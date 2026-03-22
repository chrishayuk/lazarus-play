"""
KV Cache Extension — Hybrid Tests
===================================
1. Content-only entries (exclude BOS, structural tokens)
2. Attention routing at L29 + 12-byte injection at L30
3. Scale test with more entries
"""

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from chuk_lazarus.inference import UnifiedPipeline, UnifiedPipelineConfig
from chuk_lazarus.models_v2.families.gemma.model import clip_residual
import json


def load_model():
    print("Loading model...")
    config = UnifiedPipelineConfig()
    pipeline = UnifiedPipeline.from_pretrained("google/gemma-3-4b-it", pipeline_config=config)
    model = pipeline.model
    tokenizer = pipeline.tokenizer
    print(f"Model loaded. Layers: {len(model.model.layers)}")
    return model, tokenizer


def extract_kv_pre_rope(model, input_ids, layer_idx, positions=None):
    """Extract K/V PRE-RoPE at specified layer."""
    backbone = model.model
    config = backbone.config

    h = backbone.embed_tokens(input_ids)
    h = h * mx.array(config.hidden_size ** 0.5, dtype=mx.bfloat16).astype(h.dtype)

    global_mask = backbone._create_attention_mask(h, None)
    sliding_mask = (
        backbone._create_attention_mask(h, None, window_size=backbone.sliding_window)
        if backbone.sliding_window_pattern > 1 else None
    )

    for i in range(layer_idx):
        layer = backbone.layers[i]
        is_global = config.is_global_layer(i)
        mask = global_mask if is_global else sliding_mask
        output = layer(h, mask=mask, cache=None)
        h = output.hidden_states

    layer = backbone.layers[layer_idx]
    attn = layer.self_attn
    normed = layer.input_layernorm(h)
    batch_size, seq_len, _ = normed.shape

    keys = attn.k_proj(normed)
    values = attn.v_proj(normed)

    keys = keys.reshape(batch_size, seq_len, attn.num_kv_heads, attn.head_dim)
    keys = keys.transpose(0, 2, 1, 3)
    values = values.reshape(batch_size, seq_len, attn.num_kv_heads, attn.head_dim)
    values = values.transpose(0, 2, 1, 3)

    keys = attn.k_norm(keys)
    mx.eval(keys, values, h)

    if positions is not None:
        k_sel = keys[:, :, positions, :]
        v_sel = values[:, :, positions, :]
        mx.eval(k_sel, v_sel)
        return {"k": k_sel, "v": v_sel, "k_full": keys, "v_full": values, "attn": attn, "h": h}
    return {"k": keys, "v": values, "k_full": keys, "v_full": values, "attn": attn, "h": h}


def forward_with_extended_kv_hybrid(model, query_ids, stored_k_pre_rope, stored_v,
                                      attn_obj, inject_layer=29,
                                      injection_data=None, inject_at_layer=30,
                                      inject_all_entries=False):
    """
    KV extension at inject_layer + optional 12-byte injection at inject_at_layer.

    Args:
        injection_data: dict with 'embeddings' (N, 2560) and 'coefficients' (N,)
                       for each stored entry. If None, no injection.
        inject_all_entries: If True, inject the attention-weighted sum.
                          If False, inject only the top entry.
    """
    backbone = model.model
    config = backbone.config

    h = backbone.embed_tokens(query_ids)
    h = h * mx.array(config.hidden_size ** 0.5, dtype=mx.bfloat16).astype(h.dtype)

    query_len = h.shape[1]
    n_stored = stored_k_pre_rope.shape[2]

    global_mask = backbone._create_attention_mask(h, None)
    sliding_mask = (
        backbone._create_attention_mask(h, None, window_size=backbone.sliding_window)
        if backbone.sliding_window_pattern > 1 else None
    )

    # Forward L0..inject_layer-1
    for i in range(inject_layer):
        layer = backbone.layers[i]
        is_global = config.is_global_layer(i)
        mask = global_mask if is_global else sliding_mask
        output = layer(h, mask=mask, cache=None)
        h = output.hidden_states

    # === inject_layer: manual with KV extension ===
    layer = backbone.layers[inject_layer]
    attn = layer.self_attn
    residual = h
    normed = layer.input_layernorm(h)
    batch_size = 1

    queries = attn.q_proj(normed)
    keys_q = attn.k_proj(normed)
    values_q = attn.v_proj(normed)

    queries = queries.reshape(batch_size, query_len, attn.num_heads, attn.head_dim)
    queries = queries.transpose(0, 2, 1, 3)
    keys_q = keys_q.reshape(batch_size, query_len, attn.num_kv_heads, attn.head_dim)
    keys_q = keys_q.transpose(0, 2, 1, 3)
    values_q = values_q.reshape(batch_size, query_len, attn.num_kv_heads, attn.head_dim)
    values_q = values_q.transpose(0, 2, 1, 3)

    queries = attn.q_norm(queries)
    keys_q = attn.k_norm(keys_q)

    # Sequential RoPE: stored 0..N-1, query N..N+M-1
    stored_k_roped = attn.rope(stored_k_pre_rope)
    queries = attn.rope(queries, offset=n_stored)
    keys_q = attn.rope(keys_q, offset=n_stored)

    # Build extended KV (stored BEFORE query)
    keys_ext = mx.concatenate([stored_k_roped, keys_q], axis=2)
    values_ext = mx.concatenate([stored_v, values_q], axis=2)

    # GQA
    if attn.n_rep > 1:
        keys_ext = mx.repeat(keys_ext, attn.n_rep, axis=1)
        values_ext = mx.repeat(values_ext, attn.n_rep, axis=1)

    # Manual attention
    scores = (queries @ keys_ext.transpose(0, 1, 3, 2)) * attn.scale
    stored_cols = mx.zeros((query_len, n_stored), dtype=scores.dtype)
    causal = nn.MultiHeadAttention.create_additive_causal_mask(query_len).astype(scores.dtype)
    extended_mask = mx.concatenate([stored_cols, causal], axis=1)
    scores = scores + extended_mask
    weights = mx.softmax(scores, axis=-1)

    attn_to_stored = weights[0, :, -1, :n_stored]
    attn_to_query = weights[0, :, -1, n_stored:]

    context = weights @ values_ext
    context = context.transpose(0, 2, 1, 3).reshape(batch_size, query_len, -1)
    attn_output = attn.o_proj(context)

    h = clip_residual(residual, layer.post_attention_layernorm(attn_output))
    ffn_out = layer.mlp(layer.pre_feedforward_layernorm(h))
    h = clip_residual(h, layer.post_feedforward_layernorm(ffn_out))

    mx.eval(h, attn_to_stored)

    # === Determine winning entry from H4 attention ===
    h4_attn = attn_to_stored[4, :]  # H4's attention to stored entries
    mx.eval(h4_attn)
    winning_idx = int(mx.argmax(h4_attn))
    winning_weight = float(h4_attn[winning_idx])

    # === Continue through remaining layers, with optional injection ===
    for i in range(inject_layer + 1, len(backbone.layers)):
        layer_i = backbone.layers[i]
        is_global = config.is_global_layer(i)
        mask = global_mask if is_global else sliding_mask

        if i == inject_at_layer and injection_data is not None:
            # Inject 12 bytes at this layer
            if inject_all_entries:
                # Attention-weighted injection
                # Sum of (attn_weight * coeff * embedding) across entries
                # Use H4 attention as routing signal
                h4_weights = h4_attn  # (N,)
                coeffs = injection_data['coefficients']  # (N,)
                embeds = injection_data['embeddings']     # (N, 2560)
                # weighted sum
                w = (h4_weights * coeffs)  # (N,)
                injection_vec = w @ embeds  # (2560,)
                injection_vec = injection_vec.reshape(1, 1, -1)
            else:
                # Inject only the winning entry
                coeff = injection_data['coefficients'][winning_idx]
                embed = injection_data['embeddings'][winning_idx]  # (2560,)
                injection_vec = (coeff * embed).reshape(1, 1, -1)

            # Add to last position
            h_last = h[:, -1:, :] + injection_vec.astype(h.dtype)
            h = mx.concatenate([h[:, :-1, :], h_last], axis=1)
            mx.eval(h)

        output = layer_i(h, mask=mask, cache=None)
        h = output.hidden_states

    # Final
    h = backbone.norm(h)
    if model.tie_word_embeddings:
        logits = backbone.embed_tokens.as_linear(h)
    else:
        logits = model.lm_head(h)

    probs = mx.softmax(logits[0, -1, :])
    mx.eval(probs)

    return {
        "probs": probs,
        "attn_to_stored": attn_to_stored,
        "winning_idx": winning_idx,
        "winning_weight": winning_weight,
        "h4_attn": h4_attn,
    }


def compute_injection_data(model, tokenizer, token_ids, donor_prompt, layer=30):
    """
    For each token_id, compute the injection coefficient by projecting
    the donor's residual at 'layer' onto the token's unembedding direction.
    """
    backbone = model.model

    # Get token embeddings (unembedding directions)
    embed_matrix = backbone.embed_tokens.weight  # (vocab, 2560)
    embeddings = []
    for tid in token_ids:
        e = embed_matrix[tid]  # (2560,)
        embeddings.append(e)
    embeddings = mx.stack(embeddings, axis=0)  # (N, 2560)

    # Get donor residual at specified layer
    input_ids = mx.array([tokenizer.encode(donor_prompt, add_special_tokens=True)])
    h = backbone.embed_tokens(input_ids)
    h = h * mx.array(backbone.config.hidden_size ** 0.5, dtype=mx.bfloat16).astype(h.dtype)

    global_mask = backbone._create_attention_mask(h, None)
    sliding_mask = (
        backbone._create_attention_mask(h, None, window_size=backbone.sliding_window)
        if backbone.sliding_window_pattern > 1 else None
    )

    for i in range(layer + 1):
        layer_i = backbone.layers[i]
        is_global = backbone.config.is_global_layer(i)
        mask = global_mask if is_global else sliding_mask
        output = layer_i(h, mask=mask, cache=None)
        h = output.hidden_states

    donor_residual = h[0, -1, :]  # (2560,)

    # Project onto each embedding direction
    # coefficient = (donor · embed) / (embed · embed)
    coefficients = []
    for i in range(len(token_ids)):
        e = embeddings[i]
        coeff = mx.sum(donor_residual * e) / (mx.sum(e * e) + 1e-8)
        coefficients.append(coeff)

    coefficients = mx.stack(coefficients)
    mx.eval(embeddings, coefficients)

    return {
        "embeddings": embeddings,       # (N, 2560)
        "coefficients": coefficients,   # (N,)
        "token_ids": token_ids,
    }


def main():
    model, tokenizer = load_model()

    document = "Zarkov Industries, a cutting-edge robotics company, was founded in 2019 in the city of Voltara by engineer Dimitri Zarkov."
    full_context = f"<start_of_turn>user\n{document}\n\nThe city where Zarkov Industries was founded is called<end_of_turn>\n<start_of_turn>model\n"
    bare_query = "<start_of_turn>user\nThe city where Zarkov Industries was founded is called<end_of_turn>\n<start_of_turn>model\n"

    targets = {"Volt": 194328, "The": 818, "Z": 236953}

    full_ids = mx.array([tokenizer.encode(full_context, add_special_tokens=True)])
    bare_ids = mx.array([tokenizer.encode(bare_query, add_special_tokens=True)])
    full_tokens = tokenizer.convert_ids_to_tokens(full_ids[0].tolist())

    volt_pos = None
    for i, tok in enumerate(full_tokens):
        if "Volt" in tok:
            volt_pos = i
            break
    print(f"Volt at pos {volt_pos}")
    print(f"Full context: {len(full_tokens)} tokens")

    results = {}

    # ================================================================
    # TEST 1: Content-only entries (exclude BOS, structural, punctuation)
    # ================================================================
    print("\n" + "=" * 60)
    print("TEST 1: Content-only K/V entries (exclude structural tokens)")
    print("=" * 60)

    # Identify content positions (skip BOS, special tokens, punctuation)
    structural_positions = set()
    for i, tok in enumerate(full_tokens):
        if tok in ['<bos>', '<start_of_turn>', '<end_of_turn>', 'user', 'model', '\n', '\n\n']:
            structural_positions.add(i)
        elif tok in [',', '.', '-']:
            structural_positions.add(i)

    content_positions = [i for i in range(len(full_tokens)) if i not in structural_positions]
    print(f"  Total tokens: {len(full_tokens)}")
    print(f"  Structural: {len(structural_positions)}")
    print(f"  Content positions: {len(content_positions)}")
    print(f"  Content tokens: {[full_tokens[p] for p in content_positions]}")

    # Find Volt's index within content positions
    volt_content_idx = content_positions.index(volt_pos)
    print(f"  Volt at content index {volt_content_idx} (original pos {volt_pos})")

    # Extract K/V only at content positions
    kv_content = extract_kv_pre_rope(model, full_ids, layer_idx=29, positions=content_positions)

    # Test with content-only entries
    r = forward_with_extended_kv_hybrid(
        model, bare_ids, kv_content['k'], kv_content['v'],
        kv_content['attn'], inject_layer=29
    )

    p_volt = float(r["probs"][194328])
    p_z = float(r["probs"][236953])
    top_ids = mx.argsort(r["probs"])[-5:][::-1].tolist()
    top5 = [(tokenizer.decode([t]), float(r["probs"][t])) for t in top_ids]

    h4_attn = r["h4_attn"]
    mx.eval(h4_attn)
    h4_volt = float(h4_attn[volt_content_idx])
    h4_total = float(mx.sum(h4_attn))
    h4_best_idx = int(mx.argmax(h4_attn))
    h4_best_val = float(h4_attn[h4_best_idx])

    print(f"\n  P(Volt): {p_volt:.6f} ({p_volt*100:.2f}%)")
    print(f"  P(Z): {p_z:.6f}")
    print(f"  Top 5: {top5}")
    print(f"\n  H4 attention to stored entries ({len(content_positions)} content):")
    print(f"    Total: {h4_total:.4f}")
    print(f"    Volt (idx {volt_content_idx}): {h4_volt:.4f}")
    print(f"    Best: idx {h4_best_idx} = '{full_tokens[content_positions[h4_best_idx]]}' ({h4_best_val:.4f})")

    # Show top-5 H4 entries
    h4_np = np.array(h4_attn.tolist())
    top5_h4 = np.argsort(h4_np)[-5:][::-1]
    print(f"    Top 5 H4 entries:")
    for idx in top5_h4:
        tok = full_tokens[content_positions[idx]]
        print(f"      idx {idx} = '{tok}' (pos {content_positions[idx]}): {h4_np[idx]:.4f}")

    results["content_only"] = {
        "n_entries": len(content_positions),
        "P_Volt": p_volt,
        "H4_volt": h4_volt,
        "H4_best": full_tokens[content_positions[h4_best_idx]],
        "H4_best_val": h4_best_val,
    }

    # ================================================================
    # TEST 2: Content-only + K-norm filtering (top entries by K-norm)
    # ================================================================
    print("\n" + "=" * 60)
    print("TEST 2: Top-8 content entries by K-norm")
    print("=" * 60)

    k_norms = mx.sqrt(mx.sum(kv_content['k'] ** 2, axis=-1))  # (1, 4, N)
    k_norms_mean = mx.mean(k_norms, axis=1)[0]  # (N,)
    mx.eval(k_norms_mean)

    k_norms_np = np.array(k_norms_mean.tolist())
    top8_content_idx = np.argsort(k_norms_np)[-8:][::-1].tolist()

    print(f"  Top-8 by K-norm:")
    for idx in top8_content_idx:
        print(f"    idx {idx} = '{full_tokens[content_positions[idx]]}' (pos {content_positions[idx]}): K-norm={k_norms_np[idx]:.1f}")

    # Extract just those 8 entries
    top8_positions = [content_positions[idx] for idx in top8_content_idx]
    kv_top8 = extract_kv_pre_rope(model, full_ids, layer_idx=29, positions=top8_positions)

    # Find Volt in top8
    volt_in_top8 = None
    for i, pos in enumerate(top8_positions):
        if pos == volt_pos:
            volt_in_top8 = i
            break

    r = forward_with_extended_kv_hybrid(
        model, bare_ids, kv_top8['k'], kv_top8['v'],
        kv_top8['attn'], inject_layer=29
    )

    p_volt = float(r["probs"][194328])
    top_ids = mx.argsort(r["probs"])[-5:][::-1].tolist()
    top5 = [(tokenizer.decode([t]), float(r["probs"][t])) for t in top_ids]

    h4_attn = r["h4_attn"]
    mx.eval(h4_attn)
    print(f"\n  P(Volt): {p_volt:.6f} ({p_volt*100:.2f}%)")
    print(f"  Top 5: {top5}")

    print(f"\n  H4 attention to top-8 entries:")
    for i in range(8):
        tok = full_tokens[top8_positions[i]]
        v = float(h4_attn[i])
        marker = " <-- VOLT" if i == volt_in_top8 else ""
        print(f"    [{i}] '{tok}' (pos {top8_positions[i]}): {v:.4f}{marker}")

    results["top8_knorm"] = {
        "P_Volt": p_volt,
        "volt_in_top8": volt_in_top8,
        "H4_volt": float(h4_attn[volt_in_top8]) if volt_in_top8 is not None else None,
    }

    # ================================================================
    # TEST 3: Hybrid — KV routing + 12-byte injection
    # ================================================================
    print("\n" + "=" * 60)
    print("TEST 3: Hybrid — KV routing at L29 + 12-byte injection at L30")
    print("=" * 60)

    # Compute injection data for content tokens
    content_token_ids = [full_ids[0, pos].item() for pos in content_positions]
    injection_data = compute_injection_data(
        model, tokenizer, content_token_ids, full_context, layer=30
    )

    # Test 3a: Inject winning entry (whatever H4 picks)
    print("\n  3a: Inject H4's top pick (content-only entries)")
    r = forward_with_extended_kv_hybrid(
        model, bare_ids, kv_content['k'], kv_content['v'],
        kv_content['attn'], inject_layer=29,
        injection_data=injection_data, inject_at_layer=30,
        inject_all_entries=False
    )

    p_volt = float(r["probs"][194328])
    top_ids = mx.argsort(r["probs"])[-5:][::-1].tolist()
    top5 = [(tokenizer.decode([t]), float(r["probs"][t])) for t in top_ids]
    winning = content_positions[r["winning_idx"]]
    print(f"  H4 winning entry: idx {r['winning_idx']} = '{full_tokens[winning]}' (weight {r['winning_weight']:.4f})")
    print(f"  P(Volt): {p_volt:.6f} ({p_volt*100:.2f}%)")
    print(f"  Top 5: {top5}")

    results["hybrid_h4_pick"] = {
        "P_Volt": p_volt, "top5": top5,
        "winning_entry": full_tokens[winning],
        "winning_weight": r["winning_weight"],
    }

    # Test 3b: Inject Volt entry directly (oracle — what if routing was perfect)
    print("\n  3b: Inject Volt entry directly (oracle)")
    # Create injection data for just the Volt entry
    volt_injection = {
        "embeddings": injection_data['embeddings'][volt_content_idx:volt_content_idx+1],
        "coefficients": injection_data['coefficients'][volt_content_idx:volt_content_idx+1],
    }

    # Use single Volt entry
    r = forward_with_extended_kv_hybrid(
        model, bare_ids,
        kv_content['k'][:, :, volt_content_idx:volt_content_idx+1, :],
        kv_content['v'][:, :, volt_content_idx:volt_content_idx+1, :],
        kv_content['attn'], inject_layer=29,
        injection_data=volt_injection, inject_at_layer=30,
        inject_all_entries=False
    )

    p_volt = float(r["probs"][194328])
    top_ids = mx.argsort(r["probs"])[-5:][::-1].tolist()
    top5 = [(tokenizer.decode([t]), float(r["probs"][t])) for t in top_ids]
    print(f"  P(Volt): {p_volt:.6f} ({p_volt*100:.2f}%)")
    print(f"  Top 5: {top5}")

    results["hybrid_oracle_volt"] = {"P_Volt": p_volt, "top5": top5}

    # Test 3c: Attention-weighted injection (all entries)
    print("\n  3c: Attention-weighted injection (all content entries)")
    r = forward_with_extended_kv_hybrid(
        model, bare_ids, kv_content['k'], kv_content['v'],
        kv_content['attn'], inject_layer=29,
        injection_data=injection_data, inject_at_layer=30,
        inject_all_entries=True
    )

    p_volt = float(r["probs"][194328])
    top_ids = mx.argsort(r["probs"])[-5:][::-1].tolist()
    top5 = [(tokenizer.decode([t]), float(r["probs"][t])) for t in top_ids]
    print(f"  P(Volt): {p_volt:.6f} ({p_volt*100:.2f}%)")
    print(f"  Top 5: {top5}")

    results["hybrid_attn_weighted"] = {"P_Volt": p_volt, "top5": top5}

    # Test 3d: Inject at different layers
    print("\n  3d: Oracle Volt injection at different layers")
    for inject_l in [26, 28, 29, 30, 31, 32]:
        volt_inj_l = compute_injection_data(
            model, tokenizer, [194328], full_context, layer=inject_l
        )
        volt_inj_data = {
            "embeddings": volt_inj_l['embeddings'],
            "coefficients": volt_inj_l['coefficients'],
        }

        r = forward_with_extended_kv_hybrid(
            model, bare_ids,
            kv_content['k'][:, :, volt_content_idx:volt_content_idx+1, :],
            kv_content['v'][:, :, volt_content_idx:volt_content_idx+1, :],
            kv_content['attn'], inject_layer=29,
            injection_data=volt_inj_data, inject_at_layer=inject_l,
            inject_all_entries=False
        )
        p_volt = float(r["probs"][194328])
        print(f"    Inject at L{inject_l}: P(Volt)={p_volt:.6f} ({p_volt*100:.2f}%)")
        results[f"hybrid_oracle_L{inject_l}"] = {"P_Volt": p_volt}

    # ================================================================
    # SUMMARY
    # ================================================================
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    print(f"\n{'Condition':<50} {'P(Volt)':>10}")
    print("-" * 62)
    for key in ["content_only", "top8_knorm", "hybrid_h4_pick", "hybrid_oracle_volt", "hybrid_attn_weighted"]:
        if key in results:
            print(f"  {key:<48} {results[key]['P_Volt']:>10.6f}")

    for inject_l in [26, 28, 29, 30, 31, 32]:
        key = f"hybrid_oracle_L{inject_l}"
        if key in results:
            print(f"  {key:<48} {results[key]['P_Volt']:>10.6f}")

    # Save
    output_path = "/Users/christopherhay/chris-source/lazarus-play/experiments/kv_cache_extension/results_hybrid.json"
    def ser(obj):
        if isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, default=ser)
    print(f"\nSaved to {output_path}")


if __name__ == "__main__":
    main()
