"""
Synthetic KV Injection Experiment v2
=====================================
Test whether stored K/V entries at L29 with the final residual enable
the model's own attention to route to the correct fact.

Uses the model's own architecture (GemmaAttention) to correctly handle:
- Q/K RMS normalization
- RoPE with correct base frequency
- GQA (4 KV heads → 8 Q heads)
- query_pre_attn_scalar scaling
- 4 norms per block
- clip_residual for bfloat16
"""

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from chuk_lazarus.inference import UnifiedPipeline, UnifiedPipelineConfig
import json
import time


def load_model():
    """Load the model."""
    print("Loading model...")
    config = UnifiedPipelineConfig()
    pipeline = UnifiedPipeline.from_pretrained("google/gemma-3-4b-it", pipeline_config=config)
    model = pipeline.model
    tokenizer = pipeline.tokenizer
    print(f"Model loaded. Layers: {len(model.model.layers)}")
    return model, tokenizer


def native_predict(model, tokenizer, prompt, target_ids=None):
    """Run native forward and return probs for target tokens."""
    input_ids = mx.array([tokenizer.encode(prompt, add_special_tokens=True)])
    output = model(input_ids)
    logits = output.logits[0, -1, :]
    probs = mx.softmax(logits)
    mx.eval(probs)

    top_ids = mx.argsort(probs)[-5:][::-1].tolist()
    top5 = [(tokenizer.decode([t]), float(probs[t])) for t in top_ids]

    result = {"top5": top5}
    if target_ids:
        for name, tid in target_ids.items():
            result[f"P_{name}"] = float(probs[tid])
    return result


def forward_to_layer(model, input_ids, stop_after_layer):
    """
    Run model forward through layers 0..stop_after_layer.
    Returns the hidden state (residual stream) after that layer.
    """
    backbone = model.model
    config = backbone.config

    # Embed + scale
    h = backbone.embed_tokens(input_ids)
    h = h * mx.array(config.hidden_size ** 0.5, dtype=mx.bfloat16).astype(h.dtype)

    # Masks
    global_mask = backbone._create_attention_mask(h, None)
    sliding_mask = (
        backbone._create_attention_mask(h, None, window_size=backbone.sliding_window)
        if backbone.sliding_window_pattern > 1 else None
    )

    for i in range(stop_after_layer + 1):
        layer = backbone.layers[i]
        is_global = config.is_global_layer(i)
        mask = global_mask if is_global else sliding_mask
        output = layer(h, mask=mask, cache=None)
        h = output.hidden_states

    mx.eval(h)
    return h


def extract_kv_at_layer(model, input_ids, layer_idx, positions=None):
    """
    Run forward to just before layer_idx, then compute K/V projections
    at that layer. Returns post-norm, post-RoPE K and V.

    Returns K, V in shape (batch, num_kv_heads, n_positions, head_dim)
    ready for attention (already transposed, normed, RoPE'd).
    """
    backbone = model.model
    config = backbone.config

    # Forward through layers 0..layer_idx-1
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

    # At layer_idx: compute K, V manually
    layer = backbone.layers[layer_idx]
    attn = layer.self_attn

    normed = layer.input_layernorm(h)

    batch_size, seq_len, _ = normed.shape

    # Project
    keys = attn.k_proj(normed)
    values = attn.v_proj(normed)
    queries = attn.q_proj(normed)

    # Reshape to (batch, seq, heads, head_dim) then transpose to (batch, heads, seq, head_dim)
    keys = keys.reshape(batch_size, seq_len, attn.num_kv_heads, attn.head_dim)
    keys = keys.transpose(0, 2, 1, 3)
    values = values.reshape(batch_size, seq_len, attn.num_kv_heads, attn.head_dim)
    values = values.transpose(0, 2, 1, 3)
    queries = queries.reshape(batch_size, seq_len, attn.num_heads, attn.head_dim)
    queries = queries.transpose(0, 2, 1, 3)

    # Apply Q/K normalization (Gemma-specific)
    queries = attn.q_norm(queries)
    keys = attn.k_norm(keys)

    # Apply RoPE
    queries = attn.rope(queries)
    keys = attn.rope(keys)

    mx.eval(keys, values, queries)

    if positions is not None:
        k_sel = keys[:, :, positions, :]
        v_sel = values[:, :, positions, :]
    else:
        k_sel = keys
        v_sel = values

    mx.eval(k_sel, v_sel)

    return {
        "k": k_sel,           # (batch, num_kv_heads, n_pos, head_dim)
        "v": v_sel,           # (batch, num_kv_heads, n_pos, head_dim)
        "k_full": keys,       # Full K for all positions
        "v_full": values,     # Full V for all positions
        "q_full": queries,    # Full Q for all positions
        "hidden_pre_layer": h,
        "normed": normed,
        "attn": attn,
    }


def forward_with_synthetic_kv(model, input_ids, synthetic_k, synthetic_v,
                               inject_layer=29):
    """
    Run the model, but at inject_layer, extend the K/V with synthetic entries
    before computing attention. Synthetic K/V should already be post-norm and
    post-RoPE (ready for dot product).

    Args:
        model: The loaded model
        input_ids: (1, seq_len) input token IDs
        synthetic_k: (1, num_kv_heads, N, head_dim) - post-norm, post-RoPE K vectors
        synthetic_v: (1, num_kv_heads, N, head_dim) - post-norm, post-RoPE V vectors
        inject_layer: Layer at which to inject

    Returns dict with probs, attention weights to synthetic entries, etc.
    """
    backbone = model.model
    config = backbone.config

    # Embed + scale
    h = backbone.embed_tokens(input_ids)
    h = h * mx.array(config.hidden_size ** 0.5, dtype=mx.bfloat16).astype(h.dtype)

    # Masks
    seq_len = h.shape[1]
    global_mask = backbone._create_attention_mask(h, None)
    sliding_mask = (
        backbone._create_attention_mask(h, None, window_size=backbone.sliding_window)
        if backbone.sliding_window_pattern > 1 else None
    )

    # Forward through layers 0..inject_layer-1
    for i in range(inject_layer):
        layer = backbone.layers[i]
        is_global = config.is_global_layer(i)
        mask = global_mask if is_global else sliding_mask
        output = layer(h, mask=mask, cache=None)
        h = output.hidden_states

    # === Layer inject_layer: Manual decomposition with KV injection ===
    layer = backbone.layers[inject_layer]
    attn = layer.self_attn
    residual = h

    # Input layernorm
    normed = layer.input_layernorm(h)

    batch_size = 1

    # Q/K/V projections
    queries = attn.q_proj(normed)
    keys = attn.k_proj(normed)
    values = attn.v_proj(normed)

    # Reshape and transpose to (batch, heads, seq, head_dim)
    queries = queries.reshape(batch_size, seq_len, attn.num_heads, attn.head_dim)
    queries = queries.transpose(0, 2, 1, 3)
    keys = keys.reshape(batch_size, seq_len, attn.num_kv_heads, attn.head_dim)
    keys = keys.transpose(0, 2, 1, 3)
    values = values.reshape(batch_size, seq_len, attn.num_kv_heads, attn.head_dim)
    values = values.transpose(0, 2, 1, 3)

    # Q/K normalization
    queries = attn.q_norm(queries)
    keys = attn.k_norm(keys)

    # RoPE
    queries = attn.rope(queries)
    keys = attn.rope(keys)

    # === INJECT: Extend K/V with synthetic entries ===
    n_synthetic = synthetic_k.shape[2]
    keys = mx.concatenate([keys, synthetic_k], axis=2)      # (1, kv_heads, seq+N, head_dim)
    values = mx.concatenate([values, synthetic_v], axis=2)

    # GQA: repeat KV heads
    if attn.n_rep > 1:
        keys = mx.repeat(keys, attn.n_rep, axis=1)
        values = mx.repeat(values, attn.n_rep, axis=1)

    # Scaled dot-product attention (manual, to extract weights)
    total_kv_len = seq_len + n_synthetic
    scores = (queries @ keys.transpose(0, 1, 3, 2)) * attn.scale
    # scores shape: (1, num_heads, seq_len, total_kv_len)

    # Create extended causal mask
    # Original positions: causal (can only attend to earlier positions)
    # Synthetic positions: all query positions can attend to them
    causal = nn.MultiHeadAttention.create_additive_causal_mask(seq_len)
    causal = causal.astype(scores.dtype)
    # synthetic columns: no masking (value 0)
    synthetic_cols = mx.zeros((seq_len, n_synthetic), dtype=scores.dtype)
    extended_mask = mx.concatenate([causal, synthetic_cols], axis=1)  # (seq, seq+N)

    scores = scores + extended_mask
    weights = mx.softmax(scores, axis=-1)

    # Extract attention to synthetic entries (last query position)
    attn_to_synthetic = weights[0, :, -1, seq_len:]  # (num_heads, N)
    attn_to_original = weights[0, :, -1, :seq_len]   # (num_heads, seq_len)

    # Weighted sum
    context = weights @ values  # (1, num_heads, seq_len, head_dim)

    # Transpose back and project
    context = context.transpose(0, 2, 1, 3).reshape(batch_size, seq_len, -1)
    attn_output = attn.o_proj(context)

    # Post-attention layernorm + residual (Gemma uses clip_residual)
    from chuk_lazarus.models_v2.families.gemma.model import clip_residual
    h = clip_residual(residual, layer.post_attention_layernorm(attn_output))

    # FFN
    ffn_out = layer.mlp(layer.pre_feedforward_layernorm(h))
    h = clip_residual(h, layer.post_feedforward_layernorm(ffn_out))

    # === Continue through remaining layers normally ===
    for i in range(inject_layer + 1, len(backbone.layers)):
        layer_i = backbone.layers[i]
        is_global = config.is_global_layer(i)
        mask = global_mask if is_global else sliding_mask
        output = layer_i(h, mask=mask, cache=None)
        h = output.hidden_states

    # Final norm + lm_head
    h = backbone.norm(h)
    if model.tie_word_embeddings:
        logits = backbone.embed_tokens.as_linear(h)
    else:
        logits = model.lm_head(h)

    last_logits = logits[0, -1, :]
    probs = mx.softmax(last_logits)

    mx.eval(probs, attn_to_synthetic, attn_to_original)

    return {
        "probs": probs,
        "attn_to_synthetic": attn_to_synthetic,  # (num_heads, N)
        "attn_to_original": attn_to_original,    # (num_heads, seq_len)
    }


def main():
    model, tokenizer = load_model()

    # Prompts
    document = "Zarkov Industries, a cutting-edge robotics company, was founded in 2019 in the city of Voltara by engineer Dimitri Zarkov."
    full_context = f"<start_of_turn>user\n{document}\n\nThe city where Zarkov Industries was founded is called<end_of_turn>\n<start_of_turn>model\n"
    bare_query = "<start_of_turn>user\nThe city where Zarkov Industries was founded is called<end_of_turn>\n<start_of_turn>model\n"

    # Target tokens
    targets = {"Volt": 194328, "The": 818, "Z": 236953}

    # Tokenize
    full_ids = mx.array([tokenizer.encode(full_context, add_special_tokens=True)])
    bare_ids = mx.array([tokenizer.encode(bare_query, add_special_tokens=True)])

    full_tokens = tokenizer.convert_ids_to_tokens(full_ids[0].tolist())
    bare_tokens = tokenizer.convert_ids_to_tokens(bare_ids[0].tolist())

    print(f"\nFull context: {len(full_tokens)} tokens")
    print(f"Bare query: {len(bare_tokens)} tokens")

    # Find Volt position in full context
    volt_pos = None
    for i, tok in enumerate(full_tokens):
        if "Volt" in tok:
            volt_pos = i
            print(f"Found Volt at position {i}: '{tok}'")
            break

    results = {}

    # ================================================================
    # BASELINES
    # ================================================================
    print("\n" + "=" * 60)
    print("BASELINE 1: Full context (native forward)")
    print("=" * 60)
    r = native_predict(model, tokenizer, full_context, targets)
    print(f"  P(Volt): {r['P_Volt']:.6f}")
    print(f"  Top 5: {r['top5']}")
    results["baseline_full"] = r

    print("\n" + "=" * 60)
    print("BASELINE 2: Bare query (native forward)")
    print("=" * 60)
    r = native_predict(model, tokenizer, bare_query, targets)
    print(f"  P(Volt): {r['P_Volt']:.6f}")
    print(f"  P(Z): {r['P_Z']:.6f}")
    print(f"  Top 5: {r['top5']}")
    results["baseline_bare"] = r

    # Verify manual forward matches native
    print("\n" + "=" * 60)
    print("VERIFICATION: Manual forward (no injection) matches native")
    print("=" * 60)
    # Run with 0 synthetic entries to verify
    empty_k = mx.zeros((1, 4, 0, 256), dtype=mx.bfloat16)
    empty_v = mx.zeros((1, 4, 0, 256), dtype=mx.bfloat16)
    r_manual = forward_with_synthetic_kv(model, bare_ids, empty_k, empty_v, inject_layer=29)
    p_z = float(r_manual["probs"][236953])
    p_volt = float(r_manual["probs"][194328])
    print(f"  Manual P(Z): {p_z:.6f} (native: {results['baseline_bare']['P_Z']:.6f})")
    print(f"  Manual P(Volt): {p_volt:.6f}")

    # Also verify with full context
    r_manual_full = forward_with_synthetic_kv(model, full_ids, empty_k, empty_v, inject_layer=29)
    p_volt_full = float(r_manual_full["probs"][194328])
    print(f"  Manual full-ctx P(Volt): {p_volt_full:.6f} (native: {results['baseline_full']['P_Volt']:.6f})")

    results["verification"] = {
        "bare_manual_P_Z": p_z,
        "bare_manual_P_Volt": p_volt,
        "full_manual_P_Volt": p_volt_full,
        "matches": abs(p_z - results["baseline_bare"]["P_Z"]) < 0.01 and abs(p_volt_full - results["baseline_full"]["P_Volt"]) < 0.01
    }

    if not results["verification"]["matches"]:
        print("  ⚠ Manual forward doesn't match native! Debugging...")
    else:
        print("  ✓ Manual forward matches native!")

    # ================================================================
    # EXPERIMENT 1: Extract K/V from full context at Volt position
    # ================================================================
    print("\n" + "=" * 60)
    print("EXP 1a: Extract K/V at L29 from full context")
    print("=" * 60)

    kv = extract_kv_at_layer(model, full_ids, layer_idx=29, positions=[volt_pos])
    print(f"  K shape: {kv['k'].shape}")  # (1, 4, 1, 256)
    print(f"  V shape: {kv['v'].shape}")

    # K-norms across all positions
    k_norms = mx.sqrt(mx.sum(kv['k_full'] ** 2, axis=-1))  # (1, kv_heads, seq)
    k_norms_mean = mx.mean(k_norms, axis=1)  # (1, seq)
    mx.eval(k_norms_mean)
    print(f"  K-norm at Volt (pos {volt_pos}): {float(k_norms_mean[0, volt_pos]):.2f}")
    print(f"  K-norm mean: {float(mx.mean(k_norms_mean)):.2f}")

    # ================================================================
    # EXPERIMENT 1b: Inject single K/V entry (Volt) into bare query at L29
    # ================================================================
    print("\n" + "=" * 60)
    print("EXP 1b: Inject single synthetic K/V (Volt position) at L29")
    print("=" * 60)

    r = forward_with_synthetic_kv(model, bare_ids, kv["k"], kv["v"], inject_layer=29)
    p_volt = float(r["probs"][194328])
    p_z = float(r["probs"][236953])
    p_the = float(r["probs"][818])

    top_ids = mx.argsort(r["probs"])[-5:][::-1].tolist()
    top5 = [(tokenizer.decode([t]), float(r["probs"][t])) for t in top_ids]

    print(f"  P(Volt): {p_volt:.6f}")
    print(f"  P(Z): {p_z:.6f}")
    print(f"  P(The): {p_the:.6f}")
    print(f"  Top 5: {top5}")

    print(f"\n  Attention to synthetic entry per head:")
    for h_idx in range(8):
        attn_val = float(r["attn_to_synthetic"][h_idx, 0])
        print(f"    H{h_idx}: {attn_val:.6f}")

    results["exp1b_single_volt"] = {
        "P_Volt": p_volt, "P_Z": p_z, "P_The": p_the,
        "top5": top5,
        "attn_per_head": {f"H{i}": float(r["attn_to_synthetic"][i, 0]) for i in range(8)},
    }

    # ================================================================
    # EXPERIMENT 1c: Inject ALL K/V entries from full context
    # ================================================================
    print("\n" + "=" * 60)
    print("EXP 1c: Inject ALL K/V entries from full context at L29")
    print("=" * 60)

    r = forward_with_synthetic_kv(model, bare_ids, kv["k_full"], kv["v_full"], inject_layer=29)
    p_volt = float(r["probs"][194328])
    p_z = float(r["probs"][236953])
    p_the = float(r["probs"][818])

    top_ids = mx.argsort(r["probs"])[-5:][::-1].tolist()
    top5 = [(tokenizer.decode([t]), float(r["probs"][t])) for t in top_ids]

    print(f"  P(Volt): {p_volt:.6f}")
    print(f"  P(Z): {p_z:.6f}")
    print(f"  P(The): {p_the:.6f}")
    print(f"  Top 5: {top5}")

    # H4 attention to Volt position among all synthetic entries
    h4_syn = r["attn_to_synthetic"][4, :]  # H4's attention to all 54 synthetic entries
    mx.eval(h4_syn)
    h4_volt = float(h4_syn[volt_pos])
    h4_best = int(mx.argmax(h4_syn))
    print(f"\n  H4 attention to Volt (pos {volt_pos}): {h4_volt:.6f}")
    print(f"  H4 best synthetic entry: pos {h4_best} ('{full_tokens[h4_best]}', attn={float(h4_syn[h4_best]):.6f})")

    # Total attention to synthetic per head
    print(f"\n  Total attention to ALL synthetic entries per head:")
    for h_idx in range(8):
        total = float(mx.sum(r["attn_to_synthetic"][h_idx, :]))
        print(f"    H{h_idx}: {total:.6f}")

    results["exp1c_all_kv"] = {
        "P_Volt": p_volt, "P_Z": p_z, "P_The": p_the,
        "top5": top5,
        "H4_volt_attn": h4_volt,
        "H4_best_entry": h4_best,
        "H4_best_token": full_tokens[h4_best],
    }

    # ================================================================
    # EXPERIMENT 1d: Top-K by K-norm
    # ================================================================
    print("\n" + "=" * 60)
    print("EXP 1d: Inject top-8 K/V entries by K-norm at L29")
    print("=" * 60)

    k_norms_np = np.array(k_norms_mean[0].tolist())
    k_norms_np[:4] = 0  # Exclude special tokens
    top8_pos = np.argsort(k_norms_np)[-8:][::-1].tolist()

    print(f"  Top-8 positions: {[(p, full_tokens[p]) for p in top8_pos]}")

    kv_top8 = extract_kv_at_layer(model, full_ids, layer_idx=29, positions=top8_pos)
    r = forward_with_synthetic_kv(model, bare_ids, kv_top8["k"], kv_top8["v"], inject_layer=29)

    p_volt = float(r["probs"][194328])
    top_ids = mx.argsort(r["probs"])[-5:][::-1].tolist()
    top5 = [(tokenizer.decode([t]), float(r["probs"][t])) for t in top_ids]

    print(f"  P(Volt): {p_volt:.6f}")
    print(f"  Top 5: {top5}")

    print(f"  Attention to synthetic entries per head:")
    for h_idx in range(8):
        syn_attn = r["attn_to_synthetic"][h_idx, :]
        mx.eval(syn_attn)
        best = int(mx.argmax(syn_attn))
        print(f"    H{h_idx}: best=entry {best} (pos {top8_pos[best]}, '{full_tokens[top8_pos[best]]}', attn={float(syn_attn[best]):.6f}), total={float(mx.sum(syn_attn)):.6f}")

    results["exp1d_top8"] = {
        "P_Volt": p_volt,
        "top5": top5,
        "positions": [(p, full_tokens[p]) for p in top8_pos],
    }

    # ================================================================
    # EXPERIMENT 2: Try different injection layers
    # ================================================================
    print("\n" + "=" * 60)
    print("EXP 2: Inject ALL K/V at different layers")
    print("=" * 60)

    for inject_layer in [24, 26, 29, 30]:
        print(f"\n  --- Layer {inject_layer} ---")
        kv_layer = extract_kv_at_layer(model, full_ids, layer_idx=inject_layer)
        r = forward_with_synthetic_kv(model, bare_ids, kv_layer["k_full"], kv_layer["v_full"],
                                       inject_layer=inject_layer)
        p_volt = float(r["probs"][194328])
        top_ids = mx.argsort(r["probs"])[-3:][::-1].tolist()
        top3 = [(tokenizer.decode([t]), float(r["probs"][t])) for t in top_ids]
        print(f"  P(Volt): {p_volt:.6f}")
        print(f"  Top 3: {top3}")

        # Total attention to synthetic
        total_syn = [float(mx.sum(r["attn_to_synthetic"][h, :])) for h in range(8)]
        print(f"  Total attn to synthetic: {['%.4f' % t for t in total_syn]}")

        results[f"exp2_layer{inject_layer}"] = {
            "P_Volt": p_volt, "top3": top3,
            "total_attn_per_head": total_syn,
        }

    # ================================================================
    # EXPERIMENT 3: Hybrid — KV routing at L29 + residual injection at L30
    # ================================================================
    print("\n" + "=" * 60)
    print("EXP 3: Hybrid — attention routing at L29 + residual from donor at L30")
    print("=" * 60)

    # Get the full-context residual at L30 (last position)
    donor_h_L30 = forward_to_layer(model, full_ids, stop_after_layer=30)
    donor_last = donor_h_L30[:, -1:, :]  # (1, 1, 2560)

    # Get bare query residual at L30
    bare_h_L30 = forward_to_layer(model, bare_ids, stop_after_layer=30)
    bare_last = bare_h_L30[:, -1:, :]

    # Inject donor's L30 last-position residual into bare query
    # Continue from L31 onwards
    h_injected = mx.concatenate([bare_h_L30[:, :-1, :], donor_last], axis=1)

    # Continue through remaining layers
    from chuk_lazarus.models_v2.families.gemma.model import clip_residual
    backbone = model.model
    config = backbone.config
    h = h_injected
    global_mask = backbone._create_attention_mask(h, None)
    sliding_mask = (
        backbone._create_attention_mask(h, None, window_size=backbone.sliding_window)
        if backbone.sliding_window_pattern > 1 else None
    )

    for i in range(31, len(backbone.layers)):
        layer = backbone.layers[i]
        is_global = config.is_global_layer(i)
        mask = global_mask if is_global else sliding_mask
        output = layer(h, mask=mask, cache=None)
        h = output.hidden_states

    h = backbone.norm(h)
    logits = model.lm_head(h) if not model.tie_word_embeddings else backbone.embed_tokens.as_linear(h)
    probs = mx.softmax(logits[0, -1, :])
    mx.eval(probs)

    p_volt = float(probs[194328])
    top_ids = mx.argsort(probs)[-5:][::-1].tolist()
    top5 = [(tokenizer.decode([t]), float(probs[t])) for t in top_ids]
    print(f"  P(Volt): {p_volt:.6f}")
    print(f"  Top 5: {top5}")
    results["exp3_hybrid_L30_residual"] = {"P_Volt": p_volt, "top5": top5}

    # ================================================================
    # SUMMARY TABLE
    # ================================================================
    print("\n" + "=" * 60)
    print("SUMMARY TABLE")
    print("=" * 60)
    print(f"\n{'Condition':<45} {'P(Volt)':>10}")
    print("-" * 57)

    summary_order = [
        ("Full context (native)", "baseline_full"),
        ("Bare query (native)", "baseline_bare"),
        ("Bare + 1 synth KV (Volt pos)", "exp1b_single_volt"),
        ("Bare + ALL synth KV (54 entries)", "exp1c_all_kv"),
        ("Bare + top-8 synth KV", "exp1d_top8"),
        ("Bare + ALL KV @ L24", "exp2_layer24"),
        ("Bare + ALL KV @ L26", "exp2_layer26"),
        ("Bare + ALL KV @ L29", "exp2_layer29"),
        ("Bare + ALL KV @ L30", "exp2_layer30"),
        ("Bare + donor L30 residual (L31+)", "exp3_hybrid_L30_residual"),
    ]

    for label, key in summary_order:
        if key in results:
            pv = results[key].get("P_Volt", 0)
            print(f"  {label:<43} {pv:>10.6f}")

    # Save results
    output_path = "/Users/christopherhay/chris-source/lazarus-play/experiments/synthetic_kv_injection/results_v2.json"
    def make_serializable(obj):
        if isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, default=make_serializable)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
