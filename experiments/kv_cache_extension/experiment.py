"""
KV Cache Extension - Same RoPE Frame
======================================
Inject stored K/V entries into the L29 KV cache as real positions
in the same RoPE frame as the query. Eliminate the style mismatch
by making everything one sequence.

Key difference from synthetic_kv_injection:
- Extract K/V PRE-RoPE from document
- Re-apply RoPE at sequential positions BEFORE the query
- Query tokens get RoPE positions AFTER the stored entries
- Everything in one continuous positional frame
"""

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from chuk_lazarus.inference import UnifiedPipeline, UnifiedPipelineConfig
from chuk_lazarus.models_v2.families.gemma.model import clip_residual
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

    top_ids = mx.argsort(probs)[-10:][::-1].tolist()
    top10 = [(tokenizer.decode([t]), float(probs[t])) for t in top_ids]

    result = {"top10": top10}
    if target_ids:
        for name, tid in target_ids.items():
            result[f"P_{name}"] = float(probs[tid])
    return result


def extract_kv_pre_rope(model, input_ids, layer_idx, positions=None):
    """
    Run forward to layer_idx, extract K/V BEFORE RoPE application.
    Returns pre-RoPE K and V (post-norm, post-projection, post-QK-norm).

    This allows re-applying RoPE at different positions later.
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

    # At layer_idx: compute K, V manually (PRE-RoPE)
    layer = backbone.layers[layer_idx]
    attn = layer.self_attn

    normed = layer.input_layernorm(h)
    batch_size, seq_len, _ = normed.shape

    # Project
    keys = attn.k_proj(normed)
    values = attn.v_proj(normed)

    # Reshape to (batch, heads, seq, head_dim)
    keys = keys.reshape(batch_size, seq_len, attn.num_kv_heads, attn.head_dim)
    keys = keys.transpose(0, 2, 1, 3)
    values = values.reshape(batch_size, seq_len, attn.num_kv_heads, attn.head_dim)
    values = values.transpose(0, 2, 1, 3)

    # Apply Q/K normalization (Gemma-specific) but NOT RoPE
    keys = attn.k_norm(keys)

    mx.eval(keys, values, h)

    if positions is not None:
        k_sel = keys[:, :, positions, :]
        v_sel = values[:, :, positions, :]
        mx.eval(k_sel, v_sel)
        return {
            "k_pre_rope": k_sel,
            "v": v_sel,
            "k_full_pre_rope": keys,
            "v_full": values,
            "hidden_pre_layer": h,
            "attn": attn,
        }
    else:
        return {
            "k_pre_rope": keys,
            "v": values,
            "k_full_pre_rope": keys,
            "v_full": values,
            "hidden_pre_layer": h,
            "attn": attn,
        }


def apply_rope_at_positions(attn, k, positions):
    """
    Apply RoPE to K vectors at specified position indices.

    k: (batch, heads, n_entries, head_dim)
    positions: list of position indices, len = n_entries

    Returns k with RoPE applied as if entries were at those positions.
    """
    batch, heads, n_entries, head_dim = k.shape

    # Create a dummy tensor at the right positions
    # RoPE in MLX typically uses the sequence position dimension
    # We need to apply RoPE one entry at a time at different positions

    # Actually, the Gemma RoPE implementation uses the sequence length
    # to determine positions (0, 1, 2, ..., seq_len-1).
    # We need to pad to max_position and select.

    max_pos = max(positions) + 1

    # Create a padded tensor
    k_padded = mx.zeros((batch, heads, max_pos, head_dim), dtype=k.dtype)

    # Place entries at their target positions
    for i, pos in enumerate(positions):
        k_padded = k_padded.at[:, :, pos, :].add(k[:, :, i, :])

    # Apply RoPE to full padded sequence
    k_roped = attn.rope(k_padded)

    # Extract at target positions
    k_result = k_roped[:, :, positions, :]
    mx.eval(k_result)

    return k_result


def apply_rope_at_positions_v2(attn, k, positions):
    """
    Apply RoPE at specific positions by computing sin/cos directly.
    More memory efficient than padding.

    k: (batch, heads, n_entries, head_dim) - pre-RoPE
    positions: list of int positions
    """
    batch, heads, n_entries, head_dim = k.shape

    # For Gemma, RoPE is applied to pairs of dimensions
    # The rope object has a method that applies based on sequence position
    # Let's use the rope directly but with offset

    # Build entries one at a time
    results = []
    for i, pos in enumerate(positions):
        # Create single-entry tensor and apply rope with offset
        k_single = k[:, :, i:i+1, :]  # (batch, heads, 1, head_dim)
        # Apply rope - it applies to positions 0..seq_len-1
        # We need to use offset to shift to the correct position
        k_roped = attn.rope(k_single, offset=pos)
        results.append(k_roped)

    k_out = mx.concatenate(results, axis=2)
    mx.eval(k_out)
    return k_out


def forward_with_extended_kv(model, query_ids, stored_k_pre_rope, stored_v,
                              attn_obj, inject_layer=29,
                              stored_positions=None,
                              position_mode="sequential"):
    """
    Run the model on query tokens, but at inject_layer, extend the KV cache
    with stored entries IN THE SAME RoPE FRAME.

    Key difference from prior experiment:
    - Stored K entries are PRE-RoPE
    - RoPE is applied at sequential positions BEFORE the query
    - Query tokens get shifted RoPE positions

    Args:
        model: The loaded model
        query_ids: (1, query_len) input token IDs
        stored_k_pre_rope: (1, kv_heads, N, head_dim) - pre-RoPE K
        stored_v: (1, kv_heads, N, head_dim) - V vectors
        attn_obj: The attention module (for RoPE)
        inject_layer: Layer at which to inject
        stored_positions: Custom positions for stored entries (None = sequential from 0)
        position_mode: "sequential" (0..N-1, query N..N+M-1),
                       "before_query" (query_start-N..query_start-1, query 0..M-1),
                       "original" (keep original positions),
                       "no_rope" (no RoPE on stored entries)

    Returns dict with probs, attention weights, etc.
    """
    backbone = model.model
    config = backbone.config

    # Embed + scale
    h = backbone.embed_tokens(query_ids)
    h = h * mx.array(config.hidden_size ** 0.5, dtype=mx.bfloat16).astype(h.dtype)

    query_len = h.shape[1]
    n_stored = stored_k_pre_rope.shape[2]

    # Masks for L0..L28
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

    # === Layer inject_layer: Manual decomposition with KV extension ===
    layer = backbone.layers[inject_layer]
    attn = layer.self_attn
    residual = h

    # Input layernorm
    normed = layer.input_layernorm(h)
    batch_size = 1

    # Q/K/V projections for query
    queries = attn.q_proj(normed)
    keys_q = attn.k_proj(normed)
    values_q = attn.v_proj(normed)

    # Reshape
    queries = queries.reshape(batch_size, query_len, attn.num_heads, attn.head_dim)
    queries = queries.transpose(0, 2, 1, 3)
    keys_q = keys_q.reshape(batch_size, query_len, attn.num_kv_heads, attn.head_dim)
    keys_q = keys_q.transpose(0, 2, 1, 3)
    values_q = values_q.reshape(batch_size, query_len, attn.num_kv_heads, attn.head_dim)
    values_q = values_q.transpose(0, 2, 1, 3)

    # Q/K normalization
    queries = attn.q_norm(queries)
    keys_q = attn.k_norm(keys_q)

    # === Apply RoPE based on position_mode ===
    if position_mode == "sequential":
        # Stored entries: positions 0..N-1
        # Query tokens: positions N..N+M-1

        # Apply RoPE to stored K at positions 0..N-1
        stored_k_roped = attn.rope(stored_k_pre_rope)  # positions 0..N-1 (default)

        # Apply RoPE to query Q and K at positions N..N+M-1
        queries = attn.rope(queries, offset=n_stored)
        keys_q = attn.rope(keys_q, offset=n_stored)

    elif position_mode == "before_query":
        # Query at positions 0..M-1 (natural)
        # Stored at positions M..M+N-1 (after query)
        # But placed BEFORE in KV cache
        queries = attn.rope(queries)
        keys_q = attn.rope(keys_q)
        stored_k_roped = attn.rope(stored_k_pre_rope, offset=query_len)

    elif position_mode == "original":
        # Keep original RoPE (stored K already has original positions)
        # This is what the prior experiment did
        # Note: stored_k_pre_rope is actually PRE-rope here
        # We need the caller to pass post-RoPE K for this mode
        stored_k_roped = stored_k_pre_rope  # Assume already has RoPE
        queries = attn.rope(queries)
        keys_q = attn.rope(keys_q)

    elif position_mode == "no_rope":
        # No RoPE on stored entries at all
        stored_k_roped = stored_k_pre_rope
        queries = attn.rope(queries)
        keys_q = attn.rope(keys_q)

    elif position_mode == "matched_relative":
        # Match the relative distance from the full context
        # In full context, query (pos 53) attends to Volt (pos 28)
        # Relative distance = 25
        # Here: query at pos 0..M-1, stored at negative offsets
        # Use query_len + some offset for stored entries
        queries = attn.rope(queries, offset=n_stored)
        keys_q = attn.rope(keys_q, offset=n_stored)
        stored_k_roped = attn.rope(stored_k_pre_rope)  # 0..N-1

    else:
        raise ValueError(f"Unknown position_mode: {position_mode}")

    # === Build extended KV cache ===
    # Place stored entries BEFORE query entries in the KV cache
    keys_ext = mx.concatenate([stored_k_roped, keys_q], axis=2)    # (1, kv_heads, N+M, head_dim)
    values_ext = mx.concatenate([stored_v, values_q], axis=2)       # (1, kv_heads, N+M, head_dim)

    total_len = n_stored + query_len

    # GQA: repeat KV heads
    if attn.n_rep > 1:
        keys_ext = mx.repeat(keys_ext, attn.n_rep, axis=1)
        values_ext = mx.repeat(values_ext, attn.n_rep, axis=1)

    # Scaled dot-product attention (manual, to extract weights)
    scores = (queries @ keys_ext.transpose(0, 1, 3, 2)) * attn.scale
    # scores: (1, num_heads, query_len, N+M)

    # Create attention mask
    # Stored entries (first N columns): all query positions can attend
    # Query entries (last M columns): causal mask
    stored_cols = mx.zeros((query_len, n_stored), dtype=scores.dtype)
    causal = nn.MultiHeadAttention.create_additive_causal_mask(query_len)
    causal = causal.astype(scores.dtype)
    extended_mask = mx.concatenate([stored_cols, causal], axis=1)  # (M, N+M)

    scores = scores + extended_mask
    weights = mx.softmax(scores, axis=-1)

    # Extract attention to stored entries (last query position)
    attn_to_stored = weights[0, :, -1, :n_stored]     # (num_heads, N)
    attn_to_query = weights[0, :, -1, n_stored:]       # (num_heads, M)

    # Weighted sum
    context = weights @ values_ext  # (1, num_heads, query_len, head_dim)

    # Transpose back and project
    context = context.transpose(0, 2, 1, 3).reshape(batch_size, query_len, -1)
    attn_output = attn.o_proj(context)

    # Post-attention layernorm + residual
    h = clip_residual(residual, layer.post_attention_layernorm(attn_output))

    # FFN
    ffn_out = layer.mlp(layer.pre_feedforward_layernorm(h))
    h = clip_residual(h, layer.post_feedforward_layernorm(ffn_out))

    # === Continue through remaining layers normally ===
    # Need new masks for the query-length sequence
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

    mx.eval(probs, attn_to_stored, attn_to_query)

    return {
        "probs": probs,
        "attn_to_stored": attn_to_stored,  # (num_heads, N)
        "attn_to_query": attn_to_query,    # (num_heads, M)
    }


def forward_to_layer(model, input_ids, stop_after_layer):
    """Run model forward through layers 0..stop_after_layer."""
    backbone = model.model
    config = backbone.config

    h = backbone.embed_tokens(input_ids)
    h = h * mx.array(config.hidden_size ** 0.5, dtype=mx.bfloat16).astype(h.dtype)

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


def extract_kv_post_rope(model, input_ids, layer_idx, positions=None):
    """Extract K/V WITH RoPE already applied (for 'original' position mode)."""
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
    keys = attn.rope(keys)  # WITH RoPE

    mx.eval(keys, values)

    if positions is not None:
        k_sel = keys[:, :, positions, :]
        v_sel = values[:, :, positions, :]
        mx.eval(k_sel, v_sel)
        return {"k": k_sel, "v": v_sel, "k_full": keys, "v_full": values}
    return {"k": keys, "v": values, "k_full": keys, "v_full": values}


def report_results(tokenizer, probs, attn_to_stored, target_ids, stored_tokens=None):
    """Pretty-print results."""
    top_ids = mx.argsort(probs)[-5:][::-1].tolist()
    top5 = [(tokenizer.decode([t]), float(probs[t])) for t in top_ids]

    result = {"top5": top5}
    for name, tid in target_ids.items():
        p = float(probs[tid])
        result[f"P_{name}"] = p
        print(f"  P({name}): {p:.6f} ({p*100:.2f}%)")
    print(f"  Top 5: {top5}")

    if attn_to_stored is not None:
        n_stored = attn_to_stored.shape[1]
        print(f"\n  Attention to stored entries per head (N={n_stored}):")
        result["attn_per_head"] = {}
        for h_idx in range(8):
            total = float(mx.sum(attn_to_stored[h_idx, :]))
            best_idx = int(mx.argmax(attn_to_stored[h_idx, :]))
            best_val = float(attn_to_stored[h_idx, best_idx])
            tok_str = stored_tokens[best_idx] if stored_tokens else f"entry_{best_idx}"
            print(f"    H{h_idx}: total={total:.4f}, best=entry {best_idx} ('{tok_str}', {best_val:.4f})")
            result["attn_per_head"][f"H{h_idx}"] = {
                "total": total, "best_idx": best_idx,
                "best_token": tok_str, "best_weight": best_val
            }

    return result


def main():
    model, tokenizer = load_model()

    # Prompts
    document = "Zarkov Industries, a cutting-edge robotics company, was founded in 2019 in the city of Voltara by engineer Dimitri Zarkov."
    full_context = f"<start_of_turn>user\n{document}\n\nThe city where Zarkov Industries was founded is called<end_of_turn>\n<start_of_turn>model\n"
    bare_query = "<start_of_turn>user\nThe city where Zarkov Industries was founded is called<end_of_turn>\n<start_of_turn>model\n"

    targets = {"Volt": 194328, "The": 818, "Z": 236953}

    full_ids = mx.array([tokenizer.encode(full_context, add_special_tokens=True)])
    bare_ids = mx.array([tokenizer.encode(bare_query, add_special_tokens=True)])

    full_tokens = tokenizer.convert_ids_to_tokens(full_ids[0].tolist())
    bare_tokens = tokenizer.convert_ids_to_tokens(bare_ids[0].tolist())

    print(f"Full context: {len(full_tokens)} tokens")
    print(f"Bare query: {len(bare_tokens)} tokens")

    # Find Volt position
    volt_pos = None
    for i, tok in enumerate(full_tokens):
        if "Volt" in tok:
            volt_pos = i
            print(f"Volt at position {i}: '{tok}'")
            break

    results = {}

    # ================================================================
    # BASELINES
    # ================================================================
    print("\n" + "=" * 60)
    print("BASELINE: Full context")
    print("=" * 60)
    r = native_predict(model, tokenizer, full_context, targets)
    print(f"  P(Volt): {r['P_Volt']:.6f}")
    print(f"  Top: {r['top10'][:5]}")
    results["baseline_full"] = r

    print("\n" + "=" * 60)
    print("BASELINE: Bare query")
    print("=" * 60)
    r = native_predict(model, tokenizer, bare_query, targets)
    print(f"  P(Volt): {r.get('P_Volt', 0):.6f}")
    print(f"  P(Z): {r['P_Z']:.6f}")
    print(f"  Top: {r['top10'][:5]}")
    results["baseline_bare"] = r

    # ================================================================
    # EXPERIMENT 1a: Extract pre-RoPE K/V from full context at L29
    # ================================================================
    print("\n" + "=" * 60)
    print("EXP 1a: Extract pre-RoPE K/V at L29 from full context")
    print("=" * 60)

    kv_pre = extract_kv_pre_rope(model, full_ids, layer_idx=29, positions=[volt_pos])
    print(f"  K (pre-RoPE) shape: {kv_pre['k_pre_rope'].shape}")  # (1, 4, 1, 256)
    print(f"  V shape: {kv_pre['v'].shape}")

    # Also extract post-RoPE for comparison
    kv_post = extract_kv_post_rope(model, full_ids, layer_idx=29, positions=[volt_pos])

    # K-norms
    k_norms_pre = mx.sqrt(mx.sum(kv_pre['k_full_pre_rope'] ** 2, axis=-1))
    k_norms_mean = mx.mean(k_norms_pre, axis=1)
    mx.eval(k_norms_mean)
    print(f"  K-norm at Volt (pos {volt_pos}): {float(k_norms_mean[0, volt_pos]):.2f}")

    results["exp1a"] = {
        "volt_pos": volt_pos,
        "k_norm_volt": float(k_norms_mean[0, volt_pos]),
    }

    # ================================================================
    # EXPERIMENT 1b: Single entry, SEQUENTIAL position mode
    # ================================================================
    print("\n" + "=" * 60)
    print("EXP 1b: Single Volt K/V entry — SEQUENTIAL positions")
    print("=" * 60)
    print("  Stored entry at pos 0, query at pos 1..20")

    r = forward_with_extended_kv(
        model, bare_ids, kv_pre['k_pre_rope'], kv_pre['v'],
        kv_pre['attn'], inject_layer=29, position_mode="sequential"
    )
    res = report_results(tokenizer, r["probs"], r["attn_to_stored"], targets,
                         stored_tokens=[full_tokens[volt_pos]])
    results["exp1b_sequential"] = res

    # ================================================================
    # EXPERIMENT 1c: Single entry, ORIGINAL position mode (prior experiment)
    # ================================================================
    print("\n" + "=" * 60)
    print("EXP 1c: Single Volt K/V entry — ORIGINAL positions (post-RoPE)")
    print("=" * 60)
    print(f"  Stored entry keeps RoPE from pos {volt_pos}, query at pos 0..19")

    r = forward_with_extended_kv(
        model, bare_ids, kv_post['k'], kv_post['v'],
        kv_post['attn'] if hasattr(kv_post, 'attn') else kv_pre['attn'],
        inject_layer=29, position_mode="original"
    )
    res = report_results(tokenizer, r["probs"], r["attn_to_stored"], targets,
                         stored_tokens=[full_tokens[volt_pos]])
    results["exp1c_original"] = res

    # ================================================================
    # EXPERIMENT 1d: Single entry, NO RoPE on stored
    # ================================================================
    print("\n" + "=" * 60)
    print("EXP 1d: Single Volt K/V entry — NO RoPE on stored")
    print("=" * 60)

    r = forward_with_extended_kv(
        model, bare_ids, kv_pre['k_pre_rope'], kv_pre['v'],
        kv_pre['attn'], inject_layer=29, position_mode="no_rope"
    )
    res = report_results(tokenizer, r["probs"], r["attn_to_stored"], targets,
                         stored_tokens=[full_tokens[volt_pos]])
    results["exp1d_no_rope"] = res

    # ================================================================
    # EXPERIMENT 1e: Single entry, BEFORE_QUERY mode
    # ================================================================
    print("\n" + "=" * 60)
    print("EXP 1e: Single Volt K/V entry — BEFORE_QUERY positions")
    print("=" * 60)
    print(f"  Query at pos 0..19, stored at pos 20")

    r = forward_with_extended_kv(
        model, bare_ids, kv_pre['k_pre_rope'], kv_pre['v'],
        kv_pre['attn'], inject_layer=29, position_mode="before_query"
    )
    res = report_results(tokenizer, r["probs"], r["attn_to_stored"], targets,
                         stored_tokens=[full_tokens[volt_pos]])
    results["exp1e_before_query"] = res

    # ================================================================
    # EXPERIMENT 1f: Single entry, MATCHED_RELATIVE mode
    # ================================================================
    print("\n" + "=" * 60)
    print("EXP 1f: Single Volt K/V entry — MATCHED_RELATIVE positions")
    print("=" * 60)
    print(f"  Stored at pos 0, query at pos 1..20 (same relative gaps)")

    r = forward_with_extended_kv(
        model, bare_ids, kv_pre['k_pre_rope'], kv_pre['v'],
        kv_pre['attn'], inject_layer=29, position_mode="matched_relative"
    )
    res = report_results(tokenizer, r["probs"], r["attn_to_stored"], targets,
                         stored_tokens=[full_tokens[volt_pos]])
    results["exp1f_matched_relative"] = res

    # ================================================================
    # EXPERIMENT 2: ALL entries from full context, best position mode
    # ================================================================
    print("\n" + "=" * 60)
    print("EXP 2: ALL K/V entries from full context — position modes")
    print("=" * 60)

    # Extract full pre-RoPE K/V
    kv_all_pre = extract_kv_pre_rope(model, full_ids, layer_idx=29)
    kv_all_post = extract_kv_post_rope(model, full_ids, layer_idx=29)

    for mode in ["sequential", "no_rope", "matched_relative"]:
        print(f"\n  --- Position mode: {mode} ---")

        if mode == "original":
            k_use = kv_all_post['k']
            v_use = kv_all_post['v']
        else:
            k_use = kv_all_pre['k_pre_rope']
            v_use = kv_all_pre['v']

        r = forward_with_extended_kv(
            model, bare_ids, k_use, v_use,
            kv_all_pre['attn'], inject_layer=29, position_mode=mode
        )

        p_volt = float(r["probs"][194328])
        top_ids = mx.argsort(r["probs"])[-5:][::-1].tolist()
        top5 = [(tokenizer.decode([t]), float(r["probs"][t])) for t in top_ids]

        print(f"  P(Volt): {p_volt:.6f} ({p_volt*100:.2f}%)")
        print(f"  Top 5: {top5}")

        # H4 attention to Volt
        h4_stored = r["attn_to_stored"][4, :]
        mx.eval(h4_stored)
        h4_volt = float(h4_stored[volt_pos])
        h4_best = int(mx.argmax(h4_stored))
        h4_total = float(mx.sum(h4_stored))
        print(f"  H4: total to stored={h4_total:.4f}, Volt attn={h4_volt:.4f}")
        print(f"  H4 best: entry {h4_best} ('{full_tokens[h4_best]}', {float(h4_stored[h4_best]):.4f})")

        results[f"exp2_{mode}"] = {
            "P_Volt": p_volt, "top5": top5,
            "H4_volt_attn": h4_volt, "H4_total_stored": h4_total,
            "H4_best_entry": h4_best, "H4_best_token": full_tokens[h4_best],
        }

    # Also test original (post-RoPE) for comparison
    print(f"\n  --- Position mode: original (prior experiment baseline) ---")
    r = forward_with_extended_kv(
        model, bare_ids, kv_all_post['k'], kv_all_post['v'],
        kv_all_pre['attn'], inject_layer=29, position_mode="original"
    )
    p_volt = float(r["probs"][194328])
    top_ids = mx.argsort(r["probs"])[-5:][::-1].tolist()
    top5 = [(tokenizer.decode([t]), float(r["probs"][t])) for t in top_ids]
    print(f"  P(Volt): {p_volt:.6f} ({p_volt*100:.2f}%)")
    print(f"  Top 5: {top5}")
    h4_stored = r["attn_to_stored"][4, :]
    mx.eval(h4_stored)
    h4_volt = float(h4_stored[volt_pos])
    h4_total = float(mx.sum(h4_stored))
    print(f"  H4: total to stored={h4_total:.4f}, Volt attn={h4_volt:.4f}")
    results["exp2_original"] = {
        "P_Volt": p_volt, "top5": top5,
        "H4_volt_attn": h4_volt, "H4_total_stored": h4_total,
    }

    # ================================================================
    # EXPERIMENT 3: Best mode + 12-byte injection at L30
    # ================================================================
    print("\n" + "=" * 60)
    print("EXP 3: Hybrid — KV extension at L29 + residual injection at L30")
    print("=" * 60)

    # Get full-context residual at L30 for injection
    donor_h_L30 = forward_to_layer(model, full_ids, stop_after_layer=30)
    donor_last = donor_h_L30[:, -1:, :]

    # Get bare query residual at L30 (for comparison)
    bare_h_L30 = forward_to_layer(model, bare_ids, stop_after_layer=30)

    # Inject donor's L30 residual into bare query at L31+
    h_injected = mx.concatenate([bare_h_L30[:, :-1, :], donor_last], axis=1)

    backbone = model.model
    config = backbone.config
    h = h_injected
    global_mask = backbone._create_attention_mask(h, None)
    sliding_mask = (
        backbone._create_attention_mask(h, None, window_size=backbone.sliding_window)
        if backbone.sliding_window_pattern > 1 else None
    )

    for i in range(31, len(backbone.layers)):
        layer_i = backbone.layers[i]
        is_global = config.is_global_layer(i)
        mask = global_mask if is_global else sliding_mask
        output = layer_i(h, mask=mask, cache=None)
        h = output.hidden_states

    h = backbone.norm(h)
    logits = model.lm_head(h) if not model.tie_word_embeddings else backbone.embed_tokens.as_linear(h)
    probs = mx.softmax(logits[0, -1, :])
    mx.eval(probs)

    p_volt = float(probs[194328])
    top_ids = mx.argsort(probs)[-5:][::-1].tolist()
    top5 = [(tokenizer.decode([t]), float(probs[t])) for t in top_ids]
    print(f"  Full-context L30 residual injected at L31+:")
    print(f"  P(Volt): {p_volt:.6f} ({p_volt*100:.2f}%)")
    print(f"  Top 5: {top5}")
    results["exp3_L30_injection"] = {"P_Volt": p_volt, "top5": top5}

    # ================================================================
    # SUMMARY TABLE
    # ================================================================
    print("\n" + "=" * 60)
    print("SUMMARY TABLE")
    print("=" * 60)

    summary_rows = [
        ("Full context (baseline)", "baseline_full", "P_Volt"),
        ("Bare query (baseline)", "baseline_bare", "P_Volt"),
        ("1 entry — sequential RoPE", "exp1b_sequential", "P_Volt"),
        ("1 entry — original RoPE (prior exp)", "exp1c_original", "P_Volt"),
        ("1 entry — no RoPE on stored", "exp1d_no_rope", "P_Volt"),
        ("1 entry — before_query RoPE", "exp1e_before_query", "P_Volt"),
        ("1 entry — matched_relative RoPE", "exp1f_matched_relative", "P_Volt"),
        ("ALL entries — sequential", "exp2_sequential", "P_Volt"),
        ("ALL entries — no_rope", "exp2_no_rope", "P_Volt"),
        ("ALL entries — matched_relative", "exp2_matched_relative", "P_Volt"),
        ("ALL entries — original (prior)", "exp2_original", "P_Volt"),
        ("L30 residual injection (L31+)", "exp3_L30_injection", "P_Volt"),
    ]

    print(f"\n{'Condition':<45} {'P(Volt)':>10} {'H4 attn':>10}")
    print("-" * 67)
    for label, key, pkey in summary_rows:
        if key in results:
            pv = results[key].get(pkey, 0)
            h4 = results[key].get("H4_volt_attn", results[key].get("attn_per_head", {}).get("H4", {}).get("best_weight", ""))
            h4_str = f"{h4:.4f}" if isinstance(h4, float) else str(h4)
            print(f"  {label:<43} {pv:>10.6f} {h4_str:>10}")

    # Save
    output_path = "/Users/christopherhay/chris-source/lazarus-play/experiments/kv_cache_extension/results.json"
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
