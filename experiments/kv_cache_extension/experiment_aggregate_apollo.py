"""
Attention-Aggregate Routing at Apollo Scale (N=50 windows)
==========================================================
Test multi-head aggregate routing with 50 Apollo transcript windows.
8 K-norm sampled entries per window = 400 total entries.

5 queries:
  porridge → W170
  baseball → W169
  landing  → W370
  weather  → W169
  news     → W169
"""

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import json
import time
from chuk_lazarus.inference import UnifiedPipeline, UnifiedPipelineConfig
from chuk_lazarus.models_v2.families.gemma.model import clip_residual


def load_model():
    config = UnifiedPipelineConfig()
    pipeline = UnifiedPipeline.from_pretrained("google/gemma-3-4b-it", pipeline_config=config)
    return pipeline.model, pipeline.tokenizer


def extract_kv_pre_rope(model, input_ids, layer_idx, positions=None):
    """Extract K/V at layer_idx BEFORE RoPE."""
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
    mx.eval(keys, values)
    if positions is not None:
        return {"k": keys[:, :, positions, :], "v": values[:, :, positions, :], "attn": attn}
    return {"k": keys, "v": values, "attn": attn}


def select_k_norm_positions(keys, n_select=8, exclude_first=1):
    """Select positions with highest K-norm (averaged across heads).
    Excludes first `exclude_first` positions (usually BOS)."""
    # keys shape: (1, n_heads, seq_len, head_dim)
    k_norms = mx.sqrt(mx.sum(keys[0] ** 2, axis=-1))  # (n_heads, seq_len)
    avg_norms = mx.mean(k_norms, axis=0)  # (seq_len,)
    mx.eval(avg_norms)

    norms_np = np.array(avg_norms.astype(mx.float32))
    # Exclude first positions
    norms_np[:exclude_first] = 0.0
    # Select top-n
    top_indices = np.argsort(norms_np)[::-1][:n_select]
    top_indices = sorted(top_indices.tolist())  # sort by position
    return top_indices, [float(norms_np[i]) for i in top_indices]


def build_window_store(model, tokenizer, window_text, window_key,
                       n_entries=8, layer=29):
    """Build K-vector store for a single window using K-norm sampling."""
    # Wrap in chat template
    prompt = f"<start_of_turn>user\n{window_text}<end_of_turn>\n<start_of_turn>model\n"
    input_ids = mx.array([tokenizer.encode(prompt, add_special_tokens=True)])

    # Extract all K/V at L29
    kv = extract_kv_pre_rope(model, input_ids, layer_idx=layer)

    # Select top-N by K-norm
    positions, norms = select_k_norm_positions(kv['k'], n_select=n_entries)

    # Extract selected positions
    k_selected = kv['k'][:, :, positions, :]
    v_selected = kv['v'][:, :, positions, :]
    mx.eval(k_selected, v_selected)

    tokens = tokenizer.convert_ids_to_tokens(input_ids[0].tolist())
    selected_tokens = [tokens[p] if p < len(tokens) else '?' for p in positions]

    return {
        "k": k_selected,
        "v": v_selected,
        "attn": kv['attn'],
        "window_key": window_key,
        "positions": positions,
        "k_norms": norms,
        "selected_tokens": selected_tokens,
        "n_entries": len(positions),
    }


def query_aggregate_routing(model, tokenizer, query_ids, all_k, all_v,
                            entry_to_window, n_windows, window_keys,
                            inject_layer=29):
    """
    Run query with combined KV store. Return per-window aggregate attention.
    """
    backbone = model.model
    config = backbone.config

    h = backbone.embed_tokens(query_ids)
    h = h * mx.array(config.hidden_size ** 0.5, dtype=mx.bfloat16).astype(h.dtype)

    query_len = h.shape[1]
    n_stored = all_k.shape[2]

    global_mask = backbone._create_attention_mask(h, None)
    sliding_mask = (
        backbone._create_attention_mask(h, None, window_size=backbone.sliding_window)
        if backbone.sliding_window_pattern > 1 else None
    )

    for i in range(len(backbone.layers)):
        layer = backbone.layers[i]
        is_global = config.is_global_layer(i)
        mask = global_mask if is_global else sliding_mask

        if i == inject_layer:
            attn = layer.self_attn
            residual = h
            normed = layer.input_layernorm(h)

            queries = attn.q_proj(normed).reshape(1, query_len, attn.num_heads, attn.head_dim).transpose(0, 2, 1, 3)
            keys_q = attn.k_proj(normed).reshape(1, query_len, attn.num_kv_heads, attn.head_dim).transpose(0, 2, 1, 3)
            values_q = attn.v_proj(normed).reshape(1, query_len, attn.num_kv_heads, attn.head_dim).transpose(0, 2, 1, 3)

            queries = attn.q_norm(queries)
            keys_q = attn.k_norm(keys_q)

            stored_k_roped = attn.rope(all_k)
            queries = attn.rope(queries, offset=n_stored)
            keys_q = attn.rope(keys_q, offset=n_stored)

            keys_ext = mx.concatenate([stored_k_roped, keys_q], axis=2)
            values_ext = mx.concatenate([all_v, values_q], axis=2)

            if attn.n_rep > 1:
                keys_ext = mx.repeat(keys_ext, attn.n_rep, axis=1)
                values_ext = mx.repeat(values_ext, attn.n_rep, axis=1)

            scores = (queries @ keys_ext.transpose(0, 1, 3, 2)) * attn.scale
            stored_cols = mx.zeros((query_len, n_stored), dtype=scores.dtype)
            causal = nn.MultiHeadAttention.create_additive_causal_mask(query_len).astype(scores.dtype)
            scores = scores + mx.concatenate([stored_cols, causal], axis=1)
            weights = mx.softmax(scores, axis=-1)

            # Full attention over stored entries: (8, n_stored)
            all_heads_attn = weights[0, :, -1, :n_stored]
            total_stored = mx.sum(weights[0, :, -1, :n_stored])
            total_query = mx.sum(weights[0, :, -1, n_stored:])
            mx.eval(all_heads_attn, total_stored, total_query)

            context = weights @ values_ext
            context = context.transpose(0, 2, 1, 3).reshape(1, query_len, -1)
            attn_output = attn.o_proj(context)

            h = clip_residual(residual, layer.post_attention_layernorm(attn_output))
            ffn_out = layer.mlp(layer.pre_feedforward_layernorm(h))
            h = clip_residual(h, layer.post_feedforward_layernorm(ffn_out))
        else:
            output = layer(h, mask=mask, cache=None)
            h = output.hidden_states

    all_heads_np = np.array(all_heads_attn.astype(mx.float32))  # (8, n_stored)

    # Compute per-window aggregates
    def window_aggregate(heads):
        """Sum attention across specified heads, per window."""
        window_totals = np.zeros(n_windows)
        for idx in range(n_stored):
            wi = entry_to_window[idx]
            for head in heads:
                window_totals[wi] += all_heads_np[head][idx]
        return window_totals

    h4_agg = window_aggregate([4])
    multi_agg = window_aggregate([2, 3, 4, 5])
    all_agg = window_aggregate(list(range(8)))

    # Also compute per-head argmax
    h4_only = all_heads_np[4]
    h4_argmax_idx = int(np.argmax(h4_only))
    h4_argmax_window = entry_to_window[h4_argmax_idx]

    return {
        "h4_argmax_window_idx": h4_argmax_window,
        "h4_argmax_window_key": window_keys[h4_argmax_window],
        "h4_argmax_weight": float(h4_only[h4_argmax_idx]),
        "h4_aggregate": h4_agg,
        "multi_head_aggregate": multi_agg,
        "all_head_aggregate": all_agg,
        "total_stored_attn": float(total_stored),
        "total_query_attn": float(total_query),
        "n_stored": n_stored,
    }


def main():
    model, tokenizer = load_model()

    # Load Apollo windows
    with open("archive/routing/vector_routing_windows.json") as f:
        win_data = json.load(f)
    with open("archive/routing/vector_routing_prompts.json") as f:
        prompt_data = json.load(f)

    windows = win_data['windows']
    window_keys = list(windows.keys())  # 50 keys
    n_windows = len(window_keys)

    targets = prompt_data['targets']
    query_prompts = prompt_data['query_prompts']

    print(f"Windows: {n_windows}, Queries: {len(query_prompts)}")
    print(f"Targets: {targets}")

    # ================================================================
    # Build stores for all 50 windows
    # ================================================================
    print("\nBuilding stores (8 K-norm entries per window)...")
    stores = []
    all_k_list = []
    all_v_list = []
    entry_to_window = []

    for wi, wkey in enumerate(window_keys):
        wtext = windows[wkey]
        # Strip any chat template wrapping that might be in the window text
        if wtext.startswith('<start_of_turn>'):
            # Already wrapped — extract the content
            raw_text = wtext.split('\n', 2)[-1] if '\n' in wtext else wtext
        else:
            raw_text = wtext

        store = build_window_store(model, tokenizer, raw_text, wkey, n_entries=8)
        stores.append(store)
        all_k_list.append(store['k'])
        all_v_list.append(store['v'])
        for _ in range(store['n_entries']):
            entry_to_window.append(wi)

        if (wi + 1) % 10 == 0:
            print(f"  Built {wi + 1}/{n_windows} stores "
                  f"({sum(len(s['positions']) for s in stores)} total entries)")

    # Combine all K/V
    combined_k = mx.concatenate(all_k_list, axis=2)
    combined_v = mx.concatenate(all_v_list, axis=2)
    mx.eval(combined_k, combined_v)
    total_entries = combined_k.shape[2]
    print(f"\nCombined: {total_entries} entries across {n_windows} windows")

    # ================================================================
    # Run queries
    # ================================================================
    results = {}

    for qname in ['porridge', 'baseball', 'landing', 'weather', 'news']:
        target_window = targets[qname]
        target_idx = window_keys.index(target_window)
        query_text = query_prompts[qname]
        query_ids = mx.array([tokenizer.encode(query_text, add_special_tokens=False)])

        print(f"\n{'─' * 70}")
        print(f"Query: {qname} (target: {target_window}, idx={target_idx})")
        print(f"{'─' * 70}")

        r = query_aggregate_routing(
            model, tokenizer, query_ids,
            combined_k, combined_v,
            entry_to_window, n_windows, window_keys
        )

        # Results for each method
        for method_name, agg_array in [
            ("H4 argmax", None),
            ("H4 aggregate", r['h4_aggregate']),
            ("Multi-head agg (H2-H5)", r['multi_head_aggregate']),
            ("All-head agg (H0-H7)", r['all_head_aggregate']),
        ]:
            if agg_array is None:
                # argmax
                winner_key = r['h4_argmax_window_key']
                correct = winner_key == target_window
                print(f"  {method_name}: → {winner_key} "
                      f"(weight={r['h4_argmax_weight']:.4f}) "
                      f"{'✓' if correct else '✗'}")
            else:
                winner_idx = int(np.argmax(agg_array))
                winner_key = window_keys[winner_idx]
                correct = winner_key == target_window

                # Rank of correct window
                sorted_indices = np.argsort(agg_array)[::-1]
                correct_rank = int(np.where(sorted_indices == target_idx)[0][0]) + 1

                # Margin
                sorted_vals = agg_array[sorted_indices]
                margin = sorted_vals[0] - sorted_vals[1] if len(sorted_vals) > 1 else 0
                ratio = sorted_vals[0] / sorted_vals[1] if sorted_vals[1] > 0 else float('inf')

                # Top 5
                top5 = [(window_keys[sorted_indices[i]], float(sorted_vals[i]))
                        for i in range(min(5, len(sorted_vals)))]

                print(f"  {method_name}: → {winner_key} "
                      f"(correct rank={correct_rank}/50, margin={margin:.4f}, {ratio:.2f}×) "
                      f"{'✓' if correct else '✗'}")
                if not correct:
                    print(f"    Top-5: {top5}")
                    print(f"    Target {target_window} score: {agg_array[target_idx]:.4f}")

        print(f"  Attention budget: stored={r['total_stored_attn']:.1%}, query={r['total_query_attn']:.1%}")

        results[qname] = {
            "target": target_window,
            "target_idx": target_idx,
            "h4_argmax_window": r['h4_argmax_window_key'],
            "h4_argmax_correct": r['h4_argmax_window_key'] == target_window,
            "h4_aggregate_winner": window_keys[int(np.argmax(r['h4_aggregate']))],
            "h4_aggregate_correct": window_keys[int(np.argmax(r['h4_aggregate']))] == target_window,
            "h4_aggregate_target_rank": int(np.where(np.argsort(r['h4_aggregate'])[::-1] == target_idx)[0][0]) + 1,
            "multi_agg_winner": window_keys[int(np.argmax(r['multi_head_aggregate']))],
            "multi_agg_correct": window_keys[int(np.argmax(r['multi_head_aggregate']))] == target_window,
            "multi_agg_target_rank": int(np.where(np.argsort(r['multi_head_aggregate'])[::-1] == target_idx)[0][0]) + 1,
            "all_agg_winner": window_keys[int(np.argmax(r['all_head_aggregate']))],
            "all_agg_correct": window_keys[int(np.argmax(r['all_head_aggregate']))] == target_window,
            "all_agg_target_rank": int(np.where(np.argsort(r['all_head_aggregate'])[::-1] == target_idx)[0][0]) + 1,
            "attention_budget_stored": float(r['total_stored_attn']),
            "attention_budget_query": float(r['total_query_attn']),
            "h4_aggregate_scores": {window_keys[i]: float(r['h4_aggregate'][i]) for i in range(n_windows)},
            "multi_agg_scores": {window_keys[i]: float(r['multi_head_aggregate'][i]) for i in range(n_windows)},
        }

    # ================================================================
    # Summary
    # ================================================================
    print("\n" + "=" * 80)
    print("SUMMARY — Apollo N=50")
    print("=" * 80)

    methods = [
        ("Entry argmax (H4)", "h4_argmax_correct"),
        ("H4 aggregate", "h4_aggregate_correct"),
        ("Multi-head agg", "multi_agg_correct"),
        ("All-head agg", "all_agg_correct"),
    ]

    print(f"\n{'Method':<25} {'porridge':>10} {'baseball':>10} {'landing':>10} {'weather':>10} {'news':>8} {'Score':>8}")
    print("─" * 85)
    for method_name, key in methods:
        scores = []
        for qname in ['porridge', 'baseball', 'landing', 'weather', 'news']:
            c = results[qname][key]
            scores.append("✓" if c else "✗")
        total = sum(1 for qname in results if results[qname][key])
        print(f"{method_name:<25} {scores[0]:>10} {scores[1]:>10} {scores[2]:>10} {scores[3]:>10} {scores[4]:>8} {total}/5")

    # Target ranks
    print(f"\nTarget window ranks (lower=better):")
    for method_name, key in [("H4 agg", "h4_aggregate_target_rank"),
                              ("Multi agg", "multi_agg_target_rank"),
                              ("All agg", "all_agg_target_rank")]:
        ranks = [results[q][key] for q in ['porridge', 'baseball', 'landing', 'weather', 'news']]
        avg_rank = sum(ranks) / len(ranks)
        print(f"  {method_name}: {ranks} (avg={avg_rank:.1f})")

    # Save
    output = {
        "experiment": "attention-aggregate-routing-apollo",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "n_windows": n_windows,
        "entries_per_window": 8,
        "total_entries": total_entries,
        "sampling": "K-norm top-8",
        "results": results,
    }
    with open("experiment_aggregate_apollo_results.json", "w") as f:
        json.dump(output, f, indent=2)
    print("\nResults saved to experiment_aggregate_apollo_results.json")


if __name__ == "__main__":
    main()
