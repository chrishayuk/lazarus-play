"""
Attention-Aggregate Routing: Sum H4 attention per passage instead of argmax per entry.
====================================================================================
Hypothesis: Individual entries are dominated by K-norm bias, but passage-level
aggregates are query-dependent. Sum across 32-40 entries per passage dilutes
K-norm attractors and reveals content signal.

Prior result: H4 argmax at N=3 (116 entries) → 1/3 correct.
             Store-level totals at N=2 discriminate (0.222 vs 0.130).
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
    """Extract K/V at layer_idx BEFORE RoPE. Returns pre-RoPE K and V."""
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
        return {"k": keys[:, :, positions, :], "v": values[:, :, positions, :], "attn": attn}
    return {"k": keys, "v": values, "attn": attn}


def build_store(model, tokenizer, document, answer_token_id, content_positions,
                full_ids, layer=29, inject_layer=30, scale=2.0):
    """Build K-vector store + injection coefficient for a passage."""
    backbone = model.model
    kv = extract_kv_pre_rope(model, full_ids, layer_idx=layer, positions=content_positions)

    embed = backbone.embed_tokens.weight[answer_token_id]
    embed_norm_sq = mx.sum(embed * embed)

    h = backbone.embed_tokens(full_ids)
    h = h * mx.array(backbone.config.hidden_size ** 0.5, dtype=mx.bfloat16).astype(h.dtype)
    global_mask = backbone._create_attention_mask(h, None)
    sliding_mask = (
        backbone._create_attention_mask(h, None, window_size=backbone.sliding_window)
        if backbone.sliding_window_pattern > 1 else None
    )
    for i in range(inject_layer + 1):
        layer_i = backbone.layers[i]
        is_global = backbone.config.is_global_layer(i)
        mask = global_mask if is_global else sliding_mask
        output = layer_i(h, mask=mask, cache=None)
        h = output.hidden_states
    donor_residual = h[0, -1, :]
    coeff = mx.sum(donor_residual * embed) / (embed_norm_sq + 1e-8)
    scaled_coeff = scale * coeff
    mx.eval(kv['k'], kv['v'], embed, coeff, scaled_coeff)

    return {
        "k": kv['k'],
        "v": kv['v'],
        "attn": kv['attn'],
        "answer_token_id": answer_token_id,
        "answer_embed": embed,
        "coefficient": float(coeff),
        "scaled_coefficient": float(scaled_coeff),
        "content_positions": content_positions,
    }


def combine_stores(*stores):
    """Combine multiple stores into one. Track origin of each entry."""
    all_k = []
    all_v = []
    entry_map = []
    for si, store in enumerate(stores):
        n = store['k'].shape[2]
        all_k.append(store['k'])
        all_v.append(store['v'])
        for j in range(n):
            entry_map.append((si, j))
    combined_k = mx.concatenate(all_k, axis=2)
    combined_v = mx.concatenate(all_v, axis=2)
    mx.eval(combined_k, combined_v)
    return combined_k, combined_v, entry_map


def query_full_attention(model, tokenizer, query_ids, combined_k, combined_v,
                         entry_map, stores, inject_layer=29, delivery_layer=30,
                         scale=2.0):
    """
    Run query through combined store. Return FULL attention matrix (8 heads × N entries).
    Also inject winning entry and generate logits.
    """
    backbone = model.model
    config = backbone.config

    h = backbone.embed_tokens(query_ids)
    h = h * mx.array(config.hidden_size ** 0.5, dtype=mx.bfloat16).astype(h.dtype)

    query_len = h.shape[1]
    n_stored = combined_k.shape[2]

    global_mask = backbone._create_attention_mask(h, None)
    sliding_mask = (
        backbone._create_attention_mask(h, None, window_size=backbone.sliding_window)
        if backbone.sliding_window_pattern > 1 else None
    )

    all_heads_attn = None

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

            stored_k_roped = attn.rope(combined_k)
            queries = attn.rope(queries, offset=n_stored)
            keys_q = attn.rope(keys_q, offset=n_stored)

            keys_ext = mx.concatenate([stored_k_roped, keys_q], axis=2)
            values_ext = mx.concatenate([combined_v, values_q], axis=2)

            if attn.n_rep > 1:
                keys_ext = mx.repeat(keys_ext, attn.n_rep, axis=1)
                values_ext = mx.repeat(values_ext, attn.n_rep, axis=1)

            scores = (queries @ keys_ext.transpose(0, 1, 3, 2)) * attn.scale
            stored_cols = mx.zeros((query_len, n_stored), dtype=scores.dtype)
            causal = nn.MultiHeadAttention.create_additive_causal_mask(query_len).astype(scores.dtype)
            scores = scores + mx.concatenate([stored_cols, causal], axis=1)
            weights = mx.softmax(scores, axis=-1)

            # Full attention matrix: (8, n_stored) at last query token
            all_heads_attn = weights[0, :, -1, :n_stored]  # (8, n_stored)
            # Also get total attention to stored vs query tokens
            total_stored = mx.sum(weights[0, :, -1, :n_stored], axis=-1)  # (8,)
            total_query = mx.sum(weights[0, :, -1, n_stored:], axis=-1)   # (8,)
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

    h = backbone.norm(h)
    if model.tie_word_embeddings:
        logits = backbone.embed_tokens.as_linear(h)
    else:
        logits = model.lm_head(h)

    probs = mx.softmax(logits[0, -1, :])
    mx.eval(probs)

    all_heads_np = np.array(all_heads_attn.astype(mx.float32))  # (8, n_stored)
    total_stored_np = np.array(total_stored.astype(mx.float32))
    total_query_np = np.array(total_query.astype(mx.float32))

    return {
        "all_heads_attn": all_heads_np,  # (8, n_stored)
        "total_stored_per_head": total_stored_np,  # (8,)
        "total_query_per_head": total_query_np,    # (8,)
        "probs": probs,
        "entry_map": entry_map,
        "n_stored": n_stored,
    }


def compute_aggregates(all_heads_np, entry_map, n_stores):
    """Compute per-store aggregate attention for various head combinations."""
    n_entries = all_heads_np.shape[1]

    results = {}

    # 1. H4 only — per-entry argmax
    h4 = all_heads_np[4]
    h4_argmax = int(np.argmax(h4))
    h4_argmax_store = entry_map[h4_argmax][0]

    # 2. H4 aggregate — sum per store
    h4_store_totals = np.zeros(n_stores)
    for idx in range(n_entries):
        si = entry_map[idx][0]
        h4_store_totals[si] += h4[idx]
    h4_agg_winner = int(np.argmax(h4_store_totals))

    # 3. Multi-head aggregate (H2+H3+H4+H5)
    multi_heads = [2, 3, 4, 5]
    multi_store_totals = np.zeros(n_stores)
    for head in multi_heads:
        for idx in range(n_entries):
            si = entry_map[idx][0]
            multi_store_totals[si] += all_heads_np[head][idx]
    multi_agg_winner = int(np.argmax(multi_store_totals))

    # 4. All-head aggregate (H0-H7)
    all_store_totals = np.zeros(n_stores)
    for head in range(8):
        for idx in range(n_entries):
            si = entry_map[idx][0]
            all_store_totals[si] += all_heads_np[head][idx]
    all_agg_winner = int(np.argmax(all_store_totals))

    # 5. Per-head argmax winners
    head_argmax_winners = []
    for head in range(8):
        head_attn = all_heads_np[head]
        win_idx = int(np.argmax(head_attn))
        head_argmax_winners.append(entry_map[win_idx][0])

    # 6. Per-head aggregate winners
    head_agg_winners = []
    for head in range(8):
        head_store_totals = np.zeros(n_stores)
        for idx in range(n_entries):
            si = entry_map[idx][0]
            head_store_totals[si] += all_heads_np[head][idx]
        head_agg_winners.append(int(np.argmax(head_store_totals)))

    return {
        "h4_argmax_entry": h4_argmax,
        "h4_argmax_store": h4_argmax_store,
        "h4_argmax_weight": float(h4[h4_argmax]),
        "h4_store_totals": [float(x) for x in h4_store_totals],
        "h4_agg_winner": h4_agg_winner,
        "multi_head_store_totals": [float(x) for x in multi_store_totals],
        "multi_agg_winner": multi_agg_winner,
        "all_head_store_totals": [float(x) for x in all_store_totals],
        "all_agg_winner": all_agg_winner,
        "head_argmax_winners": head_argmax_winners,
        "head_agg_winners": head_agg_winners,
    }


def main():
    model, tokenizer = load_model()

    structural = {'<bos>', '<start_of_turn>', '<end_of_turn>', 'user', 'model', '\n', '\n\n', ',', '.', '-'}

    # ================================================================
    # Build three stores
    # ================================================================
    # Store A: Zarkov / Voltara
    doc_a = "Zarkov Industries, a cutting-edge robotics company, was founded in 2019 in the city of Voltara by engineer Dimitri Zarkov."
    full_a = f"<start_of_turn>user\n{doc_a}\n\nThe city where Zarkov Industries was founded is called<end_of_turn>\n<start_of_turn>model\n"
    query_a = "<start_of_turn>user\nThe city where Zarkov Industries was founded is called<end_of_turn>\n<start_of_turn>model\n"

    full_ids_a = mx.array([tokenizer.encode(full_a, add_special_tokens=True)])
    query_ids_a = mx.array([tokenizer.encode(query_a, add_special_tokens=True)])
    full_tokens_a = tokenizer.convert_ids_to_tokens(full_ids_a[0].tolist())
    content_pos_a = [i for i, t in enumerate(full_tokens_a) if t not in structural]

    VOLT_ID = 194328
    store_a = build_store(model, tokenizer, doc_a, VOLT_ID, content_pos_a, full_ids_a)
    n_a = store_a['k'].shape[2]
    print(f"Store A (Zarkov/Voltara): {n_a} entries, coeff={store_a['coefficient']:.1f}")

    # Store B: Strand / Castellan
    doc_b = "Helena Strand, a former diplomat, discovered the ancient city of Castellan while researching trade routes in 2021."
    full_b = f"<start_of_turn>user\n{doc_b}\n\nThe ancient city discovered by Helena Strand is called<end_of_turn>\n<start_of_turn>model\n"
    query_b = "<start_of_turn>user\nThe ancient city discovered by Helena Strand is called<end_of_turn>\n<start_of_turn>model\n"

    full_ids_b = mx.array([tokenizer.encode(full_b, add_special_tokens=True)])
    query_ids_b = mx.array([tokenizer.encode(query_b, add_special_tokens=True)])
    full_tokens_b = tokenizer.convert_ids_to_tokens(full_ids_b[0].tolist())
    content_pos_b = [i for i, t in enumerate(full_tokens_b) if t not in structural]

    pred_b = model(full_ids_b)
    pred_probs_b = mx.softmax(pred_b.logits[0, -1, :])
    mx.eval(pred_probs_b)
    cast_id = int(mx.argmax(pred_probs_b))
    cast_tok = tokenizer.decode([cast_id])
    print(f"Store B answer token: '{cast_tok}' (id={cast_id})")

    store_b = build_store(model, tokenizer, doc_b, cast_id, content_pos_b, full_ids_b)
    n_b = store_b['k'].shape[2]
    print(f"Store B (Strand/Castellan): {n_b} entries, coeff={store_b['coefficient']:.1f}")

    # Store C: Voss / Kelvara
    doc_c = "Professor Elara Voss conducted the first successful quantum teleportation experiment at the Meridian Research Institute in the remote mountain town of Kelvara in 2023."
    full_c = f"<start_of_turn>user\n{doc_c}\n\nThe town where the first quantum teleportation experiment took place is called<end_of_turn>\n<start_of_turn>model\n"
    query_c = "<start_of_turn>user\nThe town where the first quantum teleportation experiment took place is called<end_of_turn>\n<start_of_turn>model\n"

    full_ids_c = mx.array([tokenizer.encode(full_c, add_special_tokens=True)])
    query_ids_c = mx.array([tokenizer.encode(query_c, add_special_tokens=True)])
    full_tokens_c = tokenizer.convert_ids_to_tokens(full_ids_c[0].tolist())
    content_pos_c = [i for i, t in enumerate(full_tokens_c) if t not in structural]

    pred_c = model(full_ids_c)
    pred_probs_c = mx.softmax(pred_c.logits[0, -1, :])
    mx.eval(pred_probs_c)
    kelv_id = int(mx.argmax(pred_probs_c))
    kelv_tok = tokenizer.decode([kelv_id])
    print(f"Store C answer token: '{kelv_tok}' (id={kelv_id})")

    store_c = build_store(model, tokenizer, doc_c, kelv_id, content_pos_c, full_ids_c)
    n_c = store_c['k'].shape[2]
    print(f"Store C (Voss/Kelvara): {n_c} entries, coeff={store_c['coefficient']:.1f}")

    # Combine all three
    combined_k, combined_v, entry_map = combine_stores(store_a, store_b, store_c)
    stores = [store_a, store_b, store_c]
    store_names = ["A (Zarkov/Voltara)", "B (Strand/Castellan)", "C (Voss/Kelvara)"]
    print(f"\nCombined: {n_a} + {n_b} + {n_c} = {n_a + n_b + n_c} entries")

    all_results = {}

    # ================================================================
    # EXPERIMENTS 1-3: Aggregate routing at N=3
    # ================================================================
    queries = [
        ("Zarkov city", query_ids_a, 0),
        ("Director surname", query_ids_b, 1),
        ("Teleportation town", query_ids_c, 2),
    ]

    print("\n" + "=" * 80)
    print("AGGREGATE ROUTING: 3 Stores, 3 Queries, 116 Entries")
    print("=" * 80)

    for name, qids, correct_store in queries:
        print(f"\n{'─' * 70}")
        print(f"Query: {name} (correct: {store_names[correct_store]})")
        print(f"{'─' * 70}")

        r = query_full_attention(model, tokenizer, qids, combined_k, combined_v,
                                  entry_map, stores)

        agg = compute_aggregates(r['all_heads_attn'], entry_map, 3)

        # Print results
        print(f"\n  Entry-level argmax (H4):")
        print(f"    Winner: entry {agg['h4_argmax_entry']} → store {agg['h4_argmax_store']} "
              f"({store_names[agg['h4_argmax_store']]})")
        print(f"    Weight: {agg['h4_argmax_weight']:.4f}")
        correct_argmax = agg['h4_argmax_store'] == correct_store
        print(f"    Correct: {'✓' if correct_argmax else '✗'}")

        print(f"\n  H4 aggregate (sum per passage):")
        for si in range(3):
            flag = " ← WINNER" if si == agg['h4_agg_winner'] else ""
            correct_flag = " (correct)" if si == correct_store else ""
            print(f"    Store {store_names[si]}: {agg['h4_store_totals'][si]:.4f}{flag}{correct_flag}")
        correct_h4_agg = agg['h4_agg_winner'] == correct_store
        print(f"    Correct: {'✓' if correct_h4_agg else '✗'}")

        print(f"\n  Multi-head aggregate (H2+H3+H4+H5):")
        for si in range(3):
            flag = " ← WINNER" if si == agg['multi_agg_winner'] else ""
            correct_flag = " (correct)" if si == correct_store else ""
            print(f"    Store {store_names[si]}: {agg['multi_head_store_totals'][si]:.4f}{flag}{correct_flag}")
        correct_multi = agg['multi_agg_winner'] == correct_store
        print(f"    Correct: {'✓' if correct_multi else '✗'}")

        print(f"\n  All-head aggregate (H0-H7):")
        for si in range(3):
            flag = " ← WINNER" if si == agg['all_agg_winner'] else ""
            correct_flag = " (correct)" if si == correct_store else ""
            print(f"    Store {store_names[si]}: {agg['all_head_store_totals'][si]:.4f}{flag}{correct_flag}")
        correct_all = agg['all_agg_winner'] == correct_store
        print(f"    Correct: {'✓' if correct_all else '✗'}")

        print(f"\n  Per-head argmax → aggregate comparison:")
        for head in range(8):
            argmax_s = agg['head_argmax_winners'][head]
            agg_s = agg['head_agg_winners'][head]
            match = "=" if argmax_s == agg_s else "≠"
            print(f"    H{head}: argmax→{argmax_s} {match} agg→{agg_s}")

        print(f"\n  Attention budget per head (stored vs query):")
        for head in range(8):
            stored_pct = r['total_stored_per_head'][head] * 100
            query_pct = r['total_query_per_head'][head] * 100
            print(f"    H{head}: stored={stored_pct:.1f}%  query={query_pct:.1f}%")

        # Store result
        all_results[name] = {
            "correct_store": correct_store,
            "h4_argmax_store": agg['h4_argmax_store'],
            "h4_argmax_correct": correct_argmax,
            "h4_agg_winner": agg['h4_agg_winner'],
            "h4_agg_correct": correct_h4_agg,
            "h4_store_totals": agg['h4_store_totals'],
            "multi_agg_winner": agg['multi_agg_winner'],
            "multi_agg_correct": correct_multi,
            "multi_head_store_totals": agg['multi_head_store_totals'],
            "all_agg_winner": agg['all_agg_winner'],
            "all_agg_correct": correct_all,
            "all_head_store_totals": agg['all_head_store_totals'],
            "head_argmax_winners": agg['head_argmax_winners'],
            "head_agg_winners": agg['head_agg_winners'],
            "attention_budget": {
                "stored_per_head": [float(x) for x in r['total_stored_per_head']],
                "query_per_head": [float(x) for x in r['total_query_per_head']],
            },
        }

    # ================================================================
    # Summary
    # ================================================================
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    methods = [
        ("Entry argmax (H4)", "h4_argmax_correct"),
        ("H4 aggregate", "h4_agg_correct"),
        ("Multi-head agg (H2-H5)", "multi_agg_correct"),
        ("All-head agg (H0-H7)", "all_agg_correct"),
    ]

    print(f"\n{'Method':<30} {'Zarkov':>10} {'Director':>10} {'Teleport':>10} {'Score':>8}")
    print("─" * 70)
    for method_name, key in methods:
        scores = []
        for name in ["Zarkov city", "Director surname", "Teleportation town"]:
            c = all_results[name][key]
            scores.append("✓" if c else "✗")
        total = sum(1 for name in all_results if all_results[name][key])
        print(f"{method_name:<30} {scores[0]:>10} {scores[1]:>10} {scores[2]:>10} {total}/3")

    # Margins
    print(f"\nH4 aggregate margins:")
    for name in ["Zarkov city", "Director surname", "Teleportation town"]:
        r = all_results[name]
        totals = r['h4_store_totals']
        correct = r['correct_store']
        sorted_totals = sorted(enumerate(totals), key=lambda x: x[1], reverse=True)
        winner = sorted_totals[0]
        runner_up = sorted_totals[1]
        margin = winner[1] - runner_up[1]
        ratio = winner[1] / runner_up[1] if runner_up[1] > 0 else float('inf')
        correct_flag = "✓" if winner[0] == correct else "✗"
        print(f"  {name}: {winner[1]:.4f} vs {runner_up[1]:.4f} "
              f"(margin={margin:.4f}, {ratio:.2f}×) {correct_flag}")

    # Save to JSON
    output = {
        "experiment": "attention-aggregate-routing",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "stores": {"A": n_a, "B": n_b, "C": n_c, "total": n_a + n_b + n_c},
        "results": all_results,
    }
    with open("experiment_aggregate_results.json", "w") as f:
        json.dump(output, f, indent=2)
    print("\nResults saved to experiment_aggregate_results.json")


if __name__ == "__main__":
    main()
