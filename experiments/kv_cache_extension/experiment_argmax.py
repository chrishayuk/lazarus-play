"""
Attention-Argmax Injection: Connect H4 routing to injection delivery
====================================================================
Two stores in the KV cache. H4 argmax selects the winner.
Inject ONLY the winner at L30. Does the model route correctly?
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

    # Forward to inject_layer to get donor residual for coefficient
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
    entry_map = []  # (store_index, position_within_store)
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


def query_with_combined_store(model, tokenizer, query_ids, combined_k, combined_v,
                              entry_map, stores, inject_layer=29, delivery_layer=30,
                              scale=2.0, top_k_report=10):
    """
    Full pipeline with combined store:
    1. L0-28: normal on query
    2. L29: extend KV with ALL stored entries, run attention
    3. Read H4 attention over stored entries → argmax → winning entry
    4. Look up which store the winner belongs to
    5. L30: inject that store's answer at 2× coefficient
    6. L31-33: normal → logits
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

    winning_idx = None
    winning_weight = None
    h4_attn_stored = None
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

            # RoPE: stored at 0..N-1, query at N..N+M-1
            stored_k_roped = attn.rope(combined_k)
            queries = attn.rope(queries, offset=n_stored)
            keys_q = attn.rope(keys_q, offset=n_stored)

            keys_ext = mx.concatenate([stored_k_roped, keys_q], axis=2)
            values_ext = mx.concatenate([combined_v, values_q], axis=2)

            if attn.n_rep > 1:
                keys_ext = mx.repeat(keys_ext, attn.n_rep, axis=1)
                values_ext = mx.repeat(values_ext, attn.n_rep, axis=1)

            scores = (queries @ keys_ext.transpose(0, 1, 3, 2)) * attn.scale
            # Mask: stored entries visible to all query tokens, causal among query tokens
            stored_cols = mx.zeros((query_len, n_stored), dtype=scores.dtype)
            causal = nn.MultiHeadAttention.create_additive_causal_mask(query_len).astype(scores.dtype)
            scores = scores + mx.concatenate([stored_cols, causal], axis=1)
            weights = mx.softmax(scores, axis=-1)

            # H4 attention over stored entries (last query token)
            h4_attn_stored = weights[0, 4, -1, :n_stored]
            # All heads over stored entries (last query token)
            all_heads_attn = weights[0, :, -1, :n_stored]  # (8, n_stored)
            mx.eval(h4_attn_stored, all_heads_attn)

            winning_idx = int(mx.argmax(h4_attn_stored))
            winning_weight = float(h4_attn_stored[winning_idx])

            # Continue attention computation normally
            context = weights @ values_ext
            context = context.transpose(0, 2, 1, 3).reshape(1, query_len, -1)
            attn_output = attn.o_proj(context)

            h = clip_residual(residual, layer.post_attention_layernorm(attn_output))
            ffn_out = layer.mlp(layer.pre_feedforward_layernorm(h))
            h = clip_residual(h, layer.post_feedforward_layernorm(ffn_out))

        elif i == delivery_layer:
            # Inject the winning store's answer
            win_store_idx = entry_map[winning_idx][0]
            store = stores[win_store_idx]
            inject_vec = store['scaled_coefficient'] * store['answer_embed']
            h_last = h[:, -1:, :] + inject_vec.reshape(1, 1, -1).astype(h.dtype)
            h = mx.concatenate([h[:, :-1, :], h_last], axis=1)

            output = layer(h, mask=mask, cache=None)
            h = output.hidden_states
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

    # Build top-k report for H4 attention
    h4_np = np.array(h4_attn_stored.astype(mx.float32))
    top_indices = np.argsort(h4_np)[::-1][:top_k_report]
    top_entries = []
    for rank, idx in enumerate(top_indices):
        si, pi = entry_map[idx]
        top_entries.append({
            "rank": rank + 1,
            "global_idx": int(idx),
            "store": si,
            "weight": float(h4_np[idx]),
        })

    # Per-head argmax over stored entries
    all_heads_np = np.array(all_heads_attn.astype(mx.float32))
    head_winners = []
    for head in range(8):
        head_attn = all_heads_np[head]
        head_win = int(np.argmax(head_attn))
        head_si, head_pi = entry_map[head_win]
        head_winners.append({
            "head": head,
            "winning_store": head_si,
            "weight": float(head_attn[head_win]),
        })

    return {
        "probs": probs,
        "winning_idx": winning_idx,
        "winning_store": entry_map[winning_idx][0],
        "winning_weight": winning_weight,
        "top_entries": top_entries,
        "head_winners": head_winners,
        "n_stored": n_stored,
    }


def main():
    model, tokenizer = load_model()

    # ================================================================
    # Build stores
    # ================================================================
    structural = {'<bos>', '<start_of_turn>', '<end_of_turn>', 'user', 'model', '\n', '\n\n', ',', '.', '-'}

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
    print(f"Store A (Zarkov/Voltara): {len(content_pos_a)} entries, coeff={store_a['coefficient']:.1f}")
    print(f"  Tokens: {[full_tokens_a[p] for p in content_pos_a]}")

    # Store B: Strand / Castellan
    doc_b = "Helena Strand, a former diplomat, discovered the ancient city of Castellan while researching trade routes in 2021."
    full_b = f"<start_of_turn>user\n{doc_b}\n\nThe ancient city discovered by Helena Strand is called<end_of_turn>\n<start_of_turn>model\n"
    query_b = "<start_of_turn>user\nThe ancient city discovered by Helena Strand is called<end_of_turn>\n<start_of_turn>model\n"

    full_ids_b = mx.array([tokenizer.encode(full_b, add_special_tokens=True)])
    query_ids_b = mx.array([tokenizer.encode(query_b, add_special_tokens=True)])
    full_tokens_b = tokenizer.convert_ids_to_tokens(full_ids_b[0].tolist())
    content_pos_b = [i for i, t in enumerate(full_tokens_b) if t not in structural]

    # Find Castellan answer token
    pred_b = model(full_ids_b)
    pred_probs_b = mx.softmax(pred_b.logits[0, -1, :])
    mx.eval(pred_probs_b)
    cast_id = int(mx.argmax(pred_probs_b))
    cast_tok = tokenizer.decode([cast_id])
    print(f"Store B answer token: '{cast_tok}' (id={cast_id}, P={float(pred_probs_b[cast_id]):.4f})")

    store_b = build_store(model, tokenizer, doc_b, cast_id, content_pos_b, full_ids_b)
    print(f"Store B (Strand/Castellan): {len(content_pos_b)} entries, coeff={store_b['coefficient']:.1f}")
    print(f"  Tokens: {[full_tokens_b[p] for p in content_pos_b]}")

    # Combine stores
    combined_k, combined_v, entry_map = combine_stores(store_a, store_b)
    n_a = store_a['k'].shape[2]
    n_b = store_b['k'].shape[2]
    print(f"\nCombined: {n_a} (A) + {n_b} (B) = {n_a + n_b} entries")

    stores = [store_a, store_b]
    store_names = ["A (Zarkov/Voltara)", "B (Strand/Castellan)"]

    # ================================================================
    # EXPERIMENT 1: Two Stores, Two Queries
    # ================================================================
    print("\n" + "=" * 70)
    print("EXPERIMENT 1: Two Stores, Two Queries — H4 Argmax Routing")
    print("=" * 70)

    results_exp1 = {}

    for name, qids, correct_store in [
        ("Zarkov city", query_ids_a, 0),
        ("Director surname", query_ids_b, 1),
    ]:
        r = query_with_combined_store(
            model, tokenizer, qids, combined_k, combined_v,
            entry_map, stores, top_k_report=10
        )

        correct = r['winning_store'] == correct_store
        answer_tok_id = stores[r['winning_store']]['answer_token_id']
        answer_tok = tokenizer.decode([answer_tok_id])

        # Get probabilities for both answers
        p_volt = float(r['probs'][VOLT_ID])
        p_cast = float(r['probs'][cast_id])

        top_id = int(mx.argmax(r['probs']))
        top_tok = tokenizer.decode([top_id])
        top_p = float(r['probs'][top_id])

        print(f"\n  Query: '{name}'")
        print(f"  H4 argmax → entry {r['winning_idx']} from store {store_names[r['winning_store']]} (weight={r['winning_weight']:.4f})")
        print(f"  Correct store: {store_names[correct_store]} → {'✓ CORRECT' if correct else '✗ WRONG'}")
        print(f"  Injected answer: '{answer_tok}'")
        print(f"  P(Volt)={p_volt:.4f}, P(Cast first tok)={p_cast:.4f}")
        print(f"  Top prediction: '{top_tok}' P={top_p:.4f}")

        print(f"\n  H4 attention top-10 over stored entries:")
        store_a_in_top5 = 0
        store_b_in_top5 = 0
        for e in r['top_entries']:
            s_name = "A" if e['store'] == 0 else "B"
            flag = " ←" if e['rank'] == 1 else ""
            print(f"    #{e['rank']}: store {s_name}, global_idx={e['global_idx']}, weight={e['weight']:.4f}{flag}")
            if e['rank'] <= 5:
                if e['store'] == 0:
                    store_a_in_top5 += 1
                else:
                    store_b_in_top5 += 1

        print(f"  Top-5 composition: {store_a_in_top5} from A, {store_b_in_top5} from B")

        print(f"\n  Per-head argmax winners:")
        for hw in r['head_winners']:
            s_name = "A" if hw['winning_store'] == 0 else "B"
            print(f"    H{hw['head']}: store {s_name} (weight={hw['weight']:.4f})")

        # Margin: #1 weight minus #2 weight
        margin = r['top_entries'][0]['weight'] - r['top_entries'][1]['weight'] if len(r['top_entries']) > 1 else 0
        concentration = r['top_entries'][0]['weight']

        results_exp1[name] = {
            "winning_idx": r['winning_idx'],
            "winning_store": r['winning_store'],
            "winning_store_name": store_names[r['winning_store']],
            "correct_store": correct_store,
            "correct": correct,
            "winning_weight": r['winning_weight'],
            "margin_over_2nd": margin,
            "top1_concentration": concentration,
            "P_Volt": p_volt,
            "P_Cast": p_cast,
            "top_prediction": top_tok,
            "top_prediction_prob": top_p,
            "top_10_entries": r['top_entries'],
            "head_winners": r['head_winners'],
            "top5_from_A": store_a_in_top5,
            "top5_from_B": store_b_in_top5,
        }

    # Summary
    all_correct = all(v['correct'] for v in results_exp1.values())
    print("\n" + "-" * 70)
    print(f"EXPERIMENT 1 RESULT: {'PASS — H4 argmax routes correctly' if all_correct else 'FAIL — H4 argmax does NOT discriminate'}")
    print("-" * 70)

    # Save results
    with open("experiment_argmax_results.json", "w") as f:
        json.dump({
            "experiment": "attention-argmax-injection",
            "exp1_two_stores_two_queries": results_exp1,
            "exp1_pass": all_correct,
            "store_a_entries": n_a,
            "store_b_entries": n_b,
            "total_entries": n_a + n_b,
        }, f, indent=2, default=str)
    print("\nResults saved to experiment_argmax_results.json")

    if not all_correct:
        print("\n*** STOPPING: H4 argmax failed. Keywords stay. ***")
        return

    # ================================================================
    # EXPERIMENT 2: Three Stores, Three Queries
    # ================================================================
    print("\n" + "=" * 70)
    print("EXPERIMENT 2: Three Stores, Three Queries")
    print("=" * 70)

    # Store C: A third novel passage
    doc_c = "Professor Elara Voss conducted the first successful quantum teleportation experiment at the Meridian Research Institute in the remote mountain town of Kelvara in 2023."
    full_c = f"<start_of_turn>user\n{doc_c}\n\nThe town where the first quantum teleportation experiment took place is called<end_of_turn>\n<start_of_turn>model\n"
    query_c = "<start_of_turn>user\nThe town where the first quantum teleportation experiment took place is called<end_of_turn>\n<start_of_turn>model\n"

    full_ids_c = mx.array([tokenizer.encode(full_c, add_special_tokens=True)])
    query_ids_c = mx.array([tokenizer.encode(query_c, add_special_tokens=True)])
    full_tokens_c = tokenizer.convert_ids_to_tokens(full_ids_c[0].tolist())
    content_pos_c = [i for i, t in enumerate(full_tokens_c) if t not in structural]

    # Find answer token for store C
    pred_c = model(full_ids_c)
    pred_probs_c = mx.softmax(pred_c.logits[0, -1, :])
    mx.eval(pred_probs_c)
    kelv_id = int(mx.argmax(pred_probs_c))
    kelv_tok = tokenizer.decode([kelv_id])
    print(f"Store C answer token: '{kelv_tok}' (id={kelv_id}, P={float(pred_probs_c[kelv_id]):.4f})")

    store_c = build_store(model, tokenizer, doc_c, kelv_id, content_pos_c, full_ids_c)
    print(f"Store C (Voss/Kelvara): {len(content_pos_c)} entries, coeff={store_c['coefficient']:.1f}")

    # Combine all three
    combined_k3, combined_v3, entry_map3 = combine_stores(store_a, store_b, store_c)
    n_c = store_c['k'].shape[2]
    stores3 = [store_a, store_b, store_c]
    store_names3 = ["A (Zarkov/Voltara)", "B (Strand/Castellan)", "C (Voss/Kelvara)"]
    print(f"Combined: {n_a} + {n_b} + {n_c} = {n_a + n_b + n_c} entries")

    results_exp2 = {}
    for name, qids, correct_store in [
        ("Zarkov city", query_ids_a, 0),
        ("Director surname", query_ids_b, 1),
        ("Teleportation town", query_ids_c, 2),
    ]:
        r = query_with_combined_store(
            model, tokenizer, qids, combined_k3, combined_v3,
            entry_map3, stores3, top_k_report=5
        )

        correct = r['winning_store'] == correct_store
        answer_tok = tokenizer.decode([stores3[r['winning_store']]['answer_token_id']])

        print(f"\n  Query: '{name}'")
        print(f"  H4 argmax → store {store_names3[r['winning_store']]} (weight={r['winning_weight']:.4f})")
        print(f"  Correct: {store_names3[correct_store]} → {'✓' if correct else '✗'}")
        top5_summary = [(e['store'], round(e['weight'], 4)) for e in r['top_entries'][:5]]
        print(f"  Top-5: {top5_summary}")

        results_exp2[name] = {
            "winning_store": r['winning_store'],
            "correct_store": correct_store,
            "correct": correct,
            "winning_weight": r['winning_weight'],
        }

    exp2_pass = all(v['correct'] for v in results_exp2.values())
    print(f"\nEXPERIMENT 2 RESULT: {'PASS — 3/3' if exp2_pass else 'FAIL'}")

    # ================================================================
    # EXPERIMENT 3: Attention Distribution Analysis
    # ================================================================
    print("\n" + "=" * 70)
    print("EXPERIMENT 3: H4 Attention Distribution Analysis")
    print("=" * 70)

    # Re-run exp1 queries with full distribution analysis
    results_exp3 = {}
    for name, qids, correct_store in [
        ("Zarkov city", query_ids_a, 0),
        ("Director surname", query_ids_b, 1),
    ]:
        r = query_with_combined_store(
            model, tokenizer, qids, combined_k, combined_v,
            entry_map, stores, top_k_report=len(entry_map)
        )

        h4_weights = [e['weight'] for e in r['top_entries']]
        top1 = r['top_entries'][0]
        top2 = r['top_entries'][1] if len(r['top_entries']) > 1 else None

        margin = top1['weight'] - (top2['weight'] if top2 else 0)
        ratio = top1['weight'] / top2['weight'] if top2 and top2['weight'] > 0 else float('inf')
        total_stored_attn = sum(h4_weights)
        concentration = top1['weight'] / total_stored_attn if total_stored_attn > 0 else 0

        # Count how many of top-5 are from correct store
        top5_correct = sum(1 for e in r['top_entries'][:5] if e['store'] == correct_store)

        print(f"\n  Query: '{name}' (correct store = {correct_store})")
        print(f"  #1: store={'A' if top1['store']==0 else 'B'} weight={top1['weight']:.6f}")
        if top2:
            print(f"  #2: store={'A' if top2['store']==0 else 'B'} weight={top2['weight']:.6f}")
        print(f"  Margin: {margin:.6f} ({ratio:.1f}× #2)")
        print(f"  Top-1 concentration: {concentration:.1%} of total stored attention")
        print(f"  Top-5 from correct store: {top5_correct}/5")

        # Store-level aggregation
        store_totals = [0.0, 0.0]
        for e in r['top_entries']:
            store_totals[e['store']] += e['weight']
        print(f"  Total attn to Store A: {store_totals[0]:.4f}")
        print(f"  Total attn to Store B: {store_totals[1]:.4f}")

        results_exp3[name] = {
            "top1_store": top1['store'],
            "top1_weight": top1['weight'],
            "top2_store": top2['store'] if top2 else None,
            "top2_weight": top2['weight'] if top2 else None,
            "margin": margin,
            "ratio": ratio,
            "concentration": concentration,
            "top5_from_correct": top5_correct,
            "store_a_total": store_totals[0],
            "store_b_total": store_totals[1],
        }

    # Save all results
    final = {
        "experiment": "attention-argmax-injection",
        "exp1": {"results": results_exp1, "pass": all_correct},
        "exp2": {"results": results_exp2, "pass": exp2_pass},
        "exp3": {"results": results_exp3},
        "stores": {
            "A_entries": n_a, "B_entries": n_b, "C_entries": n_c,
            "total_2store": n_a + n_b, "total_3store": n_a + n_b + n_c,
        }
    }
    with open("experiment_argmax_results.json", "w") as f:
        json.dump(final, f, indent=2, default=str)
    print("\n\nAll results saved to experiment_argmax_results.json")


if __name__ == "__main__":
    main()
