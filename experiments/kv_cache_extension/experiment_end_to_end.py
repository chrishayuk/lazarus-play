"""
End-to-End: KV Extension Routing + 2x Injection
=================================================
Test the full pipeline:
1. KV extension at L29 (content-only entries)
2. H4 attention routes to winning entry
3. 2x coefficient 12-byte injection at L30
4. Generate
"""

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from chuk_lazarus.inference import UnifiedPipeline, UnifiedPipelineConfig
from chuk_lazarus.models_v2.families.gemma.model import clip_residual


def load_model():
    config = UnifiedPipelineConfig()
    pipeline = UnifiedPipeline.from_pretrained("google/gemma-3-4b-it", pipeline_config=config)
    return pipeline.model, pipeline.tokenizer


def extract_kv_pre_rope(model, input_ids, layer_idx, positions=None):
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
    """
    Build the complete store for a document:
    - K vectors at L29 for routing (content positions only)
    - Answer token embedding + scaled coefficient for injection
    """
    backbone = model.model

    # Extract K/V at L29
    kv = extract_kv_pre_rope(model, full_ids, layer_idx=layer, positions=content_positions)

    # Get answer embedding
    embed = backbone.embed_tokens.weight[answer_token_id]
    embed_norm_sq = mx.sum(embed * embed)

    # Get full-context residual at inject_layer to compute coefficient
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


def query_with_store(model, tokenizer, query_ids, store, inject_layer=29,
                     delivery_layer=30, use_routing=True, scale=2.0):
    """
    Full pipeline:
    1. L0-28: normal on query
    2. L29: extend KV with stored entries, run attention
    3. Read H4 attention → winning entry
    4. L30: inject 2x coefficient * answer embedding
    5. L31-33: normal
    """
    backbone = model.model
    config = backbone.config

    h = backbone.embed_tokens(query_ids)
    h = h * mx.array(config.hidden_size ** 0.5, dtype=mx.bfloat16).astype(h.dtype)

    query_len = h.shape[1]
    n_stored = store['k'].shape[2]

    global_mask = backbone._create_attention_mask(h, None)
    sliding_mask = (
        backbone._create_attention_mask(h, None, window_size=backbone.sliding_window)
        if backbone.sliding_window_pattern > 1 else None
    )

    winning_idx = None
    winning_weight = None
    h4_attn = None

    for i in range(len(backbone.layers)):
        layer = backbone.layers[i]
        is_global = config.is_global_layer(i)
        mask = global_mask if is_global else sliding_mask

        if i == inject_layer and use_routing:
            # Manual decomposition with KV extension
            attn = layer.self_attn
            residual = h
            normed = layer.input_layernorm(h)

            queries = attn.q_proj(normed).reshape(1, query_len, attn.num_heads, attn.head_dim).transpose(0, 2, 1, 3)
            keys_q = attn.k_proj(normed).reshape(1, query_len, attn.num_kv_heads, attn.head_dim).transpose(0, 2, 1, 3)
            values_q = attn.v_proj(normed).reshape(1, query_len, attn.num_kv_heads, attn.head_dim).transpose(0, 2, 1, 3)

            queries = attn.q_norm(queries)
            keys_q = attn.k_norm(keys_q)

            stored_k = attn.rope(store['k'])
            queries = attn.rope(queries, offset=n_stored)
            keys_q = attn.rope(keys_q, offset=n_stored)

            keys_ext = mx.concatenate([stored_k, keys_q], axis=2)
            values_ext = mx.concatenate([store['v'], values_q], axis=2)

            if attn.n_rep > 1:
                keys_ext = mx.repeat(keys_ext, attn.n_rep, axis=1)
                values_ext = mx.repeat(values_ext, attn.n_rep, axis=1)

            scores = (queries @ keys_ext.transpose(0, 1, 3, 2)) * attn.scale
            stored_cols = mx.zeros((query_len, n_stored), dtype=scores.dtype)
            causal = nn.MultiHeadAttention.create_additive_causal_mask(query_len).astype(scores.dtype)
            scores = scores + mx.concatenate([stored_cols, causal], axis=1)
            weights = mx.softmax(scores, axis=-1)

            h4_attn = weights[0, 4, -1, :n_stored]
            mx.eval(h4_attn)
            winning_idx = int(mx.argmax(h4_attn))
            winning_weight = float(h4_attn[winning_idx])

            context = weights @ values_ext
            context = context.transpose(0, 2, 1, 3).reshape(1, query_len, -1)
            attn_output = attn.o_proj(context)

            h = clip_residual(residual, layer.post_attention_layernorm(attn_output))
            ffn_out = layer.mlp(layer.pre_feedforward_layernorm(h))
            h = clip_residual(h, layer.post_feedforward_layernorm(ffn_out))
        elif i == delivery_layer:
            # Inject BEFORE the layer processes
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

    return {
        "probs": probs,
        "winning_idx": winning_idx,
        "winning_weight": winning_weight,
        "h4_attn": h4_attn,
    }


def main():
    model, tokenizer = load_model()

    # ================================================================
    # PROBE 1: Zarkov / Voltara
    # ================================================================
    print("=" * 60)
    print("PROBE 1: Zarkov / Voltara")
    print("=" * 60)

    doc1 = "Zarkov Industries, a cutting-edge robotics company, was founded in 2019 in the city of Voltara by engineer Dimitri Zarkov."
    full1 = f"<start_of_turn>user\n{doc1}\n\nThe city where Zarkov Industries was founded is called<end_of_turn>\n<start_of_turn>model\n"
    query1 = "<start_of_turn>user\nThe city where Zarkov Industries was founded is called<end_of_turn>\n<start_of_turn>model\n"

    full_ids1 = mx.array([tokenizer.encode(full1, add_special_tokens=True)])
    query_ids1 = mx.array([tokenizer.encode(query1, add_special_tokens=True)])
    full_tokens1 = tokenizer.convert_ids_to_tokens(full_ids1[0].tolist())

    # Content positions (exclude structural)
    structural = {'<bos>', '<start_of_turn>', '<end_of_turn>', 'user', 'model', '\n', '\n\n', ',', '.', '-'}
    content_pos1 = [i for i, t in enumerate(full_tokens1) if t not in structural]

    # Answer token: "Volt" (194328)
    store1 = build_store(model, tokenizer, doc1, 194328, content_pos1, full_ids1,
                         layer=29, inject_layer=30, scale=2.0)

    print(f"  Store: {len(content_pos1)} entries, coeff={store1['coefficient']:.1f}, 2x={store1['scaled_coefficient']:.1f}")

    # Test scales
    for scale in [1.0, 1.5, 2.0, 3.0]:
        store1_s = {**store1, "scaled_coefficient": scale * store1['coefficient']}
        r = query_with_store(model, tokenizer, query_ids1, store1_s)
        p_volt = float(r["probs"][194328])
        top_id = int(mx.argmax(r["probs"]))
        top_tok = tokenizer.decode([top_id])
        win_tok = full_tokens1[content_pos1[r["winning_idx"]]] if r["winning_idx"] is not None else "?"
        print(f"  scale={scale:.1f}x: P(Volt)={p_volt:.4f}, H4→'{win_tok}' ({r['winning_weight']:.3f}), top='{top_tok}'")

    # ================================================================
    # PROBE 2: Helena Strand / Castellan
    # ================================================================
    print("\n" + "=" * 60)
    print("PROBE 2: Helena Strand / Castellan")
    print("=" * 60)

    doc2 = "Helena Strand, a former diplomat, discovered the ancient city of Castellan while researching trade routes in 2021."
    full2 = f"<start_of_turn>user\n{doc2}\n\nThe ancient city discovered by Helena Strand is called<end_of_turn>\n<start_of_turn>model\n"
    query2 = "<start_of_turn>user\nThe ancient city discovered by Helena Strand is called<end_of_turn>\n<start_of_turn>model\n"

    full_ids2 = mx.array([tokenizer.encode(full2, add_special_tokens=True)])
    query_ids2 = mx.array([tokenizer.encode(query2, add_special_tokens=True)])
    full_tokens2 = tokenizer.convert_ids_to_tokens(full_ids2[0].tolist())

    content_pos2 = [i for i, t in enumerate(full_tokens2) if t not in structural]

    # Find the answer token
    pred = model(full_ids2)
    pred_probs = mx.softmax(pred.logits[0, -1, :])
    mx.eval(pred_probs)
    answer_id2 = int(mx.argmax(pred_probs))
    answer_tok2 = tokenizer.decode([answer_id2])
    p_answer2 = float(pred_probs[answer_id2])
    print(f"  Full context predicts: '{answer_tok2}' (id={answer_id2}, P={p_answer2:.4f})")

    # Find the answer token for "Castellan"
    # Tokenize to find the right token
    cast_tokens = tokenizer.encode("Castellan")
    print(f"  'Castellan' tokens: {[(tokenizer.decode([t]), t) for t in cast_tokens]}")
    # Use the first significant token
    cast_id = cast_tokens[0] if len(cast_tokens) == 1 else cast_tokens[0]
    for t in cast_tokens:
        tok_text = tokenizer.decode([t])
        if 'Cast' in tok_text or 'cast' in tok_text or 'Cas' in tok_text:
            cast_id = t
            break

    store2 = build_store(model, tokenizer, doc2, answer_id2, content_pos2, full_ids2,
                         layer=29, inject_layer=30, scale=2.0)
    print(f"  Store: {len(content_pos2)} entries, coeff={store2['coefficient']:.1f}")

    for scale in [1.0, 2.0, 3.0]:
        store2_s = {**store2, "scaled_coefficient": scale * store2['coefficient']}
        r = query_with_store(model, tokenizer, query_ids2, store2_s)
        p_ans = float(r["probs"][answer_id2])
        top_id = int(mx.argmax(r["probs"]))
        top_tok = tokenizer.decode([top_id])
        win_tok = full_tokens2[content_pos2[r["winning_idx"]]] if r["winning_idx"] is not None else "?"
        print(f"  scale={scale:.1f}x: P({answer_tok2})={p_ans:.4f}, H4→'{win_tok}' ({r['winning_weight']:.3f}), top='{top_tok}'")

    # ================================================================
    # PROBE 3: Multi-fact document
    # ================================================================
    print("\n" + "=" * 60)
    print("PROBE 3: Multi-fact document — 2 facts, 2 queries")
    print("=" * 60)

    doc3 = "Zarkov Industries was founded in Voltara. The company's chief scientist is Dr. Meera Patel."
    full3a = f"<start_of_turn>user\n{doc3}\n\nThe city where Zarkov Industries was founded is called<end_of_turn>\n<start_of_turn>model\n"
    full3b = f"<start_of_turn>user\n{doc3}\n\nThe chief scientist of Zarkov Industries is<end_of_turn>\n<start_of_turn>model\n"
    query3a = "<start_of_turn>user\nThe city where Zarkov Industries was founded is called<end_of_turn>\n<start_of_turn>model\n"
    query3b = "<start_of_turn>user\nThe chief scientist of Zarkov Industries is<end_of_turn>\n<start_of_turn>model\n"

    # Get answer tokens
    full_ids3a = mx.array([tokenizer.encode(full3a, add_special_tokens=True)])
    full_ids3b = mx.array([tokenizer.encode(full3b, add_special_tokens=True)])

    pred3a = model(full_ids3a)
    p3a = mx.softmax(pred3a.logits[0, -1, :])
    mx.eval(p3a)
    ans3a_id = int(mx.argmax(p3a))
    ans3a = tokenizer.decode([ans3a_id])
    print(f"  Query A answer: '{ans3a}' (id={ans3a_id}, P={float(p3a[ans3a_id]):.4f})")

    pred3b = model(full_ids3b)
    p3b = mx.softmax(pred3b.logits[0, -1, :])
    mx.eval(p3b)
    ans3b_id = int(mx.argmax(p3b))
    ans3b = tokenizer.decode([ans3b_id])
    print(f"  Query B answer: '{ans3b}' (id={ans3b_id}, P={float(p3b[ans3b_id]):.4f})")

    # Build ONE store per fact (each with its own answer token)
    full_tokens3a = tokenizer.convert_ids_to_tokens(full_ids3a[0].tolist())
    content_pos3 = [i for i, t in enumerate(full_tokens3a) if t not in structural]

    # Store for fact A (Voltara)
    storeA = build_store(model, tokenizer, doc3, ans3a_id, content_pos3, full_ids3a,
                         layer=29, inject_layer=30, scale=2.0)

    # Store for fact B (Meera Patel)
    storeB = build_store(model, tokenizer, doc3, ans3b_id, content_pos3, full_ids3b,
                         layer=29, inject_layer=30, scale=2.0)

    query_ids3a = mx.array([tokenizer.encode(query3a, add_special_tokens=True)])
    query_ids3b = mx.array([tokenizer.encode(query3b, add_special_tokens=True)])

    # Test: Query A with store A (should route to Voltara)
    print(f"\n  Query A ('city founded') with Store A (Voltara answer):")
    r = query_with_store(model, tokenizer, query_ids3a, storeA)
    p = float(r["probs"][ans3a_id])
    win = full_tokens3a[content_pos3[r["winning_idx"]]]
    top_id = int(mx.argmax(r["probs"]))
    print(f"    P({ans3a})={p:.4f}, H4→'{win}', top='{tokenizer.decode([top_id])}'")

    # Test: Query A with store B (wrong store — should NOT get Meera)
    print(f"  Query A ('city founded') with Store B (Patel answer):")
    r = query_with_store(model, tokenizer, query_ids3a, storeB)
    p = float(r["probs"][ans3b_id])
    top_id = int(mx.argmax(r["probs"]))
    print(f"    P({ans3b})={p:.4f}, top='{tokenizer.decode([top_id])}'")

    # Test: Query B with store B (should route to Patel)
    print(f"  Query B ('chief scientist') with Store B (Patel answer):")
    r = query_with_store(model, tokenizer, query_ids3b, storeB)
    p = float(r["probs"][ans3b_id])
    win = full_tokens3a[content_pos3[r["winning_idx"]]]
    top_id = int(mx.argmax(r["probs"]))
    print(f"    P({ans3b})={p:.4f}, H4→'{win}', top='{tokenizer.decode([top_id])}'")

    # Test: Query B with store A (wrong store)
    print(f"  Query B ('chief scientist') with Store A (Voltara answer):")
    r = query_with_store(model, tokenizer, query_ids3b, storeA)
    p = float(r["probs"][ans3a_id])
    top_id = int(mx.argmax(r["probs"]))
    print(f"    P({ans3a})={p:.4f}, top='{tokenizer.decode([top_id])}'")

    # ================================================================
    # PROBE 4: No routing — just inject at L30 from bare query
    # ================================================================
    print("\n" + "=" * 60)
    print("PROBE 4: Pure injection (no KV extension routing)")
    print("=" * 60)

    for scale in [1.5, 2.0, 3.0]:
        store1_nr = {**store1, "scaled_coefficient": scale * store1['coefficient']}
        r = query_with_store(model, tokenizer, query_ids1, store1_nr, use_routing=False)
        p_volt = float(r["probs"][194328])
        top_id = int(mx.argmax(r["probs"]))
        top_tok = tokenizer.decode([top_id])
        print(f"  Volt scale={scale:.1f}x (no routing): P(Volt)={p_volt:.4f}, top='{top_tok}'")


if __name__ == "__main__":
    main()
