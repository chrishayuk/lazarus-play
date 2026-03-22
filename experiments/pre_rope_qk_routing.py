"""Pre-RoPE Q·K Routing Experiment (MLX version)

Extract pre-RoPE K-vectors (after q/k_norm, before RoPE) and Q-vectors
(completion template). Compute Q·K in 256D without positional bias.

Pipeline: residual → input_layernorm → k_proj → reshape → k_norm → [RoPE]
                                                              ↑
                                                    PRE-ROPE (pure content)
"""

import json
import time
import mlx.core as mx
import mlx.nn as nn
import numpy as np
from pathlib import Path
from mlx_lm import load

MODEL_ID = "google/gemma-3-4b-it"
LAYER = 29
D_HEAD = 256
N_HEADS = 8
N_KV_HEADS = 4

# GQA: 2 query heads per KV head
QH_TO_KVH = {0: 0, 1: 0, 2: 1, 3: 1, 4: 2, 5: 2, 6: 3, 7: 3}

PASSAGES = {
    "zarkov": "Zarkov Industries, a cutting-edge robotics company, was founded in 2019 in the city of Voltara by engineer Dimitri Zarkov. The company quickly became known for its innovative approach to humanoid robotics, developing the first commercially viable household assistant robot. Voltara's mild climate and proximity to major shipping routes made it an ideal location for Zarkov's manufacturing facilities.",
    "strand": "Helena Strand, a former diplomat, discovered the ancient city of Castellan while researching trade routes in 2021. The ruins of Castellan, hidden deep in the Carpathian mountains, revealed a previously unknown civilization that had thrived for centuries. Strand's discovery earned her international recognition and the prestigious Archaeological Achievement Award.",
    "kelvara": "Professor Elara Voss conducted the first successful quantum teleportation experiment at the Meridian Research Institute in the remote mountain town of Kelvara in 2023. The experiment demonstrated that macroscopic objects could be transported instantaneously across distances of up to 100 meters, fundamentally changing our understanding of quantum mechanics.",
}

QUERIES = {
    "zarkov": "Transcript query: The city where Zarkov Industries was founded is called\nAnswer:",
    "strand": "Transcript query: The ancient city discovered by Helena Strand is called\nAnswer:",
    "kelvara": "Transcript query: The town where the first quantum teleportation experiment took place is called\nAnswer:",
}

APOLLO_QUERIES = {
    "porridge": ("Transcript query: Who won the porridge eating contest?\nAnswer:", "W170"),
    "baseball": ("Transcript query: What were the baseball scores?\nAnswer:", "W169"),
    "landing": ("Transcript query: What did they say when they landed on the moon?\nAnswer:", "W370"),
    "weather": ("Transcript query: What was the weather in Minneapolis?\nAnswer:", "W169"),
    "news": ("Transcript query: What happened with Thor Heyerdahl's boat?\nAnswer:", "W169"),
}


def extract_pre_rope_qk(model, tokenizer, text, layer_idx=LAYER, positions=None):
    """Extract pre-RoPE Q and K vectors at specified layer.

    Runs forward through layers 0..layer_idx-1, then manually computes
    Q/K projections with norms but WITHOUT RoPE.

    Args:
        model: MLX model
        tokenizer: tokenizer
        text: input text
        layer_idx: target layer
        positions: list of token positions to extract (None = all)

    Returns:
        q_heads: dict {head_idx: mx.array (n_pos, 256)} pre-RoPE Q
        k_heads: dict {kv_head_idx: mx.array (n_pos, 256)} pre-RoPE K
        tokens: list of token strings
        input_ids: token IDs
    """
    # Tokenize
    input_ids = mx.array(tokenizer.encode(text))[None]  # (1, seq_len)
    seq_len = input_ids.shape[1]

    lm = model.language_model if hasattr(model, 'language_model') else model
    inner = lm.model

    # Embedding
    h = inner.embed_tokens(input_ids)
    # Gemma scales embeddings
    h = h * (inner.embed_tokens.weight.shape[-1] ** 0.5)

    # Create causal mask
    mask = nn.MultiHeadAttention.create_additive_causal_mask(seq_len)
    mask = mask.astype(h.dtype)

    # Forward through layers 0 to layer_idx-1
    for i in range(layer_idx):
        layer = inner.layers[i]
        h = layer(h, mask=mask)

    # Now h is the residual stream at input to layer_idx
    # Apply input_layernorm
    target_layer = inner.layers[layer_idx]
    normed = target_layer.input_layernorm(h)

    # Q/K projections (linear, no RoPE yet)
    attn = target_layer.self_attn
    q_raw = attn.q_proj(normed)  # (1, seq_len, n_heads * d_head)
    k_raw = attn.k_proj(normed)  # (1, seq_len, n_kv_heads * d_head)

    # Reshape to heads
    q_raw = q_raw.reshape(1, seq_len, N_HEADS, D_HEAD)      # (1, L, 8, 256)
    k_raw = k_raw.reshape(1, seq_len, N_KV_HEADS, D_HEAD)   # (1, L, 4, 256)

    # Apply q_norm / k_norm (per-head RMSNorm)
    q_normed = attn.q_norm(q_raw)    # (1, L, 8, 256)
    k_normed = attn.k_norm(k_raw)    # (1, L, 4, 256)

    # Select positions
    if positions is not None:
        q_normed = q_normed[:, positions, :, :]
        k_normed = k_normed[:, positions, :, :]

    mx.eval(q_normed, k_normed)

    # Split into per-head dicts
    q_heads = {}
    for h_idx in range(N_HEADS):
        q_heads[h_idx] = q_normed[0, :, h_idx, :]  # (n_pos, 256)

    k_heads = {}
    for kh_idx in range(N_KV_HEADS):
        k_heads[kh_idx] = k_normed[0, :, kh_idx, :]  # (n_pos, 256)

    # Get token strings
    ids = input_ids[0].tolist()
    token_strs = [tokenizer.decode([tid]) for tid in ids]

    return q_heads, k_heads, token_strs, ids


def k_norm_top_positions(k_heads, kv_head, top_k=8):
    """Find positions with highest K-vector norms."""
    k = k_heads[kv_head]  # (seq_len, 256)
    norms = mx.linalg.norm(k, axis=-1)  # (seq_len,)
    mx.eval(norms)
    norms_np = np.array(norms.astype(mx.float32))
    top_idx = np.argsort(-norms_np)[:top_k]
    return top_idx.tolist(), norms_np


def cosine_matrix(a, b):
    """Pairwise cosine similarity between rows of a and b."""
    a = a.astype(mx.float32)
    b = b.astype(mx.float32)
    a_norm = a / mx.linalg.norm(a, axis=-1, keepdims=True)
    b_norm = b / mx.linalg.norm(b, axis=-1, keepdims=True)
    cos = a_norm @ b_norm.T
    mx.eval(cos)
    return np.array(cos)


def run_experiment_1(model, tokenizer):
    """Experiment 1: Pre-RoPE K quality check."""
    print("\n" + "=" * 60)
    print("EXPERIMENT 1 — Pre-RoPE K Quality Check")
    print("=" * 60)

    passage_k = {}
    passage_info = {}

    for name, text in PASSAGES.items():
        _, k_heads, token_strs, _ = extract_pre_rope_qk(
            model, tokenizer, text
        )
        top_pos, norms = k_norm_top_positions(k_heads, kv_head=2, top_k=8)
        passage_k[name] = k_heads[2][mx.array(top_pos)]  # (8, 256)
        mx.eval(passage_k[name])

        print(f"\n{name}: K-norm top-8 positions:")
        for pos in top_pos:
            print(f"  pos={pos} token='{token_strs[pos]}' norm={norms[pos]:.2f}")
        passage_info[name] = {"positions": top_pos, "tokens": [token_strs[p] for p in top_pos]}

    # Intra vs inter passage cosine
    names = list(passage_k.keys())
    intra_cosines = []
    for name in names:
        k = passage_k[name]
        cos = cosine_matrix(k, k)
        mask = np.triu(np.ones_like(cos, dtype=bool), k=1)
        mean_cos = cos[mask].mean()
        intra_cosines.append(float(mean_cos))
        print(f"\n{name} intra-passage mean cosine: {mean_cos:.4f}")

    inter_cosines = []
    for i, n1 in enumerate(names):
        for j, n2 in enumerate(names):
            if j <= i:
                continue
            cos = cosine_matrix(passage_k[n1], passage_k[n2])
            mean_cos = float(cos.mean())
            inter_cosines.append(mean_cos)
            print(f"{n1} vs {n2} inter-passage mean cosine: {mean_cos:.4f}")

    mean_intra = np.mean(intra_cosines)
    mean_inter = np.mean(inter_cosines)
    ratio = mean_intra / mean_inter if mean_inter != 0 else float("inf")

    print(f"\n--- Summary ---")
    print(f"Mean intra-passage cosine: {mean_intra:.4f}")
    print(f"Mean inter-passage cosine: {mean_inter:.4f}")
    print(f"Discrimination ratio:      {ratio:.2f}x")

    return {
        "intra_cosines": dict(zip(names, intra_cosines)),
        "inter_cosines": inter_cosines,
        "mean_intra": float(mean_intra),
        "mean_inter": float(mean_inter),
        "ratio": float(ratio),
        "passage_info": passage_info,
    }


def run_experiment_2(model, tokenizer):
    """Experiment 2: Pre-RoPE Q·K at N=3."""
    print("\n" + "=" * 60)
    print("EXPERIMENT 2 — Pre-RoPE Q·K at N=3")
    print("=" * 60)

    # Build K store
    store = {}
    store_info = {}
    for name, text in PASSAGES.items():
        _, k_heads, token_strs, _ = extract_pre_rope_qk(model, tokenizer, text)
        top_pos, norms = k_norm_top_positions(k_heads, kv_head=2, top_k=8)
        store[name] = k_heads[2][mx.array(top_pos)]  # (8, 256)
        mx.eval(store[name])
        store_info[name] = [(p, token_strs[p]) for p in top_pos]

    # Query each
    results = {}
    for query_name, query_text in QUERIES.items():
        q_heads, _, _, _ = extract_pre_rope_qk(
            model, tokenizer, query_text, positions=[-1]
        )
        q_h4 = q_heads[4]  # (1, 256)

        print(f"\n--- {query_name} query ---")
        scores = {}
        for passage_name, k_store in store.items():
            qk = mx.matmul(k_store, q_h4.T).squeeze(-1)  # (8,)
            mx.eval(qk)
            qk_np = np.array(qk.astype(mx.float32))
            max_score = float(qk_np.max())
            mean_score = float(qk_np.mean())
            best_idx = int(qk_np.argmax())
            best_pos, best_tok = store_info[passage_name][best_idx]
            scores[passage_name] = {
                "max": max_score, "mean": mean_score,
                "best_token": best_tok, "best_pos": best_pos,
                "all": qk_np.tolist(),
            }
            print(f"  {passage_name}: max={max_score:.2f} mean={mean_score:.2f} best='{best_tok}'")

        max_scores = {n: s["max"] for n, s in scores.items()}
        winner = max(max_scores, key=max_scores.get)
        correct = winner == query_name
        sorted_scores = sorted(max_scores.values(), reverse=True)
        margin = sorted_scores[0] / sorted_scores[1] if sorted_scores[1] != 0 else float("inf")

        print(f"  → {winner} {'✓' if correct else '✗'} margin={margin:.2f}x")
        results[query_name] = {
            "scores": scores, "winner": winner,
            "correct": correct, "margin": float(margin),
        }

    # Summary
    n_correct = sum(1 for r in results.values() if r["correct"])
    print(f"\nResult: {n_correct}/3 correct")
    return results


def run_experiment_3(model, tokenizer):
    """Experiment 3: Pre-RoPE Q·K at N=50 (Apollo)."""
    print("\n" + "=" * 60)
    print("EXPERIMENT 3 — Pre-RoPE Q·K at N=50 (Apollo)")
    print("=" * 60)

    windows_path = Path("archive/routing/vector_routing_windows.json")
    with open(windows_path) as f:
        window_data = json.load(f)
    windows = window_data["windows"]

    # Build K store for all 50 windows
    print("Building K-vector store (50 windows × 8 positions)...")
    t0 = time.time()
    store = {}
    store_info = {}

    for i, (wid, text) in enumerate(windows.items()):
        _, k_heads, token_strs, _ = extract_pre_rope_qk(model, tokenizer, text)
        top_pos, norms = k_norm_top_positions(k_heads, kv_head=2, top_k=8)
        store[wid] = k_heads[2][mx.array(top_pos)]  # (8, 256)
        mx.eval(store[wid])
        store_info[wid] = [(p, token_strs[p]) for p in top_pos]
        if (i + 1) % 10 == 0:
            elapsed = time.time() - t0
            print(f"  {i+1}/50 windows ({elapsed:.0f}s)")

    print(f"Store built in {time.time()-t0:.0f}s")

    # Test each query
    results = {}
    for query_name, (query_text, target_window) in APOLLO_QUERIES.items():
        q_heads, _, _, _ = extract_pre_rope_qk(
            model, tokenizer, query_text, positions=[-1]
        )
        q_h4 = q_heads[4]  # (1, 256)

        # Score all windows
        window_scores = {}
        for wid, k_store in store.items():
            qk = mx.matmul(k_store, q_h4.T).squeeze(-1)
            mx.eval(qk)
            qk_np = np.array(qk.astype(mx.float32))
            best_idx = int(qk_np.argmax())
            window_scores[wid] = {
                "max": float(qk_np.max()),
                "mean": float(qk_np.mean()),
                "best_idx": best_idx,
                "best_token": store_info[wid][best_idx][1],
            }

        # Rank by max Q·K
        ranked = sorted(window_scores.items(), key=lambda x: -x[1]["max"])
        target_rank = next(i + 1 for i, (w, _) in enumerate(ranked) if w == target_window)

        print(f"\n--- {query_name} (target: {target_window}) ---")
        print(f"  Target rank: #{target_rank}/50")
        for rank, (wid, sc) in enumerate(ranked[:5]):
            marker = " ← TARGET" if wid == target_window else ""
            print(f"    #{rank+1}: {wid} max={sc['max']:.2f} token='{sc['best_token']}'{marker}")
        if target_rank > 5:
            sc = window_scores[target_window]
            print(f"    #{target_rank}: {target_window} max={sc['max']:.2f} token='{sc['best_token']}' ← TARGET")

        results[query_name] = {
            "target": target_window,
            "target_rank": target_rank,
            "target_score": window_scores[target_window]["max"],
            "top1": ranked[0][0],
            "top1_score": ranked[0][1]["max"],
            "top5": [(wid, s["max"]) for wid, s in ranked[:5]],
            "full_ranking": [(wid, s["max"]) for wid, s in ranked],
        }

    # Summary
    print(f"\n{'Query':<12} {'Target':>8} {'Rank':>6} {'Top-1':>8} {'Score':>8} {'TgtScore':>9}")
    for qn, r in results.items():
        print(f"{qn:<12} {r['target']:>8} #{r['target_rank']:>5} {r['top1']:>8} {r['top1_score']:>8.2f} {r['target_score']:>9.2f}")

    return results


def run_experiment_4(model, tokenizer):
    """Experiment 4: Multi-head pre-RoPE at N=50."""
    print("\n" + "=" * 60)
    print("EXPERIMENT 4 — Multi-Head Pre-RoPE Q·K at N=50")
    print("=" * 60)

    windows_path = Path("archive/routing/vector_routing_windows.json")
    with open(windows_path) as f:
        window_data = json.load(f)
    windows = window_data["windows"]

    # Build multi-KV-head store
    print("Building multi-head K-vector store...")
    t0 = time.time()
    store = {}
    store_info = {}

    for i, (wid, text) in enumerate(windows.items()):
        _, k_heads, token_strs, _ = extract_pre_rope_qk(model, tokenizer, text)
        # Use KV-head 2 norms for position selection
        top_pos, _ = k_norm_top_positions(k_heads, kv_head=2, top_k=8)
        pos_arr = mx.array(top_pos)
        store[wid] = {
            "kvh1": k_heads[1][pos_arr],  # (8, 256) for H2/H3
            "kvh2": k_heads[2][pos_arr],  # (8, 256) for H4/H5
        }
        mx.eval(store[wid]["kvh1"], store[wid]["kvh2"])
        store_info[wid] = [(p, token_strs[p]) for p in top_pos]
        if (i + 1) % 10 == 0:
            print(f"  {i+1}/50 ({time.time()-t0:.0f}s)")

    results = {}
    for query_name, (query_text, target_window) in APOLLO_QUERIES.items():
        q_heads, _, _, _ = extract_pre_rope_qk(
            model, tokenizer, query_text, positions=[-1]
        )

        h4_scores = {}
        multi_scores = {}

        for wid, wdata in store.items():
            # H4 only
            qk_h4 = float(mx.max(mx.matmul(wdata["kvh2"], q_heads[4].T)))

            # Multi: H2+H3 (kvh1) + H4+H5 (kvh2)
            qk_h2 = float(mx.max(mx.matmul(wdata["kvh1"], q_heads[2].T)))
            qk_h3 = float(mx.max(mx.matmul(wdata["kvh1"], q_heads[3].T)))
            qk_h5 = float(mx.max(mx.matmul(wdata["kvh2"], q_heads[5].T)))

            h4_scores[wid] = qk_h4
            multi_scores[wid] = qk_h2 + qk_h3 + qk_h4 + qk_h5

        ranked_h4 = sorted(h4_scores.items(), key=lambda x: -x[1])
        ranked_multi = sorted(multi_scores.items(), key=lambda x: -x[1])

        rank_h4 = next(i+1 for i, (w, _) in enumerate(ranked_h4) if w == target_window)
        rank_multi = next(i+1 for i, (w, _) in enumerate(ranked_multi) if w == target_window)

        print(f"\n{query_name}: H4=#{rank_h4} Multi=#{rank_multi} {'✓ improved' if rank_multi < rank_h4 else '= same' if rank_multi == rank_h4 else '✗ worse'}")

        results[query_name] = {
            "target": target_window,
            "h4_rank": rank_h4,
            "multi_rank": rank_multi,
        }

    return results


def main():
    model, tokenizer = load(MODEL_ID)
    print(f"Model loaded: {MODEL_ID}")

    # Verify weight shapes
    attn = model.language_model.model.layers[LAYER].self_attn
    print(f"L{LAYER} q_proj: {attn.q_proj.weight.shape}")
    print(f"L{LAYER} k_proj: {attn.k_proj.weight.shape}")
    print(f"L{LAYER} q_norm: {attn.q_norm.weight.shape}")
    print(f"L{LAYER} k_norm: {attn.k_norm.weight.shape}")

    exp1 = run_experiment_1(model, tokenizer)
    exp2 = run_experiment_2(model, tokenizer)
    exp3 = run_experiment_3(model, tokenizer)
    exp4 = run_experiment_4(model, tokenizer)

    # Save results
    all_results = {
        "exp1_k_quality": {
            "mean_intra": exp1["mean_intra"],
            "mean_inter": exp1["mean_inter"],
            "ratio": exp1["ratio"],
        },
        "exp2_n3": {
            name: {"correct": r["correct"], "margin": r["margin"], "winner": r["winner"]}
            for name, r in exp2.items()
        },
        "exp3_n50": {
            name: {"target": r["target"], "rank": r["target_rank"],
                   "top1": r["top1"], "score": r["target_score"],
                   "top5": r["top5"]}
            for name, r in exp3.items()
        },
        "exp4_multihead": exp4,
    }

    output_path = Path("experiments/pre_rope_qk_results.json")
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
