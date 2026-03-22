"""
Synthetic KV Injection Experiments at L29.

Tests whether injecting stored K/V entries at L29's copy head
allows the model to retrieve answers via its own attention,
without external routing.

Experiments:
  2: Extract K/V at target position from document forward pass
  4: Single synthetic KV entry injection
  5: Multiple entries (1 correct + 10 noise)
  6: RoPE position mismatch analysis
  7: Scale test with multiple queries
"""

import json
import os
import random
import sys
import time

import mlx.core as mx
import mlx.nn as nn
import numpy as np

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DOCUMENT = (
    "Zarkov Industries was established in the mid-1990s as a "
    "pioneering manufacturer of industrial filtration systems. "
    "Its headquarters, built on a former industrial lot, became "
    "a landmark of Voltara's commercial district.\n\n"
    "Zarkov Industries was founded in the city of"
)
QUERY = "Zarkov Industries was founded in the city of"
TARGET = "Volt"
TARGET_TOKEN_ID = 89711
VOLT_POS = 39  # Position of " Volt" in the document tokenization

OUTDIR = os.path.dirname(os.path.abspath(__file__))

# Gemma 3 4B-it architecture constants
SLIDING_WINDOW = 512
SLIDING_WINDOW_PATTERN = 6  # Every 6th layer (0-indexed: 5,11,17,23,29) is global


def is_global_layer(layer_idx: int) -> bool:
    return (layer_idx + 1) % SLIDING_WINDOW_PATTERN == 0


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model():
    """Load Gemma 3 4B-it via chuk-lazarus."""
    from chuk_lazarus.inference.unified import UnifiedPipeline

    print("Loading google/gemma-3-4b-it ...")
    t0 = time.time()
    pipeline = UnifiedPipeline.from_pretrained("google/gemma-3-4b-it")
    print(f"Loaded in {time.time() - t0:.1f}s")
    return pipeline


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class ModelHelper:
    """Wraps the loaded model for convenient access."""

    def __init__(self, pipeline):
        self.model = pipeline.model
        self.tokenizer = pipeline.tokenizer
        self.config = pipeline.config

        from chuk_lazarus.introspection.hooks import ModelHooks
        helper = ModelHooks(self.model, model_config=self.config)

        self.layers = helper._get_layers()
        self.embed = helper._get_embed_tokens()
        self.final_norm = helper._get_final_norm()
        self.embedding_scale = helper._get_embedding_scale()

        # Determine correct lm_head (Gemma uses tied embeddings)
        if getattr(self.model, "tie_word_embeddings", False):
            self.lm_head = self.embed.as_linear
            print("  Using tied embedding projection for lm_head")
        else:
            self.lm_head = helper._get_lm_head()
            print("  Using explicit lm_head")
        self.num_layers = len(self.layers)

        # L29 attention params
        attn = self.layers[29].self_attn
        self.num_heads = attn.num_heads          # 8
        self.num_kv_heads = attn.num_kv_heads    # 4
        self.head_dim = attn.head_dim            # 256
        self.n_rep = self.num_heads // self.num_kv_heads  # 2
        self.attn_scale = attn.scale             # 256^(-0.5) = 1/16

        print(f"  layers={self.num_layers}, heads={self.num_heads}, "
              f"kv_heads={self.num_kv_heads}, head_dim={self.head_dim}, "
              f"n_rep={self.n_rep}, scale={self.attn_scale:.6f}")

    # ---- tokenization ----

    def tokenize(self, text: str) -> mx.array:
        return mx.array(self.tokenizer.encode(text, add_special_tokens=True))

    # ---- logit extraction ----

    def get_logits(self, h: mx.array) -> mx.array:
        """Apply final_norm + lm_head to last-position hidden state."""
        h_last = h[:, -1:, :]          # (1, 1, 2560)
        h_normed = self.final_norm(h_last)
        logits = self.lm_head(h_normed)  # (1, 1, vocab)
        return logits[0, 0]             # (vocab,)

    def prob_of_token(self, logits: mx.array, token_id: int) -> float:
        probs = mx.softmax(logits)
        mx.eval(probs)
        return float(probs[token_id].item())

    def top_k_predictions(self, logits: mx.array, k: int = 10) -> list[dict]:
        probs = mx.softmax(logits)
        mx.eval(probs)
        probs_np = np.array(probs.astype(mx.float32))
        order = np.argsort(-probs_np)
        results = []
        for i in range(min(k, len(probs_np))):
            tid = int(order[i])
            results.append({
                "token": self.tokenizer.decode([tid]),
                "token_id": tid,
                "probability": round(float(probs_np[tid]), 6),
            })
        return results

    # ---- forward pass helpers ----

    def _make_mask(self, seq_len: int, dtype) -> mx.array:
        """Causal mask.  Seq ≤ 512 so sliding == global."""
        mask = nn.MultiHeadAttention.create_additive_causal_mask(seq_len)
        return mask.astype(dtype)

    def forward_to_layer(self, input_ids: mx.array, target_layer: int) -> mx.array:
        """Run layers [0, target_layer] inclusive.  Returns post-target_layer h."""
        if input_ids.ndim == 1:
            input_ids = input_ids[None, :]

        h = self.embed(input_ids)
        if self.embedding_scale is not None:
            h = h * mx.array(self.embedding_scale, dtype=h.dtype)

        mask = self._make_mask(input_ids.shape[1], h.dtype)

        for i in range(target_layer + 1):
            layer_out = self.layers[i](h, mask=mask, cache=None)
            h = layer_out.hidden_states

        mx.eval(h)
        return h

    def forward_from_layer(self, h: mx.array, start_layer: int) -> mx.array:
        """Run layers [start_layer, num_layers) on h.  Returns pre-final-norm h."""
        mask = self._make_mask(h.shape[1], h.dtype)

        for i in range(start_layer, self.num_layers):
            layer_out = self.layers[i](h, mask=mask, cache=None)
            h = layer_out.hidden_states

        mx.eval(h)
        return h

    # ---- L29 K/V extraction ----

    def extract_kv_at_layer29(self, input_ids: mx.array):
        """Forward to L28, then project Q/K/V at L29.

        Returns:
            h_pre_l29   – hidden entering L29, shape (1, S, 2560)
            q           – post-q_norm, post-RoPE, (1, 8, S, 256)
            k           – post-k_norm, post-RoPE, (1, 4, S, 256)
            v           – post-v_proj only,       (1, 4, S, 256)
            k_pre_rope  – post-k_norm, pre-RoPE,  (1, 4, S, 256)
        """
        h = self.forward_to_layer(input_ids, 28)

        block = self.layers[29]
        attn = block.self_attn
        B, S, _ = h.shape

        normed = block.input_layernorm(h)

        q = attn.q_proj(normed).reshape(B, S, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        k = attn.k_proj(normed).reshape(B, S, self.num_kv_heads, self.head_dim).transpose(0, 2, 1, 3)
        v = attn.v_proj(normed).reshape(B, S, self.num_kv_heads, self.head_dim).transpose(0, 2, 1, 3)

        # Q/K normalization
        q = attn.q_norm(q)
        k_normed = attn.k_norm(k)

        # Snapshot pre-RoPE K
        k_pre_rope = k_normed

        # Apply RoPE
        q = attn.rope(q)
        k_roped = attn.rope(k_normed)

        mx.eval(h, q, k_roped, v, k_pre_rope)
        return h, q, k_roped, v, k_pre_rope

    # ---- custom L29 with injected K/V ----

    def run_layer29_with_injected_kv(
        self,
        h_input: mx.array,
        injected_k: mx.array,
        injected_v: mx.array,
    ) -> tuple[mx.array, mx.array]:
        """Run L29 with extra K/V entries appended to the KV cache.

        Args:
            h_input:    (1, S_q, 2560) – hidden state entering L29
            injected_k: (1, num_kv_heads, N, head_dim) – post-k_norm, post-RoPE
            injected_v: (1, num_kv_heads, N, head_dim) – post-v_proj

        Returns:
            h_out:        (1, S_q, 2560)
            attn_weights: (1, 8, S_q, S_q + N)
        """
        block = self.layers[29]
        attn = block.self_attn
        B, S_q, _ = h_input.shape
        N = injected_k.shape[2]

        # 1. Input norm
        normed = block.input_layernorm(h_input)

        # 2. Project Q, K, V from query tokens
        q = attn.q_proj(normed).reshape(B, S_q, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        k_q = attn.k_proj(normed).reshape(B, S_q, self.num_kv_heads, self.head_dim).transpose(0, 2, 1, 3)
        v_q = attn.v_proj(normed).reshape(B, S_q, self.num_kv_heads, self.head_dim).transpose(0, 2, 1, 3)

        # 3. Q/K norm
        q = attn.q_norm(q)
        k_q = attn.k_norm(k_q)

        # 4. RoPE on query Q and K
        q = attn.rope(q)
        k_q = attn.rope(k_q)

        # 5. Append injected K/V (already post-norm, post-RoPE)
        k_full = mx.concatenate([k_q, injected_k], axis=2)   # (1, 4, S_q+N, 256)
        v_full = mx.concatenate([v_q, injected_v], axis=2)    # (1, 4, S_q+N, 256)

        # 6. Repeat KV heads for GQA
        k_exp = mx.repeat(k_full, self.n_rep, axis=1)  # (1, 8, S_q+N, 256)
        v_exp = mx.repeat(v_full, self.n_rep, axis=1)

        # 7. Build mask: causal for self-tokens + fully visible for injected
        causal = nn.MultiHeadAttention.create_additive_causal_mask(S_q).astype(h_input.dtype)
        inject_cols = mx.zeros((S_q, N), dtype=h_input.dtype)
        mask = mx.concatenate([causal, inject_cols], axis=1)  # (S_q, S_q+N)

        # 8. Manual scaled dot-product attention (to capture weights)
        # scores: (1, 8, S_q, S_q+N)
        scores = (q @ k_exp.transpose(0, 1, 3, 2)) * attn.scale
        scores = scores + mask[None, None, :, :]
        attn_weights = mx.softmax(scores, axis=-1)

        # output: (1, 8, S_q, 256)
        attn_out = attn_weights @ v_exp

        # 9. Reshape + O-projection
        attn_out = attn_out.transpose(0, 2, 1, 3).reshape(B, S_q, -1)
        r = attn.o_proj(attn_out)

        # 10. Residual connections with Gemma's 4-norm pattern
        h = h_input + block.post_attention_layernorm(r)

        # 11. FFN
        r_ffn = block.mlp(block.pre_feedforward_layernorm(h))
        out = h + block.post_feedforward_layernorm(r_ffn)

        mx.eval(out, attn_weights)
        return out, attn_weights


# ---------------------------------------------------------------------------
# Experiment runners
# ---------------------------------------------------------------------------

def experiment_2(M: ModelHelper) -> dict:
    """Extract K/V at target position and noise positions."""
    print("\n" + "=" * 60)
    print("EXPERIMENT 2: Extract K/V at target position")
    print("=" * 60)

    doc_ids = M.tokenize(DOCUMENT)
    print(f"Document tokens: {doc_ids.shape[0]}")

    h_doc, q_doc, k_doc, v_doc, k_doc_pre_rope = M.extract_kv_at_layer29(doc_ids)

    # Target K/V at Volt position
    k_volt = k_doc[:, :, VOLT_POS:VOLT_POS+1, :]  # (1, 4, 1, 256)
    v_volt = v_doc[:, :, VOLT_POS:VOLT_POS+1, :]

    norms = {
        f"kv_head_{kh}": {
            "k_norm": float(mx.linalg.norm(k_volt[0, kh, 0]).item()),
            "v_norm": float(mx.linalg.norm(v_volt[0, kh, 0]).item()),
        }
        for kh in range(M.num_kv_heads)
    }
    for kh, n in norms.items():
        print(f"  {kh}: k_norm={n['k_norm']:.4f}, v_norm={n['v_norm']:.4f}")

    # Noise positions
    random.seed(42)
    all_positions = list(range(1, doc_ids.shape[0]))
    all_positions.remove(VOLT_POS)
    noise_positions = sorted(random.sample(all_positions, 10))
    print(f"Noise positions: {noise_positions}")

    noise_tokens = []
    for pos in noise_positions:
        tok = M.tokenizer.decode([int(doc_ids[pos].item())])
        noise_tokens.append(tok)
        print(f"  pos {pos}: '{tok}'")

    k_noise = mx.concatenate(
        [k_doc[:, :, p:p+1, :] for p in noise_positions], axis=2
    )  # (1, 4, 10, 256)
    v_noise = mx.concatenate(
        [v_doc[:, :, p:p+1, :] for p in noise_positions], axis=2
    )

    k_volt_pre_rope = k_doc_pre_rope[:, :, VOLT_POS:VOLT_POS+1, :]

    # Save (cast to float32 for numpy compatibility with bfloat16)
    def to_np(x):
        return np.array(x.astype(mx.float32))

    np.save(f"{OUTDIR}/k_volt.npy", to_np(k_volt))
    np.save(f"{OUTDIR}/v_volt.npy", to_np(v_volt))
    np.save(f"{OUTDIR}/k_noise.npy", to_np(k_noise))
    np.save(f"{OUTDIR}/v_noise.npy", to_np(v_noise))
    np.save(f"{OUTDIR}/k_volt_pre_rope.npy", to_np(k_volt_pre_rope))
    print("Saved K/V vectors.")

    result = {
        "description": "Extracted K/V at L29 for Volt position and 10 noise positions",
        "volt_position": VOLT_POS,
        "k_volt_shape": list(k_volt.shape),
        "v_volt_shape": list(v_volt.shape),
        "norms": norms,
        "noise_positions": noise_positions,
        "noise_tokens": noise_tokens,
    }
    return result, k_volt, v_volt, k_noise, v_noise, k_volt_pre_rope


def experiment_4(
    M: ModelHelper,
    k_volt: mx.array,
    v_volt: mx.array,
) -> dict:
    """Single synthetic KV entry injection at L29."""
    print("\n" + "=" * 60)
    print("EXPERIMENT 4: Synthetic KV Injection (Single Entry)")
    print("=" * 60)

    query_ids = M.tokenize(QUERY)
    S_q = query_ids.shape[0]
    print(f"Query tokens: {S_q}")

    # Forward query through L0–L28
    h_pre = M.forward_to_layer(query_ids, 28)

    # Inject single K/V entry (all KV heads, from Volt position)
    h_post, attn_w = M.run_layer29_with_injected_kv(h_pre, k_volt, v_volt)

    # Continue L30–L33
    h_final = M.forward_from_layer(h_post, 30)

    logits = M.get_logits(h_final)
    p_volt = M.prob_of_token(logits, TARGET_TOKEN_ID)
    top_preds = M.top_k_predictions(logits)

    print(f"P(Volt) = {p_volt:.4f}  ({p_volt * 100:.1f}%)")
    for p in top_preds[:5]:
        print(f"  {p['token']:>15s}  {p['probability']:.4f}")

    # Attention weights: last query pos → injected entry (last column)
    attn_np = np.array(attn_w.astype(mx.float32))
    last_q = S_q - 1

    per_head = {}
    for h_idx in range(M.num_heads):
        w_inject = float(attn_np[0, h_idx, last_q, -1])
        w_bos = float(attn_np[0, h_idx, last_q, 0])
        per_head[f"H{h_idx}"] = {
            "attn_to_injected": round(w_inject, 6),
            "attn_to_BOS": round(w_bos, 6),
        }
        if h_idx in (4, 5):
            print(f"  H{h_idx}: inject={w_inject:.4f}, BOS={w_bos:.4f}")

    result = {
        "description": "Single K/V entry injected at L29 (all KV heads)",
        "P_Volt": round(p_volt, 6),
        "top_predictions": top_preds[:10],
        "per_head_attention": per_head,
    }
    with open(f"{OUTDIR}/exp4_single_injection.json", "w") as f:
        json.dump(result, f, indent=2)

    return result, h_pre  # h_pre reused by exp 5


def experiment_5(
    M: ModelHelper,
    h_pre: mx.array,
    k_volt: mx.array,
    v_volt: mx.array,
    k_noise: mx.array,
    v_noise: mx.array,
    noise_positions: list[int],
    doc_ids: mx.array,
) -> dict:
    """11 entries: 1 correct (Volt) + 10 noise.  Volt is first."""
    print("\n" + "=" * 60)
    print("EXPERIMENT 5: Synthetic KV Injection (11 Entries)")
    print("=" * 60)

    query_ids = M.tokenize(QUERY)
    S_q = query_ids.shape[0]

    k_all = mx.concatenate([k_volt, k_noise], axis=2)   # (1, 4, 11, 256)
    v_all = mx.concatenate([v_volt, v_noise], axis=2)

    h_post, attn_w = M.run_layer29_with_injected_kv(h_pre, k_all, v_all)
    h_final = M.forward_from_layer(h_post, 30)

    logits = M.get_logits(h_final)
    p_volt = M.prob_of_token(logits, TARGET_TOKEN_ID)
    top_preds = M.top_k_predictions(logits)

    print(f"P(Volt) = {p_volt:.4f}  ({p_volt * 100:.1f}%)")
    for p in top_preds[:5]:
        print(f"  {p['token']:>15s}  {p['probability']:.4f}")

    # Attention analysis at H4
    attn_np = np.array(attn_w.astype(mx.float32))
    last_q = S_q - 1
    inject_start = S_q
    volt_weight = float(attn_np[0, 4, last_q, inject_start])
    noise_weights = [
        float(attn_np[0, 4, last_q, inject_start + 1 + i])
        for i in range(10)
    ]
    bos_weight = float(attn_np[0, 4, last_q, 0])

    print(f"H4 attn → Volt entry:  {volt_weight:.4f}")
    print(f"H4 attn → BOS:         {bos_weight:.4f}")
    print(f"H4 attn → noise (max): {max(noise_weights):.4f}")

    noise_detail = []
    for i, (pos, w) in enumerate(zip(noise_positions, noise_weights)):
        tok = M.tokenizer.decode([int(doc_ids[pos].item())])
        noise_detail.append({
            "position": pos,
            "token": tok,
            "H4_attention": round(w, 6),
        })

    all_heads_volt = {
        f"H{h}": round(float(attn_np[0, h, last_q, inject_start]), 6)
        for h in range(M.num_heads)
    }

    result = {
        "description": "11 K/V entries injected (1 Volt + 10 noise)",
        "P_Volt": round(p_volt, 6),
        "top_predictions": top_preds[:10],
        "H4_attn_to_volt": round(volt_weight, 6),
        "H4_attn_to_BOS": round(bos_weight, 6),
        "noise_entries": noise_detail,
        "all_heads_attn_to_volt": all_heads_volt,
    }
    with open(f"{OUTDIR}/exp5_multi_injection.json", "w") as f:
        json.dump(result, f, indent=2)

    return result


def experiment_6(
    M: ModelHelper,
    h_pre: mx.array,
    k_volt: mx.array,
    v_volt: mx.array,
    k_volt_pre_rope: mx.array,
) -> dict:
    """RoPE position mismatch analysis."""
    print("\n" + "=" * 60)
    print("EXPERIMENT 6: RoPE Position Mismatch")
    print("=" * 60)

    query_ids = M.tokenize(QUERY)
    S_q = query_ids.shape[0]

    attn = M.layers[29].self_attn
    block = M.layers[29]
    B = 1

    # --- Compute Q·K scores under three conditions ---

    # Get query Q (post-norm, post-RoPE) at last position
    normed = block.input_layernorm(h_pre)
    q_raw = attn.q_proj(normed).reshape(B, S_q, M.num_heads, M.head_dim).transpose(0, 2, 1, 3)
    q_normed = attn.q_norm(q_raw)
    q_pre_rope = q_normed  # snapshot
    q_post_rope = attn.rope(q_normed)
    mx.eval(q_pre_rope, q_post_rope)

    # Q head 4, last position (maps to KV head 2)
    q_h4_pre = q_pre_rope[0, 4, -1, :]
    q_h4_post = q_post_rope[0, 4, -1, :]
    # KV head 2
    k_kv2 = k_volt[0, 2, 0, :]               # post-RoPE at doc pos 39
    k_kv2_pre = k_volt_pre_rope[0, 2, 0, :]  # pre-RoPE

    # Re-RoPE K at position S_q (appended right after query)
    k_pre_reshaped = k_volt_pre_rope  # (1, 4, 1, 256)
    k_reroped = attn.rope(k_pre_reshaped, offset=S_q)
    k_kv2_reroped = k_reroped[0, 2, 0, :]

    # Re-RoPE K at position 0
    k_reroped_0 = attn.rope(k_pre_reshaped, offset=0)
    k_kv2_reroped_0 = k_reroped_0[0, 2, 0, :]

    mx.eval(k_kv2_reroped, k_kv2_reroped_0)

    scale = attn.scale
    qk_a = float((q_h4_post * k_kv2).sum().item()) * scale
    qk_b = float((q_h4_pre * k_kv2_pre).sum().item()) * scale
    qk_c = float((q_h4_post * k_kv2_reroped).sum().item()) * scale
    qk_d = float((q_h4_post * k_kv2_reroped_0).sum().item()) * scale

    print(f"(a) Q·K mismatched RoPE (doc pos 39):           {qk_a:.4f}")
    print(f"(b) Q·K pre-RoPE (content only):                {qk_b:.4f}")
    print(f"(c) Q·K re-RoPE at pos {S_q} (append after query): {qk_c:.4f}")
    print(f"(d) Q·K re-RoPE at pos 0:                       {qk_d:.4f}")

    # --- Full injection with re-RoPE'd K at pos S_q ---
    h_post_c, _ = M.run_layer29_with_injected_kv(h_pre, k_reroped, v_volt)
    h_final_c = M.forward_from_layer(h_post_c, 30)
    logits_c = M.get_logits(h_final_c)
    p_volt_c = M.prob_of_token(logits_c, TARGET_TOKEN_ID)

    # --- Full injection with re-RoPE'd K at pos 0 ---
    h_post_d, _ = M.run_layer29_with_injected_kv(h_pre, k_reroped_0, v_volt)
    h_final_d = M.forward_from_layer(h_post_d, 30)
    logits_d = M.get_logits(h_final_d)
    p_volt_d = M.prob_of_token(logits_d, TARGET_TOKEN_ID)

    # --- Injection with pre-RoPE K (no positional encoding at all) ---
    h_post_b, _ = M.run_layer29_with_injected_kv(h_pre, k_volt_pre_rope, v_volt)
    h_final_b = M.forward_from_layer(h_post_b, 30)
    logits_b = M.get_logits(h_final_b)
    p_volt_b = M.prob_of_token(logits_b, TARGET_TOKEN_ID)

    print(f"\nP(Volt) with original RoPE (Exp 4): (already measured)")
    print(f"P(Volt) with pre-RoPE K (no pos):   {p_volt_b:.4f}  ({p_volt_b*100:.1f}%)")
    print(f"P(Volt) with re-RoPE at pos {S_q}:     {p_volt_c:.4f}  ({p_volt_c*100:.1f}%)")
    print(f"P(Volt) with re-RoPE at pos 0:       {p_volt_d:.4f}  ({p_volt_d*100:.1f}%)")

    result = {
        "description": "RoPE position mismatch analysis for K injection",
        "QK_scores": {
            "a_original_rope_doc_pos_39": round(qk_a, 4),
            "b_pre_rope_content_only": round(qk_b, 4),
            "c_reroped_at_S_query": round(qk_c, 4),
            "d_reroped_at_pos_0": round(qk_d, 4),
        },
        "P_Volt": {
            "b_pre_rope": round(p_volt_b, 6),
            "c_reroped_at_S_query": round(p_volt_c, 6),
            "d_reroped_at_pos_0": round(p_volt_d, 6),
        },
        "S_query": S_q,
    }
    with open(f"{OUTDIR}/exp6_rope_analysis.json", "w") as f:
        json.dump(result, f, indent=2)

    return result


def experiment_7(
    M: ModelHelper,
    k_doc: mx.array,
    v_doc: mx.array,
    doc_ids: mx.array,
) -> dict:
    """Scale test: 8-entry synthetic KV cache, 3 different queries."""
    print("\n" + "=" * 60)
    print("EXPERIMENT 7: Scale Test (3 queries, 8-entry cache)")
    print("=" * 60)

    doc_len = doc_ids.shape[0]
    step = max(1, (doc_len - 1) // 8)
    sample_positions = [1 + i * step for i in range(8)]
    sample_positions = [min(p, doc_len - 1) for p in sample_positions]

    print("Sample positions:")
    for pos in sample_positions:
        tok = M.tokenizer.decode([int(doc_ids[pos].item())])
        print(f"  pos {pos}: '{tok}'")

    k_samples = mx.concatenate(
        [k_doc[:, :, p:p+1, :] for p in sample_positions], axis=2
    )
    v_samples = mx.concatenate(
        [v_doc[:, :, p:p+1, :] for p in sample_positions], axis=2
    )
    mx.eval(k_samples, v_samples)

    queries = [
        ("Zarkov Industries was founded in the city of", "Volt", 89711),
        ("The headquarters of Zarkov Industries was built on a", "former", 4937),
        ("Zarkov Industries was established in the", "mid", 5453),
    ]

    results = []
    for query_text, target_tok, target_id in queries:
        q_ids = M.tokenize(query_text)
        S_q = q_ids.shape[0]

        h_pre = M.forward_to_layer(q_ids, 28)
        h_post, attn_w = M.run_layer29_with_injected_kv(h_pre, k_samples, v_samples)
        h_final = M.forward_from_layer(h_post, 30)

        logits = M.get_logits(h_final)
        p_target = M.prob_of_token(logits, target_id)
        top_preds = M.top_k_predictions(logits, 5)

        # Which injected entry gets most H4 attention?
        attn_np = np.array(attn_w.astype(mx.float32))
        inject_w = attn_np[0, 4, S_q - 1, S_q:]
        best_idx = int(np.argmax(inject_w))
        best_pos = sample_positions[best_idx]
        best_tok = M.tokenizer.decode([int(doc_ids[best_pos].item())])

        entry = {
            "query": query_text,
            "target": target_tok,
            "target_id": target_id,
            "P_target": round(float(p_target), 6),
            "top_predictions": top_preds,
            "best_attended_idx": best_idx,
            "best_attended_pos": best_pos,
            "best_attended_token": best_tok,
            "inject_attention_H4": [round(float(w), 6) for w in inject_w],
        }
        results.append(entry)

        print(f"\nQuery: '{query_text}'")
        print(f"  P({target_tok}) = {p_target:.4f}  ({p_target*100:.1f}%)")
        print(f"  Best attended: pos {best_pos} ('{best_tok}'), w={inject_w[best_idx]:.4f}")
        for p in top_preds[:3]:
            print(f"    {p['token']:>15s}  {p['probability']:.4f}")

    with open(f"{OUTDIR}/exp7_scale_test.json", "w") as f:
        json.dump(results, f, indent=2)

    return {"queries": results}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def verify_forward_pass(M: ModelHelper):
    """Verify that our manual layer-by-layer forward matches the model's own forward."""
    print("\n" + "=" * 60)
    print("VERIFICATION: Manual forward vs model forward")
    print("=" * 60)

    query_ids = M.tokenize(QUERY)
    if query_ids.ndim == 1:
        query_ids_2d = query_ids[None, :]
    else:
        query_ids_2d = query_ids

    # Full model forward
    output = M.model(query_ids_2d)
    logits_model = output.logits[0, -1, :]
    mx.eval(logits_model)

    # Manual forward: all layers + final_norm + lm_head
    h = M.forward_to_layer(query_ids, M.num_layers - 1)
    logits_manual = M.get_logits(h)
    mx.eval(logits_manual)

    # Compare
    model_probs = mx.softmax(logits_model)
    manual_probs = mx.softmax(logits_manual)
    mx.eval(model_probs, manual_probs)

    model_top = int(mx.argmax(logits_model).item())
    manual_top = int(mx.argmax(logits_manual).item())

    model_tok = M.tokenizer.decode([model_top])
    manual_tok = M.tokenizer.decode([manual_top])

    diff = float(mx.max(mx.abs(logits_model - logits_manual)).item())

    print(f"Model top-1:  '{model_tok}' (id={model_top}, p={float(model_probs[model_top].item()):.4f})")
    print(f"Manual top-1: '{manual_tok}' (id={manual_top}, p={float(manual_probs[manual_top].item()):.4f})")
    print(f"Max logit diff: {diff:.6f}")

    if model_top != manual_top:
        print("WARNING: Top-1 tokens differ!")
    else:
        print("OK: Top-1 tokens match.")

    # Also verify split forward: L0-28, L29 (normal), L30-33
    h_pre = M.forward_to_layer(query_ids, 28)

    # Run L29 normally
    mask = M._make_mask(query_ids.shape[0] if query_ids.ndim == 1 else query_ids.shape[1], h_pre.dtype)
    layer_out = M.layers[29](h_pre, mask=mask, cache=None)
    h_post = layer_out.hidden_states
    mx.eval(h_post)

    h_final = M.forward_from_layer(h_post, 30)
    logits_split = M.get_logits(h_final)
    mx.eval(logits_split)

    split_top = int(mx.argmax(logits_split).item())
    split_tok = M.tokenizer.decode([split_top])
    diff_split = float(mx.max(mx.abs(logits_model - logits_split)).item())
    print(f"Split top-1:  '{split_tok}' (id={split_top})")
    print(f"Max logit diff (split vs model): {diff_split:.6f}")

    # Verify custom L29 (no injection — just self-attend)
    # This tests run_layer29_with_injected_kv with N=0 injected entries
    # Equivalent: run with injected_k/v of shape (1, 4, 0, 256)
    empty_k = mx.zeros((1, M.num_kv_heads, 0, M.head_dim), dtype=h_pre.dtype)
    empty_v = mx.zeros((1, M.num_kv_heads, 0, M.head_dim), dtype=h_pre.dtype)
    h_custom, _ = M.run_layer29_with_injected_kv(h_pre, empty_k, empty_v)
    h_final_custom = M.forward_from_layer(h_custom, 30)
    logits_custom = M.get_logits(h_final_custom)
    mx.eval(logits_custom)

    custom_top = int(mx.argmax(logits_custom).item())
    custom_tok = M.tokenizer.decode([custom_top])
    diff_custom = float(mx.max(mx.abs(logits_model - logits_custom)).item())
    print(f"Custom L29 (no inject) top-1: '{custom_tok}' (id={custom_top})")
    print(f"Max logit diff (custom vs model): {diff_custom:.6f}")

    if diff_custom > 1.0:
        print("ERROR: Custom L29 diverges significantly from model!")
        print("  Debugging: checking intermediate values...")
        # Check if h_post and h_custom match
        diff_h = float(mx.max(mx.abs(h_post - h_custom)).item())
        print(f"  L29 output diff: {diff_h:.6f}")

    return diff_custom < 1.0


def main():
    pipeline = load_model()
    M = ModelHelper(pipeline)

    # Verify the forward pass pipeline first
    if not verify_forward_pass(M):
        print("\nFORWARD PASS VERIFICATION FAILED. Debugging required.")
        sys.exit(1)

    doc_ids = M.tokenize(DOCUMENT)

    # Experiment 2: Extract K/V
    exp2_result, k_volt, v_volt, k_noise, v_noise, k_volt_pre_rope = experiment_2(M)

    # Experiment 4: Single entry injection
    exp4_result, h_pre = experiment_4(M, k_volt, v_volt)

    # Experiment 5: Multi entry injection
    exp5_result = experiment_5(
        M, h_pre, k_volt, v_volt, k_noise, v_noise,
        exp2_result["noise_positions"], doc_ids,
    )

    # Experiment 6: RoPE analysis
    exp6_result = experiment_6(M, h_pre, k_volt, v_volt, k_volt_pre_rope)

    # Experiment 7: Scale test
    # Re-extract full K/V from document (need all positions)
    _, _, k_doc, v_doc, _ = M.extract_kv_at_layer29(doc_ids)
    exp7_result = experiment_7(M, k_doc, v_doc, doc_ids)

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Exp 4 — Single injection:  P(Volt) = {exp4_result['P_Volt']:.4f}")
    print(f"Exp 5 — Multi injection:   P(Volt) = {exp5_result['P_Volt']:.4f}")
    print(f"Exp 6 — RoPE analysis:")
    for k, v in exp6_result["P_Volt"].items():
        print(f"  {k}: P(Volt) = {v:.4f}")
    print(f"Exp 7 — Scale test:")
    for q in exp7_result["queries"]:
        print(f"  '{q['query'][:40]}...' → P({q['target']}) = {q['P_target']:.4f}")

    # Write combined summary
    summary = {
        "exp2": exp2_result,
        "exp4": exp4_result,
        "exp5": exp5_result,
        "exp6": exp6_result,
        "exp7": exp7_result,
    }
    with open(f"{OUTDIR}/summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nAll results saved to {OUTDIR}/")


if __name__ == "__main__":
    main()
