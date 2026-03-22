# SPEC: chuk-mlx (Lazarus) v12 — Knowledge Store Architecture

**Author:** Chris Hay
**Date:** 2026-03-22
**Status:** Engineering Spec (updated with experiment results)
**Repository:** chuk-mlx (chuk-lazarus)
**Prior:** v10 (1D injection), v11.1 (RoPE fix), additive-multidim (7f89fd70),
multi-position injection (f2ba7184)

---

## 1. The Core Insight

The 56 GB KV cache is an O(1/N) search engine delivering 8 bytes of
content. Even with the full cache loaded — every position, every layer,
every head — H5 drops from 26% attention to <0.5% at 2,500 tokens.
The softmax denominator grows linearly with sequence length. At 370,000
tokens, the model has perfect information and can't use it.

**The industry scales context windows (128K, 1M, 10M tokens) to fight
this with brute force. More memory. More positions. More hardware.
And at every scale, the same 1/N dilution.**

The model is brilliant at comprehension within 200 tokens. It's
terrible at retrieval across 370,000 tokens. The solution: find
which 50-512 tokens matter (external routing), and give the model
only those (focused delivery). The comprehension was never the
bottleneck. The addressing was.

---

## 2. What the Experiments Proved

### The Markov property in production (patch_all_positions)

`patch_all_positions=True` at L30 produces KL=0.0 with the donor.
Bit-perfect. All 7/10 facts. Works at L14, L24, L30 — all KL=0.0.
The full residual stream at any layer IS the complete document state.
Downstream layers rebuild their KV from the injected state.

### Facts are positional

"Coyle" lives at position 99. "23" lives at position 112. "Corby"
lives at position 88. The last-position residual carries topic
direction only ("porridge championship", "John"). Narrative details
require the positional KV entries where those tokens appear.

Last-position-only injection: P(correct fact) = 0%. Topic correct,
every detail confabulated.

### Grammar is load-bearing

| Context style | Facts correct |
|--------------|:------------:|
| Full sentence: "John Coyle, an Irishman, won...23 bowls of instant oatmeal" | 7/10 |
| Compressed: "John Coyle won porridge championship, 23 bowls oatmeal" | 6/10 |
| Telegram: "John Coyle, 23 bowls, Corby, 35 competitors" | 0/10 |

Without connective tissue ("of instant oatmeal", "porridge eating
championship"), the model can't resolve ambiguity ("23 bowls" →
bowling, not oatmeal). Grammar is not optional.

### The minimum viable context

A well-formed fact sentence of ~25-50 tokens delivers the same
narrative quality as a full 512-token window. The model needs syntax
to resolve semantics. But it doesn't need 512 tokens — just the
fact-bearing sentence with connective tissue intact.

### The softmax bottleneck is architectural, not a storage problem

Even with the full 56 GB KV cache loaded, H5 can't route at 2,500+
tokens. This is not fixable by loading more tokens. It's a property
of softmax attention. The model has the information. It can't find it.
External routing bypasses this entirely.

---

## 3. Three Delivery Tiers

The knowledge store supports three delivery mechanisms. Same routing
index. Same build pipeline. Different storage/speed/quality tradeoffs.

### Tier 1: Fact Sentence Replay (~210 KB total, ~1-2s query)

```
Store:  routing index (175 KB) + fact sentences (~35 KB text)
Query:  route → prepend fact sentence to query → prefill 50-90 tokens → generate
Result: 7/10 facts, grammatically correct
```

During build, the model reads each window and generates a focused
fact summary sentence (~25-50 tokens). Stored as text. At query time,
the fact sentence is prepended to the query. The model prefills
50-90 tokens (vs 512 for the full window, vs 200 for focused passage).

**The practical default.** Tiny store. Fast query. Correct narrative.

### Tier 2: Focused Context Replay (~175 KB total, ~3.8s query)

```
Store:  routing index (175 KB) + window token lists (for decoding)
Query:  route → decode focused passage (~200 tokens) → prefill → generate
Result: 7/10 facts, grammatically correct
```

The current v10 architecture. Decodes ~200 tokens from the stored
window token lists. The model reads real transcript text. No
generated summaries — the original text, focused around the
routing match.

**The conservative default.** No generated content in the store.
Original text only. Slightly slower than Tier 1.

### Tier 3: Markov Residual Injection (~461 MB total, <0.5s query)

```
Store:  routing index (175 KB) + L30 residual streams (461 MB bfloat16)
Query:  route → load residual stream → patch_all_positions at L30 → generate
Result: 7/10 facts, KL=0.0 (bit-perfect match to full context)
```

The full Markov state transplant. Zero tokens prefilled. The stored
L30 residual stream replaces the hidden state entirely. L31-L33
rebuild their KV from the injected state. Bit-perfect output.

**The proof.** No context window whatsoever. No tokens. No prefill.
The document state is a tensor on disk. Load it. Inject it. Done.
Larger store but fastest query time and provably exact output.

### Comparison

| | Tier 1 (fact sentence) | Tier 2 (context replay) | Tier 3 (Markov residual) |
|-|:----------------------:|:-----------------------:|:------------------------:|
| Store (Apollo 176 win) | ~210 KB | ~175 KB | ~461 MB |
| Tokens prefilled | 50-90 | ~200 | 0 |
| Query time | ~1-2s | ~3.8s | <0.5s |
| KL vs full context | ~0 | ~0 | 0.0 (exact) |
| Facts correct | 7/10 | 7/10 | 7/10 |
| Generated content in store | Yes (sentences) | No (original text) | No (residual tensors) |
| Compression vs 56 GB | 267,000× | 320,000× | 121× |

---

## 4. Why This Works (and Why the KV Cache Fails)

### The KV cache is an O(1/N) search engine

```
Softmax attention:
  weight(position_i) = exp(Q·K_i) / Σ_j exp(Q·K_j)

As N grows:
  denominator grows linearly
  each position's weight shrinks as 1/N
  at N=370,000: each position gets ~0.0003% of attention budget
  H5 can't give >0.5% to any single answer token
  → routing fails
  → the model can't find the answer
  → even with PERFECT information in the KV cache
```

### The Lazarus architecture bypasses the bottleneck

```
External routing: TF-IDF token matching → O(1) per window
  "porridge" matches window 170 in 0.3ms
  No attention. No softmax. No 1/N dilution.

Focused delivery: 50-512 tokens of relevant context
  Within 200 tokens, H5 gives 26% to the answer
  The model's comprehension is perfect at this scale
  No architectural change needed — just fewer positions

The KV cache existed for addressing (search 370K positions)
We replaced the addressing (token matching, 0.3ms)
The comprehension was never the problem
```

### Both tiers reconstruct what attention needs

The full KV cache gives L31-L33 attention 370,000 positions to search.
H5 uses maybe 10 of them. All three delivery tiers give attention only
the positions that contain facts:

- **Tier 1/2:** prefill 50-200 tokens → L31-L33 build KV from
  real token processing → attention reads the fact positions
- **Tier 3:** inject L30 residual stream → L31-L33 rebuild KV from
  stored state → attention reads the same positions, same hidden states

Both produce the same output because both give attention the same
content to attend to. The 56 GB was overhead for finding those positions.

---

## 5. Build Pipeline

```
BUILD (once, ~164-180 seconds for 90K tokens):

  Pass 1 — Token Index (fast, no model):
    Tokenise into 512-token windows
    Compute IDF across all windows
    Store unique token IDs per window
    Store ordered token lists per window

  Pass 2 — Prefill + Residual Capture (one forward per window):
    Chain boundary residuals (Markov property)
    Store full L30 residual stream per window (bfloat16)  [Tier 3]
    Store boundary residual for next window

  Pass 3 — Model-Native Keywords + Fact Sentences (one generation per window):
    Completion-template: window text + "Topic:"
    Generate 5 topic tokens per window (routing keywords)
    Add variant token IDs (case/space) to routing index
    Generate fact summary sentence per window (~25-50 tokens) [Tier 1]

  Output:
    routing index:       175 KB  (all tiers)
    fact sentences:      ~35 KB  (Tier 1)
    window token lists:  ~57 KB  (Tier 2, for decoding)
    residual streams:    461 MB  (Tier 3)
```

### Pass 2: Residual stream capture

```python
for wid, chunk_ids in enumerate(windows):
    w_ids = mx.array(chunk_ids)[None]

    h = kv_gen.prefill_to_layer(
        w_ids,
        target_layer=config.crystal_layer,
        initial_residual=boundary_residual,
    )
    # h: (1, seq_len, 2560)

    # Tier 3: store full residual stream
    if store_residuals:
        stream = h[0].astype(mx.bfloat16)
        mx.eval(stream)
        residual_streams[wid] = stream

    # Chain boundary for next window
    boundary_residual = h[:, -1:, :]
    mx.eval(boundary_residual)
```

### Pass 3: Fact sentence generation (Tier 1)

```python
for wid, chunk_ids in enumerate(windows):
    window_text = tokenizer.decode(chunk_ids, skip_special_tokens=True)
    summary_prompt = (
        f"{window_text}\n\n"
        "Summarise the key facts in one sentence."
    )
    summary_ids = tokenizer.apply_chat_template(
        [{"role": "user", "content": summary_prompt}],
        add_generation_prompt=True,
    )
    summary_text = generate(model, summary_ids, max_tokens=50)
    fact_sentences[wid] = summary_text
```

---

## 6. Store Format (v12)

```
knowledge_store/
├── manifest.json              # version, tier, metadata
├── window_tokens.npz          # unique token IDs per window (routing)
├── window_token_lists.npz     # ordered tokens per window (Tier 2 decode)
├── idf.json                   # IDF table (routing weights)
├── keywords.json              # model-generated topic keywords (routing)
├── entries.npz                # 1D injection entries (optional, entity names)
├── fact_sentences.json        # generated fact summaries (Tier 1)
├── residuals/                 # L30 residual streams (Tier 3)
│   ├── window_000.npy         # (seq_len, 2560) bfloat16
│   ├── window_001.npy
│   └── ...
└── boundary_residual.npy      # final boundary (for future chaining)
```

All tiers share the routing index (window_tokens, idf, keywords).
Tier 1 adds fact_sentences.json. Tier 3 adds the residuals/ directory.
The manifest records which tier was built.

Residuals are loaded LAZILY — only the winning window(s) at query time.

---

## 7. Query Pipelines

### Tier 1: Fact Sentence

```python
window_ids = store.route_top_k(query, tokenizer, k=3)
fact_text = store.get_fact_sentence(window_ids[0])

content = f"{fact_text}\n\n{query}"
prompt_ids = tokenizer.apply_chat_template(
    [{"role": "user", "content": content}],
    add_generation_prompt=True,
)
generated = generate(model, prompt_ids, max_tokens=80)
```

### Tier 2: Context Replay

```python
window_ids = store.route_top_k(query, tokenizer, k=3)
passages = [store.get_focused_passage(wid, query, tokenizer)
            for wid in window_ids]
context = "\n".join(passages)

content = f"{context}\n\n{query}"
prompt_ids = tokenizer.apply_chat_template(
    [{"role": "user", "content": content}],
    add_generation_prompt=True,
)
generated = generate(model, prompt_ids, max_tokens=80)
```

### Tier 3: Markov Residual

```python
window_ids = store.route_top_k(query, tokenizer, k=1)
donor_stream = store.load_residual(window_ids[0])

prompt_ids = prepare_prompt(tokenizer, query)
h_query = kv_gen.prefill_to_layer(prompt_ids, target_layer=crystal_layer)

donor = donor_stream.reshape(1, -1, donor_stream.shape[-1])
inject_logits, kv_store = kv_gen.prefill_from_layer(
    donor, start_layer=crystal_layer + 1,
)
generated = generate_from_logits(kv_gen, inject_logits, kv_store,
                                  donor.shape[1], max_tokens=80)
```

### Multi-Window (Tier 3)

```python
streams = [store.load_residual(wid) for wid in window_ids[:3]]
combined = mx.concatenate(streams, axis=0)  # (1536, 2560)
```

---

## 8. Implementation Checklist

### Phase 1: Tier 1 — Fact Sentence Store

- [ ] Add fact sentence generation to Pass 3 (build pipeline)
- [ ] Store fact_sentences.json (window_id → text)
- [ ] Query pipeline: prepend fact sentence to query
- [ ] Test: 50-90 token prefill produces correct narrative
- [ ] Measure: query time improvement vs 200-token replay

### Phase 2: Tier 3 — Markov Residual Store

- [ ] Capture full L30 residual stream per window in Pass 2
- [ ] Save as bfloat16 .npy files in residuals/ directory
- [ ] Lazy loading: load_residual(window_id) on demand
- [ ] Implement generate_with_markov_injection
- [ ] Handle KV merging (L0-L30 from query, L31-L33 from donor)
- [ ] Test: KL ≈ 0.0 through the CLI
- [ ] Multi-window concatenation

### Phase 3: Compression (Tier 3 stretch goal)

- [ ] Test PCA rank-50 per window: KL vs full
- [ ] Test PCA rank-20: KL vs full
- [ ] If rank-50 gives KL < 0.01: implement compressed storage
- [ ] Compressed store target: <20 MB for Apollo

### Phase 4: CLI Integration

- [ ] `--tier` flag: `sentence` / `context` / `residual`
- [ ] Default: `context` (current behaviour, backwards compatible)
- [ ] `lazarus knowledge build --tier all` stores everything
- [ ] `lazarus knowledge build --tier sentence` stores routing + sentences
- [ ] `lazarus knowledge build --tier residual` stores routing + residuals

---

## 9. Key Numbers

| Metric | Tier 1 (sentence) | Tier 2 (context) | Tier 3 (Markov) |
|--------|:-----------------:|:-----------------:|:---------------:|
| Store total | ~210 KB | ~175 KB | ~461 MB |
| Tokens prefilled | 50-90 | ~200 | 0 |
| Query time | ~1-2s | ~3.8s | <0.5s |
| KL vs full context | ~0 | ~0 | 0.0 |
| Facts correct | 7/10 | 7/10 | 7/10 |
| Compression vs 56 GB | 267,000× | 320,000× | 121× |
| Build time | ~180s | ~164s | ~164s |

### Tier 3 with PCA compression

| Rank | Store total | Compression | KL (TBD) |
|-----:|:----------:|:-----------:|:--------:|
| 50 | ~18 MB | 3,100× | TBD |
| 20 | ~7 MB | 8,000× | TBD |
| 9 | ~3.2 MB | 17,500× | TBD |

---

## 10. What This Means

Three delivery tiers. One routing index. Same correct answers.

**Tier 1** is the smallest (210 KB). Model-generated fact sentences.
50-90 tokens of prefill. ~1-2 seconds. For mobile, edge, constrained
deployments. The model reads its own summary and produces correct output.

**Tier 2** is the most conservative (175 KB). Original transcript text.
200 tokens of prefill. ~3.8 seconds. For deployments that need original
source text, not generated summaries. Shipping today.

**Tier 3** is the purest (461 MB). Zero tokens. KL=0.0. The Markov
property as a product. For deployments that need bit-perfect output
and have storage. The proof that the context window is truly eliminated.

All three bypass the softmax bottleneck. All three give the model
focused content within its attention capacity. All three produce
correct, grounded, factual answers.

370,000 tokens. 175 kilobytes to 461 megabytes. On a MacBook.

There is no context window.

---

## 11. Future: Fixing the Softmax Bottleneck

The current architecture bypasses the softmax bottleneck with external
routing. The model's own attention can't route at scale. This is
the remaining open problem.

If the softmax bottleneck is fixed:
- No external routing needed
- The model reads the full document and finds facts by itself
- The KV cache becomes useful again (but smaller — don't need all layers)
- Tier 3 residual injection still faster (no prefill)

Possible directions:
- Sparse attention patterns (learned pre-filtering before softmax)
- W_K token identity signal (fix RoPE-content interaction at distance)
- Multi-scale attention (local for grammar, global for facts)
- Dark space routing (99.4% of residual is invisible to unembedding —
  route through the dark space instead of through W_K)

This is the research frontier. Videos 4, 5, 6. The product ships now.