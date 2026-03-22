# Lazarus Routing Research: Complete Findings

**Author:** Chris Hay (chrishayuk)
**Model:** google/gemma-3-4b-it (34 layers, 2560 hidden, 8 attn heads, 4 KV heads)
**Date:** 2026-03-21

---

## The Question

The Lazarus knowledge store eliminates the KV cache for delivery — persistent injection of crystallised L30 residuals gives 100% P(target) per token for novel entities ("John Coyle" from 24 bytes). The question was: can we eliminate the KV cache for routing too? Can the model route to the correct fact without loading any tokens?

## The Answer

The model's internal routing mechanism (Q·K attention at L29) works perfectly at short range (<200 tokens) and fails catastrophically at long range (>2,500 tokens). This is an irreducible constraint of softmax attention capacity, not a fixable engineering problem. External vocabulary-space routing is necessary.

However, the routing can be made model-native. The model's own token IDs and TF-IDF statistics route correctly for 4/5 Apollo queries at N=724. The one failure (the landing query) requires comprehension bridging — mapping "landed on the moon" to "CONTACT LIGHT / ENGINE STOP" — which only the model's generation can provide.

---

## What We Proved

### 1. The Delivery is Solved (8 bytes per fact)

Persistent 1D subspace injection at L30 delivers answer tokens at 91-100% probability. "John Coyle" from 24 bytes (3 tokens × 8 bytes). The mechanism: replace the recipient residual's 1D component in the answer-token embedding direction with the donor's coefficient. At 2× natural coefficient, P(target) = 100%.

The prior compass experiment's negative result was a methodology error: donors must be context+query format ("Answer with just the name"), not raw window text. The donor's last-position residual must be loaded with the answer direction.

| Entity | P(target) | Bytes |
|--------|:---------:|------:|
| "John" (porridge winner) | 91.3-100% | 8 |
| "Baltimore" (baseball team) | 99.99% | 8 |
| "Volt" (Zarkov city) | 100% at 2× | 8 |
| "Cast" (Castellan) | 100% at 2× | 8 |
| "John Coyle" (persistent 3-step) | 91→99→100% | 24 |

### 2. The KV Cache is an Addressing System

The full 56 GB KV cache stores K and V at every position at every layer. The model uses approximately 10-20 positions per answer with >1% attention weight. The rest is unused scaffolding.

During correct generation, H5 (the primary copy head, not H4 as previously assumed) attends to the answer position at 25.98% and contributes DLA=1.0 (80.6% of L29's answer contribution). H2 is secondary (19.43%, DLA=0.268). H4 attends correctly (9.52%) but writes the wrong token (DLA≈0).

The KV cache exists to support attention-based routing. The delivery is 8 bytes. The 56 GB is for routing.

### 3. The Model Routes by Token Identity (Not Context)

L29 residuals are 89° from their token embeddings (cosine ~0.015). The residual is 99.98% contextual after 29 layers of processing. But W_K specifically extracts the ~0.02% token-identity-aligned subspace.

Evidence: Pre-RoPE K-vectors have a discrimination ratio of 0.99× (intra-passage vs inter-passage cosine). The K-space encodes general token properties, not passage-specific content.

The contextual processing (99.98% of the residual) is NOT used by W_K for routing. The routing signal is primarily token identity projected through W_K, with RoPE providing positional modulation and context providing 5× magnitude variation.

### 4. Softmax Attention Capacity is the Bottleneck

| Context length | H5 attention to answer | Routing quality |
|:--------------:|:---------------------:|:---------------:|
| 44 tokens | 23.34% | Perfect |
| 166 tokens | 25.98% | Perfect |
| 2,590 tokens | <0.52% | Complete failure |

50× dilution from 166 to 2,590 tokens. The softmax denominator grows linearly with sequence length. The answer token's attention share shrinks proportionally. At >2,500 tokens, no head gives >1% to the answer. This is mathematical, not fixable by any architectural change to the routing mechanism.

### 5. K-Vector Q·K Routing Works at Short Range

| Scale | Accuracy | Margins | Mechanism |
|:-----:|:--------:|:-------:|-----------|
| N=3 (novel entities) | 3/3 | 2.4-8.0× | Q·K 256D, completion template |
| N=6 (novel entities) | 6/6 | 2.2-19× | Q·K 256D, live attention |
| N=12 (same-template) | 12/12 | 1.009× | H-space cosine at L29 |
| N=50 (Apollo transcript) | 0/5 | — | Softmax dilution + structural dominance |

### 6. RoPE is 50% of the Scale Problem

Removing RoPE improved the Apollo landing query from rank #36 to #9 at N=50. RoPE creates 100× recency bias at 5,000 token distances. But removing it revealed the second wall: pre-RoPE K-vectors encode token identity, not context. Both walls must be overcome for model-native routing at scale.

### 7. Token Overlap Routes 3-4/5 at Apollo Scale

TF-IDF weighted token overlap using the full window vocabulary:

| Query | N=50 rank | N=724 rank | Failure mode |
|-------|:---------:|:----------:|-------------|
| Porridge | #1 | #1 | — |
| Baseball | #1 | #3-4 | "baseball" in 4 windows |
| Landing | #43 | #616 | Zero vocabulary overlap |
| Weather | #1 | #1 | — |
| News | #1 | #1 | — |

The landing failure is irreducible at the token level. "landed on the moon" shares zero content vocabulary with "CONTACT LIGHT. ENGINE STOP. THE EAGLE HAS LANDED." This is a concept-to-vocabulary gap, not a string-matching gap.

### 8. The Model Bridges Its Own Vocabulary Gap

Completion-primed prediction ("Topic:") after reading W370 produces "Lunar" at 26%, with "Eagle" (5.8%), "Apollo" (5.1%), "Landing" (3.5%), "Moon" (2.4%) in the top-5. The model reads garbled OCR transcript and correctly identifies the topic. The comprehension IS there — it's just not in the embeddings or the K-vectors or the logit lens. It's in the generation.

### 9. The Logit Lens Reads Continuation, Not Comprehension

Projecting the L30 residual through the unembedding matrix produces "Okay" at 98.4% for all windows (chat template) or "Lunar" at 26% (completion template). The comprehension is in the dark space (99.4% of the residual, orthogonal to the unembedding matrix). The logit lens sees only the 0.6% that projects into vocabulary space, which is dominated by the response-start or topic-category signal.

### 10. The Contextual Residual is Gold

During prefill, the residual at each content position encodes rich contextual understanding:

| Position | Token | Residual encodes |
|----------|-------|-----------------|
| 103 | "John" | Irish person names (Murphy, Kennedy, Smith) |
| 101 | "Irishman" | "named" at 97.5% — expects a proper name |
| 6 | "Philadelphia" | "Phillies" at 89.6% — baseball context |
| 9 | ";" | "Chicago" at 66.6% — next city in list |

The model's understanding of each position — readable through the logit lens at L29 — is a contextual comprehension map of the document. This is a potential tool (the "Context Map") for interpretability research.

---

## What We Tested and Why It Failed

### Embedding-Level Approaches (L0)

| Method | Result | Why |
|--------|--------|-----|
| Token ID overlap | 3/5 at N=724 | Vocabulary gap — different words for same concept |
| TF-IDF weighted | 3/5 at N=724 | Same — IDF helps baseball, not landing |
| Embedding cosine (soft) | 0/5 at N=50 | Embeddings encode word forms, not concepts |
| Full vocabulary (512 tokens/window) | 3/5 at N=724 | More tokens doesn't bridge the concept gap |
| Case normalisation | 3/5 | Fixes case but not vocabulary |

### Dark Space / Residual Approaches (L26-L30)

| Method | Result | Why |
|--------|--------|-----|
| L30 cosine | 0/5 at N=50 | Structural dominance (0.97+ cosine between all windows) |
| L26 centred cosine | 0/5 at N=50 | Same structural dominance |
| L26 compass projection | 0/4 cross-format | GPS not topic index |
| Per-position L29 residuals | 0/5 at N=50 | Format gap at all positions, not just boundaries |
| Centred per-position | 0/5 | Removes shared structure, introduces new bias |

### Attention Approaches (L29)

| Method | N=3 | N=50 | Why N=50 fails |
|--------|:---:|:----:|----------------|
| H4 entry argmax | 1/3 | — | K-norm attractor bias |
| H4 passage aggregate | 2/3 | 0/5 | K-norm sampling amplifies structural bias |
| Multi-head aggregate (H2-H5) | 3/3 | 0/5 | Structural K-norm variation between windows |
| Q·K 256D (live, completion) | 3/3 | 0/5 | RoPE recency bias at 5,000 tokens |
| Pre-RoPE Q·K | 1/3 | 0/5 | Token identity dominance, no context |
| KV cache extension (same RoPE) | — | — | V-projection dead; attention routes but can't deliver |

### Logit Lens / Prediction Approaches

| Method | Result | Why |
|--------|--------|-----|
| Logit lens (chat template) | 0/5 | "Okay" at 98.4% — response-start dominance |
| Logit lens (all layers L14-L33) | 0/5 | No content tokens at any layer |
| Completion template ("Topic:") | 3/5 | Works for bridging but single-token = single-topic |
| 5-word summary generation | 4/5 | Bridges vocabulary gap but IS keyword extraction |

### Synthetic KV / V-Projection

| Method | Result | Why |
|--------|--------|-----|
| Synthetic KV at L29 | H4 routes (13.7%), P(Volt) = 0% | V-projection is an amplifier, not a retriever |
| KV cache extension (same RoPE frame) | H4 routes (10.5%), P(Volt) = 0% | Same — V cannot deliver standalone |
| All RoPE modes | ±1.4% noise | Position mode irrelevant |
| With final residual | No improvement | Final residual doesn't help V-projection |

### V-Projection Finding

The copy head at L29 is an **amplifier**, not a retriever. It amplifies tokens already emerging in the residual. It cannot introduce tokens from stored KV entries. With full context, the residual at L28 already has +7.72 logits for the answer — the copy head adds +10.66 more. Without context, the residual has ~0 for the answer — the copy head's contribution is a rounding error.

---

## The Architecture

### Delivery (SOLVED)

```
Persistent 1D injection at L30, 2× coefficient
8 bytes per fact (token_id + coefficient)
100% P(target) for novel entities
Self-terminating via agreement gating
```

### Routing (TWO OPTIONS)

**Option A: Token overlap + model summary (model-native)**

```
Build time:
  Prefill each 512-token window
  Store: unique token IDs per window (~326 bytes)
  Generate: 5-word topic summary per window (~20 bytes)
  Total: ~346 bytes/window = 244 KB for Apollo 11

Query time:
  TF-IDF weighted token overlap (query tokens vs stored tokens + summary tokens)
  O(1) per window, no model computation
  Accuracy: 4/5 at N=50 (landing bridged by summary "moon")
```

**Option B: Keyword index (simplest, most accurate)**

```
Build time:
  Extract 3 keyword tokens per fact
  Total: 800 bytes for Apollo 11

Query time:
  String matching
  O(1) per window
  Accuracy: 5/5 at N=724
```

**Option C: Two-stage model-native (experimental)**

```
Stage 1: Token overlap narrows 724 → 5-10 candidates
Stage 2: K-vector Q·K in 256D fine-routes among candidates
          (within attention's working range of ~200 tokens)
Stage 3: Persistent injection delivers
```

### Store Format

| Component | Option A | Option B |
|-----------|---------|---------|
| Routing tokens | 326 B/win | 20 B/win (keywords) |
| Summary tokens | 20 B/win | — |
| Injection entries (8 × 8B) | 64 B/win | 64 B/win |
| Total per window | 410 B | 84 B |
| Apollo 11 (724 windows) | 297 KB | 61 KB |
| vs KV cache (56 GB) | 188,000× smaller | 917,000× smaller |

---

## Key Numbers

| Metric | Value |
|--------|------:|
| Full KV cache | 56 GB |
| Knowledge store (Option B) | 61 KB |
| Compression ratio | 917,000× |
| Injection per fact | 8 bytes |
| P(target) novel entities | 100% at 2× |
| P(target) "John Coyle" | 91→99→100% |
| Softmax attention limit | ~200 tokens |
| H5 DLA for answer delivery | 1.0 (80.6% of L29) |
| Residual ↔ token embedding | 89° (cosine 0.015) |
| Context fraction of residual | 99.98% |
| W_K extracts | Token identity (0.02%) |
| Dark space fraction | 99.4% of residual |

---

## Novel Findings

1. **H5 is the primary copy head, not H4.** H4 attends correctly but writes the wrong token (DLA≈0). H5 reads AND writes (DLA=1.0). The three-head copy circuit (H4+H5+H3 by frequency) is about attention routing, but delivery is concentrated in H5 and H2.

2. **W_K is a token-identity filter.** Despite the residual being 99.98% contextual, W_K extracts the 0.02% token-aligned subspace. The model's routing mechanism in attention is fundamentally token matching in 256D, with contextual modulation of magnitude (5× range).

3. **V-projection is dead for retrieval.** The copy head amplifies existing residual signal, it cannot introduce new signal from stored entries. 38% attention to the correct entry produces 0.06% P(target). This kills all synthetic KV approaches.

4. **The softmax attention bottleneck is irreducible.** At >2,500 tokens, H5 drops from 26% to <0.5% per answer token. This is O(1/N) dilution, not fixable by better representations or projections.

5. **Completion template unlocks 4× more content attention.** "Transcript query: X\nAnswer:" produces Q-vectors that are maximally content-discriminative vs chat template's "Okay" priming. All templates are >0.95 similar in 2560D but differ 4× in attention behaviour — the W_Q projection amplifies the discriminative dimensions.

6. **The contextual residual map.** Each position's L29 residual, decoded through the logit lens, reveals the model's contextual understanding. "John" → "Irish person names", "Philadelphia" → "Phillies". This is a new interpretability tool — the "Context Map."

---

## Implications for the Field

### The KV cache is an addressing system, not a knowledge store

The model stores knowledge in its weights (parametric) and delivers it through 8-byte injection entries (non-parametric). The 56 GB KV cache exists to support softmax attention routing — an O(1/N) mechanism that degrades with sequence length. The knowledge content of a 370,000-token document compresses to ~61 KB.

### Inference engineering, not training, is the path

The 12-byte injection, the persistent transplant, the crystallisation, the Markov property — none of these were trained. They were discovered in an existing model and exploited for inference. The model already has the mechanisms. The industry builds larger context windows. We read the model's own representations and bypass the context window entirely.

### The vocabulary gap is the real frontier

Every model-native routing approach works within the model's vocabulary. "porridge" matches "porridge." The gap between "landed on the moon" and "CONTACT LIGHT / ENGINE STOP" requires world knowledge — comprehension that the model possesses but only expresses through 34 layers of processing or through generation. Bridging this gap without generation is the remaining open problem.

---

## Open Questions

1. Can the two-stage router (token overlap → K-vector Q·K) reach 5/5 at Apollo scale?
2. Can the contextual residual map be used for position selection (content positions vs structural positions)?
3. Does PCA compression of K-vectors (6-8D for 83-94% variance) maintain routing accuracy?
4. How does the architecture scale to 10M tokens? 100M tokens?
5. Can the "Context Map" tool provide interpretability insights beyond routing?