# Pre-RoPE Q·K Routing Results

**Experiment:** 76a56876
**Model:** google/gemma-3-4b-it (MLX)
**Date:** 2026-03-21

## Summary

Removing RoPE improved landing from #36→#9 (removing positional
recency bias). But revealed a SECOND wall: pre-RoPE K-vectors
encode token identity, not passage-specific content. K-norm
sampling captures structural tokens (CDR, Apollo, Roger, \<bos\>)
that appear across many windows. The 256D head space without
RoPE has discrimination ratio 0.99× (no clustering).

**Two walls, not one:**
1. RoPE creates positional bias (fixed by pre-RoPE)
2. Pre-RoPE K-space encodes token identity, not context

**Keywords confirmed final** — for the third time.

---

## Experiment 1 — Pre-RoPE K Quality Check

K-norm top-8 positions per passage. Pairwise cosine within vs
across passages (KV-head 2, 256D).

| Passage | Top K-norm tokens |
|---------|---|
| Zarkov | climate, Dimitri, ark, \<bos\>, robotics, Volt, Industries, - |
| Strand | diplomat, Archaeological, Strand, \<bos\>, " ", Carpathian, Strand, Cast |
| Kelvara | El, ara, teleport, quantum, Kel, Voss, mountain, \<bos\> |

| Metric | Value |
|--------|------:|
| Mean intra-passage cosine | 0.788 |
| Mean inter-passage cosine | 0.799 |
| **Discrimination ratio** | **0.99×** |

**No discrimination.** Pre-RoPE K-vectors at high-norm positions
are equally similar within and across passages. The K-space
encodes general token properties, not passage-specific content.

---

## Experiment 2 — Pre-RoPE Q·K at N=3

Pre-extracted K-vectors from 3 passages (8 per passage).
Pre-extracted Q from bare completion-template query. Q·K in 256D.

| Query | Zarkov max | Strand max | Kelvara max | Winner | Correct? |
|-------|:---:|:---:|:---:|---|:---:|
| Zarkov | 224 | 224 | 224 | Zarkov | ✓ (luck) |
| Strand | **240** | 240 | 240 | Zarkov | ✗ |
| Kelvara | **237** | 234 | 234 | Zarkov | ✗ |

**1/3 correct** (zarkov wins by rounding, margin 1.00×).

\<bos\> dominates: all passages yield identical max Q·K scores.
The bare query's Q-vector matches \<bos\>'s K-vector equally
everywhere because \<bos\> encodes generic sequence-start, not
content.

### Why live attention (with RoPE) got 3/3

Live attention puts all passages IN CONTEXT. The query's
residual stream at L29 contains information about all passages.
The Q-vector is contextualized — it encodes "looking for X
in the context of passages A, B, C." The pre-RoPE Q from
a bare query lacks this context.

---

## Experiment 3 — Pre-RoPE Q·K at N=50 (Apollo)

| Query | Pre-RoPE rank | Live+RoPE rank | Target | Top-1 | Top-1 token |
|-------|:---:|:---:|---|---|---|
| Porridge | #23 | #5 | W170 | W660 | Buzz |
| Baseball | #20 | #5 | W169 | W195 | Apollo |
| **Landing** | **#9** | **#36** | **W370** | W180 | CDR |
| Weather | #46 | #5 | W169 | W195 | Apollo |
| News | #48 | #5 | W169 | W150 | Roger |

**0/5 correct.** But landing improved #36→#9 (recency bias removed).

### What dominates

Structural tokens common across all windows:
- **CDR** — Commander designation, appears in most windows
- **Apollo** — mission name, appears everywhere
- **Roger** — acknowledgment, appears everywhere
- **Buzz** — crew member, appears in many windows
- **Tape** — transcription marker, appears everywhere

These tokens have high K-norms and produce high Q·K with
every query because they encode the document's identity
(Apollo transcript), not window-specific content.

### Landing: the best pre-RoPE result

Landing improved the most because:
1. With RoPE: recency bias pushed W370 to #36 (positional)
2. Without RoPE: structural tokens still dominate (#9)
   but W370 scores higher than average because it contains
   "Houston", "Tranquility", "EAGLE" — structurally important
   AND content-relevant tokens

The convergence of structural importance and content relevance
for landing-specific tokens is why landing benefits most from
RoPE removal. Other queries' targets (W169, W170) contain
news content that doesn't overlap with Apollo structural tokens.

---

## Experiment 4 — Multi-Head Pre-RoPE

| Query | H4 rank | Multi-head rank | Improved? |
|-------|:---:|:---:|:---:|
| Porridge | #23 | #19 | ✓ |
| Baseball | #20 | #5 | ✓ |
| Landing | #9 | #20 | ✗ |
| **Weather** | **#46** | **#1** | **✓** |
| News | #48 | #37 | ✓ |

Multi-head (H2+H3+H4+H5) helps 4/5 queries but hurts landing
(#9→#20). Different heads carry different biases. Weather jumps
to #1 — H2/H3 may encode weather-specific routing that H4 misses.

---

## Root Cause Analysis

### The two-wall model

```
Full attention (works)
  = Contextual residual → W_Q/W_K → Q·K with context → RoPE
                    ↑                      ↑               ↑
              Full document         Content + context    Position
              in context            from 28 layers       awareness

Pre-RoPE stored K (fails)
  = Bare passage residual → W_K → K without RoPE
              ↑                         ↑
         No query context        Token identity only
         No cross-passage        No passage-specific
         information             content
```

The content-discriminative routing signal requires THREE things:
1. **Contextualized residual** — the query must be grounded in
   the document's space (all passages visible)
2. **Position-aware Q·K** — RoPE within the attention window
   provides relative position, which helps distinguish nearby
   tokens from distant ones
3. **Content matching** — the 256D projection amplifies content

Pre-RoPE removes #2. Bare-query extraction removes #1.
Only #3 remains, and it's insufficient alone.

### Why N=3 worked with live attention

Live attention provides ALL THREE:
1. All passages in context → contextualized residual
2. RoPE within 500 tokens → ~flat, doesn't hurt
3. 256D projection → content matching dominates

At N=50 with live attention, #2 overwhelms #3 (RoPE 100×
bias). Pre-RoPE fixes #2 but breaks #1 (K-vectors from
individual passages, no cross-context).

### The fundamental tension

The model's attention mechanism routes using CONTEXTUALIZED
representations — K-vectors that encode not just "this token
is CONTACT" but "this token is CONTACT in the context of a
lunar landing transcript at position 88." Stripping context
(pre-extracting K from bare passages) removes the contextual
information that makes routing work.

But keeping context requires the full KV cache, which is what
we're trying to avoid.

---

## Comparison Table (Updated)

| Method | N=3 | N=50 land. | N=50 all | Mechanism |
|--------|:---:|:---:|:---:|---|
| Token overlap | 3/3 | #43 | 4/5 | Vocabulary intersection |
| Keyword | 3/3 | #1 | 5/5 | Regex phrases |
| Q·K live+RoPE | 3/3 | #36 | 0/5 | Attention (contextual + positional) |
| **Q·K pre-RoPE** | **1/3** | **#9** | **0/5** | **Content only (no context, no RoPE)** |
| Q·K pre-RoPE multi-head | — | #20 | 0/5 | Multi-head content only |

---

## What This Means

### The 256D content signal is real but insufficient

The W_Q/W_K projection creates a 256D subspace. Within this
subspace, the completion template produces semantically-aligned
Q-vectors (proven at N=3 with live attention). But the content
signal in pre-RoPE K-space is:

1. **Dominated by token identity** — "CDR" maps to similar K
   regardless of which window it appears in
2. **Lacking context** — the K-vector for "CONTACT" in a bare
   window doesn't know it's about lunar landing
3. **Not passage-discriminative** — cosine ratio 0.99× means
   K-vectors from different passages are interchangeable

### RoPE was 50% of the problem

Removing RoPE improved landing from #36 to #9 — a 4× rank
improvement. But it also revealed that even without positional
bias, the content signal alone can't route.

### The model needs context to route

The model's routing mechanism is inherently contextual. The
K-vectors that route attention are built from contextual
residuals (28 layers of processing), not raw token embeddings.
Stripping them from context strips the routing signal.

This is WHY the full KV cache works: it preserves the contextual
K-vectors built during prefill. Any compressed representation
that discards this context will lose the routing signal.

### Keywords work because they're context-free

Keywords operate in vocabulary space: "porridge" matches
"porridge" regardless of context. They're the only
position-invariant, context-invariant routing mechanism
available.

**Architecture confirmed for the final time:**
**Keyword route → persistent inject.**

---

## Store Budget Comparison

| Approach | Size (Apollo) | Accuracy | Position-invariant? | Context-free? |
|----------|---:|:---:|:---:|:---:|
| Keywords | 25 KB | 5/5 | ✓ | ✓ |
| Pre-RoPE K | 2.94 MB | 0/5 | ✓ | ✗ (needs context) |
| Post-RoPE K | 2.94 MB | 0/5 | ✗ (RoPE bias) | ✗ |
| Full KV cache | 56 GB | 5/5 | N/A | N/A |

Keywords: 25 KB, 5/5, no model access needed.
Full KV: 56 GB, 5/5, full model access.
Pre-RoPE: 2.94 MB, 0/5, model weights needed.

The 117× compression from keywords to pre-RoPE K buys nothing.
