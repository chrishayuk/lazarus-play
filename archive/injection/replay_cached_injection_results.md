# Replay-Cached Injection — Experiment Results

**Experiment ID:** dd304510-82a6-442d-b545-f58a1043bbf6
**Model:** google/gemma-3-4b-it
**Date:** 2026-03-19
**Builds on:** format_matched_pca16_results.md, addressing_precision_results.md

---

## The Question

The format-matched PCA-16 experiment established that H-space routing requires
document context in both the fact vector and the query vector. This creates a
circular dependency: V-injection exists to avoid loading the document, but
H-space routing requires loading it.

**Proposed fix:** Cache hidden states during replay fallback. When K-space routing
fails and the engine falls back to replay (full document + query forward pass), the
L29 hidden state is format-matched by construction. Cache it. Build a PCA-16 routing
index from cached replay entries. Next time a similar query arrives, route against the
cache instead of falling back again.

The binary gate: does the bare-query hidden state (no document) route against a
replay-cached hidden state (with document)?

---

## Experiment 1b — The Three-Condition Comparison

### Setup

Three prompts for the Namath fact ("Joe Namath agreed to sell his penthouse apartment"):

**Prefill** — single-fact template, 34 tokens:
```
<start_of_turn>user
Joe Namath agreed to sell his penthouse apartment.
Based on the above, what did Joe Namath agree to sell?<end_of_turn>
<start_of_turn>model
```

**Replay** — full 12-fact document + query, 170 tokens:
```
<start_of_turn>user
[all 12 facts]
Based on the above, what did Joe Namath agree to sell?<end_of_turn>
<start_of_turn>model
```

**Bare query** — question only, 18 tokens:
```
<start_of_turn>user
What did Joe Namath agree to sell?<end_of_turn>
<start_of_turn>model
```

Hidden states extracted at L29, last token position (the model-turn newline).

### Results

| Pair | Raw cosine | Raw angle | Centered angle |
|---|---|---|---|
| Prefill ↔ Replay | 0.9971 | 4.34° | 54.23° |
| Prefill ↔ Bare | 0.9821 | 10.86° | 153.57° |
| **Replay ↔ Bare** | **0.9826** | **10.70°** | **152.20°** |

**Norms:** Prefill = 59,608 · Replay = 59,873 · Bare = 66,195

**Dim-443 spike:** Prefill = 58,624 · Replay = 58,880 · Bare = 65,024

---

## The Binary Gate: Closed

**Replay ↔ Bare = 10.70°.** Prefill ↔ Bare = 10.86°. The improvement from
using a replay-cached vector instead of a prefill vector is **0.16°** — noise.

After centering (removing the shared template component), the result is starker:
bare query is **152°** from the replay hidden state. Nearly opposite direction.

The replay hidden state provides zero improvement over prefill for routing against
bare queries. The proposal fails at the first measurement.

---

## Why It Fails: The Document/No-Document Wall

The dim-443 spike reveals the mechanism.

| Context | Tokens | Dim-443 |
|---|---|---|
| Bare query | 18 | 65,024 |
| Single-fact template | 34 | 58,624 |
| Full-document replay | 170 | 58,880 |

The spike magnitude tracks **prompt length and document presence**, not semantic
content. Bare queries (no document) produce a systematically larger spike than any
document-context prompt, regardless of how similar the question text is.

This means bare queries occupy a categorically different region of L29 hidden space.
Not a different neighbourhood — a different hemisphere. The 152° post-centering angle
confirms this: the non-spike component of the bare query points roughly opposite to
both prefill and replay vectors.

The format gap is **document-presence-driven**, not content-driven. The previous
experiment showed content matters (generic padding makes things worse). This
experiment shows document presence matters more: no document context = wrong
hemisphere, full stop.

---

## What This Kills

The entire replay-caching architecture:

```
Stage 2: Geometric cache (NEW)
  H_query = hidden state at L29 (bare query, no document)
  H_centered = H_query - corpus_mean
  H_pca = PCA_basis @ H_centered
  scores = cache_pca @ H_pca   ← cache built from replay (with document)
```

The bare-query H and the replay-cached H are in opposite hemispheres after centering.
PCA-16 of the replay corpus cannot produce a basis that bridges a 152° gap.
The cosine scores would be negative — the highest score would be the most wrong entry.

No variant of this architecture works:
- **Different injection layer** — the document/no-document wall exists throughout the network
- **More PCA dimensions** — more dimensions of an opposite-hemisphere vector don't help
- **Corpus-mean centering** — the corpus mean is computed from document-context vectors; bare queries remain outliers
- **Combined prefill + replay basis** — both prefill and replay are on the same side of the wall

---

## What Survives

The routing architecture from `addressing_precision_results.md` is unchanged:

```
Query arrives
│
├─ Entity string match (for entity-explicit queries, ~75–83% coverage)
│   YES → inject (100% confident, 0% wrong injection)
│
├─ K-space adaptive Q·K (for entity-implicit queries)
│   max > mean × 2.0?
│   YES → inject
│
└─ Replay fallback (remaining ~15%)
    Load windows, generate (2s)
```

The geometric cache was proposed to reduce the replay fallback rate. That goal stands,
but geometry alone cannot achieve it. The 40% K-space ceiling (without entity string
filter) and ~85% hybrid ceiling (with entity string filter) are the correct numbers.

---

## The Geometry of the Wall

| Routing method | Namath ↔ Bare angle | Resolution |
|---|---|---|
| K-space (W_K projected, 256D) | — | FAILS (crowded) |
| H-space raw (2560D) | 10.86° | FAILS (spike dominates) |
| H-space centered (2560D) | 153.57° | FAILS (opposite hemisphere) |
| H-space replay-cached + centered | 152.20° | FAILS (same as raw centered) |
| Entity string filter | — | CORRECT (bypasses geometry) |

The entity string filter works because it bypasses the geometric wall entirely.
String matching doesn't care what hemisphere the hidden state is in.

---

## Residual Open Question

The 152° result raises a question not addressed here: **what is the bare-query
hidden state actually encoding?**

If it's nearly opposite to the document-context vectors, it's not just "less
information" — it's pointing at something specific. The dim-443 spike is larger
in bare queries, suggesting the model is in a different computational mode
(perhaps pure parametric retrieval vs. document retrieval). That's an interesting
finding in its own right, but it doesn't change the routing architecture.

---

## Summary

| Question | Answer |
|---|---|
| Does replay caching close the format gap? | No |
| Does Replay ↔ Bare < Prefill ↔ Bare? | No (10.70° vs 10.86°, difference < noise) |
| After centering, is the gap small? | No (152°, nearly opposite) |
| Is the gap content-driven? | No — document-presence-driven |
| Is the geometric cache viable? | No |
| Does the 40%/85% routing architecture survive? | Yes, unchanged |

**The replay path is not the learning path. It's just the slow path.**
The document/no-document divide is a hard geometric wall at L29.
Entity string filter remains the only viable route above the K-space ceiling.
