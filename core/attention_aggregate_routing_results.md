# Attention Aggregate Routing Results

**Experiment ID:** 9d2ca4a1-0aa8-47dd-942d-672c5005084f
**Model:** google/gemma-3-4b-it
**Date:** 2026-03-21

## Hypothesis

Individual H4 entry-level argmax fails at N=3 (1/3) because K-norm bias
dominates. But store-level totals from the prior experiment discriminate
(Zarkov: A=22.2% vs B=13.0%). The fix: sum H4 attention per passage and
pick the passage with the highest total. K-norm attractors in one entry
are diluted by the other entries in the passage.

Extension: use multiple heads (H2+H3+H4+H5) for ensemble discrimination.

## Experiments 1-3: N=3 (116 Entries, 3 Stores)

Three stores: A (Zarkov/Voltara, 40 entries), B (Strand/Castellan, 32),
C (Voss/Kelvara, 44). Three queries.

### Results

| Method | Zarkov→A | Director→B | Teleportation→C | Score |
|--------|:---:|:---:|:---:|:---:|
| Entry argmax (H4) | ✓ | ✗ | ✗ | 1/3 |
| H4 aggregate | ✓ | ✓ | ✗ | 2/3 |
| **Multi-head agg (H2-H5)** | **✓** | **✓** | **✓** | **3/3** |
| **All-head agg (H0-H7)** | **✓** | **✓** | **✓** | **3/3** |

### H4 Aggregate Scores

| Query | Store A | Store B | Store C | Winner | Margin |
|-------|:---:|:---:|:---:|:---:|:---:|
| Zarkov | **0.191** | 0.117 | 0.147 | A ✓ | 0.044 (1.30×) |
| Director | 0.178 | **0.204** | 0.135 | B ✓ | 0.026 (1.14×) |
| Teleportation | **0.219** | 0.098 | 0.172 | A ✗ | 0.047 (1.27×) |

H4 aggregate improves from 1/3 to 2/3 but still fails Teleportation —
entry 21 (Volt token, K-norm=0.16) pulls Store A ahead of Store C.

### Multi-Head Aggregate Scores (H2+H3+H4+H5)

| Query | Store A | Store B | Store C | Winner | Margin |
|-------|:---:|:---:|:---:|:---:|:---:|
| Zarkov | **0.541** | 0.523 | 0.504 | A ✓ | 0.019 (1.04×) |
| Director | 0.519 | **0.819** | 0.519 | B ✓ | 0.300 (1.58×) |
| Teleportation | 0.553 | 0.502 | **0.620** | C ✓ | 0.067 (1.12×) |

Multi-head works because K-norm bias in H4 doesn't correlate with H2/H3
biases. Content signal reinforces across heads; noise doesn't.

### Attention Budget (N=116)

| Head | Stored % | Query % |
|------|:---:|:---:|
| H4 | 45-52% | 48-54% |
| H5 | 45-50% | 49-54% |
| H2, H3 | 28-43% | 57-71% |
| H0, H1 | 14-24% | 76-86% |
| H6, H7 | 0.1-2.4% | 97-100% |

H6 and H7 contribute almost nothing to stored entries. The effective
routing heads are H2-H5.

## Experiment 4: Apollo N=50 (400 Entries)

50 sampled Apollo transcript windows, 8 K-norm entries each = 400 entries.
5 queries: porridge→W170, baseball→W169, landing→W370, weather→W169, news→W169.

### Results

| Method | porridge | baseball | landing | weather | news | Score |
|--------|:---:|:---:|:---:|:---:|:---:|:---:|
| Entry argmax (H4) | ✗ | ✗ | ✗ | ✗ | ✗ | 0/5 |
| H4 aggregate | ✗ | ✗ | ✗ | ✗ | ✗ | 0/5 |
| Multi-head agg (H2-H5) | ✗ | ✗ | ✗ | ✗ | ✗ | 0/5 |
| All-head agg (H0-H7) | ✗ | ✗ | ✗ | ✗ | ✗ | 0/5 |

### Target Window Ranks

| Method | porridge | baseball | landing | weather | news | Avg |
|--------|:---:|:---:|:---:|:---:|:---:|:---:|
| H4 agg | 27 | 29 | 28 | 32 | 29 | 29.0 |
| Multi agg | 37 | 31 | 23 | 24 | 29 | 28.8 |
| All agg | 23 | 28 | 21 | 30 | 30 | 26.4 |

**Random chance = 25.5.** Average ranks of 26-29 are indistinguishable
from random. The attention mechanism has zero content discrimination
at N=50.

### Why It Fails: W465 Dominates Everything

W465 wins 4/5 queries across all aggregation methods. It's not a
content match — it's a **structural K-norm attractor**.

K-norm sampling selects the 8 highest-norm entries per window. Some
windows have structurally higher K-norms than others (W465, W450, W315).
When you sum 8 high-norm entries, these windows dominate regardless
of query content.

The aggregate doesn't cancel K-norm bias — it amplifies it. Each of
the 8 entries was SELECTED for high K-norm. Summing 8 high-K-norm
entries gives the window with the highest structural norms an
unbeatable advantage.

### Attention Budget at N=400

~75% of attention per head goes to stored entries (up from 45-52% at
N=116). The budget is not the bottleneck — the entries get plenty of
attention. The problem is pure discrimination failure.

## Experiments 5-6: Skipped

N=50 fails so completely (random ranking) that N=725 and detailed
attention budget analysis are moot.

## Why N=3 Works and N=50 Doesn't

At N=3 (116 entries): each store has 32-44 entries. K-norm attractors
exist (entry 21 = Volt, weight 0.16) but are ONE entry out of 40.
The other 39 entries per store provide enough content-dependent signal
to overcome the attractor when aggregating across 4 heads. Margin is
thin (0.019-0.067) but positive.

At N=50 (400 entries): the K-norm sampling strategy selects entries
BECAUSE they have high K-norm. Some windows have structurally higher
K-norms (OCR artifacts, timestamps, formatting). These windows get
high aggregate attention for every query. The content signal (if any)
is buried under structural K-norm variation between windows.

**The fundamental problem:** K-norm sampling creates a store where
entries are selected by K-norm magnitude. Aggregate attention over
K-norm-selected entries is dominated by K-norm magnitude — the very
thing we're trying to average out.

## Conclusion

**Multi-head aggregate routing works at N=3 (3/3) but is random at N=50 (0/5).**

The aggregate hypothesis was partially correct:
- ✓ Multiple heads cancel head-specific K-norm bias (fixes N=3)
- ✗ Aggregation cannot cancel structural K-norm variation between passages
- ✗ K-norm sampling amplifies the problem (selects FOR high K-norm)

**Keywords confirmed as necessary for Apollo-scale routing.** No pure
attention mechanism — entry-level, aggregate, or multi-head — can
discriminate content across 50 heterogeneous transcript windows.

The architecture remains:
```
keyword index → filter to N≤3 candidates → attention aggregate routes → inject
```

The multi-head aggregate is a useful SECOND stage (3/3 at N=3) but
cannot replace the keyword pre-filter.

## Architecture Update

```
Stage 1: Keyword index    — filter 725→3 windows  (string matching)
Stage 2: Multi-head agg   — select 1 of 3         (H2+H3+H4+H5 aggregate)
Stage 3: Entry selection   — best entry in window   (H4 argmax within winner)
Stage 4: Persistent inject — 12 bytes at L30 at 2×  (1D subspace injection)
```

Stage 2 is NEW. It replaces the prior assumption that H4 argmax could
select from N>2 candidates. Multi-head aggregate handles the keyword
index's short-list more robustly than any single-head method.
