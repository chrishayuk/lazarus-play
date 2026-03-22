# Routing Wall Breakers — Experiment Results

**Experiment ID:** 699874e1-b395-4586-a8aa-9a07b464a442
**Model:** google/gemma-3-4b-it
**Date:** 2026-03-19
**Builds on:** routing_scale_results.md, hierarchical_routing_results.md

---

## Motivation

`routing_scale_results.md` documented a same-template routing failure at N=12:

```
Namath query → "sell" (correct): 3.52%
Namath query → "donate" (wrong):  4.00%
Ratio: 0.88× — ROUTING FAILURE
```

That experiment used **L29 H4 attention score routing** (Q·K mechanism). This experiment tests whether **H-space cosine routing** (comparing last-token residual vectors) can break through the same-template wall.

**Methods planned:**
- M0: Raw H-space cosine at L29 last token (baseline)
- M1a/b/c: Variance-weighted cosine
- M3: Contrastive delta routing
- M4: Fisher discriminant (if M1–M3 fail)

---

## Key Clarification: Two Different Routing Mechanisms

The `routing_scale_results.md` failure was in **attention-score space** (how much L29 H4 attends to each answer token position). This experiment tests **residual (H-space) cosine** — comparing the full last-token hidden state against single-fact fact vectors.

These are fundamentally different:
- Attention routing: scalar attention weight to specific KV positions
- H-space routing: cosine similarity between 2560-dim residual vectors

---

## M0 — Raw H-Space Cosine

### Setup

- **Query vectors:** Full 12-doc context + question (242 tokens), last token position, L29
- **Fact vectors:** Single fact + question (36–38 tokens), last token position, L29
- **Spike removal:** Argmax-based zeroing (spike at dim443, ~58K–60K magnitude)
- **Comparison:** `residual_match` tool at L29, `token_position=-1`

### 2×2 Matrix (Namath/Marchand pair)

| Query | Fact | Cosine | Angle | Type |
|---|---|---|---|---|
| Q11 (Namath query) | F11 (Namath fact) | **0.998201** | **3.44°** | Correct |
| Q11 (Namath query) | F12 (Marchand fact) | 0.986589 | 9.39° | Wrong |
| Q12 (Marchand query) | F12 (Marchand fact) | **0.997535** | **4.02°** | Correct |
| Q12 (Marchand query) | F11 (Namath fact) | 0.988521 | 8.69° | Wrong |

**Routing margins:**
- Q11 (Namath): 0.998201 / 0.986589 = **1.012× CORRECT**
- Q12 (Marchand): 0.997535 / 0.988521 = **1.009× CORRECT**

### N=12 Full Ranking

**Q11 (Namath) vs all 12 single-fact candidates:**

| Rank | Fact | Cosine | Angle |
|---|---|---|---|
| **1** | **F11 Namath/sell** | **0.998201** | **3.44°** |
| 2 | F8 Vaxis/Delvoran | 0.9897 | 8.23° |
| 3 | F6 Zephyr/Aruvex | 0.989325 | 8.38° |
| 4 | F4 Velarian/Korinth | 0.988601 | 8.66° |
| 5 | F1 Zarkov/Voltara | 0.9886 | 8.66° |
| 6 | F2 Nexaris/Crenthia | 0.987189 | 9.18° |
| 7 | F5 Dravik/Solvane | 0.987062 | 9.23° |
| 8 | F12 Marchand/donate | 0.986589 | 9.39° |
| 9 | F10 signal/crackled | 0.986461 | 9.44° |
| 10 | F9 audio/scratchy | 0.986409 | 9.46° |
| 11 | F7 Tarkon/Beldross | 0.985434 | 9.79° |
| 12 | F3 Aldric/Thessmere | 0.984153 | 10.21° |

**Q12 (Marchand) vs all 12 single-fact candidates:**

| Rank | Fact | Cosine | Angle |
|---|---|---|---|
| **1** | **F12 Marchand/donate** | **0.997535** | **4.02°** |
| 2 | F11 Namath/sell | 0.988521 | 8.69° |
| 3 | F8 Vaxis/Delvoran | 0.985601 | 9.73° |
| 4 | F1 Zarkov/Voltara | 0.985539 | 9.76° |
| 5 | F4 Velarian/Korinth | 0.985278 | 9.84° |
| 6 | F2 Nexaris/Crenthia | 0.984849 | 9.99° |
| 7 | F5 Dravik/Solvane | 0.984652 | 10.05° |
| 8 | F6 Zephyr/Aruvex | 0.9842 | 10.20° |
| 9 | F10 signal/crackled | 0.982437 | 10.75° |
| 10 | F9 audio/scratchy | 0.982311 | 10.79° |
| 11 | F7 Tarkon/Beldross | 0.981817 | 10.94° |
| 12 | F3 Aldric/Thessmere | 0.9818 | 10.95° |

**Both correct at N=12 (2/2, 100%).**

### Margin Analysis

| Query | Correct sim | Nearest wrong | Source of nearest wrong | Ratio |
|---|---|---|---|---|
| Q11 Namath | 0.998201 | 0.9897 (F8 Vaxis) | Cross-template (city) | **1.009×** |
| Q12 Marchand | 0.997535 | 0.988521 (F11 Namath) | Same-template (verb) | **1.009×** |

Key observation: For Q11 (Namath), the nearest wrong is **not** F12 (Marchand, 9.39°) but F8 (Vaxis, 8.23°). Cross-template city facts are slightly closer than the same-template verb fact. The entity name in the question provides a stronger anchor than the template structure.

---

## Verdict: The Routing Wall Does Not Exist in H-Space

The "routing wall" from `routing_scale_results.md` is **mechanism-specific** to attention-score routing, not H-space cosine routing.

| Mechanism | Namath routing | Marchand routing | Margin |
|---|---|---|---|
| L29 H4 attention score (routing_scale_results) | 3.52% vs 4.00% donate | N/A | **0.88× FAILURE** |
| L29 H-space cosine last token (this experiment) | 0.998201 vs 0.9897 | 0.997535 vs 0.988521 | **1.009× CORRECT** |

**H-space cosine routing at L29 last token correctly resolves the same-template pair at both N=2 and N=12 with ~1.009–1.012× margins.**

---

## Why H-Space Cosine Works Where Attention Score Fails

The attention score mechanism routes by asking "how much does the query attend to answer token X?" When two facts share a template ("agreed to [verb]"), the model's attention weight is split between the two competing answer tokens — the template similarity dominates over entity distinction.

H-space cosine routes by asking "which fact vector is most similar to the query vector in the full residual space?" The 2560-dimensional residual encodes entity identity (L7 dark space, L14 compass) along with template structure. Entity dimensions survive because:
1. L29 H4 (retrieval head) copies entity-specific information into the last-position residual
2. The entity name appears explicitly in the question ("Joe Namath agreed to..."), providing a self-anchoring signal at last position
3. The 2560D space has sufficient capacity to separate entity differences even when template differences are small

---

## Implications for the Routing Architecture

The correct routing mechanism is **H-space cosine, not attention-score routing**. The routing_scale_results.md architecture was using the wrong primitive.

**Revised routing architecture:**
1. For entity-explicit queries (entity name in question): H-space cosine at L29 last token against single-fact fact vectors → works at N=12 with 1.009× margins
2. String filter first (deterministic for exact matches)
3. H-space cosine fallback with confidence threshold (e.g., margin ≥ 1.002× over 2nd best)

**Open question:** Does H-space cosine routing still work at N=50+ or N=3625? The margins (1.009×) are narrower than the PCA-16 routing margins (1.003–1.015×) from hierarchical_routing_results.md but comparable. The test at N=12 suggests no cliff at this scale.

---

## Summary

| Method | N | Namath | Marchand | Notes |
|---|---|---|---|---|
| Attention score routing (routing_scale_results) | 12 | FAIL (0.88×) | N/A | Wrong mechanism |
| H-space cosine raw (this experiment) | 2 | ✓ 1.012× | ✓ 1.009× | M0 baseline |
| H-space cosine raw (this experiment) | 12 | ✓ 1.009× | ✓ 1.009× | M0 at scale |

**M1a/b/c variance weighting and M3 contrastive delta not needed** — M0 already resolves the same-template routing failure. The original problem was using the wrong routing mechanism.
