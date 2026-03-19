# Hierarchical Routing — Experiment Results

**Experiment ID:** 8a42e949-eea1-4154-a904-703f8d59eecd
**Model:** google/gemma-3-4b-it
**Date:** 2026-03-19
**Builds on:** format_matched_pca16_results.md, hidden_space_routing.md, addressing_precision_results.md

---

## Motivation

Previous experiments established that:
1. H-space PCA-16 routing works (9/12) with full-doc context but fails with bare queries (format gap ~13–16°)
2. The format gap is content-driven: standardised single-fact templates cannot bridge it
3. W_K collapses entity-separating dimensions (Namath/Marchand: 73.87° in H → coin-flip in K)

**Hypothesis (this experiment):** A two-level geospatial routing architecture might survive the format gap:
- Level 1 (K-space Q·K): coarse cluster assignment by template type
- Level 2 (per-template PCA): fine entity discrimination within cluster

The per-template PCA hypothesis: within a homogeneous cluster (all "city" questions or all "verb" questions), the dominant content variation is removed, leaving format-invariant entity dimensions.

---

## Experiment 3 — Format Gap Survival Test (KEY TEST)

### Setup

- 4 bare queries (no document context): Q1 (Zarkov), Q2 (Nexaris), Q11 (Namath), Q12 (Marchand)
- 4 full-doc candidate fact vectors: F1 (Zarkov), F2 (Nexaris), F11 (Namath), F12 (Marchand)
- Method: `residual_match` at L29, raw cosine, last token position
- Each bare query ranked against all 4 full-doc candidates

### Results

| Query (bare, ~17-19 tok) | Correct candidate | Correct sim | Angle | Best wrong sim | Ratio | Status |
|---|---|---|---|---|---|---|
| Q1 "...Zarkov Industries founded in?" | F1 (Zarkov) | 0.9906 | 7.85° | 0.9858 (F11/Namath) | **1.005×** | ✓ CORRECT |
| Q2 "...Nexaris Corporation founded in?" | F2 (Nexaris) | 0.9918 | 7.33° | 0.9856 (F1/Zarkov) | **1.006×** | ✓ CORRECT |
| Q11 "...Joe Namath agree to do?" | F11 (Namath) | 0.9841 | 10.25° | 0.9790 (F1/Zarkov) | **1.005×** | ✓ CORRECT |
| Q12 "...Sylvia Marchand agree to do?" | F12 (Marchand) | 0.9923 | 7.11° | 0.9872 (F11/Namath) | **1.005×** | ✓ CORRECT |

**4/4 correct in raw H-space cosine with bare queries vs full-doc fact vectors.**

### Unexpected Finding: Entity Name in Query Bridges the Format Gap

This result appears to contradict the format_matched_pca16 conclusion that bare-query routing fails.
The key difference is WHERE entity information appears:

**Previous failures (F9–F12 in hidden_space_routing.md):**
- F9/F10: "audio quality", "signal quality" — NO entity name in query. Bare question:
  "What was the audio quality like?" — entity-implicit. No anchor.
- F11/F12: Those FACT vectors were truncated (61–83 tok) vs query full-context (99–104 tok).
  The content gap was between truncated doc and full doc, not bare vs full.

**This experiment's queries:**
- All four queries contain the entity name explicitly in the question itself:
  "What city was **Zarkov Industries** founded in?" (17 tokens)
  "What did **Joe Namath** agree to do?" (17 tokens)
- The entity name at the last token position provides sufficient L29 routing signal
- The full document in the fact prompt is irrelevant to what L29 last-token encodes
  (L29 H4 already attended to the entity name tokens in the question)

**Why it works:** L29 H4 is a retrieval head that copies entity information into the last-position residual. When the entity name appears in the QUERY ITSELF, L29 encodes it regardless of whether the preceding document is present. Bare query with entity name ≈ full-doc query with entity name at last-position L29.

### Margin Analysis

Routing margins are consistently ~1.005× (narrow but positive):

| Query pair | Within-cluster separation | Cross-cluster separation |
|---|---|---|
| Q1 (Zarkov) best wrong: F11 Namath | 0.9906 vs 0.9858 → **1.005×** | Cross-cluster distance ~10° |
| Q2 (Nexaris) best wrong: F1 Zarkov | 0.9918 vs 0.9856 → **1.006×** | City-city wrong is further than city-verb |
| Q11 (Namath) best wrong: F1 Zarkov | 0.9841 vs 0.9790 → **1.005×** | Verb-city distance slightly larger |
| Q12 (Marchand) best wrong: F11 Namath | 0.9923 vs 0.9872 → **1.005×** | Same-template pair: 9.17° apart |

**The same-template pair (Namath/Marchand, Q11/Q12) gets 1.005× discrimination** — matching the
full-doc result from format_matched_pca16_results.md (1.009× for full-doc Namath/Marchand).
Bare-query routing for entity-explicit queries is nearly as good as full-doc routing.

**Cross-cluster note:** Q11 (Namath/verb) routes wrong to F1 (Zarkov/city) rather than F12 (Marchand/verb).
This means within-cluster discrimination does NOT fail here — the entity name signal overrides template signal.

### N=12 Verification

Tested Q1, Q11, Q12 against all 12 full-doc candidates:

| Query | Correct sim | Correct rank | Best wrong sim | Best wrong fact | Ratio | Status |
|---|---|---|---|---|---|---|
| Q1 (Zarkov) | 0.9906 (7.85°) | **1/12** | 0.9877 (8.98°) | F5 Dravik | **1.003×** | ✓ CORRECT |
| Q11 (Namath) | 0.9841 (10.25°) | **1/12** | 0.9790 (11.76°) | F1 Zarkov | **1.005×** | ✓ CORRECT |
| Q12 (Marchand) | 0.9923 (7.11°) | **1/12** | 0.9872 (9.17°) | F11 Namath | **1.005×** | ✓ CORRECT |

**3/3 correct at N=12.** Margins hold.

**City cluster effect on Q1:** With 8 city candidates instead of 4 mixed-type, the nearest wrong shifts from
F11/Namath (9.67°) to F5/Dravik (8.98°) — tightening from 1.005× to 1.003×. The 8 same-template city
facts cluster ~0.9877–0.9813 around Q1, but all below 0.9906. Entity names in the query prevent
within-cluster collapse.

**Verb cluster stability:** Q11 and Q12 margins are unchanged from N=4 — the same-template pair
(Namath/Marchand) remains the hardest discrimination at 1.005×, and adding 10 more candidates
didn't introduce anything closer than F1/Zarkov as the nearest wrong.

---

## Reframing the Format Gap Problem

The previous three experiments established a taxonomy of format gap severity:

| Scenario | F-Q angle | Entity signal | Routing |
|---|---|---|---|
| Same-doc, same question structure | 0–6° | Entity in doc context | ✓ Works |
| Bare query (entity in question) vs full-doc fact | **7–12°** | Entity in question tokens | **✓ Works (this experiment)** |
| Bare query (entity-implicit) vs full-doc fact | ~14° | No entity anchor | ✗ Fails |
| Bare query vs truncated-context fact | ~14° | Entity in different context depth | ✗ Fails |

**Revised conclusion:** The format gap is only fatal when the query lacks an entity anchor.
Entity-explicit queries (where the entity name appears in the question) are self-anchoring:
L29 last-position encodes the entity name regardless of preceding context length.

This is why the original routing_scale_results experiment worked for "Zarkov" queries but failed
for "Namath agreed to sell" — the Zarkov answer token ("Voltara") is distinctive, but more
importantly, the routing mechanism was different (attention score, not H-space cosine).

---

## Architecture Implications

### What This Means for the Hierarchical Routing Spec

The original motivation for hierarchical routing was to survive the format gap via per-template PCA.
The format gap survival test shows that for entity-explicit queries, the format gap is not the
bottleneck — raw H-space routing already works at 4/4.

**Revised architecture question:** Is hierarchical routing needed at all for entity-explicit queries?

| Query type | Bare routing (raw H-space) | Current production approach |
|---|---|---|
| Entity-explicit (entity name in question) | **Works at ~1.005× margin** | String filter (reliable) |
| Entity-implicit (no entity name) | Fails (~14° format gap) | Adaptive K-space Q·K |

String filter handles entity-explicit better (deterministic, no margin risk).
Raw H-space handles entity-explicit surprisingly well but with narrow 1.005× margins.

### When Hierarchical Routing Would Add Value

Per-template PCA (Level 2) would strengthen margins from 1.005× to higher ratios for entity-explicit
queries — buying robustness at scale (N=50+). The concern is that 1.005× margins may not survive
when 12 city facts are all competing rather than 4 mixed-type candidates.

**N=12 result:** Margins hold (1.003–1.005×). String filter + H-space fallback is sufficient for
entity-explicit queries at N=12 scale without hierarchical PCA.

---

## Experiment 3 Conclusion

**The format gap hypothesis was partially wrong.** The gap is fatal ONLY for entity-implicit queries.
For entity-explicit queries (entity name in the question), L29 raw H-space cosine correctly routes
bare queries to full-doc fact vectors at 4/4 (N=4) and 3/3 (N=12) with 1.003–1.005× margins.

**Per-template PCA is not needed for entity-explicit queries.** The string filter handles these
deterministically. H-space cosine routing is a viable fallback when string matching misses, but
the narrow margins (1.003×) at N=12 suggest it should be used with a confidence threshold.

---

## Next Steps

1. **N=12 test**: Test Q1/Q2/Q11/Q12 bare queries against full 12-candidate set to check margin survival
2. **Entity-implicit routing**: Test Q9/Q10 (audio/signal quality, no entity name) — expect failure
3. **Experiment 1** (K-space clustering): Verify city/verb/tech clusters are separable in K-space
4. **Experiment 2** (Per-template PCA): Complete dimensionality table for city cluster
5. **Save results** to experiment 8a42e949

---

## Summary Table

| Method | Query type | N candidates | Accuracy | F-Q angle | Margins |
|---|---|---|---|---|---|
| H-space raw cosine | Entity-explicit bare | 4 | **4/4 (100%)** | 7–12° | 1.005–1.006× |
| H-space raw cosine | Entity-explicit bare | **12** | **3/3 (100%)** | 7–12° | **1.003–1.005×** |
| H-space PCA-16, full-doc context | Any (full-doc) | 12 | 9/12 (75%) | 0° | 1.001–1.015× |
| Entity string filter + K-space Q·K | Entity-explicit | 12 | ~85–92% | N/A | N/A |
| H-space PCA-16, bare query (previous) | Entity-implicit | 12 | ~0% | 13–16° | <1.0× |
