# Hidden-Space Routing

**Experiment ID:** 25a66189-dd6f-4a22-b1bd-6ec8fda8c62a
**Model:** google/gemma-3-4b-it
**Date:** 2026-03-18
**Builds on:** addressing_precision_results.md (Exp 1-5)

---

## The Question

The addressing-precision experiment established that entity identity is well-separated
in 2560D hidden space at L29 (Namath vs Marchand: 73.87°, cosine 0.278), but W_K
collapses this to 256D K-space where same-template entities collide (0.88× score ratio —
effectively a coin flip). The fix proposed was an entity string filter. But the deeper
question remained: **can routing in the unprojected 2560D space exploit the geometric
separation that W_K throws away?**

If yes, routing in H-space would resolve entity collisions without string matching,
without language dependence, and without model retraining.

---

## Setup

- 12-fact stress test from addressing-precision: 8 city facts (Zarkov/Nexaris/Aldric/
  Tesserin/Korath/Delvani/Quorbin/Essani) + audio quality + signal strength + Namath + Marchand
- Hidden states extracted at L29 last token position
- F-vectors (facts): model-turn completion stubs (F1–F8, ntok 93–94) + truncated user
  context (F9–F12, ntok 61–83)
- Q-vectors (queries): full context + question + model-turn (Q1–Q12, ntok 99–104)
- Dominant spike dimension: **dim 443**, magnitude 42K–65K across all vectors

---

## EXP A — Raw H-Space Cosine (2560D)

Dim 443 dominates. Its 42K–65K magnitude is shared across all 24 vectors, compressing
every pairwise cosine similarity into a narrow 0.97–0.99 band. Routing becomes noise.

| Metric | Value |
|--------|-------|
| Diagonal (correct pairs) mean cosine | 0.9883 ± 0.0104 |
| Off-diagonal mean cosine | 0.9775 ± 0.0073 |
| Discrimination ratio | **1.011×** |
| Routing accuracy | **8/12 (66.7%)** |

Failures: Q9 (audio), Q10 (signal), Q11 (Namath), Q12 (Marchand) — all route to F1 (Zarkov).

**Acid test — Namath/Marchand:**
- Q11: F11=0.97782, F12=0.97500, ratio=**1.003×** → WRONG
- Q12: F12=0.97552, F11=0.97401, ratio=**1.002×** → WRONG

Raw dot product is worse still (8.3%) — norm differences (42K–67K range) dominate direction.

**Finding:** Raw H-space routing is unusable. The template/positional spike in dim 443
wipes out all entity discrimination. This is a structural property of the residual stream,
not an accident.

---

## EXP B — Mean-Centered H-Space Cosine

Subtract the corpus mean hidden state across all fact vectors before computing cosine.
This removes the shared template component (and eliminates the dim-443 spike).

| Subset | Accuracy |
|--------|----------|
| Entity routing Q1–Q8 | **8/8 (100%)** |
| Non-entity routing Q9–Q12 | 1/4 (25%) |
| Overall | **9/12 (75%)** |

**Acid test — centered:**
- Q11 (Namath): F11=0.427, F12=0.371, ratio=**1.15×** → **CORRECT** ✓
- Q12 (Marchand): F12=0.231, F10=0.252, ratio=**0.92×** → WRONG (routes to F10)

**Finding:** The 73.87° entity separation at L29 is real and usable. After centering
removes the spike, Namath is correctly discriminated. Marchand's failure is not a
geometry problem — it's a format mismatch (see EXP D).

---

## EXP C — PCA Dimensionality Sweep

PCA automatically removes the dim-443 spike: it absorbs into PC1 (template signal),
and entity geometry appears in PCs 2–16.

| Dims | Accuracy | Namath | Marchand | Storage/fact |
|------|----------|--------|----------|--------------|
| 256D via W_K (K-space baseline) | 7/8 cities | WRONG | WRONG | 512 bytes |
| PCA 8D (82.3% variance) | 6/12 (50%) | WRONG | WRONG | 16 bytes |
| **PCA 16D (99.0% variance)** | **9/12 (75%)** | **CORRECT** | WRONG | **32 bytes** |
| PCA 23D (100% variance) | 9/12 (75%) | CORRECT | WRONG | 46 bytes |
| Full 2560D raw | 8/12 (66.7%) | WRONG | WRONG | 5120 bytes |

Minimum dims to resolve Namath: **16** (32 bytes/fact). Marginal returns disappear past
16D — all remaining variance is in the format gap, not entity geometry.

**Critical comparison:** PCA-16 beats W_K-256 for entity routing at 1/32 the storage.
W_K is a worse projection than PCA for routing purposes — it optimises for attention
patterns, not entity discrimination, and it discards the entity-separating dimensions.

---

## EXP D — Format Mismatch: Root Cause of F9–F12 Failures

The F9–F12 failures (audio, signal, Namath, Marchand in the first run) trace back to
an extraction artifact: fact vectors used truncated user-context prompts while query
vectors used full chat-turn prompts with longer context.

| Pair | F format | F ntok | Q ntok | F–Q angle |
|------|----------|--------|--------|-----------|
| F1–F8 / Q1–Q8 | Model-turn completion | 93–94 | 99–101 | **4.6–6.1°** |
| F9–F12 / Q9–Q12 | Truncated user context | 61–83 | 99–104 | **12.1–14.1°** |

The 2–3× larger angular gap for mismatched pairs cannot be bridged by centering.
Replacing truncated F9–F12 with model-turn versions (ntok 95–98) doesn't help —
the gap persists at 13–14° and accuracy drops further to 66.7%. The issue is positional
encoding / context length, not just chat-turn formatting.

**Consequence:** Format matching is a hard prerequisite for H-space routing. Fact and
query vectors must be extracted from prompts of comparable length and structure.
This is not an arbitrary requirement — it follows from positional encoding baking
context length into the hidden state.

---

## Summary Table

| Method | Accuracy | Namath | Marchand | Storage/fact |
|--------|----------|--------|----------|--------------|
| K-space Q·K, fixed threshold | ~50% | WRONG | WRONG | 512 bytes |
| K-space Q·K, adaptive threshold | ~60% | WRONG | WRONG | 512 bytes |
| H-space raw cosine | 66.7% | WRONG | WRONG | 5120 bytes |
| H-space mean-centered cosine | 75.0% | **CORRECT** | WRONG† | 5120 bytes |
| H-space PCA-16 | 75.0% | **CORRECT** | WRONG† | 32 bytes |
| **Entity string filter + adaptive Q·K** | **~85–92%** | **CORRECT** | **CORRECT** | **532 bytes** |

†Marchand failure is a format artifact in this experimental run, not a geometry limit.

---

## What W_K Actually Does

This experiment reframes W_K from pure bottleneck to mixed function:

**What W_K breaks:** Entity discrimination for same-template facts. It collapses Namath
and Marchand from 73.87° apart in H-space to coin-flip territory in K-space.

**What W_K provides:** Implicit template removal. It projects out the dominant positional
spike (dim 443) and exposes entity-specific signal in a compact 256D space. Without this
removal, raw H-space routing fails completely.

The W_K "bug" is specific: it collapses entity-distinguishing directions while removing
the template component. A routing-optimal projection would keep the template removal and
preserve entity separation. That would require retraining W_K with a routing-aware
objective (or replacing it with an entity-preserving projection like PCA-16). At
inference time, W_K is fixed — we can't change its behaviour.

---

## Architecture Implication

H-space routing with centering or PCA achieves the same geometric result as the entity
string filter, but costs more:

| Fix | Storage cost | Requires format matching | Language dependent |
|-----|-------------|--------------------------|-------------------|
| Entity string filter | +12 bytes (name string) | No | Yes (verbatim name) |
| H-space PCA-16 | +32 bytes | **Yes** | No |
| H-space full 2560D | +5120 bytes | **Yes** | No |

For same-template entity collision (the Namath/Marchand problem), the entity string
filter is strictly cheaper and doesn't require format matching. The hybrid architecture
spec in `v_injection_architecture.md` remains the recommended approach.

H-space PCA-16 becomes interesting only if: (a) paraphrased entity queries are needed
("the quarterback" instead of "Joe Namath") where string matching fails, AND (b) format
matching can be guaranteed at extraction time. For the Apollo 11 use case, entity names
appear verbatim in queries, so neither condition holds.

---

## Open Question: Format-Matched Extraction

The one untested case: would H-space PCA-16 with properly format-matched F9–F12 vectors
achieve 12/12 accuracy? The geometry says yes — Marchand's failure was 0.92× (close),
and the angular gap was 12–14° (format-induced). With 4–6° gaps (format-matched), the
centering should resolve it. Validating this would confirm that H-space routing is a
viable fallback for entity-implicit queries where string matching is unavailable.
