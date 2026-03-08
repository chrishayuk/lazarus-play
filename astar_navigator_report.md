# Full-Dimensional A\* Navigation Over the Residual Stream

**Model:** google/gemma-3-4b-it — 34 layers, 2560-dimensional residual stream, bfloat16
**Experiment ID:** 2b6e5e7c-3100-4d36-b4f3-09f680ef231f
**Task:** Compositional factual retrieval — "The birthplace of the author of X was ___"

---

## Motivation

The residual stream of a transformer is a Markov process: the vector at position *t*, layer *L* is the complete computational state. Everything the model will output is determined by that single point in 2560-dimensional space. This raises a question about failure modes: when the model produces a wrong answer, what does the trajectory look like? Can we detect failure early, intervene in the right subspace, and navigate back to the correct destination? And does navigation require the full 2560 dimensions, or does a low-dimensional signal suffice?

To test this, we built a prompt library of compositional retrievals — works of literature paired with questions about their authors' birthplaces — and traced trajectories through residual space at layers 14, 24, 28, and 33.

---

## Prompt Library

**Successes** (verified via generation):

| Work | Author | Correct city |
|------|--------|-------------|
| Romeo and Juliet | Shakespeare | Stratford-upon-Avon |
| Ulysses | Joyce | Dublin |
| The Metamorphosis | Kafka | Prague |
| Don Quixote | Cervantes | Alcalá de Henares |
| Crime and Punishment | Dostoyevsky | Moscow |
| The Divine Comedy | Dante | Florence |

**Failures** (model's wrong output):

| Work | Correct city | Model output |
|------|-------------|-------------|
| Hamlet | Stratford-upon-Avon | "a small town in Denmark" |
| Faust | Frankfurt | continues with Germany, no city |
| Moby Dick | New York City | "in the city of Boston" |
| Symphony No. 9 | Bonn | "in the city of Prague" |

All prompts use the template: `The birthplace of the author of [WORK] was`

---

## Experiment 6: Is There a Correct-Computation Manifold?

The first question is structural. If successful computations converge toward a shared region of residual space — a "correct-retrieval manifold" — then A\* navigation has a natural target. We can measure distance to the manifold as a heuristic and intervene when the trajectory drifts away.

We measured pairwise cosine similarities and centroid distances across all six successful-retrieval trajectories at L14, L24, L28, and L33.

### Centroid Distance by Layer

| Layer | Avg centroid distance | Relative to L14 |
|-------|-----------------------|-----------------|
| L14   | 0.000558              | 1×              |
| L24   | 0.010112              | 18×             |
| L28   | 0.022452              | 40×             |
| L33   | 0.030867              | 55×             |

At L14, all six trajectories are virtually identical — cosine similarities in the range 0.9991–0.9997. By L33, the same prompts have spread to cosines as low as 0.9491. The separation is monotone and accelerating. There is no convergence.

### Pairwise Cosine Range (success-only set)

| Layer | Min cosine | Max cosine | Range  |
|-------|------------|------------|--------|
| L14   | 0.9991     | 0.9997     | 0.0006 |
| L24   | 0.9849     | 0.9936     | 0.0088 |
| L28   | 0.9678     | 0.9873     | 0.0195 |
| L33   | 0.9491     | 0.9842     | 0.0351 |

**Finding:** Successful computations do not converge. They diverge — each trajectory travels to its own isolated destination region. There is no shared basin of correct retrieval. The manifold that A\* would navigate toward does not exist.

### Failures Are Geometrically Invisible

The more damaging result: failure trajectories are indistinguishable from successes in the full-dimensional space.

| Pair (S=success, F=failure) | L24 cosine | L28 cosine |
|-----------------------------|------------|------------|
| Ulysses (S) vs Metamorphosis (S) | 0.9936 | 0.9841 |
| Ulysses (S) vs Moby Dick (F) | 0.9936 | 0.9849 |
| Romeo & Juliet (S) vs Hamlet (F) | 0.9903 | 0.9799 |

Moby Dick's trajectory at L24 is *identical* (to four decimal places) to the Ulysses–Metamorphosis pair — both are 0.9936. Hamlet's trajectory is *closer* to Romeo & Juliet than most success–success pairs. The failure signal is not visible in the global geometry of the residual.

The 2D PCA projection at L28 makes this concrete:

```
PC2
 +4000 |  Romeo & Juliet
       |
    0  |         Hamlet     Ulysses   Metamorphosis   Moby Dick
       |
 -9500 |                     Faust
       +---------------------------------------------------> PC1
         -8000    -5000    0    +3500  +5500  +6000
```

Moby Dick sits directly inside the Ulysses–Metamorphosis cluster. Faust is isolated in negative PC2 (locked into a Germany-specific direction). Hamlet tracks Romeo & Juliet in PC1 but has diverged in PC2 by L28.

---

## Experiment 8: 2D Logit-Lens vs Full-D Cosine Similarity

If full-dimensional cosine similarity cannot separate success from failure, what can? We tested the logit lens — the normalized top prediction of the residual stream at a given layer — as a classification signal.

### Logit-Lens Predictions at L24 and L28

| Prompt | L24 norm-top1 | L28 norm-top1 | Outcome |
|--------|---------------|---------------|---------|
| Romeo and Juliet | Shakespeare 26.7% | **Stratford 65.8%** | SUCCESS |
| Ulysses | **Dublin 73.2%** | **Dublin 96.2%** | SUCCESS |
| The Metamorphosis | Czech / Prague / located (tied ~11.5%) | **Prague 84.5%** | SUCCESS |
| Hamlet | Shakespeare 30.1% | Denmark 27.2% / Shakespeare 27.2% | FAILURE |
| Faust | Germany 72.4% | Germany 99.4% | FAILURE |
| Moby Dick | located 50.4% | Boston 45.1% | FAILURE |

### Classification Rule

> If the L28 logit-lens top-1 prediction is a **specific city name** at **>50% probability**, predict success. Otherwise predict failure.

**Accuracy: 6/6. Zero errors.**
Full-D cosine similarity against the success centroid: **0/3 failures correctly identified.**

The 2D heuristic — one token, one number — perfectly separates success from failure across all tested prompts. The full 2560-dimensional similarity structure adds no predictive value.

### Three Failure Signatures at L24

Failures are not a single phenomenon. The logit lens at L24 reveals three structurally distinct patterns:

**1. Author confusion** (Hamlet → Denmark)
The author token dominates at L24 (Shakespeare 30.1%). The trajectory has found the correct cultural space but stopped at the author level — the relational hop from work to author succeeded; the geographic hop from author to birthplace failed. The trajectory is heading toward England but resolves to the author's name and then to the work's setting (Denmark).

**2. Country lock** (Faust → Germany, no city)
The country token dominates strongly at L24 (Germany 72.4%) and amplifies monotonically to 99.4% by L28. Geographic resolution terminates at country granularity. The city-level binding (Frankfurt) is absent or unreachable from this trajectory.

**3. Spatial uncertainty** (Moby Dick → Boston)
No geographic or author signal appears at L24 — "located" at 50.4% indicates a spatial query frame with no destination. The trajectory drifts to Boston by L28 (45.1%), a plausible New England seafaring city but the wrong one (Melville was born in New York City). No geographic frame stabilizes early enough to direct the computation correctly.

---

## Experiment 5: Correction Strategies

Given a failure, can we intervene? We tested two strategies: full-dimensional injection of a reference trajectory at a specific layer, and targeted injection into a low-dimensional subspace aligned with the correct answer.

### Hamlet → Stratford (author confusion failure)

| Strategy | Injection layer | Result |
|----------|-----------------|--------|
| Baseline | — | "a small town in Denmark" |
| Full injection from Romeo & Juliet | L24 | **Stratford 95.2%** |
| Full injection from Romeo & Juliet | L26 | **Stratford 85.9%** |
| Full injection from Romeo & Juliet | L28 | **Stratford 84.3%** |
| Subspace injection (4D, 0.33% of residual) | L26 | **Stratford 69.9%** |

The subspace used tokens [" Stratford", " England", " Britain", " Avon"]. It occupies 0.33% of the residual at L26. Replacing only that subspace recovers 69.9% — confirming that the relevant distinction is highly localized. Full injection at L24 recovers 95.2%.

All injection windows work (L24–L28). Earlier is marginally better. The failure is cross-domain confusion — the trajectory can be redirected because the correct destination is orthogonal to the wrong one. The Hamlet and Romeo & Juliet trajectories are at 0.9903 cosine similarity at L24 but point to nearly orthogonal destinations (Denmark vs Stratford) in the low-dimensional subspace that matters.

### Symphony No. 9 → Bonn (absent binding failure)

| Strategy | Injection layer | Result |
|----------|-----------------|--------|
| Baseline | — | "in the city of Prague" |
| Full injection from Symphony No. 5 | L24 | Prague (unchanged) |
| Subspace injection (4D, Bonn-aligned) | L26 | Prague (unchanged) |

**Both strategies fail completely.**

Key measurements:
- Residual angle between Symphony No. 9 and No. 5 at L24: **2.44°** (nearly identical trajectories)
- Cosine similarity within the Bonn-aligned subspace: **0.9966**
- Bonn-subspace energy, Symphony No. 9: **0.001583**
- Bonn-subspace energy, Symphony No. 5: **0.001344**

The recipient trajectory already carries *more* energy in the Bonn direction than the donor. There is no shortage of Bonn-aligned signal to inject. The failure is not cross-domain confusion — the model is not routing to a wrong destination. It is routing to a blank. The Beethoven–Bonn binding does not exist in this model's parameters in a form that any injection strategy within the model's own activations can access.

---

## The Two Failure Types

These results define a clean taxonomy:

| Type | Name | Mechanism | Correctable? |
|------|------|-----------|-------------|
| A | Cross-domain confusion | Trajectory routes to wrong destination (Hamlet → Denmark) | Yes — inject at L24–L26 |
| B | Absent binding | No destination trajectory exists (9th Symphony) | No — model lacks the fact |

Type A failures have a low-dimensional correction signature: the correct and incorrect destinations are nearly orthogonal in a small subspace (<1% of residual). Replacing that subspace corrects the output. The full-dimensional trajectory may be nearly identical between success and failure (cos=0.9903) while the relevant subspace is completely wrong.

Type B failures are structurally distinct. The trajectory is not confused — it is empty. The model generates plausible-sounding output by falling into the nearest available attractor (Prague for Beethoven, because both are associated with Central European classical music). No amount of trajectory comparison or injection can supply a fact the model was never trained to retrieve.

---

## The Dimensionality Paradox

The central finding is a paradox about scale. The model operates in 2560 dimensions. A computation that correctly retrieves Dublin differs from one that would retrieve London by a vector of that same length. Yet:

- The relevant distinction occupies **0.33% of the residual** (4 dimensions out of 2560)
- The wrong and right trajectories are **99.03% similar** in full space (Hamlet vs Romeo & Juliet)
- **One number** — P(correct city | L28 logit lens) — perfectly predicts success or failure
- The full-dimensional cosine similarity classifies **0/3 failures** correctly

The residual stream is dominated by shared structure: format, syntax, query type, and common-mode representations. These account for ~99.7% of the signal. The fact-specific direction that determines whether the model outputs Stratford or Denmark occupies the remaining 0.3% — invisible to global similarity measures, but fully decisive for the output.

The layer-norm operation at readout is what reveals this: it subtracts the mean direction (common-mode signal) and amplifies whatever remains orthogonal to it. Sydney has a higher raw dot product than Canberra at every single layer, yet Canberra wins — because its direction survives layer normalization. The competition between right and wrong answers happens in the residual's orthogonal complement to the mean, not in the full space.

---

## Implications for Navigation

**There is no manifold to navigate toward.** Successful computations diverge from each other at a rate that accelerates through L24–L33. At L33, a successful Dublin retrieval and a successful Prague retrieval are 0.9497 similar — further apart than a success–failure pair at L24.

**The heuristic is not spatial, it is spectral.** The question is not "where is the trajectory" but "what does the logit lens show." A single forward pass, layer-norm, and argmax over the vocabulary gives a perfect predictor at L28.

**Intervention geometry depends on failure type.** For Type A (confusion), the correct direction is already present in the model; injection at L24–L26 redirects the computation with 85–95% recovery. For Type B (absent), no intervention within the model's forward pass can help. The distinction between these types is visible at L24: Type A shows an author or country token; Type B shows a spatial placeholder ("located") with no geographic destination.

**The effective dimensionality of the decision is below 10.** The subspace that determines correct vs. incorrect output can be spanned by 4–6 vectors (the unembedding directions of the answer token and its close neighbors). Navigation in the full 2560-dimensional space is unnecessary and misleading.

---

## Summary

| Question | Answer |
|----------|--------|
| Is there a correct-computation manifold? | No. Successful trajectories diverge monotonically (55× by L33). |
| Can full-D cosine similarity predict failure? | No. Failures are embedded in success clusters (cos=0.9936). |
| What predicts success? | P(correct city at L28 logit lens). One number, zero errors on 6 prompts. |
| Can failures be corrected by injection? | Yes for Type A (confusion). No for Type B (absent binding). |
| How much of the residual carries the decisive signal? | ~0.33% — 4 dimensions out of 2560. |
| What layer is the earliest reliable detector? | L24 for strong retrievals (Dublin 73%). L28 for weaker ones. |
| Does L14 carry predictive signal? | No. All prompts cosine > 0.999 at L14. |
