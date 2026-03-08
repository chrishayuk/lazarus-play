# Dark Amplification: Freeing Space for the Signal at Inference Time

**Experiment:** 735887e4-4420-4a75-bd31-07329a0e6d46
**Model:** google/gemma-3-4b-it
**Prompts:** "The birthplace of the author of Hamlet was" and variants

---

## Hypothesis

The dark entity signal at L7 occupies 8 dimensions of a 2560D residual — 0.3% of the norm. It is
buried under a ~70% format attractor (PC00). The hypothesis: rotate or amplify those dark dimensions
at inference time to let the entity signal survive the viewport. No weight changes. One cheap operation
at L7. The 27 noisy viewport layers then process a residual where the entity signal is loud.

This hypothesis was **largely wrong**. But the investigation revealed why, and a working intervention
that targets the right layer.

---

## Experiment 1 — Signal-to-Noise at L7

### Setup

- PC00 at L7: computed via `residual_atlas` over 40 diverse prompts. Stored as `pc00_l7`.
  PC1 explains 61.9% of variance, norm 5030. Decodes to `<unused>` tokens — pure dark.
- Entity chart at L7: computed via `compute_subspace` over 16 "The birthplace of X was" prompts.

### Critical flaw in the entity chart

Entity_chart_l7 PC1 = **85.8% of variance**. This is not the entity-specific variation — it is the
shared template format signal ("The birthplace of X was" all share the same structure). The entity-
*specific* variation is in PC2-10, which totals only 14.2%.

Computing PCA over same-template prompts gives the template direction as PC1. The planned "8D entity
chart amplification" would have amplified the format attractor, not the dark signal.

### Measured SNR (last token position, L7)

| Quantity | Value |
|----------|-------|
| Residual norm | 4,597 |
| Fraction in PC00 (rank 5) | 92.1% |
| Fraction in entity_chart (rank 10) | 93.7% |
| Entity signal outside PC00 | 7.9% ≈ 363 units |
| SNR (entity / format) | **1:12** |

The predicted SNR was 1:250 (0.3% / 70%). The measured SNR is 1:12 — more favourable. But this
measurement is at the **last token position only**, and the relevant entity signal is distributed
across all positions. The last-position SNR is misleadingly optimistic.

**Residual angle between Shakespeare (direct) and Hamlet 2-hop at L7: 3.49°**
Difference vector magnitude: ~1,129 (on 4,597–5,766 norm vectors).

### Direction angle measurements at L7

All vocabulary token directions are nearly orthogonal to the L7 residual (~88–90°):

| Direction | Angle to Hamlet residual | Dot product |
|-----------|--------------------------|-------------|
| token: Stratford | 89.81° | 14.5 |
| token: Denmark | 88.56° | 116.4 |
| token: Shakespeare | 88.13° | 152.9 |
| token: England | 89.29° | 57.2 |

Even Denmark (the wrong answer) has a barely larger projection than Stratford (the right answer).
The entity signal lives in the dark space, not in the vocabulary direction space. Token-direction
angles are the wrong instrument for measuring it.

### Logit lens — Hamlet 2-hop

| Layer | Top predictions | Notes |
|-------|----------------|-------|
| L0–L7 | wolves (99%+) | Pure dark, no readable signal |
| L14 | famously, supposed | Entity emerging, hedging |
| L22 | the, a, what, not | Confused |
| **L26** | **Denmark 43.8%, Shakespeare 34.0%** | **Stage 3a conflation peak** |
| L30 | known, Denmark, Shakespeare | Regression |
| L33 | a 18.3%, Stratford 5.9% | Final uncertain output |

Shakespeare direct at L26: "debated" 25.8%, "discovered" 9.5% — even the direct prompt is uncertain
at L26. The model knows Shakespeare's birthplace was contested; the two-hop version conflates
Shakespeare's nationality with the Hamlet setting.

---

## Experiment 2 — PC00 Subtraction

Direct PC00 subtraction is not natively supported — `patch_all_positions=True` is incompatible with
`subspace_only=True` (tool constraint). Testing via self-injection:

Injecting the PC00 subspace component from Hamlet into Hamlet (self-injection) gives KL=0.001 —
essentially a no-op, confirming the tool works correctly. The PC00 fraction of the last-position
Hamlet L7 residual is **92.1%**.

No direct PC00 subtraction test was run. The structural constraint (multi-positional signal) makes
last-position PC00 subtraction moot regardless.

---

## Experiments 3 & 4 — Mean-Difference Steering at L7 (Fails)

Three steering vectors computed at L7:

| Vector | Positive | Negative | Norm | Separability |
|--------|----------|----------|------|-------------|
| stratford_l7 | Shakespeare direct (4 variants) | Hamlet 2-hop (4 variants) | 672.7 | 0.0045 |
| entity_delta_l7 | Shakespeare (6 diverse templates) | Goethe (6 matched templates) | 129.6 | **0.0003** |
| entity_delta_l14 | Shakespeare (5 templates) | Goethe (5 templates) | 543.9 | **0.0002** |

**Separability is essentially zero at all layers.** The entity direction is NOT the mean centroid
difference direction. A linear probe at L7 reads entity identity at 99.6% — the classes are linearly
separable — but the probe's hyperplane normal differs from the centroid difference vector. The
mean-difference method finds template/length variation instead.

`steer_and_generate` with stratford_l7:
- α=20: Bengali character artifacts (ী) + Shakespeare repetition — incoherent
- α=50: Pure Bengali artifacts

The vector points toward Bengali text encoding, not entity identity. The 6-token vs 9-token template
difference dominates the mean.

Matching templates (Shakespeare vs Goethe, same structure) reduces the norm to 129.6 but gives
separability 0.0003 — even worse. Entity identity in dark space is genuinely not in the
mean-difference direction.

---

## Experiment 7 — The Divergence Curve

All-position injection (`patch_all_positions=True`) tested at 8 layers to measure how the
Shakespeare and Hamlet residuals diverge:

| Layer | Angle | Donor norm | Recipient norm | KL (donor→injected) |
|-------|-------|------------|----------------|---------------------|
| L7 | 3.49° | 4,597 | 4,597 | **0.0** |
| L10 | **2.48°** | 13,753 | 13,686 | **0.0** |
| L14 | 2.89° | 23,479 | 22,328 | **0.0** |
| L18 | 4.05° | 32,686 | 30,241 | **0.0** |
| L20 | 4.89° | 35,096 | 31,377 | **0.0** |
| L24 | 9.12° | 43,495 | 39,830 | **0.0** |
| L26 | 10.47° | 51,197 | 46,689 | **0.0** |
| L30 | 11.50° | 63,459 | 57,327 | **0.0** |

**Markov holds universally (KL=0.0) at every tested layer.** recipient_injected_kl = 1.707
consistently — the injection is a substantial change, always in the right direction.

### The story the curve tells

```
L7  ──── 3.49° ─── entity identity partially encoded across positions
L10 ──── 2.48° ─── MINIMUM ANGLE: entity fully resolved, both prompts at same Shakespeare state
L14 ──── 2.89° ─── entity compass stable (entity compass experiment confirms 100% probe accuracy)
L18 ──── 4.05° ─── beginning to fork
L24 ──── 9.12° ─── BIG JUMP (+5.1° in 4 layers): L24 Head 1 writes attribution error
L26 ──── 10.47° ── L26 FFN commits to Denmark conflation
L30 ──── 11.50° ── locked
```

The difference vector **shrinks** from L7 (1,129) to L10–L14 (~749): entity identity
*converges* by L10. Both prompts arrive at the same Shakespeare entity state. Then it
**explodes** to 8,885 by L26 as L24 Head 1 writes the attribution error.

### The entity signal is not lost — it is overwritten

The dark signal is correctly computed and available at L10. The Markov state at L10 contains
the right answer. L24 Head 1 then reads the "Hamlet" setting context and writes Denmark,
actively overwriting the correct entity attribution. The fix must target L24–L26, not L7.

### Why single-position amplification fails

The entity encoding at L7 is distributed across all 9 token positions. From prior work:
- All-position injection at L7: KL=0.0 ✓
- Last-position-only injection: KL=1.55 ✗

`steer_and_generate` operates on a single token position at each generation step. This cannot
substitute for an all-position correction of a multi-position encoding. No scalar amplification
of the last-token position can replicate the effect of correcting all 9 positions simultaneously.

---

## Experiment 6 — Endpoint Protection (Fails)

**Test:** inject entity_chart_l7 subspace into L33 (replace L7 entity coordinates at output).

| Metric | Value |
|--------|-------|
| KL (injected vs baseline) | 10.88 |
| Injected top-1 | " a" at 99.4% |

**Why it fails:** L7 residual norm = 4,597. L33 residual norm = 68,088. The network amplifies
residuals ~15× through the depth. Injecting L7-scale coordinates at L33 creates a severely
underscaled subspace component. The output collapses to a near-degenerate distribution.

The template direction is stable (subspace cosine similarity 0.996 from L7 to L33 — the direction
is preserved), but the **magnitude** is completely incompatible. Endpoint protection by L7
coordinate transplant is not viable.

---

## Experiment 5 — Rotation

Not testable with available tools. `patch_all_positions=True` is incompatible with subspace
injection; computing an explicit rotation matrix requires vector arithmetic outside the tool suite.
The multi-positional nature of the entity signal also means last-position rotation would fail for the
same reason as other last-position interventions.

---

## Subspace Structure at L24 and L26

Entity subspaces computed at L24 and L26 from the same 16-prompt entity-varying set:

| Layer | PC1 variance | PC1–PC7 for 80% | Character |
|-------|-------------|-----------------|-----------|
| L7 | **85.8%** | PC1 alone | Template-dominated, useless |
| L24 | 31.0% | ~6 PCs | Genuine multi-dim entity variation |
| L26 | 24.6% | ~7 PCs | Genuine multi-dim entity variation |

At L26 the entity-specific variance is distributed across 10 dimensions with no dominant direction.
This is the right representation for surgical subspace work.

### Subspace injection at L26 (last position only)

| Subspace | KL (donor→injected) | KL (recipient→injected) | Notes |
|----------|---------------------|-------------------------|-------|
| entity_chart_l26 (10D PCA) | 0.691 | 1.231 | 31.7% toward Shakespeare; "located in..." |
| entity_chart_l24 (10D PCA) | 0.455 | 1.286 | 55% toward Shakespeare; "located in..." |
| token embeddings (Stratford/Denmark/etc.) | 0.951 | 0.113 | Near no-op; tokens = 0.5% of L26 residual |

The PCA subspace injections partially move the output toward Shakespeare but do not land on Stratford.
The output shifts to "located in [?]" — the attribution is partially fixed but the entity-specific
location is not recovered. This is because only the last-position subspace component is injected;
the preceding "Hamlet" tokens at earlier positions continue to push Denmark.

Token embedding directions ("Stratford", "Denmark") are only 0.5–0.9% of the L26 residual. The
L26 representation lives in the **unembedding** space (aligned with the output weight matrix W_U),
not the input embedding space. Token-direction subspace injection is the wrong instrument here.

---

## Experiment 3 — L26 Attribution Steering: The Working Fix

### Steering vector

**attribution_fix_l26** computed at L26:
- Positive: Shakespeare, Einstein, Beethoven, Mozart, Darwin, Newton (direct entity prompts)
- Negative: author of Hamlet, author of Relativity, composer of 5th, composer of Don Giovanni,
  author of Origin, author of Principia (indirect "X of Y" prompts)
- Norm: 4,893 (~10% of L26 residual)
- Separability: 0.0046

Separability is still near zero even at L26 with a 10.47° angle between classes. The within-class
variance (0.975–0.984 cosine similarity within each class) dominates the between-class mean
difference. The vector is not a precise discriminative direction, but it has enough structure at
L26 to work at the right α.

### Alpha sweep

| α | Output | Verdict |
|---|--------|---------|
| 1 | "a small town in Denmark..." | No change |
| 3 | "a small town in **England**... Stratford" | Partial: right country, incoherent structure |
| **5** | **"a small, quiet village in the East of England. The village of Stratford-upon-Avon was the birthplace of William Shakespeare."** | **Correct and coherent** |
| 10 | "a humble beginnings. Homer was a simple **b** **b** **b**..." | Wrong entity, incoherent |

**α=5 is the sweet spot.** The steering vector at L26 with α=5 produces a correct, fluent answer.

### Mechanism

`steer_and_generate` applies the vector continuously at L26 during each generation step — not as
a one-time initial injection, but re-applied at every token. At each step the model's L26
computation is nudged from "Hamlet setting context → Denmark" toward "author identity → England".
The continuous pressure accumulates: the first token becomes "a" (describing England), the next
tokens commit to "small, quiet village in the East of England", and finally Stratford-upon-Avon
and William Shakespeare are named explicitly.

This is not dark signal amplification at L7. It is **L26 attribution redirection**.

---

## Experiment 8 — Cross-Entity Generalization

The same vector (attribution_fix_l26, α=5) tested on other multi-hop failures:

| Prompt | Baseline | Steered (α=5) | Verdict |
|--------|----------|---------------|---------|
| Hamlet birthplace | "a small town in Denmark..." | "East of England...Stratford-upon-Avon...Shakespeare" | ✓ **Fixed** |
| Relativity birthplace | "small town in Switzerland...Ulm, Germany" | "...parents in Ulr[m]..." | ~ Partial (heading toward Ulm, garbled) |
| 3-hop Hamlet language | ": Denmark" (then Stratford in continuation) | "**German**...Shakespeare...Stratford-upon-Avon" | ~ Partial (entity correct, language wrong) |
| 5th symphony composer birthplace | "Prague...then Beethoven" | "Zunik, Niemieje, Preroll..." | ✗ Hallucination |
| Crime & Punishment birthplace | "St. Petersburg...Moscow" | "Krupnic...Raskorn..." | ✗ Hallucination |

### Why partial generalization

- **Hamlet:** Was in the training set as negative → positive direction precisely targets this failure. ✓
- **Relativity:** Was in training set. Vector was computed including "author of Relativity" as a
  negative. Partial fix — heading toward Ulm but garbled.
- **3-hop language:** Not in training set. Entity chain attribution is fixed (correctly names
  Shakespeare and Stratford), but the final hop (language of Stratford) gives German instead of
  English. The L26 steer fixes the *entity*, not the *predicate computation*.
- **5th symphony:** "Composer of the fifth symphony" was in the negative training set. At α=5 the
  vector actively pushes away from this prompt's own L26 representation → hallucination.
- **Crime & Punishment:** Not in training set. Dostoevsky/Raskolnikov confusion; the vector drives
  into no-man's-land.

The vector **does not generalise universally** at α=5. It is overfit to its 6 entity pairs.
A per-entity vector at lower α, or a larger training set covering more entities, would be needed
for broad coverage.

---

## Synthesis

### What the data says about the original hypothesis

| Claim | Result |
|-------|--------|
| Entity signal is ~250× quieter than format noise | SNR is ~12× at last position (not 250×), but last position is the wrong slice |
| PC00 is structural scaffolding the viewport needs | PC00 subspace self-injection KL=0.001 — format signal is mostly a bias, but removing it via vector arithmetic is not natively supported |
| Rotating 8D entity chart into dominant direction will help | entity_chart_l7 is the template direction, not the entity direction. The planned rotation would have amplified format noise. |
| Single amplification at L7 fixes viewport corruption | Fails. Entity signal is multi-positional; last-position interventions cannot substitute for all-position context. |
| The entity signal decays through the viewport | The opposite: entity correctly resolves at L10 (minimum angle). L24 Head 1 then actively *writes* the error. |

### The correct picture

1. **L7–L10:** Multi-hop entity identity resolves. Both the direct and indirect prompts arrive at
   the same Shakespeare entity state by L10 (minimum angle 2.48°). The dark signal is working.

2. **L10–L24:** The correctly-resolved entity state is processed. Around L18 the prompts begin to
   diverge as the Hamlet contextual signal begins to dominate the indirect prompt.

3. **L24 (Head 1):** The "contextual attribute bridge" reads the dominant context ("Hamlet" setting)
   and writes Denmark attribution, actively overwriting the correctly-resolved entity state.

4. **L26 FFN:** Commits to the conflation. Stage 3a bottleneck confirmed — logit lens shows Denmark
   43.8%, Shakespeare 34.0% at this layer.

5. **L26 steering at α=5:** Continuously nudges the model away from setting-attribution at L26
   during generation. Works for training entities. Cheap but not universal.

### Comparison table

| Method | Extra compute | Hamlet birthplace accuracy | Notes |
|--------|--------------|---------------------------|-------|
| Standard | None | ✗ "Denmark..." | Self-corrects after first token |
| All-position donor injection (any layer L7–L30) | 1 forward pass | ✓ KL=0.0 | Universal; Markov holds throughout |
| Last-position subspace injection (L24–L26) | 1 forward pass | ~ Partial (KL 0.45–0.69 from donor) | Right direction, insufficient magnitude |
| steer_and_generate L26 α=5 | 1 steering vector | ✓ Correct and coherent | Works for training entities only |
| steer_and_generate L7 mean-diff | 1 steering vector | ✗ Bengali artifacts | Wrong direction |
| Endpoint protection L7→L33 | 1 forward pass | ✗ KL=10.88 | 15× scale mismatch |

### The revised story of multi-hop failure

The dark signal is not the problem. The dark signal is correct at L7 and fully resolved at L10.
The problem is that L24 Head 1 reads the *surface context* ("Hamlet" as a setting token) and
writes attribution based on the setting rather than the author. The viewport does not "corrupt"
a weak signal — it is a specific circuit at L24 that fires on context and overrides entity identity.

The fix is at L26, not L7. Continuous steering at α=5 at the conflation layer redirects this
specific misattribution. It is cheap (one vector, no extra forward pass), but it is not a general
dark signal amplifier — it is a targeted L26 attribution corrector.

---

## Tool Constraints Discovered

- `patch_all_positions=True` is **incompatible with `subspace_only=True`**. All-position subspace
  injection is not supported — only full tensor replacement or last-position subspace injection.
- `steer_and_generate` applies the vector at every generated token (continuous), not as a
  one-time initial perturbation.
- Token embedding directions ("Stratford", "Denmark") are ~0.5% of L26 residual. The L26
  representation is in the unembedding output space, not the input embedding space.
- `compute_steering_vector` separability score is consistently near zero for entity identity tasks
  at all layers (L7, L14, L26). The metric reflects within-class variance dominance, not the
  true discriminability (which probes confirm is 99.6%+ at L7).
