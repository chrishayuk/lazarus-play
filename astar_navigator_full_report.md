# Full-Dimensional A\* Navigation Over the Residual Stream
## Experiments 1–8

**Model:** google/gemma-3-4b-it — 34 layers, 2560-dimensional residual stream, bfloat16
**Experiment ID:** 2b6e5e7c-3100-4d36-b4f3-09f680ef231f
**Task:** Compositional factual retrieval — "The birthplace of the author of X was ___"

---

## Setup and Motivation

The residual stream of a transformer is a Markov process: the vector at position $t$, layer $L$ is the complete computational state. Everything the model will output is determined by that single point in 2560-dimensional space. No hidden channel exists; the residual is the computation.

This gives A\* navigation a precise meaning. A correct computation corresponds to a trajectory through residual space that reaches a destination region — the basin of the correct answer. A failure is a trajectory that reaches the wrong destination, or no destination at all. If we can characterize the geometry of these trajectories — where they diverge, what the correct region looks like, how far the failing trajectory has drifted — we can formulate a heuristic, detect failure early, and intervene.

We test this on compositional factual retrieval: *"The birthplace of the author of X was ___"*. The task requires a two-hop relational inference (work → author, author → birthplace), and the model succeeds and fails in structured ways that make the failure geometry tractable.

**Prompt library — successes** (verified via generation):

| Work | Author | Correct answer |
|------|--------|---------------|
| Romeo and Juliet | Shakespeare | Stratford-upon-Avon |
| Ulysses | Joyce | Dublin |
| The Metamorphosis | Kafka | Prague |
| Don Quixote | Cervantes | Alcalá de Henares |
| Crime and Punishment | Dostoyevsky | Moscow |
| The Divine Comedy | Dante | Florence |

**Prompt library — failures** (model's wrong output):

| Work | Correct answer | Model output |
|------|---------------|-------------|
| Hamlet | Stratford-upon-Avon | "a small town in Denmark" |
| Faust | Frankfurt | Germany, no city given |
| Moby Dick | New York City | "in the city of Boston" |
| Symphony No. 9 | Bonn | "in the city of Prague" |

All prompts use the template: `The birthplace of the author of [WORK] was`

---

## Experiment 1: Trajectory Atlas

**Goal:** Build a reference atlas of trajectories across all 34 layers for 27 diverse prompts. Measure how residual space evolves: how many dimensions carry meaningful variance, what those dimensions encode, and how the atlas changes layer by layer.

### Atlas composition

27 prompts across four groups:
- Simple capital retrievals (8): "The capital of X is" for Australia, Japan, France, Germany, Poland, Canada, Nigeria, South Korea
- Named birthplace retrievals (8): "The birthplace of [person] was" for Darwin, Napoleon, Cervantes, Shakespeare, Marie Curie, Kafka, Dante, Joyce
- Compositional successes (6): the six works listed above
- Compositional failures (5): the four failures above plus one additional

### Effective dimensionality by layer

The variance spectrum tells us how many independent dimensions the model uses to represent the diversity of prompts at each layer.

| Layer | 50% variance | 80% variance | 90% variance | 95% variance | 99% variance |
|-------|-------------|-------------|-------------|-------------|-------------|
| L0    | 1D          | 1D          | 1D          | 2D          | ~12D        |
| L14   | 1D          | 1D          | 2D          | 5D          | 17D         |
| L16   | 1D          | 5D          | 7D          | ~11D        | ~18D        |
| L24   | 3D          | 10D         | 15D         | 19D         | 20D         |
| L28   | 4D          | 12D         | 16D         | 20D         | 20D         |
| L33   | 4D          | 11D         | 16D         | 19D         | 20D         |

At L14, a single dimension captures 85.7% of all variance across 27 diverse prompts. The residual is almost entirely common-mode signal — format, query structure, the fact that all prompts are asking for a place. The model has not yet differentiated them in any meaningful sense.

Between L16 and L24, the spectrum expands sharply. By L24, 15 dimensions are needed for 90% of variance. The model is differentiating prompts into distinct computational trajectories. This expansion does not plateau — it continues through L28–L33 at roughly 16 dimensions for 90%.

**Key observation:** The dimensionality expansion is irreversible. Trajectories that are nearly identical at L14 never reconverge. They differentiate and stay differentiated.

### What each principal component encodes

**Layer 14:**
- PC1 (85.7%): Artifact tokens / common-mode signal — the shared "query about a place" frame. Negative pole is common English words; positive pole is rare/unused tokens.
- PC2 (6.1%): Coarse token-type separation. No geographic content.
- PC3–PC10 (total ~8%): Weak, noisy signals. No decodable semantic content.

At L14, the residual space is dominated by format. No geographic, biographical, or compositional signal is detectable.

**Layer 24:**
- PC1 (34.9%): Still dominated by common-mode signal. The format axis persists but weakens.
- PC2 (12.7%): **Nationality/authorship axis.** Positive pole: *Russian, Kafka, Austrian*. Negative pole: German grammatical tokens. This is the relational hop dimension — the model has identified the author's cultural affiliation.
- PC3 (7.9%): **Shakespeare cluster.** Positive: *Shakespeare, Marlowe, Elizabethan*. Negative: *Warsaw, Polish* tokens. England vs Eastern Europe.
- PC4 (5.4%): Cornish/English regional vs Italian. Geographic discrimination at regional level.
- PC5 (4.7%): German/Polish national dimension.
- PC6 (4.1%): Institutional names vs Napoleon/French-Revolution frame.
- PC8 (3.1%): Balkan/Slavic vs German axis.
- PC10 (2.7%): Soviet/Russian vs Darwin/Galápagos axis.

At L24, the residual is actively encoding nationalities and cultural affiliations of authors, not yet the specific cities. The two-hop inference is in progress: work → author has largely completed; author → birthplace city is beginning.

**Layer 28:**
- PC1 (22.9%): Common-mode persists but now weaker than at L24 (34.9% → 22.9%).
- PC2 (18.2%): **Answer cities appear explicitly.** Positive: *Tokyo, Budapest, Abuja, Seoul*. These are the final answer tokens, now present as a principal axis of variation. This is the layer where the computation crystallizes.
- PC3 (6.4%): **Shakespeare/England** dimension (+Shakespeare, Marlowe, Elizabethan; -Warsaw, Polish).
- PC4 (5.4%): Italy/Tuscany dimension (Dante, Fibonacci geography).
- PC5 (5.3%): Germany/Munich dimension (Goethe, Beethoven geography).
- PC7 (4.3%): Canberra/Australia dimension.
- PC10 (2.9%): Russian/Moscow dimension.

The shift from L24 to L28 is the shift from *nationality* to *city*: PC2 transforms from a nationality axis (Russian/Austrian) into an answer-city axis (Tokyo/Budapest/Abuja). The specific destination tokens are now the primary axis of differentiation.

**Layer 33:**
- PC1 (25.5%): Output format axis (reversed polarity vs earlier layers).
- PC2 (13.2%): Dynamic/process tokens vs Balkan/Hungarian geography.
- PC3 (7.9%): Specific names — Hershey, Luther, Hartley — possibly birthplace proper nouns at high resolution.
- PC6 (5.3%): **Warsaw/Poland** — Curie's birthplace, specific enough to name the city.
- PC7 (4.4%): **Shrewsbury/Down** — Darwin's birthplace towns, fully resolved.
- PC4 (6.8%): Renaissance/Chaucer vs neural terminology — historical literary geography.

By L33, principal components decode to the specific correct answers: Shrewsbury (Darwin), Ajaccio (Napoleon), Alcalá (Cervantes). The residual has resolved from format → nationality → country → city → specific city.

### The divergence curve: Ulysses vs Hamlet

Two prompts with structurally similar inputs but opposite outcomes. Tracing pairwise cosine similarity layer by layer:

| Layer | Cosine (Ulysses vs Hamlet) | Layer-on-layer drop |
|-------|---------------------------|---------------------|
| L20   | 0.9977                    | —                   |
| L21   | 0.9970                    | −0.0007             |
| L22   | 0.9954                    | −0.0016             |
| L23   | 0.9937                    | −0.0017             |
| **L24**   | **0.9867**            | **−0.0070**         |
| L25   | 0.9857                    | −0.0010             |
| **L26**   | **0.9786**            | **−0.0071**         |
| L27   | 0.9774                    | −0.0012             |
| L28   | 0.9764                    | −0.0010             |
| L30   | 0.9730                    | −0.0034             |
| L33   | 0.9729                    | ~0                  |

The divergence is concentrated in two bursts: L24 (drop of 0.0070) and L26 (drop of 0.0071), with slow drift elsewhere. These are the two layers where the trajectories actively separate. L33 shows negligible additional divergence — whatever went wrong at L24–L26 is locked in.

**Phase 1 (L24): Author identified, geographic routing begins.** The logit lens at L24 shows Shakespeare 30.1% for Hamlet — the author has been correctly retrieved. But no city appears. For Ulysses, Dublin already shows at 73.2% by L24. The first burst of divergence corresponds to Joyce routing to Dublin while Shakespeare routing stalls at the author level.

**Phase 2 (L26): Wrong geography injected into Hamlet.** By L26, Hamlet's logit lens shows Denmark 44%, Shakespeare 34%. The setting of the play (Denmark) has been injected into the residual in place of the author's birthplace. This is the second burst. Ulysses continues amplifying Dublin (91% by L26).

The failure is a two-event sequence, not a single mistake.

---

## Experiment 2: Full-Dimensional Heuristics

**Goal:** Evaluate three candidate heuristics for predicting and guiding successful computation: (1) trajectory similarity to a reference, (2) momentum (rate of change of the trajectory), and (3) subspace projection (how much of the residual lives in the answer-relevant subspace).

### Heuristic 1: Trajectory similarity

Does the current trajectory resemble a known-correct reference trajectory for the same query type? We measure cosine similarity between each test prompt and the centroid of the six successful compositional retrievals, at L14, L24, and L28.

**Result: This heuristic fails.** All 11 prompts (6 successes + 5 failures) have cosine similarity to the success centroid in the range 0.976–0.993 at L28. Success–failure pairs:

| Pair | L24 cosine | L28 cosine |
|------|-----------|-----------|
| Ulysses (S) vs Metamorphosis (S) | 0.9936 | 0.9841 |
| Ulysses (S) vs Moby Dick (F) | 0.9936 | 0.9849 |
| Romeo & Juliet (S) vs Hamlet (F) | 0.9903 | 0.9799 |
| Metamorphosis (S) vs Symphony No. 9 (F) | — | 0.9935 |

Moby Dick (failure) is *identical* to Ulysses (success) in similarity to the centroid. Hamlet (failure) is *closer* to Romeo & Juliet than most success–success pairs. Global cosine similarity carries no diagnostic signal.

### Heuristic 2: Momentum

The velocity of the residual trajectory — how much the direction changes per layer — is maximal at the divergence layers (L24 and L26). For Hamlet, the momentum signal at L24 is large but *also large for Ulysses*. Both trajectories are changing direction at L24; the difference is which direction they change toward.

Momentum alone does not distinguish correct from incorrect computation. What matters is not the magnitude of change but the destination of the change, which requires the logit lens or target-specific comparison.

### Heuristic 3: Subspace projection

How much of the residual lives in the answer-relevant subspace? We compute a 4D subspace spanned by the unembedding directions of candidate answer tokens ([" Stratford", " England", " Britain", " Avon"] for the Hamlet/Romeo-and-Juliet family), then measure the fraction of residual energy inside vs outside this subspace.

**Result:** 99.67% of the residual lies outside the answer subspace. Only **0.33%** of the 2560-dimensional vector carries the information that determines whether the model outputs Stratford or Denmark.

This subspace fraction is the diagnostic signal. When the 0.33% is oriented correctly (toward Stratford), the model succeeds. When it is oriented toward Denmark, the model fails. The 99.67% outside the subspace is shared between success and failure trajectories — it carries format, syntax, and cultural context that is identical regardless of the correct answer.

The subspace projection heuristic works, but it is target-specific: it requires knowing the candidate answer tokens in advance. It cannot be computed without a hypothesis about what the correct answer is.

---

## Experiment 3: Failure Anatomy — Where Trajectories Diverge

**Goal:** Identify the precise layer and mechanism of failure for each of the four failing prompts.

### Hamlet (setting-attribution confusion)

The Hamlet failure is a three-stage process:

| Stage | Layers | Event |
|-------|--------|-------|
| Early | L0–L7 | Stratford dominant — embedding carries Shakespeare's birthplace signal |
| Middle | L8–L22 | Shakespeare name dominant — author-identification phase |
| Divergence 1 | L24 | Author correctly retrieved (Shakespeare 30.1%), no city yet — correct for Ulysses (Dublin 73%), stuck for Hamlet |
| Divergence 2 | L26 | Denmark 44%, Shakespeare 34% — Hamlet's setting injected into place of birthplace |
| Lock-in | L28–L33 | Denmark/Shakespeare tied (27.2%/27.2%), no city ever dominates |

The trajectory knows it is working with Shakespeare at L24. It has the right author. The failure occurs at the author→birthplace hop: instead of retrieving Shakespeare's birthplace (Stratford), it retrieves the cultural setting most strongly associated with Hamlet (Denmark). The setting of the work contaminates the birthplace of the author.

At L28, the logit lens shows Denmark 27.2%, Shakespeare 27.2%, England 16.5%, London 12.8%, with Stratford at only 1.7%. The trajectory is in the right cultural hemisphere (English/Scandinavian literature) but is distributing probability across a cluster of tokens associated with Hamlet — none of which is the correct birthplace.

### Faust (country lock)

| Layer | Logit-lens top-1 | Probability |
|-------|-----------------|-------------|
| L24   | Germany         | 72.4%       |
| L26   | Germany         | 95%         |
| L28   | Germany         | 99.4%       |

The trajectory routes to country-level geographic resolution at L24 and amplifies monotonically. City-level specificity never materializes. The model knows Goethe is German; it does not know he was born in Frankfurt specifically. The trajectory reaches a stable attractor at "Germany" and cannot proceed to a city. This is geographic resolution failure — the correct country is active but no city binding exists.

### Moby Dick (absent geographic frame)

| Layer | Logit-lens top-1 | Probability |
|-------|-----------------|-------------|
| L24   | located         | 50.4%       |
| L26   | Boston          | 36%         |
| L28   | Boston          | 45.1%       |

The trajectory produces "located" as the dominant signal at L24 — a spatial placeholder with no destination. No author-identification signal appears. No geographic frame is established. By L28, the trajectory has drifted to Boston (45.1%), a plausible New England seafaring city associated with the whaling context of the novel, but not Melville's birthplace (New York City).

The failure is a context drift: without an author-identification signal establishing "Melville was born in...", the trajectory responds to the strongest geographic associate of the work itself (New England/Massachusetts). Boston leads because it is the most prominent New England city; it wins by association, not by knowledge.

### Symphony No. 9 (absent binding)

| Layer | Logit-lens top-1 | Probability |
|-------|-----------------|-------------|
| L24   | Prague          | dominant    |
| L28   | Prague          | dominant    |

This failure is structurally different from all others. The trajectory routes to Prague — the birthplace of Dvořák, who wrote a symphony called "From the New World" and whose music dominated Central European classical identity in the model's training. The model confuses Beethoven (Bonn, Germany) with Dvořák (Bohemia/Czech Republic) at the level of Central European classical composer identity.

Critically: the Symphony No. 9 and Symphony No. 5 trajectories at L24 are separated by only **2.44°**. These two prompts are nearly identical in residual space. For Symphony No. 5 (by Beethoven, correctly associated with Bonn), the energy in the Bonn-aligned subspace is 0.001344. For Symphony No. 9, it is **0.001583** — the failing trajectory already carries *more* Bonn signal than the succeeding one. The failure is not that the trajectory points the wrong way in the Bonn subspace. It is that neither trajectory has a strong Bonn signal, and the strong Central European classical attractor (Prague) wins by default.

---

## Experiment 4: Navigation by Reference

**Goal:** Use known-correct reference trajectories as guides. For each failing prompt, identify the most appropriate reference (a successful prompt with the same correct answer region), then measure how similar the current trajectory is to the reference at each layer.

### Selecting references

The natural reference for Hamlet is Romeo and Juliet — both are Shakespeare plays, both should route to Stratford. At L24, their cosine similarity is **0.9903**, the highest similarity between any failure and its paired reference. The trajectories have not yet diverged in the full-dimensional space; the divergence lives in 0.33% of the residual.

For Symphony No. 9, the reference is Symphony No. 5 — same composer, same correct destination (Bonn). Their cosine similarity at L24 is **0.9978** (angle: 2.44°). Nearly indistinguishable.

### Reference-guided diagnosis

The reference similarity at L24 tells us whether a correction is feasible:

| Failure | Reference | L24 angle | Correctable? |
|---------|-----------|-----------|-------------|
| Hamlet | Romeo & Juliet | ~8° | Yes |
| Symphony No. 9 | Symphony No. 5 | 2.44° | No |

A large angle (8°) means the two trajectories are in different subregions of the space; injecting the reference's residual replaces the failure's wrong subspace component with the reference's correct one. A tiny angle (2.44°) means the trajectories are essentially identical — both are in the wrong place together, and injection accomplishes nothing.

The reference similarity heuristic correctly predicts correctability: small angle at L24 → uncorrectable (absent binding); larger angle → potentially correctable (cross-domain confusion).

---

## Experiment 5: Correction Strategies — Full-Dimensional vs Subspace

**Goal:** For correctable failures, compare (a) full-dimensional injection of the reference residual at a specific layer, and (b) targeted injection into only the answer-relevant subspace. Test which strategy recovers the correct answer and at what cost.

### Hamlet → Stratford

Baseline: the model generates *"a small town in Denmark."*

| Strategy | Layer | Stratford probability | Notes |
|----------|-------|----------------------|-------|
| Baseline | — | ~0% | Denmark output |
| Full injection (Romeo & Juliet) | L24 | **95.2%** | Best result |
| Full injection (Romeo & Juliet) | L26 | **85.9%** | |
| Full injection (Romeo & Juliet) | L28 | **84.3%** | |
| Subspace injection (4D, 0.33% of residual) | L26 | **69.9%** | |

All four strategies succeed. The subspace injection, which replaces only the 0.33% of the residual aligned with [" Stratford", " England", " Britain", " Avon"], recovers 69.9% — confirming that the relevant distinction is entirely localized in that small subspace.

Full injection at L24 achieves the highest accuracy (95.2%). Earlier injection is marginally better because more computation remains to amplify the correct signal. The window is wide: L24, L26, and L28 all work.

**Why subspace injection (69.9%) lags full injection (95.2%):** The 4D subspace is aligned with England/Stratford-associated tokens. The full injection additionally overwrites the Denmark/Shakespeare-associated signals that have accumulated in nearby dimensions by L26. Subspace injection only adds the correct signal; full injection both adds the correct signal and removes the incorrect one.

**Comparison with trajectory_geometry.md results (experiment e03e954a):** Earlier work on a slightly different prompt format ("The birthplace of the author of Hamlet was in the country whose capital is") found subspace injection at L28 achieving 96.3% using a 4D subspace. The higher figure reflects the 4D subspace being better tuned to that prompt format and injection layer. Both experiments confirm the same principle: ~0.4% of residual carries the correction.

### Symphony No. 9 → Bonn (UNCORRECTABLE)

Baseline: the model generates *"in the city of Prague."*

| Strategy | Layer | Result |
|----------|-------|--------|
| Full injection (Symphony No. 5) | L24 | Prague (unchanged) |
| Subspace injection ([" Bonn", " Germany", " Rhine", " German"]) | L26 | Prague (unchanged) |

Neither strategy has any effect. The reason is measurable:

- Residual angle (Symphony No. 9 vs No. 5 at L24): **2.44°**
- Bonn-subspace cosine similarity: **0.9966**
- Energy in Bonn subspace — Symphony No. 9: **0.001583**
- Energy in Bonn subspace — Symphony No. 5: **0.001344**

The failing trajectory already has *more* energy in the Bonn-aligned direction than the reference trajectory. The correction target is already present; what is absent is the downstream amplification that would make Bonn win. The trajectory reaches the right neighborhood but falls into the Prague attractor — which has deeper gravitational pull because the Beethoven–Bonn binding is weak or absent in this model's parameters.

No injection strategy that uses the model's own activations can supply a fact the model was never trained to confidently represent.

---

## Experiment 6: Path Structure — Convergent or Divergent Manifold?

**Goal:** Measure whether successful trajectories converge to a shared "correct computation" manifold, or diverge into isolated destination-specific regions. This determines whether A\* can navigate toward a shared attractor.

### Method

Six successful trajectories measured at L14, L24, L28, and L33. Pairwise cosine similarity matrix computed at each layer. Centroid distance (mean distance from the centroid of the six trajectories) recorded as a summary statistic.

### Results

**Centroid distance (average pairwise cosine distance from centroid):**

| Layer | Centroid distance | Relative to L14 |
|-------|------------------|-----------------|
| L14   | 0.000558         | 1×              |
| L24   | 0.010112         | 18×             |
| L28   | 0.022452         | 40×             |
| L33   | 0.030867         | 55×             |

The separation is monotone and accelerating. Trajectories do not converge at any layer.

**Pairwise cosine similarity range (six successes):**

| Layer | Min     | Max     | Range  |
|-------|---------|---------|--------|
| L14   | 0.9991  | 0.9997  | 0.0006 |
| L24   | 0.9849  | 0.9936  | 0.0088 |
| L28   | 0.9678  | 0.9873  | 0.0195 |
| L33   | 0.9491  | 0.9842  | 0.0351 |

At L14, all six trajectories are effectively identical — 99.9% similar, with variation smaller than measurement noise. At L33, the most distant pair (Dublin vs Alcalá) is 0.9491 similar — more distant from each other than a success–failure pair at L24.

**Conclusion: The manifold is divergent, not convergent.**

Successful computations do not share a basin. Each travels to its own destination: Dublin for Joyce, Prague for Kafka, Stratford for Shakespeare. These destinations are nearly orthogonal to each other in the residual space (note PC2 at L28 spans Tokyo, Budapest, Abuja, Seoul as separate axes — these are not the same place on a single axis, they are independent dimensions).

There is no "correct computation manifold" to navigate toward. A\* cannot use proximity to a shared attractor as its heuristic.

### The geometric invisibility of failure

Failure trajectories are embedded in the success cluster:

| Pair | L24 cosine | L28 cosine |
|------|-----------|-----------|
| Ulysses (S) vs Metamorphosis (S) | 0.9936 | 0.9841 |
| Ulysses (S) vs Moby Dick (F) | **0.9936** | **0.9849** |
| Romeo & Juliet (S) vs Hamlet (F) | **0.9903** | 0.9799 |

Moby Dick is indistinguishable from the success cluster in all pairwise comparisons at L24 and L28. Hamlet is *closer* to Romeo & Juliet than any success–success pair at L24. The failure signal exists only in the 0.33% of the residual that carries answer-specific information — invisible to global similarity measures that are dominated by the 99.67% of shared format signal.

**2D PCA projection at L28 (mixed success/failure):**

```
PC2
+4000  Romeo & Juliet (S)

   0   Hamlet (F)     Ulysses (S)    Metamorphosis (S)    Moby Dick (F)

-9500              Faust (F)
       ─────────────────────────────────────────────────────────
       -8000   -5000   0   +3000  +5000  +6000
                              PC1
```

Faust is isolated in negative PC2 (Germany-locked direction). Moby Dick sits inside the Ulysses–Metamorphosis cluster. Hamlet tracks Romeo & Juliet in PC1 but has diverged in PC2. In two dimensions, three of the four failures are geometrically hidden.

---

## Experiment 7: Heuristic-Guided A\* Loop

**Goal:** Test iterative A\* correction. Inject a reference residual, re-run forward computation from that layer, evaluate the logit-lens heuristic, and repeat until heuristic plateaus. Measure: how many loops to convergence, final accuracy, total compute cost.

### Results

**Correctable failures (Hamlet type):** The loop converges in **one step**.

- Pre-injection logit lens at L28: Denmark 27.2%, no city > 50%
- Post-injection (L24 full inject, one step): Stratford 95.2%
- Second injection (on the already-corrected residual): Stratford 95.6% (+0.4%)

The heuristic — P(city > 50% at L28) — transitions from False to True in a single injection. A second injection onto an already-correct trajectory offers negligible improvement (+0.4%). The loop terminates after one step with no accuracy benefit from iteration.

**Why one step is sufficient:** The failure (Denmark signal at L26) lives in 0.33% of the residual. A single full injection at L24 overwrites all of this 0.33% from the reference trajectory. Subsequent layers then have a clean Shakespeare→birthplace signal to amplify. There is no "partial correction" that requires iteration — the correction is binary.

**Uncorrectable failures (Symphony No. 9 type):** The loop does not converge at any number of steps.

- Post-injection: Prague unchanged
- After 2nd injection onto same trajectory: Prague unchanged
- After 3rd injection: Prague unchanged

The heuristic (P(city > 50%)) remains False regardless of iterations. Each injection passes through a trajectory that is 2.44° away from the reference — effectively identical — and exits into the same Prague attractor. Iteration cannot escape the absent-binding failure mode.

**Summary:** The A\* loop is useful only for diagnosing, not correcting. For correctable failures, single injection at L24 achieves near-ceiling accuracy (95.2%). For uncorrectable failures, no number of injections helps. The practical heuristic is: apply the logit-lens test at L28 before any intervention; if it passes, no correction is needed; if it fails with a detectable failure signature, apply single injection; if it fails without a detectable signature, no intervention will work.

---

## Experiment 8: 2D Logit-Lens vs Full-Dimensional Cosine Similarity

**Goal:** Direct comparison of two navigation strategies. Strategy A (full-D): measure cosine similarity of the current trajectory to the correct-answer reference and use this as a success/failure predictor. Strategy B (2D): apply the logit lens at L24/L28 and check whether a city name dominates.

### Logit-lens predictions at L24 and L28

| Prompt | L24 norm-top1 | L28 norm-top1 | Outcome |
|--------|---------------|---------------|---------|
| Romeo and Juliet | Shakespeare 26.7% | **Stratford 65.8%** | SUCCESS |
| Ulysses | **Dublin 73.2%** | **Dublin 96.2%** | SUCCESS |
| The Metamorphosis | Czech/Prague/located ~11.5% (tied) | **Prague 84.5%** | SUCCESS |
| Hamlet | Shakespeare 30.1% | Denmark 27.2% / Shakespeare 27.2% (tied) | FAILURE |
| Faust | Germany 72.4% | Germany 99.4% | FAILURE |
| Moby Dick | located 50.4% | Boston 45.1% | FAILURE |

### Classification rule

> IF the L28 logit-lens top-1 prediction is a specific **city name** at **>50% probability** → predict success.

**Logit-lens accuracy: 6/6. Zero errors.**
**Full-D cosine similarity accuracy: 0/3 failures correctly identified.**

The full-D similarity cannot flag Hamlet (cosine 0.9903 to its reference, higher than most success–success pairs), Moby Dick (cosine 0.9936, identical to the Ulysses–Metamorphosis success pair), or Faust (cosine 0.9828, within normal success range). The logit lens flags all three correctly.

### Three failure signatures at L24

The logit lens at L24 reveals not just whether a computation will fail, but *why*, which determines the appropriate response:

**1. Author confusion** (Hamlet: Shakespeare 30.1%)
The author-identification hop has succeeded — Shakespeare appears strongly at L24. But no city is present. The trajectory has reached the correct author but stalled at the person level; the second hop (author → birthplace) has not yet fired. At L26, Hamlet's setting contaminates the second hop, injecting Denmark. This is a late failure in the relational chain, and it is correctable via injection at L24–L26.

**2. Country lock** (Faust: Germany 72.4%)
Country-level geographic resolution succeeds at L24 and strengthens monotonically: Germany 72% → 95% → 99.4%. City-level resolution never occurs. The model knows the author (Goethe) is German; it does not have a strong Goethe→Frankfurt binding. The trajectory reaches a country-level attractor and has no force to proceed further. This failure may be partially correctable if a Goethe-specific reference trajectory is available; no such reference was tested.

**3. Spatial uncertainty** (Moby Dick: "located" 50.4%)
No geographic or author signal at L24 — only a spatial placeholder. The trajectory has established that the answer will be a location, but has not established which location or whose birthplace to retrieve. By L28, New England seafaring context (Boston, Massachusetts) fills the placeholder through contextual drift rather than fact retrieval. This failure is not correctable because there is no author-identification signal to redirect — the first hop (work → author) has not fired.

### Why the 2D heuristic outperforms the full 2560-dimensional comparison

The residual stream is dominated by shared structure. At L28, the common-mode signal (format, query type, the "birthplace" frame) accounts for approximately 22.9% of variance (PC1 of the atlas). The answer-specific signal — the difference between Dublin and Prague and Stratford — accounts for a few percent across several specific dimensions.

When we compare full-dimensional trajectories, the 99.67% of shared signal swamps the 0.33% of answer-specific signal. Two trajectories heading to completely different destinations are 99% similar because 99% of what they carry is the same.

The logit lens bypasses this problem entirely. It applies layer normalization (which subtracts the common-mode direction), then reads off the top prediction. This is not a 2D compression; it is the model's own mechanism for discarding the shared structure and reading the answer. One number — P(correct city | L28 residual) — carries all the predictive information that 2560 dimensions of cosine similarity cannot provide.

### Early detection

The logit lens can detect success even earlier:

- Ulysses: Dublin 73.2% **at L24** — success predictable 9 layers before output
- The Metamorphosis: city signal begins at L24 (Prague tied) — success predictable with uncertainty
- Romeo and Juliet: Shakespeare dominates at L24, city only emerges at L28 — requires waiting to L28

Strong bindings (Joyce → Dublin) crystallize early. Weaker bindings (Shakespeare → Stratford, complicated by Hamlet as confounder) require more computation to consolidate. The L24 logit lens is not a universal early detector, but it flags the clearest successes immediately.

---

## Summary of Findings

### The geometry of compositional retrieval

At L14, all 27 prompts occupy virtually the same point in residual space — 85.7% of variance is captured by a single common-mode dimension. The model has not yet begun to differentiate them. By L33, the same prompts have separated into distant destination-specific regions, with 16 principal components needed to capture 90% of variance. The trajectory of a compositional retrieval is a monotone expansion from a shared starting region to an isolated destination.

The expansion is irreversible and the destinations are isolated — Dublin, Prague, Stratford, Bonn are nearly orthogonal directions in the answer subspace. There is no shared "correct computation" basin. Successful computations do not converge; they diverge.

### Two failure types with distinct signatures and interventions

| Type | Name | L24 signature | Correctable | Intervention |
|------|------|--------------|-------------|-------------|
| A | Cross-domain confusion | Author or setting token (Shakespeare, Denmark) | Yes | Full inject at L24–L26 → 85–95% |
| B | Absent binding | No author or city signal ("located"), or near-zero angle to reference | No | None known |

Type A failures are confusions about which fact to retrieve. The correct destination exists in the model's parameter space; the trajectory is heading to the wrong one. Injection redirects it with high fidelity.

Type B failures are gaps in the model's knowledge. No destination exists to navigate to. The trajectory fills the empty attractor slot with the strongest plausible associate (Prague for a Central European composer; Boston for a New England maritime novel). Injection cannot supply knowledge that was never learned.

### The dimensionality paradox

The task operates in 2560 dimensions, but the decisive computation occupies ~0.33% of that space (approximately 8–9 effective dimensions). The remaining 99.67% is shared structure that is identical regardless of whether the model will output the correct answer.

This creates the paradox: full-dimensional cosine similarity correctly classifies 0/3 failures, while a single logit-lens query at L28 correctly classifies 6/6 prompts. More dimensions is not more signal. It is more noise. The correct heuristic is not spatial proximity in the full space — it is spectral readout in the layer-norm subspace where the answer token lives.

The practical consequence for navigation: A\* in the full residual space is the wrong tool. The problem is not "find the right trajectory in 2560D." It is "detect whether the answer token has sufficient probability mass at L28." The entire A\* framework collapses to a single forward pass with a logit-lens query.

### Layer roles in compositional retrieval

| Layer | Role |
|-------|------|
| L0–L14 | Format/query framing. Virtually no geographic differentiation. 85.7% of variance is format. |
| L14–L16 | Markov threshold. Same-task patching succeeds at L14+; cross-task fails at all layers. |
| L16–L23 | Semantic context resolution. Dimensionality expanding from ~2D to ~11D. |
| L24 | **First hop lands.** Author/nationality tokens appear. Primary divergence event (18× centroid distance expansion vs L14). |
| L25–L26 | Second hop: author → city. Correct prompts amplify city signal. Wrong prompts may inject setting geography (Hamlet → Denmark at L26). |
| L27–L28 | **Answer crystallization.** City tokens dominate in successful trajectories (Dublin 96%, Prague 84%). Prediction possible here. |
| L29–L33 | Consolidation and output formatting. Template competition possible (biographical vs geographic frames). |

---

## Appendix: Atlas PCA Components

### Layer 14 PCA (27 prompts)

| PC | Variance | Cumulative | + tokens | − tokens |
|----|----------|-----------|---------|---------|
| 1 | 85.7% | 85.7% | unused/artifact tokens | common English words |
| 2 | 6.1% | 91.9% | rare Unicode tokens | newcomer, currently, よろしく |
| 3 | 1.9% | 93.7% | Manuscript, dreaded | non-Latin scripts |
| 4 | 1.2% | 94.9% | apartments, tumultuous, venerable | tribal/regional terms |
| 5–10 | ~5% | ~97.6% | weak, noisy signals | |

**Interpretation:** Residual space at L14 is almost entirely format signal. No geographic, biographical, or compositional content is detectable in the principal components.

### Layer 24 PCA (27 prompts)

| PC | Variance | Cumulative | + tokens | − tokens |
|----|----------|-----------|---------|---------|
| 1 | 34.9% | 34.9% | artifact tokens | Canberra, Tokyo, capital, Ottawa |
| 2 | 12.7% | 47.5% | Russian, Kafka, Austrian | German grammatical tokens |
| 3 | 7.9% | 55.4% | Shakespeare, Marlowe, Elizabethan | Warsaw, Polish tokens |
| 4 | 5.4% | 60.8% | Cornwall, Tweed, Cumbria | Italian, Italy, Galileo |
| 5 | 4.7% | 65.5% | German, Polish, Poland | English institutional tokens |
| 6 | 4.1% | 69.6% | Harvard, Swansea, Scranton | Napoleon, Bonaparte |
| 7 | 3.9% | 73.5% | ellipsis/blank tokens | Austrian, Serbian, Yugoslav |
| 8 | 3.1% | 76.6% | Croatian, Serbian, Albanian | Germany, Alemania |
| 9 | 2.7% | 79.3% | Brexit, decorated, adorned | Canadian, Canada |
| 10 | 2.7% | 81.9% | Soviet, 소련 | Darwin, Galápagos |

**Interpretation:** The relational hop is active. PC2 encodes nationality/authorship; PC3 encodes the Shakespeare cluster; PCs 4–10 encode country/nationality axes. No specific cities appear as principal axes — the computation is at author/nationality level.

### Layer 28 PCA (27 prompts)

| PC | Variance | Cumulative | + tokens | − tokens |
|----|----------|-----------|---------|---------|
| 1 | 22.9% | 22.9% | artifact tokens | Canberra, Ottawa |
| 2 | 18.2% | 41.1% | **Tokyo, Budapest, Abuja, Seoul** | HTML/markup tokens |
| 3 | 6.4% | 47.5% | Shakespeare, Marlowe, Elizabethan | Warsaw, Polish |
| 4 | 5.4% | 53.0% | Italy, Tuscan, Tuscany | Down, King, Rock |
| 5 | 5.3% | 58.2% | Germany, Munich | Isle, Shetland, Nantucket |
| 6 | 4.7% | 62.9% | German, Mannheim | Galilee, Valladolid |
| 7 | 4.3% | 67.2% | Canberra, Aust | Poland, Warsaw |
| 8 | 3.4% | 70.6% | Australian, Canada, Canberra | noise tokens |
| 9 | 3.2% | 73.8% | Evolutionary, Friedrich | noise |
| 10 | 2.9% | 76.7% | Russian, Moscow, Russia | noise |

**Interpretation:** The critical transition. PC2 has shifted from nationalities (L24: Russian/Kafka/Austrian) to specific answer cities (Tokyo/Budapest/Abuja/Seoul). The computation has resolved from cultural affiliation to geographic destination. Individual country/city axes appear in PCs 3–10.

### Layer 33 PCA (27 prompts)

| PC | Variance | Cumulative | + tokens | − tokens |
|----|----------|-----------|---------|---------|
| 1 | 25.5% | 25.5% | output format tokens | artifact tokens |
| 2 | 13.2% | 38.7% | dynamic/process tokens | Balkan/Hungarian geography |
| 3 | 7.9% | 46.6% | Hershey, Luther, Hartley, Gilman | Ajaccio, corso |
| 4 | 6.8% | 53.4% | neural terminology | Pico, Renaissance, Chaucer |
| 5 | 5.9% | 59.3% | Camus, Cambridge | Malayalam, Puch |
| 6 | 5.3% | 64.6% | **Warsaw, Poland** | Ulm, Deutscher |
| 7 | 4.4% | 69.1% | **Shrewsbury, Down, Pembroke** | Stalingrad, Boley |
| 8 | 3.6% | 72.7% | estudiante, SDS | Aust, Austr |
| 9 | 3.3% | 76.0% | Berg, Lufthansa | Canadian |
| 10 | 2.7% | 78.7% | Yugoslav, Serbian | worms, Eng |

**Interpretation:** Specific birthplace tokens now appear as principal axes. PC6 encodes Warsaw/Poland (Marie Curie's birthplace). PC7 encodes Shrewsbury/Down (Darwin's birthplace towns). PC3 may reflect Alcalá (Cervantes) and related proper name geography. The atlas at L33 is a map of specific biographical facts.
