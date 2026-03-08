# Trajectory Geometry: The Residual Stream as a Dynamical System

*Experiment ID: e03e954a-3c8e-40de-849e-49bf1faed051*
*Model: google/gemma-3-4b-it (34 layers, 2560 hidden dim)*

---

## Overview

We treat the residual stream as a point moving through 2560-dimensional space. Each layer
applies a force — attention and FFN — that displaces the point. The trajectory through depth
IS the computation. The answer is wherever the point lands at layer 33.

We ask: where are the basins of attraction? Where is the boundary? What forces deflect the
trajectory? And what is the minimal correction needed to keep it in the right basin?

The short answer: **the basins aren't where you think they are, the boundary doesn't exist
where you'd look for it, and the minimal correction is shockingly small — 0.33% of the vector.**

---

## Prompts and Baselines

Three compositional failures and two simple successes:

| Prompt | Expected | Top-1 | Prob |
|--------|----------|-------|------|
| "The birthplace of Beethoven was in the country whose capital is" | Berlin | Berlin | 36.9% |
| "The birthplace of Einstein was in the country whose capital is" | Berlin | Berlin | 29.5% |
| "The birthplace of the author of Hamlet was in the country whose capital is" | London | Copenhagen | 26.6% |
| "The capital of the country that borders France to the north is" | Brussels | Paris | 26.0% |
| "The capital of Germany is" | Berlin | Berlin | **89.1%** |

Beethoven and Hamlet technically get the right top-1 token (Berlin/Copenhagen), but the
distribution is dangerously diffuse. France-north genuinely fails. Germany is the clean baseline.

---

## Experiment 1: The Trajectories

We run `residual_trajectory` for all prompts, measuring the angle between the last-position
residual and a set of reference tokens at every layer.

### Beethoven vs Germany: where do they diverge?

Cosine similarity between Beethoven and Germany last-position residuals:

| Layer | Cosine |
|-------|--------|
| L0 | 0.997 |
| L24 | 0.988 |
| L26 | 0.985 |
| L28 | 0.985 |
| L33 | 0.977 |

**The divergence is gradual and monotonic from L24 onward.** There is no sudden flip at L31.
The biographical forces act continuously across 10 layers, not as a single discrete event.

### Beethoven's reference-token trajectory (dominant token in reference set)

```
L0-L3:   Berlin     (embedding similarity)
L4-L10:  Germany    (compositional structure processing)
L11-L33: Vienna     (biographical context dominates reference set)
```

**Vienna is the dominant reference token from L11 all the way to L33.** Yet Berlin wins the
greedy prediction at L33 (36.9%). This apparent paradox is the first major finding.

### Germany's trajectory

```
L0-L3:   Berlin     (embedding similarity)
L4-L23:  Vienna     (generic "European city" detour)
L24-L25: Germany/Vienna
L26-L33: Berlin     (capital fact store activates, stays)
```

Germany also passes through a "Vienna-dominant" phase (L4-L23), but L26 FFN permanently
flips it to Berlin. Beethoven never gets that flip.

### Einstein's trajectory

Einstein shows **Vienna-dominant from L18 onward** in reference-token space — structurally
identical to Beethoven. Yet Einstein's logit lens tells a completely different story.

---

## Experiment 2: The Logit Lens

What does the model "think" the answer is at each layer, if computation stopped there?

### Beethoven logit lens (normalized top predictions)

| Layer | #1 | #2 | #3 |
|-------|----|----|-----|
| L24 | Berlin 43% | London 18% | Moscow 7% |
| L26 | **Moscow 36%** | Berlin 28% | Vienna 9% |
| L28 | Berlin 45% | Moscow 24% | Vienna 9% |
| L31 | Berlin 41% | **Vienna 25%** | Rome 7% |
| L33 | Berlin 37% | **Bonn 20%** | Vienna 15% |

Key events:
- **L24**: Compositional circuit fires correctly — Berlin 43%.
- **L26**: Capital-query frame adds Moscow as a red herring (generic capital of large country).
- **L28**: Berlin peaks at 45%. This is the correct answer's best moment.
- **L31**: Vienna surges from 9% to 25% — biographical override active.
- **L33**: **Bonn emerges at 20%** — the literal birthplace fact. Three-way dissolution.

### Germany simple logit lens

| Layer | #1 |
|-------|----|
| L24 | Berlin 83% |
| L26 | Berlin **99.9%** |
| L28 | Berlin **99.9%** |
| L31 | Berlin 99.4% |
| L33 | Berlin 88.7% |

Germany achieves near-certainty by L26 and holds it. The contrast with Beethoven is stark.

### Einstein logit lens

| Layer | #1 | #2 | #3 |
|-------|----|----|-----|
| L24 | **London** 17% | Berlin 13% | Paris 10% |
| L26 | Canberra 13% | Moscow 13% | Berlin 12% |
| L28 | **Moscow** 15% | Tokyo/Beijing/Berlin ~11% | |
| L31 | **Rome** 22% | Paris 12% | Berlin 8% |
| L33 | Berlin 30% | in 7% | Rome 7% |

Einstein "succeeds" (Berlin wins) but never with a coherent signal. London leads at L24,
Moscow at L28, Rome at L31. Berlin wins at L33 by attrition over incoherent scatter.
**This is not a confident success — it is an uncertain non-failure.**

### The three-way dissolution

Beethoven's failure is not a binary Berlin→Vienna flip. It is a three-way pull:
1. **Berlin** (correct: capital of country of birth)
2. **Bonn** (biographical: actual city of birth)
3. **Vienna** (biographical: city of life and work)

Each of these facts activates a different memory circuit. They compete and jointly erode
the correct answer. The model has too much accurate biographical knowledge.

---

## Experiment 3: The Raw Angle Paradox

We use `token_space` to measure the angle between the residual and the actual prediction
tokens (space-prefixed: " Berlin", " Vienna", " Bonn") at each layer.

### Beethoven residual angles to answer tokens

| Layer | ∠ Berlin | ∠ Vienna | ∠ Bonn | Nearest |
|-------|----------|----------|--------|---------|
| L28 | 88.45° | **86.18°** | 87.36° | Vienna |
| L29 | 88.63° | **86.36°** | 87.37° | Vienna |
| L30 | 88.48° | **86.17°** | 87.19° | Vienna |
| L31 | 89.11° | **86.70°** | 87.23° | Vienna |
| L32 | 89.52° | **87.10°** | 87.48° | Vienna |
| L33 | 89.49° | 87.11° | **86.89°** | Bonn |

**Vienna is geometrically closer than Berlin to Beethoven's residual at every single layer.**
Berlin is always the farthest. No basin boundary crossing occurs in raw angle space.

### Germany simple at L28 (for comparison)

| Token | ∠ to residual |
|-------|---------------|
| Berlin | 87.76° |
| Vienna | **86.84°** |
| Bonn | 87.43° |
| Nearest | **Vienna** |

Germany *also* has Vienna as the geometrically nearest token in raw space at L28. Yet it
outputs Berlin 99.9%.

### The resolution: layer norm as common-mode rejection

The **pairwise angles between tokens are fixed** — they depend only on the unembedding matrix
weights. Berlin↔Vienna is always 63.1°. Berlin↔Bonn is always 71.9°. What changes is where
the residual sits relative to all of them.

After layer norm:
1. The **mean direction** is subtracted (common-mode rejection)
2. The residual is rescaled
3. Logit scores are computed in the normalized space

The mean direction at these layers points toward the "generic famous European city associated
with Beethoven" cluster. Vienna and Bonn are correlated with this mean; Berlin is more
orthogonal to it. After subtraction, Berlin's relative component survives better.

This is identical to the Sydney/Canberra mechanism documented in `geometry.md`:
*Sydney has higher raw dot product at every layer, yet Canberra wins after layer norm.*

**The Berlin basin does not exist in raw geometry. It exists only in normalized space.**

### Quantifying the mechanism

Raw projections onto the residual at L28:

|  | Vienna projection | Berlin projection | Ratio |
|--|-------------------|-------------------|-------|
| Germany | 3018 | 2137 | **1.41×** |
| Beethoven | 4072 | 1656 | **2.46×** |

Germany has a much more balanced Vienna/Berlin ratio — enough Berlin survives normalization
to win. Beethoven's Vienna projection is so much larger that after normalization, Berlin barely
leads. That's why Germany gets 99.9% and Beethoven gets 37%.

---

## Experiment 4: Force Decomposition

We run `residual_decomposition` on Beethoven's prompt at L24-L33.

**FFN dominates every single layer.** There are no attention-dominant layers in this range.

| Layer | FFN fraction |
|-------|-------------|
| L24 | 56.9% |
| L25 | 55.7% |
| L26 | 61.0% |
| L27 | 53.2% |
| L28 | 54.8% |
| L29 | 62.0% |
| **L30** | **74.7%** (most FFN-pure) |
| L31 | 58.9% |
| L32 | 67.3% |
| **L33** | **70.9%** (largest total output) |

L30 and L33 are the most FFN-dominated layers. L30 is also where the contextual-bleed
template competition operates in the France-north case.

### Weight geometry finding

Running `weight_geometry` at L30 and L31 reveals **no capital-city specific neurons** by
weight norm alone. The top neurons push toward random language particles, punctuation, and
non-geographic tokens.

This is consistent with the earlier finding that the capital-city feature occupies <0.6%
of the residual (a 6-10D subspace). Capital push requires activation **patterns** across many
neurons — not single-neuron specialization. The landscape is distributed, not sparse.

---

## Experiment 5: The Minimal Correction

Can we fix Beethoven by nudging the trajectory at a specific layer?

### Full injection: Germany → Beethoven

We replace Beethoven's last-position residual at various layers with Germany's residual:

| Injection layer | Berlin probability | KL(donor, injected) | Residual angle |
|-----------------|-------------------|---------------------|----------------|
| L24 | 78.7% | 0.042 | 9.05° |
| L26 | 81.5% | 0.024 | 9.96° |
| L28 | 79.6% | 0.032 | 10.04° |
| L30 | 81.7% | 0.021 | 11.31° |

All four work similarly well (~80%). **No sweet spot.** The biographical deflecting force
is persistent and uniform across L24-L30 — there is no single critical layer to target.

The residual angle between Germany and Beethoven at L28 is 10°. After injection, Berlin
rises from 36.9% to 79.6%. But why not higher?

The remaining gap (79.6% vs Germany's 88.7%) is because L29-L33 **attention layers continue
to read the Beethoven token at earlier positions**. Even with the correct last-position
residual, the historical context of "Beethoven" keeps pulling the trajectory slightly.

### Subspace injection: the surgical correction

We replace only a small subspace of the residual — the component in the direction of the
answer tokens — rather than the whole vector.

**2D subspace (Berlin/Vienna only):**

| Subspace dim | Fraction of residual | Subspace cosine | Orthogonal cosine | Berlin after |
|--------------|---------------------|-----------------|-------------------|--------------|
| 2D | **0.33%** | 0.946 | 0.985 | **97.5%** |
| 5D | 0.50% | 0.918 | 0.985 | 96.7% |
| 7D | 0.54% | 0.919 | 0.985 | 95.7% |

The 2D subspace injection **outperforms** the full injection (97.5% vs 79.6%).

This is the key result. The subspace cosine of 0.946 means the two prompts have a
meaningful difference (5.4% disagreement) in just 2 dimensions. The orthogonal cosine of
0.985 means they are 98.5% identical everywhere else.

**Replacing just 2 dimensions — 0.33% of the 2560-dimensional vector — takes Berlin from
36.9% to 97.5%.** The full injection overwrites correct information and is actually worse.

### Cross-problem corrections

The same principle works universally across all three failures:

| Problem | Subspace tokens | Dim | Fraction | Before | After |
|---------|-----------------|-----|---------|--------|-------|
| Beethoven | Berlin, Vienna | 2D | 0.33% | 36.9% | **97.5%** |
| Hamlet | London, Copenhagen, England, Denmark | 4D | 0.42% | 26.6% | **96.3%** |
| France-north | Brussels, Paris, Belgium, France | 4D | 0.77% | 26.0% | **98.5%** |

Each problem has a problem-specific correction subspace. They are in different parts of
the embedding space (German geography vs English geography vs Franco-Belgian geography).
**There is no single universal correction direction.** But the mechanism is universal:
identify the 2-4 answer-relevant token embeddings, extract the subspace they span, and
replace just that component at L28.

### What is in the correction subspace?

The subspace is literally the span of the answer token embedding vectors. It encodes
the comparison between the correct answer (" Berlin") and the competing wrong answers
(" Vienna", " Bonn"). The model's compositional circuit has correctly assembled Germany
identity everywhere else — only this tiny answer-encoding component is miscalibrated.

The compositional failure is not about forgetting Germany. It's about the biographical
memory writing the wrong answer into the 2D "answer selection" subspace while leaving
the other 2558 dimensions correct.

---

## Experiment 6: The Einstein Immunity Question

Why does Einstein succeed? Both Einstein and Beethoven were born in Germany. Both prompts
have the same structure. Yet Einstein outputs Berlin 29.5% and Beethoven 36.9% (so
Beethoven is actually *more* confident — Einstein "succeeds" with lower probability).

At L28, Beethoven and Einstein are **0.9982 cosine similar** (3.4° apart). They are nearly
the same point in residual space. Yet:

- Injecting Einstein's residual into Beethoven at L28 makes things *worse*: Berlin drops from
  36.9% to 15.6%. Einstein's residual brings in Rome 11.3%, Canberra 2.4%, Jakarta 2.1%.
- Einstein's "success" is not a clean Berlin signal — it's a diffuse signal where Berlin
  narrowly wins over a field of random capitals.

Einstein doesn't have a strong competing attractor (Princeton, Zurich, Bern are not capitals).
His biographical associations scatter diffusely across many cities. Beethoven has a specific
strong attractor (Vienna) plus a literal-birthplace fact (Bonn). Vienna is coherent and wins.

**Einstein is immune not because his circuit works better, but because his biographical
associations don't coherently point toward a specific wrong answer.**

### At L33, across the three prompts

Cosine similarities at L33:
- Beethoven ↔ Einstein: **0.9947** (very close — both compositional)
- Beethoven ↔ Germany: 0.9770 (further — different prompt type)
- Einstein ↔ Germany: 0.9720 (furthest)

In PCA 2D projection at L33: Germany sits on one side (dominant PC1 = capital-fact signal),
Beethoven and Einstein sit near each other on the other side (compositional-structure signal).
The two compositional prompts are more similar to each other at L33 than either is to the
simple prompt, even though their entities and final answers differ.

---

## The Full Picture: What Goes Wrong in Compositional Reasoning

Putting it all together, here is what happens when a transformer processes
"The birthplace of Beethoven was in the country whose capital is":

**L0-L3**: Embedding. Raw signal is near-uniform across answer tokens. Berlin slightly closest.

**L4-L23**: Compositional processing. The model builds a "Germany" representation in the
last-position residual via the L24 Head 1 contextual attribute bridge circuit. The trajectory
passes through a long "generic European city" phase where Vienna is the nearest reference
token — but this is the common mode, not the signal.

**L24**: Compositional circuit fires. Logit lens: Berlin 43%. The correct answer is present.

**L26**: Capital-query frame adds Moscow as a red herring (standard capital-of-large-country
association). Logit lens: Moscow 36%, Berlin 28%.

**L28**: Peak Berlin (45%). The trajectory is in the Berlin basin in normalized space. This
is where the computation is most correct.

**L29-L31**: Biographical override. The "Beethoven" token at earlier positions continues to
be read by attention. L29-L31 FFN neurons, seeing the Beethoven context in their input
(via accumulated earlier-position signals), add Vienna-associated content. Vienna grows from
9% to 25%.

**L31**: The Bonn birthplace fact activates. Berlin 41%, Vienna 25%, Bonn beginning to emerge.

**L33**: Three-way dissolution. Berlin 37%, Bonn 20%, Vienna 15%. No decisive attractor.
The FFN at L33 has the largest output of any layer — a massive final force — but it is
divided across three competing cities.

**The correction**: At L28, the computation is 99.67% correct. Only 0.33% of the residual
(2 dimensions) encodes the wrong answer in the answer-selection subspace. Replacing those
2 dimensions with the Germany simple prompt's values at L28 takes Berlin from 36.9% to 97.5%
and eliminates Bonn and Vienna completely.

---

## Key Findings Summary

### 1. The basin boundary doesn't exist in raw space

Vienna is geometrically closer than Berlin to Beethoven's residual at every single layer from
L11 through L33. There is no moment where Berlin is closest in raw angle space. The "Berlin
basin" exists only in layer-norm space, where the common-mode "famous European city"
direction is subtracted.

### 2. The failure is three-way dissolution, not a binary flip

Beethoven's L33 distribution: Berlin 37%, Bonn 20%, Vienna 15%. Three separate
biographical memory circuits activate — country capital, literal birthplace, and city of
residence — each partially correct, jointly incoherent.

### 3. The Subspace Correction Principle

The entire compositional failure lives in 2-7 dimensions (0.33-0.77% of the residual) at
L28. This is the "answer selection subspace" spanned by the competing answer token embeddings.
Replacing only this component with the correct prompt's value at L28 achieves 96-98.5%
confidence across all three problems.

**Subspace injection outperforms full injection**: replacing 0.33% of the vector achieves
97.5% vs replacing 100% achieves only 79.6%. Full injection overwrites correct signals.

### 4. Einstein's "immunity" is biographical diffusion, not computational accuracy

Einstein doesn't have a coherent competing attractor (Princeton is not a capital). His
biographical associations scatter randomly, so Berlin wins by default. At L28, Beethoven and
Einstein are nearly identical (0.9982 cosine) but Einstein's residual is worse for Berlin
because it encodes diffuse uncertainty, not a corrective signal.

### 5. The deflecting force is persistent, not localized

Full injection works equally well at L24, L26, L28, and L30 (~80% each). The biographical
override is not a single layer's operation — it is distributed across L29-L33 attention
reads of the Beethoven context. Any single-layer full correction is equally effective but
incomplete, because subsequent attention keeps reading the Beethoven token.

### 6. FFN dominates, but weight geometry shows no specialization

FFN is dominant at all layers L24-L33 (55-75% of displacement norm). Yet no Vienna- or
Berlin-specific neurons are found by weight norm at L30/L31. Capital push requires
activation patterns across the distributed FFN, consistent with the capital feature
occupying <0.6% of the residual.

---

## Implications

**For interpretability:** The answer selection subspace (2-7D at L28) is a clean locus of
the compositional failure. It can be identified by taking the span of the competing answer
token embeddings. This gives a mechanistic, token-grounded interpretation of what the model
"debates" internally.

**For intervention:** A deployable correction is: at L28, project the residual onto the
answer-token subspace, compare with the correct answer's expected projection, and replace
just that component. This requires knowing the correct answer (oracle) but not knowing the
model internals.

**For architecture:** The persistence of the biographical override (uniform across L24-L30)
suggests that the problem is not a localized circuit malfunction but the sustained action of
the biographical memory reading Beethoven at earlier positions. CoT's strategy of rephrasing
the input to avoid the "Beethoven" token is mechanistically correct — it removes the biographical
context from the force field that the trajectory passes through.

**For theory:** Compositional reasoning in transformers is not fragile in the usual sense.
At L28, 99.67% of the computation is correct. The failure is a 0.33% miscalibration in the
answer subspace, caused by biographical memory writing its answer into the same tiny subspace
that the compositional circuit writes its answer. The two circuits compete in 2 dimensions
out of 2560.
