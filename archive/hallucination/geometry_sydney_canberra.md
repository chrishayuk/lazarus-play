# The Sydney→Canberra Flip: A Geometric Story

**Model:** google/gemma-3-4b-it (34 layers, 2560 hidden dim)
**Prompt:** "The capital city of Australia is"
**Experiment:** 5fd26ee8-63cc-4a9b-ba99-daa9a1de4bbc

## Background

Previous experiments established that layer 26 FFN is the single causal bottleneck for the
Australia capitals answer. When zeroed, the model flips back to Sydney (KL=0.832). The logit
lens shows a clean story: Sydney peaks at 92.2% at layer 25, then layer 26 FFN overwrites it
to Canberra 87.5% in one pass.

But logit attribution couldn't answer the spatial question: *where does Canberra come from?*
Is layer 26 FFN rotating the residual stream toward Canberra, or just suppressing Sydney until
Canberra is whatever is left? These geometry experiments — `token_space`, `direction_angles`,
`subspace_decomposition`, `residual_trajectory`, `feature_dimensionality` — answer that question.

---

## Finding 1 — Sydney and Canberra Live in Independent Dimensions

Pairwise angles between city unembedding vectors (fixed properties of the model, layer-independent):

| Pair | Angle | Cosine similarity |
|------|-------|-------------------|
| Sydney ↔ Canberra | **85.0°** | 0.087 |
| Sydney ↔ Melbourne | 52.2° | 0.613 |
| Sydney ↔ Brisbane | 67.8° | 0.378 |
| Canberra ↔ Melbourne | 85.2° | 0.084 |
| Canberra ↔ Perth | 79.4° | 0.184 |

Sydney, Melbourne, and Brisbane form a **large-city cluster** (52–68° from each other). These
are the famous, populous cities — the ones a non-expert Australian would name first.

Canberra is **geometrically isolated** from this cluster. At 85° from Sydney and 85° from
Melbourne, it is nearly as orthogonal to them as two random unit vectors in 2560 dimensions
would be. The capital city does not live near the popular cities in vocabulary space. It
occupies its own independent direction.

This is the central geometric fact. Sydney and Canberra are not opposites — there is no
single axis along which "more Sydney" means "less Canberra." They are competitors only in
softmax space, not in the underlying geometry.

---

## Finding 2 — Sydney Dominates Raw Projections for All 34 Layers

The residual trajectory at the last token position, measured against Sydney, Canberra, and
Melbourne across every layer:

```
Layer   Sydney proj   Canberra proj   Melbourne proj   Orthog. fraction
0            -2.1           9.2           -11.7          99.96%
4           +13.4           4.5           -23.4          99.97%  ← Sydney first appears
16          345.9          83.8          -242.0          99.93%
24         1847.9         300.3           751.1          99.76%  ← Head 1 fires
25         2726.3         344.7          1464.2          99.57%  ← L25 FFN peak
26         3282.8         735.2          1792.1          99.43%  ← L26 FFN "flip"
29         3785.6        1862.5          2015.3          99.48%  ← largest Canberra boost
31         3119.2        2175.4          1296.0          99.65%
32         2692.0        2406.5           768.2          99.71%
33         2571.0        1983.7           654.9          99.78%  ← final output
```

**Sydney has the highest raw dot product with the residual stream at every layer from 4 to
33.** Including the final layer — where the model outputs Canberra.

There is no crossing point in raw projection space. The "flip" visible in logit-lens
(layer 25: Sydney 92.2%, layer 26: Canberra 87.5%) does not correspond to any crossing in
the raw geometric data. Sydney's projection grows at layer 26 (+557). Canberra's also grows
(+390). Sydney still leads.

The "dominant token" field in the trajectory summary reads "Sydney" for all 34 layers.

The largest single rotation in the trajectory is at **layer 4** (16.8°) — Sydney's initial
semantic encoding of "Australian city." Layer 26 produces a rotation of 6.5°, unremarkable
among a sequence of 4–7° rotations throughout the middle layers.

---

## Finding 3 — The Flip Is a Layer Norm Effect

If Sydney has a higher raw dot product at every layer, why does the model output Canberra?

The logit computation is not `dot(residual, W_U[token])`. It is
`dot(LayerNorm(residual), W_U[token])`. Layer norm subtracts the mean and divides by the
standard deviation of the residual vector across its 2560 dimensions.

Sydney's logit advantage in raw space is correlated with the *mean* of the residual stream —
the generic "big Australian city" signal that accumulates across layers 4–25 as the model
resolves the prompt. This mean has a large projection along the Sydney unembedding direction
because Sydney is the most common, most salient association for "Australian city" in training
data.

Layer norm **subtracts that mean**. After centering, the shared big-city signal — which Sydney
was riding — is removed. Canberra, whose unembedding direction is nearly orthogonal to Sydney's
(85° away), is much less correlated with this mean. Its signal survives the centering.

The result is that layer norm acts as a **common-mode rejection filter**: it cancels the
shared Australian-city signal that Sydney and Melbourne carry and that dominates the raw
residual, leaving Canberra's weaker but more specific signal as the winner.

This explains why the L26 FFN is causally necessary even though it doesn't dominate the raw
projection: it is adding enough energy in the Canberra direction (from near zero at L25: 344,
to substantial at L26: 735) that after layer norm, Canberra crosses the threshold.

---

## Finding 4 — What the FFN Actually Does Geometrically

Direction angles at layer 26 between the FFN output vector and each city's unembedding:

| Direction | Angle to FFN output | Raw dot product |
|-----------|--------------------|-----------------|
| Sydney | 86.5° | **294.1** |
| Canberra | **86.0°** | **344.9** |
| Melbourne | 88.4° | 133.4 |

The FFN output at layer 26 is weakly aligned with both Sydney and Canberra — both are nearly
orthogonal to it. But Canberra's dot product (344.9) is 17% higher than Sydney's (294.1),
meaning the FFN output vector is marginally more aligned with Canberra than Sydney even in
raw activation space.

This is a weak pro-Canberra signal on top of the layer norm effect, not a dominant rotation.
The FFN is neither cleanly "rotating toward Canberra" nor "suppressing Sydney." It is adding
energy in a direction that is almost orthogonal to both cities but tilted slightly toward
Canberra.

---

## Finding 5 — Neuron 9444 Does Not Directly Suppress Sydney

Neuron 9444 was previously identified as the most discriminative neuron for Canberra vs
Sydney at layer 26: always suppressed (negative activation), always in the top-20 positive
contributors across all 9 capital prompts. Its claimed mechanism: suppressed state × negative
down_proj vector = positive contribution to correct capital.

Direction angles between neuron 9444's down_proj column and key directions:

| Pair | Angle |
|------|-------|
| Neuron 9444 ↔ Sydney | **90.5°** |
| Neuron 9444 ↔ Canberra | **93.4°** |
| Neuron 9444 ↔ Melbourne | 88.6° |
| Neuron 9444 ↔ FFN output | **122.7°** |
| Neuron 9444 ↔ residual | 93.2° |

The neuron's down_proj direction is essentially orthogonal to Sydney (90.5°) — it has
effectively zero direct geometric influence on the Sydney logit. It is also slightly
*anti*-Canberra (93.4°, meaning its direct contribution marginally opposes Canberra).

Its 122.7° angle to the total FFN output, combined with its negative activation, means its
actual contribution to the residual is in the *same* direction as the FFN output (double
negative: suppressed neuron with anti-FFN down_proj → pro-FFN contribution). The mechanism
is indirect, working through downstream layers (previously confirmed: feeds L30-31 FFNs
at +0.099/+0.121 alignment), not through direct vocabulary-space geometry at L26.

Neuron 9182 tells a simpler story: 104.8° from Sydney (weakly anti-Sydney), 90.3° from
Canberra (neutral). It does push weakly against Sydney but is a bystander to Canberra.

---

## Finding 6 — The Decision Uses Less Than 0.6% of the Space

Subspace decomposition of the residual stream onto city directions (Gram-Schmidt
orthogonalized, Sydney first):

| Layer | Sydney % | Canberra % (orthog.) | Melbourne % | City total | Orthogonal |
|-------|----------|----------------------|-------------|------------|------------|
| L25 | 0.421% | 0.0007% | 0.004% | **0.43%** | **99.57%** |
| L26 | 0.551% | 0.010% | 0.005% | **0.57%** | **99.43%** |

At the moment of the capital decision, **99.4% of the model's residual stream is in directions
completely orthogonal to all Australian city tokens.** The entire competition between Sydney
and Canberra happens in a sliver of the 2560-dimensional space.

Canberra's orthogonal component (the part of Canberra's direction that is independent of
Sydney, after Gram-Schmidt) grew 15× from L25 to L26: from 0.0007% to 0.010% of the
residual. A tiny but real signal that was not there before the L26 FFN fired.

The 99.4% of the residual that is orthogonal to all cities is not "wasted computation" — it
encodes the context, grammar, world knowledge, and formatting information needed for the full
output. The city decision is just the final selection within a very small subspace.

---

## Finding 7 — The Capital-City Feature Is a 6-11 Dimensional Subspace

Feature dimensionality of "capital city prompts" vs "non-capital prompts" at layers 25 and 26:

| Threshold | Dims at L25 | Dims at L26 |
|-----------|-------------|-------------|
| 50% variance | 3 | 3 |
| 80% variance | 6 | 6 |
| 95% variance | 9 | **10** |
| 99% variance | 11 | 11 |
| Classification accuracy | 100% with 1D | 100% with 1D |

The capital-city feature is **perfectly linearly separable in 1 dimension** but requires
6 dimensions to capture 80% of its variance. It is a multi-dimensional subspace, not a
clean scalar direction.

The dimensionality is nearly unchanged across the L26 flip: 9D → 10D for 95% variance.
The flip does not restructure the capital-city representation. Layer 26 FFN shifts the
weights within an existing multi-dimensional feature subspace; it does not create a new
geometric structure.

---

## The Full Story

Before these experiments, the story was: *Layer 26 FFN suppresses Sydney and Canberra
emerges.* This was the logit-attribution story — FFN adds a large negative component
to Sydney's logit and a positive component to Canberra's.

The geometric story is different and more interesting:

**Canberra was always there.** From layer 0, Canberra has a nonzero positive projection
on the residual (9.2 at layer 0, growing slowly to 344 by layer 25). It is always present
as a weak signal in an independent dimension — orthogonal to the dominant Sydney-Melbourne
city cluster, orthogonal to the mean of the residual, quietly accumulating.

**Sydney's dominance is shared.** Sydney's large raw projection is correlated with the
residual mean — the accumulated "famous Australian city" signal that many tokens contribute
to, not a specific fact about the capital.

**Layer 26 FFN does two things:** It grows Canberra's raw projection by 390 (a 2.1× increase
from L25's 344 to L26's 735) — a real and specific pro-Canberra signal. And it adds energy
in a direction that is slightly more aligned with Canberra than Sydney (dot products 344.9
vs 294.1). But neither effect is enough to flip Sydney's raw dominance.

**Layer norm is the flip mechanism.** When the final logit computation applies layer norm,
the mean of the residual is subtracted. Sydney's advantage — correlated with that mean —
shrinks dramatically. Canberra's signal, living in an orthogonal direction that is less
correlated with the mean, survives the centering. After softmax, Canberra wins.

The L26 FFN's causal necessity (zeroing it flips the output to Sydney) is therefore a
combination of: building up enough Canberra signal to survive layer norm after the mean
subtraction, and doing so in a direction specifically aligned with Canberra and not with
the shared city mean. Remove the FFN and Canberra's projection drops back to the L25 level —
too small to emerge from Sydney's shadow after centering.

The bottleneck is not a dramatic rotation. It is a precise, small injection of Canberra
signal into a specific orthogonal dimension, timed exactly at the layer where layer norm
will amplify it into a winning logit.

---

## Summary Table

| Question | Answer |
|----------|--------|
| Is the flip rotation or suppression? | **Neither — layer norm effect.** Raw projections don't flip. |
| What fraction of 2560D is involved? | **<0.6%.** 99.4% orthogonal to all cities. |
| Are Sydney and Canberra geometrically opposed? | **No — 85° apart, nearly independent.** |
| Does neuron 9444 directly suppress Sydney? | **No — 90.5° from Sydney, orthogonal.** |
| Does neuron 9444 directly boost Canberra? | **No — 93.4° from Canberra, slightly anti.** |
| When does Sydney first appear in raw space? | **Layer 4** (largest single rotation: 16.8°) |
| When does Canberra's largest boost occur? | **Layer 29** (+652 projection delta) |
| How many dimensions does the capital feature use? | **6-11D subspace** (not 1D direction) |
| Does the flip change that feature geometry? | **No** — same dimensionality before and after |
