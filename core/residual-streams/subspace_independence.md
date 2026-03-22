# The Subspace Is the Address: Local Coordinate Frames in the Factual Landscape

*Experiment ID: fd54bf19-f62a-4727-bb96-9488fc543afc*
*Model: google/gemma-3-4b-it (34 layers, 2560 hidden dim)*

---

## Overview

A previous experiment failed. We tried injecting the "Canberra" prediction into an unrelated
prompt by using the Sydney and Canberra token-embedding vectors as a basis — the intuitive
idea that facts live in vocabulary-space directions. It produced Sydney, not Canberra.
The conclusion: token embeddings are the wrong coordinate system.

This experiment tests the right one.

The theory is that the residual stream at layer 26 is a landscape where factual predictions
live as directions — not as globally separable regions, but as specific orientations in a
2560-dimensional space. The capital-of-Australia prediction occupies roughly 15 dimensions
(0.6% of the map). We've proven the Markov property with KL=0.0 patching: downstream layers
only read the current residual state, nothing else.

The question this experiment answers: **can we identify and surgically extract those 15
dimensions, graft them into a completely unrelated prompt, and have the unrelated prompt
predict Canberra — while preserving everything else?**

If yes: the factual region is genuinely independent. You can transplant a fact without
disturbing the surrounding computation. If no: the regions interact nonlinearly — orthogonal
in the raw space doesn't mean independent when read through layer norm and attention.

The short answer is yes. But the mechanism is surprising, and the path to finding it
reveals something deeper about how the landscape is organised.

---

## Setup: What PCA Coordinate System to Use

The natural tool is `compute_subspace`: run a set of prompts through the model at a target
layer, collect the last-position residuals, mean-centre them, and compute the top principal
components via SVD. The result is a basis for the directions of maximum variation among
those prompts.

We start with the obvious choice: 15 capital-city prompts.

```
"The capital of Australia is", "The capital of France is", "The capital of Japan is", ...
```

PCA at layer 26, rank=10. The variance breakdown is clean:

| Component | Variance | Cumulative |
|-----------|----------|------------|
| PC1 | 31.2% | 31.2% |
| PC2 | 11.9% | 43.0% |
| PC3 | 10.0% | 53.1% |
| PC4 | 9.8% | 62.8% |
| PC5 | 8.3% | 71.1% |
| PC6 | 7.7% | 78.8% |
| PC7 | 6.5% | 85.3% |
| PC8 | 5.6% | 90.9% |

Seven components explain 85.3%, eight explain 90.9%. The factual landscape appears compact.

---

## Phase 1: The Raw Landscape at Layer 26

Before any injection, `decode_residual` at layer 26 reveals something striking.

**Raw top tokens (before layer norm) across three very different prompts:**

| Prompt | Raw top-1 | Raw top-2 | Raw top-3 |
|--------|-----------|-----------|-----------|
| "The capital of Australia is" | ꗜ | 𒉶 | PLDNN |
| "Translate to French: The weather is" | ꗜ | 𒉶 | PLDNN |
| "The largest planet is" | ꗜ | 𒉶 | PLDNN |

The top-8 raw tokens are **identical** across all three prompts, in nearly the same order.
The raw residual — before layer normalisation — points at the same garbage tokens regardless
of what the model is computing. The mean direction is also identical: dominated by "cause",
"crashing", multilingual tokens.

**After layer norm:**

| Prompt | Norm top-1 | Probability |
|--------|------------|-------------|
| Australia | Canberra | 93.8% |
| Translation | beautiful | 47.6% |
| Planet | Jupiter | 54.5% |

The factual signal is entirely a layer-norm effect. The common mode — the enormous shared
component that dominates the raw residual — is subtracted, and what remains are the small
orthogonal directions specific to each computation. Canberra wins not by having a large raw
projection, but by being nearly orthogonal to the mean. Layer norm amplifies it by removing
everything that isn't specific.

This is why the geometry of the Canberra flip is so counterintuitive: Sydney has a *higher*
raw dot product at every single layer, yet Canberra wins. The capital-city feature doesn't
live in the dominant directions of the residual — it lives in the quiet, nearly orthogonal
complement.

---

## Phase 2: The Independence Test — Why the First Approach Fails

We take the 10-component within-family PCA subspace and inject Australia's last-position
residual (projected onto that subspace) into three unrelated prompts at layer 26.

```
inject_residual(
    donor="The capital of Australia is",       # → Canberra 92%
    recipient="Translate to French: The weather is",
    layer=26, subspace_only=True, subspace_name="capital_city_L26"
)
```

| Recipient | Injected top-1 | KL(donor, injected) | Orthogonal cosine |
|-----------|----------------|---------------------|-------------------|
| Translation | good (96%) | 11.35 | 0.929 |
| Fairy tale | "," (82%) | 17.98 | 0.890 |
| Planet | Jupiter (27%) | 7.90 | 0.927 |

All three fail. No recipient predicts Canberra.

The diagnostic is in one number: **subspace_cosine_similarity = 0.974** for all three
recipients. This measures the cosine similarity between the donor's projection onto the
10D subspace and the recipient's projection onto the same subspace. A value of 0.974 means
the translation prompt and the Australia prompt already look nearly identical inside the
PCA space. We are injecting something that was essentially already there.

The within-family PCA captures the directions of maximum variation *among capital-city
prompts* — how Australia differs from France, from Japan, from Germany. But those directions
are also high-energy for every other prompt. All residuals at layer 26 project similarly
onto these directions, because the within-family variation is small relative to the
universal structure of the residual space.

The factual signal for Canberra — the fine-grained direction that makes Canberra win after
layer normalisation — is not one of the top variance directions within the capital-city
family. It's something else.

**Control:** full injection (subspace_only=False) at the same layer produces Canberra 93.2%
with KL=0.018. The setup is correct. The Markov property holds. The problem is the
coordinate system.

---

## Phase 3: The Right Coordinate System

The insight: a PCA over capital-city prompts tells you how capitals differ *from each other*.
We need to know how capital-city queries differ *from everything else*.

We rebuild the subspace with a mixed training set: five capital prompts plus eight unrelated
prompts spanning translation, creative writing, astronomy, arithmetic, chemistry, and narrative.

```python
prompts = [
    "The capital of Australia is",
    "The capital of France is",
    "The capital of Japan is",
    "The capital of Germany is",
    "The capital of Brazil is",
    # --- domain boundary ---
    "Translate to French: The weather is",
    "Once upon a time in a land far",
    "The largest planet is",
    "My favourite colour is",
    "The first president of the United States was",
    "Two plus two equals",
    "The chemical formula for water is",
    "She opened the door and saw",
]
```

PCA at layer 26, rank=10. Same architecture, different training set. Now the principal
components are shaped by the domain boundary: they capture what makes capital-query residuals
different from translation/creative/factual/arithmetic residuals.

The result:

| Recipient | Injected top-1 | KL(donor, injected) | Orthogonal cosine |
|-----------|----------------|---------------------|-------------------|
| Translation | **Canberra (94.1%)** | **0.021** | **0.9997** |
| Fairy tale | **Canberra (87.9%)** | **0.015** | **0.9999** |
| Planet | **Canberra (90.3%)** | **0.0035** | **0.9999** |

All three pass. KL divergences are effectively zero — the injected output is statistically
indistinguishable from the original donor output.

The orthogonal cosine is the key diagnostic. It measures similarity between donor and
recipient in the 2550-dimensional space *outside* the 10D subspace. Values of 0.9997-0.9999
mean the orthogonal components are essentially identical. The 10D subspace has captured
everything that differs between the donor and recipient. The remaining 2550 dimensions —
grammar, task format, narrative context, everything else — are identical between them and
are untouched by the injection.

### The task context survives completely

What does the model generate after the injection? The translation prompt, after receiving
Australia's factual signal:

> "Canberra. Here are a few options for translating..."

The fairy tale prompt:

> "Canberra, there lived a young woman named Elara"

Canberra was transplanted into the computation at layer 26, but the downstream layers still
see the translation task structure, the fairy tale narrative structure. The fact was grafted
in; the surrounding computation ran normally.

The planet prompt was fully overridden — no translation or narrative context to preserve,
so the model just talks about Canberra. But the other two show the independence cleanly:
**the 10D factual region and the 2550D task-context region operate independently.**

---

## Phase 4: How Many Dimensions Does a Fact Need?

We vary the rank of the cross-domain subspace and measure when the injection starts working.

| Rank | Top-1 | Prob | KL(donor) | Orthog. cosine | Result |
|------|-------|------|-----------|----------------|--------|
| 3 | nice | 10% | 5.96 | 0.946 | FAIL |
| 6 | called | 8% | 5.88 | 0.951 | FAIL |
| **8** | **Canberra** | **58.9%** | **0.357** | **0.980** | **PASS** |
| **10** | **Canberra** | **94.1%** | **0.021** | **0.9997** | **PASS** |

The transition is sharp. Rank 6 fails completely; rank 8 produces Canberra at 58.9%.
Rank 10 achieves near-perfect transfer.

**The minimum bandwidth for a capital-city factual prediction is 7-8 dimensions out of
2560** — 0.3% of the residual space. This is consistent with previous feature-dimensionality
measurements (6-10 dimensions for the capital-city subspace) and tighter than the earlier
estimate of ~15 dimensions for the Sydney→Canberra flip.

The orthogonal cosine tracks the success exactly: 0.946 and 0.951 (rank 3 and 6) fail;
0.980 and 0.9997 (rank 8 and 10) succeed. The threshold is somewhere around 0.96-0.97.
Below that, the subspace doesn't capture enough of the difference between donor and recipient
to carry the factual signal. Above it, the orthogonal space (grammar, context) is
effectively identical between them, and the injection transfers cleanly.

---

## Phase 5: At What Depth Does the Factual Region Form?

The cross-domain subspace was computed at layer 26. We now vary the injection layer, using
the same subspace, to find when the factual region becomes independently addressable.

| Inject layer | Top-1 | KL(donor) | Residual angle | Result |
|---|---|---|---|---|
| L18 | nice (33%) | 18.0 | 5.8° | FAIL |
| L22 | "in" (66%) / Canberra #2 (17.7%) | 1.61 | 9.2° | PARTIAL |
| **L24** | **Canberra (92.4%)** | **0.064** | 13.3° | **PASS** |
| **L26** | **Canberra (94.1%)** | **0.021** | 17.2° | **PASS** |
| L30 | Canberra (52.3%) | 0.494 | 18.6° | PASS |

At layer 18, the two prompts are only 5.8° apart — almost identical residuals. The
cross-domain subspace (computed at L26 where prompts are 17.2° apart) barely captures
any variation when applied at L18. The injection is a near-no-op.

At layer 22, Canberra appears at rank 2 with 17.7% probability, but "in" wins. The factual
signal is starting to form but can't yet overcome the surrounding context.

**Layer 24 is the entry point.** Canberra wins at 92.4% — almost as strong as layer 26.

This is not coincidental. Layer 24 Head 1 is the contextual attribute bridge: the attention
head that reads the subject token and writes the relevant relational attribute ("Australian",
"capital city", etc.) into the last-position residual. The factual coordinate frame
crystallises exactly at the layer that *writes* it. Once Head 1 has acted, the capital-city
prediction exists as a coherent independent direction. Below L24, it doesn't exist yet.

Layer 30 still works (Canberra 52.3%) but with lower confidence — the L26-computed subspace
is being used outside its natural domain, and some coupling has accumulated across the
intervening layers.

A note on time-travel: capturing the Australia donor's residual *at layer 30* and injecting
its subspace projection into the translation recipient *at layer 26* still produces Canberra
at 59.8%. The factual signal persists forward in the donor and can be back-ported four
layers. The landscape is not perfectly Markov at the subspace level — the factual region
is stable enough to survive being read at a later layer.

---

## Phase 6: The Coordinate Frame Is the Address

The deepest result of the experiment. We build two country-specific subspaces:

- `australia_only_L26` (rank=8): Australia vs. 8 unrelated prompts
- `france_specific_L26` (rank=8): France vs. 8 unrelated prompts

Then we run a 2×2 experiment: two donors (Australia, France), two subspaces (Australia,
France), same recipient (translation prompt), same layer (26).

| Donor | Subspace | Injected top-1 | KL(donor) | Orthog. cosine |
|-------|----------|----------------|-----------|----------------|
| France (Paris 80.9%) | france_specific | **Paris 89.3%** | 0.048 | **1.0** |
| Australia (Canberra 92%) | australia_only | **Canberra 93.2%** | 0.018 | **1.0** |
| France (Paris 80.9%) | **australia_only** | **Canberra 45.9%** | 7.12 | 0.965 |
| Australia (Canberra 92%) | **france_specific** | **Paris 84.3%** | 14.01 | 0.947 |

The diagonal works. Each country injected through its own subspace produces the correct
capital with near-perfect fidelity (KL < 0.05, orthogonal cosine = 1.0).

The off-diagonal is the finding. **France injected through Australia's subspace outputs
Canberra. Australia injected through France's subspace outputs Paris.**

The generated text confirms it completely:

- france_specific subspace: *"Paris is beautiful today."* (both France donor AND Australia donor)
- australia_only subspace: *"Canberra. Here are a few options for translating..."* (both donors)

The output follows the coordinate frame, not the vector content. The donor's residual is
being projected onto the subspace and used to replace the corresponding directions in the
recipient — but the subspace defines *which* directions those are. The Australia subspace
spans directions that distinguish "capital-of-Australia query" from everything else; when
France's vector is projected onto those directions, the projection captures Australia-like
components of France's residual, and the resulting injection produces Canberra.

Put differently: **the subspace is the address, not the content.** Injecting through a
country's subspace sends the recipient to that country's region of the factual landscape,
regardless of whose residual you're reading from.

---

## What This Means for the Structure of the Landscape

### Local coordinate frames, not global regions

The Markov Bandwidth Theory described the residual as a map where every prediction coexists
as a direction in 2560-dimensional space. This experiment refines that picture.

The factual landscape is not organised as globally separable regions — "Canberra lives here,
Paris lives there, and they're orthogonal to each other." It's organised as **local
coordinate frames**: each country's factual prediction lives in its own private 8-dimensional
neighbourhood, defined by the directions that distinguish that country's capital query from
all other computations.

These frames are independent of the task-context space (orthogonal cosine ~1.0 in all
successful injections — the 2550D task space is completely preserved). But they are not
independent of each other: Australia's frame and France's frame are distinct subspaces, and
projecting one country's residual through another country's frame sends you to the wrong
neighbourhood.

The geometry is more like local coordinate charts on a manifold than like independent axes
in a global space.

### Why token embeddings failed

The previous experiment used the Sydney and Canberra unembedding vectors as a basis. These
are directions in the 262,208-dimensional vocabulary space, projected into the 2560D residual
space as "the direction that would maximally predict Sydney" and "the direction that would
maximally predict Canberra." But these are global vocabulary directions — defined by the
model's final linear layer, not by how the model organises knowledge at intermediate layers.

The local coordinate frame for "Australia capital query" at layer 26 is computed from
how the *activations* vary across prompts, not from how the vocabulary is structured.
The two coordinate systems don't align. Using vocabulary directions to navigate the
activation landscape is like using longitude and latitude to navigate a subway map.

### The L24 crystallisation point

The factual region doesn't exist below layer 24. This is not a smooth gradient — at L22,
the prompts are 9° apart and the cross-domain subspace can't transfer the prediction. At
L24, it works at 92.4%. Head 1 at layer 24 writes the contextual attribute bridge (the
"Australian" direction, the "capital city" direction) into the last-position residual.
Before that act, the local coordinate frame for this fact doesn't exist. After it, the
frame is stable through at least layer 30.

The circuit creates the coordinate frame, and the coordinate frame is what makes the
prediction injectable.

### The 0.3% number

Eight dimensions out of 2560 — 0.3% of the residual — are sufficient to specify a
capital-city prediction completely. The remaining 99.7% (grammar, task format, narrative
context, language register) can be whatever they need to be from the recipient prompt.
Factual prediction and linguistic competence are encoded in independent subspaces of the
residual.

This is both a measure of efficiency and a measure of robustness. The model needs only
8 dimensions to specify "Canberra"; the other 2552 dimensions are free for the task at
hand. Conversely, you only need to perturb 8 dimensions to redirect a factual prediction
entirely — the surrounding 2552 dimensions don't protect the fact.

---

## Summary

| Question | Answer |
|----------|--------|
| Does the factual region operate independently? | Yes — orthogonal cosine ≈ 1.0 after correct subspace injection |
| Which PCA coordinate system works? | Cross-domain (capital + unrelated), not within-family |
| Why did within-family PCA fail? | It captures universal high-energy directions present in all residuals |
| Why did token embeddings fail? | Vocabulary directions ≠ activation-space local coordinate frames |
| Minimum dimensions for a factual prediction? | 7-8 (0.3% of 2560D) |
| At what depth does the factual region form? | Layer 24 — exactly where Head 1 writes the attribute bridge |
| Is the output determined by the donor or the subspace? | The subspace — France via Australia's frame → Canberra; Australia via France's frame → Paris |
| What is the map's structure? | Local coordinate frames, ~8D each, per fact-type. Independent of task-context space. |

The landscape metaphor needs one update: it's not a flat map with labelled regions.
It's a collection of local atlases — each fact has its own chart, defined by how that
fact's computation differs from everything else at the layer where it crystallises.
To navigate to a fact, you need the right chart. The chart is the address.
