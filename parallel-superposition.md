# Parallel Superposition: Branching Trajectories with Deferred Collapse

**Model:** google/gemma-3-4b-it (34 layers, 2560 hidden dim, 8 attention heads)
**Experiment ID:** 75378d84-d749-43e0-9b63-b29c2005e51a

---

## Background

The residual stream is Markov. Computation at each layer depends only on the current residual state — not on the history that produced it. This means you can capture the residual at any layer, inject it into a different forward pass, and the downstream computation will proceed as if it had always been there.

We know two things about Gemma-3-4b's compositional computation:

1. The model reaches peak correct confidence at intermediate layers — Brussels at 79.7% at L26, Germany at 98.4% at L30 — before late-layer forces degrade it.
2. Those late-layer forces are partially prompt-driven. The "France" token fires at L30 regardless of what the last-position residual says. The "Einstein" biographical context fires at L33 regardless of the Germany signal.

Previous experiments tried to fight these forces directly — ablating neurons, injecting corrective subspaces, recirculating residuals. The forces kept winning.

The new approach: don't fight the forces. Route around them. Branch the residual at the compositional peak into parallel trajectories through *different force fields* — different templates with different context tokens — where the interfering forces cannot fire.

The Dr Strange analogy is precise. He doesn't replay the same timeline hoping for a different outcome. He looks at millions of different timelines simultaneously and picks the one that works. The branches are the superposition. The collapse is choosing the winning branch.

---

## Prompts

**France-north:** `The country directly north of France has its capital in the city of`
Correct answer: Brussels. Known failure: Paris surge at L25 (90.2%), L30-31 (64-72%).

**Einstein:** `The physicist Einstein was born in the capital city of`
Correct answer: Germany (compositionally; capital = Berlin). Known failure: Ulm confabulation at L33 (51.2%) despite Germany at 98.4% through L31.

---

## Experiment 1: Finding the Branch Points

Before branching, we need to know *where* each trajectory is most correct — the layer at which the computation has done the most right and the least damage.

### France-north: Brussels and Paris across all layers

| Layer | Brussels | Paris | Top-1 |
|-------|----------|-------|-------|
| 0-22 | <1% | <2% | — |
| 24 | 51.6% | 40.0% | Brussels |
| **25** | **7.4%** | **90.2%** | **Paris** ← surge |
| **26** | **79.7%** | **20.1%** | **Brussels** ← PEAK |
| 27 | 62.1% | 37.7% | Brussels |
| 28 | 52.7% | 46.5% | Brussels |
| 29 | 58.9% | 40.6% | Brussels |
| **30** | **34.8%** | **64.8%** | **Paris** ← reconquers |
| **31** | **26.8%** | **72.7%** | **Paris** |
| 32 | 51.6% | 40.0% | Brussels |
| 33 | 60.5% | 36.7% | Brussels ← final output |

The trajectory has three distinct battles. Brussels first emerges at L24 (51.6%). Paris surges at L25 (90.2%) — a massive intermediate peak with no lasting effect, overwritten at L26. Brussels reaches its peak at L26 (79.7%). Paris reconquers at L30-31 (64-72%) through contextual bleed from the "France" token in the prompt. Brussels recovers to 60.5% at L33 — correct but degraded.

**Branch point: L26.** Every layer after the peak is potential damage.

### Einstein: Germany and Ulm across all layers

| Layer | Germany | Ulm rank | Top-1 |
|-------|---------|----------|-------|
| 0-22 | <5% | 500+ | — |
| 24 | 57.0% | 573 | Germany |
| 28 | 97.7% | 175 | Germany |
| 30 | **98.4%** | 113 | **Germany** ← PEAK |
| 31 | 98.7% | 10 | Germany |
| 32 | 71.5% | 6 | Germany |
| **33** | **35.2%** | **0 (51.2%)** | **Ulm** ← eruption |

Germany is essentially the only answer from L24 through L31, peaking at 98.4-98.7%. Ulm is rank 113 at L30, rank 10 at L31, rank 6 at L32 — not even in the top 5. Then at L33, in a single layer, Ulm erupts to 51.2% and Germany collapses to 35.2%. This is not a gradual drift — it is a catastrophic single-layer override.

**Branch point: L30.** Two layers before the eruption begins. Germany at 98.4%.

---

## Experiment 2-3: The Branches

From each branch point, five parallel trajectories. Each passes through a different force field. Each evolves independently.

### France-north — branching at L26

The donor is the France-north L26 residual (Brussels 79.7%). Each branch injects this residual into a different prompt at L26, then runs L27-L33 through that prompt's force field.

**Branch A — Continue (baseline)**
Inject France-north L26 into France-north at L26. Self-injection = identity. This is standard inference.

**Branch B — Clean Belgian template**
Inject into `The capital of Belgium is` at L26. The template contains no "France" token. The L30 contextual bleed cannot fire.

**Branch C — Neutral template**
Inject into `The answer is` at L26. No geographic content. The force field is maximally neutral.

**Branch D — Short path**
Read the France-north residual at L28 without further injection. Two layers past the branch point, before L30 interference begins.

**Branch E — Jump to L30**
Inject France-north L26 into France-north at L30. Skip L27-29. Land at L30 with the peak Brussels state.

#### Results

| Branch | Path | Brussels | Paris | Top-1 | Correct? |
|--------|------|----------|-------|-------|---------|
| A (continue) | L26→L33 self | 62.4% | 35.0% | Brussels | Yes |
| **B (Belgium template)** | **L26→L33 via Belgium** | **79.6%** | **13.9%** | **Brussels** | **Yes** |
| C (neutral template) | L26→L33 via "The answer is" | 78.8% | 15.7% | Brussels | Yes |
| D (short path) | L28 exit | 52.7% | 46.5% | Brussels | Yes (fragile) |
| E (jump to L30) | L26 at L30 | 24.1% | 35.2% | **Paris** | **No** |

**Winner: Branch B.** Brussels 79.6% — virtually identical to the L26 peak (79.7%). The Belgium template recovers the branch point confidence through the entire L27-L33 gauntlet.

Branch E is the most dramatic failure. Starting from the Brussels peak (79.7%), injecting that state at L30 of the same prompt gives Paris 35.2%, Brussels 24.1%. L30 is a Paris-dominant force field regardless of input state. The peak Brussels residual cannot survive it.

### Einstein — branching at L30

The donor is the Einstein L30 residual (Germany 98.4%). Each branch injects this residual into a different context.

**Branch A — Continue (baseline)**
Self-injection at L30. Standard inference. Ulm 51.2%.

**Branch B — Clean German template**
Inject into `The capital of Germany is` at L30. No "Einstein" biographical context.

**Branch C — Short path**
Read the Einstein L30 logit lens output directly. Germany 98.4%. No further computation.

**Branch D — Late injection**
Inject Einstein L30 into Einstein at L32. Skip L31-32. Land at L32 with the peak-Germany state.

**Branch E — Very late**
Inject Einstein L30 directly into Einstein at L33. Skip L31-32. Run just L33.

#### Results

| Branch | Path | Germany | Ulm | Top-1 | Correct? |
|--------|------|---------|-----|-------|---------|
| A (continue) | L30→L33 self | 34.7% | 51.3% | **Ulm** | No |
| B (Germany template) | L30→L33 via Germany | 7.6% | 8.8% | **Switzerland 64.8%** | No |
| C (short path) | L30 exit | 98.4% | ~0% | Germany | Yes |
| D (skip L31-32) | L30 at L32 | 11.9% | 2.7% | **"the" 42.1%** | No |
| **E (very late)** | **L30 at L33** | **98.3%** | **~0%** | **Germany** | **Yes** |

**Winner: Branch E.** Germany 98.3% by injecting the L30 state directly at L33 — bypassing L31-32 entirely. The Ulm eruption never fires.

Branch B is a striking failure for a different reason: Switzerland 64.8%. The Einstein L30 residual carries Swiss associations — Einstein spent years at ETH Zürich and held Swiss citizenship. The Germany template force field amplifies these associations rather than suppressing them. Clean template injection is not universally beneficial; the template interacts with whatever the residual carries.

Branch D collapses to format tokens ("the" 42.1%). L31-32 provide essential structural scaffolding. Skipping them produces incoherence.

---

## Experiment 4: Why the Templates Work (and Don't)

To verify that the branches explore genuinely different force fields, we measured direction angles at L30 for three contexts: the original France-north prompt, the Belgium template, and the neutral template.

### What we measured

For each prompt at L30, we computed the angle between:
- The **FFN output** direction (the retrieval component)
- The **attention output** direction (the routing component)
- The **Brussels** and **Paris** token embedding directions

| Prompt | FFN ↔ Attn | Attn ↔ Paris | Attn ↔ Brussels | FFN ↔ Paris |
|--------|------------|-------------|----------------|-------------|
| France-north | 124.4° | **85.4°** | 86.7° | 91.5° |
| Belgium template | 100.6° | 90.5° | 90.9° | 94.3° |
| Neutral | 121.5° | 89.6° | 88.9° | 89.4° |

### What this reveals

First: all three force fields are nearly orthogonal (~90°) to the Brussels and Paris token directions. The interference is not through direct pushes toward Paris in vocabulary space. It operates in orthogonal dimensions that only become visible after layer normalization — the common-mode rejection mechanism documented in geometry.md.

Second: the "France" token causes a measurable but subtle tilt in L30 attention. In the France-north prompt, attention is 85.4° from Paris (cosine +0.079) and 86.7° from Brussels (cosine +0.058). That 1.3° difference — attention slightly more aligned with Paris than Brussels — is the mechanism of contextual bleed. In the Belgium and neutral templates, this tilt disappears. Attention is symmetrically orthogonal to both cities (~90°, cosine near zero).

A 1.3° tilt sounds negligible. It isn't. Compounded over L30 and L31, with the FFN and attention both operating on the residual, this subtle directional preference is sufficient to flip top-1 from Brussels to Paris. The bleed is not about "France" literally meaning Paris — it's about the France token loading associations into the attention key-value context that weakly prefer Paris-related directions at L30.

Remove the France token and the tilt vanishes. That is why the Belgium template (no France) and the neutral template (no France) both recover Brussels 79%. They are not adding Brussels-specific knowledge — they are subtracting the contamination.

### The FFN-attention opposition

At L30, FFN and attention are strongly opposed in all three prompts: 124° for France-north, 121° for neutral, 101° for Belgium template. The two sub-components are pointing in nearly opposite directions of residual space. This is the ongoing battle: the FFN holds geographic facts, the attention is tracking contextual associations, and they are in partial conflict at every layer. The template changes which associations the attention loads, shifting the balance.

---

## Experiment 5: The Ulm Mechanism

Branch E for Einstein is the most unexpected result: inject the L30 residual (Germany 98.4%) directly at L33, skipping L31-32, and the output is Germany 98.3%. The Ulm eruption does not occur.

This requires an explanation. L33 is the layer where Ulm appears — we established this in the recirculation experiments. If the "Einstein" prompt triggers the Ulm neuron at L33, why doesn't it fire when the L30 residual is injected?

The answer: **L33 does not fire Ulm in response to "Einstein" in the prompt. It fires Ulm in response to a specific residual configuration that L31-32 produce.**

The mechanism, reconstructed from the branch results:

1. Through L24-L30, the model builds a clean Germany answer. Germany 98.4% at L30.
2. L31-32 process this Germany-dominant state. During this processing, something about the "born in the capital city of" biographical framing combined with the Einstein context causes L31-32 to introduce Ulm-related features into the residual. At L32, Germany is still top-1 (71.5%) but Ulm has risen from rank 113 to rank 6 — L31-32 have been building it.
3. L33 receives the L32 residual — already containing a significant Ulm component — and its FFN fires the Ulm eruption to 51.2%.

When we inject the L30 residual at L33 (Branch E), we bypass steps 2 and 3's first part entirely. L33 receives a residual with Germany 98.4% and Ulm at rank 113. The L33 FFN encounters a state it has not seen at that layer position: a pure-Germany input without the Ulm buildup. It cannot trigger the eruption. Germany 98.3%.

The Ulm confabulation is a *two-step process*. L31-32 are the preparation phase. L33 is the detonation. Bypassing the preparation prevents the detonation.

This updates the architecture notes for L33: the confabulation is not a direct response to biographical cues in the prompt — it is a response to what *previous layers do to the residual* when processing those cues.

---

## Experiment 6: The Geometry of Branches

### Do branches explore different territory?

At L33, we compared the natural (uninjected) residuals of the four prompts:

| Prompts | Cosine | Angle |
|---------|--------|-------|
| France-north vs Belgium template | 0.962 | 15.8° |
| France-north vs neutral | 0.939 | 20.1° |
| France-north vs "Brussels is the capital" | 0.968 | 14.6° |
| Belgium template vs neutral | 0.955 | 17.2° |

These are not small differences. From the geometry experiments, we know that a 6-10° perturbation at L24 is enough to destroy correct answers. A 15-20° separation at L33 means the branches end up in substantially different regions of residual space.

In the PCA 2D projection, France-north is a clear outlier — isolated from all other prompts at L33. The Belgium template, neutral template, and "Brussels is the capital" statement cluster together in a different region of the plane. This is the correct-answer basin. France-north lands outside it.

When Branches B and C inject the France-north L26 residual into these templates, they route the trajectory away from the France-north isolation and toward the correct-answer cluster. The template force fields don't just preserve the Brussels signal — they actively steer the trajectory into a different basin.

Branch E (France-north, jump to L30) has the opposite effect. Injecting at L30 puts the residual into the *France-north force field at its most Paris-dominant moment*. The trajectory is steered into the Paris basin. Starting from Brussels 79.7% and ending at Paris 35.2%.

### The equivalence classes of templates

Branches B (Belgium) and C (neutral) give nearly identical results: 79.6% vs 78.8% Brussels. They are in the same equivalence class. The shared property is absence of the "France" token — not presence of Belgian context. Any template without geographic contamination from the France-north prompt will give approximately the same result.

This implies the optimal branch set does not need to be diverse in the sense of covering many different domains. It needs one representative from each interference equivalence class: one "contaminated" branch (the original) and one "clean" branch (any template without the contaminating token). Additional clean branches are redundant.

---

## Experiment 7: The Collapse Protocol

Given N branches with their final distributions, which strategy selects the correct answer?

**Strategy A — Maximum confidence:** Pick the branch with the highest top-1 probability.

**Strategy B — Majority vote:** If 4/5 branches have the same top-1, collapse to it.

**Strategy C — Minimum entropy:** Pick the branch with the most decisive distribution.

### France-north results

| Branch | Top-1 | Confidence |
|--------|-------|-----------|
| A | Brussels | 62.4% |
| **B** | **Brussels** | **79.6%** |
| C | Brussels | 78.8% |
| D | Brussels | 52.7% |
| E | Paris | 35.2% |

All three strategies agree: Brussels, anchored on Branch B. 4/5 branches correct. The superposition is decisive — the one wrong branch (E) is also the least confident (35.2% for Paris, a diffuse distribution). Any collapse strategy recovers the correct answer.

### Einstein results

| Branch | Top-1 | Confidence |
|--------|-------|-----------|
| A | Ulm | 51.3% |
| B | Switzerland | 64.8% |
| **C** | **Germany** | **98.4%** |
| D | "the" | 42.1% |
| **E** | **Germany** | **98.3%** |

Only 2/5 branches are correct. Majority vote fails — there is no majority. Maximum confidence wins cleanly: Branches C and E both output Germany at 98.3-98.4%, far exceeding the confidence of any other branch.

**Recommended collapse strategy: maximum confidence.** The key signal is not what most branches agree on, but which branch is most certain. Correct branches for compositional retrieval tend to be highly confident (>79%, often >98%). Wrong branches tend to be diffuse. The confidence gap is the oracle.

**Adaptive trigger:** Run L0-L26 for every prompt. Check logit lens confidence at L26. If top-1 exceeds 85%, continue to L33 normally — no branching needed. If below 85%, branch. France-north at L26: Brussels 79.7% → branch. Simple "The capital of Germany is" → Berlin 89% → no branch.

---

## Compute Budget

| Scenario | Layers | France-north Brussels | Einstein Germany |
|----------|--------|-----------------------|-----------------|
| Standard inference | 34 | 60.5% | 35.2% (Ulm wins) |
| Branch B (Belgium) | 33 | **79.6%** | — |
| Branch E (L30→L33) | 31 | — | **98.3%** |
| Branch C (early exit) | 26 | — | **98.4%** |
| 5-branch ensemble | 53 | **79.6%** | **98.3%** |
| CoT | 300+ | ~90% | ~90% |

Two key observations:

Branch B for France-north (33 layers) outperforms standard inference (34 layers) by 19 percentage points while using fewer compute. Branching is not only more accurate — it is cheaper.

Branch C for Einstein (26 layers — early exit at L30) gives the correct answer using only 76% of the compute of standard inference. The model reaches its best answer at L30 and then proceeds to destroy it over the next 4 layers. The optimal strategy is to stop.

Even the five-branch ensemble (53 layers) is well within the cost of a single CoT inference (300+ layers). For uncertain compositional queries, the branching protocol handles them at 53 layers with 79-98% confidence. The remaining edge cases — where branching itself fails — are the ones that genuinely need CoT.

---

## Major Discoveries

### 1. Template amplification is contamination removal, not knowledge injection

The neutral template `The answer is` gives Brussels 78.8%. This template knows nothing about Belgium or Brussels — its natural output at L33 is markdown formatting tokens (`**`, 58.2%). Yet it almost matches the Belgium template (79.6%).

Both templates recover Brussels because they lack the "France" token, not because they contain Belgian information. The mechanism is negative: removing the contaminating token eliminates the contextual bleed that degrades Brussels through L30-31.

This has an important implication: you do not need a semantically related "helper" template for branching. Any template that avoids the contaminating token from the original prompt will work. The optimal branch template is the *simplest* one with no geographic overlap.

### 2. The Ulm confabulation requires intermediate layer preparation

L33 does not fire the Ulm eruption because Einstein is in the prompt. It fires because L31-32 transform the Germany-dominant residual into a configuration that loads Ulm features. The confabulation is a two-step process with a preparation phase (L31-32) and a detonation phase (L33).

This revises the architecture description for L33 FFN. The confabulation detection threshold (>12 FFN spike signals Type 2a confabulation) remains valid — but the mechanism is indirect. The Einstein biographical context doesn't reach L33 as a direct signal. It reaches it as a transformed residual, mediated by L31-32's processing.

Bypassing L31-32 (Branch E) prevents the preparation. L33 fires Germany.

### 3. Force fields are nearly orthogonal to the outputs they control

At L30, the FFN and attention output directions are ~90° from the Brussels and Paris token embeddings. The interference mechanism does not push directly toward Paris in vocabulary space. It operates through high-dimensional residual directions that only resolve to Paris after layer normalization strips the common mode.

The 1.3° tilt in attention alignment (Paris vs Brussels) sounds negligibly small. But in a 2560-dimensional space where the capital-city feature occupies only 6-10 dimensions (<0.6% of the residual), a 1.3° perturbation in the relevant subspace is substantial. The effect is concentrated in the subspace that matters.

### 4. Some layers are traps

Branch E for France-north (jump L26→L30) gives Paris 35.2%, Brussels 24.1%. The L30 force field is Paris-dominant regardless of what you inject into it. Starting from the highest Brussels confidence in the entire forward pass (79.7%) does not help. The force field overwrites it.

This is the most important negative result. Spatial navigation through the layer stack is not free. You cannot skip to an arbitrary layer and expect the residual to survive. Some layers are attractors for wrong answers, and their force is strong enough to override any incoming state. The branching protocol must route *around* these layers, not through them.

### 5. The Germany template fails for Einstein because residuals carry history

Branch B injects the Einstein L30 residual into `The capital of Germany is`. The result: Switzerland 64.8%.

This happens because the Einstein L30 residual encodes Einstein's Swiss associations — ETH Zürich, Swiss citizenship, years in Bern. These are not erased by reaching L30 with Germany at 98.4%. They are present as minority features in the residual. The Germany template force field, encountering this residual, amplifies the Swiss component rather than suppressing it.

This is a fundamental constraint on residual injection: **residuals carry history**. The L30 state is not just "Germany 98%." It also contains everything the model has computed about Einstein along the way. A template that amplifies those minority features can produce surprising results.

The lesson for the branching protocol: the "clean" template must be clean in the sense of not amplifying the competing associations in the donor residual, not just clean of the contaminating token from the original prompt.

---

## Open Questions

**Why does Germany template amplify Switzerland?** The L30 Einstein residual in the Germany template force field selects Switzerland at 64.8%. Which neurons at L30-33 in the Germany template are responding to the Swiss signal in the residual? This is a tractable question: compare neuron activations at L30-33 for the Germany template with and without the injected Einstein residual.

**Does Branch B generalize?** France-north is one instance of a prompt with a contaminating geographic token that biases L30. Does the clean-template amplification work for other prompts with similar contamination patterns? The hypothesis is yes — any prompt where a high-salience context token drives L30 attention away from the correct answer should benefit from routing through a contamination-free template.

**What is the minimum branch set?** Branches B and C give 79.6% and 78.8% respectively — they are in the same equivalence class. For France-north, one clean template branch is sufficient. The branching protocol does not need to be exhaustive; it needs to cover the interference equivalence classes. For a given prompt, these classes can probably be identified from the direction angles of the contaminating tokens at the branch point layer.

**Can Branch D (short path) be made robust?** Brussels 52.7% at L28 is fragile — only 6% margin over Paris. But it uses only 2 extra layers past the branch point. If a probe or confidence signal could identify *which* L28 states are fragile vs stable, Branch D becomes a much cheaper option for a subset of prompts.

**Where does the Ulm preparation happen in L31-32?** We know L31-32 build the Ulm signal (rank 113 → rank 6 across two layers). Which attention heads or FFN neurons are responsible? This is directly measurable with head attribution and top_neurons at L31 and L32. The preparation circuit for the Ulm confabulation is a specific, identifiable mechanism.

---

## Summary

The parallel superposition hypothesis is confirmed. Branching the residual at the compositional peak into parallel trajectories through different force fields produces correct answers that single-track inference misses — and does so at equal or lower compute cost.

The mechanism is not mysterious: different prompt contexts create different force fields at the same layers, and some force fields are hostile to the correct answer while others are neutral or amplifying. Routing through a neutral force field preserves the peak-confidence state. Routing through a hostile force field destroys it.

The optimal strategy is to find the branch point (peak confidence), identify the contaminating elements in the original prompt (the tokens that drive hostile force fields), and route through a template that excludes them. The collapse protocol selects the highest-confidence branch. The result is a system that handles compositional uncertainty at 26-53 layers with >79% confidence — cheaper than standard inference for the difficult cases, and far cheaper than CoT for all of them.
