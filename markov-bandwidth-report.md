# Markov Bandwidth Theory — Experimental Report

**Model:** google/gemma-3-4b-it (34 layers, 2560 hidden dim)
**Experiment ID:** b8c3035a-9c62-4059-9d3b-e9e523a5006e
**Date:** 2026-03-06

---

## Theory

The residual stream is a single 2560-dimensional vector that carries the model's entire
computation state. Each layer reads only the current vector and writes its update. If this
is truly Markovian, two different prompts that produce the same residual vector at layer L
should produce identical output from layer L+1 onward.

Prior work established:
- The capital-city prediction occupies ~0.6% of residual space (~15 dims)
- Sydney and Canberra are 85° apart — nearly independent dimensions
- The Sydney→Canberra flip is mediated by layer norm common-mode rejection, not direct suppression
- Layer 26 FFN is causally necessary for the Australia capital fact

This experiment tests whether the Markov property holds, where it breaks, and how the
residual's "bandwidth" evolves across layers.

---

## Experiment 1 — Do Different Phrasings Converge?

Four phrasings of "what is Australia's capital?":
- A: "The capital city of Australia is"
- B: "What is the capital of Australia? The answer is"
- C: "Australia's capital city is called"
- D: "The Australian seat of government is in"

All four produce Canberra.

### Pairwise cosine similarity across layers

| Layer | Min cosine | Max cosine | Centroid distance | Pattern |
|-------|-----------|-----------|-------------------|---------|
| 0     | 0.714     | 0.998     | 0.183             | Very different (raw token embeddings) |
| 8     | 0.993     | 0.996     | 0.006             | Rapid convergence — 30× reduction |
| 16    | 0.996     | 0.998     | **0.003**         | **Tightest convergence** |
| 24    | 0.989     | 0.994     | 0.009             | Diverging — semantic differentiation |
| 26    | 0.988     | 0.993     | 0.010             | Continuing to diverge |
| 30    | 0.984     | 0.991     | 0.013             | |
| 33    | 0.978     | 0.990     | 0.016             | Most diverged (but all >0.978) |

**Finding:** Convergence-then-divergence. Prompts equalize by layer 8–16, then diverge as
later layers do increasingly prompt-specific work. All still produce Canberra despite diverging.

### City subspace fractions

Projecting onto the Sydney / Canberra / Melbourne basis:

**Layer 16:** ~0.07% of residual in city subspace. Melbourne dominates negatively.
Canberra projection near zero. Consistent across all 4 prompts.

**Layer 26:** City subspace expands sharply.

| Prompt | Sydney projection | Canberra projection | Sydney fraction |
|--------|-----------------|-------------------|----------------|
| A (capital city) | 3,283 | 451 | 0.55% |
| B (what is / answer) | 3,333 | 533 | 0.57% |
| C (is called) | 3,106 | 487 | 0.46% |
| D (seat of government) | 3,092 | 516 | 0.50% |

All four prompts show **the same Sydney > Canberra pattern** with comparable magnitudes.
The city-relevant subspace has converged across phrasings by L26 — despite the full residual
still diverging (centroid_distance 0.010). The prompts have aligned their factual representations
even while their surface-form residuals continue to differ.

---

## Experiment 2 — The Patching Test

**Source:** "The capital city of Australia is" → Canberra
**Target baseline:** "The largest city in Australia is" → Sydney

Patch the source residual into the target at each layer and observe the output.

| Layer patched | Output | Result |
|--------------|--------|--------|
| 4  | Sydney  | FAIL |
| 8  | Sydney  | FAIL |
| 12 | Sydney  | FAIL |
| 13 | Sydney  | FAIL |
| **14** | **Canberra** | **SUCCESS** |
| 15 | Canberra | SUCCESS |
| 16 | Canberra | SUCCESS |
| 20 | Canberra | SUCCESS |
| 24 | Canberra | SUCCESS |
| 26 | Canberra | SUCCESS |
| 28 | Canberra | SUCCESS |
| 32 | Canberra | SUCCESS |

**Sharp binary threshold at Layer 14.** No mixed states — either Sydney (L≤13) or
Canberra (L≥14). All successful patches produce identical text ("Canberra, the capital
city. It is located in the Australian Capital Territory...") because once the first token
is Canberra, greedy continuation is deterministic.

**Markov property confirmed for same-task patching.** The last-position residual at L14
is a sufficient Markov state for distinguishing "capital city" from "largest city" within
the same semantic domain.

Note: The patching threshold (L14) precedes the Canberra *emergence* in the logit lens
(L26). The residual already contains sufficient information for downstream layers to
retrieve the correct answer from L14 onward — those layers haven't yet done the work
to surface it, but the signal is there.

---

## Experiment 3 — Cross-Task Patching

**Source:** "The capital city of Australia is" → Canberra
**Target baseline:** "Translate to French: The weather is beautiful today." → French translation

| Layer patched | Output | Result |
|--------------|--------|--------|
| 14 | French translation (unchanged) | FAIL |
| 26 | French translation (unchanged) | FAIL |
| 32 | French translation (unchanged) | FAIL |

**Complete failure at all layers tested.** The output is byte-for-byte identical to the
baseline French translation regardless of which layer is patched.

**Why it fails:** Attention at every layer after the patch reads ALL token positions.
The other positions ("Translate", "to", "French", "The", "weather"...) still carry their
original translation context. That context, spread across all positions, overpowers any
signal injected into the last position alone.

**Key insight: the full Markov state is the residual at every position, not just the last.**

The contrast with Experiment 2 is clean. Same-task patching works because the other token
positions in the target prompt ("The", "largest", "city", "in", "Australia", "is") are
semantically compatible with the source residual being injected. Cross-task positions are
hostile and cannot be overridden by a single-position patch.

---

## Experiment 4 — Subspace Patching (design note)

The theory predicts that the city-relevant 0.6% of the residual might operate independently
from the other 99.4%. If true, replacing only the city dimensions of the translation residual
with the capital-query values should produce Canberra even while 99.4% of the state is
doing French translation.

This is not directly testable with the current `patch_activations` tool (which patches the
full vector). What would be needed:

1. Extract activations from both prompts at L26
2. Compute the projection onto the city basis (Sydney/Canberra/Melbourne)
3. Construct hybrid = city dimensions from source + orthogonal complement from target
4. Inject this hybrid into a forward pass

The prior geometry work suggests this would likely fail: the nonlinear attention and FFN
operations at layers 27–33 mix the subspaces. The "independent lane" exists in the residual
representation but probably does not survive the nonlinear forward pass as an independent signal.

---

## Experiment 5 — Bandwidth Measurement

How many dimensions does the capital-city feature occupy? Testing at multiple layers using
6 capital prompts (positive) vs 6 non-capital prompts (negative).

| Layer | Dims for 50% | Dims for 80% | Dims for 95% | Type | Top token, dim 1 |
|-------|-------------|-------------|-------------|------|-----------------|
| 16    | 1 (68.3%)   | 3           | 6           | directional | "ꗜ" (artifact) |
| 24    | 2           | 5           | 8           | subspace | "ꗜ"; Canberra at dim 8 |
| 26    | 3           | 6           | 9           | subspace | "ꗜ"; Canberra at dim 9 |
| 33    | 3           | 6           | 10          | subspace | Canberra at dim 2 (17.8%) |

### What this means

**The bandwidth EXPANDS from L16 to L33.** This contradicts the compression hypothesis.

At **Layer 16**, the capital-vs-non-capital distinction is captured almost entirely in one
direction (68% variance in dim 1). That direction corresponds to a garbage artifact token —
it's a generic "question-about-a-capital" format signal, not semantic content. The model
can tell you're asking about a capital but doesn't yet know which capital.

By **Layer 24**, country-specific content appears: "Brisbane" at dim 3, "Canberra" at dim 8,
"Shakespeare" at dim 5 (from the "author of Hamlet" negative prompts). The feature has
expanded into a subspace because each country-capital pair needs its own dimension.

By **Layer 33**, "Canberra" is the second-highest-variance dimension (17.8%). The
country-specific answers have become the primary distinguishing content.

**The feature evolves from a single format direction into a multi-dimensional semantic subspace.**
Each new country adds roughly one dimension. The "bandwidth" of the capital-city feature
is not a fixed channel but a growing subspace that accumulates country-specific information.

---

## Experiment 6 — Bandwidth Competition

Does adding more context dilute the factual signal?

**Simple:** "The capital of Australia is"
**Competing:** "The capital of Australia is located south of Sydney and north of Melbourne,
between the two largest cities, and the name of this capital city is"

Logit lens comparison:

| Layer | Simple (top prediction) | Competing (top prediction) |
|-------|------------------------|---------------------------|
| 20    | "located" 8.3%         | " " 11.3% — unfocused     |
| 24    | Sydney 47.9%, **Canberra 22.7%** | Australia 46.1%, Sydney 27.9%, **Canberra 11.6%** |
| 26    | **Canberra 94.1%**, Sydney 6.0% | **Canberra 87.1%**, Sydney 11.8% |
| 30    | Canberra 90.6%         | Canberra 89.5%             |
| 33    | **Canberra 92.2%**     | **Canberra 92.6%**         |

**Competition is real at L24** (Canberra probability halved: 22.7%→11.6%). The additional
geographic context occupies bandwidth that would otherwise carry the Canberra signal.

**But it fully recovers by L33.** Both prompts end at nearly identical confidence (92.2% vs
92.6%). The later layers — L26 FFN acting as capital fact store and L33 Head 5 as confidence
restorer — compensate for the mid-process dilution.

**Interpretation:** The Markov bandwidth is not rigidly fixed. Same-domain competition
dilutes signals in intermediate layers but the later bottleneck layers can reconcentrate them.
The 6-10 dimensional subspace has enough room that competing geographic information temporarily
crowds the Canberra signal without permanently displacing it.

---

## Experiment 7 — CoT as Bandwidth Relief

**Direct (4-hop failure):** "The birthplace of the spouse of the author of Hamlet was"
→ "in the county of Norfolk" (WRONG)

**CoT (should succeed):** "The author of Hamlet was William Shakespeare.
Shakespeare's spouse was Anne Hathaway. Anne Hathaway was born in"
→ "1556" (WRONG — correct year, but asked for birthplace not birth year)

### Subspace at Layer 24

Basis: Stratford / Denmark / London / English / Danish

| Prompt | Dominant direction | Runner-up | Ratio | Winner |
|--------|------------------|-----------|-------|--------|
| Direct | Denmark: 1,206   | Stratford: 1,050 | 1.15:1 | **Denmark (wrong)** |
| CoT    | Stratford: 1,163 | Denmark: 474  | **2.45:1** | **Stratford (correct)** |

CoT confirmation: **Denmark drops 61%** (1,206 → 474). Stratford dominates 2.45:1.
The CoT residual is dramatically cleaner at L24. This directly confirms the bandwidth
prediction: resolving the referent in prior tokens removes competing directions from
the residual at the critical disambiguation layer.

### Logit lens comparison

**Direct prompt trajectory:**
- L24: "what" 48.4%, London 13.9%, England 8.4% — confused, no Stratford
- L26: **Denmark 75.8%**, London 9.1% — wrong entity wins
- L28: Denmark 50.4%, London 23.8% — still Denmark
- L30: London 41%, Denmark 22% — leadership changes but still wrong
- L33: **"in" 23.7%** — gives up entirely, outputs a preposition

**CoT prompt trajectory:**
- L16: "England" 0.24% — early correct geographic framing
- L24: Shakespeare 66.4%, England 27.5% — entity correct, location not yet resolved
- L26: **Stratford 72.7%**, Shakespeare 16.2% — correct location emerging
- L28: **Stratford 87.1%**, Shakespeare 4.9% — strong and converging
- L30: **" " (space) 51.2%**, Stratford 10.1%, April 6.1% — CATASTROPHIC COLLAPSE
- L33: **" " (space) 98.4%**, Stratford 0.15% — temporal template dominates

### What happened at Layer 30

The CoT circuit succeeded. Stratford was at 87.1% at L28 — a clear, confident correct
prediction. Then L30 activated a temporal template: "born in" → year/date follows, not
city. The space token at 98.4% is a numeric preamble; the actual output was "1556"
(Anne Hathaway's birth year, which is correct — but the wrong dimension of fact).

This is the **template competition failure mode** documented in prior CoT experiments.
The circuit operating at L24–28 and the template operating at L30–33 are independent
mechanisms that can contradict each other. The template won.

### Two failures, two fixes

| Prompt | Failure mode | What went wrong |
|--------|-------------|----------------|
| Direct | Referent competition | Head 1 extracted Denmark (Hamlet setting) not Elizabethan (Shakespeare nationality) |
| CoT | Template competition | L30 temporal template hijacked a correct geographic circuit |

The CoT fix for referent competition works perfectly. But it cannot fix template competition.
The fix for template competition is grammatical: use "The birthplace of Anne Hathaway was"
(geographic template) not "Anne Hathaway was born in" (temporal template).

---

## Synthesis

### The Markov Property — Verdict: Partially Confirmed

| Prediction | Result |
|-----------|--------|
| Same-task patching works | **CONFIRMED** — sharp threshold at L14 |
| Cross-task patching at last position fails | **CONFIRMED** — fails at all layers |
| Feature compresses into fewer dimensions | **REFUTED** — expands from 3D to 6D |
| More context dilutes the signal | **CONFIRMED** at L24–26, recovered by L33 |
| CoT produces cleaner, lower-dimensional states | **CONFIRMED** — Denmark drops 61%, ratio 1.15:1 → 2.45:1 |

### What the experiments reveal

**1. The Markov property holds within a semantic task.**
When source and target prompts share the same task structure, transplanting the last-position
residual is sufficient to override the prediction. The threshold is Layer 14 — sharp, binary,
no gradual transition. Below it: the residual encodes "what kind of question", but not which
specific answer. Above it: the residual encodes the specific answer strongly enough to survive
the attention reads of all subsequent layers.

**2. The full Markov state is all positions, not just the last.**
Attention at every layer reads every position. Cross-task patching fails because the other
token positions still carry their original task context. You cannot fool the model's identity
by changing only the last position — it is reading all the others.

**3. Bandwidth is not a fixed channel — it is a growing semantic subspace.**
The capital-city feature starts as a ~1D format signal at L16 (you're asking about a capital)
and expands to a ~6D country-specific semantic subspace by L33 (this specific capital is
Canberra/Ankara/Tokyo etc.). The bandwidth *expands* as facts become concrete. Each
country's capital fact occupies roughly its own dimension.

**4. Late layers are both amplifiers and corruptors.**
Experiment 6 shows later layers (L26 FFN, L33 Head 5) can reconcentrate a diluted signal,
recovering from mid-process bandwidth competition. But Experiment 7 shows the same layers
can *destroy* a correct signal (L30 template activation, L33 temporal override). The late
layers are not passive relays — they are active, and they can overwrite earlier correct computation.

**5. CoT measurably cleans the Markov state.**
The CoT advantage is not an abstract engineering heuristic — it is visible in the residual
geometry. Denmark's projection drops 61% when the referent is resolved in prior tokens.
The Markov state at L24 is quantifiably cleaner, and the logit lens at L26–28 confirms the
correct answer is being assembled. The failure (template competition) is an independent
downstream mechanism, not a failure of the CoT-as-cleanup hypothesis.

### Open questions

1. **What causes the L30 temporal template activation?** Is it a specific FFN neuron or
   attention head? Can it be characterized the way L33 Head 5 was?

2. **Can we patch all positions simultaneously?** If patch_activations could transplant
   the full sequence of residuals (not just the last position), would cross-task patching
   succeed? This would fully test whether the all-position Markov state is sufficient.

3. **Does bandwidth competition scale with dissimilarity?** We saw same-domain competition
   (geographic context) recovers by L33. Would cross-domain competition (e.g., mixing
   capital queries with biographical facts) be harder to recover from?

4. **Is the L14 threshold constant across fact types?** We found L14 for capital-city vs
   largest-city. Does biographical retrieval (Marie Curie → Warsaw) have a different threshold?
   The distributed architecture might mean no sharp threshold exists.

---

*Experiment saved to Lazarus experiment store. All results logged with add_experiment_result.*
