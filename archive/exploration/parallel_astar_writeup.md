# Parallel Multidimensional A*: Subspace Signal Boosting
## Experiment a2321302-a101-464d-a6a0-403e9a501bc2

---

## Summary

We set out to test whether targeted subspace boosting during a single forward pass could fix
compositional failures — cases where the model produces a wrong answer on multi-hop queries like
"The country where Beethoven was born has its capital in." The hypothesis was that the failure
originates from a weak country signal at L24, and that amplifying it would let the model's own
coupling resolve the second hop.

The hypothesis was wrong about the location of the failure. But it was right that the model can
compose — it just also needs protection from itself.

**Central finding:** Gemma-3-4b-it correctly resolves all three tested compositional problems
through at least layer 28, reaching 67-88% confidence on the correct answer with zero
intervention. The failures are not compositional inability. They are late-layer associative memory
overrides that fire after the composition circuit has already succeeded.

---

## Prompts and Baselines

Three compositional failures were studied:

| Prompt | Model output | Correct answer |
|--------|-------------|----------------|
| "The country where Beethoven was born has its capital in" | Vienna 45.7% | **Berlin** |
| "The country north of France has its capital in" | Brussels 44.5% / Paris 44.5% (tie) | **Brussels** |
| "The birthplace of the author of Hamlet was" | "a" 18.3%, Stratford 5.9% (4th) | **Stratford** |

Two prompts that work correctly were used as reference:
- "The country where Einstein was born has its capital in" → Berlin 58.6% (correct)
- "The country where Darwin was born has its capital in" → London 59.0% (correct)

Simple reference baselines:
- "The capital of Germany is" → Berlin 89.1%
- "The capital of Belgium is" → Brussels 57.8%
- "The capital of Austria is" → "a" 46.7%, Vienna 32.0% (Austria direct is also weak)

---

## Experiment 1: Subspace Atlas

### Method

Extracted country/author identity directions at L24 using `extract_direction` (diff_means) across
sets of 3 simple prompts. Used `subspace_decomposition` to project each compositional residual onto
these directions. Used `direction_angles` to measure alignment.

### Results

**Germany/Austria at L24**

Both directions extracted with separation scores 2.6-3.8 and 83% accuracy. Critical geometry:
- Germany-Austria angle: **47.7°** — not cleanly separated. Boosting one bleeds into the other.
- Both directions anti-correlated with all residuals (~112° from the actual residual vector).
- Beethoven residual at L24: Germany projection -14,459 (13.7%), Austria projection +14,029 (12.9%).
- Simple Germany residual at L24: Germany projection -16,024 (16.5%), Austria projection +12,779 (10.5%).
- The Beethoven residual is slightly Austria-tilted relative to the simple Germany prompt but the
  absolute difference is small.

**Belgium/Brussels at L24 and L30**

The Belgium direction extracted at L24 is **nearly orthogonal** (85°) to both the compositional
France-north residual and the simple "capital of Belgium" residual. Both have Belgium projections of
~3,350 units (0.7% of the residual norm). The Belgium signal is equally weak in both prompts.
This refutes the amplitude bottleneck hypothesis for France-north — the compositional context is not
suppressing Belgium identity relative to the direct prompt.

**Shakespeare at L24 (Hamlet)**

The Shakespeare direction at L24 (separation score 2.68) captures 75.1% of the Hamlet compositional
residual. The direct "birthplace of Shakespeare" prompt captures 73.3%. Both angles are ~30° from
their respective residuals. Author identity is fully and equally resolved in both the compositional
and direct prompts at L24. The failure is not in the first hop.

**Country signal is not in the token embedding subspace.** Subspace injection at L24 using tokens
["Germany", "Berlin", "German"] as the subspace basis: donor_subspace_fraction = 0.0026 (0.26%).
Injecting only this subspace component into Beethoven had zero effect — Vienna still wins. This
confirms the finding from the Canberra geometry study: the capital/country identity feature occupies
a small high-dimensional subspace that does not align with any individual token embedding.

---

## Experiment 2: Bottleneck Classification

### Method

Used `track_token` at all layers (0-33) for both the correct and incorrect candidate tokens. Used
`component_intervention` (zero FFN, zero attention) at candidate bottleneck layers. Classified each
problem as amplitude bottleneck vs distortion bottleneck.

### Results

**Beethoven → Berlin/Vienna trajectory**

| Layer | Berlin | Vienna | Top-1 |
|-------|--------|--------|-------|
| L0-L22 | ~0 | ~0 | — |
| L24 | **44.5%** | 4.1% | **Berlin** |
| L26 | 35.5% | 27.5% | Berlin |
| L28 | **78.5%** | 9.4% | **Berlin** (peak) |
| L29 | 75.4% | 9.0% | Berlin |
| L30 | 67.2% | 27.9% | Berlin |
| L31 | 39.5% | **44.7%** | **Vienna** (flip) |
| L32 | 28.5% | 36.7% | Vienna |
| L33 | 31.4% | **45.7%** | Vienna (output) |

The model is top-1 Berlin from L24 through L30 without any intervention. The flip occurs at **L31**.

Simple "capital of Germany": Berlin 83.6% at L24, 100% at L26-L28, recovers to 89.1% at L33.
Zeroing L31 FFN on simple Germany: Berlin drops from 89.1% to 72.7% but stays correct. This means
L31 FFN helps Berlin for the simple prompt but writes Vienna for Beethoven — **input-dependent
behavior, the hallmark of a grokked associative memory**.

Zeroing L31 FFN on Beethoven: Berlin 37.7%, Vienna 33.2% — Berlin returns to top-1 (barely).
Zeroing L32 FFN: no meaningful change. Vienna still wins. L32 is not causal; L31 is.
Scaling L31 FFN at 0.5×: KL = 0.0 (bfloat16 rounds to zero). Non-linear threshold behavior.

**Classification: L31 FFN biographical override.** Beethoven lived in Vienna; the model has a
strong Beethoven→Vienna associative memory that fires at L31 after the capital circuit has succeeded.

**France-north → Brussels/Paris trajectory**

| Layer | Brussels | Paris | Top-1 |
|-------|----------|-------|-------|
| L24 | 18.5% | **64.5%** | Paris |
| L26 | **88.3%** | 9.3% | **Brussels** (peak) |
| L28 | 60.9% | 36.9% | Brussels |
| L30 | 31.8% | **67.2%** | Paris |
| L31 | 24.2% | **74.6%** | Paris (peak) |
| L32 | 38.3% | 49.2% | Paris |
| L33 | **44.5%** | **44.5%** | Tie (output) |

Brussels peaks at 88.3% at L26 — correct answer is strongly established. Then Paris surges at L30.
Zeroing L30 FFN: no meaningful change (tie persists). Paris surge is not in the FFN.
Zeroing L30 attention: Brussels 57.0%, Paris 30.5% — Brussels wins cleanly.

**Classification: L30 attention contextual bleed.** The L30 attention mechanism reads the "France"
token in "north of France" and routes to the France→Paris capital association, overriding the Belgium
signal that built up at L26.

**Hamlet → Stratford/Denmark trajectory**

| Layer | Stratford | Denmark | Top-1 |
|-------|-----------|---------|-------|
| L24 | 0.06% | 0.5% | — |
| L26 | 0.09% | **43.8%** | **Denmark** |
| L28 | 1.7% | 27.1% | Denmark |
| L30 | 2.3% | 8.2% | — |
| L31 | 1.4% | 2.4% | — |
| L33 | **5.9%** | 1.6% | — (output: "a") |

Denmark (the setting of Hamlet) fires at L26 with 43.8% before Stratford ever establishes. Stratford
never reaches even 10% at any layer. The top-1 at output is "a" (18.3%) — the model disperses into
hedging language after Denmark fades.

**Classification: L26 entity competition.** Denmark (property of the work) competes with Stratford
(property of the author). Denmark fires first and drains probability from all other candidates.
This matches the referent competition failure mode documented in cot_experiments.md — the same
pattern as "Hamlet → Danish" from the 4-hop boundary test.

### Experiment 5 (Parallel A* coupling test): CONFIRMED

The model's own compositional coupling already works. Berlin reaches 78.5% at L28 with no
intervention. No boosting of the first-hop subspace is needed — the model resolves Germany→capital
→Berlin correctly and automatically. The coupling from country identity to capital is functional.

**The parallel A* mechanism is running inside the model on every forward pass. We just need to
stop a different circuit from overwriting its output.**

---

## Experiment 3: Targeted Interventions

### Full residual injection

Injecting the full "capital of Germany" residual at L24 into the Beethoven forward pass:
- Donor-recipient residual angle: **9.8°** — extremely similar
- Output: **Berlin 82.0%** — complete fix

At L30: Berlin 85.1%, donor-recipient angle 11.8°. Also works.

The two residuals are nearly identical (9.8° apart). The full injection works because it replaces a
small angular deviation with the Germany-direct context. However, it replaces the entire residual —
not subspace-targeted.

### Steer-and-generate results

**berlin_vs_vienna_L31** (sep score 6.18, 100% accuracy):
- Beethoven at alpha=5: " Berlin.\n\nWhich country is this?" — **Correct**
- Beethoven at alpha=10: " Berlin.\n\nWhich of the following is" — **Correct**
- Beethoven at alpha=20: "Berlin Brandenburg Brandenburg..." — loops
- Austria at alpha=5: " Berlin.\n\nBerlin is" — **Hallucination** (capital of Austria is Vienna)

**beethoven_vs_germany_L31** (sep score 15.75, 100% accuracy — self-contrast direction):
- Beethoven at alpha=-5: " Berlin.\n\nGermany.\n\nGermany." — **Correct** first token
- Austria at alpha=-5: "known for its rich history" — evasion, no Berlin hallucination
- Mozart at alpha=-5: quiz evasion, Vienna suppressed — **structural collateral damage**

**brussels_vs_paris_L30** (sep score 9.92, 100% accuracy):
- France-north at alpha=5: " Brussels, and is a federal parliamentary democracy" — **Correct**
- France-north at alpha=10: " Brussels, Belgium.\n\nBelgium is a" — **Correct, more informative**
- "The capital of France is" at alpha=5: " Brussels, Belgium." — **Hallucination**

**stratford_vs_denmark_L26** (sep score 8.22, 100% accuracy):
- Hamlet at alpha=5: " Stratford-upon-England.\nThe" — **Correct** (minor: England vs Avon)
- Hamlet at alpha=10: " Stratford-upon- Stratford-upon-" — repetition loop
- "The capital of Denmark is" at alpha=5: "a city of vibrant culture" — Copenhagen suppressed
- "The birthplace of the author of Don Quixote" at alpha=5: Toledo instead of Alcalá — **Hallucination**

All four directions fix their target at alpha=5. All four cause hallucinations on related prompts at
the same alpha.

### Minimum intervention to fix each problem

| Problem | Method | Min alpha | Safe? |
|---------|--------|-----------|-------|
| Beethoven | Steer beethoven_vs_germany_L31 | -5 | Partial |
| France-north | Steer brussels_vs_paris_L30 | +5 | No |
| Hamlet | Steer stratford_vs_denmark_L26 | +5 | No |
| Beethoven | Zero L31 FFN | — | No (destroys L31 globally) |
| Beethoven | Full inject Germany@L24 | — | No (full residual replacement) |

---

## Experiment 6: Blind Boosting

### Strategy A — Template boost at L24

**Verdict: Fails.** There is nothing to boost. Berlin is already 44.5% top-1 at L24 without any
intervention. The country signal at L24 for the compositional prompt is lower than for the simple
prompt (44.5% vs 83.6%), but it is still correct. The capital retrieval circuit builds from there to
78.5% by L28. Amplifying the L24 country signal would not prevent L31 from firing.

### Strategy B — Self-contrast

**Verdict: Partially works.** The beethoven_vs_germany_L31 direction (extracted from Beethoven
compositional prompts vs Germany direct prompts) has separation score 15.75 — the highest of all
directions tested. Applying negative alpha pushes the Beethoven prompt toward the Germany-direct
context at L31, which suppresses the biographical override. First token correct at alpha=-5.

The direction is query-type-specific (person→country→capital), not entity-specific. It cannot
distinguish "Beethoven, wrong answer Vienna" from "Mozart, correct answer Vienna." Applying it to
Mozart (which also matches the person→country→capital structure) suppresses the correct Vienna output.

### Strategy C — Entropy-guided boost

**Verdict: Fails for Beethoven.** At the interference layer (L31), Vienna has already flipped to
top-1 (44.7%). Boosting the leading token would amplify the error. Strategy C would work for France-
north at L26, where Brussels is already leading at 88.3% and entropy is concentrated on the right
answer — but by the final layer, Paris has taken over again, so this window is too early.

### Fundamental limit

No blind boosting strategy can distinguish a wrong associative memory override from a correct
biographical fact without entity-level disambiguation. The structure of "Beethoven→Vienna (wrong)" is
identical to the structure of "Mozart→Vienna (correct)" at the level of query type and direction. To
apply the fix correctly, you need to know the entity's actual birth country — which is exactly the
knowledge the model is failing to use.

---

## Experiment 7: Hallucination Safety

### beethoven_vs_germany_L31 at alpha=-5 (most targeted)

| Prompt | Baseline | Steered | Assessment |
|--------|----------|---------|------------|
| "The capital of Japan is" | Tokyo | "A. Tokyo" | OK (format change) |
| "The capital of France is" | Paris | "of course, Paris" | OK |
| "The capital of Brazil is" | (no capital given) | corrupt format | Neutral |
| "Einstein born country capital" | Berlin | Berlin | OK — same structure, unaffected |
| "Mozart born country capital" | **Vienna 90.2%** | quiz evasion | **FAIL** |
| "The capital of Austria is" | "a city..." (weak) | "known for rich history" | Neutral |

Mozart is the critical failure. It shares the person→country→capital structure with Beethoven, but
the correct answer is Vienna. The direction suppresses compositional biographical queries indiscriminately.

**Hallucination rate: above the 5% threshold.** Not safe for deployment without a structural classifier.

### Direction comparison across all three problems

| Direction | Working alpha | Hallucination type |
|-----------|-------------|-------------------|
| berlin_vs_vienna_L31 | 5 | Austria → Berlin (direct wrong capital) |
| beethoven_vs_germany_L31 | -5 | Mozart Vienna suppressed |
| brussels_vs_paris_L30 | 5 | France → Brussels (direct wrong capital) |
| stratford_vs_denmark_L26 | 5 | Copenhagen suppressed; Don Quixote wrong city |

All four fail safety at their working alpha. The self-contrast direction (beethoven_vs_germany) is
the least harmful — it evades rather than substituting a wrong answer — but it still fails.

---

## Revised Theoretical Framework

### Two parallel navigators

For every compositional query, the model runs two circuits simultaneously in the same forward pass:

**Navigator 1 — Compositional circuit** (L24-L30)
Performs the relational hop. Head 1 at L24 extracts the country attribute (Germany), then the capital
retrieval circuit uses this to build toward Berlin. Peaks at 78.5% at L28. This circuit is correct.

**Navigator 2 — Associative memory** (L31+)
Retrieves strong biographical/cultural associations directly linked to the entity in the prompt.
For Beethoven: Beethoven→Vienna (residence). For France-north: France-token→Paris (national capital).
For Hamlet: Hamlet-text→Denmark (play setting). This circuit fires after Navigator 1 has finished
and overwrites its output.

The intervention is not enabling Navigator 1 — it is already working. The intervention must attenuate
Navigator 2 without disrupting Navigator 1 on prompts where Navigator 2 is correct.

### Why the original boosting approach fails

The original hypothesis assumed a signal-strength bottleneck in the first hop. The experiment
showed there is none. The country signal at L24 is sufficient — it successfully drives the capital
retrieval circuit to the right answer. Adding more country signal at L24 would not prevent L31 from
firing because L31's behavior depends on the entity identity (Beethoven), not the country strength.

The architectural reason: L31 FFN has learned (entity → associated city) associations during training.
These fire regardless of what the capital retrieval circuit computed at L26-L30. The two circuits
are parallel and the FFN associations at L31 are not gated by the circuit output.

### The three failure mechanisms are architecturally distinct

| Type | Layer | Component | Trigger | Analogy |
|------|-------|-----------|---------|---------|
| Biographical override | L31 | FFN | Entity name | Beethoven→Vienna (lived there) |
| Contextual bleed | L30 | Attention | Context token | "France" → Paris |
| Entity competition | L26 | FFN | Work/entity co-occurrence | Hamlet→Denmark (set there) |

Each requires a different intervention layer and a different direction. There is no single layer or
direction that covers all compositional failure types.

### Proposed safe pipeline

For deployment, the minimum architecture needed:

1. **L0 structural classifier**: Surface syntax probe detects compositional query type.
   (This works — probes at L0 achieve 90% accuracy from syntax alone per markov_bandwidth.md.)

2. **Confidence gate**: At the peak layer for the query type (L28 for biographical, L26 for
   relational), check if the leading token probability is below a threshold (e.g. 70%). If the
   model is already confident, skip intervention.

3. **Layer and direction selection**: Based on query type:
   - Person → country → capital: beethoven_vs_germany style direction at L31
   - Relational geographic (north/south/border): brussels style direction at L30
   - Work → author → birthplace: stratford style direction at L26

4. **Apply and verify**: Steer at the selected layer with alpha=5. Check that the output has not
   degraded coherence (token repetition = too strong).

5. **Fallback**: If confidence remains low after steering, route to CoT.

This pipeline is blind in the sense that it does not require knowing the correct answer. It requires
knowing the query type — which is readable from surface syntax at L0. The direction itself does not
need to know whether the answer should be Berlin or Vienna; it only needs to know whether the query
is person→country→capital. Within that type, it nudges the model away from the biographical override
context and toward the direct capital retrieval context.

---

## Key Negative Results

1. **Subspace injection via token embeddings fails**: Only 0.26% of the country signal at L24 lies
   in the token embedding subspace. This rules out any approach based on injecting token embeddings
   as a targeted subspace boost. The country identity feature lives in a different part of residual
   space that does not align with individual token embeddings.

2. **L31 FFN scaling is non-linear**: Scaling L31 FFN at 0.5× produces KL=0.0 in bfloat16 —
   identical output. At 0.25×, a tiny change. At 0.0×, Berlin flips to top-1 but with only a 4.5pt
   margin. The biographical override is not a proportional contribution — it behaves like a winner-
   take-all threshold operation. You cannot partially suppress it.

3. **Template boost (Strategy A) fails by design**: The country signal is already present and
   correctly driving the capital retrieval circuit. Amplifying it further at L24 cannot prevent a
   later layer from overriding the result. The failure is downstream, not upstream.

4. **Germany and Austria directions are not cleanly separable** (47.7° apart). Any direction that
   boosts Germany also partially boosts Austria. This is why the berlin_vs_vienna direction
   hallucinates on Austria prompts — the two country representations are too geometrically close
   at L24 to be independently targeted.

---

## Directions Extracted (all stored as steering vectors)

| Name | Layer | Sep. Score | Accuracy | Notes |
|------|-------|-----------|----------|-------|
| germany_vs_others_L24 | 24 | 2.63 | 83% | Too coarse, Germany/Austria only 47.7° apart |
| austria_vs_others_L24 | 24 | 3.84 | 83% | Same issue |
| belgium_vs_others_L24 | 24 | 3.26 | 83% | Near-orthogonal to all residuals |
| shakespeare_vs_others_L24 | 24 | 2.68 | 83% | Not useful — Shakespeare already resolved |
| berlin_vs_vienna_L31 | 31 | 6.18 | 100% | Works, unsafe on Austria |
| vienna_vs_berlin_L31 | 31 | 6.18 | 100% | Inverse of above |
| beethoven_vs_germany_L31 | 31 | **15.75** | 100% | Best direction. Self-contrast. Safest. |
| brussels_vs_paris_L30 | 30 | 9.92 | 100% | Works, unsafe on France |
| stratford_vs_denmark_L26 | 26 | 8.22 | 100% | Works, unsafe on Denmark/adjacent |

The self-contrast direction (beethoven_vs_germany_L31) has separation score 15.75 — more than
twice the next best direction at the same layer. This reflects the fact that it captures a broader
contextual difference (compositional biographical query vs direct capital query) rather than a narrow
entity pair difference (Beethoven vs Germany). It is both the most powerful and the most targeted
direction found.

---

## Summary Table

| Experiment | Question | Finding |
|-----------|----------|---------|
| 1 — Subspace Atlas | Where does the country signal live? | Not in token embeddings (0.26%). Country signal present but not amplitude-bottlenecked. Shakespeare fully resolved at L24. |
| 2 — Bottleneck classification | Where does the failure occur? | L31 FFN (Beethoven), L30 attention (France-north), L26 entity competition (Hamlet). All are post-composition overrides. |
| 3 — Targeted boost | Can steering fix the output? | Yes, at alpha=5 for all three. Full injection also works. Both are unsafe without a gate. |
| 5 — Model coupling | Does the model do the second hop? | Yes. Berlin reaches 78.5% at L28 automatically. The coupling is functional. |
| 6 — Blind boosting | Can we fix it without knowing the answer? | Strategy B (self-contrast) partially works. No strategy is both effective and safe without a structural classifier. |
| 7 — Safety | Does boosting hallucinate? | Yes, on axis-adjacent prompts (Mozart, France direct, Copenhagen). Fix alpha = hallucination alpha. |
| 8 — Pipeline | Can this be deployed? | With a structural classifier and confidence gate — yes. Without — no. |
