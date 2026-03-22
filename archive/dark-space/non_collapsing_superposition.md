# Non-Collapsing Superposition: Residual Recirculation

**Experiment ID:** `e70a5b16-fec2-4e15-a61d-21b60ff9dbf4`
**Model:** `google/gemma-3-4b-it` — 34 layers, 2560 hidden dim, 8 heads, bfloat16
**Date:** 2026-03-07

---

## Hypothesis

A forward pass through a transformer leaves the superposition of competing answers unresolved at L33. Instead of collapsing to a token and paying the cost of an embedding lookup, take the L33 residual — the pre-observation quantum state — and inject it directly at L24, where the compositional circuit begins. Run L24-L33 again. Derived answers get reinforced. Associative shortcuts don't compound. Observe only when the superposition has resolved.

The two test cases:

- **France-North**: "The country directly north of France has its capital in the city of" → Brussels 60.6% vs Paris 36.8% at L33, with Brussels peaking at 79.5% at L26.
- **Einstein**: "The physicist Einstein was born in the capital city of" → Ulm 51.1% at L33, but Germany 97.5% at L28 before a catastrophic single-layer override.

---

## A Necessary Preamble: Tool Limitation

The experiment as designed requires injecting a prompt's L33 residual *back at L24* — taking a later layer's output and feeding it into an earlier input. `inject_residual` cannot do this. The tool captures the donor's residual **at the injection layer** (running L0→L on the donor), then continues from L+1. Injecting a prompt into itself at any layer is always identity by construction. There is no time-travel in a forward-only tool.

This is not a workaround failure. It is a genuine architectural constraint. True L33→L24 recirculation would require a `donor_layer` parameter distinct from the `injection_layer` — the ability to say "capture from layer 33, inject at layer 24." The tool has neither.

What follows uses the closest achievable approximation throughout: a **strong-signal donor** whose last-position residual at layer L is injected into the recipient's last-position residual at the same layer L. This tests the same underlying question — does providing a richer, more answer-committed state at the start of the compositional circuit produce a better final answer? — but from a different donor rather than a time-shifted self.

---

## Experiment 1 — Baseline Superposition Geometry

### Baselines

| Prompt | Top-1 | Top-1 % | Top-2 | Top-2 % |
|--------|-------|---------|-------|---------|
| France-North | Brussels | 60.6% | Paris | 36.8% |
| Einstein | Ulm | 51.2% | Germany | 35.2% |
| Capital of Germany | Berlin | 89.1% | — | — |
| Capital of France | Paris | 80.9% | — | — |

### France-North: The Complete Oscillation Map

Layer-by-layer logit lens across L24-L33, both competing tokens:

| Layer | Brussels | Paris | Top-1 | Event |
|-------|---------|-------|-------|-------|
| L24 | 51.6% | 40.0% | Brussels | Compositional circuit begins |
| **L25** | **7.4%** | **90.2%** | **Paris** | Attention assault — "France" in prompt fires |
| **L26** | **79.7%** | 20.1% | **Brussels** | L26 FFN counterattack — compositional peak |
| L27 | 62.1% | 37.7% | Brussels | Decay begins |
| L28 | 52.7% | 46.5% | Brussels | Near-parity |
| L29 | 58.9% | 40.6% | Brussels | Brief recovery |
| **L30** | 34.8% | **64.8%** | **Paris** | Contextual bleed from "France" in prompt |
| **L31** | 26.8% | **72.7%** | **Paris** | Paris peak |
| L32 | 51.6% | 40.0% | Brussels | Brussels recovery |
| L33 | 60.5% | 36.8% | Brussels | Final output |

Three separate battles inside one forward pass:

1. **L25**: Attention fires on "France" in the prompt tokens, wiping Brussels from 51.6% to 7.4%. Paris surges to 90.2%.
2. **L26**: The L26 FFN (capital fact store) counterattacks, pushing Brussels back to 79.7%. Paris collapses to 20.1%.
3. **L30-31**: Contextual bleed reasserts. "France" in the prompt causes a second associative surge; Paris peaks at 72.7% at L31. Brussels bottoms at 26.8%.
4. **L32-33**: A second Brussels recovery brings the final output to 60.5%.

The final answer is correct — Brussels wins — but only after losing *twice*. Each win and loss is driven by fixed weights responding to fixed prompt structure.

### Einstein: Explosive Single-Layer Override

Germany and Ulm across L24-L33:

| Layer | Germany | Ulm | Ulm rank | Event |
|-------|---------|-----|----------|-------|
| L24 | 57.0% | ~0% | 573 | Germany dominant from the start |
| L25 | 70.7% | ~0% | 626 | Germany strengthens |
| L26 | 83.2% | ~0% | 180 | FFN amplifies Germany |
| L27 | 96.9% | ~0% | 206 | Near-certainty |
| L28 | **97.6%** | ~0% | 175 | **Peak** |
| L30 | **98.4%** | ~0% | 113 | Maximum confidence |
| L31 | 98.4% | 0.001% | 10 | Ulm barely appears |
| L32 | 71.5% | 0.07% | 6 | Germany drops; Ulm rising |
| **L33** | 35.2% | **51.2%** | **0** | **Ulm erupts from nothing** |

Ulm is rank 500+ through L30 — not merely weak, genuinely absent. Its subspace projection at L28 is **negative** (−297): Ulm actively points away from the residual direction. Then at L32-L33, something writes it with massive force. Germany drops from 98.4% to 35.2% in two layers while Ulm materialises at 51.2%.

This is qualitatively different from France-North. There is no competition. There is a clean, stable computation (Germany 97%+ for 7 consecutive layers) followed by a catastrophic single-layer override.

### Geometry

**Near-total perpendicularity.** Both L24 and L33 residuals are nearly perpendicular to all answer tokens. For France-North:

- Brussels: 86.7° from L24 residual, 86.3° from L33 residual
- Paris: 88.4° from L24 residual, 88.0° from L33 residual

The entire Brussels/Paris competition lives in less than 2° of angular difference between tokens that are themselves 67.6° apart. The answer is encoded in tiny angular variations in a high-dimensional space, not in large directional movements.

**Subspace sizes.** The answer subspace is a vanishingly small fraction of the total residual:

| Prompt | Layer | Dominant answer projection | Fraction of residual |
|--------|-------|--------------------------|---------------------|
| France-North | L26 (peak) | Brussels: 2990 | 0.57% |
| France-North | L33 (final) | Brussels: 2783 | 0.43% |
| Einstein | L28 (peak) | Germany: 2903 | 0.26% |
| Einstein | L33 (final) | Germany: 2039 / Ulm: 1349 | 0.17% |

Brussels component shrinks 7% from L26 to L33. Paris shrinks 26%. The Brussels/Paris projection ratio *improves* (5.2→6.5) even as both decay — the L30-31 Paris surge leaves a residue that L32-33 partially cleans, but not fully.

At L28, Ulm's projection is **−297** (negative). By L33 it is **+1349**. The L33 FFN wrote +1646 units of Ulm projection from nothing, while simultaneously subtracting 864 units from Germany.

**Inter-prompt geometry.** At L24, all prompts are cosine ≥0.985 to each other (centroid distance 0.010) — roughly 10° apart, essentially the same region of space. By L33, france-north has separated to cosine 0.96-0.97 from control prompts (centroid 0.027). The L33 residual norms:

- France-North: L24=35,344, L33=42,572 — a 20% norm growth
- Einstein: L24 lower, L28=57,167, L33=59,711

The L33 residual is geometrically *within distribution* of L24 space. Injection compatibility is not the limiting factor for why the experiment is hard. The tool limitation is the limiting factor.

---

## Experiment 2 — Injection Approximation

### Design

Since true L33→L24 injection is not achievable, the approximation uses:

- **France-North**: donor = "The capital of Belgium is" → recipient = france-north, at layers L24/L26/L28/L30/L32
- **Einstein**: donor = "The capital of Germany is" → recipient = einstein, at layers L28/L32

Only the **last position** is replaced (patch_all_positions=False). The other positions retain the recipient's own residuals. The donor's last-position residual at layer L carries the strong correct-answer signal that the L33→L24 loop was supposed to provide.

### France-North Results

Residual angles between donor (Belgium) and recipient (france-north) at each injection layer: 9.9° (L24), 11.6° (L26), 11.3° (L28), 13.2° (L30), 13.7° (L32). Close enough to be in-distribution. The question is whether this small angular difference changes the output.

| Inject at | Brussels % | Paris % | Delta Brussels | KL(donor→injected) |
|-----------|-----------|---------|---------------|-------------------|
| Baseline | 60.6% | 36.8% | — | — |
| L24 | 62.8% | absent | +2.2pp | 0.017 |
| L26 | 60.9% | absent | +0.3pp | 0.012 |
| L28 | 61.2% | absent | +0.6pp | 0.009 |
| **L30** | **65.1%** | absent | **+4.5pp** | 0.016 |
| L32 | 59.1% | absent | −1.5pp | 0.002 |

Brussels improves modestly (0.3-4.5pp depending on layer). But **Paris always disappears**, replaced by "a" (23-28% as rank 2). The injected output matches the donor's distribution closely (KL < 0.02), not a blend of donor and recipient.

The injection does not reinforce Brussels *within* the compositional register. It transplants the Belgium **fact-completion register** — the distribution that follows "The capital of Belgium is" looks like "Brussels / a / the / one / known…", not like the question-answering distribution france-north produces. The genre changes, not just the answer.

The best injection point is L30 (+4.5pp), precisely where the Paris surge peaks — not L24 (the compositional circuit entry). This suggests the effective mechanism is **counter-signal at the interference layer**, not head-start at the circuit origin. Injecting at L32 (just before the final Brussels recovery) actually hurts (−1.5pp), because it overwrites the recovery before it can happen.

### Einstein Results

Residual angles: L28=16°, L32=19.4°. Still within distribution.

| Inject at | Top-1 | Result | KL(donor→injected) |
|-----------|-------|--------|-------------------|
| Baseline | Ulm | 51.2% Ulm, 35.2% Germany | — |
| **L28** | **Berlin** | **85.6% Berlin** — Ulm absent | 0.007 |
| **L32** | **Berlin** | **86.1% Berlin** — Ulm absent | 0.004 |

Complete Ulm suppression at both injection points. Even at L32, just one layer before the eruption, providing a strong German-capital signal fully prevents the L33 Ulm-writing FFN from triggering. The donor output is Berlin (the capital of Germany), not Germany — the fact-completion frame takes over.

Since Ulm exists only in L33, any strong-signal injection at L28-L32 can prevent it. The Ulm confusion is not embedded in the residual through the main computation. It requires active L33 FFN writing, and that writing can be blocked with a late injection.

---

## Experiment 4 — Inside the Loop

The layer-by-layer trajectory from Experiment 1 *is* the trajectory a recirculation loop would follow. The within-loop computation uses the same weights responding to the same prompt structure. The key interference events:

- **L25**: fires 90.2% Paris because the prompt contains the token "France." This does not depend on the incoming residual state at L24 — it depends on the key-value representations of "France" in the earlier positions, which are fixed by the prompt.
- **L30-31**: fires 72.7% Paris for the same reason — contextual bleed from the prompt token "France."

Both interference events are prompt-driven, not residual-driven. A recirculation loop that feeds the L33 residual back into L24 would encounter the *same* prompt-driven interference at L25 and L30. The oscillation would repeat with slightly different magnitudes but the same qualitative structure.

This is the core reason why the loop is not transformative for France-North: the interference is structural, encoded in the prompt's own tokens. Recirculation cannot remove the token "France" from the prompt.

---

## Experiment 5 — The Answer Subspace Through the Loop

From the subspace decompositions:

**France-North:**

| Layer | Brussels projection | Brussels % of residual | Paris projection | Paris % of residual |
|-------|--------------------|-----------------------|-----------------|---------------------|
| L26 | 2990 | 0.572% | 577 | 0.021% |
| L33 | 2783 | 0.427% | 429 | 0.010% |

The Brussels component shrinks from L26 (peak) to L33 (output). If the loop carried the L33 state back to L24, the starting Brussels projection would be 2783 — lower than the L24 baseline of ~2009. But the L26 FFN would fire again and push it back toward 2990. Whether the final L33 state after one loop cycle ends higher or lower than 60.5% depends on whether the L30-31 Paris surge, operating on the new baseline, takes more or less from Brussels than the first time.

Given the prompt-driven nature of the L30-31 surge (independent of incoming residual), the second loop would likely produce a similar Paris surge, leaving Brussels in the 62-67% range. Marginal improvement, not resolution.

**Einstein:**

| Layer | Germany projection | Ulm projection |
|-------|-------------------|----------------|
| L28 | +2903 | **−297** (negative) |
| L33 | +2039 | +1349 |

The loop would feed the L33 state (Germany +2039, Ulm +1349) back into L24. The circuit would then run for 7 layers dominated by Germany (as in the original pass). When it reaches L33 again, the incoming Germany signal would be stronger (the circuit had a stronger start), and the Ulm-writing FFN would be less able to override a higher Germany projection. The loop plausibly produces Germany > Ulm in the second iteration — perhaps decisively.

---

## Experiment 7 — Controls and Safety

### The Safety Failure

Single-position last-token injection at L24 is **not safe** for already-correct prompts:

| Test | Baseline | After injection | Verdict |
|------|---------|----------------|---------|
| Belgium "is" → Germany "is" at L24 | Berlin 89.1% | **Brussels 58.7%** | DESTROYED |
| Belgium "is" → France "is" at L24 | Paris 81.0% | **Brussels 59.9%** | DESTROYED |

Residual angles between donor and recipient: 6.4° (Belgium→Germany) and 6.8° (Belgium→France). A 6-7° angular perturbation — smaller than the 10° typical gap at L24 — is sufficient to completely flip high-confidence correct answers. Berlin 89.1% becomes Brussels 58.7%.

The last-position residual at L24 has **dominant causal influence** on the final output. The other 5 positions of "The capital of Germany is" carry strong Germany context through the key-value cache, but they cannot overcome a 6° perturbation at the last position. This is consistent with the L24 Head 1 attention pattern: it fires 59.8% on the country/topic token and writes that information into the last position, making the last position the primary carrier of the answer decision by L24.

Any pipeline applying L24 last-position injection must therefore be restricted exclusively to prompts identified as compositional failures. Applying it to simple facts reliably destroys the answer.

### Cross-Contamination: The Amplifier Effect

Injecting france-north's ambiguous L24 last-position residual (Brussels 60.6%, Paris 36.8%) into "The capital of Germany is" at L24:

- **Injected output: Brussels 85.3%, Paris 7.9%, Berlin 1.6%**
- KL(france-north→injected) = 0.34 — the Germany template does substantial extra work
- Residual angle: 9.8° between the two prompts at L24

The germany-simple template amplifies the ambiguous 60/37 Brussels/Paris signal to 85/8. The simple-fact template's "answer slot" expectations concentrate probability onto the dominant competing token. The output is stronger Brussels than france-north produces from its own computation.

This is an unexpected finding: **simple-fact templates act as probability concentrators for compositional residuals**. The compositional signal, run through 5-6 more layers of a clean fact-lookup template, gains confidence in its dominant component. This could form the basis of a different correction mechanism — route the ambiguous residual through a high-confidence template to concentrate probability, then observe — though this requires generating a token and reformulating the query.

---

## Synthesis

### What the Hypothesis Predicted

The L33 residual contains the superposition of Brussels and Paris, with Brussels slightly ahead. Feeding it back to L24 gives the compositional circuit (L24-L28) a head start. L26 FFN reinforces Brussels again. The L30-31 Paris surge, operating on a stronger Brussels baseline, takes less. Each iteration strengthens Brussels. Entropy drops. The superposition resolves.

### What the Evidence Shows

**The oscillation is weight-baked, not residual-driven.**

The L25 and L30-31 Paris surges fire from the prompt's own "France" tokens, not from the state of the last-position residual. No matter what state is fed into L24, those interference events fire from the same fixed tokens. Recirculation through the same weights sees the same oscillation with slightly different amplitudes.

The loop would improve france-north by an estimated 3-8pp (consistent with the 2-5pp observed from the injection approximation). It would not resolve the superposition to >80% Brussels. That resolution requires either removing the interfering "France" tokens (CoT, different framing), surgical subspace correction at L28, or observation at L26 before the interference.

**The two failure types are not equivalent.**

For france-north (two-sided competition), the loop is marginally helpful. For einstein (late-layer override), a single targeted injection is already sufficient and the loop is unnecessary.

**The L33 residual is not the right feedback signal for france-north.**

The L33 residual already reflects the outcome of the L30-31 interference. Feeding it back into L24 means the second iteration starts from a state that carries traces of the Paris surge that followed the L26 peak. Starting from the L26 state — the peak Brussels, before the interference — would be more useful. But that is early exit (observe at L26), not recirculation.

---

## Revised Intervention Taxonomy

| Failure type | Signature | Best intervention | Compute cost |
|-------------|-----------|-----------------|-------------|
| Late override (Einstein/Ulm) | Answer rank 500+ through L31, erupts at L32-L33 | Inject at L31-32 using pre-override residual | +2 layers |
| Two-sided competition (France-North) | Both answers compete throughout L24-L33; L26 is the peak | Early exit at L26 | −7 layers |
| Distributed/redundant retrieval | Correct answer dominant throughout; no peak-then-decay | No intervention needed | 0 |
| Template competition | Circuit succeeds at L26-28; template overwrites at L30+ | CoT with geo/naming template | +374 layers |

**Early exit at L26 beats recirculation for France-North.** Brussels at 79.7% is the model's best answer, before the structural interference. The task is to observe *before* the interference, not loop after it.

**Targeted late injection beats recirculation for Einstein.** The Ulm override is blockable at L32 with a single injection. No loop required.

**The loop's real value** would emerge in failures where: (a) the late override layer is unknown or variable, (b) the pre-override state is not accessible via logit lens, or (c) the model is large enough that layer-by-layer monitoring is expensive. In those cases, feeding the final state back through the compositional layers could suppress spurious late writes without knowing exactly which layer to target.

---

## Key Numbers

| Measurement | Value |
|-------------|-------|
| France-North L25 Paris peak | 90.2% |
| France-North L26 Brussels peak | 79.7% |
| France-North L31 Paris peak | 72.7% |
| France-North L33 final | Brussels 60.5%, Paris 36.8% |
| France-North answer subspace at L26 | 0.57% of residual |
| France-North answer subspace at L33 | 0.44% of residual |
| Einstein Germany peak | 98.4% (L30-L31) |
| Einstein Ulm rank through L30 | 500+ |
| Einstein Ulm at L33 | 51.2% (from rank 500+ to top-1 in 3 layers) |
| Einstein Ulm subspace projection at L28 | −297 (negative) |
| Best injection improvement (France-North) | +4.5pp Brussels (L30 injection) |
| Einstein after L28 injection | Berlin 85.6% — complete Ulm suppression |
| Einstein after L32 injection | Berlin 86.1% — complete Ulm suppression |
| Safety threshold | 6° perturbation at L24 last position destroys correct answers |
| Cross-contamination amplifier | 60/37 → 85/8 Brussels/Paris when compositional residual runs through simple-fact template |
| Inter-prompt cosine at L24 | ≥0.985 (≤10° between all prompts) |
| Inter-prompt cosine at L33 | 0.96-0.97 (15-19° between france-north and controls) |
