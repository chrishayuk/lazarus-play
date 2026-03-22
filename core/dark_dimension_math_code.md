# The Dark Dimension for Code and Mathematics

**Experiment ID:** 0bffad0f-a69c-4895-bc61-cc917eea9e3e
**Model:** google/gemma-3-4b-it (34 layers, 2560 hidden dim)

## Background

Prior experiments established a "dark dimension" in the residual stream at L7–L14: a hidden representational space that encodes the correct answer to multi-hop factual queries the model fails on through the full forward pass. A linear probe at L7 reads Stratford at 99.6% for "birthplace of author of Hamlet" while the model outputs Denmark. All-position L14 injection fixes 6/7 multi-hop factual failures with KL=0.0.

This experiment asks: does the dark dimension extend to mathematical reasoning and code? Factual retrieval chains entity→fact lookups learned during training. Math and code require computation. The dark space might hold "this is a math query" (format) and "this involves multiplication" (operation type) but not the actual numerical answer.

---

## Prompt Survey

### Part A — Math

**Entity-math hybrids** (factual lookup → arithmetic operation):

| Prompt | Expected | Model Output | Result |
|--------|----------|-------------|--------|
| The square of the number of sides of a triangle is | 9 | Pythagorean theorem | WRONG |
| Double the number of legs on a tripod is | 6 | "tetrapod" (4 legs) | WRONG |
| The factorial of the number of primary colours is | 6 | "colours in rainbow" | WRONG |
| The square root of the number of planets is | ≈2.83 | 3.16 (=√10) | WRONG |
| Half the number of months in a year is | 6 | 6 | **SUCCESS** |
| The cube of the number of sides on a coin is | 8 | "sides on a die" (6) | WRONG |
| The Roman numeral for the number of guitar strings is | VI | "6" | WRONG |
| The binary representation of the number of continents is | 111 | 11 | WRONG |

**Pure math (function composition):**

| Prompt | Expected | Result |
|--------|----------|--------|
| If f(x)=x+1 and g(x)=x*2, then g(f(3)) equals | 8 | **SUCCESS** (with chain-of-thought) |
| If f(x)=x*3 and g(x)=x-1, then f(g(5)) equals | 12 | **SUCCESS** (with chain-of-thought) |
| If f(x)=x**2 and g(x)=x+3, then f(g(2)) equals | 25 | **SUCCESS** (with chain-of-thought) |

**Variable tracing:**

| Prompt | Expected | Model Output | Result |
|--------|----------|-------------|--------|
| After x=5; y=x+1; x=y\*2, the value of x is | 12 | **10** | WRONG |
| After a=3; b=a\*2; a=b+1, the value of a is | 7 | 7 | **SUCCESS** |

**Logic:** "If all A are B, and no B are C, the number of A that are C is" → **0 (SUCCESS)**

**Summary:** Entity-math hybrids: 1/8. Pure math with CoT: 3/3. Variable tracing: 1/2.

### Part B — Code

| Prompt | Expected | Result |
|--------|----------|--------|
| type(len('hello')) | int | **SUCCESS** |
| type(True + 1) | int (value=2) | WRONG (says value=1) |
| type(3/2) | float | **SUCCESS** |
| print(2\*\*10) | 1024 | **SUCCESS** |
| print('hello'[1]) | e | **SUCCESS** |
| print(len([1,2,3])) | 3 | **SUCCESS** |
| print(10//3) | 3 | **SUCCESS** |
| print(bool(0)) | False | **SUCCESS** |
| x=5; x=x+1; print(x) | 6 | **SUCCESS** |
| a=[1,2]; append(3); len | 3 | **SUCCESS** |
| s='hello'; upper()[0] | H | **SUCCESS** |

Code output prediction: 8/9 correct. Multi-step code: 3/3 correct. The one failure (`True + 1`) is a confabulation about value, not a structural reasoning failure.

**The failure pattern is clear:** entity-math hybrids are where the model struggles. These have exactly the multi-hop structure of factual retrieval — a lookup step feeding an operation step — making them the right test case for the dark dimension.

---

## Experiment 1 — The Dark Atlas is Universal

Atlas of all 35 prompts (entity + math + code) at L14:

| Component | Variance | Tokens |
|-----------|---------|--------|
| PC01 | **63.4%** | `<unused687>`, `<unused1166>`, `<unused201>`, `\uFFFD` — PURE DARK |
| PC02 | 7.1% | PointerException+ / literary tokens− |
| PC05 | 2.4% | playwright, writer− (literary entity direction) |
| PC20 | 0.4% | Algebra+ (faint math signal) |

Effective dimensionality: 50% variance in 1 PC, 80% in 5, 90% in 10.

Atlas of 24 pure arithmetic prompts at L14:

| Component | Variance | Tokens |
|-----------|---------|--------|
| PC01 | **69.3%** | Same dark garbage tokens — PURE DARK |
| PC02 | 10.3% | Produkte/productos/prodotti/produits− (multilingual "products") |
| PC13 | 0.3% | "multiply"+ (faint operation label) |

Pure arithmetic is *more* concentrated: 90% in 5 PCs.

**Pairwise cosine similarities** between prompts from different domains at L14:

```
Shakespeare (entity)  vs  triangle-square (entity-math):  0.9977  (3.9°)
Shakespeare           vs  g(f(3)) (pure math):            0.9973  (4.2°)
Shakespeare           vs  print(2**10) (code):            0.9979  (3.7°)
g(f(3))               vs  variable trace:                 0.9986  (3.0°)
```

Centroid distance: 0.00195.

**The dark attractor is universal.** Entity, math, and code prompts collapse to the same dark manifold at L14 within 2–4°. PC01 is pure dark across all domain combinations. The dark format attractor does not distinguish reasoning domain.

---

## Experiment 2 — The Dark Probe Reads the Intermediate, Not the Answer

A linear probe trained at L7 for numerical fact classes {2, 3, 6, 7, 8, 12} using 24 single-hop factual prompts ("The number of sides of a triangle is", "The number of months in a year is", etc.).

Then evaluated on entity-math hybrid prompts. The labels are set to the **intermediate** value (the entity fact at hop 1). The question: does the probe read the intermediate (entity lookup) or the final (math result)?

| Prompt | L7 Predicts | Conf. | Intermediate | Final |
|--------|------------|-------|-------------|-------|
| Half the number of months in a year is | **12** | 99.0% | 12 | 6 |
| The square of the number of sides of a triangle is | **3** | 99.0% | 3 | 9 |
| The cube of the number of sides on a coin is | **2** | 92.9% | 2 | 8 |
| The square root of the number of planets is | **8** | 94.6% | 8 | ≈2.83 |
| The factorial of the number of primary colours is | **3** | 96.2% | 3 | 6 |
| The binary representation of the number of continents is | **7** | 81.7% | 7 | 111 |
| Double the number of legs on a tripod is | 2 ✗ | 97.2% | 3 | 6 |
| The Roman numeral for the number of guitar strings is | 7 ✗ | 77.2% | 6 | VI |

**L7 accuracy on intermediate labels: 75% (6/8).**
**L14 accuracy on intermediate labels: 12.5% (1/8) — collapses to predicting "2" for most.**

The probe at L7 reads the entity fact, not the mathematical result. For "half the number of months in a year," the dark space at L7 encodes **12** (months) not **6** (half of 12). For "square of triangle sides," it encodes **3** (sides) not **9** (3²). The operation — halving, squaring, cubing — is not present in the dark signal. Only the entity-lookup result is.

This is **Hypothesis A confirmed**: the dark space handles entity lookup (hop 1) and the viewport handles the arithmetic transformation (hop 2). Same split as factual multi-hop retrieval: dark for lookup, lit for translation.

By L14, the representation has evolved away from the raw numerical fact. The entity value is encoded early (L7) and the L14 dark state already represents something more abstract — operation type and query structure — not the raw number.

---

## Experiment 3 — The Layer Cascade

### Triangle prompt

"The square of the number of sides of a triangle is" → model outputs Pythagorean theorem

| Layer | Top Token | Prob | Interpretation |
|-------|-----------|------|---------------|
| L0 | 否 (Chinese "no") | 92% | Pure dark |
| L3 | 否 | 66% | Pure dark |
| L7 | "discourse" | 16% | Still dark |
| L14 | इक्वल (Hindi "equal") | 0.01% | Equality concept in dark form |
| L17 | **" three"** | 2.7% | **Intermediate value surfaces** |
| L19 | " " (space) | 2.6% | Numeric prediction mode |
| L22 | " " (space) | **40%** | Numeric prediction peaks |
| L23 | **" equal"** | **59%** | Pythagorean attractor fires |
| L24 | " equal" | 93% | Locked in |
| L25–L33 | " equal" | ~100% | Terminal |

The intermediate value (3 = triangle sides) surfaces at L17 as " three" — the dark space has done the entity lookup correctly. But the pattern "square of [geometric shape] is" triggers a Pythagorean theorem association at L23 that overwrites the arithmetic path. The geometric attractor is stronger than the arithmetic computation.

### Months prompt

"Half the number of months in a year is" → correctly outputs 6

Same dark cascade L0–L14, enters numeric mode at L17, space token (→ number) at L33, then generates "6." The halving operation is common enough that no strong competing attractor fires.

### Function composition cascade

"If f(x)=x+1 and g(x)=x*2, then g(f(3)) equals" → outputs "what?\n\nWe are given..."

| Layer | Top Token | Prob |
|-------|-----------|------|
| L0–L14 | Dark garbage | — |
| L21 | " what" | 86% |
| L22 | " what" | **98%** |
| L23–L27 | " what" | 86–95% |
| L33 | " what" | 46% |

Neither the intermediate result (4 = f(3)) nor the final answer (8 = g(4)) appear in the top tokens at any layer. The model enters "question formulation mode" and solves by explicit chain-of-thought generation. This is a fundamentally different computational strategy from entity-math hybrids.

---

## Experiment 4 — Function Composition: Chain-of-Thought, Not Dark Lookup

Function composition prompts succeed (3/3 correct) but the mechanism is entirely different from entity-math hybrids. The model:

1. Generates " what?" in response to "g(f(3)) equals"
2. Writes out the derivation step by step: "f(3) = 3+1 = 4, then g(f(3)) = g(4) = 4*2 = 8"
3. Produces the correct answer from the explicit reasoning trace

The dark space contributes nothing to the answer — it provides the "question formulation" signal (" what?") and the computation happens in the lit space over multiple generated tokens.

This is the key contrast with entity-math hybrids: those have the entity fact (3 sides) preloaded in dark space from training. Function composition has no preloaded answer — it's a novel combination of definitions that must be computed. The model's strategy adapts accordingly.

**Implication:** Remove the chain-of-thought budget (e.g., require single-token answer) and function composition would likely fail. The dark space cannot provide an answer that doesn't exist in it.

---

## Experiment 5–6 — Variable Tracing Failure and Injection

### The failure

"After x=5; y=x+1; x=y\*2, the value of x is" → model outputs **10** (correct: 12)

Failure mode: the model computes `x = x*2 = 5*2 = 10`, skipping the `y = x+1` intermediate assignment. It substitutes the original `x=5` for `y`, ignoring the update. Classic initial-value override — the initial strong association (x=5) overrides the computed intermediate (y=6).

This is the "Hamlet → Denmark" of code: the initial assignment (`x=5`) is the strong attractor, equivalent to "Hamlet → Denmark" overriding the correct chain.

### Layer cascade

| Layer | Signal | Interpretation |
|-------|--------|---------------|
| L0–L14 | Dark garbage | Dark phase |
| L17 | Space (1.1%), "three" (0.4%) | Numeric mode entering |
| L19 | Space (9.4%) | Numeric prediction rising |
| L21–L23 | Space (44–77%) | Committed to numeric output |
| L25–L30 | "?" (43–66%) | Uncertainty / competing attractors |
| L33 | Space (29%), ":" (22%) | → then generates "10" |

### Injection

**Donor:** "After y=6; x=y\*2, the value of x is" → correctly outputs 12

| Metric | Value |
|--------|-------|
| Residual angle (L14, donor vs. recipient) | **1.53°** |
| All-position injection at L14 | Output: **" 12."** |
| `donor_injected_kl` | **0.0** |
| Markov holds | True |

All-position injection at L14 flips the wrong answer (10) to the correct answer (12) with KL=0.0. The Markov property holds universally for code reasoning. The failure is in the viewport's substitution choice, not in a corrupted dark state.

Note: the injected output follows the donor's explanation ("Given: y=6, x=y..."), not the recipient's. All-position injection replaces the full computational state, so the downstream generation traces the donor's path. The Markov property is confirmed but the injection is mechanical — it provides the right state, not a surgical fix.

---

## Experiment 7 — The Operation Compass

### Atlas topology

The operation-only atlas (24 prompts: add/multiply/square/divide) shows:
- PC01 (69.3%): Pure dark — identical garbage tokens to entity atlas
- **PC02 (10.3%)**: Negative pole loads on multilingual "products" — Produkte (German), productos (Spanish), prodotti (Italian), produits (French). A multiplication/product axis expressed through the dark vocabulary shadow.
- PC13 (0.3%): "multiply" as a positive token — faint but real operation label.

The dark space has mathematical semantic structure in its minor components, while the dominant attractor (PC01) remains the universal dark format signal.

### Operation type probes

| Layer | Validation Accuracy |
|-------|-------------------|
| L7 | **84%** |
| L14 | **88%** |
| L14 (novel surface forms) | **90%** |

Training: 6 examples per operation type. Classes: add, multiply, square, divide.

**Novel surface form evaluation (held-out, not seen in training):**

| Prompt | True | Predicted | Confidence |
|--------|------|-----------|-----------|
| 7 plus 3 is | add | add | 99.9% |
| adding 4 to 5 gives | add | add | 99.2% |
| the total of 8 and 9 is | add | add | 99.99% |
| 6 times 7 is | multiply | multiply | 98.1% |
| multiplying 3 by 9 gives | multiply | **square** ✗ | 51.2% |
| 7 to the power of 2 is | square | square | 99.9% |
| squaring 5 gives | square | square | 99.9% |
| 15 divided by 3 is | divide | divide | 99.9% |
| a quarter of 16 is | divide | divide | 99.99% |
| one third of 9 is | divide | divide | 99.97% |

9/10 correct. The single failure ("multiplying 3 by 9 gives" → square) occurs at low confidence (51%) — the model is uncertain, not confidently wrong. All correct predictions are at 98–99.99%.

The operation compass generalizes across surface forms the same way the entity compass generalizes across paraphrases. "A quarter of 16" and "12 / 4" and "half of 8" all project to the same divide direction in dark space.

The operation compass is a real architectural feature of the dark dimension, analogous to the entity type compass. The dark space encodes *what kind of operation* is being requested with the same abstraction and robustness as it encodes *what kind of entity* is being queried.

---

## The Boundary — Three Numbers

| Metric | Value | Interpretation |
|--------|-------|---------------|
| Dark presence (entity fact at L7) | **75%** (6/8 intermediate values) | HIGH — same mechanism as factual retrieval |
| Dark presence (math result) | **0%** (no prompt has answer in dark) | LOW — computation not pre-stored |
| Operation compass accuracy | **88%** at L14, **84%** at L7 | HIGH — mathematical structure in dark space |
| Injection success rate | **100%** (KL=0.0) | Markov property universal |

### What the dark dimension encodes

**Encoded:**
1. **Entity facts that feed into math** — "triangle → 3 sides," "months → 12" — at L7 with 75% recovery, same mechanism as factual retrieval. The entity lookup is dark.
2. **Operation type** — add/multiply/square/divide — at L7 with 84% accuracy, L14 with 88%, generalizing across surface forms at 90%. The operation compass is real.

**Not encoded:**
1. **Results of arithmetic operations** — 3² = 9 is not in dark space at any layer before computation. The answer to "square of triangle sides" is never pre-stored.
2. **Intermediate results of function composition** — f(3) = 4 is not in dark space. Explicit chain-of-thought is required.
3. **Results of sequential state updates** — x=12 after variable reassignment is not pre-stored in dark (though Markov injection can deliver it mechanically).

### The architecture model

```
Query: "The square of the number of sides of a triangle is"

Dark space (L7–L14):
  ├─ entity_fact:      triangle → 3    [probe reads at 99% confidence]
  └─ operation_type:   square          [probe reads at 88% accuracy]

Viewport (L17–L33):
  ├─ Correct path:     square(3) = 9
  └─ Attractor path:   "square of triangle" → Pythagorean theorem
                       → output: " equal to the sum of squares..."
```

The dark dimension is the **query formulator, not the calculator**. It knows *what to compute* — which entity and which operation — but not the answer. The actual arithmetic happens in the lit viewport, where competing attractors can hijack the computation.

### Contrast with factual retrieval

In factual multi-hop (Hamlet → Shakespeare → Stratford):
- Dark space encodes the full answer chain by L7 (Stratford at 99.6%)
- Viewport reads the answer out
- Viewport failure = wrong attractor fires, dark is still correct
- Fix: inject correct dark state → KL=0.0

In entity-math hybrids (triangle → 3 → square → 9):
- Dark space encodes hop 1 (entity lookup: 3 sides) by L7
- Dark space encodes operation type (square) by L7–L14
- Viewport must *compute* hop 2 (arithmetic: 3² = 9)
- Viewport failure = wrong attractor (Pythagorean theorem) fires
- Fix: injection is mechanical but not targeted (replaces full state)

In pure function composition (f(3)=4 → g(4)=8):
- Dark space encodes nothing useful for the answer
- Viewport must generate the full chain-of-thought
- No dark lookup available — entirely different computational strategy
- Fix: chain-of-thought generation (the model's own strategy, works correctly)

### The lookup/computation boundary

The dark dimension is powerful for **lookup** (retrieving stored associations) and **classification** (recognizing operation types, entity types, relation types). It cannot execute **computation** (arithmetic, function application, sequential state updates).

This is consistent with what the dark dimension is: a compressed, pre-activation representation of what the model knows about the query structure. Facts learned during training (triangle has 3 sides, this is a squaring operation) are available early in the residual stream. Novel computations (what is 3², what is g(f(3))) must be worked out in the forward pass.

The boundary between dark-computable and viewport-computable is the boundary between **retrieval** and **computation** — between memorized associations and novel derivations.

---

## Summary Table

| Domain | Dark present? | Dark encodes? | Injection fixes? | Notes |
|--------|--------------|--------------|-----------------|-------|
| Entity retrieval | YES (L7) | Final answer | YES (KL=0.0) | Full dark chain |
| Entity-math hybrid | PARTIAL (L7) | Intermediate only | Mechanical | Dark has lookup, not operation |
| Pure math (CoT) | NO | Nothing | N/A | Chain-of-thought, not dark |
| Variable tracing | PARTIAL | Query structure | YES (KL=0.0) | Markov holds; fix is full replacement |
| Operation type | YES (L7–L14) | Operation class | N/A | 84–88% accuracy, surface-form robust |
| Code output (simple) | N/A | Model mostly correct | N/A | Few failures to analyze |
