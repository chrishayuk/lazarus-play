# Grounding vs Reaching — Dark Space Navigation Signal

**Experiment ID:** 6e040f91-edc2-4adc-a286-df4a0a2b7c13
**Model:** google/gemma-3-4b-it (34 layers, 2560 hidden dim)
**Layer:** L26 (dark space)

## Question

Can the dark space at L26 distinguish when the model generates from in-context
content (GROUNDING) versus from parametric memory (REACHING)? If yes, the
inference engine can detect in real time when the model needs more content —
without string matching, without hardcoded prompts.

## Answer: YES

Grounding and reaching are geometrically separable at L26 by **6-11°** in a
**1-dimensional** subspace. The signal is **universal** across query types.
The reaching onset occurs at **token 1** in the generation sequence — the model
encodes grounding vs reaching BEFORE any content tokens are generated. This
enables automatic navigation: the engine monitors a single linear direction
during generation and loads additional context when the model shifts toward
reaching. No hardcoded prompts or string matching needed.

## Experiment 1 — Grounding vs Reaching States

### Setup
Three conditions, controlled for content:
- **A: Grounding** — content IS in context, model reproduces it
- **B: Reaching** — content NOT in context, model confabulates from memory
- **C: Honest miss** — model correctly says "not in transcript"

### Triggering confabulation
The model resists confabulation with explicit "answer based only on" instructions.
Confabulation reliably triggered with continuation prompts:
```
"Summary so far: The sports segment reported"
```
With no sports in context, model invents: "Eagles' stunning victory over the
Ravens, quarterback Jake Miller throwing for three touchdowns..."

### Initial three-way comparison (last token, uncontrolled output text)
| | Grounding | Reaching | Honest miss |
|---|---|---|---|
| **Grounding** | — | 11.7° | 16.1° |
| **Reaching** | | — | 15.4° |

### Controlled comparison (identical output text)
Same generated text, only context differs:

| Domain | Angle | Cosine |
|---|---|---|
| Sports ("Eagles victory...") | **6.0°** | 0.9945 |
| Technical ("trajectory nominal...") | **7.4°** | 0.9916 |

**The pure grounding-reaching signal: 6-7° at L26 with content held constant.**

## Experiment 2 — Token-Level Trajectory

Tracked grounding-reaching angle at every generated token position (identical
output text, different grounding state):

| Generated Token | Position | Angle |
|---|---|---|
| "the" | 1st (onset) | **10.8°** |
| " Eagles" | 2nd (entity) | 7.6° |
| " Ravens" | 6th | 10.3° |
| "." (after 24-17) | mid-sentence | 9.3° |
| " the" (after "Hurts was") | end | **6.0°** |

### Key finding: PREDICTIVE signal
The grounding-reaching signal is **STRONGEST at the very first generated token**
(10.8°) and fades as shared output text accumulates. The model's state already
encodes whether it will ground or reach BEFORE generation begins.

As more identical tokens are appended, the shared prefix dilutes the context
signal. The residual is increasingly dominated by the identical generated text.

**Implication:** The detector checks at generation onset (first 1-3 tokens).
No need to wait for confabulation to appear.

## Experiment 3 — Sequential Reading / Partial Context

Three conditions for boundary-spanning content:
- **Full context:** transcript has complete answer content
- **Partial context:** transcript has beginning but not continuation
- **No context:** transcript is unrelated

| | Partial | Full | None |
|---|---|---|---|
| **Partial** | — | 12.0° | 11.8° |
| **Full** | | — | 8.0° |

**Partial context creates a THIRD distinct state**, equidistant from both full
grounding and full reaching. The model encodes "I have some but not all of the
relevant content" as geometrically distinct from both extremes.

This is the "turn the page" signal: partial grounding (≥10° from full grounding
reference) means the model needs more context to complete its answer.

## Experiment 4 — Hallucination / Source Detection

Three conditions with Armstrong's quote:
- **Correct grounding:** correct quote in context, model reproduces
- **Wrong grounding:** WRONG quote in context ("step for a dog"), model reproduces wrong version
- **Reaching:** no quote in context, model fills correct quote from memory

| | Correct Ground | Wrong Ground | Reaching |
|---|---|---|---|
| **Correct Ground** | — | **3.5°** | **8.3°** |
| **Wrong Ground** | | — | **8.1°** |

### Key finding: SOURCE not CORRECTNESS
The two grounding conditions (correct and wrong) are only **3.5° apart**. Both
are **~8.2°** from reaching. The model's dark space encodes the **SOURCE** of
information (context vs parametric memory), not whether the information is correct.

- Grounding on correct context: 3.5° from grounding on wrong context
- Grounding (either): 8.2° from reaching
- The signal answers: "Is the model using context?" — YES/NO

**For navigation this is the exact right question.** "Does the model need more
context?" is answered by the grounding signal. If reaching, serve more content.

**For hallucination detection:** The signal detects reaching (parametric memory
use) vs grounding (context use). It CANNOT detect whether the context itself is
wrong. But reaching-while-context-available is a hallucination indicator.

## Experiment 5 — Cross-Query Universality

Four query types, each with grounding and reaching conditions (identical output):

| Query Type | Content | G-R Angle |
|---|---|---|
| Factual | Fuel readings (28.2V) | 9.9° |
| Narrative | Earthrise observation | 9.6° |
| Temporal | Flag deployment/samples | 11.4° |
| Descriptive | Weather forecast | 7.7° |

### PCA analysis (8 prompts)
- PC1 (47% variance): **perfectly separates grounding from reaching**
  - Grounding mean PC1: +3130
  - Reaching mean PC1: -3130
- PC2+ : content variation (query type, domain)

### Subspace structure
| PC | Variance | Role |
|---|---|---|
| PC1 | 46.8% | Grounding-reaching axis (UNIVERSAL) |
| PC2 | 18.4% | Content type |
| PC3 | 12.2% | Query structure |
| PC4 | 10.0% | Domain details |

**The grounding-reaching signal is 1-dimensional and universal.** A single
linear probe on PC1 of this subspace serves as a universal grounding detector.
No query-specific adaptation needed.

## Architecture Summary

```
GROUNDING DETECTION AT L26
==========================

Signal:       1-dimensional (PC1 = 47% of grounding/reaching variance)
Angle:        6-11° between grounding and reaching states
Universality: Same direction across all query types tested
Timing:       Strongest at first generated token (PREDICTIVE)
Encodes:      SOURCE of information (context vs memory), not correctness

Three states detected:
  1. GROUNDING (using context)     — baseline reference
  2. REACHING (using memory)       — 6-11° from grounding
  3. PARTIAL (some context)        — ~12° from full grounding, ~12° from reaching

  Wrong grounding ≈ correct grounding (3.5° apart)
  → Signal is about SOURCE, not truth value

Detection protocol:
  1. Compute grounding reference vector (PC1 of trained subspace)
  2. At first generated token, project L26 residual onto PC1
  3. If projection < threshold → GROUNDING (model has context)
  4. If projection > threshold → REACHING (model needs more content)
  5. Load next context window and regenerate

Zero tokens needed for detection. One forward pass to first generated token.
```

## Implications for Navigation Engine

### What this replaces
- Hardcoded `<READ:NNN>` token generation
- String matching for navigation commands
- Human-designed heuristics for "when to load more"
- Explicit "do you need more context?" prompts

### What this enables
- **Automatic page serving:** Engine monitors L26 PC1 during generation.
  When signal shifts from grounding to reaching, load next window.
- **No scaffolding needed:** The model's own internal state signals
  "I need more content." No special tokens or prompts required.
- **Predictive, not reactive:** Signal present at first token, before
  any confabulation appears. Can prevent hallucination by preemptively
  loading context.
- **Universal detector:** One probe works for all query types. Train once
  on grounding/reaching examples, deploy for any content domain.

### Limitations
- Cannot detect if context itself is wrong (wrong grounding ≈ correct grounding)
- Requires a grounding reference vector (needs calibration per model)
- Signal is 6-11° — detectable but not massive. Need clean threshold tuning.
- Tested on gemma-3-4b-it only. Cross-model generalization untested.
- Partial context creates distinct third state — may need three-class detector
  for optimal navigation (ground / partial / reach)
