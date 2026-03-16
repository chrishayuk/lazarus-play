# Latent Chain Generation: Why It's Architecturally Impossible

**Experiment ID:** 0c83facb-8609-4600-9ad6-16700f3bf667
**Model:** google/gemma-3-4b-it (34 layers, 2560 hidden dim, 8 heads)
**Date:** 2026-03-15

## The Question

Can we generate multi-token answers by chaining residual states through the model
WITHOUT projecting to tokens at intermediate steps? The model processes its own
output residual as input to the next step, staying in 2560D continuous space.
Tokens are only decoded at the very end.

## The Answer: No. Here's Why.

Latent chain generation is **architecturally impossible** in standard transformers.
The obstacle is not fixable by normalization, layer selection, or clever injection.
It's structural.

---

## Four Levels of Failure

### Level 1: Norm Mismatch (addressable, not the real problem)

L33 output norms: 19,000-33,000. L0 embedding norms: ~880. Ratio: 22-37x.

| Prompt   | L0 norm | L33 norm | Ratio | cos(L0,L33) |
|----------|---------|----------|-------|-------------|
| France   | 876     | 19,976   | 22.8x | 0.526       |
| Germany  | 880     | 18,758   | 21.3x | 0.422       |
| 2+2      | 896     | 33,193   | 37.0x | 0.831       |
| sky      | 880     | 19,877   | 22.6x | 0.351       |
| water    | 879     | 22,946   | 26.1x | 0.447       |

Could be fixed by normalizing or injecting at a compatible layer. Not the real problem.

### Level 2: Prediction State vs Processed-Token State (fundamental)

The L33 residual at position N encodes "predict token X next." Position N+1
needs "token X was just processed." These are fundamentally different.

**Evidence:** Cross-position injection (L33 from position N → position N+1) at
every layer tested (L0, L4, L8, L12, L16, L20, L24, L28, L32) **replays the
donor's prediction** rather than advancing the sequence.

| Inject Layer | Injected Top1 | KL(donor,inj) | Advances? |
|-------------|---------------|---------------|-----------|
| L0          | ** (99.99%)   | 13.09         | NO        |
| L8          | ** (99.99%)   | 13.20         | NO        |
| L16         | ** (99.99%)   | 13.19         | NO        |
| L24         | ** (81.0%)    | 2.21          | NO        |
| L28         | The (99.2%)   | 3.64          | NO        |
| L32         | The (73.4%)   | 3.93          | NO        |

At L28-L32, the injected model produces the SAME token the donor predicted ("The"),
not the NEXT token in the sequence (" capital"). The chain is stuck.

### Level 3: KV Cache Is the Information Carrier (architectural)

Entity identity doesn't live in the current position's residual. It lives in the
KV cache of previous positions.

**Proof:** Beethoven vs Mozart at L14: cos = 0.9999 (virtually identical residuals).

- **Single-position injection** (replace one position's residual): NO EFFECT.
  Model still outputs the recipient's answer (Salzburg for Mozart).
  KL(recipient, injected) = 0.000064.

- **All-position injection** (replace all KV entries): COMPLETE OVERRIDE.
  Mozart context outputs Beethoven's answer (Bonn). KL(donor, injected) = 0.0.

The difference is the KV cache. Single-position injection leaves previous positions'
KV entries untouched. The model attends to those entries and gets the original entity.
All-position injection replaces the entire attention history.

### Level 4: L33 Discontinuity (makes chaining impossible)

Consecutive autoregressive steps produce L33 residuals that are nearly **orthogonal**.

| Layer | Cosine range across 5 steps | Interpretation |
|-------|----------------------------|----------------|
| L14   | 0.996 - 0.999              | FROZEN          |
| L26   | 0.939 - 0.964              | Moderate change |
| L33   | **0.109 - 0.413**          | CHAOTIC         |

L33 minimum cosine = 0.109 (84 degrees apart). There is no smooth manifold to
chain on. Each token gets a bespoke output vector organized by ROLE (what to
predict next), not by CONTENT (what entity is being discussed):

- Same role, different entities (B_step1 vs M_step1): cos = **0.866**
- Same entity, different roles (B_step0 vs B_step1): cos = **0.351**

---

## The Static-Dynamic Gradient

The deepest finding: layers exist on a gradient from static (L14) to chaotic (L33).

**L14 (Entity Compass):** cos > 0.996 across all steps. Even Beethoven vs Mozart =
0.99999. The dark space at L14 is a FIXED REFERENCE FRAME encoding the question
type and template format. Entity identity lives in <0.001% of the variance,
readable only by trained probes.

**L26 (Fact Layer):** cos 0.939-0.964. The answer signal "charges" before emission
(Vienna projection peaks at 2119 right before "Bonn" is emitted) then "discharges"
after (crashes to 583). ~6% change per step.

**L33 (Output):** cos 0.109-0.413. Completely discontinuous. Organized by prediction
role, not content. No manifold to chain.

This gradient is WHY latent chaining can't work:
- **L14**: Continuous but static. Chaining adds nothing (state doesn't change).
- **L33**: Information-rich but discontinuous. Can't form a chain.
- No layer is BOTH information-rich enough to advance AND continuous enough to chain.

---

## What the Dark Space Actually Is

**Not this:** A hidden computation space that processes information across tokens
in parallel, performing latent reasoning that could replace CoT.

**Actually this:** A fixed reference frame (L14 entity compass) that encodes WHAT
QUESTION is being asked. The model RE-DERIVES it from the KV cache at each position.
It doesn't need to survive the token bottleneck because it's reconstructed from
scratch via attention to previous positions.

**The token bottleneck is not the information bottleneck.** Information flows through
the KV cache (cross-position attention), not through the token→embedding pathway.

---

## Why CoT Can't Be Replaced

All-position L14 injection into a "dummy" recipient produces KL=0.0 but generates
the SAME CoT as the donor:

- "Beethoven was born in Bonn, Germany. The capital of Germany is **Berlin**."

The dark space doesn't contain pre-computed answers. It contains question encodings.
The actual computation IS the autoregressive token generation. There is no shortcut.

---

## What Would Need to Change

For latent chain generation to work, the architecture would need:

1. **Residual-to-residual connections across positions** (recurrence)
2. **A trained transition function** T: prediction_state → processed_token_state
3. **KV entries writable from injected residuals**, not just from embedded tokens

None of these are possible with existing transformer weights.

---

## Positive Findings

### L14→L0 Answer Shortcut
L14 residual injected at L0 produces the answer token directly (Paris at 99.7%
from France prompt). The entity compass, processed from scratch through all 34
layers, bypasses formatting and surfaces the raw answer.

### L26 Answer Charging
The Vienna signal at L26 peaks right before answer emission (2119), crashes after
(583). The fact layer "charges up" the answer signal and discharges when emitted.

### Role-Based L33 Clustering
L33 residuals cluster by generation step role, not entity identity. Two different
entities at the same step (cos 0.87) are far more similar than the same entity at
different steps (cos 0.35). The output layer is organized by "what KIND of token
to emit."

### Entity Identity Paradox
Probes read entity identity at L14 with 100% accuracy, but entities are cos=0.9999
in full 2560D space. The entity-specific signal is <0.001% of total variance. The
probe finds a needle in a 2560-dimensional haystack.

---

## Implications

1. **CoT is not a workaround for lossy token projection.** It's the actual
   computation. The model genuinely needs each autoregressive step.

2. **The token bottleneck doesn't lose dark space information** because the
   information pathway is through KV cache, not through tokens.

3. **Transformers are fundamentally sequential across positions.** Parallelism
   exists within each position (layers process in sequence, but each layer
   processes all dimensions simultaneously). But across positions, the model
   must embed, attend, and compute at each step.

4. **The "99.4% dark space" finding needs reinterpretation.** The dark space
   is not hidden computation — it's a high-dimensional background signal
   (template format, positional encoding, universal features) that happens to
   be orthogonal to vocabulary directions. The entity-specific signal that
   matters is vanishingly small (<0.001%) and lives in the KV cache pathway.
