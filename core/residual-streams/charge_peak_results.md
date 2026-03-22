# The Charge Peak: Capturing Facts From the Residual Stream

**Experiment ID:** 993f5f93-cc27-48de-b976-0efcc066b3c6
**Model:** google/gemma-3-4b-it (34 layers, 2560 hidden dim, 8 heads)

## Summary of Key Findings

### 1. The Fact Position Encodes "What Comes Next," Not "What I Am"

At pos 11 (" Volt") during prefill, the logit lens shows:
- L0-L14: noise
- L18-L25: "ville" / "city" (contextual type)
- L26-L30: " City" top-1
- L33: "ara" top-1 (next-token prediction)

**" Volt" NEVER appears in the top 10 at its own position.** The residual at
the fact position encodes the NEXT token ("ara"), not its identity (" Volt").
99.4-99.8% of the residual energy is orthogonal to all fact-related tokens.
The fact identity lives in dark space.

### 2. The Generation Position Has a Sign Flip at L29

At pos 43 (" of") — the generation position — the " Volt" projection:

| Layer | Volt projection | Status |
|---|---|---|
| L0 | -20.3 | Suppressed |
| L14 | -653.3 | Deep suppression |
| L22 | **-797.2** | Pre-retrieval low |
| L23 | -414.0 | Recovering |
| **L29** | **+1,555.5** | **SIGN FLIP = RETRIEVAL** |
| L33 | +1,655.2 | Amplified |

The swing from -797 to +1,556 (a delta of 2,353 units) between L22 and L29
IS the retrieval event. L29 H4 copies the fact signal from pos 11's KV cache
to the generation position.

### 3. The Retrieval Circuit: Three Dedicated Heads

**L23 H1** — First Read (DLA +1.55, 97.6% of layer)
- Attends to pos 11 (" Volt") at **75.8%** weight
- The scout head: high focus, moderate DLA

**L29 H4** — Main Copy (DLA +1.74, 99.1% of layer)
- Attends to pos 11 at **44.9%** weight (+ 50.8% bos)
- The retrieval head: highest DLA, the signal transfer event

**L30 H0+H3** — Reinforcement (DLA +1.57 + 1.41)
- Each attends to pos 11 at 16-18% weight
- More distributed, consolidating the signal

### 4. Novel Facts Are Attention-Dominated; Parametric Facts Are FFN-Dominated

**DLA at generation position:**

| Metric | Voltara (novel) | Paris (parametric) |
|---|---|---|
| Total attention DLA | **+22.0** | -32.6 |
| Total FFN DLA | -12.7 | **+41.2** |
| Dominant component | **ATTENTION** | **FFN** |
| Peak layer | L29 attn (+12.81) | L25 FFN (+8.00) |

Complete mirror images. Novel fact retrieval reads from KV cache via
attention. Parametric fact retrieval reads from weights via FFN.

### 5. The 5.1KB Answer: Single Residual Injection at L29

**Fact-position injection (pos 11 residual → generation position):**

| Layer | Top-1 | Prob | KL | Why |
|---|---|---|---|---|
| L23 | "ara" | 36.3% | 0.015 | Reproduces fact pos output (next token) |
| L29 | "ara" | 38.6% | 0.024 | Same — fact pos predicts "ara" not "Volt" |

**Generation-position injection (last pos residual → last pos):**

| Layer | Top-1 | Prob | KL | Why |
|---|---|---|---|---|
| L23 | " Z" | 22.8% | 3.12 | FAILS — retrieval hasn't happened yet |
| **L29** | **" Volt"** | **99.5%** | **0.000808** | **PERFECT — 5.1KB single residual** |
| L33 | " Volt" | 99.3% | 0.000001 | Even more perfect |

**A single 2560-dimensional bfloat16 vector (5.1KB) from the generation
position at L29 reconstructs the answer with KL = 0.0008.**

The asymmetry: the fact-position residual carries "what comes next" (ara).
The generation-position residual carries "the answer" (Volt). To capture
the answer, you must capture it at the DESTINATION after the retrieval
heads fire (L29+), not at the SOURCE.

### 6. Charge Magnitude Is a Weak Fact Detector; Attention Weight Is Strong

At L29, the total fact-related subspace fraction:
- Fact positions (Volt, ara): 0.24-0.30%
- Filler positions (The, people, research): 0.12-0.17%

Only a ~2x gap. Too weak for reliable detection.

The real detector: **L29 H4 attention weight** puts 44.9% on the fact
position and <1% on any filler position. The attention mechanism itself
IS the fact detector.

### 7. L29+L30 Attention DLA: Universal Novel/Parametric Classifier

Tested across 4 facts (2 novel, 2 parametric):

| Fact | Type | L29+L30 attn DLA | L25 FFN | L33 FFN |
|---|---|---|---|---|
| Voltara | Novel | **+17.69** | +0.44 | +1.38 |
| Wavebreak | Novel | **+17.44** | +1.03 | +0.75 |
| Paris | Parametric | **-0.63** | +8.00 | -1.13 |
| Hornet | Parametric | **-0.56** | +2.72 | +18.88 |

**L29+L30 attention DLA > +5: novel fact (KV retrieval)**
**L29+L30 attention DLA < +1: parametric fact (FFN retrieval)**

Gap is 30:1. Clean binary separator.

### 8. Hornet: Extreme Parametric Signature

- Embedding logit: **88.0** (vs 22.5 for Paris, 11.7 for Voltara)
- L0 suppression: **-45.25** (massive)
- L33 FFN: **+18.88** (the confabulation amplifier — but here CORRECT)
- Total attention: **-114.1** (overwhelmingly suppressive)
- Total FFN: **+81.5**

The L33 FFN spike (+18.88) matches the Type 2 confabulation signature
(L33 FFN >12). This means the L33 spike detects FFN-driven retrieval,
NOT necessarily confabulation. A confident correct parametric fact
(Hornet at 100%) fires the same L33 FFN amplifier as a confabulation.
The difference is whether the FFN content is correct.

## Architecture Summary

```
NOVEL FACT RETRIEVAL (KV pathway):
  Embedding → L0 suppression → Dark layers (L5-L22, near zero)
  → L23 H1 first read (75.8% → fact pos)
  → L29 H4 main copy (44.9% → fact pos)
  → L30 H0+H3 reinforcement
  Attention-dominated: attn +22, FFN -13

PARAMETRIC FACT RETRIEVAL (weight pathway):
  Embedding → L0 suppression → Dark layers
  → L23 FFN (+3-5)
  → L25 FFN universal amplifier (+3-8)
  → L26 FFN (+3)
  → L33 FFN amplifier (+0 to +19)
  FFN-dominated: attn -33 to -114, FFN +41 to +82

FACT POSITION (source):
  Residual encodes "what comes next" (next-token prediction)
  Identity in dark space. FFN builds the V vector.
  Logit lens shows contextual type ("City"), not identity ("Volt")

GENERATION POSITION (destination):
  Receives fact signal via attention at L29.
  Sign flip: Volt proj goes from -797 (L22) to +1556 (L29).
  Single 5.1KB residual at L29 = complete answer (KL=0.0008).
```

## Positions (corrected from protocol)

The protocol assumed " Volt" at pos 14 — actual position is **pos 11**.
- Pos 11: " Volt" (token_id 89711)
- Pos 12: "ara" (token_id 2032)
- Pos 43: " of" (generation position, last token)

## Experimental Details

- 10 steps saved to experiment 993f5f93
- Tags: charge-peak-updated, l23-charge, l29-charge, v-vector-trajectory,
  k-vector-addressability, peak-injection, residual-vs-kv, charge-detection,
  novel-vs-parametric-charge, dual-circuit-charge, fact-detector,
  prefill-charge-curve
