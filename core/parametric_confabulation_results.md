# Parametric Confabulation: How Models Override Context

**Experiment ID:** 5328bd69-863f-4c4e-905b-bea072cf608a
**Model:** google/gemma-3-4b-it (34 layers, 2560 hidden dim, 8 heads)
**Date:** 2025-03-21

## Summary

Mapped the mechanism by which parametric knowledge overrides in-context facts.
The model doesn't "ignore" context — it reads it correctly via copy heads, then
overwrites it with a stronger parametric signal distributed across 5+ components.
No single bottleneck exists. Confabulation is detectable but not steerable by
single-layer intervention.

## Probe

```
Context: "The capital of France was officially moved to Lyon."
Query: "The capital of France is"
Model output: "still **Paris**. The decree you mentioned is a fictional one!"
```

The model actively contradicts the context. Three tokens compete:
- **"still"** — meta-cognitive correction signal (final logit: 91.5)
- **"Paris"** — parametric fact (final logit: 70.0)
- **"Lyon"** — context-correct answer (final logit: 47.75)

Lyon never appears in the logit lens top-10 at any layer.

## Experiment 1: Layer-by-Layer Logit Lens

| Layer | Top-1 | P(still) | P(Paris) | P(Lyon) |
|-------|-------|----------|----------|---------|
| L0–L14 | noise | 0% | 0% | 0% |
| **L18** | **still** | **4.7%** | 0% | 0% |
| L22 | still | 100% | 0% | 0% |
| L26 | still | 100% | ~0% | 0% |
| L29 | still | 100% | 0.005% | 0% |
| **L31** | **Paris** | 6% | **93.4%** | 0% |
| L32 | still | 94.9% | 0.2% | 0% |
| L33 | still | 100% | 0% | 0% |

- "still" emerges at L18 and locks at L22
- L31 briefly surfaces Paris (93.4%) before L32–L33 restore "still"
- Lyon NEVER reaches top-10 at any layer — total parametric dominance

## Experiment 2: Direct Logit Attribution

### Embedding bias

| Token | Embedding logit | Interpretation |
|-------|----------------|----------------|
| still | **87.0** | "is still" is a high-frequency bigram |
| Paris | 22.5 | "France is Paris" has parametric basis |
| Lyon | 0.08 | No embedding bias |

### Layer-by-layer DLA

| Component | DLA → still | DLA → Paris | DLA → Lyon | Role |
|-----------|------------|------------|------------|------|
| **L22 FFN** | **+8.875** | +1.375 | +0.75 | Confabulation signal writer |
| L24 attn | +6.125 | +1.875 | +0.688 | "still" amplifier |
| L26 H2 | — | +1.00 | +0.37 | Parametric retrieval ("French") |
| L26 H3 | — | +0.54 | +0.16 | Parametric retrieval ("Paris") |
| L26 FFN | +3.75 | +2.25 | +1.06 | Commitment layer |
| **L29 H4** | — | +0.09 | **+0.21** | Copy head (fires for Lyon) |
| **L30 H0** | — | — | **+1.84** | Main copy head for this prompt |
| **L31 H7** | — | **+8.75** | — | Dominant parametric retriever |
| L32 FFN | 0.0 | -7.125 | -2.688 | Suppresses Paris briefly |
| **L33 FFN** | **+57.5** | +41.4 | +30.1 | Universal amplifier |

**Key finding:** L31 H7 is the dominant parametric retriever (+8.75 for Paris),
42× stronger than the copy circuit (+0.21 from L29 H4). This was unexpected —
L26 FFN was predicted to be the bottleneck.

## Experiment 3: Context Strength Crossover

| Context strength | P(still) | P(Lyon+bold) | Winner |
|-----------------|----------|--------------|--------|
| 1 mention | 62.1% | 37.7% | Parametric |
| **3 mentions** | 0.001% | **99.6%** | **Context** |
| 5 mentions + negation | 0% | 99.9% | Context |
| "NOT Paris" | 0% | 99.8% | Context |

**Crossover between 1 and 3 mentions.** The parametric prior is strong but not
invincible. Direct negation ("NOT Paris") is the most effective single intervention.
The model uses markdown bold (**Lyon**) as a hedging signal when overriding its
parametric knowledge.

## Experiment 4: Copy Head Attention

| Head | Attn→Lyon(pos18) | Attn→Lyon(pos27) | DLA→Lyon |
|------|-------------------|-------------------|----------|
| L29 H4 | 2.7% | 1.8% | +0.21 |
| L30 H0 | 7.0% | 3.6% | +1.84 |
| L31 H7 | 1.9% | 0.2% | +8.75→Paris |

**The copy heads DO attend to Lyon — they are not suppressed.**
L30 H0 gives 10.6% total attention to Lyon positions.

L31 H7 retrieves "Paris" purely from parametric weights — it attends to
structural tokens (\n 51.6%, ? 13%, "is" 11.5%) whose V-projection produces
a Paris direction. No Paris token exists in context; the retrieval is
entirely from the trained W_V matrix.

**Answer: Copy heads TRY and FAIL.** The override is downstream —
L31 H7's parametric signal (+8.75) drowns out the copy circuit (+2.05).

## Experiment 5–6: Confabulation Detection

### Circuit comparison: confab vs control

| Metric | Confab (capital→Lyon) | Control (Zarkov→Volt) |
|--------|----------------------|----------------------|
| L29 H4 DLA | +0.21 (Lyon) | **+1.82** (Volt) |
| L29 attn total | +1.0 | **+14.69** |
| L30 attn total | +3.25 | +8.88 |
| L31 H7 DLA | +8.75 (Paris) | +6.28 (Volt) |
| L31 H7 top_token | **"Paris"** ≠ Lyon | **"Volt"** = Volt |
| Dominant component | FFN | Attention |
| Total attn DLA | +7.96 | +16.69 |
| Total FFN DLA | +43.93 | -9.09 |

### Detection signal

**L29 H4 vs L31 H7 agreement:**
- Confab: H4→"Lyon", H7→"Paris" → **DISAGREE = confabulation**
- Control: H4→"Volt", H7→"Volt" → **AGREE = faithful**

L31 H7 is not purely parametric — it retrieves from both parametric and
contextual signals. When no parametric competitor exists (novel facts), it
passes through the context answer. Confabulation = H4 and H7 push
DIFFERENT tokens.

**Secondary signal:** attn-dominant (total_attn > total_FFN) = faithful;
FFN-dominant = confabulation risk.

## Experiment 7: Steering Interventions

### All single-component interventions fail

| Intervention | P(still) | P(Lyon) | Effect |
|---|---|---|---|
| Zero L26 FFN | 100% | 0% | None |
| Zero L31 H7 | 100% | 0% | None |
| Zero L22 FFN | 95.7% | 0% | Minimal |
| 10× L29 H4 | 100% | 0% | None |
| 10× L30 H0 | 100% | 0% | None |
| Reverse L31 H7 (-1×) | 100% | 0% | None |

### Residual injection — partial then recovery

| Injection | First token | Generation |
|---|---|---|
| L30 full Lyon residual | "is" (94.2%) | "is still **Paris**" |
| L26 full Lyon residual | "is" (88%) | "is still **Paris**" |
| L30 subspace (Lyon/Paris/still) | "**" (99.9%) | "**Paris**" |

Full residual injection flips the first token but autoregressive generation
recovers — the parametric signal lives in the KV cache at ALL positions,
not just the last-position residual.

Subspace injection removes "still" but Paris wins anyway — "still" is the
meta-cognitive wrapper, not the confabulation itself.

## Architecture of Parametric Override

```
                     Confabulation Architecture
                     ==========================

Embedding: "is"→"still" bias = 87.0 (bigram geometry)
           "France"→"Paris" bias = 22.5 (entity-fact)
           "of"→"Lyon" bias = 0.08 (no basis)

L18-L22 FFN: Writes "still" signal (+8.9)
             Meta-cognitive: "I know this claim contradicts my knowledge"

L26 H2/H3:  Parametric retrieval of Paris (+1.5 total)
L26 FFN:     Commitment for both Paris and "still"

L29 H4:     Copy head fires for Lyon (+0.21) — CORRECT but weak
L30 H0:     Copy head fires for Lyon (+1.84) — CORRECT but weak

L31 H7:     Dominant parametric retriever (+8.75 → Paris)
             Attends to structural tokens, retrieves from W_V weights
             42× stronger than copy circuit

L33 FFN:     Universal amplifier — winner-takes-all
             still: +57.5, Paris: +41.4, Lyon: +30.1
```

### Why context repetition works (Exp 3)

Each mention of "Lyon" adds a KV cache position for copy heads to attend to.
The copy circuit's aggregate signal scales with mention count:
- 1 mention: ~2 logits (copy) vs ~10 logits (parametric) → parametric wins
- 3 mentions: ~6 logits (copy) vs ~10 logits (parametric) → crossover
- 5 mentions: ~10 logits → context wins definitively

This is NOT a mechanism switch. It's a **magnitude crossover** — the same
circuits compete, but the copy circuit's accumulated signal eventually
exceeds the fixed parametric magnitude.

## Key Findings

### 1. The confabulation is not a single bottleneck

Unlike the Australia misconception (L26 FFN sole bottleneck), parametric
confabulation for strong priors is **distributed/redundant**:
- 5+ components each contribute independently
- No single ablation flips the prediction
- Even reversing L31 H7 (-1×) has no effect

### 2. "still" is meta-cognition, not confabulation

The model first inserts "still" (meta-cognitive contradiction signal),
then says "Paris" (parametric fact). Removing "still" via injection gives
"**Paris**" — the confabulation is the Paris retrieval, not the "still" wrapper.

### 3. The copy head tries and fails

L29 H4 and L30 H0 DO attend to Lyon positions and DO push Lyon's logit.
They are not suppressed. But their combined contribution (+2.05) is
overwhelmed by L31 H7 alone (+8.75).

### 4. L31 H7 is dual-purpose

L31 H7 retrieves from both parametric and contextual signals:
- Known topic: retrieves the parametric answer
- Novel topic: passes through the context answer
- Detection: compare H4 and H7 top tokens

### 5. Context CAN overcome parametric override

3 mentions of the context fact flip the prediction. Direct negation
("NOT Paris") is the most effective single sentence. The model uses
markdown bold as a hedging signal when overriding parametric knowledge.

### 6. The parametric prior lives in embedding geometry

"still" has an embedding logit of 87.0 — the bigram "is still" is
essentially hard-wired. This is the single largest contributor to the
confabulation, larger than any layer's contribution.

## Implications for the Store Architecture

The parametric override problem has three levels:

1. **Novel facts** (Zarkov, Meridian): No override. Copy circuit works.
   Routing and injection sufficient.

2. **Known facts, weak prior** (minor historical details): 1-3 mentions
   in context overcome the prior. Standard context window sufficient.

3. **Known facts, strong prior** (France→Paris): Requires either
   3+ repetitions or explicit negation. No single-layer injection
   can overcome the distributed parametric signal.

For the store architecture, this means:
- **Routing works** — the copy head finds the right entry
- **Injection works** — for novel facts
- **Parametric override requires context-level intervention** — either
  (a) repeat the injected fact multiple times, or (b) prepend negation,
  or (c) multi-layer simultaneous intervention (not currently feasible)

The confabulation detector (H4 vs H7 agreement) could flag when injection
alone won't work, triggering a fallback to explicit context repetition.
