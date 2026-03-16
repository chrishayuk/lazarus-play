# Attention-Free Inference Results

**Experiment ID:** 6759d6ff-8e09-4b61-8522-e3b8e45cc395
**Model:** google/gemma-3-4b-it (34 layers, 2560 hidden dim, 8 heads)

## Executive Summary

**The hypothesis that attention can be fully removed during generation is WRONG in its strong form but PARTIALLY CORRECT in its weak form.**

- **Strong claim (FFN-only generation): IMPOSSIBLE.** FFN has zero cross-position information flow. Without attention, all prompts produce identical gibberish (the FFN fixed point for the "\n" token).
- **Weak claim (partial attention removal): CONFIRMED.** L25-L33 attention (9/34 layers, 26%) can be completely zeroed — even during prefill — with coherent, relevant output maintained.
- **Attention horizon discovered:** Most layers' attention is critical only at the first generation token; by token 7, only L0 and L4 remain individually critical.
- **Composability failure:** 28/34 layers are individually dispensable (KL≈0 when zeroed one at a time), but this redundancy doesn't compose — removing all 28 simultaneously destroys the model.

## Experiment 1 — Attention Budget During Generation

### Residual Decomposition (Attention vs FFN Norms)
- **L0-L9:** Attention dominant (69-86% of residual norm)
- **L10-L14:** Mixed, roughly equal
- **L15-L33:** FFN dominant (53-78%), especially L33 FFN=75-78%
- **Peak:** L33 total norm ~50K; FFN=38-40K, attention=10-13K

### Attention Pattern During Generation (at generated token positions)
| Layer group | Self | BOS | Prompt | Prev gen | Verdict |
|---|---|---|---|---|---|
| L0 | <1% | 5-6% | **72-81%** | 12-22% | **GENUINE RETRIEVAL** |
| L7 | 3-5% | 35-47% | 22-37% | 26-27% | Mixed retrieval |
| L14 | 13-27% | 22-27% | 26-39% | 20-27% | Broad gathering |
| L20 | 10-13% | 35-44% | 22-29% | 16-30% | BOS growing |
| **L25-L26** | 2-8% | **65-79%** | 11-19% | 5-8% | **BOS DOMINATED** |
| L31 | 22-27% | 54-59% | 10-12% | 5-13% | BOS + self |
| L33 | 24-29% | 32% | 31-32% | 7-13% | Still reads prompt |

**Key insight:** L25-L26 send 65-79% of attention to BOS during generation — a learned bias, not retrieval. These layers are doing almost no information gathering.

## Experiment 2 — Progressive Attention Ablation

### Single-Layer Ablation (all 34 layers, France/Paris prompt)
**Critical layers (KL > 1.0):** L0 (7.47), L4 (11.19), L5 (1.10), L6 (14.0), L14 (1.32), L15 (1.32)

**28 out of 34 layers SAFE (KL < 0.02)** when individually zeroed. Only 6 layers' attention matters for first-token prediction.

### Group Ablation (multi-token generation)
| Layers ablated | # layers | Output quality |
|---|---|---|
| **L25-L33** | **9** | **Coherent, relevant output (47% token match)** |
| L22-L33 | 12 | Degraded English, wrong facts |
| L20-L33 | 14 | Mostly gibberish |
| L16-L33 | 18 | Gibberish |
| L0-L14 | 15 | Gibberish |
| All 34 | 34 | Identical gibberish regardless of input |

**The cliff is at L22-L25.** L25-L33 is the maximal safe simultaneous ablation set.

### Composability Failure
- Individual: 28/34 safe → Simultaneous: only 9/34 safe
- Root cause: each layer is redundant because 33 others compensate. Remove all simultaneously and no compensation is available.
- L0+L4 only (keeping 2 layers): gibberish. Not enough to carry the model.

## Experiment 3 — Logit Attribution

### France → "Paris"
- Embedding: -7.28 (negative)
- Total attention: **+23.5 logits (29%)**
- Total FFN: **+64.2 logits (80%)**
- **FFN dominates 2.7× over attention**
- L33 FFN alone: **+45.25** (56% of all FFN)

### Beethoven → "Be"
- Embedding: +53.75 (high — "Be" is natural continuation)
- L0 attention: **-55.7** (embedding war)
- Total attention: **-38.4** (NET NEGATIVE)
- Total FFN: **+72.7**
- L33 FFN: **+52.5** (72% of FFN total)

**FFN provides 71-80% of target token logit. L33 FFN alone provides 56-72%.**

## Experiment 4 — FFN-Only Generation

**Result: IMPOSSIBLE.** All prompts → identical output: "prefabricatedcession ổn indicate نسبت smug賁 sewn..."

Without attention, each position's FFN operates only on its own embedding. The last token is always "\n" (after "model" in chat template), so the output is the same regardless of prompt. **Attention is the sole mechanism for cross-position information transfer.**

## The Attention Horizon

### Critical Layers Decay Over Generation
| Layer | Token 1 (KL) | Token 7 (KL) | Token 10 (KL) | Verdict |
|-------|-------------|--------------|----------------|---------|
| **L0** | **7.47** | **9.44** | **8.13** | **PERMANENTLY CRITICAL** |
| **L4** | **11.19** | **5.81** | **1.07** | Critical but declining |
| L5 | 1.10 | 0.0002 | 0.0 | Safe by token 7 |
| L6 | 14.0 | 0.0002 | — | Safe by token 7 |
| L14 | 1.32 | 0.0 | 0.0 | Safe by token 7 |
| L15 | 1.32 | 0.0 | — | Safe by token 7 |
| L25-L33 | 0.0 | 0.0 | 0.0 | Always safe |

**Attention horizon:**
- Token 1: 6 layers need attention (L0, L4, L5, L6, L14, L15)
- Token 7+: 2 layers need attention (L0, L4)
- Token 15+: Likely 1 layer (L0 only, as L4 continues declining)

### What L0 Does (Why It's Irreplaceable)
L0 is the **continuous prompt reader**. At every generation step, it:
- H5: reads topic words ("capital", "of", "France") — 11%, 9%, 5%
- H6: reads entities ("known" 26%, "Paris" 10%, "which" 15%)
- H0: reads content words ("breakdown" 43%, "known" 31%)
- H1: reads context ("which" 17%, "known" 16%, "for" 10%)

L0 sends 72-81% of attention to the prompt at every generation step. It continuously re-reads the prompt and writes a summary into the residual. Without this summary, downstream layers (which are FFN-dominant) have no context to work with.

### What L4 Does (Why It Fades)
L4 is a **template/formatting layer**:
- H2: 94% BOS — pure learned bias
- H7: reads template tokens (start_of_turn, model, BOS)
- Importance decays as generation format stabilizes

## Proposed Architecture

### Practical (verified)
```
Prefill:  Full attention all 34 layers
Generate: L0-L24 attention + L25-L33 FFN-only
Savings:  26% attention compute, ~13% total, proportional KV cache reduction
```

### Theoretical (from single-layer tests, not composable)
```
Token 1:   L0 + L4 + L5 + L6 + L14 + L15 attention (82% skipped)
Token 7+:  L0 + L4 attention only (94% skipped)
Token 15+: L0 attention only (97% skipped)
```
This is the upper bound if composability could be achieved (e.g., through distillation or fine-tuning to make remaining layers compensate).

## Why the Hypothesis Was Partially Wrong

The original hypothesis: "The Markov property means the residual carries all needed information, so attention during generation is redundant."

**What's correct:**
- The residual IS the complete Markov state
- FFN IS the dominant computation engine (71-80% of logit)
- L25-L33 attention IS genuinely expendable (BOS-dominated, not doing retrieval)

**What's wrong:**
- Attention doesn't just READ state — it also WRITES necessary transformations
- Even BOS-dominated attention heads provide a learned bias that downstream FFN layers expect
- The 99.4% dark space carries context, but attention is needed to MAINTAIN it across generation steps
- L0 must continuously re-read the prompt because the residual's prompt summary degrades without refreshing

**The fundamental insight:** Attention serves two functions:
1. **Information gathering** (reading KV cache, copying from other positions) — genuinely needed only at L0
2. **Local transformation** (a learned linear bias from BOS/self attention) — individually dispensable but collectively necessary

Function 1 concentrates in L0. Function 2 is distributed across all layers and is what creates the composability failure.

## Connection to Previous Work

- **Markov property (929cdef0):** Confirmed. The residual is the complete state. But "complete state" includes the attention-written components.
- **Dark space (446690dc):** The 99.4% dark space carries entity identity, but attention is required to maintain it across generation steps via L0's continuous prompt reading.
- **L25 FFN amplifier:** Confirmed as attention-independent. L25 FFN works fine without L25 attention.
- **L26 FFN fact store:** Also attention-independent. L26 FFN fires correctly without L26 attention.
- **Latent chain generation (0c83facb):** This experiment explains WHY latent chaining failed. Residual chaining without attention lacks the continuous L0 prompt refresh needed at each step.
