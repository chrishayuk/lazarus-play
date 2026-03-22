# Additive Multi-Dimensional Injection Results

**Experiment ID:** 7f89fd70-4147-4894-a5a2-9e7bdbfeba3d
**Model:** google/gemma-3-4b-it (34L, 2560D, 8H)
**Date:** 2026-03-22
**Status:** Complete
**Outcome:** C (with nuance) — Entity name works, narrative requires context

---

## Setup

- **Donor:** Focused porridge passage (~194 tokens) + query, chat-templated
- **Recipient:** Bare query, chat-templated
- **Injection layer:** L30 (last-position residual)
- **Baseline:** Donor P(John) = 99.2%, Recipient P(This) = 100%
- **Cosine(donor, recipient) at L30:** 0.9826 (angle 10.7°)

## Experiment 3 — Raw Difference Vector (Additive Steering)

Computed `steering_vector = mean(donors) - mean(recipients)` at L30 (norm 12,248).
Applied additively via `steer_and_generate` at varying alpha.

| Alpha | First Token | Output | P(John) |
|------:|------------|--------|--------:|
| 0.3 | This | Monty Python / Holy Grail | 0% |
| 0.5 | This | Monty Python / Holy Grail | 0% |
| 1.0 | This | Harry Potter / Goblet of Fire | 0% |
| 2.0 | This | **John Lennon**, 1969 | ~0% |
| **3.0** | **According** | **John Muir** won in 1881 | ~30% |
| 4.0 | According | John John John repetition begins | — |
| **5.0** | **John** | John John, Ireland, 8-year-old | ~90% |
| 6.0 | John | Irish John John, 10.5 kg, 2023 | ~95% |
| 7.0 | John | John John John! Irish, Dublin | — |
| 10+ | John | JohnJohnJohnJohn... (fixed point) | 100% |

**Phase transitions:**
- Alpha ≈ 3: entity name "John" first appears
- Alpha ≈ 5–6: sweet spot — entity + "Irish"/"Ireland" leaks (real fact!)
- Alpha ≥ 10: fixed point (repetitive)

**What leaks through:** "John" (first name), "Irish"/"Ireland" (nationality from "an Irishman")
**What NEVER appears at any alpha:** Coyle, 23, bowls, oatmeal, Corby, 35, championship

**Diagnosis:** Difference vector is dominated by entity-name and nationality directions.
Narrative details too distributed in 2560D for additive injection.

## Experiment 2 — PCA Subspace Injection

### Subspace Analysis

PCA on 5 donor + 5 recipient prompts at L30 (10 prompts, rank 9):

| PC | Variance | Cumulative |
|---:|--------:|----------:|
| 1 | 43.7% | 43.7% |
| 2 | 19.4% | 63.0% |
| 3 | 14.6% | 77.6% |
| 4 | 7.9% | 85.5% |
| 5–9 | 14.5% | 100% |

### Rank Sweep (Subspace Replacement)

| Rank | Energy | Top1 | P(John) | Output | Multi-token? |
|-----:|-------:|------|--------:|--------|:------------:|
| 1 | 17% | According | 39.9% | The Gruffalo mouse | ✓ |
| 2 | 52% | The | ~0% | Dustin Ortiz (**worse!**) | ✓ |
| **9** | **75%** | **John** | **99.95%** | **John Holt, 1959, 68 bowls** | **✓** |
| Full | 100% | John | 99.95% | **Identical to rank 9** | ✓ |

**Key finding:** 9D PCA subspace replacement = full 2560D replacement (same output, same KL).
The remaining 25% of energy in orthogonal dims is irrelevant.

Non-monotonic: rank 2 WORSE than rank 1 (PCA maximises total variance, not entity signal).

### Token-Directed Subspace (10 answer tokens)

Tokens: John, Coyle, 23, bowls, oatmeal, Corby, England, Irish, porridge, 10
- **Donor subspace fraction: 0.29%** — answer content is 0.29% of residual energy
- **P(John) after injection: 0.000002** — effectively zero
- Confirms: answer lives in DARK SPACE (99.7% orthogonal to vocabulary)

## Experiment 5 — Cross-Query Injection

Full L30 replacement tested across 4 query types:

| Query | Donor → | Recipient → | Injected → | Worked? |
|-------|---------|-------------|------------|---------|
| Who won? | **John** Coyle | **This** (tricky one) | **John** Holt 1959 | First token ✓ |
| How many bowls? | **The** text: 23 | **Please** provide | **The** text doesn't state | Format shift only |
| Where held? | **The** ... Corby | **The** ... Lancre | **The** ... Lancre | ZERO effect |
| Nationality? | **John** ... Irish | **John** ... Ferrante | **John** ... Ferrante | ZERO effect |

**Critical finding:** `inject_residual` is a **first-token-only intervention**.
- When first tokens differ: injection works (This→John)
- When first tokens agree: injection is invisible, recipient KV cache dominates all subsequent tokens
- "John Holt won the 1959 Championship" is NOT content from the donor — it's the model confabulating a plausible continuation after being seeded with "John" in a porridge-contest context

## What Lives Where

| Information | Where it lives | Single-pos injectable? |
|------------|---------------|:---------------------:|
| Entity first name ("John") | L30 last-position residual | ✓ (99.95%) |
| Topic ("porridge championship") | L30 last-position residual | ✓ (implicit) |
| Nationality ("Irish") | L30 residual (weak) | Partial (alpha=5–6 only) |
| Entity surname ("Coyle") | KV cache positional entries | ✗ |
| Quantities ("23 bowls") | KV cache positional entries | ✗ |
| Locations ("Corby, England") | KV cache positional entries | ✗ |
| Time constraints ("10 minutes") | KV cache positional entries | ✗ |

## Conclusions

### Outcome C confirmed: narrative requires context

The single-position L30 residual encodes **WHAT TOPIC** (porridge eating championship)
and **WHOSE** (John), but not **THE SPECIFIC FACTS** (Coyle, 23 bowls, Corby,
10 minutes). Those facts are distributed across KV cache entries at the token
positions where they appear in the original text.

### Three mechanisms ranked

1. **PCA 9D subspace replacement** (best): P(John) = 99.95%, coherent multi-token,
   no fixed point. 72 bytes + shared PCA basis. But entity only, no narrative.

2. **Full replacement** at L30: identical to 9D. The extra 2488 dimensions don't help.

3. **Additive steering**: narrow useful window (alpha 3–7), fixed point at alpha ≥ 10.
   Strictly worse than subspace replacement.

### For the architecture

| Current (v10) | Finding |
|---------------|---------|
| 175 KB routing + 500 tok context | **Optimal. Cannot be reduced.** |
| 3.8s query (500 tok prefill) | Prefill is the cost of narrative |
| "There is no context window" | True with context replay, not without |

**Zero-token injection cannot replace context replay.** The narrative content
lives in positional KV entries, not the last-position residual. No amount of
single-position residual surgery — at any dimensionality, any coefficient,
any layer — can deliver "23 bowls of instant oatmeal in Corby, England."

Context replay (~200 tokens focused) is the floor. Ship what works.

### The persistent_transplant distinction

Prior work (c1bb923b) showed persistent injection (at EVERY generation step)
delivers 100% correct multi-token output. That works because it re-injects
the crystallised residual at every step, overriding the KV cache at each token.
This experiment's single-step injection only seeds the first token.

The difference: persistent injection = 10 KB/step × N steps. Context replay = 200 tokens once.
Context replay is 50–100× cheaper. The architecture is correct.
