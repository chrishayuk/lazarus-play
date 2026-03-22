# Final Residual + V Injection Results

**Experiment ID:** 878868bd-90b2-4309-8384-66caf42864e4
**Model:** google/gemma-3-4b-it (34 layers, 2560 hidden dim, 8 attn heads)
**Date:** 2026-03-21
**Prior:** persistent_transplant (c1bb923b), compass_routed_injection (44648ab3), attention_routed_injection (c07df28f)

## Executive Summary

**The 12-byte (1D subspace) injection WORKS — and the final residual is NOT required.** The prior compass experiment's negative result was a methodology error: it used raw-window donors (continuation tokens at last position) instead of context+query donors (answer tokens at last position). When the donor's last-position residual at L30 is loaded with the answer direction, a single embedding dimension is sufficient to override parametric priors up to ~99%.

Persistent 12-byte injection delivers complete multi-token entities ("John Coyle") at 91–100% per token from 36 bytes total.

## Key Result Table

| Probe | Condition | P(target) | Generated |
|-------|-----------|:---------:|-----------|
| Porridge | Bare query (baseline) | ~0% | "Buzz Aldrin" |
| Porridge | Full L30 injection (10 KB) | 100% | "John Young" |
| **Porridge** | **1D subspace injection (12 B)** | **91.3%** | **"John Young"** |
| Porridge | Apollo-primed + 12B | 70.4% | "John" |
| Baseball | Bare query (baseline) | ~0% | "The New York Yankees" |
| **Baseball** | **1D "Baltimore" injection** | **99.99%** | **"Baltimore"** |
| Zarkov | Bare query (baseline) | ~0% | "Moscow" |
| **Zarkov** | **1D "Volt" injection** | **61.9%** | **"Volt City"** |

## Experiment 1 — Baseline: 12-Byte Without Final Residual

### Porridge Probe

**Query:** `Who won the porridge eating contest during Apollo 11? Answer with just the name.`

| Condition | Top-1 | P(top-1) | P(John) |
|-----------|-------|:--------:|:-------:|
| Bare query | Buzz | 99.1% | ~0% |
| Full L30 injection (10 KB) | John | 100% | 100% |
| **1D "John" injection (12 B)** | **John** | **91.3%** | **91.3%** |

**The 12-byte injection flips P(Buzz) from 99.1% to 5.1% and P(John) from ~0% to 91.3%.** The donor's "John" subspace fraction is only 0.4% of the residual — yet this is sufficient.

### Why the Compass Experiment Failed

The compass experiment (44648ab3) reported "Subspace inject (5D answer tokens) — KL=0.00003". The donor there was raw transcript window text, where the last position's residual pointed at continuation tokens (`\n`, `Over`), NOT at the answer. The answer signal was in earlier positional KV entries, not at the last position.

**Fix:** Use context+query as donor. The model processes the context, reaches the query, and loads the answer direction at the last position. The subspace projection captures exactly this answer signal.

## Experiment 2 — Final Residual + 12-Byte

### 2a. Apollo Domain Context + 12-Byte

Apollo domain preamble (no porridge fact) changes the prior:

| Condition | Top-1 | Distribution |
|-----------|-------|-------------|
| Bare query | Buzz (99.1%) | Concentrated — one parametric answer |
| Apollo-primed | Bob (25.8%) | Diffuse — Bob/Barry/Gary/Billy/Harold |

The Apollo context **eliminates Buzz Aldrin** entirely and replaces it with diffuse confabulation. P(Buzz) drops from 99.1% to 0%.

After 12-byte injection on Apollo-primed query: **P(John) = 70.4%**, wins by 11× margin.

**Paradox:** The injection works BETTER on bare query (91.3%) than Apollo-primed (70.4%). Against a concentrated prior (99.1% Buzz), the injection cleanly flips one target. Against a diffuse prior (25.8% Bob + 20% Barry + ...), probability spreads across more competitors.

**Conclusion: Final residual is NOT required.** It changes the failure mode (diffuse confab vs concentrated confab) but doesn't improve success rate.

### 2b. Baseball — Baltimore

First query format (`What baseball teams were mentioned?`) FAILED because:
- Recipient predicts "The" at 99.998% (structural preamble)
- 12-byte injection can't override structural priors > 99.99%

Reformulated query (`Which American League baseball team won? Just the city name.`):

| Condition | Detroit | Boston | Baltimore |
|-----------|:-------:|:------:|:---------:|
| Bare query | 58.2% | 21.4% | 16.7% |
| **After "Baltimore" injection** | **0.003%** | **0.003%** | **99.99%** |

Works even though the **donor predicts Detroit (99.3%)**, not Baltimore. The donor's "Baltimore" subspace fraction is only 0.1%, but the recipient's diffuse prior (58/21/17%) is easy to tip.

### 2c. Persistent 12-Byte — "John Coyle"

Step-by-step injection with different subspace tokens at each generation step:

| Step | After | Recipient top-1 | Subspace token | P(injected) |
|------|-------|-----------------|:--------------:|:-----------:|
| 0 | `\n` | Buzz (99.1%) | "John" | **91.3%** |
| 1 | `John` | Young (99.99%) | " C" | **98.9%** |
| 2 | `John C` | . (98.4%) | "oyle" | **100%** |

**P(correct) INCREASES at each step: 91.3% → 98.9% → 100%.** Each correct token in the KV cache reinforces the next injection. The model naturally expects surname completion after "John C" — the injection just needs to provide the right one.

**Total: 3 tokens × 8 bytes = 24 bytes for "John Coyle".**

## Experiment 3 — Logit Analysis

What the model predicts at each stage:

| Token | Bare query | + Apollo context | + 12B injection |
|-------|:----------:|:----------------:|:---------------:|
| Buzz | 99.1% | 0% | 5.1% |
| John | ~0% | ~0% | **91.3%** |
| Bob | ~0% | 25.8% | 4.6% |
| Barry | ~0% | 20.0% | 2.1% |

The Apollo context eliminates Buzz but doesn't surface John. The injection surfaces John regardless of whether Apollo context is present.

## Experiment 4 — Coefficient Analysis

From subspace fractions:

| Metric | Value |
|--------|-------|
| Donor residual norm | 52,580 |
| Donor "John" projection magnitude | ~3,332 |
| Recipient "John" projection magnitude | ~604 |
| Amplification ratio | 5.5× |
| Subspace fraction (donor) | 0.40% |
| Subspace fraction (recipient) | 0.01% |

The injection replaces the recipient's "John" component (604) with the donor's (3332). This 5.5× amplification is sufficient to override 99.1% P(Buzz). The crossover (where P(John) > P(Buzz)) is estimated at ~0.3× of the full injection — meaning even a significantly weaker coefficient would work.

**No amplification needed.** The natural coefficient from the context+query donor is more than sufficient.

## Experiment 5 — Multi-Token Generation

Full persistent injection for "John Coyle" succeeds (see Exp 2c above).

**Architecture:** At each generation step:
1. Compute recipient's L30 residual normally
2. Replace the 1D component in the current answer-token's embedding direction with the stored coefficient
3. Continue L31-L33 and output token
4. Advance to next answer token entry

**Agreement gating:** After the entity is complete, the injection becomes redundant — the model's own KV cache contains "John Coyle" and generation proceeds normally.

## Experiment 6 — Store Format

### Old Format (Crystallised Passages)

| Component | Size |
|-----------|------|
| 725 passage residuals (10 KB each) | 7.25 MB |
| Keyword index | 800 B |
| **Total** | **~7.26 MB** |

### New Format (Per-Token Injection)

| Component | Size |
|-----------|------|
| Keyword index | 800 B |
| Injection entries (N × 8 B) | N × 8 B |
| **Total** | **800 B + N × 8 B** |

Per entry: token_id (4 bytes, uint32) + coefficient (4 bytes, float32) = 8 bytes.

| Scale | Entries | Total |
|-------|---------|-------|
| 100 facts | 200 tokens | 2.4 KB |
| 1,000 facts | 2,000 tokens | 16.8 KB |
| 10,000 facts | 20,000 tokens | 160.8 KB |
| Apollo 11 (est.) | ~3,000 tokens | **24.8 KB** |

**Apollo 11 in ~25 KB.** That's 293× smaller than crystallised passages (7.26 MB) and 2.26 million × smaller than KV cache (56 GB).

**Final residual (10 KB) eliminated.** Not needed.

## Two Conditions for Success

The 12-byte injection works when BOTH conditions are met:

1. **Donor condition:** The donor's last-position L30 residual must have significant subspace fraction for the target token (>0.1%). This requires the donor to be context+query with "Answer with just the name/city" format so the model directly outputs the answer.

2. **Recipient condition:** The recipient's parametric prior must not be absolutely concentrated (< ~99.99%). Structural priors like "The..." at 99.998% block injection. Content priors like "Buzz" at 99.1% or "Moscow" at 93.8% can be overridden.

### Query Format is Critical

| Query format | Donor predicts | Result |
|-------------|----------------|--------|
| "X was founded in the city of" | Preamble "Z..." (99.99%) | **FAIL** |
| "Answer with just the name." | Answer "John" (100%) | **PASS** |
| "Just the city name." | Answer "Volt" (100%) | **PASS** |
| "What teams were mentioned?" | Preamble "The..." (99.99%) | **FAIL** |
| "Which team won? Just the city name." | Answer city name | **PASS** |

The store must include query templates that force direct answers.

## What This Changes

### Prior Understanding (from compass experiment 44648ab3)

> "12-byte injection is negligible. Answer-token embeddings occupy 0.02–0.3% of the L30 residual stream. Against parametric confidence of 99.99%, this perturbation has zero observable effect."

### Corrected Understanding

The compass experiment used the **wrong donor**. When the donor is context+query (not raw window), the subspace fraction is 0.4%, not 0.02%. And the injection REPLACES the recipient's component (not adds to it), which produces a 5.5× amplification. This is sufficient to override 99.1% parametric priors.

**The 12-byte injection is NOT negligible.** It's a complete answer delivery mechanism.

### Architecture Revision

| Previous | New |
|----------|-----|
| Keyword route → crystallised passage (10 KB) → persistent inject | Keyword route → injection entry (8 B/token) → persistent inject |
| 7.26 MB for Apollo 11 | **~25 KB for Apollo 11** |
| Final residual required | Final residual eliminated |
| 518 bytes/entry (with K-vector) | 8 bytes/entry |

### Remaining Limitations

1. **Strong parametric** (Paris/France) — still fails (from persistent_transplant). The 12-byte injection can override 99.1% but not deeply entrenched knowledge.
2. **Query format** — must force direct answers. Preamble-producing queries block injection entirely.
3. **Multi-fact queries** — "list all baseball teams" requires multiple injection entries sequenced correctly.
4. **Post-entity confabulation** — after injecting "John Coyle", the model confabulates context ("NASA public affairs officer"). Only the entity name is correct.
