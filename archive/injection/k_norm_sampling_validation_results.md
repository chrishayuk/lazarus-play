# K-Norm Sampling Validation: Partial Fix, Deeper Problem

**Experiment ID:** 2a36b216-c9a1-4be8-8af3-546e446328b8
**Model:** google/gemma-3-4b-it (34 layers, 2560 hidden dim, 8 attn heads)
**Date:** 2026-03-20
**Prior:** apollo_routing_diagnosis (12b05e7b) — identified two problems: (1) interval sampling captures punctuation, (2) H4 is positional copy head not semantic retriever

## Executive Summary

K-norm sampling **does improve store quality** — content tokens get 5.2x more H4 attention
than punctuation, and equalized-position routing succeeds for 4/8 heads. However, a **third
problem** emerged that neither the diagnosis nor K-norm sampling addresses:

**The model's parametric knowledge overrides context for well-known topics.** Even with the
actual baseball scores in the prompt, the model confabulates a fictional story about
"Buzz Aldrin accidentally recording a Detroit Tigers game." This parametric override is the
binding constraint — routing and injection are downstream of a model that refuses to read
its own context.

## Results Summary

| Exp | Question | Answer |
|-----|----------|--------|
| 1 | Do content tokens get more H4 attention? | **YES.** 5.2x per-position, 12.4x per-sample |
| 2 | Does K-norm sampling capture better positions? | **YES.** 87.5% content (28/32) vs 68% interval (19/28) |
| 3a | Does H4 route correctly in full context? | **NO.** Routes to structural markers (CC/CDR) for ALL queries |
| 3b | Does routing work with equalized positions? | **PARTIAL.** 4/8 heads route correctly (keyword matching) |
| 5a | Does L30 injection shift generation? | **YES.** Mentions real cities (Boston, Baltimore, Detroit) from scores |
| 5b | Does injection deliver exact facts? | **NO.** Facts require KV cache, not residual stream |
| 5c | Does full context fix the model? | **NO.** Parametric override — confabulates even with correct context |

## Experiment 1 — K-Norm Distribution (Baseball Window)

**Method:** Used H4 attention weights from the last (query) position as K-norm proxy.
Chat template query: "What were the baseball scores during the Apollo 11 mission?"
447-token baseball window from Apollo 11 transcript.

### Content vs Punctuation Attention

| Category | Count | Mean H4 attn/position | Sum H4 attn |
|----------|-------|----------------------|-------------|
| Content positions (30) | 30 | 0.000858 | 0.0266 |
| Punctuation positions (48) | 48 | 0.000165 | 0.0082 |
| **Content/Punct ratio** | — | **5.2x** | **3.2x** |

Top content positions by H4 attention:
- Minneapolis (0.011), Norman (0.006), Thor (0.004), Houston (0.002), National (0.0006)

### K-Norm Sampling vs Interval Sampling

| Metric | K-norm (top-32) | Interval (every 16th) |
|--------|----------------|----------------------|
| Content tokens | 28 (87.5%) | 19 (68%) |
| Noise tokens | 4 (12.5%) | 9 (32%) |
| Mean H4 attention (excl BOS) | 0.003120 | 0.000251 |
| **Ratio** | — | **12.4x** |

**K-norm sampling selects:**
Minneapolis, snow, CC, Roger, Norman, Thor, weather, radio, CMP, yesterday, sports,
Thursday, Houston, St, navigator, repairing, sleep, today...

**Interval sampling selects:**
weather, has, a, the, comma, on, was, footing, and, is, period, endurance, Shen, 0,
showing, Lem, yesterday, space, semicolon, American, 4, Boston, Irishman, space, from...

K-norm captures 12.4x more relevant content per sample.

## Experiment 3a — Full-Context Routing (3 Windows)

**Method:** Concatenated 3 transcript windows (scratchy + baseball + landing) with
chat-template query. Measured per-window H4 attention.

### H4 Routes to Structural Markers, Not Content

| Query | W1(scratch) | W2(baseball) | W3(landing) | H4 Winner | Correct? |
|-------|------------|-------------|------------|-----------|----------|
| Baseball scores | 17.5% | **2.4%** | 16.2% | W1 | NO |
| Eagle landed | 35.3% | 2.3% | 19.5% | W1 | NO |
| Audio quality | 34.1% | 1.4% | 14.6% | W1 | YES (accident) |
| Capsule communicator | 32.1% | 1.4% | 18.1% | W1 | N/A |

**H4 routes to W1 for ALL queries** — attending to CC/CDR/LMP structural tokens
(densest in W1), not content.

**All other heads (H0-H3, H5-H7) route to W3 for ALL queries** — pure RoPE
positional bias (W3 is nearest to query).

**Zero heads do content-dependent routing in full context.**

## Experiment 3b — Equalized-Position Store Routing

**Method:** Created 6 compact "store entries" (keyword summaries) at equal distances
from the query. Removes RoPE positional confound.

### Equalized Routing: 4/8 Heads Route Correctly

| Head | Routes to | Content attn | Correct? |
|------|-----------|-------------|----------|
| H0 | **A(baseball)** | 0.0057 | YES |
| H1 | **A(baseball)** | 0.0072 | YES |
| H2 | E(sports) | 0.0044 | no |
| H3 | D(technical) | 0.0029 | no |
| H4 | F(scratchy) | 0.0485 | no — "Apollo" keyword match |
| H5 | F(scratchy) | 0.0087 | no — "Apollo" keyword match |
| H6 | **A(baseball)** | 0.0002 | YES |
| H7 | **A(baseball)** | 0.0001 | YES |

**Key insight:** With equalized positions, heads do **keyword-based routing** —
matching tokens that appear in both query and entry. H4 and H5 are misled by
"Apollo" appearing in Entry F (scratchy section also mentions Apollo).

**Improvement:** From 0/8 correct (full-context) to 4/8 correct (equalized).
The routing mechanism works when RoPE positional bias is removed.

## Experiment 5 — Injection and Parametric Override

### Three-Way Generation Comparison

**Parametric only (no context):**
> "Houston Mission Control was trying to listen to a baseball game — the Houston Astros
> versus the New York Mets..."

Completely confabulated. Wrong teams, wrong narrative.

**Full context (actual scores in prompt):**
> "Buzz Aldrin, while on the moon, was using a portable tape recorder... He accidentally
> recorded a baseball game — specifically, a Detroit Tigers game..."

**Still confabulates despite having the correct scores in context.** The model's
parametric association "Apollo 11 + baseball" produces a fictional narrative that
overrides the actual transcript data.

**L30 single-position injection:**
> "NASA intentionally broadcast baseball scores... Boston Red Sox vs. Baltimore Orioles
> (Score: Orioles 5, Red Sox 3)... New York Yankees vs. Detroit Tigers..."

Injection **shifted the topic** — mentions Boston, Baltimore, Detroit (cities that
appear in the actual scores: Baltimore 3, Detroit 4, Boston at NY). But wraps them in
a confabulated narrative. KL from donor: 4.4. KL from recipient: 21.1.

**Full Markov injection (patch_all_positions=True):**
- Perfectly replicates donor output (KL = 0.0)
- But donor itself confabulates — Markov holds but is useless when the donor fails

### The Parametric Override Problem

| Condition | Reads context? | Output quality |
|-----------|---------------|----------------|
| Parametric only | N/A | Confabulated (Astros vs Mets) |
| Full context | **NO** | Confabulated (Tigers recording) |
| L30 injection | Partially | Shifted topic, still confabulated |
| Full Markov | Matches donor | Donor itself fails |

**The model treats "Apollo 11 baseball" as a famous anecdote and generates its
parametric version regardless of context.** This is a Type 2 confabulation (L33 FFN
spike) — the parametric signal is too strong to override.

## Root Cause: Three Compounding Problems

### Problem 1: Sampling (SOLVED by K-norm)
- Interval sampling captures punctuation, K-norm captures content
- 12.4x improvement in store quality
- **Fix: sample at high-K-norm positions**

### Problem 2: Positional Bias (PARTIALLY SOLVED by store format)
- Full-context routing dominated by RoPE distance and structural markers
- Equalized-position store removes RoPE bias: 4/8 heads route correctly
- **Fix: store format inherently equalizes positions**

### Problem 3: Parametric Override (UNSOLVED)
- Model confabulates "Apollo 11 baseball" even with correct context
- This is upstream of routing and injection — the model doesn't read its context
- L30 injection can shift topic but not override parametric knowledge
- **No fix within the current architecture**

## What Each Outcome Means

**K-norm sampling partially works (Exp 1-2: YES, Exp 3b: PARTIAL):**
- Ship K-norm sampling for store construction — 12.4x better content capture
- Use equalized-position store format (already planned)
- Multi-head consensus (not just H4) for routing

**But the binding constraint is parametric override (Exp 5):**
- For well-known topics where the model has strong parametric associations,
  context injection cannot override the confabulation
- This affects all injection architectures, not just K-norm
- The model needs to be taught to prefer context over parametric knowledge
  (potentially via steering or fine-tuning)

## Recommended Path Forward

1. **K-norm sampling: ADOPT.** Replace interval sampling with top-N by K-norm.
   12.4x better content capture, 87.5% content tokens vs 68%.

2. **Equalized-position store: KEEP.** The store format already removes RoPE
   bias. 4/8 heads route correctly in equalized format vs 0/8 in full context.

3. **Multi-head routing: INVESTIGATE.** H0/H1/H6/H7 route correctly for
   baseball when equalized. H4 alone is insufficient — use head ensemble.

4. **Parametric override: UNSOLVED.** For topics where the model has strong
   parametric associations, injection cannot override confabulation. Options:
   - Steering vector to suppress parametric knowledge
   - Fine-tuning to teach context preference
   - Prompt engineering to force context reading
   - Different query formulation that doesn't trigger parametric association

5. **Novel facts: UNAFFECTED.** The parametric override only applies to
   well-known topics. For novel facts (Zarkov-style), the architecture works
   because there's no parametric knowledge to override. The Apollo transcript
   baseball scores are a worst case — the model "knows" (incorrectly) about
   Apollo 11 and baseball.

## Key Numbers

- **5.2x** — content vs punctuation H4 attention ratio per position
- **12.4x** — K-norm vs interval sampling mean attention ratio
- **87.5%** — content tokens in K-norm top-32 (vs 68% interval)
- **4/8** — heads that route correctly with equalized positions
- **0/8** — heads that route correctly in full context (RoPE bias)
- **4.39** — KL divergence from donor after L30 injection (shifted toward correct topic)
- **21.15** — KL divergence from recipient after L30 injection (far from parametric)
- **0.0** — KL divergence from donor after full Markov injection (perfect match)
- **0/3** — conditions where the model reads the actual baseball scores from context
