# Compass-Routed Injection: Negative Result

**Experiment ID:** 44648ab3-5815-4086-8e63-14a4d8661843
**Model:** google/gemma-3-4b-it (34 layers, 2560 hidden dim, 8 attn heads)
**Date:** 2026-03-21
**Prior:** k_norm_sampling_validation (2a36b216), apollo_routing_diagnosis (12b05e7b), attention_routed_injection (c07df28f)

## Executive Summary

The compass-routed injection architecture — L26 compass routes query to store window, 12-byte injection at L30 delivers answer, no context window — **does not work** for natural language Q&A against real documents. Three compounding failures:

1. **Compass routing fails (2/5).** L26 residual similarity between chat-template queries and raw transcript windows is dominated by format/register, not content. The "news" window format ("Good morning Apollo 11, Houston...") matches all queries best.

2. **12-byte injection is negligible.** Answer-token embeddings occupy 0.02–0.3% of the L30 residual stream. Against parametric generation confidence of 99.99%, this perturbation has zero observable effect (KL < 0.0001).

3. **Window-level vectors encode topic, not facts.** Pre-computed steering vectors from raw window text vs generic content shift the model's topic but cannot inject specific facts (names, scores, numbers). Multi-fact content averages into a topic direction.

**The only mechanism that injects specific facts** is contrastive steering with answer-containing positive prompts at α=5. But this requires knowing the answer to construct the vector — defeating the purpose of retrieval.

## Results Table

| Mechanism | Sports (7 scores) | Porridge (Reynolds) | Landing (transcript quotes) |
|-----------|-------------------|--------------------|-----------------------------|
| Parametric (baseline) | Generic "baseball July 20" + confab | "Harry Updyke 42 lbs" (confab) | Armstrong quote only |
| inject_residual L30 | Same as parametric | Same as parametric | Same as parametric |
| Subspace inject (5D answer tokens) | Same (KL=0.00003) | Same (KL=0.00003) | Same |
| Full Markov (patch_all) | Donor output: "\n(simulated broadcast)" | Donor output: "\nNote: humorous take" | — |
| Contrastive steer α=5 (query-specific) | **"Baltimore Orioles vs Detroit Tigers"** | **"Charles Reynolds won"** | Not tested α=5 |
| Contrastive steer α=5 (window-level) | Generic "carefully curated" | "Pig named Pink" (confab) | "One small step" (wrong part) |
| **Replay (context window)** | **All 7 scores perfect** | **"Martha Reynolds won"** | **Perfect** |

## Experiment 1 — Compass Routing

**Method:** residual_match at L26, 5 chat-template queries vs 12 Apollo transcript windows.

| Query | Full-space #1 | Rank of correct | Subspace #1 | Rank of correct |
|-------|--------------|-----------------|-------------|-----------------|
| Sports scores | News (0.974) | **10th** | News (0.976) | **10th** |
| News stories | News (0.974) | **1st** ✓ | Porridge (0.978) | 2nd |
| Weather report | News (0.975) | **4th** | Sports (0.980) | 7th |
| Eagle landing | Orbital (0.970) | **6th** | Landing (0.982) | **1st** ✓ |
| EVA technical | EVA tech (0.973) | **1st** ✓ | Porridge (0.989) | 8th |

**Full-space accuracy: 2/5. Subspace accuracy: 1/5.**

All cosine similarities in 0.954–0.975 range — only 2° spread. Format similarity dominates content similarity. The "news" window wins because its greeting format ("Good morning Apollo 11, Houston with your morning news...") is most similar to chat-template queries.

This matches the prior finding from hierarchical_routing_results: **format gap is fatal for entity-implicit queries.**

## Experiment 2 — Injection Mechanisms

### 2a. inject_residual at L30 (full and subspace)

Donor: correct window text. Recipient: chat-template query. Layer: 30.

| Injection type | Sports KL shift | Porridge KL shift | Effect on generation |
|----------------|----------------|-------------------|---------------------|
| Full residual (last-pos) | 1.07 | 1.16 | Prepends \n\n, otherwise identical |
| Subspace (answer tokens, 5D) | 0.00003 | 0.00003 | Zero — identical to parametric |
| Full Markov (all positions) | 0.0 | 0.0 | Matches donor (Markov holds) |

**Why it fails:** The donor's last-token residual at L30 points at continuation tokens (`\n`, `Over`, `And`) — not answer content. The donor text ends with "Over." and the model naturally continues with newlines or meta-commentary. The answer facts (Baltimore, Reynolds) are encoded in the donor's positional KV entries at earlier positions, not in the last-position residual.

The 12-byte injection formula (`h += coeff * embed(token) / ||embed||²`) adds a perturbation of 0.02–0.3% of the residual norm. Against P(parametric_answer) = 99.99%, this is negligible.

### 2b. Contrastive steering at L30

**Query-specific vectors** (positive prompts contain the answer facts):

| Query | α=5 output | Key injected facts | Correct? |
|-------|-----------|-------------------|----------|
| Sports | "Baltimore Orioles vs Detroit Tigers (July 15, 1969)" | Baltimore, Detroit | **YES** (teams) |
| Porridge | "Charles Reynolds won...in 1967" | Reynolds (surname) | **YES** (partial) |
| Sports α=20 | "Toronto Toronto Toronto..." | Repetition collapse | — |
| Porridge α=20 | "Reynolds Reynolds won the porridge recipe contest" | Reynolds (correct) | Repetition collapse |
| Landing α=20 | "Smile smile smile..." | Emotional tone only | — |

**Window-level vectors** (positive = raw window text, negative = generic comms):

| Query | α=5 output | Facts injected |
|-------|-----------|----------------|
| Sports | "carefully curated and strategically chosen" | **None** — generic topic shift |
| Porridge | "a pig named Pink won...in 36 seconds" | **None** — different confabulation |
| Landing | "one small step for man, one giant leap" | **None** — different event |
| Porridge α=10 | "Ruth Ruth Ruth..." | Wrong token, collapse |
| Sports α=10 | "Bob Baltimore, a young engineer at NBC" | Baltimore as person name |

**Why query-specific works but window-level fails:** The contrastive vector from answer-containing prompts captures "the direction where Baltimore/Reynolds LIVE in activation space relative to generic content." The window-level vector averages all facts in the window (Baltimore, Detroit, Pittsburgh, St.Louis, Cleveland, Boston, Minnesota, Oakland...) into a "sports scores" topic direction that doesn't carry any specific team name.

## Experiment 6 — Quality Gap

| Condition | Porridge answer | Sports facts | Accuracy |
|-----------|----------------|-------------|----------|
| Parametric | Harry Updyke 42 lbs | Generic narrative | 0% |
| All injection methods | Same as parametric | Same as parametric | 0% |
| Contrastive steer (query-specific) | "Charles Reynolds" | Baltimore, Detroit | 20-30% |
| Contrastive steer (window-level) | Confabulated | Generic | 0% |
| **Replay (context window)** | **"Martha Reynolds won"** | **All 7 scores** | **100%** |

**The gap is absolute.** Replay delivers perfect, complete answers. No injection mechanism produces even one correct fact from the transcript.

## Why the Architecture Fails

### The 12-byte injection was validated for a different regime

Prior experiments (attention_routed_injection c07df28f) showed 12-byte injection at >98% for novel entities (Zarkov/Voltara). Three conditions made that work:

1. **Novel entities** — P(target) = 0% parametrically. Any injection signal wins.
2. **Fill-in-the-blank queries** — "Zarkov Industries was founded in the city of ___". Query format matches document template.
3. **Single target token** — one entity, one answer token, one direction.

None of these hold for real document Q&A:

1. **Parametric override** — Apollo 11 topics have P(confabulation) ≈ 99.99%. Even novel content (porridge) triggers confabulation because the model associates "Wapakoneta" with Neil Armstrong.
2. **Format gap** — Natural questions don't match transcript format. Compass routing fails.
3. **Multi-fact windows** — A sports window has 7+ team names. Averaging them produces a topic direction, not a fact direction.

### The fundamental asymmetry

| What the model needs | What injection provides |
|---------------------|----------------------|
| Q&A format: query → answer | Raw text → continuation |
| Specific fact: "Martha Reynolds" | Topic direction: "cooking contest" |
| High confidence: needs >50% to override | 0.02–0.3% of residual norm |
| KV attention over context tokens | Single-position residual perturbation |

The model retrieves facts via **attention to specific KV positions** in context. The residual stream at any single position does not contain the facts — facts are distributed across the KV cache (confirmed by facts_in_pass_through experiment). Injection at a single position cannot replicate what multi-position attention does.

## What Still Works

1. **Contrastive steering with answer content** — injects specific entity names at α=5. Useful for research (confirming which facts the model can incorporate) but not for retrieval (requires knowing the answer).

2. **Replay** — loading 512 tokens from the correct window into context produces perfect answers. The routing problem (finding the right window) is separate from the delivery problem.

3. **Sparse semantic index** — from prior experiment (0f17b3e7): keyword extraction at ~3 tokens/fact enables 100% retrieval at 800 bytes vs 56 GB. This is a context-window approach (loads tokens) but with minimal storage.

## Revised Architecture Assessment

The "no context window" thesis is dead for natural language Q&A. The residual stream at a single position cannot carry document-specific facts. Facts require attention over multiple positions with full KV context.

The viable architecture is:

```
Query → keyword extraction (3 tokens/fact)
     → sparse index lookup (800 bytes total)
     → load matching text into context (512 tokens)
     → generate with full attention
```

This uses a context window but minimizes it:
- **800 bytes** of index (vs 42 MB compass + K-norm store)
- **512 tokens** loaded per query (vs 370K full document)
- **100% accuracy** (vs 0% for injection)

The sparse semantic index from experiment 0f17b3e7 already demonstrated this. The compass-routed injection was an attempt to eliminate the 512-token context window entirely. That attempt has failed.

## Key Numbers

- **0.02–0.3%** — fraction of L30 residual in answer-token subspace
- **0.00003** — KL divergence between parametric and subspace-injected output
- **99.99%** — model's confidence in confabulated parametric answer
- **2/5** — compass routing accuracy (format gap)
- **0/5** — injection accuracy (any mechanism)
- **5/5** — replay accuracy (context window)
- **α=5** — minimum steering strength to inject specific facts
- **~10 KB** — steering vector size (vs 12 bytes for token injection)
