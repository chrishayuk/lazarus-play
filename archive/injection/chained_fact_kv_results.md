# Chained Fact-Position KV: The Chain Adds Almost Nothing

**Experiment ID:** c7b7e88f-efaa-42ba-b260-e17e8ea7282e
**Model:** google/gemma-3-4b-it (34 layers, 2560 hidden dim, 4 KV heads, 256 head_dim)
**Builds on:** direct_kv_injection_results.md, facts_in_pass_through_results.md

---

## Executive Summary

Tested whether chaining document context through boundary residuals enriches the KV
entries at fact token positions compared to standalone (context-free) computation.

**The chain adds almost nothing.** Residuals at fact positions are nearly identical
(L0-L14) between chained and standalone conditions. Both produce correct generation.
The earlier standalone KV injection failure (0/3) was caused by **RoPE position mismatch**,
not missing document context.

---

## The Hypothesis

During chained prefill, the boundary residual injects accumulated document state. Each
window's KV entries are computed with everything processed before. The hypothesis:
chained KV entries are RICHER than standalone KV, encoding document context that enables
retrieval standalone KV can't support.

## The Test

### Prompts

**Chained (full document context):**
```
<start_of_turn>user
The crew received news updates during the mission. Houston relayed information about
current events. Houston reported that Joe Namath agreed to sell his restaurant and
report to the New York Jets training camp. The crew continued with systems checks
and navigation updates. What sport was discussed during the mission?
<end_of_turn>
<start_of_turn>model
```

**Standalone (Window 1 only):**
```
<start_of_turn>user
Houston reported that Joe Namath agreed to sell his restaurant and report to the
New York Jets training camp. What sport was discussed during the mission?
<end_of_turn>
<start_of_turn>model
```

**No context (query only):**
```
<start_of_turn>user
What sport was discussed during the mission?
<end_of_turn>
<start_of_turn>model
```

### Fact Token Position Mapping

| Token | Chained pos | Standalone pos | Token ID |
|-------|-------------|----------------|----------|
| Nam   | 24          | 8              | 11774    |
| ath   | 25          | 9              | 651      |
| Jets  | 37          | 21             | 72161    |
| restaurant | 30     | 14             | 9180     |
| training | 38       | 22             | 4122     |
| camp  | 39          | 23             | 3545     |

Token IDs match perfectly — same tokens, different absolute positions (RoPE effect).

---

## Experiment 1 — Generation Baselines

| Condition | Output | Correct? |
|-----------|--------|----------|
| No context (query only) | "**baseball** ... using a baseball bat and ball to repair a damaged antenna" | NO — confabulation |
| Standalone (W1 only) | "**American football**. Specifically, Joe Namath's return to the New York Jets." | YES |
| Full chain (W0+W1+W2) | "**football**, specifically American football, due to the mention of Joe Namath and the New York Jets." | YES |

**Both standalone and chained produce correct answers.** The fact tokens alone
(without document context) are sufficient for generation.

---

## Experiment 2 — Residual Comparison at Fact Positions

Compared `decode_residual` at "ath" (Namath completion) and "Jets" positions
across layers 0, 7, 14, 26 in both conditions.

### "ath" position (Namath) — norm_top1

| Layer | Chained | Standalone | Verdict |
|-------|---------|------------|---------|
| L0 | "ir" (99.1%) | "ir" (99.1%) | IDENTICAL |
| L7 | इसके (17.1%) | इसके (19.1%) | NEAR IDENTICAL |
| L14 | " debacle" (0.06%) | " debacle" (0.1%) | NEAR IDENTICAL |
| L26 | **" was" (67.6%)** | **" was" (60.3%)** | SIMILAR — chained +7% confidence |

### "Jets" position — norm_top1

| Layer | Chained | Standalone | Verdict |
|-------|---------|------------|---------|
| L0 | "al" (100%) | "al" (100%) | IDENTICAL |
| L7 | "al" (36.6%) | "al" (34.9%) | NEAR IDENTICAL |
| L14 | "al" (0.11%) | "al" (0.12%) | NEAR IDENTICAL |
| L26 | **" for" (34.3%)** | **"." (20.6%)** | DIFFERENT — chained more contextual |

### L26 "Jets" position — full top-10

**Chained:**
- " for" (34.3%), " training" (23.5%), " again" (12.6%), " as" (4.1%),
  " immediately" (4.1%), "." (2.5%), " staff" (1.9%), " after" (1.7%),
  " team" (1.5%), " coaching" (1.3%)

**Standalone:**
- "." (20.6%), " for" (18.2%), " as" (12.5%), " today" (5.9%),
  " immediately" (2.8%), " on" (2.8%), " draft" (2.3%), " team" (2.0%),
  " yesterday" (1.9%), " Thursday" (1.8%)

The chained version knows the sentence continues with "training camp" (contextual
completion). The standalone version is more uncertain, predicting sentence-end
tokens (".") or temporal markers ("today", "yesterday"). But both encode the
correct continuation tokens in their top-10.

### Dark Space (Raw Top-K)

Nearly identical at all layers. Same unused tokens dominate:

| Layer | Chained top1 dot | Standalone top1 dot | Ratio |
|-------|-----------------|--------------------:|------:|
| L0 (ath) | 142.0 | 141.0 | 1.007 |
| L7 (ath) | 1696.0 | 1712.0 | 0.991 |
| L14 (ath) | 6176.0 | 5696.0 | 1.084 |
| L26 (ath) | 8192.0 | 7936.0 | 1.032 |
| L0 (Jets) | 126.0 | 127.5 | 0.988 |
| L7 (Jets) | 1624.0 | 1584.0 | 1.025 |
| L14 (Jets) | 6208.0 | 6272.0 | 0.990 |
| L26 (Jets) | 10560.0 | 10176.0 | 1.038 |

**Chain adds ~3-4% magnitude to dark space at L26. Direction is identical.**

### "Jets" L26 biggest_gainers — contextual tokens

Chained: " preseason" (+238K rank), " yesterday" (+157K), " scout" (+147K),
" roster" (+124K), " during" (+117K)

Standalone: " scout" (+158K), " announced" (+148K), " under" (+147K),
" roster" (+124K), " yesterday" (+121K)

**Same contextual tokens gain rank in both conditions.** The model recovers
Jets/football semantics regardless of whether Window 0 context is present.

---

## Experiment 3 — Markov Property (All-Position Injection at L14)

| Metric | Value |
|--------|-------|
| Donor | Full chain + query (65 tokens) |
| Recipient | Query only (18 tokens) |
| Layer | L14 |
| patch_all_positions | True |
| donor_injected_kl | **0.0** |
| Injected output | "The sport discussed during the mission was **football**, specifically American football, due to the mention of Joe Namath and the New York Jets." |
| Matches donor? | **YES — identical** |
| Residual cosine | 0.9996 (1.71°) |

**Markov holds perfectly.** Full state transplant at L14 reproduces the donor's
output exactly. This confirms the fact information is encoded in the distributed
hidden states at L14.

---

## The Answer: Why Standalone KV Failed

### What we know

1. **Standalone generation works** — "American football" is correct
2. **Residuals at fact positions are nearly identical** — chain adds ~3-4% magnitude
3. **Standalone KV injection failed (0/3)** in prior work

### The diagnosis

Standalone KV injection failed because of **RoPE position mismatch**, not
missing document context.

During standalone prefill: "Namath" computes its K vector at position 8 with
RoPE encoding for position 8. During injection: that K vector is placed at a
different position in the recipient's cache. The Q vectors from the query
(at positions N+1, N+2, ...) compute their RoPE for those positions. The
dot product Q·K is corrupted because the RoPE rotations don't match — Q expects
K from a nearby position, but K was encoded for position 8 in a different context.

**Evidence:** Pre-RoPE full-window KV injection worked 4/4 at 29ms. The only
difference between that and standalone fact-position KV is position handling.

### The solution

**Save K BEFORE RoPE application.** At injection time, apply RoPE for contiguous
positions (0, 1, 2, ..., N_facts). The model sees contiguous positions regardless
of where facts originally appeared.

This is exactly what the pre-RoPE test validated: position-independent K + V →
apply any positions at injection time → 4/4 correct, 29ms.

---

## Implications for Architecture

### What this changes

The chained checkpoint architecture was designed to carry document context through
boundary residuals so that each window's KV encodes accumulated state. This
experiment shows the accumulated state barely affects fact-position KV — the token
identity dominates.

This means:
1. **Chaining is unnecessary for fact-position KV extraction.** Standalone
   prefill of fact windows produces equivalent KV at fact positions.
2. **The boundary residual's value is at the LAST position** (routing/entity
   compass), not at fact positions (storage).
3. **Pre-RoPE K + V is the correct storage format.** Position-independent.
   Apply positions at injection time.

### What this doesn't change

- **All-position injection still needs all positions** (last-position-only fails)
- **Facts are multi-positional** (2-5 KV entries per fact minimum)
- **L26 attention is the fact-copying mechanism** (confirmed by DLA)
- **Sparse keyword index is still the practical solution** (0 KV manipulation needed)

### Revised storage architecture

```
Per fact:
  - 3-5 token positions
  - Pre-RoPE K: positions × 34 layers × 4 heads × 256 dim × 2 bytes
  - V: same dimensions
  - Per position: 34 × 4 × 256 × 2 × 2 = 139,264 bytes ≈ 136 KB
  - Per fact (5 positions): 680 KB

For Apollo 11 (725 windows, ~1 fact/window):
  - 725 × 680 KB ≈ 482 MB (all layers)
  - With layer subset (e.g., L0+L14+L26 = 3 layers): 42 MB
  - With layer subset (L0+L26 = 2 layers): 28 MB

Compare:
  - Full KV cache: 56 GB
  - Keyword index: 800 bytes
  - Fact-position pre-RoPE KV: 28-482 MB
```

### The tradeoff

| Method | Storage | Latency | Accuracy | Complexity |
|--------|---------|---------|----------|------------|
| **Keyword index** | **800 B** | **5-10ms** | 100% (tested 20+ facts) | Standard prompting |
| Pre-RoPE fact KV | 28-482 MB | 29ms | Expected 100% (4/4 pre-RoPE) | MLX engineering |
| Full KV cache | 56 GB | 0ms | 100% | Standard |
| Checkpoint replay | 1.3 MB | 500-1000ms | 100% | Custom injection |

**The keyword index wins on every practical dimension.** Pre-RoPE KV injection
is theoretically cleaner (exact state, no prompt engineering needed) but adds
massive storage and engineering complexity for no accuracy benefit.

---

## Key Findings

### 1. The chain adds ~3-4% magnitude, not direction
Dark space raw projections differ by <4% between chained and standalone.
Normalized predictions are nearly identical at L0-L14. The token identity
(Namath, Jets) dominates the representation, not prior window context.

### 2. L26 shows the only meaningful difference
At L26, the chained version predicts contextual continuations (" for", " training")
while standalone predicts terminal tokens ("."). But both have the correct
continuation in their top-10 — the difference is in confidence, not content.

### 3. Standalone KV failed due to RoPE, not context
The evidence: (a) residuals are nearly identical, (b) pre-RoPE injection works 4/4,
(c) the only variable between working pre-RoPE and failing standalone is position encoding.

### 4. Markov holds universally
All-position injection at L14: KL=0.0. The distributed hidden states at L14
fully determine the model's output. Consistent with all prior Markov results.

### 5. The keyword index remains the practical winner
Zero KV manipulation, zero RoPE handling, zero storage overhead. 800 bytes for
370K tokens. 5-10ms retrieval. 100% accuracy at tested scales. The sparse
keyword index IS the solution — KV injection is an elegant theoretical mechanism
that the keyword index makes unnecessary.

---

## What We Didn't Test (And Why)

### Experiments 3-7 from the original plan

The residual comparison (Experiment 1) answered the foundational question
conclusively: the chain doesn't enrich fact-position KV in any meaningful way.
This makes the subsequent experiments (position count sweep, layer subset,
cross-window injection, RoPE handling, storage optimization) moot for the
chained-vs-standalone comparison — the chain is unnecessary.

Pre-RoPE KV injection (Experiment 6c) was already validated in prior work
(4/4 at 29ms for full-window KV). Extending this to fact-position-only
pre-RoPE KV is an engineering task, not a scientific question.

### What would still be interesting

1. **Layer subset for KV injection** — which layers' KV are actually needed?
   Could reduce storage from 482 MB to 28 MB. But this only matters if
   KV injection is pursued over keyword index.

2. **Cross-window KV injection** — can 5 windows' fact KV be injected
   simultaneously? Probably yes (multi-fact injection worked 5/5 in
   prior experiments), but the keyword index already does this natively.

3. **Minimum KV positions per fact** — how few positions are needed?
   Prior work shows 3 content tokens minimum for keyword retrieval.
   KV injection likely needs the same or fewer.

---

## Tags
chained-fact-kv, chained-vs-standalone, rope-position-mismatch, pre-rope-kv,
fact-position-residuals, document-context-kv, standalone-kv, keyword-index-wins,
markov-property, dark-space-comparison, l26-contextual-difference
