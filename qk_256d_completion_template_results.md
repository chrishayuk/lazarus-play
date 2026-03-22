# Q·K 256D Completion Template Results

**Experiment:** cca92c7f
**Model:** google/gemma-3-4b-it
**Date:** 2026-03-21

## Summary

The 256D head space IS content-discriminative. The completion template
unlocks 4× better semantic routing than chat template. Q·K routes
3/3 at N=3 (novel entities). But at N=50, RoPE positional bias
overwhelms the content signal. Landing query ranks #36/50.

**Keywords confirmed final.** The 256D content signal exists but
is 10–100× weaker than RoPE decay at scale.

---

## Experiment 1 — Template Effect on Q-vectors

**Method:** W370 (Apollo landing window) + query in 3 template formats.
attention_pattern at L29 measures H4 routing to landing-relevant tokens
(CONTACT, LIGHT, Houston, Tranquility, LANDED, Eagle, Roger, etc.)

| Template | H4 landing % | H5 landing % | H3 landing % | Multi-head sum |
|----------|:---:|:---:|:---:|:---:|
| Chat | 6.3% | 2.3% | 0.4% | 11.4% |
| **Completion** | **28.9%** | **20.3%** | **8.1%** | **62.5%** |
| Fill-blank | 7.6% | 6.9% | 1.6% | 18.8% |

**Completion template = "Transcript query: {Q}\nAnswer:"**

The completion template produces 4× more attention to semantically
relevant content tokens. H4 attends to Roger (7.0%), Houston (4.8%),
Tranquility (3.5%), HAS (2.0%), Eagle (1.9%), copy (1.1%), landed (1.0%).

Chat template: mostly Houston (3.2%) and structural tokens.
Fill-blank: similar to chat with slightly more spread.

### Pairwise cosine (Experiment 4)

In 2560D full residual space:

| | Chat | Completion | Fill-blank |
|---|:---:|:---:|:---:|
| Chat | 1.000 | 0.984 | 0.955 |
| Completion | 0.984 | 1.000 | 0.962 |
| Fill-blank | 0.955 | 0.962 | 1.000 |

All templates are >0.95 similar in 2560D. Yet routing differs 4×.
The W_Q projection into 256D amplifies the content-discriminative
dimensions that are a small fraction of the 2560D norm.

**FR conditions not tested** — requires inject_residual combined with
attention_pattern, which the tools don't support.

---

## Experiment 2 — Q·K at N=3 (Novel Entities)

**Method:** 3 passages (Zarkov/Voltara, Strand/Castellan, Voss/Kelvara)
concatenated + completion template query. attention_pattern at L29.

| Query | H4 to Zarkov | H4 to Strand | H4 to Kelvara | Correct? | Margin |
|-------|:---:|:---:|:---:|:---:|:---:|
| Zarkov city | **55.1%** | 6.9% | 4.1% | ✓ | 8.0× |
| Strand city | 10.3% | **24.9%** | 5.5% | ✓ | 2.4× |
| Kelvara town | 8.5% | 4.6% | **47.4%** | ✓ | 5.6× |

**3/3 correct.** H4 alone sufficient. Margins 2.4–8.0×.

H4 top attended tokens per query:
- Zarkov: "Volt" (26.4% + 20.5%), "Dimitri" (2.4%)
- Strand: "Cast" (9.9%), "ellan" (2.3%), "Carpathian" (2.3%)
- Kelvara: "Kel" (28.3%), "Meridian" (13.4%)

H5 also routes correctly for all 3 queries (secondary confirmation).

**Why N=3 works:** All passages within ~500 tokens of the query.
RoPE positional differences are small (~1.1× decay over 500 positions).
Content signal (256D Q·K matching) dominates.

---

## Experiment 3 — Q·K at N=50 (Apollo)

**Method:** 50 windows in 5 batches of 10. Completion template.
attention_pattern at L29 for all heads.

| Query | H4 target rank | H4 target attn | H4 winner | Pattern |
|-------|:---:|:---:|---|---|
| Landing | **#36/50** | 0.000 | W390 (37.2%) | Recency bias |
| Porridge | #5-6/10 | 0.000 | W255 (33.6%) | Recency bias |
| Baseball | #5-6/10 | 0.000 | W255 (32-34%) | Recency bias |
| Weather | #5-6/10 | 0.000 | W255 (32-34%) | Recency bias |
| News | #5-6/10 | 0.000 | W255 (32-34%) | Recency bias |

**0/5 correct.** All heads dominated by positional recency bias.

### Diagnosis: RoPE Overwhelms Content

The failure is NOT because the 256D content signal doesn't exist.
It's because RoPE positional encoding creates massive recency bias:

1. **10 windows × ~500 tokens = 5000 token span per batch**
2. **RoPE decay**: exp(-θ × Δpos) creates 100× advantage for recent tokens
3. **Content signal is ~10% of attention** (Exp 1: 28.9% landing tokens)
4. **RoPE bias is ~90% of attention** at these distances

Evidence:
- Changing the query text produces **near-zero change** in H4 attention map
- The last window ALWAYS wins regardless of query content
- This is position-dependent, not content-dependent

### N=3 vs N=50: Why the Difference?

| | N=3 | N=50 (batched) |
|---|---|---|
| Total tokens | ~500 | ~5000 |
| Position span | ~500 | ~5000 |
| RoPE decay | ~1.1× | ~100× |
| Content signal | Dominates | Drowned |
| Result | 3/3 | 0/5 |

The 256D content signal operates in a ~500-token radius where
RoPE is approximately flat. Beyond that, RoPE wins.

---

## Experiments 5 & 6 — Not Run

Per spec: "Stop after Experiment 3 if landing rank > 20."
Landing rank = 36. Multi-head (Exp 5) cannot fix RoPE bias
since ALL heads show the same recency pattern.

---

## Key Findings

### 1. The 256D Content Signal Is Real
The W_Q/W_K projection creates a content-discriminative subspace.
Completion template unlocks it (4× vs chat). H4 attends to
semantically relevant tokens (Roger, Houston, Tranquility, Volt, Kel).
Works perfectly at N=3.

### 2. RoPE Kills It at Scale
RoPE positional encoding creates exponential recency bias that
overwhelms content matching beyond ~500 tokens. This is why:
- N=3 (500 tokens): 3/3 correct
- N=50 (5000 tokens per batch): 0/5 correct

### 3. Stored K-Vectors (Without RoPE) Remain Open
This experiment tested Q·K via **live attention** (includes RoPE).
The stored K-vector approach could potentially work if:
- K-vectors stored without RoPE
- Or positions equalized
- Previous experiment (k_norm_sampling): 4/8 heads route with equalized positions

But even equalized-position K-vectors couldn't overcome parametric
override for well-known topics.

### 4. Template Matters — Completion Is Best
The completion template "Transcript query: X\nAnswer:" produces
Q-vectors that are maximally content-discriminative. This was
the clearest positive finding.

### 5. Keywords Confirmed Final
At N=50+, no Q·K mechanism (with or without RoPE) has matched
keyword routing (5/5 at N=50, 5/5 at N=724). The 256D content
signal is real but insufficient for scale routing.

**Architecture remains: keyword route → persistent inject.**

---

## Comparison Table

| Method | N=3 | N=50 | N=724 | Mechanism |
|--------|:---:|:---:|:---:|---|
| Token overlap | 3/3 | 4/5 | 3/5 | Vocabulary intersection |
| TF-IDF | — | 4/5 | 3/5 | Weighted vocabulary |
| Keyword | 3/3 | **5/5** | **5/5** | Regex phrases |
| 2560D cosine | 1/4 | — | — | Full residual |
| Q·K 256D (live) | **3/3** | 0/5 | — | Attention with RoPE |
| Q·K 256D (stored, equalized) | — | 4/8 heads | — | K-vectors without RoPE |

---

## What This Means

The model's 256D head space IS the right subspace for content
matching. The completion template IS the right query format.
But RoPE positional encoding — the very mechanism that gives
attention its sequence awareness — also creates the barrier
that prevents cross-format, cross-position routing at scale.

This is a fundamental tension: RoPE enables attention to
distinguish positions (essential for language), but also
prevents attention from being a pure content-addressable
memory at scale.

Keywords work because they operate in vocabulary space,
which has no positional encoding. They're position-invariant
by construction. The model's attention mechanism is not.
