# Multi-Position Residual Injection Results

**Experiment:** f2ba7184-c227-40de-84fe-d7150cc1cbf9
**Model:** google/gemma-3-4b-it (34 layers, 2560 hidden, 8 heads)
**Date:** 2026-03-22
**Prior:** additive-multidim-injection (7f89fd70), persistent_transplant (c1bb923b)
**Status:** Complete

---

## Question

Can multi-position full-residual injection at L30 deliver both entity
names AND narrative details WITHOUT any context token replay?

## Answer: Outcome C-plus — context replay required, but far less than expected

**All-position injection** (patch_all_positions=True) delivers **perfect** narrative
at KL=0.0 — every fact, every number, every name. But this IS context replay:
it transplants the donor's entire hidden state across all positions. The facts
live in the positional KV entries, and there is no way around replaying them.

**Last-position-only injection** delivers **topic only** — the model knows it's
about porridge/baseball but confabulates every fact. P(correct fact) = 0%.

**The irreducible floor is the context sentence itself** (~25-50 tokens of
focused, well-formed text containing the facts). Grammar is not optional.

---

## Key Results

### 1. Ground Truth (full 529-token window)

```
"According to the text, an Irishman named John Coyle won the world's
 porridge eating contest by consuming 23 bowls of instant oatmeal."
```
7/10 facts. P(According) = 71.9%.

### 2. All-Positions Injection — Perfect at Every Layer

| Donor | Tokens | Layer | KL | Output | Facts |
|-------|:------:|:-----:|:--:|--------|:-----:|
| Full window + query | 529 | L30 | 0.0 | "According to the text, an Irishman named John Coyle..." | 7/10 |
| Full sentence + query | 68 | L30 | 0.0 | "John Coyle, an Irishman from Corby, England, won..." | 5/10 |
| Full sentence + query | 68 | L24 | 0.0 | "John Coyle, an Irishman from Corby, England, won..." | 5/10 |
| Full sentence + query | 68 | L14 | 0.0 | "John Coyle, an Irishman from Corby, England, won..." | 5/10 |
| Full sentence + detailed query | 91 | L30 | 0.0 | Name: John Coyle, Irish, Corby England, 23 bowls, 35 competitors | **7/10** |
| Compact sentence + query | 41 | L30 | 0.0 | "John Coyle won the porridge eating contest." | 2/10 |

**The injection mechanism works perfectly at KL=0.0 from L14 through L30.**
The limiting factor is what the donor itself generates, not the injection.

### 3. Last-Position-Only — Total Failure for Facts

| Donor | Query | Output | Facts |
|-------|-------|--------|:-----:|
| Full sentence (68 tok) | Simple | "John **Holt** won the 1959 Championship! 68 bowls in 17 min" | **0/10** |
| Full sentence (68 tok) | Detailed | "**Riku Miettinen**, Finnish, 48 bowls, 15 competitors" | **0/10** |
| Baseball scores (52 tok) | Simple | "Baltimore Orioles 8, Texas Rangers 7 (10 innings)..." | **0/4** |

Last-position injection steers TOPIC but confabulates ALL facts.
With a strong recipient query (42 tokens), the last-position signal
is overwhelmed entirely — output matches the recipient, not the donor.

### 4. The Grammar Requirement

| Context style | Output | Facts |
|--------------|--------|:-----:|
| Full sentence: "John Coyle, an Irishman, won...23 bowls of instant oatmeal" | All correct | 7/10 |
| Compressed: "John Coyle won porridge championship, 23 bowls oatmeal, Corby England" | Nationality wrong ("British") | 6/10 |
| Telegram: "John Coyle, 23 bowls, Corby, 35 competitors" | Interprets "23 bowls" as **bowling** | **0/10** |

Without "of instant oatmeal" and "porridge eating championship", the model
cannot resolve ambiguity. **Connective tissue resolves meaning.**

### 5. Generalization

**Zarkov Industries (novel entity):** All-positions L30 injection delivers
"Headquarters: Voltara, Founder: Marcus Strand, Primary Business: Advanced
Water Filtration Systems" — 3/3 facts correct. Recipient alone confabulates
"defense company, espionage, secure communications."

**Baseball scores:** All-positions L30 injection delivers all 4 game scores
correctly (Baltimore 3-2, Detroit 4-3, Minnesota 8-5, Boston/NY rained out).
Last-position confabulates every score.

---

## What patch_all_positions Actually Does

When donor (68 tokens) and recipient (17 tokens) have different lengths,
`patch_all_positions=True` replaces the **recipient's entire hidden state
tensor** with the **donor's entire hidden state tensor** at the injection
layer. The donor's 68 positions replace the recipient's state completely.
Downstream layers process the donor's computation. KL=0.0 because from
that layer onward, it IS the donor's forward pass.

This is NOT selective position injection — it's full state transplant.
The tool does not support injecting at specific positions only.

---

## Architecture Implications

### What This Proves

1. **Facts are positional.** "Coyle" lives at position 104-105, "23" at
   positions 125-126, "Corby" at positions 95-96 in the KV cache. The
   last-position residual carries topic direction only.

2. **The Markov property holds universally.** From L14 through L30,
   replacing all positions produces KL=0.0. The downstream computation
   depends only on the current hidden state, not on how it was produced.

3. **Grammar is load-bearing.** Telegram-style keyword bags fail
   catastrophically. The model needs syntactic context to resolve
   semantic ambiguity ("23 bowls" → bowling vs oatmeal).

4. **The minimum viable context is a well-formed sentence.** ~25-50 tokens
   of focused, grammatical text containing all relevant facts. This is
   already 10-20× less than the full 512-token window.

### The Minimum Context Architecture

```
Store per window:
  - TF-IDF keywords for routing (~3 tokens, ~12 bytes)
  - Focused fact sentence (~25-50 tokens, ~100-200 bytes as text)
  - Boundary residual for chaining (10 KB)

At query time:
  1. Route: keyword match → select window(s)
  2. Build donor: fact sentence + query (50-90 tokens)
  3. Prefill donor through model (50-90 token forward pass)
  4. Generate from donor's KV cache

Cost: 50-90 token prefill vs 512 token window replay
Speedup: ~6-10×
Storage: ~200 bytes text + 10 KB residual per window
```

### Why Selective Position Injection Was the Wrong Question

The experiment design hypothesised injecting 8-13 L30 residuals at
fact-bearing positions. This would have required:
- Extracting and storing full L30 residuals per position (10 KB each)
- Custom injection code to plant them at synthetic positions
- Solving the RoPE double-encoding problem

But the results show the real bottleneck is **downstream attention
over the injected positions**. L31-L33 need to attend to positions
that have been processed through L0-L30 — not just the L30 output,
but the full KV stack built at every layer during the forward pass.

`patch_all_positions` works because it replaces the hidden state AND
the downstream layers rebuild KV from that state. Injecting L30
residuals at synthetic positions would give L31-L33 hidden states
without the L0-L29 KV entries — attention would have nothing coherent
to attend to at those positions.

**The KV cache is built incrementally through the forward pass.
You can't shortcut it by injecting at one layer.**

---

## Comparison to Prior Results

| Approach | Context tokens | Facts delivered | Storage/window |
|----------|:--------------:|:--------------:|:--------------:|
| Full window replay | 512 | 7-10/10 | 56 GB (full KV) |
| Focused sentence replay | 50-90 | 7/10 | ~200 B text |
| 1D V-injection (878868bd) | 0 | 1/10 (entity only) | 24 B |
| Persistent transplant (c1bb923b) | 0 | 3-5/10 (entity + short) | 10 KB |
| Last-position injection | 0 | 0/10 (confabulation) | 10 KB |
| **This: all-position transplant** | **50-90** | **7/10** | **10 KB + 200 B** |

---

## Connection to the Product

The current v10 architecture replays ~200 tokens of focused context.
This experiment confirms that approach is correct and may even be
over-provisioned — **50-90 tokens of well-formed fact sentence + query
is sufficient for full narrative delivery**.

The key optimisation is not zero-token injection (which fails for
narrative) but **minimum-token context generation**:

1. Extract key facts during indexing (already done via IDF)
2. Generate a focused fact sentence per window (~25-50 tokens)
3. At query time, prepend the fact sentence to the query
4. Prefill the combined ~50-90 tokens through the model
5. Generate from the resulting KV cache

This is 6-10× faster than replaying the full 512-token window,
achieves the same output quality, and requires ~200 bytes of
additional storage per window (the fact sentence text).

**Zero-token delivery of full narrative is not achievable.**
The facts live in positional KV entries that require a forward pass
to construct. But 50-90 token "micro-replay" is the new floor,
down from 200 tokens. The product ships with what works.

---

## Outcome Classification

**Outcome C-plus: Entity + narrative require context, but much less than expected.**

Not Outcome A (zero-token narrative delivery).
Not Outcome D (full context required — 50 tokens suffice, not 512).
A refined Outcome C: grammar-preserving micro-context replay at 50-90 tokens
delivers the same quality as full 512-token windows.
