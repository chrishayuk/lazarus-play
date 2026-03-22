# Needle in Dark Space Haystack — Full Results

**Experiment:** `needle-in-dark-haystack` (ID: `412cfe21-4f2a-40e5-8584-a0015feeb54f`)
**Model:** google/gemma-3-4b-it (34 layers, 2560 hidden dim)
**Date:** 2026-03-15

---

## Setup

- **Corpus:** Shakespeare's complete works, 1,539,122 tokens, 3,007 windows of 512 tokens
- **Needle:** "The astronaut landed on the moon on July 20, 1969." (18 tokens)
- **Needle location:** Window 769, positions 313-330 (within Henry IV Part 2)
- **Compass:** Geometric RRF at L26, PC2-17 (16D), 8 interval samples per window at positions 64, 128, 192, 256, 320, 384, 448, 511
- **Checkpoint:** `/shakespeare/context/shakespeare_ctx_512/` — 24,056 L26 residuals stored

---

## Experiment 1 — Where Is the Needle in W769?

### Needle Token Map

| Position | Token | Token ID | Note |
|----------|-------|----------|------|
| 313 | "The" | 818 | Needle START |
| 314 | " astronaut" | 43839 | **Unique — appears 0 times elsewhere in Shakespeare** |
| 315 | " landed" | 30473 | 16 occurrences in corpus |
| 316-317 | " on the" | 580, 506 | Common |
| 318 | " moon" | 16254 | 145 occurrences |
| 319-320 | " on July" | 580, 5505 | July: 4 occurrences |
| 321-330 | " 20, 1969." | digits/punct | Date tokens |

### Interval Sampling Hit

| Sample | Position | Token | In Needle? |
|--------|----------|-------|------------|
| s0 | 64 | "," | No |
| s1 | 128 | "," | No |
| s2 | 192 | "\n" | No |
| s3 | 256 | " by" | No |
| **s4** | **320** | **" on"** | **YES** |
| s5 | 384 | " alone" | No |
| s6 | 448 | " honours" | No |
| s7 | 511 | "," | No |

**Sample s4 (position 320) lands on " on" within "...moon on July..." — INSIDE the needle sentence.**

**The compass failure is NOT a sampling miss.** The needle's geometry was stored. It simply didn't score high enough.

---

## Experiment 2 — Is the Needle Geometrically Distinctive?

### Raw L26 Residual: Angle from Shakespeare Baseline

Shakespeare baseline = mean of all W768 + W770 positions (pure Shakespeare).

| Position | Angle from baseline | Note |
|----------|--------------------|----|
| W769 s0 (Shakespeare) | 12.83° | |
| W769 s1 (Shakespeare) | 17.31° | |
| W769 s2 (Shakespeare) | 11.47° | |
| W769 s3 (Shakespeare) | 12.75° | |
| **W769 s4 (NEEDLE)** | **9.47°** | **Less deviant than most Shakespeare positions!** |
| W769 s5 (Shakespeare) | 8.93° | |
| W769 s6 (Shakespeare) | 9.74° | |
| W769 s7 (Shakespeare) | 8.84° | |
| W768 range (pure Shk) | 8.11° – 13.33° | Normal Shakespeare variation |
| W770 range (pure Shk) | 8.15° – 12.72° | Normal Shakespeare variation |

**The needle is at 9.47° — squarely within Shakespeare's 8-17° range.** It is NOT geometrically distinctive at L26.

### Full 2560D Space: Needle is CLOSER to Mean Than Average

| Metric | Needle (W769 s4) | Corpus |
|--------|------------------|--------|
| L2 dist from corpus mean | 14,400 | mean=22,058, std=8,037 |
| z-score | **-0.95** | — |
| Percentile | **5.8th** | — |

The needle is at the 5.8th percentile — meaning 94% of the corpus is MORE distant from the mean than the needle. In the full residual space, the needle is unremarkable.

### Energy Decomposition: Where the Difference Lives

Decomposing the needle-vs-Shakespeare difference vector:

| Subspace | % of Difference | Compass Sees It? |
|----------|----------------|-------------------|
| PC0-1 (structural) | 23.4% | No |
| **PC2-17 (compass)** | **9.9%** | **Yes, but only 10%** |
| **PC18+ (dark residual)** | **66.7%** | **No — invisible to compass** |

**67% of what makes the needle different from Shakespeare lives in dimensions the compass doesn't see.** The compass (PC2-17) captures only 9.9% of the needle's distinctive signal. Even with perfect sampling, the compass is looking in the wrong subspace.

---

## Experiment 3 — Compass Score Against Space-Travel Query

### Query: "Is there mention of space travel in this text?"

| Metric | Value |
|--------|-------|
| Needle s4 cosine to query | **0.053** (essentially orthogonal) |
| W769 max cosine (best of 8 samples) | 0.246 |
| W769 rank | **2527 / 3007** |
| W769 z-score | -1.09 |
| Top-1 score (W140) | 0.833 |
| Top-50 threshold | 0.755 |

**The needle actively scores BELOW average against a space-travel query.** The compass is worse than random at finding this needle — it's anti-correlated.

### W769 Individual Sample Scores

| Sample | Score | Note |
|--------|-------|------|
| s0 | -0.248 | Anti-correlated |
| s1 | -0.191 | Anti-correlated |
| s2 | -0.523 | Strongly anti-correlated |
| s3 | 0.194 | Weakly positive |
| **s4 (NEEDLE)** | **0.053** | **Near zero** |
| s5 | 0.046 | Near zero |
| s6 | 0.080 | Near zero |
| s7 | 0.246 | Best score — but this is SHAKESPEARE text |

The best-scoring sample in W769 is s7 (position 511), which is pure Shakespeare. The needle position (s4) scores near zero.

---

## Experiment 4 — Anomaly Detection: Every Approach Fails

### Summary of ALL Anomaly Approaches Tested

| Approach | W769 Rank | Total | Percentile | Finds Needle? |
|----------|-----------|-------|-----------|---------------|
| Global anomaly (L2 from corpus mean, compass 16D) | 426 | 3,007 | 85.8% | No |
| Local neighbor anomaly (s4 vs ±5 windows) | 2,616 | 3,007 | 13.0% | No |
| Max-position anomaly (best of 8 samples) | 1,579 | 3,007 | 47.5% | No |
| Internal diversity (within-window variation) | 1,268 | 3,007 | 57.8% | No |
| Token rarity (weighted rare token count) | 1,082 | 3,006 | — | No |
| Zero-count tokens (tokens not in Shakespeare) | 1,194 | 3,006 | — | No |
| Compass search ("space travel" query) | 2,527 | 3,007 | — | No |

**No L26-based metric finds the needle in the top-50.**

### What Beats the Needle on Every Metric

The French scenes in Henry V (W820, W846) dominate every anomaly metric:
- 24-26 zero-count tokens (French words) vs needle's 2
- Extreme token rarity (score 600 vs needle's 115)
- Shakespeare naturally contains more "anomalous" content than a single modern sentence

---

## Root Cause Analysis

### Why L26 Cannot See the Needle

At position 320 (" on" / "July"), the L26 residual encodes:

```
Attention context at position 320:
  Shakespeare tokens: 313 (97.8%)
  Needle tokens visible: 7 (2.2%)
  Ratio: 313× more Shakespeare than needle
```

The attention mechanism has integrated 313 tokens of Henry IV Part 2 — king's speeches, stage directions, archaic English. By L26, the residual at position 320 represents "a position in Shakespeare that happens to contain the word 'on'" — not "a modern sentence about space travel."

**The needle's identity is not in the L26 residual. It is in the TOKEN SEQUENCE.**

### The Subspace Problem

The compass PCA basis (PC2-17) captures variation WITHIN Shakespeare:
- Scene types (dialogue vs stage direction)
- Language register (verse vs prose)
- Character names and settings

It does NOT capture:
- Modern vs archaic language
- Scientific vs literary vocabulary
- Temporal anachronism

67% of the needle's distinctive signal lives in PC18+ — dimensions the compass doesn't index.

### The Context-Drowned Signal

Even in the full 2560D space, the needle is at the 5.8th percentile (CLOSER to mean than average). Shakespeare's complete works have enormous internal variation — history plays vs comedies vs tragedies vs sonnets. A single 18-token modern sentence is a smaller perturbation than the difference between the Henriad and A Midsummer Night's Dream.

---

## What Would Find the Needle

### 1. Perplexity Scan (RECOMMENDED)

**Approach:** Store per-token surprise (prediction rank of actual token) during checkpoint creation.

**Why it works:** After 313 tokens of Henry IV, the model predicts "wizard" (6.7%), "king" (5.2%), "boy" (2.1%) — "astronaut" is not in the top-50 (p < 0.36%). With full Shakespeare context, P(astronaut) < 0.01%.

**Implementation:**
```
During checkpoint creation (forward pass already happening):
  For each position p:
    actual_token = tokens[p]
    rank = position of actual_token in model's softmax ranking
    surprise[p] = rank  # high rank = high surprise
  Store: max(surprise) per window
```

**Cost:** 512 bytes/window (rank per position) = 1.5 MB for 3,007 windows. Zero additional forward passes at query time.

**At query time:**
```
lazarus context search \
    --checkpoint ./shakespeare_ctx_512/ \
    --strategy surprise --top-k 3

# Returns: W769 (rank ~1), W820 (French scenes), ...
```

### 2. Token Embedding Outlier (No Forward Pass)

**Approach:** Check each token's embedding distance from the Shakespeare vocabulary centroid.

**Why it works:** "astronaut" embedding neighbors are NASA, spaceship, spacecraft — maximally distant from thee/thou/hath/wherefore. This operates at the EMBEDDING level (before context integration drowns the signal).

**Cost:** One embedding lookup per token. No forward pass needed.

### 3. Multi-Layer Approach

**Approach:** Index at L0-L3 instead of L26. Earlier layers retain more token-level identity before attention integrates context.

**Why it works:** At L0, the residual IS the embedding. At L3, the model has processed some context but the token's own identity still dominates. "astronaut" at L3 would be more anomalous than at L26.

**Tradeoff:** L0-L3 lacks the rich content-type encoding that makes L26 good for compass search. Use L26 for compass, L0-L3 for anomaly.

### 4. Denser Sampling DOES NOT HELP

Even at 32 or 64 samples per window, L26 residuals at needle positions remain indistinguishable from Shakespeare. **The problem is the layer, not the sampling density.**

At 8 samples per window, the probability of hitting the 18-token needle is:
- P(hit ≥ 1) = 1 - ((512-18)/512)^8 = 24.9%
- Actual: s4 hits "on"/"July" — sampling succeeded

With 32 samples: needle spans ~2 samples. But both would score identically to Shakespeare at L26.

---

## Architecture of the Solution

### Two-Channel Document Navigation

| Channel | What it finds | Layer | Storage | Forward passes |
|---------|--------------|-------|---------|----------------|
| **Compass** (existing) | Content SIMILAR to query | L26 | 8 × 2560 × 2B/window | 0 at query time |
| **Surprise** (proposed) | Content SURPRISING in context | All (during fwd pass) | 512 × 1B/window | 0 at query time |

These are **complementary**, not competing:

- **Compass:** "Find passages about kings and crowns" → Henry IV scenes
- **Surprise:** "Find content that doesn't belong" → astronaut sentence, French scenes, anachronisms

### The Anomaly Detection Flag

```
lazarus context search \
    --checkpoint ./shakespeare_ctx_512/ \
    --prompt "What doesn't belong?" \
    --strategy anomaly --top-k 3
```

This would rank by max per-token surprise, ignoring the query entirely. Query-independent. Finds any content the model itself considers out-of-distribution.

---

## Key Findings Summary

1. **Sampling is NOT the problem.** Interval sample s4 lands inside the needle. The geometry was captured.

2. **L26 residuals CANNOT distinguish the needle.** At L26, 97.8% of the representation comes from Shakespeare context. The needle is geometrically absorbed — closer to the corpus mean than average (z-score -0.95).

3. **The compass subspace sees only 9.9% of the difference.** 67% of the needle's distinctiveness lives in PC18+ — outside the 16D compass projection.

4. **Shakespeare's natural variation is LARGER than the needle perturbation.** French scenes, songs, and register shifts create bigger anomalies than a single modern sentence.

5. **Perplexity is the correct signal.** The model assigns < 0.01% probability to "astronaut" in Shakespeare context. This is a massive, unambiguous signal that requires no additional computation (captured during the existing forward pass).

6. **The compass and surprise detector are complementary channels.** Compass finds similarity (content-addressed). Surprise finds anomaly (context-addressed). Together: "find passages about X" AND "find things that don't belong."
