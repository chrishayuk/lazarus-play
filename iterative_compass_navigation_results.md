# Iterative Compass Navigation — Results

**Experiment ID:** 1ba1fdb2-c942-4324-8135-e2530d88f952
**Model:** google/gemma-3-4b-it (34 layers, 2560 hidden dim)
**Date:** 2026-03-14

## Summary Finding

**Reading content shifts the L26 residual by 7.0–7.5 degrees — well above the 5-degree
significance threshold. The shift is content-dependent (different content produces shifts
3.89 degrees apart). The shifts are 2-dimensional: PC1 (78%) encodes "I am reading
something" (context presence); PC2 (15%) encodes "what I am reading" (content type).
The tonal dimension is UNTOUCHED (only 0.25° shift). Iterative compass navigation is
geometrically viable: the model's L26 compass rotates through content-space when it reads,
providing a mechanism for content-directed document exploration.**

**However: diminishing returns. Reading a second window adds only 3.6° beyond the first
read's 7° shift. The PC1 component (context presence) dominates — to extract the navigational
signal (PC2), subtract the context-presence baseline first.**

---

## Experiment 1 — Baseline Query Residuals

### Two Queries at L26

| Query Pair | Cosine | Angle |
|------------|--------|-------|
| "Find 3 amusing moments" vs "What sport and teams?" | 0.9688 | **14.34°** |

Two different queries produce residuals 14.34° apart at L26 — confirming that query intent
is well-separated in compass space.

### Direction to Amusing Steering Vector

| Prompt | Angle to amusing SV |
|--------|---------------------|
| "Find amusing moments" (bare query) | **121.36°** |
| After reading W170 (porridge/baseball) | **121.61°** |
| After reading W118 (Earth in window) | **121.52°** |

All three are within **0.25°** of each other on the tonal axis. The tonal channel is
completely invariant to what the model reads. The shift is orthogonal to tone.

---

## Experiment 2 — Residual Shift After Reading Content

### The Core Measurement

| Comparison | Cosine | Angle | Interpretation |
|------------|--------|-------|----------------|
| Query → Post-read-W170 (porridge/baseball) | 0.9926 | **6.97°** | **Significant shift** |
| Query → Post-read-W118 (Earth window) | 0.9915 | **7.48°** | **Significant shift** |
| Post-read-W170 → Post-read-W118 | 0.9977 | **3.89°** | Different content → different shift |
| Query → "What sport?" (different query) | 0.9688 | **14.34°** | Baseline for different intent |

### Key Findings

1. **Reading shifts the compass by ~7°.** This is half the distance between completely
   different queries (14.3°). The compass has clearly moved.

2. **Different content produces different shifts.** W170 (sports/porridge) and W118
   (Earth/magnificent) produce residuals 3.89° apart. The shift direction is
   content-dependent, not just a generic "context-present" signal.

3. **The shift is NOT tonal.** All post-read residuals are within 0.25° of the bare
   query on the amusing steering vector axis (121.4° ± 0.13°). Reading amusing content
   does NOT make the residual more "amusing" — the tonal compass is independent.

### What This Means

The model's L26 state encodes what it has just read in its content-type dimensions,
but NOT in its tonal dimensions. When the model reads about porridge eating contests
versus the magnificent Earth view, the compass rotates through content-space (sports
vs nature/observation), but the model's tonal orientation remains fixed.

This is exactly what the previous tonal experiment found: the tonal and content-type
channels are orthogonal (zero top-10 overlap in compass vs tonal search).

---

## Experiment 3 — Compass Re-Routing

### 15-Window Ranking Comparison

Tested routing against 15 representative windows across 4 conditions:

| Rank | Bare Query | Post-read W170 | Post-read W118 | Post-read Both |
|------|-----------|----------------|----------------|----------------|
| 1 | Morning news (0.971) | Morning news (0.973) | Morning news (0.972) | Morning news (0.972) |
| 2 | Flight plan (0.970) | **Breakfast** (0.972) | **Breakfast** (0.972) | **Breakfast** (0.971) |
| 3 | TV broadcast (0.970) | TV broadcast (0.971) | TV broadcast (0.971) | TV broadcast (0.971) |
| 4 | **Breakfast** (0.969) | Flight plan (0.971) | Flight plan (0.971) | Flight plan (0.970) |
| 5 | Moon bet (0.969) | Moon bet (0.971) | Moon bet (0.970) | Moon bet (0.970) |

### Consistent Rank Changes After Reading

- **"Breakfast magnificent" rises from rank 4 → rank 2** in ALL post-read conditions.
  After reading any Apollo 11 content, the compass preferentially points toward
  human-interest/narrative windows over planning windows.

- **"Flight plan discussion" drops from rank 2 → rank 4.** Procedural/planning content
  becomes less aligned after reading human-interest content.

- **"President" drops from rank 7 → 9-10.** Formal/institutional content drops.

### Limitation

With only 15 windows, these rank changes are modest (+2, -2). In the full 725-window
library where windows are clustered within 0.001 cosine of each other, a 7° shift
would produce **massive re-ranking** — potentially swapping hundreds of window positions.
The full-library experiment would be decisive.

---

## Experiment 4 — Multi-Step Navigation Simulation

### Cumulative Shift Trajectory

| Round | Context | Shift from Origin | Additional from Previous |
|-------|---------|-------------------|------------------------|
| 0 | Bare query | 0° | — |
| 1 | + W170 (porridge/baseball) | **6.97°** | 6.97° |
| 2 | + W170 + W118 (both) | **7.39°** | **3.58°** |

### Diminishing Returns

The first read shifts the compass by 7°. The second read adds only 3.6° more.
The total shift after reading TWO windows (7.39°) is barely more than reading ONE (6.97°).

**The shift asymptotes.** The dominant component is "bare query → query with context"
(PC1, 78% of variance). Once the model has ANY context, adding more context produces
only incremental content-specific adjustments.

### Implication for Navigation Design

For iterative navigation, **use each window's residual independently**, not cumulatively:

```
Round 1: Extract L26 from [query] → route → find W₁
Round 2: Extract L26 from [W₁ content + query] → route → find W₂
Round 3: Extract L26 from [W₂ content + query] → route → find W₃
```

NOT:
```
Round 3: Extract L26 from [W₁ + W₂ + query]  ← diminishing returns
```

Each round should use only the most recent window, so the content-specific signal
(PC2, 15%) isn't drowned by the context-presence signal (PC1, 78%).

---

## Experiment 5 — Shift Geometry Analysis

### PCA of Navigation Shifts (7 prompts, L26)

| Component | Variance | Cumulative | Interpretation |
|-----------|----------|------------|----------------|
| PC1 | **77.7%** | 77.7% | "Context presence" — bare query vs any context |
| PC2 | **14.6%** | 92.3% | "Content type" — which content was read |
| PC3 | 5.1% | 97.4% | Residual content variation |
| PC4 | 1.7% | 99.1% | Minor variation |
| PC5 | 0.9% | 100% | Noise |

### The Two-Dimensional Navigation Signal

The shifts are **2-dimensional** at 92.3% variance. The navigation manifold is:

```
PC1 (78%): I am reading ←→ I am not reading
PC2 (15%): Content A    ←→ Content B
```

PC1 is the same for ALL content — it captures the structural difference between
a bare question and a question embedded in a paragraph. This is NOT useful for
navigation (it doesn't distinguish content).

PC2 captures WHAT was read. This is the navigational signal.

### The Key Insight

**To extract the navigational signal, subtract the PC1 projection.**

All post-read residuals share the PC1 component (context presence). The navigational
value — which content shifts the compass where — lives in PC2. This is analogous to
centering before PCA: remove the mean to find the interesting variation.

### Orthogonality to Tonal Dimension — CONFIRMED

The tonal steering vector is at 121.4° ± 0.13° from ALL query variants (bare and
post-read). The navigation shifts are in dimensions ORTHOGONAL to the tonal signal.
This confirms the multi-channel architecture:

```
L26 Dark Space:
  Content-type compass (PC 4-19 of original)  ← shifts with reading
  Tonal compass (3D)                           ← invariant to reading
  Navigation shift (PC1-2 of this experiment)  ← new channel
```

---

## Architecture of Iterative Navigation

### What Works

1. **Reading shifts the compass** — 7° is geometrically meaningful (half the
   distance between different queries)
2. **Shifts are content-dependent** — different content produces different compass
   rotations (3.89° apart)
3. **The tonal channel is invariant** — reading doesn't change the model's tonal
   assessment, confirming channel independence
4. **Shifts are low-dimensional** — 2D captures 92.3%, making them tractable
   for extraction and manipulation

### What Doesn't Work (Yet)

1. **Diminishing returns** — second read adds only 3.6° beyond first read's 7°
2. **PC1 dominance** — 78% of the shift is "context presence" not "content type"
3. **Small-library testing** — 15 windows insufficient to show dramatic re-routing;
   full 725-window library needed for definitive test

### The Optimal Navigation Protocol

Based on these findings, iterative compass navigation should:

1. **Extract L26 residual** from [current window + query]
2. **Subtract the context-presence baseline** (PC1 of the shift subspace)
3. **Use the residual PC2-5 components** as the new compass bearing
4. **Route against the library** in the content-dependent subspace
5. **Use single-window context** per round, not cumulative

This isolates the content-specific navigation signal from the generic context-presence
signal, maximizing the compass's discriminative power across rounds.

### The Bigger Picture

This experiment confirms that the L26 dark space supports at least THREE independent
navigation channels:

| Channel | Dimensions | Behavior | How to Access |
|---------|-----------|----------|---------------|
| Content-type compass | ~16D (PC 4-19) | Static query routing | Cosine to stored residuals |
| Tonal compass | 3D | Fixed per query | Steering vector projection |
| Navigation shift | 2D | Changes with reading | PC2 of shift subspace |

The navigation shift is the new finding: the model's compass ROTATES when it reads
content, and the rotation direction depends on WHAT it reads. This enables iterative
document exploration — each reading produces a new compass bearing, directing the
model toward thematically adjacent content.

**Iterative compass navigation is geometrically viable. The model reads, the compass
shifts, and different content produces different shifts. The implementation challenge
is extracting the PC2 signal (content-type) from the PC1 noise (context-presence).**
