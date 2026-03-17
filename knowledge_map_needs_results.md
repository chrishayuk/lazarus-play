# What Knowledge Does the Map Need?

**Experiment 08821818 — google/gemma-3-4b-it (34 layers, 2560D)**

## The Answer

**One dimension per fact.** The knowledge content of a novel fact in the residual stream is a single scalar: the projection magnitude onto the answer token's embedding direction. The other 2559 dimensions are addressing infrastructure.

## Setup

Context prompt (3 novel facts, chat template + partial completion):
```
Zarkov Industries was founded in the city of Voltara in 1987.
The crew reported that the audio quality was scratchy during descent.
Joe Namath agreed to sell his restaurant and report to the New York Jets training camp.
```
Query: model turn with "Zarkov Industries was founded in the city of" → " Volt" at 100%.

Token positions: " Volt" at pos14, "ara" at pos15. 72 tokens total.

## Experiment 1 — What Transfers at L29

### The retrieval circuit

| Layer | Component | Contribution to " Volt" | Notes |
|-------|-----------|------------------------|-------|
| Embedding | — | +11.7 | Base signal |
| L0 | attn+FFN | -24.3 | Suppression (embedding war) |
| L23 | attn | **+6.9** | First copy (H1: +1.82, H3: +1.97) |
| L26 | attn | +1.9 | Secondary copy |
| **L29** | **attn** | **+16.5** | **Main copy (H4: +2.1 DLA, 101% of layer)** |
| L30 | attn | +5.4 | Cleanup (H0: +1.36, H3: +1.88) |
| L33 | attn+FFN | +19.25 | Universal amplifier (FFN: +14.0) |
| **Total** | | **42.75** | P(" Volt") = 100% |

### L29 H4 — the fact copier

- **57.8% attention to pos14** (" Volt") — the fact value position
- DLA contribution: +2.109 (101% of layer's attribution)
- All other heads: negative or near-zero
- **Output norm: 0.0284** — incredibly tiny vs 5637 total attention output
- **99.4% of H4 output is dark space** — orthogonal to ALL vocabulary tokens
- The fact transfer is encoded in dark dimensions, NOT directly in token space

### L29 attention output decomposition

| Component | Projection | Fraction of energy |
|-----------|-----------|-------------------|
| " Volt" (answer) | 2119 | **14.1%** |
| "ara" | 90 | 0.03% |
| " city" (category) | 52 | 0.008% |
| " town" (category) | 63 | 0.01% |
| " founded" (relation) | -100 | 0.03% |
| "Z" (entity) | -17 | 0.001% |
| " Industries" (entity) | -21 | 0.001% |
| **Dark space** | **5221** | **85.8%** |

### The knowledge delta (L28 → L29)

| Component | L28 (before) | L29 (after) | Delta |
|-----------|-------------|-------------|-------|
| " Volt" (answer) | **-543.6** | **+1535.3** | **+2078.9** |
| "ara" | +1428.6 | +1527.6 | +99.0 |
| " city" | +150.5 | +209.8 | +59.3 |
| " town" | +589.1 | +675.9 | +86.8 |
| " founded" | -328.9 | -367.5 | -38.6 |
| " of" | +336.9 | +366.5 | +29.6 |
| "Z" | +464.2 | +467.6 | +3.4 |
| " Industries" | -85.6 | -130.9 | -45.3 |

**The answer delta dominates**: +2079 in " Volt", 21x larger than any other change. The map already carried category/entity/relation signals. The KV added the SPECIFIC answer.

### Residual decode trajectory

| Layer | Top prediction | P(" Volt") |
|-------|---------------|-----------|
| L15 | "," (0.04%) | not in top-10 |
| L23 | " " (42%) | not in top-5 |
| L28 | " " (15.9%) | " Vol" at 5.9% |
| **L29** | **" Volt" (95.7%)** | **95.7%** |
| L30 | " Volt" | continues |
| L33 | " Volt" (100%) | 100% |

L29 is THE knowledge transfer layer. 0% to 95.7% in one layer.

## Experiment 2 — The Slot-Filler Structure

### Three facts, same template, different answers

| Fact | Answer token | P(answer) |
|------|-------------|-----------|
| Zarkov/Voltara | " Volt" | 100% |
| Nexaris/Crenthia | " Cren" | 100% |
| Aldric/Thessmere | " Thess" | 100% |

### L29 attention output — universal structure

| Fact | Answer fraction | Dark fraction | Answer projection |
|------|----------------|---------------|-------------------|
| Voltara | 14.1% | 85.8% | 2119 |
| Crenthia | 15.9% | 84.0% | 1752 |
| Thessmere | 13.1% | 86.8% | 2094 |

**~14% answer, ~86% dark is universal.** The attention output structure is invariant to which specific fact fills the slot.

### Slot similarity across layers

| Layer | Same-type cosine | Angle | Interpretation |
|-------|-----------------|-------|----------------|
| L15 (slot ready) | >0.9998 | **~1.1 deg** | Slots indistinguishable |
| L28 (pre-copy) | ~0.993 | ~6.5 deg | Some divergence |
| L29 (post-copy) | ~0.984 | ~10.3 deg | Knowledge SEPARATES representations |

The slot is generic. At L15, all three "city of ___" queries are virtually identical. The knowledge transfer at L29 roughly doubles the divergence.

### Cross-slot geometry at L15

| Pair | Cosine | Angle |
|------|--------|-------|
| City vs City (same template) | >0.9998 | 1.1 deg |
| City vs Team (Namath) | 0.9974 | 4.1 deg |
| City vs Parametric (Paris) | 0.9977 | 3.9 deg |

Different slot TYPES create 4x more angular separation than same-type fillers, but all are still very similar (cos>0.997). The L15 slot says "answer needed" not "city needed."

## Experiment 3 — The Minimum Knowledge Dimension

### PCA fails, token direction succeeds

| Method | Dims | P(" Volt") | KL from donor |
|--------|------|-----------|---------------|
| Full residual | 2560 | 100% | 0.0 |
| 5D PCA subspace | 5 | 7.6% | 2.57 |
| 5D token directions | 5 | 99.97% | 0.0003 |
| **1D answer token** | **1** | **99.997%** | **0.000031** |

PCA across 6 city prompts shows flat variance (27%, 23%, 19%, 17%, 14%) — no dominant knowledge direction. But injecting just the 1D answer token direction works perfectly.

### 1D injection — universal across facts

| Fact | Token | P(answer) | KL from donor | Energy fraction |
|------|-------|-----------|---------------|-----------------|
| Voltara | " Volt" | 99.997% | 0.000031 | 0.058% |
| Crenthia | " Cren" | 99.993% | 0.000074 | 0.40% |
| Thessmere | " Thess" | 100% | 0.0 | 0.026% |

**One dimension per fact.** The knowledge IS the answer token direction. The 86% dark space is structural scaffolding, NOT knowledge about the answer. The actual information content is a single scalar: the projection magnitude.

### Why PCA fails

PCA captures variation BETWEEN facts. Each fact has its own token direction (" Volt", " Cren", " Thess" are ~88 deg apart). N facts span ~N directions. PCA finds these, but they're not a useful basis for transferring any SPECIFIC fact. The knowledge for Voltara lives in 1D (the " Volt" direction), not in the 5D subspace of cross-fact variation.

### Storage implications

| Quantity | Value |
|----------|-------|
| Knowledge per fact | 1 scalar x 2 bytes (bf16) = **2 bytes** |
| 3,625 facts (Apollo 11) | **7.25 KB** |
| Total KV cache (370K tokens) | **56 GB** |
| **Infrastructure-to-content ratio** | **7,700,000:1** |

The KV cache is a 56 GB addressing system that delivers 7.25 KB of knowledge content. The knowledge is tiny. The machinery to store, address, and retrieve it is enormous.

## Experiment 4 — How the Map Creates the Slot

### Slot with vs without context

| Comparison (L15) | Cosine | Angle |
|-------------------|--------|-------|
| Same entity, WITH vs WITHOUT fact | 0.9978 | 3.8 deg |
| Different entity, BOTH with fact | 0.9998 | 1.1 deg |

The model knows at L15 whether an answer exists in context (3.8 deg > 1.1 deg). But the slot is still 99.8% identical regardless. Presence/absence of the filler slightly modifies the slot.

### What fills the slot without context

Without context, "Zarkov Industries was founded in the city of" produces:

| Layer | Top prediction | P(top) |
|-------|---------------|--------|
| L15 | " North" | 0.06% |
| L28 | **" Chicago"** | **74.2%** |
| L29 | **" Chicago"** | **76.2%** |

The slot gets filled by the encyclopaedia's best guess. "Industries" + "city of" leads to US industrial city (Chicago). Same slot mechanism, different filler source (FFN weights instead of KV cache).

## Experiment 5 — The Parametric Equivalent

### Circuit comparison

| Property | Novel (Voltara) | Parametric (Paris) |
|----------|----------------|-------------------|
| Source | KV cache (L29 attn) | FFN weights (L25 FFN) |
| Transfer layer | L29 (+16.5) | L25 (+8.25) |
| Dominant component | **Attention** (total +29.4) | **FFN** (total +38.0) |
| Answer fraction | **14.1%** | **5.2%** |
| Dark fraction | 85.8% | **93.8%** |
| Answer delta | +2079 | +1356 |
| Trajectory | 0% to 95.7% at L29 (discrete jump) | 97.8% at L24 (gradual) |

### The encyclopaedia activates neighborhoods

L25 FFN for "capital of France is":
- " Paris": +841 (5.2%)
- " Lyon": +291 (0.6%)
- " capital": +190 (0.3%)
- " France": +56 (0.02%)

The notebook copies ONE fact. The encyclopaedia activates related facts (Lyon, capital, France — the semantic neighborhood of Paris).

### Parametric decode trajectory

| Layer | Top prediction | P(" Paris") |
|-------|---------------|------------|
| L15 | " cities" (0.05%) | not in top |
| L24 | **" Paris" (97.8%)** | 97.8% |
| L25 | " Paris" (99.9%) | 99.9% |
| L26 | " Paris" (99.96%) | 99.96% |
| L33 | " Paris" (99.7%) | 99.7% |

Parametric: already dominant by L24 (embedding starts at +22.5). Novel: not in top-5 until L29.

### 1D injection — novel vs parametric

| Test | 1D token | Result |
|------|----------|--------|
| Voltara (novel, recipient has no fact) | " Volt" | **99.997%** (works) |
| Paris to Germany (parametric, recipient has Berlin) | " Paris" | **Berlin 99.1%** (fails) |

1D injection works for novel facts (no competing signal) but cannot override parametric facts (FFN already provides the answer). The encyclopaedia's signal is too strong for a 1D perturbation.

## Key Findings

### 1. Knowledge is one dimension per fact
The actual knowledge content of a novel fact in the residual stream is a single scalar: the projection onto the answer token's embedding direction. One dimension out of 2560. This is sufficient for KL=0.0 transfer.

### 2. 86% dark is NOT knowledge
The L29 attention output is 86% dark space. This is routing/structural scaffolding, not answer information. Only the 14% in the answer token direction carries the fact. And even that is overkill — 0.058% of energy suffices.

### 3. The slot is generic, the filler is specific
At L15, all query types produce nearly identical slots (cos>0.997). The slot says "answer needed" not "city needed." Differentiation happens at L23-L29 during retrieval. The filler (specific answer) is a tiny perturbation on a massive shared structure.

### 4. Parametric is darker than novel
The encyclopaedia (FFN) operates in 93.8% dark space vs 85.8% for the notebook (KV). Parametric facts also activate semantic neighborhoods (Lyon alongside Paris), while novel facts copy only the specific answer.

### 5. Same format, different circuit
Novel: KV to attention to discrete jump at L29. Parametric: FFN to gradual build through L20-L26. Both produce ~1500 answer delta in the same token direction, but the parametric fact is already present from the embedding (+22.5).

### 6. The KV cache is an addressing system, not a knowledge store
56 GB of KV cache to deliver 7.25 KB of knowledge. The 7.7M:1 infrastructure ratio means >99.99% of the KV cache is addressing, scaffolding, and structural information.

### 7. H4's output is 99.4% dark
The fact copier head (L29 H4) writes almost entirely to dark space. Its output norm is 0.0284 vs 5637 for total attention. The fact is encoded in dark dimensions that project to the answer through downstream layer norm amplification and the unembedding matrix.
