# Dark Space Geometric Routing — Results

**Experiment ID:** 615dc1c1
**Model:** google/gemma-3-4b-it
**Date:** 2026-03-21
**Status:** NEGATIVE RESULT — pure geometric routing in dark space does not bridge the query-to-window format gap

## Summary

The hypothesis was that L26 dark space coordinates (16D content subspace) could directly route natural language queries to matching store entries by geometric proximity. **This fails.** The dark space is the model's GPS — it tracks WHERE in the content stream the model is processing. It does NOT encode WHAT the content is about in a way that bridges the format gap between "asking about X" (query) and "reading X" (window content).

Five routing mechanisms were tested. Only direction dot products work, and they require knowing which token direction to project onto (supervised signal).

## What Was Tested

### 1. Full-Space Cosine Similarity (2560D)

Raw L26 cosine similarity between bare queries (no chat template) and window content.

| Query | Sports window | News window | Landing window | EVA window |
|-------|-------------|------------|---------------|------------|
| Sports | 0.961 | **0.974** | 0.971 | 0.969 |
| News | 0.955 | **0.968** | 0.966 | 0.964 |
| Landing | 0.930 | 0.945 | 0.945 | **0.946** |
| Technical | 0.957 | **0.973** | 0.973 | 0.973 |

**Result:** 1/4 correct. Full-space cosine is dominated by structural signal. The sports query is closest to the NEWS window, not the sports window.

With chat template, it's worse: queries cluster with EACH OTHER (0.95-0.99) because the chat template dominates the residual.

### 2. Dark Space PCA Subspace (16D)

Calibrated L26 PCA compass from 20 diverse Apollo window texts (rank=16). Built dark tables for windows and queries. Tested subspace surgery — replace query's 16D coordinates with each window's coordinates.

**Result:** ALL four surgeries produce IDENTICAL text:
> "During the Apollo 11 mission, Mission Control relayed a series of sports scores to Neil Armstrong..."

The 16D subspace carries NO routing-relevant signal. The query's semantic intent lives entirely in the orthogonal complement (which is preserved by surgery). The subspace captures structural variation between windows, not content routing information.

**Energy analysis:** 86.4% of the query's energy is in the 16D subspace. But this is STRUCTURAL energy (the dominant PCA directions of the model's L26 activation space), not content energy.

### 3. Logit Lens Top-K Comparison

Decoded the L26 residual into vocabulary space (after layer norm) for queries and windows.

| Prompt | Top-5 predictions |
|--------|-------------------|
| Sports query | \n\n (98%), What, radio, **sports**, **Sports** |
| Sports window | \n (94%), **Chicago, Boston, Detroit, Philadelphia** |
| News query | **Today** (48%), **today** (20%), \n\n, **read**, **news** |
| News window | \n (59%), **It** (36%), Its, **Scientists**, The |

**Result:** No top-k overlap. The sports query predicts "sports/baseball" but the sports window predicts city names (from the scores being read). They predict different tokens for different completion contexts. **Logit lens top-k is NOT a routing signal.**

### 4. Direction Dot Products (Token Embedding Projections)

Measured the L26 residual's dot product with specific token embedding directions.

| Prompt | → scores | → news | → forecast | → weather | → Houston | → Apollo |
|--------|----------|--------|------------|-----------|-----------|----------|
| Sports query | **1430** | 1079 | 1377 | 793 | 546 | 112 |
| Sports window | **1180** | 326 | 1036 | 550 | 1148 | -178 |
| News query | 742 | **2009** | 1548 | 904 | 272 | -77 |
| News window | 967 | 917 | 1408 | 328 | 303 | 122 |

**Result:** ROUTES CORRECTLY when the right direction is known:
- Sports query nearest to sports window by " scores" dot product: |1430-1180|=250 < |1430-967|=463 ✓
- News query nearest to news window by " news" dot product: |2009-917|=1092 < |2009-326|=1683 ✓

**But:** Requires knowing WHICH token to project onto. This is keyword matching in embedding space, not unsupervised geometric routing.

### 5. Mixed PCA Content Components

PCA of combined query+window prompts at L26 found content-routing dimensions:

| PC | Variance | Positive direction | Negative direction |
|----|----------|-------------------|-------------------|
| PC2 | 19% | forecast, Forecast | Apollo, Houston, Neil |
| PC6 | 6% | storms, weather, rain | Shuttle, robotics |

These separate weather from space content. But using them for routing requires supervised selection of which PCs encode content vs format.

## The Core Finding

**The dark space is the model's GPS, not its topic index.**

- **Dark space** (L26 PCA subspace, orthogonal to vocabulary) encodes WHERE the model is in the content stream — its navigational position. Two windows reading different sports content have similar dark coordinates. But a query ASKING about sports has completely different dark coordinates because "asking about X" and "reading X" are different navigational states.

- **Vocabulary space** (L26 residual projected onto token embeddings) encodes WHAT the content is about. Both the sports query and sports window project onto " scores" with similar magnitude. This IS the shared routing signal — but it lives in vocabulary-interpretable dimensions, not dark space.

The compass works for window-to-window comparison because both are in "reading" mode — same format, different content. The dark coordinates correlate with content type WITHIN format. But query-to-window is cross-format, and the dark coordinates don't bridge that gap.

## Architecture Implications

### What Doesn't Work for Cross-Format Routing
- Pure dark space distance (16D PCA)
- Full-space cosine similarity (2560D)
- Logit lens top-k overlap
- Subspace surgery at L26

### What Works
- **Sparse semantic index** (keyword extraction from windows, 3 tokens/fact, already proven at 100% retrieval)
- **Direction dot products** (L26 residual projected onto content-specific token embeddings)
- These are both vocabulary-space mechanisms

### Recommended Combined Architecture

```
Stage 1: Keyword routing (vocabulary space)
  Query → extract keywords → match against sparse index → candidate windows
  Already proven: 100% retrieval at ~3 tokens/fact, 800 bytes vs 56 GB

Stage 2: Dark space fine routing (within candidates)
  Compass coordinates → nearest entries within candidate windows
  This IS same-format comparison (window context → entry positions)

Stage 3: L30 injection (12 bytes/entry)
  Proven mechanism, >98% delivery
```

This is a 3-stage pipeline, not the 1-stage pure geometry originally hoped for. But the keyword index is cheap (800 bytes total), the compass works for within-window routing, and injection is proven. The full system works — it's just not pure geometry.

### What Dark Space IS Good For

1. **Window-to-window comparison** (compass routing — proven)
2. **Content type clustering** (windows cluster by content in dark space)
3. **Navigational state tracking** (where the model is in its processing)
4. **Same-format fine routing** (which entry within a window)

### The 72 bytes/entry Store

The original plan for 72 bytes/entry (dark coords + token_id + coefficient) is still viable for WITHIN-WINDOW routing (Stage 2). But the routing to the right window (Stage 1) needs the sparse keyword index, not dark space coordinates. Total store:

| Component | Size | Purpose |
|-----------|------|---------|
| Sparse keyword index | ~800 bytes | Cross-format routing (Stage 1) |
| Dark coords per entry | 64 bytes × 23,200 = 1.45 MB | Within-window routing (Stage 2) |
| Token ID + coefficient | 6 bytes × 23,200 = 136 KB | Injection payload (Stage 3) |
| **Total** | **~1.6 MB** | Full Apollo 11 store |

## Key Numbers

| Metric | Value |
|--------|-------|
| Full-space cosine routing accuracy | 1/4 (25%) |
| Dark space surgery differentiation | 0/4 (0%) |
| Logit lens top-k overlap | 0/4 (0%) |
| Direction dot product routing | 2/2 (100%) with known directions |
| Sports query → " scores" dot product | 1430 (query), 1180 (window) |
| News query → " news" dot product | 2009 (query), 917 (window) |
| Format gap (chat template cosine) | Queries cluster 0.97-0.99, windows 0.94-0.97 |
| Subspace energy fraction | 86.4% (structural, not routing-relevant) |
