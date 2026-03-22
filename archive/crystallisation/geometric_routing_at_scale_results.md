# Geometric Routing at Scale Results

**Experiment ID:** 45ecac84-75f0-4d35-8667-af8b4b5578e5
**Model:** google/gemma-3-4b-it (34 layers, 2560 hidden dim)
**Date:** 2026-03-21

## Summary

**NEGATIVE.** Cosine similarity at L30 does NOT route queries to correct passages within a single document. The correct passage ranks 36th out of 50 windows (bottom 28%). The keyword index is required.

## Background

The persistent transplant experiment showed cosine routing worked 3/3 at N=3 (Zarkov vs Strand vs Apollo), with margins ~0.005. This experiment tests whether that scales to N=50+ within a single document (Apollo 11 transcript, 725 passages of 512 tokens).

## Method

- Built 50 representative windows from the Apollo 11 transcript (~99K lines), sampling every ~2000 lines plus key content windows (porridge/baseball at line 23390, landing at line 50835)
- Formatted each as chat-template passage: `<start_of_turn>user\n{text}<end_of_turn>\n<start_of_turn>model\n`
- Computed cosine similarity at L30 between the porridge query (`"Who won the porridge eating contest?"`) and all 50 windows
- Used compare_activations in batches of 8

## Results

### Porridge Query Ranking (N=50)

| Rank | Window | Cosine | Note |
|------|--------|--------|------|
| 1 | W012_L24840 | 0.9770 | Random transcript |
| 2 | W028_L57960 | 0.9766 | Random transcript |
| 3 | W041_L84870 | 0.9760 | Random transcript |
| ... | ... | ... | ... |
| **36** | **W_PORRIDGE_L23390** | **0.9664** | **CORRECT** |
| ... | ... | ... | ... |
| 50 | W036_L74520 | 0.9509 | Random transcript |

### Score Distribution

- **Range:** 0.9509 - 0.9770 (spread: 0.0261)
- **Mean:** ~0.969
- **Correct window score:** 0.9664 (28th percentile)
- **Correct window rank:** 36/50

### Landing Query (Partial)

In a 7-window batch test, the landing window ranked #1 (0.9699) with 0.007 margin. But this is misleading — the query mentions "Eagle" and "Houston" which appear throughout the transcript. Not tested against full 50.

## Why It Fails

### 1. L30 encodes structure, not content

Within a single document (Apollo transcript), all windows share the same structural patterns: speaker labels (CDR, CC, LMP), timestamps (02 05 57 48), call signs (Houston, Eagle, Columbia). These dominate the L30 residual, creating a massive shared component (~0.97 cosine between all pairs).

The content-specific signal — that one window mentions "porridge" while another mentions "P22 alignment" — is negligible compared to the structural component.

### 2. Format gap is fatal

The query ("Who won the porridge eating contest?") is semantically different from the passage ("In Corby, England, an Irishman, John Coyle has won the world's porridge eating championship..."). Even though both mention porridge, the L30 residuals encode:
- Query: question-answering retrieval state
- Passage: document continuation state

These are geometrically distant in dark space.

### 3. Why N=3 worked

The 3/3 result at N=3 succeeded because:
- Zarkov (novel entity fiction) vs Strand (physics prize) vs Apollo (space transcript)
- These are **maximally different domains**
- Cross-domain separation (~0.005) >> intra-document separation (~0.001)
- L30 distinguishes document domains, not passages within a domain

## Consistency with Prior Work

This result is fully consistent with:
- **dark_space_geometric_routing_results.md:** "Pure dark space geometry does NOT route cross-format (query→window). Dark space = GPS (WHERE), not topic index (WHAT)."
- **compass_routed_injection_results.md:** "L26 compass routing 2/5 — format gap between chat queries and transcript windows."

## Implications

### What works
- **Cross-domain routing:** Cosine at L30 correctly distinguishes Zarkov from Strand from Apollo (3/3)
- **Persistent injection:** 100% per token once the right passage is identified (proven in persistent transplant experiment)

### What doesn't work
- **Intra-document routing:** Cosine at L30 cannot find the right passage within a document
- **Geometric routing at scale:** N=50 is already too many for intra-document search

### Final Architecture

```
Store:     10 KB crystallised residual per passage
Index:     Sparse keyword index (~3 tokens/fact, 800 bytes total)
Route:     Keyword filter → candidate set (N ≤ 5)
Deliver:   Persistent injection at L30 every generation step
Cost:      Apollo 11 = 7.25 MB residuals + 800 bytes index
```

**The keyword index is not optional.** It's the routing mechanism. Geometric comparison only works for cross-domain discrimination (selecting between Apollo and Shakespeare, not between Apollo passages).

The architecture is: **keyword route + persistent inject.** Not pure geometry. But the delivery mechanism (persistent injection) is the real breakthrough — 100% per token, 10 KB per passage, no replay.
