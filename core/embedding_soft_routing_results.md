# Embedding Soft Routing Results

**Experiment:** 7114c8b4 | **Model:** google/gemma-3-4b-it | **Date:** 2026-03-21

## Hypothesis

Token IDs are 1D binary match. Embeddings are 2560D with semantic gradients. "landed" and "LAND" share embedding structure (cosine ~0.47). Route queries to windows using max-cosine soft matching between query token embeddings and window token embeddings. One matmul. No keywords. Pure geometry.

## Result: NEGATIVE

Embedding soft matching is **strictly worse** than token overlap for the one query that needs it (landing). The vocabulary gap between query and window is **semantic**, not string-level. Embeddings bridge case variants but cannot bridge concept gaps.

## Experiment 1 — Embedding Similarity Diagnostic

Query: "What did they say when they landed on the moon?"

| Query token | Best match in W370 | Cosine | Notes |
|-------------|:---:|:---:|---|
| " landed" | " LAND" | 0.467 | Bridges the case/form gap |
| " moon" | " blue" | 0.144 | **Nothing space-related in W370** |
| " say" | " the" | 0.118 | No semantic match |
| " when" | " in" | 0.397 | Function word similarity |
| " they" | " you" | 0.385 | Pronoun similarity |
| " on" | " on" | 0.992 | Exact match |
| " the" | " the" | 1.000 | Exact match |

### W370 vs W000 comparison

| Query token | Best in W370 | Score | Best in W000 | Score | Winner |
|-------------|:---:|:---:|:---:|:---:|:---:|
| " landed" | " LAND" | 0.467 | " got" | 0.210 | W370 (+0.26) |
| " moon" | " blue" | 0.144 | "Lunar" | 0.465 | **W000 (+0.32)** |

**The intro page** (W000) contains "Lunar module pilot" and matches "moon" **3x better** than W370 (the actual landing window). W370 contains mission-control jargon — CONTACT LIGHT, ENGINE STOP, ACA, DETENT — not space vocabulary.

## Experiment 2-3 — Window Scoring at N=50

| Method | Porridge | Baseball | Landing | Weather | News |
|--------|:---:|:---:|:---:|:---:|:---:|
| Token overlap (baseline) | 1 | 2 | 30 | 1 | 1 |
| Soft, no IDF | 1 | **1** | 41 | 1 | 1 |
| Soft + query IDF | 1 | **1** | **50** | 1 | 1 |
| Soft + threshold 0.3 | 1 | **1** | 41 | 1 | 1 |
| Soft + threshold 0.5 | 1 | **1** | 40 | 1 | 1 |
| Soft + IDF + thresh 0.3 | 1 | **1** | 49 | 1 | 1 |

**Baseball improves** from rank 2 to 1 (soft matching bridges vocabulary variants). But **landing degrades** from rank 30 to 41 (raw) to **50 (dead last)** with IDF.

### Why soft matching makes landing WORSE

Three compounding failures:

**1. Moon has no match.** W370 contains zero space-related tokens. The window is mission-control protocol: timestamps (04 06 45 32), speaker labels (CDR, LMP, CC), checklist items (ACA out of DETENT, ENGINE ARM OFF). "moon" best-matches "blue" at 0.14 — noise.

**2. Common tokens dominate.** Every window contains pronouns and function words. "they"→"you" (0.38), "when"→"in" (0.40), "What"→"We" (0.41) — these soft matches occur at equal strength across ALL 50 windows, creating uniform noise that drowns the one distinctive signal (landed→LAND 0.47).

**3. IDF is adversarial for this corpus.** In a transcript *about* a moon landing:
- " landed" IDF = 0.00 (appears in many/all windows)
- " moon" IDF = 0.00 (same)
- " say" IDF = 3.22 (rare — but matches nothing in W370)

IDF upweights the tokens with NO signal (say→0.12) and zeros out the ONE distinctive match (landed→LAND at 0.47). This drives landing to rank 50.

## Landing detail — why W370 scores low

Per-token breakdown for W370:

| Query token | Match in W370 | Cosine | IDF | Weighted |
|-------------|:---:|:---:|:---:|:---:|
| " landed" | " LAND" | 0.467 | 0.00 | 0.000 |
| " moon" | " blue" | 0.144 | 0.00 | 0.000 |
| " say" | " the" | 0.118 | 3.22 | 0.379 |
| " when" | " in" | 0.397 | 2.12 | 0.841 |
| " they" | " you" | 0.385 | 2.12 | 0.816 |
| "What" | "We" | 0.406 | 2.30 | 0.935 |

The two distinctive tokens contribute **exactly zero** to the IDF-weighted score. The entire score comes from function-word noise shared with every other window.

## Experiment 5-6 — Cost (Moot)

- 4.7 ms per query across 50 windows
- ~534 KB store for 724 windows
- Cost is reasonable but irrelevant — accuracy is worse than baseline

## Why Embeddings Cannot Bridge This Gap

The gap between "landed on the moon" and W370's content is not a string-level gap. It is a **concept-level gap**:

| What the query says | What W370 contains |
|----|----|
| "landed" | "CONTACT LIGHT", "ENGINE STOP" |
| "moon" | "04 06 45 42", "CDR", "LMP" |
| "they say" | "Tranquility Base here" |

The embedding matrix encodes morphological relationships (land↔LAND, moon↔Moon↔lunar). It does **not** encode that "CONTACT LIGHT" means "we just landed on the moon." That requires world knowledge — exactly what the model computes through 34 layers of processing, not what it stores in the embedding matrix.

The earlier embedding_neighbors result was misleading: "landed"→"landing" at 0.64 is real, but **"landing" doesn't appear in W370 either**. The embedding space maps word forms, not concepts.

## Architecture Implications

This is the third negative result for unsupervised routing of the landing query:
1. **Token overlap** — rank 43/724 (vocabulary gap)
2. **Dark space geometry** — 1/4 (GPS not topic index)
3. **Embedding soft matching** — rank 50/50 (concept gap, worse than baseline)

Versus the positive result:
- **Keywords** — rank 1/724, 5/5 queries, 800 bytes, zero computation

The landing query is the **hardest routing problem** in this corpus because the answer window uses domain-specific jargon that shares zero vocabulary (even softly) with the natural-language question. This is precisely the case that requires human-readable keyword extraction — a string like "moon landing" that a human would write but that never appears literally in the transcript.

Keywords remain the routing mechanism. 800 bytes. 5/5. No alternatives found.
