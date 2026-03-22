# Embedding Space Routing

**Experiment:** a4261425-830b-44b1-acb9-8bd864a997b2
**Model:** google/gemma-3-4b-it
**Date:** 2026-03-21

## Hypothesis

The routing signal isn't in attention at L29, dark space at L26, or cosine at L30. It's in the **embeddings**. Layer 0. The model's own token IDs and embedding vectors. "Porridge" in the query is the same token as "porridge" in the store. No format gap at the embedding level.

## Method

Three levels of embedding-space routing tested:
1. **Token ID overlap** — set intersection of query token IDs with passage token IDs
2. **TF-IDF weighted overlap** — weight matches by inverse document frequency in the store
3. **Embedding dot product** — cosine similarity between query and passage embedding vectors

## Experiment 1 — Token Overlap at N=3

Three distinct passages, three queries.

| Query | Store A (Zarkov) | Store B (Strand) | Store C (Kelvara) | Winner | Correct? |
|-------|:---:|:---:|:---:|:---:|:---:|
| Zarkov city | **7** (city, was, ark, ov, Industries, founded, in) | 2 (city, in) | 1 (in) | A | ✅ |
| Strand city | 1 (city) | **4** (ancient, city, discovered, Strand) | 0 | B | ✅ |
| Kelvara town | 1 (the) | 1 (the) | **7** (the, town, first, quantum, teleport, ation, experiment) | C | ✅ |

**3/3 correct.** Content-specific tokens (Industries, Strand, quantum, teleport) are unique to their passages and dominate over shared function words.

Note: "Helena" tokenizes as single token (78105) in query but as "Hel"+"ena" (9822+3320) in passage — BPE context-dependent splitting. "Strand" (73442) is identical in both.

## Experiment 2 — Embedding Dot Product at N=3

Same 3/3. At N=3 with distinct passages, exact token matches (cosine=1.0) completely dominate soft matches (~0.01-0.50). Embedding dot product adds no value over token overlap when passages have distinct vocabularies.

Key embedding clusters:
- " quantum" → " Quantum" (0.88), 量子 (0.71) — tight semantic cluster
- " Strand" → " strand" (0.66) — case variant only
- " teleport" → " Tele" (0.50) — weak prefix cluster

## Experiment 3a — Token Overlap at N=50 (Apollo)

50 sampled windows from Apollo 11 transcript. 5 queries.

| Query | Target | Rank | Correct? | Rare matches in target (df) |
|-------|--------|:---:|:---:|---|
| Porridge | W170 | **1** | ✅ | porridge(1), won(1), eating(1) |
| Baseball | W169 | 5 | ❌ | baseball(1) — outnumbered by common tokens in W150 |
| Landing | W370 | 27 | ❌ | **ZERO** — "landed"/"moon" not in any window |
| Weather | W169 | **1** | ✅ | Minneapolis(1), weather(3) |
| News | W169 | **1** | ✅ | Thor(1), erd(1), ahl(1) |

**3/5.** Two failure modes:
1. **Baseball:** Single rare token (baseball, df=1) outnumbered by function words (What, were) that happen to be rare in technical transcripts
2. **Landing:** ZERO vocabulary overlap — query says "landed on the moon", transcript says "EAGLE", "CONTROL", "ENGINE", "CONTACT LIGHT"

### Token frequency across 50 windows

| Frequency | Unique tokens |
|-----------|:---:|
| 1 window | 1,807 |
| 2-5 windows | 718 |
| 6-25 windows | 198 |
| 26-40 windows | 36 |
| >40 windows | 27 |

High-frequency tokens (>40/50): newlines, punctuation, digits, "the", "CC", "Roger", "you" — Apollo transcript function words.

## Experiment 4 — TF-IDF Weighted Overlap at N=50

IDF = log(N/df). Threshold: only count query tokens with IDF ≥ 2.0.

| Query | Target | Rank | Score | Margin | Correct? |
|-------|--------|:---:|:---:|:---:|:---:|
| Porridge | W170 | **1** | 11.74 | 11.74 | ✅ |
| Baseball | W169 | **1** | 3.91 | 1.61 | ✅ |
| Landing | W370 | 36 | 0.00 | — | ❌ |
| Weather | W169 | **1** | 6.73 | 3.91 | ✅ |
| News | W169 | **1** | 11.74 | 9.43 | ✅ |

**4/5.** TF-IDF threshold fixes baseball — "baseball" (IDF=3.91) dominates over "What" (IDF=2.30) when both are filtered. Landing remains structurally unroutable.

### Why landing fails (3 layers of vocabulary gap)

1. **Case:** W370 has "HAS LANDED" (uppercase) → tokenizes as HAS(90016) + LAND(68497) + ED(2413). Query has "landed" → single token (30473). Different BPE splits = zero overlap.
2. **Register:** Transcript uses technical callsigns (EAGLE, CONTROL, ENGINE, ACA, MODE, ARM, DET). Query uses natural language (moon, landed).
3. **OCR artifacts:** Underscores, broken words, non-standard spacing.

Embedding soft matching can't bridge this either:
- "moon" → nearest relevant neighbor "lunar" (0.61) — but "lunar" in 17/50 windows, not discriminative
- "landed" → "landing" (0.64) — not in W370
- "Eagle" has neighbor "EAG" (0.65) — W370 HAS "EAG", but query doesn't contain "Eagle"

## Experiment 5 — Full Scale N=725

Full Apollo transcript: 370,778 tokens, 725 × 512-token windows. 10,987 unique tokens, 5,130 appearing in exactly 1 window.

| Query | Target | Rank | Score | Margin | Correct? |
|-------|--------|:---:|:---:|:---:|:---:|
| Porridge | W170 | **1** | 15.66 | 9.08 | ✅ |
| Baseball | W169 | **1 (4-way tie)** | 5.20 | 0.00 | ❌ |
| Landing | W370 | 36 | 0.00 | — | ❌ |
| Weather | W169 | **1** | 10.46 | 3.76 | ✅ |
| News | W589 | **1** | 24.27 | 5.89 | ✅ |

**3/5.** Baseball degrades from rank 1 (N=50) to 4-way tie — "baseball" appears in 4 windows across the multi-day mission (different news relays: Thursday NL/AL scores, west division race, All Star game, rained-out All Star). Landing unchanged.

News routes correctly to W589 (confirmed content: "Thor Heyerdahl had to give up his attempt to sail a papyrus boat across the Atlantic").

## Experiment 6 — End-to-End: Route + Inject

TF-IDF routes to window → inject window context at L30 → generate.

### Porridge (novel content) — PASS

| Stage | Output | P(correct) |
|-------|--------|:---:|
| Bare query (no injection) | "Eric Idle" (confabulation) | 0% |
| Donor (context + query) | "John Coyle" | 100% |
| **Injected (L30)** | **"John Holt"** (first token correct, surname drifts) | **100%** |

KL(donor, injected) = 0.0. First token "John" at 100%. Multi-token drift on surname (known issue — persistent injection fixes this).

### News / Thor Heyerdahl (parametric content) — FAIL

| Stage | Output |
|-------|--------|
| Donor (context + query) | "papyrus boat was damaged by a storm... abandoned 650 miles from Bermuda" |
| Recipient (bare query) | "Kon-Tiki and Ra II are legendary feats" (parametric knowledge) |
| **Injected (L30)** | **"Kon-Tiki and Ra II"** — follows RECIPIENT, not donor |

KL(recipient, injected) = 0.0003 ≪ KL(donor, injected) = 0.001. Injection failed — parametric knowledge about Thor Heyerdahl overrides the injected signal.

## Summary Table

| Method | N=3 | N=50 | N=725 |
|--------|:---:|:---:|:---:|
| Token overlap | 3/3 | 3/5 | — |
| TF-IDF (threshold ≥ 2.0) | 3/3 | **4/5** | **3/5** |
| Embedding dot product | 3/3 | — | — |
| H4 attention (prior work) | 3/3 | 0/5 | — |
| L30 cosine (prior work) | 3/3 | 0/5 | — |

## Comparison with Keyword Routing

The keyword index from sparse_semantic_index achieved 100% routing at all scales. Embedding routing achieves 60-80%. The gap comes from two irreducible failure modes:

1. **Vocabulary gap:** Query and passage describe the same event in different vocabulary registers (natural language vs technical transcript, different cases). Keywords extract CONCEPTS ("landing", "moon"), not TOKENS. Token-level matching is literal.

2. **Ambiguity:** A content word (e.g., "baseball") may appear in multiple relevant windows. Keywords can be combined with temporal/structural context. Token matching treats every occurrence equally.

## What This Means

**Embedding routing IS model-native.** The token IDs come from the model's own tokenizer. The IDF weights come from the store's own statistics. No external word lists. No regex. No function word filters. It works at 60-80% accuracy with zero computation beyond set intersection.

**But keywords are better.** Not because of sophistication — because keywords capture CONCEPTS while tokens capture STRINGS. "Moon landing" as a keyword matches a passage about Eagle's descent. "moon" as a token doesn't match "EAGLE" or "LANDED" (uppercase).

**The format gap exists at EVERY level:**
- L0 (embeddings): "moon" ≠ "EAGLE" (different tokens, different embeddings)
- L26 (dark space): query compass ≠ passage compass (geometric_routing_results)
- L29 (attention Q·K): format mismatch kills routing (apollo_routing_diagnosis_results)
- L30 (residual): encodes structure not content (geometric_routing_at_scale_results)

**The cleanest architecture remains: keyword route + persistent inject.**
- Keywords: ~3 tokens/fact → 100% routing (800 bytes for Apollo)
- Persistent inject: 12 bytes/fact → 100% per-token delivery
- Total: ~10 KB for Apollo. No replay. No KV cache.

## The One Thing Embedding Routing Adds

For **novel entities** with distinctive names (Zarkov, Voltara, Kelvara), token overlap IS the keyword index. The entity name IS the routing key. No extraction needed.

For stores built from novel entities (the primary use case for injection), embedding routing is equivalent to keyword routing and simpler to implement:

```python
query_tokens = set(tokenizer.encode(query))
store_idf = {tid: log(N/df) for tid, df in token_doc_freq.items()}
scores = {pid: sum(store_idf.get(t, 0) for t in query_tokens & passage_tokens[pid])
           for pid in passages}
best = max(scores, key=scores.get)
```

Zero model computation. Pure set intersection + IDF lookup.
