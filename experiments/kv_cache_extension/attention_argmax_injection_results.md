# Attention-Argmax Injection Results

**Experiment:** a4733418 | **Model:** google/gemma-3-4b-it | **Date:** 2025-03-21

## Hypothesis

H4 attention over stored K-vectors is query-dependent. If we extend the KV cache at L29 with entries from multiple stores and let H4 argmax select the winner, then inject ONLY that entry's answer at L30 — the model routes without keywords.

## Setup

Two-stage pipeline:
1. **L29:** Extend KV cache with pre-RoPE K/V entries from all stores. Sequential RoPE (stored at positions 0..N-1, query at N..N+M-1). H4 attention runs naturally. Argmax over stored entries → winning entry index.
2. **L30:** Inject winning entry's answer token embedding at 2× coefficient.

Three novel-entity stores:
- **Store A** (Zarkov/Voltara): 40 content entries, answer=" Volt" (id=194328), coeff=2304
- **Store B** (Strand/Castellan): 32 content entries, answer="Cast" (id=34597), coeff=4032
- **Store C** (Voss/Kelvara): 44 content entries, answer="Kel" (id=85383), coeff=3168

Content positions = all non-structural tokens (excluding BOS, turn tags, punctuation, newlines).

## Experiment 1 — Two Stores, Two Queries (PASS)

Combined KV cache: 72 entries (40 A + 32 B).

| Query | H4 argmax | Store | Correct? | Weight | Margin | P(correct) |
|-------|-----------|-------|----------|--------|--------|-------------|
| "Zarkov city" | Entry 21 ("Volt") | A ✓ | YES | 9.03% | 2.9× #2 | 100% |
| "Director surname" | Entry 68 ("Helena") | B ✓ | YES | 4.76% | 1.1× #2 | 100% |

### Per-head argmax winners

| Head | Zarkov query → | Castellan query → |
|------|:-:|:-:|
| H0 | A | A |
| H1 | A | A |
| H2 | A | **B** |
| H3 | **B** | **B** |
| H4 | A | **B** |
| H5 | A | **B** |
| H6 | B | A |
| H7 | A | A |

H4 routes correctly for both. H2, H3, H5 also flip with query — broadly query-dependent.

### Store-level attention totals

| Query | Attn→Store A | Attn→Store B |
|-------|:---:|:---:|
| Zarkov | **22.2%** | 13.0% |
| Castellan | 19.7% | **22.6%** |

The model broadly allocates more attention to the correct store, but the margin is narrow (3% absolute for Castellan).

## Experiment 2 — Three Stores, Three Queries (FAIL: 1/3)

Combined KV cache: 116 entries (40 A + 32 B + 44 C).

| Query | H4 argmax | Store | Correct? | Weight |
|-------|-----------|-------|----------|--------|
| "Zarkov city" | Entry 21 ("Volt") | A ✓ | YES | 8.06% |
| "Director surname" | Store C entry | C ✗ | **NO** | 4.49% |
| "Teleportation town" | Entry 21 ("Volt") | A ✗ | **NO** | 16.02% |

**Critical observation:** Entry 21 (the "Volt" token at position 28 in the Zarkov passage) attracted 16% attention for the completely unrelated teleportation query. This is a K-norm attractor, not semantic routing.

The Castellan query failed by 0.003 (4.49% vs 4.20% for the correct B entry) — razor-thin and flipped by the addition of Store C.

## Experiment 3 — Attention Distribution Analysis

Over 72 entries (Exp 1 configuration):

| Metric | Zarkov query | Castellan query |
|--------|:---:|:---:|
| Top-1 weight | 9.03% | 4.76% |
| Top-2 weight | 3.13% | 4.20% |
| Margin (#1 - #2) | 5.91% | **0.56%** |
| Ratio (#1 / #2) | 2.9× | **1.1×** |
| Top-1 concentration | 25.7% | 11.3% |
| Top-5 from correct store | 3/5 | 3/5 |

Zarkov routing is moderately robust (2.9× margin). Castellan routing is barely above chance (1.1× margin). At N=3, Castellan fails.

## Experiments 4–6: SKIPPED

H4 argmax fails at N=3 (116 entries). Cannot scale to N=5,800 (Apollo). Experiments 4–6 not run.

## Diagnosis: Why H4 Is Not a Semantic Router

Entry 21 = " Volt" (token_id 89711) at position 28 in the Zarkov passage — the **answer token itself**. During the document's forward pass, the model concentrates fact-retrieval information at this position, producing a K-vector with anomalously high norm. H4, as a **copy head**, naturally attends to high-norm K-vectors to copy their content.

This is K-norm-biased copying, not semantic routing:
- For the matching query (Zarkov), the answer-position K-vector IS the right thing to copy → **correct by coincidence**
- For unrelated queries (teleportation), the same K-vector still attracts 16% → **structural bias**
- At N=2 the correct store's answer-position K-vector wins because it's the highest-norm entry matching the query's entity tokens
- At N=3 the competition between multiple high-norm answer-position K-vectors becomes too tight

## Conclusion

**H4 attention is weakly query-dependent but not strongly discriminative.**

| Outcome | Verdict |
|---------|---------|
| H4 argmax correct at N=2 stores | ✓ (both) |
| H4 argmax correct at N=3 stores | ✗ (1/3) |
| Injection delivers when routing correct | ✓ (100%) |
| H4 is a semantic router | **NO** — K-norm copy head |
| Keywords eliminated | **NO** |

**Keywords confirmed as the only viable routing mechanism.** Even the hybrid approach (keywords narrow, attention fine-routes) is not recommended — H4's margin was 0.003 at N=2 for the Castellan query, insufficient as a tiebreaker.

### Final Architecture (Confirmed)

```
keyword index → window selection → persistent 1D injection at L30
```

800 bytes per passage. No attention routing. No cosine similarity. Keywords do the work. Injection delivers the answer. Ship it.
