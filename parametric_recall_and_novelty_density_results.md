# Experiment Results — Parametric Recall & Novelty Density

**Model:** google/gemma-3-4b-it
**Date:** 2026-03-16

---

## Experiment A — Parametric Recall Baseline

Measures what the model already knows *without* any context or sparse index. Tests three familiarity levels, then re-tests with entity anchors to measure the boost from priming.

### Phase A1–A4: Baseline Recall by Familiarity

| Topic | Familiarity | Score | Accuracy |
|-------|-------------|-------|----------|
| Apollo 11 | Very high | 7/9 | 78% |
| Gemini 8 | Moderate | 4/6 | 67% |
| Zarkov Expedition | Fictional | 0/0* | — |

\*All 6 Zarkov answers were **confabulations** — the model invented plausible-sounding details for a completely fictional topic.

#### Notable errors (without anchors)

| Question | Model Answer | Expected |
|----------|-------------|----------|
| Who stayed in orbit? | Buzz Aldrin | **Michael Collins** |
| UTC time of first step? | 20:17:40 UTC | **02:56 UTC** |
| Gemini 8 launch date? | March 21, 1966 | **March 16, 1966** |
| Gemini 8 landing? | Atlantic Ocean | **Pacific Ocean** |

The model confidently produces wrong answers for specific details — dates, times, and less-famous crew roles — while nailing major facts.

#### Zarkov confabulations

The fictional "Zarkov expedition" questions produced six fully fabricated answers with zero hedging:

| Question | Confabulated Answer |
|----------|-------------------|
| Who led it? | Captain James T. Kirk, 2259 |
| Objective? | Establish permanent human presence on Mars |
| Team size? | Twelve individuals |
| Discovery? | Habitable zone on Europa |
| Location? | Arctic Ocean, Siberian Deepwater Horizon |
| Equipment failure? | Primary communication system failure |

### Phase A5: Entity Anchor Boost

Re-asking the same questions with key entity names prepended to the prompt (e.g., "Key entities: Armstrong, Aldrin, Collins, Eagle, Columbia...").

| Topic | Without Anchors | With Anchors | Delta |
|-------|----------------|-------------|-------|
| Apollo 11 | 7/9 (78%) | 8/9 (89%) | +11% |
| Gemini 8 | 4/6 (67%) | 5/6 (83%) | +17% |

Anchors fixed the "who stayed in orbit" error (Collins now correctly identified) and the Gemini 8 landing location (Pacific instead of Atlantic). The UTC time error persisted — the model confuses 20:17 UTC (landing time) with 02:56 UTC (first step time) regardless of anchors.

### Hedging Rate

**0/22 answers contained hedging language** across all phases.

The model *never* says "I don't know" — not even when asked about a completely fictional expedition. This is the core problem that motivates contextual grounding: without external context, the model has no mechanism to distinguish what it knows from what it's inventing.

---

## Experiment B — Novelty Density Across Document Types

Measures per-token surprise (cross-entropy rank) across five document types. Each token is classified by how predictable it is to the model:

- **Parametric** (rank 0–2): model already knows this — top-3 prediction
- **Semi-parametric** (rank 3–50): partially predictable
- **Novel** (rank >50): genuinely new information the model couldn't predict

### Results

| Document | Tokens | Parametric | Semi | Novel | Median Rank | Mean Rank |
|----------|--------|-----------|------|-------|-------------|-----------|
| Wikipedia (Apollo 11) | 171 | 77.8% | 15.8% | 6.4% | 0 | 42.6 |
| Transcript (Apollo EVA) | 245 | 56.7% | 27.3% | 15.9% | 1 | 283.7 |
| Fiction (original) | 145 | 48.3% | 30.3% | 21.4% | 3 | 878.0 |
| Proprietary Memo | 181 | 42.5% | 35.4% | 22.1% | 4 | 982.3 |
| ML Paper (abstract) | 171 | 39.8% | 29.8% | 30.4% | 5 | 1038.7 |

### Surprise Distribution Histograms

**Wikipedia (Apollo 11)** — median rank 0, nearly all tokens are top predictions:
```
   top-1: ███████████████████████████  54.4% (93)
   top-3: ███████████  23.4% (40)
  top-10: ████   9.4% (16)
  top-50: ███   6.4% (11)
 top-200: █   3.5% (6)
  top-1K: █   2.3% (4)
 top-10K:    0.6% (1)
    10K+:    0.0% (0)
```

**Fiction (original passage)** — median rank 3, broad spread across all bins:
```
   top-1: ████████████████  33.8% (49)
   top-3: ███████  14.5% (21)
  top-10: ███████  15.9% (23)
  top-50: ██████  13.8% (20)
 top-200: █████  11.7% (17)
  top-1K: ██   4.8% (7)
 top-10K: ██   4.8% (7)
    10K+:    0.7% (1)
```

**ML Paper (abstract)** — median rank 5, heaviest tail above rank 200:
```
   top-1: ████████████  25.7% (44)
   top-3: ███████  14.0% (24)
  top-10: ███████  15.2% (26)
  top-50: ███████  14.6% (25)
 top-200: ██████  12.3% (21)
  top-1K: █████  11.1% (19)
 top-10K: ██   4.7% (8)
    10K+: █   2.3% (4)
```

**Transcript (Apollo 11 EVA)** — median rank 1, bimodal (many top-1, long tail):
```
   top-1: █████████████████████  43.7% (107)
   top-3: ██████  13.1% (32)
  top-10: ██████  12.7% (31)
  top-50: ███████  14.7% (36)
 top-200: ██   5.3% (13)
  top-1K: ██   4.9% (12)
 top-10K: ██   4.9% (12)
    10K+:    0.8% (2)
```

**Proprietary Memo** — median rank 4, significant mass above rank 1K:
```
   top-1: ██████████████  28.7% (52)
   top-3: ██████  13.8% (25)
  top-10: █████████  18.2% (33)
  top-50: ████████  17.1% (31)
 top-200: ███   6.6% (12)
  top-1K: ██   4.4% (8)
 top-10K: ███   7.2% (13)
    10K+: █   3.9% (7)
```

### Most Surprising Tokens Per Document

The highest-rank tokens reveal *what kinds of information* the model cannot predict:

| Document | Top Surprise Tokens | Peak Rank |
|----------|-------------------|-----------|
| Wikipedia | `Commander`, `Michael`, `first`, `lunar`, `pilot` | 4,245 |
| Fiction | `Thorn`(field), `crumpled`, `Elena`, `Gret`(el), `bartender` | 102,034 |
| ML Paper | `ROLL`(S), `PG`(-19), `transformers`, `surprise`, `entity` | 96,465 |
| Transcript | `CMP`, `Roger`, `CC`, `Tranqu`(ility), `LMP` | 14,550 |
| Proprietary | `migrate`, `Sarah`, `Valk`(ey), `Webb`, `auth` | 33,587 |

Pattern: **proper nouns, domain-specific abbreviations, and specific numbers** are consistently the most surprising. The model predicts syntactic structure and common phrases well but cannot anticipate *which* names, *which* numbers, or *which* technical terms appear.

### Key Observations

1. **Wikipedia is almost entirely parametric.** 78% of tokens are in the model's top-3 predictions. The model has essentially memorized this content — replaying it as context adds very little.

2. **Transcripts are surprisingly predictable.** Despite containing novel conversational content, 57% of tokens are parametric. The model predicts dialogue structure well but struggles with specific names, timestamps, and verbatim quotes.

3. **Proprietary content is the highest-value target.** With only 42% parametric tokens and a mean rank of 982, internal memos contain information the model genuinely cannot predict — names, dates, dollar figures, build numbers.

4. **ML papers have the highest novelty density.** 30% of tokens are novel, driven by specific numbers (94.7%, 340%, 2.1%), method names, and technical claims the model hasn't seen.

5. **Fiction's surprise is character-driven.** The highest-rank token across all documents (102,034) was `Thorn` from "Thornfield Arms" — a proper noun the model has no basis to predict. Character names (`Elena`, `Gretel`, `Marcus`) and specific actions (`crumpled`, `nursing`) dominate fiction's novel tokens.

### Implications for Sparse Indexing

The novelty density directly determines how much a sparse index can compress:

- **Wikipedia**: ~6% novel → index is tiny, but also unnecessary (model already knows it)
- **Proprietary memo**: ~22% novel → index captures the genuinely new facts
- **ML paper**: ~30% novel → highest index density, but highest value per token

The sweet spot for sparse indexing is content where novelty density is moderate (15–30%) — enough new information to be worth indexing, but enough parametric structure that the model can reconstruct around the novel facts.

---

## Experiment C — The Real Apollo 11 Demo

Full pipeline: pre-built checkpoint library from the Apollo 11 air-to-ground transcript, comparing three retrieval modes:

1. **Parametric only** — no context, pure model memory
2. **Sparse index** — a hand-crafted ~1KB text summary of key moments
3. **Window replay** — replaying specific KV checkpoint windows from the library

### Phase C3: Single-Fact Retrieval

Six questions spanning parametric facts, partially-parametric facts, and novel transcript details.

| Query | Type | Parametric | Sparse Index | Window Replay |
|-------|------|-----------|-------------|---------------|
| Mission commander | parametric | pass | pass | — |
| Bruce audio quality | novel | fail | pass | pass |
| Particle trajectory | novel | fail | pass | pass |
| Nixon call intro | partial | pass | pass | pass |
| Strange noises | novel | fail | pass | pass |
| Aldrin "home" quote | novel | fail | pass | pass |

**Parametric only gets 1–2/6.** It knows the commander but confabulates novel transcript details.
**Sparse index gets 6/6.** A ~1KB text summary is sufficient for single-fact retrieval.
**Window replay gets 5/5** (of those with window IDs). Full KV context provides the highest fidelity.

### Phase C4: Cross-Window Synthesis

Open-ended questions requiring reasoning across multiple transcript moments:

- **"Find 3 amusing moments"** — Parametric invents plausible but fictional moments. Sparse index identifies real moments (the "living room" audio joke, Aldrin's "home" comment, the "friends" noises).
- **"5 key moments"** — Parametric lists well-known events generically. Sparse index cites specific timestamps and verbatim quotes.
- **"Crew dynamics"** — Parametric gives a generic textbook answer. Sparse index draws on actual conversational exchanges.

---

## Summary

| Finding | Evidence |
|---------|----------|
| The model never admits ignorance | 0/22 hedged answers, even on fiction |
| Entity anchors boost recall +11–17% | Phase A5 vs A1/A4 |
| Specific details fail parametrically | Dates, times, and lesser-known roles are unreliable |
| Wikipedia is ~78% parametric | Novelty density measurement |
| Proprietary content is ~42% parametric | Highest value for sparse indexing |
| A 1KB sparse index matches window replay | 6/6 vs 5/5 on single-fact retrieval |
| Sparse index enables cross-window synthesis | Parametric confabulates; index grounds |
