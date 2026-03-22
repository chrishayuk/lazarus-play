# Vector Routing with Final Residual

**Experiment ID:** 869f60eb
**Model:** google/gemma-3-4b-it (34 layers, 2560 hidden dim)
**Date:** 2026-03-21
**Status:** NEGATIVE

## Hypothesis

Every prior routing experiment used bare queries — no document context.
The final residual (document Markov state) was never tested as a routing
variable. If the query is processed in the presence of the document's
geometric state, cosine routing might succeed where bare queries fail.

## Method

### Five conditions tested

1. **Bare** — Standard chat-wrapped query, no document context
2. **Text context** — W690 (last sampled transcript window) prepended to
   query in the same user turn
3. **Inj@L0** — W690's last-token residual injected at layer 0 of the
   query's forward pass (replaces last-token residual, continues forward)
4. **Inj@L7** — Same injection at layer 7 (entity identity layer)
5. **Inj@L14** — Same injection at layer 14 (Markov threshold / compass)

### Queries and targets

| Query | Natural language | Target window |
|-------|-----------------|---------------|
| porridge | "Who won the porridge eating contest?" | W170 |
| baseball | "What were the baseball scores?" | W169 |
| landing | "What did they say when they landed on the moon?" | W370 |
| weather | "What was the weather in Minneapolis?" | W169 |
| news | "What happened with Thor Heyerdahl's boat?" | W169 |

50 sampled windows from Apollo 11 transcript. Rank = position of target
window when all 50 are sorted by similarity to query.

## Results

### L30 centred cosine (primary metric)

| Query | Bare | Text ctx | Inj@L0 | Inj@L7 | Inj@L14 | Target |
|-------|:----:|:--------:|:------:|:------:|:-------:|--------|
| porridge | **8**/50 | 16/50 | **6**/50 | 8/50 | 23/50 | W170 |
| baseball | **5**/50 | 8/50 | 11/50 | **5**/50 | 13/50 | W169 |
| landing | 47/50 | 44/50 | **40**/50 | 46/50 | 48/50 | W370 |
| weather | 12/50 | 10/50 | 14/50 | 12/50 | **5**/50 | W169 |
| news | 22/50 | **7**/50 | 15/50 | 22/50 | 18/50 | W169 |

### L30 raw cosine

| Query | Bare | Inj@L0 | Inj@L7 | Inj@L14 | Target |
|-------|:----:|:------:|:------:|:-------:|--------|
| porridge | **12**/50 | 12/50 | 12/50 | 20/50 | W170 |
| baseball | **6**/50 | 11/50 | 6/50 | 8/50 | W169 |
| landing | **46**/50 | 46/50 | 46/50 | 47/50 | W370 |
| weather | **3**/50 | 6/50 | 5/50 | 6/50 | W169 |
| news | 18/50 | **17**/50 | 18/50 | 17/50 | W169 |

### L26 centred cosine

| Query | Bare | Inj@L0 | Inj@L7 | Inj@L14 | Target |
|-------|:----:|:------:|:------:|:-------:|--------|
| porridge | 11/50 | **10**/50 | 10/50 | 31/50 | W170 |
| baseball | 25/50 | **17**/50 | 24/50 | 29/50 | W169 |
| landing | **46**/50 | 43/50 | 45/50 | 48/50 | W370 |
| weather | **6**/50 | 20/50 | 7/50 | 21/50 | W169 |
| news | **20**/50 | 20/50 | 22/50 | 22/50 | W169 |

### Text context — all four metrics

| Query | L30 raw | L30 centred | L26 raw | L26 centred | Target |
|-------|:-------:|:-----------:|:-------:|:-----------:|--------|
| porridge (bare) | 12 | 8 | 34 | 11 | W170 |
| porridge (+ctx) | 43 | 16 | 50 | 38 | W170 |
| baseball (bare) | 6 | 5 | 15 | 25 | W169 |
| baseball (+ctx) | 13 | 8 | 16 | 24 | W169 |
| landing (bare) | 46 | 47 | 47 | 46 | W370 |
| landing (+ctx) | 39 | 44 | 43 | 42 | W370 |
| weather (bare) | 3 | 12 | 8 | 6 | W169 |
| weather (+ctx) | 8 | 10 | 12 | 9 | W169 |
| news (bare) | 18 | 22 | 36 | 20 | W169 |
| news (+ctx) | 9 | 7 | 11 | 22 | W169 |

## Analysis

### No condition consistently improves routing

- **Text context** helps news dramatically (L30 centred: 22→7, L26 raw:
  36→11) but destroys porridge (L30 raw: 12→43, L26 centred: 11→38).
  The context OVERWHELMS the query signal — the model attends to the
  W690 content, not the question.

- **Injection at L0** shows minor fluctuations (±5 ranks) with no
  systematic direction. The single-position residual replacement at the
  embedding layer is too weak to shift 2560-dim geometry meaningfully.

- **Injection at L7** is nearly identical to bare in most cases. The
  entity identity layer passes through the injection without amplification.

- **Injection at L14** is actively harmful for some queries (porridge:
  8→23, baseball: 5→13) while helping weather (12→5). The compass layer
  overwrites the query's content signal with a document-structure signal.

### Landing is unfixable by context

Landing ranks 39-48/50 across ALL conditions. This is not a context gap
— the model simply does not encode "What did they say when they landed
on the moon?" in a way that is geometrically close to the transcript
passage about the lunar landing. The query activates parametric knowledge
about Apollo 11 (well-known event), which maps to a completely different
region of activation space than the verbatim transcript.

### Why text context helps news but hurts porridge

The news query ("Thor Heyerdahl's boat") is obscure enough that the model
has no parametric commitment. Adding transcript context shifts the query
into a transcript-like region of activation space, which happens to be
closer to W169. But porridge ("porridge eating contest") is even more
unusual — adding context drowns the porridge signal in transcript-structure
noise.

## Conclusion

**The final residual is NOT the missing routing variable.**

The format gap between "asking about X" (query) and "reading about X"
(passage) is fundamental to the model's representation geometry. It is
not caused by missing document context. Adding context either:
1. Overwhelms the query signal (text context)
2. Has negligible effect (L0/L7 injection)
3. Actively interferes (L14 injection)

The three routing mechanisms that work remain:
1. **Keyword index** — 800 bytes, 100% retrieval (confirmed)
2. **H-space cosine at L29** — works cross-domain but fails intra-document
3. **Attention-routed injection** — works for novel entities with
   format-matched templates

For natural language Q&A over documents, the keyword index is the
correct architecture. The vector routing hypothesis is conclusively dead.

## Implications for the knowledge store

The persistent injection architecture (keyword route → persistent inject)
is confirmed as the right design. This experiment eliminates the last
plausible alternative to keyword routing.

Final architecture:
- **Index:** keyword extraction (~3 tokens/fact)
- **Route:** keyword match (string comparison, not vector similarity)
- **Inject:** persistent L30 residual injection (100% per-token accuracy)
- **Store size:** ~10 KB per passage, no replay needed
