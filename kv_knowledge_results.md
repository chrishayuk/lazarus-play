# Knowledge in the KV Cache

**Experiment d3bf96ea — google/gemma-3-4b-it (34 layers, 2560D)**

## The Question

The model processes tokens into KV cache. Somewhere in that cache is knowledge — every fact, relationship, entity. Prior experiments showed:
- Knowledge lives in V vectors at specific positions
- Knowledge is addressable via K vectors encoding relation templates
- Knowledge uses ~7 of 34 layers; the rest is scaffolding
- Relation context is required for retrieval

**The deeper question**: What IS knowledge in the KV cache? How much of it is knowledge vs scaffolding? How is it structured? How does it relate to parametric knowledge in FFN weights?

## Setup — Three Novel Facts

Context prompt (59 tokens, with chat template):
```
Zarkov Industries was founded in the city of Voltara in 1987.
The crew reported that the audio quality was scratchy during descent.
Joe Namath agreed to sell his restaurant and report to the New York Jets training camp.
```

### Token Map (key positions)

| Pos | Token | Role |
|-----|-------|------|
| 0 | `<bos>` | Structural |
| 1-3 | `<start_of_turn>` user `\n` | Template |
| 4-6 | Z ark ov | Entity (Zarkov) |
| 7 | Industries | Entity modifier |
| 8-9 | was founded | Relation verb |
| 10-11 | in the | Relation prep |
| 12-13 | city of | **Relation context** (Voltara) |
| 14-15 | Volt ara | **Fact value** (Voltara) |
| 16-22 | in 1 9 8 7 . | Year + separator |
| 23-26 | The crew reported that | Filler / intro |
| 27-28 | the audio | Subject |
| 29 | quality | **Relation context** (scratchy) |
| 30 | was | Relation verb |
| 31-32 | scratch y | **Fact value** (scratchy) |
| 33-35 | during descent . | Filler + separator |
| 36-38 | Joe Nam ath | Entity (Namath) |
| 39-46 | agreed to sell his restaurant and report to | Filler / narrative |
| 47-49 | the New York | Location context |
| 50 | Jets | **Fact value** (Jets) |
| 51-52 | training camp | Context |
| 53 | . | Separator |
| 54-58 | `<end_of_turn>` `\n` `<start_of_turn>` model `\n` | Template |

### Baseline Retrieval

| Query | Target | P(target) | Status |
|-------|--------|-----------|--------|
| "Zarkov Industries was founded in the city of" | " Volt" | **100%** | Novel — pure context retrieval |
| "The audio quality was" | " scratch" | **59.8%** | Novel — context retrieval with uncertainty |
| "Joe Namath reported to the" | " New" | **100%** | Mixed — Namath+Jets is parametric |

All three facts are retrievable. Voltara and Jets at 100% confidence, scratchy at 60% (model hedges with "a", "poor", "inconsistent" as alternatives — reasonable since "scratchy" is a specific adjective the model wouldn't default to).

## Experiment 1 — The Knowledge Fraction

**Goal**: How many (position, layer) entries in the KV cache carry retrievable knowledge?

### Approach

The plan was exhaustive position × layer ablation: zero K,V at each position at each layer, test all 3 queries. With 59 positions × 14 key layers × 3 queries = 2,478 interventions.

**Challenge**: `full_causal_trace` (Meng et al. 2022 style position × layer heatmap) runs one forward pass per cell. For 59 × 14 = 826 forward passes per query, this is too slow for interactive use (~15-20 minutes per query).

**Alternative approaches available**:
- `trace_token`: Layer-level causal tracing (~34 passes per query) — identifies which LAYERS matter
- `attention_pattern`: Which positions get READ at each layer — identifies which POSITIONS matter
- `component_intervention`: Zero attention or FFN at specific layers — confirms component roles
- `logit_attribution`: Decompose per-layer contribution to target logit

### Status: PENDING

The layer-level tracing and attention pattern analysis need to run. These will give us:
1. Which layers are critical for each query (from trace_token)
2. Which positions are attended to at each critical layer (from attention_pattern)
3. The cross-product identifies the knowledge footprint

### Prediction

From prior work:
- ~10 positions per fact × 7 critical layers = 70 knowledge entries per fact
- 3 facts → ~210 entries (with sharing at structural positions)
- ~210 / (59 × 34) = ~10% of total KV entries carry knowledge
- Structural entries (BOS, template) serve all queries: ~5-10 positions × 7 layers = 35-70 universal entries
- Fact-specific entries: ~40-50 per fact

## Experiments 2-7 — PENDING

The remaining experiments investigate:
- **Exp 2**: V vector trajectories — what do V vectors encode at each layer for fact vs filler positions?
- **Exp 3**: K vector anatomy — do K vectors encode relation templates or answer content?
- **Exp 4**: Multi-fact interference — at what scale do facts compete for attention?
- **Exp 5**: Parametric vs novel knowledge structure — which circuit wins when weights contradict context?
- **Exp 6**: Knowledge geometry — do V vectors cluster by fact type? Do K vectors cluster by relation?
- **Exp 7**: Knowledge compression — how many PCA dimensions carry the knowledge signal?

## Next Steps

1. Run `trace_token` for all 3 queries (layer-level causal importance)
2. Run `attention_pattern` at critical layers for all 3 queries (position-level reading)
3. Cross-reference to build the knowledge map
4. Run `decode_residual` at fact positions across layers (V vector content)
5. Run `logit_attribution` for all 3 queries (component decomposition)
6. Begin V/K geometry experiments with `compare_activations` and `compute_subspace`
