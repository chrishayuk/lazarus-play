# Archived — Routing Research Programme

The complete routing research programme for the Lazarus knowledge store on `google/gemma-3-4b-it` (34 layers, 2560D). Seventeen experiments spanning dark space geometric routing, event density detection, fact marker detection, K-space addressing, H-space routing, format matching, vector routing with final residual, two-layer routing, temporal routing for global queries, token frequency copy circuit characterisation, and the final architecture benchmark. Conducted March 2026.

The hybrid routing architecture (entity string filter + adaptive K-space Q·K + BM25 keyword fallback) and the final benchmark results (15/15 with combined routing) are absorbed into the production system. This folder is the evidence base.

## What's Here

| File | Experiment ID | Verdict | One-line summary |
|------|--------------|---------|------------------|
| `dark_space_geometric_routing_results.md` | 615dc1c1 | **Negative** | Pure geometric routing in L26 dark space does not bridge query-to-window format gap; dark space = GPS not topic index |
| `event_density_timeline_results.md` | 40f79c3a | **Partial** | Regex phrase matching works (4/5); surprise, speaker changes, and tension probe all fail as event detectors |
| `fact_markers_results.md` | 218ab701 | **Negative** | Model does not mark facts during prefill; 6/7 methods negative; local outlier at L29 is semantic novelty not fact-ness |
| `addressing_precision_results.md` | bbbfad93 | **Complete** | Entity identity 73.87° in H-space but W_K collapses it; hybrid entity string filter raises injection 50%→85%; L14 only works for known entities |
| `hidden_space_routing.md` | 25a66189 | **Complete** | Mean-centred H-space 75% (9/12); PCA-16 at 32 bytes/fact; dim 443 spike dominates raw space; format matching is hard prerequisite |
| `format_matched_pca16_results.md` | f07f5d15 | **Negative** | Single-fact templates can't bridge content gap (14-20° F-Q angle); full-document context works but defeats V-injection; K+entity filter remains correct |
| `vector_routing_final_residual_results.md` | 869f60eb | **Negative** | Final residual is NOT the missing routing variable; text context overwhelms query; injection at L0/L7/L14 negligible or harmful |
| `two_layer_routing_results.md` | 8fa47623 | **Partial** | L26 and L29 are redundant not complementary; 3D K-vector content-type subspace at 100%; routing is distributed L24-L28 not two-layer |
| `routing_scale_results.md` | d00d898e | **Complete** | N=12 stress test: same-type interference stable; token distinctiveness is binding constraint; same-template cross-domain collision (Namath→donate); 15% threshold catches all failures |
| `temporal_routing_results.md` | d74ab181 | **Complete** | Temporal stride beats compass for global queries (3.5/5 vs 2/5); angular velocity phase detection negative; 15-window sweet spot; model misinterprets operational language |
| `final_architecture_experiments.md` | 60136e50+ | **Complete** | 15/15 benchmark: injection at L10 (5/5 factual), dark probe+tool (5/5 entity-math), CoT (14/15 universal); L10 universal convergence point; Ulm two-step confabulation |
| `token_freq_probe_set.md` | — | — | 30-probe set across 5 frequency bands (code word template) |
| `token_frequency_probe_set.md` | — | — | 30-probe set across 5 frequency bands (sacred material template) |
| `vector_routing_prompts.json` | — | — | Query prompts for vector routing experiments |
| `vector_routing_windows.json` | — | — | Window data for vector routing experiments |

## Research Threads

### Thread 0: Failed Routing Hypotheses

Three early experiments testing whether model internals could replace keyword routing. All returned negative or limited results, justifying the BM25 approach.

**Dark space geometric routing** — The L26 dark space is the model's GPS (navigational state), not its topic index. "Asking about sports" and "reading sports content" occupy completely different dark coordinates. Full-space cosine is 1/4 correct. Direction dot products work but require knowing which token to project onto — keyword matching with extra steps.

**Event density signals** — Surprise and speaker changes are *anti-correlated* with events in the Apollo 11 transcript. Events use predictable operational language. The tension probe fails at all layers (val accuracy below chance). Regex phrase matching with phase constraints is the only reliable automated signal (12/15 recall, zero false positives, zero forward passes).

**Fact markers in the residual** — The model doesn't pre-index facts during prefill. 6/7 methods negative. The closest signal is semantic novelty (local outlier at L29), which partially overlaps with fact-bearing status but doesn't cleanly separate it. The retrieval circuit (L29 H4) is purely query-driven.

### Thread 1: K-Space and H-Space Routing

Four experiments progressively characterised the routing geometry:

**Addressing precision** established the core finding: entity identity is well-separated in 2560D H-space (Namath vs Marchand at 73.87°) but W_K collapses it to coin-flip territory in 256D K-space. L14 entity compass fires only for training-data entities — useless for novel facts. The fix: hybrid entity string filter (+20 bytes/fact) raises injection from 50% to 85%.

**Hidden space routing** showed mean-centred H-space achieves 75% accuracy (9/12) and PCA-16 resolves Namath at 32 bytes/fact. But a dominant spike (dim 443, magnitude 42-65K) makes raw H-space useless. Format matching between fact and query vectors is a hard prerequisite.

**Format-matched PCA-16** conclusively showed single-fact templates cannot bridge the content gap — the hidden state encodes what the model has *read*, not just prompt structure. Generic padding is worse than no padding. Full-document context works (100%) but requires replay, defeating V-injection.

**Vector routing with final residual** killed the last alternative to keyword routing — adding document context to queries either overwhelms the query signal, has negligible effect, or actively interferes.

### Thread 2: Routing Architecture

**Two-layer routing** showed L26 and L29 find the same targets for factual queries (redundant, not complementary) and both fail for tonal queries. The K-vector 3D content-type subspace achieves 100% classification between interesting and routine content. Routing is distributed across L24-L28, not isolatable to two layers.

**Routing at scale** (N=12 stress test) found same-type interference is stable (adding city facts doesn't degrade existing routing) but same-template cross-domain collision is the real failure mode (Namath→donate at 0.88×). Token distinctiveness (≥4 char first token) is the binding constraint. The 15% safety threshold catches all failures.

### Thread 3: Temporal and Global Routing

**Temporal routing** fixed global reasoning queries: timeline summary improved from 2/5 (compass) to 3.5/5 (phase-anchored or compass+temporal). Angular velocity phase detection is negative (5.29° ± 0.45%, too uniform). 15-window sweet spot at ~85 tokens/window. New failure mode: the 4B model misinterprets operational language ("ABORT STAGE RESET" → thinks crew aborted).

### Thread 4: Token Frequency Copy Circuit

The DLA matrix and ablation matrix have moved to `core/residual-streams/` as foundational findings about the copy circuit architecture. The probe sets (30 probes across 5 frequency bands in two template formats) remain here as supporting data.

### Thread 5: The Final Architecture Benchmark

**15 queries × 6 methods.** The definitive scorecard:

| Method | Accuracy | Compute |
|--------|----------|---------|
| Standard inference | 5/15 (33%) | 34 layers |
| All-position injection at L10 | 9/15 (60%) | 68 layers |
| L26 steering | 6/15 (40%) | 34 layers |
| Dark agent loop | 13/15 (87%) | 41 layers + tool |
| CoT | 14/15 (93%) | ~300 layers |
| **Combined routing** | **15/15 (100%)** | **~90 layers avg** |

L10 is the universal convergence point for factual multi-hop (all 6 pairs minimum at L10-L14). The Ulm confabulation is confirmed as a two-step process (L31-32 preparation + L33 detonation). B-injection and CoT are complementary — CoT fails exactly Q3 (Beethoven capital), injection succeeds exactly Q3.

## Key Architecture Decisions (absorbed)

| Decision | Evidence | Status |
|----------|---------|--------|
| Entity string filter for routing | Addressing precision: 50%→85% injection | Production |
| Adaptive Q·K threshold (mean×2) | Routing scale: fixed 15% fails at N>50 | Production |
| L29 H4 for novel, H5 for common tokens | Token freq DLA + ablation | Absorbed into copy head config |
| BM25 for tonal/global queries | Two-layer routing: attention can't route tonal | Production |
| Temporal stride for timeline queries | Temporal routing: 3.5/5 vs 2/5 compass | Production |
| L10 injection for factual multi-hop | Final architecture: 5/5, KL=0.0 | Production |
| Dark probe at L7 for entity-math | Final architecture: 5/5, 41 layers | Production |
| 15-window budget for global queries | Temporal routing: knee at 10-15 | Production |

## Connections to Core

| Finding | Core document |
|---------|--------------|
| W_K collapses entity separation | `charge_peak_results.md` — dual-circuit architecture |
| L10 convergence point | `markov-writeup.md` — Markov property enables injection |
| Token frequency copy heads | `copy_head_calibration_results.md` (archive/attention-architecture) |
| Format gap is content-driven | `subspace_independence.md` — local coordinate frames |
| Temporal vs compass routing | `reconstruction_at_scale_results.md` — routing quality as bottleneck |