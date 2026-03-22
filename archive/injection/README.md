# Archived — Injection Research Programme

The complete injection research programme for the Lazarus knowledge store on `google/gemma-3-4b-it` (34 layers, 2560D). Fourteen experiments spanning V-injection proof-of-concept, KV anatomy, synthetic KV injection, direct KV injection, compass-routed injection, replay caching, chained fact KV, subspace surgery, K-norm sampling, Apollo routing diagnosis, and the canonical architecture specification. Conducted March 2026.

The V-injection architecture spec and the KV anatomy circuit map have moved to `core/attention/`. The remaining twelve experiments document the proof-of-concept, alternative approaches tested and eliminated, and supporting evidence.

## What's Here

| File | Experiment ID | Verdict | One-line summary |
|------|--------------|---------|------------------|
| `v_injection_results.md` | bab74728 | **Complete** | V-injection proof: KL≈0 at L29; H4 copies at 50.78%; 1-fact to 10-fact scaling clean; cross-entity contamination = max-confidence wrong answer; 11,000:1 compression |
| `kv_knowledge_results.md` | d3bf96ea | **Incomplete** | Token map, baseline retrieval, knowledge fraction plan. Experiments 2-7 pending. |
| `synthetic_kv_injection_results.md` | 2ece587e | **Negative** | Attention routing works perfectly (58:1 discrimination) but answer retrieval fails; copy head is amplifier not standalone retriever; single-layer KV injection dead end |
| `direct_kv_injection_results.md` | d651412a | **Complete** | All-position injection KL=0.0 at every layer; last-position fails completely; facts are multi-positional; 5/5 novel facts correct; same retrieval circuit fires for injection/keywords/replay |
| `compass_routed_injection_results.md` | 44648ab3 | **Negative** | 12-byte injection negligible (0.02-0.3% of residual); compass routing 2/5; parametric override; only contrastive steering at α=5 injects specific facts; replay is the only working delivery |
| `chained_fact_kv_results.md` | c7b7e88f | **Complete** | Chain adds ~3-4% magnitude not direction; standalone KV failed due to RoPE mismatch not missing context; pre-RoPE K+V is correct storage format; keyword index wins on every practical dimension |
| `replay_cached_injection_results.md` | dd304510 | **Negative** | Bare query vs replay hidden state: 152° after centering (opposite hemisphere); document/no-document wall is fundamental; geometric cache dead |
| `subspace_surgery_dark_table.md` | 5eeee327 | **Mixed** | Surgery clean (orthogonal_cosine=1.0 always) but Stage 3a contamination is in the orthogonal complement by definition; works for same-template entity swapping; scale law: inject at same layer as table |
| `apollo_routing_diagnosis_results.md` | 12b05e7b | **Complete** | Zarkov probe ranks #1/1,473 (mechanism works); interval sampling captures punctuation not content; H4 is positional copy head not semantic retriever; chat template essential for routing |
| `k_norm_sampling_validation_results.md` | 2a36b216 | **Partial** | K-norm sampling 12.4× better than interval; equalized-position routing 4/8 heads correct; parametric override is binding constraint (confabulates even with correct context) |

## Research Timeline

### Phase 1: Proof of Concept

**V-injection** proved the mechanism: inject the L29 last-position residual from a donor (who has the fact in context) into a bare query. KL≈0 at L29+. Scales from 1 to 10 facts. Cross-entity contamination delivers wrong answer at maximum confidence — routing must be correct before injection.

**KV anatomy** mapped the complete retrieval circuit: query processing (L0→L4→L9-L15 dark routing) → fact copying (L23→L29 H4→L30) → amplification (L33 FFN). Discovered dark routing layers (L9, L10, L14, L15) that have near-zero DLA but are the most critical layers. Established the novel vs parametric double dissociation.

### Phase 2: Compression and Architecture

The architecture spec compressed storage from 5,120 bytes (full residual) to 536 bytes/fact (12-byte answer token + coefficient, 512-byte K-vector, 12-byte entity string). Hybrid routing: entity string filter (75-83% coverage, 0% false positives) + adaptive Q·K threshold (auto-scales with N) + replay fallback (~15%).

### Phase 3: Alternative Approaches (all failed or limited)

**Synthetic KV injection** — Inject stored K/V entries at L29 to let H4 find the answer via its own attention. Attention routing works perfectly (58:1 discrimination) but answer retrieval fails. The copy head amplifies a signal from L0-L28 that single-layer injection can't provide.

**Direct KV injection** — All-position injection KL=0.0 but is full state transplant (cached replay), not selective KV concatenation. True KV injection requires MLX-level engineering.

**Compass-routed injection** — 12-byte injection is negligible against parametric confidence (99.99%). Compass routing 2/5 (format gap). Only contrastive steering at α=5 injects specific facts, but requires knowing the answer. Replay is the only working delivery for real Q&A.

**Chained fact KV** — Chain adds ~3-4% to dark space magnitude at fact positions. Standalone KV failed due to RoPE mismatch, not missing context. Pre-RoPE K+V is the correct format.

**Replay caching** — Bare query vs replay hidden state: 152° after centering. Document/no-document wall is fundamental at L29.

**K-norm sampling** — 12.4× better content capture than interval sampling. But parametric override is the binding constraint — model confabulates even with correct context for well-known topics.

### Phase 4: Architecture Convergence

All alternative paths converge on the same conclusion: **keyword routing + context window replay is the correct architecture for real document Q&A.** The 12-byte injection works for novel entities with template-matched queries but cannot override parametric knowledge or bridge the format gap for natural language questions.

## Key Architecture Decisions (absorbed)

| Decision | Evidence | Status |
|----------|---------|--------|
| 12-byte injection (token ID + coefficient at L30) | V-injection: KL≈0 for novel facts | Production (knowledge store) |
| Entity string filter for routing | Addressing precision: 50%→85% injection | Production |
| Adaptive Q·K threshold (mean×2) | Routing scale: fixed 15% fails at N>50 | Production |
| L30 injection layer (not L29) | V-injection: L29 FFN anti-correlated; L30 cleanup heads pre-amplify | Production |
| Replay fallback for entity-implicit queries | Compass-routed injection: 2/5; replay: 5/5 | Production |
| Pre-RoPE K storage format | Chained fact KV: RoPE mismatch is root cause of standalone failure | Reference |
| Keyword index over KV manipulation | Chained fact KV: 800 bytes vs 28-482 MB, same accuracy | Production |

## Key Numbers

| Measurement | Value | Source |
|-------------|-------|--------|
| V-injection KL | ≈0 (0.000285 single-fact) | v_injection_results |
| H4 attention to fact position | 50.78-61.7% | v_injection + kv_anatomy |
| L29 H4 DLA | +17.09 (41.4% of final logit) | kv_anatomy |
| L33 FFN DLA | +14.00 (33.9% of final logit) | kv_anatomy |
| Critical layers | 7 of 34 (L0, L4, L9, L10, L14, L15, L29) | kv_anatomy |
| Anti-critical layers | 6 (L1, L2, L3, L5, L7, L16) | kv_anatomy |
| Compression ratio | 11,000:1 (5KB vs 56GB) | v_injection |
| Storage per fact | 536 bytes | v_injection_architecture |
| Injection rate at N=3,625 | ~85% (hybrid routing) | v_injection_architecture |
| Synthetic KV discrimination | 58:1 (routing works, retrieval doesn't) | synthetic_kv_injection |
| Document/no-document wall | 152° after centering | replay_cached_injection |
| K-norm vs interval sampling | 12.4× better content capture | k_norm_sampling |
| Chain contribution to fact KV | ~3-4% magnitude, same direction | chained_fact_kv |

## Connections to Core

| Finding | Core document |
|---------|--------------|
| L29 sign flip = retrieval event | `core/residual-streams/charge_peak_results.md` |
| Dual-circuit (novel vs parametric) | `core/residual-streams/charge_peak_results.md` |
| H4 for rare tokens, H5 for common | `core/attention/token_freq_dla_matrix.md` |
| Markov property enables injection | `core/residual-streams/markov-writeup.md` |
| L14 as entity compass | `core/residual-streams/markov-bandwidth-report.md` |
| 8D per fact, local coordinate frames | `core/residual-streams/subspace_independence.md` |
| 1D per novel fact | `core/knowledge_map_needs_results.md` + `knowledge_the_map_needs_results.md` |