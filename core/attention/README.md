# Core — Attention Circuit Architecture

How the retrieval circuit works end-to-end. Four documents on `google/gemma-3-4b-it` (34 layers, 2560D, 8 heads) covering the complete three-phase retrieval circuit, the frequency-stratified copy heads, and the canonical injection architecture specification.

The charge peak experiment in `core/residual-streams/` establishes *that* L29 H4 is the primary copy head and that novel/parametric retrieval uses different circuits. These documents explain *how* the circuit is organised, *which heads handle which tokens*, and *what the production architecture looks like*.

## What's Here

| File | One-line summary |
|------|------------------|
| `kv_anatomy_results.md` | **The circuit map.** Three-phase retrieval: query processing (L0→L4→L9→L10→L14→L15 dark routing), fact copying (L23→L29 H4→L30), amplification (L33 FFN). 7/34 layers critical, 6 anti-critical. Novel vs parametric double dissociation. Dark routing discovery. |
| `v_injection_architecture.md` | **The canonical spec.** 536 bytes/fact (12-byte answer + 512-byte K + 12-byte entity string). Entity string filter + adaptive Q·K routing. 85% injection rate at N=3,625. 330ms fast path. The production blueprint. |
| `token_freq_dla_matrix.md` | **The copy heads.** 30-probe × 8-head DLA matrix across 5 frequency bands. H4 for rare (ID >7K, DLA 1.14-1.39), H5 for common (2-7K, DLA 1.08), H3/H0 for very common (<2K, DLA 0.72/0.26). Sharp crossover at ~7K and ~2K. |
| `token_freq_ablation_matrix.md` | **The causal proof.** H4 ablation -24.7pp obsidian but +12.1pp gold. H5 ablation -45.2pp gold but 0pp obsidian. Non-linear cooperation: individual -33pp, collective -60pp (45% cooperative signal invisible to single-head analysis). |

## The Three-Phase Retrieval Circuit (kv_anatomy)

The complete circuit for novel in-context fact retrieval:

**Phase 1 — Query Processing (L0→L4→L9→L10→L14→L15):** Dark routing layers read QUERY positions from KV to build a retrieval key. They contribute near-zero DLA but are the most critical layers (KL 3.6-8.1, higher than L29's 3.7). L10 H1 reads "city" at 44.7%. L14 H7 reads "of" at 68.8%. They match the query template against relation context at the fact position.

**Phase 2 — Fact Copying (L23→L29 H4→L30):** One head (L29 H4) puts 62% attention on one position (pos 14 "Volt") and provides 41% of the output logit (+17.09 DLA). The fact signal jumps from rank 19,344 to #1 at L29 — a discrete copy operation, not gradual accumulation.

**Phase 3 — Amplification (L33 FFN):** Reads "Volt" from the residual stream (NOT from KV) and amplifies +14.0 DLA (34% of final logit).

**The seven critical layers:** L0 (context reader), L4 (relation template), L9 (query structure), L10 (query relation word), L14 (query key refinement), L15 (retrieval key finalisation), L29 (fact copy). Removing any one destroys retrieval.

**The six anti-critical layers:** L1, L2, L3, L5, L7, L16. Removing them *improves* P(answer) to 100%. They introduce noise that degrades the retrieval signal.

**Novel vs parametric double dissociation:** Novel facts use attention (L29 H4 +17.09). Parametric facts use FFN (L25 +12.4). L9/L10 act as a pathway switch — they suppress parametric while enabling KV retrieval. L33 FFN amplifies both. Completely different circuits, same injection site.

## The Production Architecture (v_injection_architecture)

The canonical specification for the knowledge store:

```
Per fact: 536 bytes
  answer_token_id  :   4 bytes    (which token to inject)
  projection_coeff :   8 bytes    (injection magnitude)
  k_vector         : 512 bytes    (256D float16, L29 KV-head-2)
  entity_string    :  ~12 bytes   (entity name for string matching)

Routing: two-stage
  Stage 1: Entity string match (~75-83% of queries, 0% false positives)
  Stage 2: Adaptive Q·K threshold (mean×2, auto-scales with N)
  Stage 3: Replay fallback (~15% of queries)

Injection: 1D at L30
  h += coefficient × embed(answer_token) / ||embed||²

Performance at N=3,625:
  Injection rate: ~85%
  Fast path latency: 330ms
  Replay latency: 2,000ms
  Average: ~505ms
  Wrong injection rate: ~0%
```

## The Frequency-Stratified Copy Heads (token_freq_dla_matrix + ablation_matrix)

| Frequency Band | Token ID Range | Dominant Head | Mean DLA | Examples |
|----------------|---------------|---------------|----------|---------|
| Very rare | >120K | **H4** | 1.39 | obsidian, Mendez, Brix, Thane, Rune, fjord |
| Rare | 60-120K | **H4** | 1.39 | Lark, Birch, Ember, basalt, Volt, Flint |
| Medium | 7-60K | **H4** | 1.14 | Portland, mercury, velvet, cobalt, architect |
| Common | 2-7K | **H5** | 1.08 | sell, build, teach, write, drive, plant |
| Very common | <2K | **H3** (+ H0) | 0.72 (0.26) | the, is, have, make, go, take |

The crossovers are sharp:
- **H4 → H5** at ~7K token ID: H4 goes from +1.14 to -0.05; H5 goes from -0.01 to +1.08
- **H5 → H3** at ~2K token ID: H5 drops out; H3 rises from 0.08 to 0.72; H0 emerges at 0.26

## Mutual Suppression

The heads aren't just specialised — they actively suppress each other's frequency bands:

- **H4 zeroed:** Obsidian drops -24.7pp (rare token loses its copy head). Gold *rises* +12.1pp (common token freed from H4's suppression).
- **H5 zeroed:** Gold drops -45.2pp (common token loses its copy head). Obsidian unchanged (H5 has no effect on rare tokens).

H4 is actively pushing common-token probability *down* while copying rare tokens. The copy circuit is competitive, not just parallel.

## Non-Linear Cooperation

Individual head ablations for obsidian sum to -33pp. Zeroing all attention drops it -60pp. The 27pp gap (45% of total effect) is cooperative signal — heads reinforcing each other in ways invisible to single-head ablation. The copy circuit has massive redundancy that only appears in collective ablation.

## Key Numbers

| Measurement | Value |
|-------------|-------|
| L29 H4 DLA | +17.09 (41.4% of final logit) |
| L33 FFN DLA | +14.00 (33.9% of final logit) |
| Critical layers | 7 of 34 (L0, L4, L9, L10, L14, L15, L29) |
| Anti-critical layers | 6 (L1, L2, L3, L5, L7, L16 — removing improves output) |
| L15 ablation KL | 8.1 (most critical dark routing layer) |
| H4 rare DLA (mean, 18 probes) | 1.14-1.39 |
| H5 common DLA (mean, 6 probes) | 1.08 |
| H3 very common DLA (mean, 6 probes) | 0.72 |
| H4→H5 crossover | ~7K token ID |
| H5→H3 crossover | ~2K token ID |
| H4 ablation (obsidian, rare) | -24.7pp |
| H5 ablation (gold, common) | -45.2pp |
| H4 ablation (gold, common) | +12.1pp (mutual suppression) |
| Non-linear cooperation | 45% of total effect (individual -33pp, collective -60pp) |
| Storage per fact | 536 bytes (12 answer + 512 K + 12 entity) |
| Injection rate at N=3,625 | ~85% (hybrid routing) |
| Fast path latency | 330ms (vs 2,000ms replay) |

## Why This Matters for Lazarus

| Principle | Architectural consequence |
|-----------|------------------------|
| Three-phase retrieval | Query processing and fact copying are separate circuits — dark routing builds the key, L29 H4 uses it |
| 7 critical layers, 6 anti-critical | Most layers are irrelevant or harmful to retrieval — the circuit is sparse |
| Dark routing (L9-L15) | Near-zero DLA but most critical — invisible to logit attribution, only visible via ablation |
| Novel vs parametric pathway switch | L9/L10 suppress parametric while enabling KV retrieval — the model has a built-in source selector |
| Frequency-stratified copy heads | Knowledge store injection must account for which head handles the target token |
| H4 for novel entities | Novel fictional names (Voltara, Zarkov) route through H4 — the primary injection path |
| H5 for common verbs | Common-word facts (sell, build) use a different copy path |
| Mutual suppression | H4 actively pushes against common tokens — the copy circuit is competitive |
| 536 bytes/fact, 85% injection | The production architecture: entity string filter + adaptive Q·K + replay fallback |
| L30 injection (not L29) | L29 FFN is anti-correlated with factual signal; L30 cleanup heads pre-amplify |

## Related Documents

- `core/residual-streams/charge_peak_results.md` — the dual-circuit discovery (novel = attention, parametric = FFN)
- `core/residual-streams/markov-writeup.md` — Markov property that enables injection
- `core/residual-streams/subspace_independence.md` — local coordinate frames explaining why 1D injection works
- `core/knowledge_map_needs_results.md` / `knowledge_the_map_needs_results.md` — 1D per fact, geometric hash
- `archive/attention-architecture/copy_head_calibration_results.md` — the DLA + ablation method that found L29 H4
- `archive/injection/v_injection_results.md` — the original V-injection proof of concept
- `archive/routing/token_freq_probe_set.md` / `token_frequency_probe_set.md` — the 30-probe sets