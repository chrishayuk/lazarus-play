# Archived — Exploration Research Programme

The interpretability research programme proper. Sixteen experiments on `google/gemma-3-4b-it` (34 layers, 2560D) exploring A* navigation over the residual stream, compositional reasoning mechanisms, broad context reasoning, translation stratigraphy, the quantum inference hypothesis, and the geometry of multi-hop failures. Conducted March 2026.

This folder contains the deepest mechanistic understanding of how the model reasons about compositional queries — the A* failure taxonomy, the CoT-as-input-simplification proof, the universal branching theory, the zoom strands architecture, and the 19 dead ends. The findings are absorbed into the Lazarus architecture design philosophy and inform every engineering decision. The experiments themselves are standalone interpretability research.

## What's Here

| File | Experiment ID | One-line summary |
|------|--------------|------------------|
| `a_star_lookahead.md` | b642415b | Head 1 misfires on Hamlet→Denmark; zeroing Head 1 doubles Stratford to 12.5% (2nd); L26 FFN is the decisive wrong-write (+5.31 Denmark vs +0.63 Stratford) |
| `astar_navigator_report.md` | 2b6e5e7c | Summary: no correct-computation manifold; 2D logit lens perfectly predicts success/failure; 0.33% of residual carries the decisive signal |
| `astar_navigator_full_report.md` | 2b6e5e7c | Full 8-experiment report: trajectory atlas, heuristic evaluation, failure anatomy, navigation by reference, correction strategies, path structure, A* loop, logit lens vs full-D |
| `astar_residual_stream.md` | 4fe30d12 | 8 experiments: layer skip fails collectively (85% individual, 0% simultaneous); portal jumps save 67.6% compute; Type A/B/C failure taxonomy; CoT dominates A* on accuracy |
| `broad_context_reasoning_results.md` | e96e2860 | Geometric routing 0/11 for humor; BM25 indicators 27.3%; 256 clean tokens/window knee; query-type routing essential; 81.8% keyword ceiling with 18.2% semantic gap |
| `cot_experiments.md` | — | **CoT is input simplification, not circuit change.** Same Head 1 circuit, same concentration. Three failure mechanisms: referent competition, cultural saturation, template competition. Minimum CoT identified per failure type. |
| `dead-ends.md` | — | **19 dead ends.** Every failed approach catalogued: surprise, speaker changes, angular velocity, tension probe, two-layer cascade, expression-mode steering, evaluative manifold, inject-all, compressed replay, operational warnings, fixed threshold, L14 routing, multi-layer K, multi-head K, entity-enhanced K, contrastive K, W_K fixing, raw H-space, PCA-16 circular dependency. |
| `gemma3_translation_stratigraphy.md` | — | Six-method unified stratigraphy of EN→FR translation. Layer 2 is the single critical node. Language identity at L1 (100%) blurs to 83% in middle layers, re-consolidates at L17+. Representations lead predictions by 16 layers. Attention net -90.4 vs FFN net +63.9 for "Le". |
| `multi_hop_branching_universal.md` | 29720a4e | Chain-peak-then-overwrite generalises across 5 domains. Three failure types (late overwrite, early shortcut, missing chain). Template framing as critical as trigger removal. Branch merging outperforms both branches. Strand orthogonality predicts merge success. |
| `multi_hop_geometric_navigation.md` | 25f2177a | No hop vector exists; each hop is entity-specific. Last-position injection fails (Head 1 reads other positions). Geometric CoT works with oracle (33 layers) but is circular. Head 1 scaling ×2 fixes amplitude failures. Planner probe peaks at L0 (surface syntax). |
| `parallel_astar_writeup.md` | a2321302 | Model correctly resolves all three failures through L28 (67-88% confidence). Failures are late-layer associative memory overrides. Navigator 1 (compositional) vs Navigator 2 (associative) — both run in parallel. Self-contrast direction sep=15.75. All steering directions hallucinate on adjacent prompts. |
| `path_trace_experiments.md` | 6a86facb | Fact presence detectable at L26; fact identity only at L33 (reversal). Accumulation sub-additive (N^0.8). Path trace identity horizon ~2000-2500 tokens. Five different facts produce nearly identical L26 perturbation. |
| `quantum_inference.md` | 8bb57c7d | Brussels 79.5% at L26 degrades to 60.6% at L33; Einstein Germany 98.2% at L30 destroyed to 35.2% at L33 by single-layer override. Hard recirculation: 93.4% after one pass. Early exit at L26-L28 gives correct answer where L33 gives wrong. |
| `synthetic_to_real_results.md` | c6d790b6 | Parametric ceiling 78% (Apollo), 17% (Gemini); 0/22 hedging; entity anchors +11-17%; novelty density 5-80% by document type; 1KB sparse index matches window replay on single-fact retrieval |
| `universal_chart_experiment.md` | 52394963 | Relational address hypothesis falsified — generic chart collapses to nearest training attractor. Private charts work universally. Two-level hierarchy: entity coordinates (relation-independent) + relation charts. Cross-relation injection works (capital→language: 86.3%). Manifold is flat but private. |
| `zoom_strands_report.md` | f87eb7ba | Seven zoom levels (artifact→collapse→domain→crystallisation→discrimination→consolidation→final). Merge point is computational (L26 FFN 4.6×) not geometric. Raw/normalised paradox: Stratford has highest L33 raw projection but 5.9% normalised. Four failure signatures. Correct answer is always in the residual. |

## Research Threads

### Thread 1: A* Navigation (4 documents)

Progressive exploration of whether geometric search can replace sequential inference:

**A* lookahead** — zeroing Head 1 at L24 doubles Stratford probability. The knowledge is in the weights; the routing is wrong. Head 4 carries the correct signal.

**A* residual stream** — the definitive 8-experiment test. Layer skipping fails collectively (85% individually safe, 0% simultaneously). Portal jumps work for same-type queries (67.6% savings). Three failure types: Type A (FFN wrong association, fixable), Type B (attention entity misfiring, unfixable by surgery), Type C (format template competition, unfixable). CoT dominates on accuracy.

**A* navigator** — the full-dimensional analysis. No correct-computation manifold exists (trajectories diverge 55× by L33). Failures are geometrically invisible in full 2560D (cos=0.9936 between success and failure). One number — P(correct city at L28 logit lens) — perfectly predicts success. The dimensionality paradox: 0.33% of the residual carries the decisive signal.

**Parallel A*** — the model already resolves all three test failures correctly through L28 (67-88%). Failures are late-layer associative memory overrides, not compositional inability. Navigator 1 (compositional, L24-L30) vs Navigator 2 (associative, L31+) run in parallel. All steering directions hallucinate on adjacent prompts at their working alpha.

### Thread 2: Compositional Reasoning (3 documents)

**CoT experiments** — the mechanistic proof that CoT is input simplification, not circuit change. Same Head 1, same concentration, same layer-by-layer structure. Three independent failure mechanisms: referent competition (fixable by naming), cultural saturation (fixable by bypassing saturated entity), template competition (fixable by phrasing). CoT can make things worse (221B Baker Street: Head 1 concentration increased from 37% to 85% on wrong token).

**Multi-hop geometric navigation** — no shared hop vector exists (direction captures template type, not entity-specific hop). Last-position injection fails because Head 1 reads from all positions. Geometric CoT works but requires an oracle planner — circular dependency. Head 1 scaling ×2 fixes Type B amplitude failures. Planner probe falsified (peaks at L0, surface syntax only).

**Multi-hop branching universal** — the chain-peak-then-overwrite pattern generalises across geography, temporal, linguistic, scientific, and literary domains. L24 Head 1 reads the trigger; L26 FFN writes the shortcut. Template framing matters as much as trigger removal. Branch merging produces outputs stronger than either individual branch (Relativity: 81.1% merged > 69.9% clean > 45.7% contaminated). Strand orthogonality (<10°) predicts merge success.

### Thread 3: Broad Context & Document Navigation

**Broad context reasoning** — geometric routing fails completely for subjective queries (0/11 for humor). BM25 indicator search ((Laughter), Czar) beats all neural routers (27.3% vs 0-9%). Context density knee at 256 clean tokens/window. OCR cleaning > more raw context. Query-type routing is essential: tone/social queries work with any conversational windows, event queries require specific windows.

**Synthetic to real** — parametric ceiling measured (Apollo 78%, Gemini 17%, fictional 0%). Zero hedging across 21 answers. Entity anchors alone boost recall 11-17%. Novelty density ranges 5-80% by document type. A 1KB sparse index matches full window replay on single-fact retrieval.

### Thread 4: Deep Architecture

**Translation stratigraphy** — six-method unified analysis of EN→FR translation. Single critical layer (L2). Language identity at L1 (100%), blurs to 83% in middle layers, re-consolidates at L17+. Representations lead predictions by 16 layers. The answer emerges from the embedding (+59.25 for "Le"), is suppressed for 33 layers (attention: -90.4), then recovered by FFN (+63.9). The model "knows" the answer from context but processes through it fully before committing.

**Quantum inference** — early exit gives correct answers where L33 gives wrong (Einstein: Germany 98.2% at L30, Ulm 51.1% at L33). Hard recirculation converges in one pass (Brussels 60.6% → 93.4%). The "quantum" framing is wrong (embeddings too correlated for interference), but the observation-timing finding is real: later is not always better.

**Universal chart** — relational address hypothesis falsified (generic chart collapses to nearest training attractor). Private charts work universally (orthogonal_cosine=1.0). Two-level hierarchy: entity coordinates (~8D, relation-independent) + relation charts (map entity to specific attribute). Cross-relation injection works (Japan capital → Japan language chart → Japanese 86.3%).

**Zoom strands** — the deepest architectural finding. Seven zoom levels from artifact (L0-7) through crystallisation (L24) to final (L33). The merge point is computational (L26 FFN writes 4.6×) not geometric (token angles are constant). The raw/normalised paradox: Stratford has the highest L33 raw projection (2706) but only 5.9% normalised — layer norm's common-mode rejection biased by the wrong association's mean. Four failure signatures (wrong write, zoom ceiling, missing strand, ambiguous entity).

**Path traces** — fact presence detectable at L26 (cosine ~0.998 fact-fact vs ~0.998 fact-nofact); fact identity only at L33 (reversal: different facts become more different from each other than from no-fact). Accumulation sub-additive (N^0.8). Identity horizon ~2000-2500 tokens.

### Thread 5: Dead Ends

**19 dead ends** catalogued. The three that hurt most: L14 entity compass (peak separation but fires only for known entities), entity-enhanced K-vectors (73.87° separation lost in W_K projection), PCA-16 routing (best geometry found, killed by circular dependency). Every dead end looked promising before the experiment.

## Key Findings That Shaped the Architecture

| Finding | Architectural consequence |
|---------|------------------------|
| No correct-computation manifold | Can't navigate toward "correct" — must detect and fix specific failures |
| 0.33% of residual is decisive | Subspace injection is the right granularity for corrections |
| CoT = input simplification | The model's circuit works fine — the input is the problem |
| Three failure types (A/B/C) | Different interventions for different failures; no universal fix |
| Head 1 has one shot | Multi-hop requires external chain resolution (CoT or injection) |
| L26 FFN is the discrimination gate | Both the source of correct answers (Canberra) and wrong answers (Denmark) |
| Late-layer override (Navigator 2) | Compositional answers are correct at L28 but destroyed at L31-L33 |
| Early exit gives correct answers | The optimal observation layer is not always L33 |
| Keyword routing beats geometric routing | For subjective/tonal queries, surface markers outperform residual geometry |
| 19 dead ends | Extensive negative results preventing re-exploration of failed paths |
| Template framing critical for branching | Not just trigger removal — the clean template's own prior must align |
| Strand orthogonality predicts merge success | <10° → merge works; >25° → collapse |
| Two-level entity/relation hierarchy | Entity coordinates are relation-independent at L26 |
| Representations lead predictions by 16 layers | The model "knows" before it "says" |

## Connections to Core

| Finding | Core document |
|---------|--------------|
| L26 FFN as discrimination gate | `core/attention/kv_anatomy_results.md` |
| Head 1 contextual attribute bridge | `core/residual-streams/markov-7-percent-report.md` |
| 0.33% answer subspace | `core/residual-streams/subspace_independence.md` |
| Layer norm as common-mode rejection | `core/trajectory_geometry_writeup.md` |
| Dual-circuit retrieval | `core/residual-streams/charge_peak_results.md` |
| L14 Markov threshold | `core/residual-streams/markov-bandwidth-report.md` |
| Parametric recall baseline | `core/parametric_recall_and_novelty_density_results.md` |