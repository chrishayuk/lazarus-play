# Archived — Tonal Probe & Navigation Signal Experiments

Four experiments on `google/gemma-3-4b-it` (34 layers, 2560D) developing dark space tonal probes for document navigation and the evaluative manifold programme. Conducted March 2026.

The tonal probe work established the navigation probe programme and the multi-channel dark space hypothesis. The evaluative manifold experiment was started but not completed. The charge peak dual-circuit finding has moved to `core/residual-streams/`.

## What's Here

| File | Experiment ID | Verdict | One-line summary |
|------|--------------|---------|------------------|
| `generation_mode_tonal_probe_results.md` | cd6d04e9 | **Complete** | Generation-mode probe 87.5-100% on humor; W170 porridge found at 100% (was rank 308/725); tonal signal is 1D at L26 (84.6% PC1); two-probe query routing architecture |
| `tonal_compass_library_results.md` | 4b57b32a | **Complete** | Tonal steering vector projected onto 5,800 stored residuals; 8.4σ dynamic range; zero top-10 overlap with compass search; orthogonal dark space channel confirmed |
| `semantic_gap_generation_probe_results.md` | a92ee5a2 | **Complete** | Gap narrowed from 81.8% to 90.9% (10/11); W37 birthday humor found at 62.35%; W613 social humor is model ceiling; judgment ⊥ tonal at 91.16° |
| `evaluative_manifold_results_v2.md` | — | **Incomplete** | Ground truth windows defined for tension/surprise/social probes; training protocol specified; experiments not run |

## Key Findings

### Tonal Probes & Multi-Channel Dark Space

Three experiments progressively built the dark space navigation probe programme:

**Library projection** — The tonal steering vector (trained on 8+8 synthetic amusing/serious prompts) projected onto 5,800 stored L26 residuals produces a weak but real signal (8.4σ range). Zero top-10 overlap with compass search confirms it's an independent dark space channel. W238 (sports/news) at rank 50/725. W170 (porridge) at rank 308/725 — diluted by surrounding context.

**Generation-mode probe** — The breakthrough. Instead of reading content residuals, read the model's *judgment* residual after generating a 20-token assessment. PC1 captures 84.6% of variance (vs 23.3% for expression-mode). W170 porridge found at 100% confidence (was rank 308). TV broadcast correctly classified as routine (awe ≠ humor). Two-probe routing: query-type probe (100% accuracy) routes to tonal or grounding probe.

**Semantic gap probe** — The gap between keyword-findable and total amusing windows narrowed from 81.8% (9/11) to 90.9% (10/11). W37 (happy birthday from lunar orbit) found at 62.35% — pragmatic humor within the 4B model's evaluative reach. W613 (Collins pranking the press about sphere-of-influence) is the true ceiling — social humor requiring theory of mind beyond this model's capacity.

**The orthogonality finding:** Judgment (generation-mode) and tonal (expression-mode) signals are 91.16° apart — perfectly orthogonal. These are completely independent dark space channels using different dimensions. Generation cannot be approximated by better reading-residual extraction. The dark space contains at least two independent evaluation systems.

### Evaluative Manifold (incomplete)

Ground truth windows defined for tension (7), surprise (7), social (7), and routine (7). Training protocol specified (replay window → assessment prompt → generate 20 tokens → extract L26). Experiments not run. The hypothesis: each evaluation type has its own dark space direction, confirming a multi-dimensional evaluative manifold.

## Connections

| Finding | Feeds into |
|---------|-----------|
| Generation-mode tonal probe | Navigation probe programme, Mode 7 exploration queries |
| Two-probe query routing | Zero-hardcoding query classification at L26 |
| Judgment ⊥ tonal (91°) | Multi-channel dark space hypothesis |
| Semantic gap 81.8% → 90.9% | Ceiling for 4B model humor evaluation |
| W613 as model ceiling | Social inference requires larger models or multi-turn |
| Tonal ⊥ compass (zero top-10 overlap) | Independent dark space navigation channels |

The dual-circuit retrieval architecture (novel vs parametric) is documented in `core/residual-streams/charge_peak_results.md`.