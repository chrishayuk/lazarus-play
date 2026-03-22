# Archived — Attention Architecture Experiments

Mechanistic interpretability experiments on `google/gemma-3-4b-it` (34 layers, 2560D, 8 heads) investigating attention's role during generation, copy head identification, and alternative generation strategies. Conducted March 2026 as part of the Lazarus project. All actionable findings have been absorbed into the working architecture.

## What's Here

| File | Experiment ID | One-line summary |
|------|--------------|------------------|
| `attention_free_inference_results.md` | 6759d6ff | L25-L33 attention safely removable; FFN provides 71-80% of logit; L0 irreplaceable |
| `copy_head_calibration_results.md` | 71541b5d | DLA + ablation combined method finds L29 H4 as true copy head; validated on 4B and 1B |
| `copy_head_calibration_ablation_matrix.md` | — | Per-head zeroing across 6 probes; L29 H4 confirmed at 4/6 consistency |
| `copy_head_calibration_dla_matrix.md` | — | Per-head Direct Logit Attribution across L27-L32, 6 probes |
| `l29_h4_fact_detection_results.md` | — | L29 H4 does NOT flag facts during raw prefill — killed the salience-head hypothesis |
| `latent_chain_generation_results.md` | 0c83facb | Latent chaining is architecturally impossible; L33 discontinuous, KV cache is the information carrier |

## Key Findings (absorbed into Lazarus)

**Copy head architecture** — L29 H4 is the primary copy head for novel/rare tokens. H3 handles common tokens, H5 handles common verbs. Injection at L30 (copy head + 1). The 85-88% positional scaling law generalises to 1B (L23 H1). The calibration algorithm (DLA top-5 → ablation consistency → injection validation) is the reference method for porting to new models.

**Attention budget** — L25-L33 attention is BOS-dominated (65-79%) and safely zeroed during generation. 26% attention compute savings, ~13% total. The cliff is at L22-L25. 28/34 layers individually safe but only 9/34 simultaneously safe (composability failure).

**Attention horizon** — Token 1 needs 6 layers; token 7+ needs 2 (L0, L4); token 15+ likely L0 only. Not composable without distillation.

**L0 is irreplaceable** — 72-81% prompt attention at every generation step. Creates the entity signal from scratch via KV cache read, not by reinforcing a degraded signal.

**Latent chaining is impossible** — L33 residuals are nearly orthogonal across consecutive steps (cos 0.109-0.413). Prediction state ≠ processed-token state. KV cache is the cross-position information carrier. This justified the pivot to checkpoint-chained prefill.

**L29 H4 is a retrieval head, not a salience head** — Only activates when a query drives retrieval. During raw prefill, attention goes to BOS (42-86%) and local context. Justified the pivot to BM25 routing for indexing.

## Related Core Documents

The following experiments from this research programme are in the core documentation (not archived):

- `bigger_map_results.md` — residual = query channel, KV = map, 27× compression
- `no_context_length_results.md` — unlimited context proof at 827K tokens, KL=0.0
- `reconstruction_at_scale_results.md` — L14 compass at 0.997 with 725 windows, three ceilings
- `trajectory_geometry_writeup.md` — compositional failures in 2-7D, subspace correction principle