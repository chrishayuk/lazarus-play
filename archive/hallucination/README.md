# Archived — Hallucination & Parametric Override Experiments

Deep geometric analysis of how the model resolves competing factual signals, weight-level safety interventions, and 1D injection for correcting parametric misconceptions. Conducted March 2026 on `google/gemma-3-4b-it` (34 layers, 2560D). These are standalone interpretability results — publishable work but not feeding into the current Lazarus engineering.

## What's Here

| File | Experiment ID | Verdict | One-line summary |
|------|--------------|---------|------------------|
| `geometry_sydney_canberra.md` | 5fd26ee8 | **Complete** | Sydney leads raw projection at all 34 layers; the flip to Canberra is a layer norm common-mode rejection effect; <0.6% of residual is "about cities" |
| `geometry_sydney_canberra.py` | — | — | Visualization source (4-panel: trajectory, competition ratio, vocab space compass, direction angles) |
| `geometry_sydney_canberra.png` | — | — | Rendered visualization |
| `geometry_sydney_canberra.pdf` | — | — | Rendered visualization (vector) |
| `hallucination_mechanisms.md` | — | — | Broader hallucination circuit analysis |
| `no_go_zones.md` | — | **Mixed** | Weight surgery works for bottleneck circuits (L26 FFN), fails for distributed circuits (L20-24 induction heads); 5× repetition attack bypasses fact store entirely |
| `parametric_override_results.md` | a25bd50a | **Partial** | 1D injection overrides 57% misconceptions (Frankenstein: 83.6%) but fails on 97% template-locked priors (goldfish); same 12-byte mechanism as novel fact injection |

## Key Findings

### The Layer Norm Flip (Sydney → Canberra)

Sydney has the highest raw dot product with the residual at every layer from L4 to L33 — including the final output layer where the model predicts Canberra. There is no crossing point in raw geometry. The "flip" is entirely mediated by layer norm, which subtracts the residual mean (dominated by the "famous Australian city" shared signal). Sydney's advantage is correlated with this mean; Canberra's signal at 85° away is orthogonal to it and survives centering. L26 FFN is causally necessary because it injects just enough Canberra-direction energy to survive the centering.

This is the same mechanism documented in the trajectory geometry core doc (Beethoven/Berlin), generalised: **basin boundaries exist only in normalised space, not raw geometry.**

### No-Go Zones: Circuit Topology Determines Surgical Viability

Bottleneck circuits (L26 FFN for capital facts) are surgically modifiable — one ablation, one flip, minimal collateral. Distributed circuits (L20-24 induction heads for repetition attacks) are not — removing the full band suppresses the attack but destroys legitimate context reading (France→Paris, Germany→Berlin all break). The model's natural robustness already handles 4/5 attack variants; only 5× exact repetition breaks through.

### Parametric Override via 1D Injection

The same 12-byte injection mechanism (token ID + coefficient at L30) that delivers novel facts can correct parametric misconceptions — but only when a same-suffix donor exists that shifts the model's prediction. Moderate priors (Frankenstein " monster" at 57%) are overridable to 83.6%. Template-locked priors (goldfish " seconds" at 97%) are immune. The injection targets a specified direction, not the donor's top-1: a 12% secondary belief in "scientist" amplifies to 75% at inference.

Four laws established: same-suffix requirement is strict, template-lock is the primary failure mode, moderate priors are overridable with viable donors, and the injection targets specified directions independently of the donor's argmax.

## Connections

| Finding | Related to |
|---------|-----------|
| Layer norm as common-mode rejection | `trajectory_geometry_writeup.md` (core) — same mechanism for Beethoven/Berlin |
| 1D parametric override at L30 | Knowledge store injection — same 12-byte format |
| <0.6% of residual encodes city decision | `bigger_map_results.md` (core) — entity signal is 8-21 dims of 2560 |
| Bottleneck vs distributed circuit topology | `attention_free_inference_results.md` (archive) — composability failure |
| L26 FFN as capital fact store | Copy head calibration (archive) — L29 H4 retrieves, L26 FFN stores parametric facts |