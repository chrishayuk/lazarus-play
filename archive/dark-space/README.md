# Archived — Dark Space Experiments

Three experiments on `google/gemma-3-4b-it` (34 layers, 2560D) investigating superposition dynamics during compositional reasoning and anomaly detection in the dark space compass. Conducted March 2026. The superposition and branching work is standalone interpretability research — potentially publishable but not currently feeding into Lazarus engineering. The needle-in-haystack result justified the pivot to perplexity-based routing.

The Markov property proofs and subspace independence theory that originated in this research programme have moved to `core/residual-streams/`.

## What's Here

| File | Experiment ID | Verdict | One-line summary |
|------|--------------|---------|------------------|
| `needle_in_dark_haystack_results.md` | 412cfe21 | **Negative** | L26 compass cannot find a modern sentence in Shakespeare (rank 2527/3007); 67% of signal in PC18+; perplexity is the correct detection mechanism |
| `non_collapsing_superposition.md` | e70a5b16 | **Research** | Three-battle oscillation map for Brussels/Paris; Einstein Ulm eruption from rank 500+ to 51.2% in 3 layers; recirculation doesn't help because interference is prompt-driven |
| `parallel-superposition.md` | 75378d84 | **Research** | Branching at compositional peak into parallel force fields; clean template recovers Brussels 79.6%; Ulm bypass at 98.3%; max-confidence collapse; 53 layers vs 300+ CoT |

## Key Findings

### Needle in Dark Haystack

The L26 compass was tested against Shakespeare's complete works (3,007 windows of 512 tokens) with an 18-token modern sentence ("The astronaut landed on the moon...") planted in Henry IV Part 2. The compass ranks the needle at 2527/3007 — worse than random. At L26, 97.8% of the representation comes from Shakespeare context. The needle is at the 5.8th percentile of distance from corpus mean (closer than average, not further).

The energy decomposition explains why: 67% of the needle's distinctive signal lives in PC18+, outside the 16D compass projection. The compass sees only 9.9% of what makes the needle different. Shakespeare's natural variation (French scenes in Henry V, songs, register shifts) creates bigger anomalies than a single modern sentence.

Perplexity (per-token surprise) is the correct signal: "astronaut" after 313 tokens of Henry IV is a massive prediction-rank outlier requiring zero additional computation (captured during the existing forward pass).

### Superposition Dynamics

The non-collapsing superposition experiment maps the full layer-by-layer trajectory for two compositional failures:

**France-north** has three distinct battles in one forward pass: L25 Paris surge (90.2%) from the "France" token in prompt → L26 FFN counterattack (Brussels 79.7%) → L30-31 contextual bleed reasserts Paris (72.7%) → L32-33 Brussels recovery (60.5%). The interference at L25 and L30-31 is prompt-driven — the "France" token fires regardless of residual state. Recirculation (feeding L33 back to L24) would encounter the same interference and produce marginal improvement (~3-8pp).

**Einstein** is qualitatively different: clean Germany computation at 98.4% for 7 consecutive layers (L24-L31), then catastrophic single-layer Ulm eruption at L33 (rank 500+ → 51.2%). The Ulm-direction projection at L28 is *negative* (−297) — Ulm actively points away from the residual. L33 FFN writes +1646 units of Ulm projection from nothing while subtracting 864 from Germany.

### Parallel Superposition (Branching)

The resolution: branch at the compositional peak into parallel trajectories through different force fields. Five branches per prompt, max-confidence collapse.

**France-north** (branch at L26): clean Belgium template recovers Brussels to 79.6% — virtually identical to the L26 peak. The neutral template ("The answer is") gives 78.8%. Both work because they lack the "France" token, not because they contain Belgian knowledge. Template amplification = contamination removal.

**Einstein** (branch at L30): injecting L30 residual directly at L33 (bypassing L31-32) gives Germany 98.3%. The Ulm confabulation is a two-step process: L31-32 preparation + L33 detonation. Bypass the preparation and the detonation can't fire. The Germany template fails (Switzerland 64.8%) because the Einstein residual carries Swiss associations that the template amplifies.

Five discoveries: (1) template amplification is contamination removal; (2) Ulm requires intermediate layer preparation; (3) force fields are ~90° to outputs (1.3° attention tilt is the mechanism); (4) max-confidence collapse beats majority vote; (5) some layers are traps (L30 is Paris-dominant regardless of input).

## Connections

| Finding | Related core document |
|---------|---------------------|
| Perplexity as anomaly signal | Routing architecture (BM25 + surprise) |
| Prompt-driven interference | `markov-7-percent-report.md` — Head 1 targeting by salience |
| L33 Ulm eruption mechanism | `trajectory_geometry_writeup.md` — three-way dissolution |
| Branching at 53 layers vs CoT at 300+ | `reconstruction_at_scale_results.md` — compute budgets |
| Template as contamination removal | `subspace_independence.md` — local coordinate frames |