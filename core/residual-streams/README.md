# Core — Residual Stream Theory

The theoretical foundation for the Lazarus architecture. Five experiments on `google/gemma-3-4b-it` (34 layers, 2560D) proving the Markov property of the residual stream, characterising its bandwidth, establishing the limits of compositional reasoning, discovering that factual subspaces are local coordinate frames, and mapping the complete dual-circuit retrieval mechanism.

Everything in the Lazarus system — checkpoint chaining, injection, the knowledge store, compass routing — depends on these results. If someone asks "why does any of this work?", start here.

## What's Here

| File | Experiment ID | One-line summary |
|------|--------------|------------------|
| `markov-writeup.md` | 929cdef0 | **The proof.** Cross-step Markov confirmed (10/10 token match). Full-sequence all-position patching KL=0.0 across 3 conditions including cross-task. Token-embedding subspace fails (produces Sydney not Canberra). |
| `markov-bandwidth-report.md` | b8c3035a | **The bandwidth.** Feature expands from 1D (L16) to 6D (L33). Sharp binary patching threshold at L14. Competing context dilutes at L24 but recovers by L33. CoT drops Denmark projection 61%. |
| `markov-7-percent-report.md` | 9bb84464 | **The limits.** Head 1's ~93% three-position coverage is constant across complexity. Failure is targeting not bandwidth. No patching threshold for compositional prompts. Chain residue encoded as competing dimensions. |
| `subspace_independence.md` | fd54bf19 | **The address theory.** Cross-domain PCA works (within-family fails). 8 dimensions per fact (0.3%). L24 crystallisation point. The subspace IS the address: France via Australia's frame → Canberra. |
| `charge_peak_results.md` | 993f5f93 | **The retrieval mechanism.** Novel facts via L29 H4 attention (+22 DLA), parametric via L25/L33 FFN (+41 to +82). Sign flip at L29 IS the retrieval event. 5.1KB single residual = KL=0.0008. 30:1 novel/parametric classifier. |

## The Proof Chain

### 1. The Markov Property (markov-writeup)

The residual stream at any layer is the complete computational state. Three tests:

- **Cross-step:** Generate 10 tokens with KV cache, then replicate with fresh forward passes. 10/10 exact match. KV cache carries zero additional information — it is pure optimisation.
- **Full-sequence:** Transplant all-position residuals at L14 from "capital of Australia" into "language spoken in France." Output: Canberra 92%, KL=0.0. The prompt history is irrelevant given the full residual state.
- **Subspace:** Token-embedding basis (Sydney/Canberra directions) injected into an unrelated prompt produces Sydney, not Canberra. The functional subspace is not aligned with vocabulary directions. This failure motivated the subspace independence experiment.

### 2. Bandwidth Characterisation (markov-bandwidth-report)

The capital-city feature is not a fixed channel — it is a growing semantic subspace:

- **L16:** ~1D format signal ("you're asking about a capital")
- **L24:** ~5D country-specific content emerging
- **L33:** ~6D with country answers as primary distinguishing content

L14 is the sharp binary patching threshold for same-task prompts. Below L14: the residual encodes task type but not specific answer. Above L14: the answer survives all subsequent attention reads.

Competing context (geographic descriptions) dilutes Canberra at L24 (22.7% → 11.6%) but L26 FFN and L33 fully reconcentrate it (92.2% → 92.6%). Late layers are both amplifiers and corruptors.

CoT measurably cleans the Markov state: Denmark's projection drops 61% when the referent is resolved in prior tokens. The Markov state at L24 is quantifiably cleaner. But CoT introduces template competition (L30 temporal template hijacks a correct geographic circuit).

### 3. Complexity Limits (markov-7-percent-report)

The effective Markov state does not widen for compositional prompts:

- **Head 1** concentrates on ~3 positions (BOS + topic + self, ~93% coverage) regardless of prompt complexity
- **The failure is targeting:** Head 1 fires on the most surface-salient entity, not the compositionally correct one. "Hamlet" → Denmark (play's setting), not Shakespeare → Stratford (author's birthplace)
- **No patching threshold exists for compositional prompts** — single-position patching never flips the output at any layer (L8 through L32 all fail)
- **Feature dimensionality scales:** Simple capitals separate in 1D; compositional prompts require 5D, encoding chain residue (Shakespeare, Polish alongside the answer)
- **Wrong-direction dominance is geometric:** Denmark:Stratford = 4.4:1 at L26, but no compositional fact store exists to override it (unlike the simple L26 FFN for capital facts)

### 4. The Address Theory (subspace_independence)

The deepest theoretical result. The factual landscape is local coordinate charts on a manifold, not globally separable regions:

- **Within-family PCA fails** (capital prompts only) because it captures universal high-energy directions present in all residuals (subspace cosine 0.974 between donor and recipient — near-identical projections)
- **Cross-domain PCA works** (capitals + unrelated prompts): Canberra 94.1% injected into a translation prompt, KL=0.021, orthogonal cosine 0.9997
- **8 dimensions per fact** (0.3% of 2560D) is the minimum bandwidth. Sharp transition: rank 6 fails, rank 8 succeeds
- **L24 is the crystallisation point** — exactly where Head 1 writes the contextual attribute bridge. Below L24 the coordinate frame doesn't exist; above it the frame is stable through L30
- **The subspace is the address, not the content:** France injected through Australia's frame → Canberra. Australia injected through France's frame → Paris. The output follows the coordinate frame, not the vector content.

### 5. The Retrieval Mechanism (charge_peak_results)

The complete empirical characterisation of how facts move through the residual stream:

- **Novel facts (KV pathway):** L23 H1 first read (75.8% attention to fact position, DLA +1.55) → L29 H4 main copy (44.9%, DLA +1.74, 99.1% of layer) → L30 H0+H3 reinforcement. Total attention DLA: +22.0, total FFN: -12.7. Attention-dominated.
- **Parametric facts (weight pathway):** L23-L26 FFN builds the signal. Total attention DLA: -32.6 to -114.1 (suppressive), total FFN: +41.2 to +81.5. FFN-dominated. Complete mirror image.
- **The sign flip:** At the generation position, Volt projection goes from -797 (L22) to +1,556 (L29) — a delta of 2,353 units. This IS the retrieval event.
- **Fact position encodes "what comes next", not "what I am":** " Volt" never appears in top-10 at its own position. The residual encodes "ara" (next token). Identity lives in dark space. You must capture the answer at the destination (generation position at L29+), not the source.
- **5.1KB single residual:** One 2560D bfloat16 vector from the generation position at L29 reconstructs the answer with KL=0.0008. First proof that single-vector injection works.
- **30:1 novel/parametric classifier:** L29+L30 attention DLA > +5 = novel (KV retrieval); < +1 = parametric (FFN retrieval). Clean binary separator.

## Key Numbers

| Measurement | Value |
|-------------|-------|
| Cross-step Markov | 10/10 token match without KV cache |
| Full-sequence patching | KL=0.0 (3 conditions, including cross-task) |
| L14 patching threshold | Sharp binary (same-task) |
| Compositional patching threshold | None (never flips at any layer) |
| Feature bandwidth | 1D (L16) → 6D (L33), expanding |
| Head 1 coverage | ~93%, constant across complexity |
| CoT Denmark reduction | 61% projection drop |
| Minimum dims per fact | 8 (0.3% of 2560D) |
| Cross-domain injection | Canberra 94.1%, KL=0.021 |
| Subspace as address | France via Australia's frame → Canberra |
| L24 crystallisation | Injection works at 92.4% from L24; fails at L22 |
| L29 sign flip (novel retrieval) | -797 → +1,556 (delta 2,353) |
| Single-residual injection | KL=0.0008 (5.1KB at L29) |
| Novel/parametric classifier | L29+L30 attn DLA, 30:1 gap |
| Novel total DLA | Attention +22, FFN -13 |
| Parametric total DLA | Attention -33 to -114, FFN +41 to +82 |

## Why This Matters for Lazarus

| Principle | Architectural consequence |
|-----------|------------------------|
| Residual is complete Markov state | Checkpoint chaining works — restoring residuals = restoring computation |
| KV cache is pure optimisation | Replay from stored tokens reconstructs identical KV cache |
| L14 threshold | Entity compass layer — where entity identity enters the last-position residual |
| Bandwidth expands, not compresses | Later layers carry richer country-specific signal; L26 is the commitment layer |
| Head 1 targets by salience | Compositional failures need CoT or branching, not deeper attention |
| Local coordinate frames | Per-fact 1D injection works because each fact has its own private 8D frame |
| L24 crystallisation | Injection targets L30 (post-crystallisation, post-copy-head) |
| 8D per fact | 12-byte knowledge store entries are sufficient — the signal is tiny but independent |
| Dual-circuit retrieval | Novel facts via attention (L29 H4), parametric via FFN (L25/L33) — different pathways, same injection site |
| Sign flip at L29 | The retrieval event is a single discrete transition, not gradual accumulation |
| Fact position ≠ answer carrier | Capture at the generation position after retrieval, not at the source token |
| 30:1 novel/parametric classifier | Grounding detector can distinguish retrieval source from DLA alone |