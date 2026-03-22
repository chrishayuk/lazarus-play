# Answer Crystallisation Experiment Results

**Experiment ID:** ed84135b-08fe-4ce8-a4d7-f24a3976a51c
**Model:** google/gemma-3-4b-it (34 layers, 2560D, 8 heads)
**Date:** 2026-03-21

## Thesis

The residual at the last position, at some layer, contains the complete
answer state after full attention over context tokens. If we capture that
vector during prefill and transplant it at query time, we replace replay
(512 tokens) with a single 10 KB vector. No context window needed.

## Probes

| Probe | Context | Query | Target | Token ID |
|-------|---------|-------|--------|----------|
| Zarkov | Novel company in Voltara | "...founded in the city of" | Volt | 89711 |
| Strand | Novel physicist, Kessler Prize | "...laureate in 2014 was Dr." | Strand | 73442 |
| Castellan | Novel observatory director | "...surname of the director was" | Cast | 10184 |

All probes: 0% target probability without context (pure confabulation).

## Experiment 1 — Crystallisation Layer (logit_lens)

| Layer | P(Volt) | P(Strand) | P(Cast) |
|-------|---------|-----------|---------|
| L0–L27 | 0% | 0% | 0% |
| L28 | 0% (Vol 3.6%) | 0% | 0% |
| **L29** | **50.4%** | 0% | 0% |
| **L30** | **92.6%** | **84.8%** | **44.1%** |
| L31 | 96.1% | 68.4% | 15.4% |
| L32 | 78.9% | 47.1% | drops |
| L33 | 96.9% | 40.2% | 29.7% |

**Finding:** Sharp one-layer crystallisation boundary.
- Zarkov: L29 (copy head H4 fires)
- Strand/Castellan: L30 (primary retrieval layer)
- Before boundary: pure confabulation. After: correct answer.

## Experiment 2 — Residual Transplant at L30

Transplant donor residual (context+query at L30) into bare query (no context).

| Probe | Bare query | Donor | **Transplant** | KL(D→I) |
|-------|-----------|-------|---------------|---------|
| Zarkov | P(Volt)=0% | 97.1% | **99.4%** | 0.028 |
| Strand | P(Strand)=0% | 40.1% | **94.9%** | 1.54 |
| Castellan | P(Cast)=0% | 29.5% | **89.1%** | 1.38 |

**THE TRANSPLANT WORKS.** All three probes produce the correct first token.

**Amplification effect:** Injected P(target) EXCEEDS donor P(target) for
Strand (94.9% vs 40.1%) and Castellan (89.1% vs 29.5%). The clean KV
cache environment (no competing context signals) amplifies the transplanted
answer direction.

## Experiment 3 — Layer Sweep

| Transplant Layer | Zarkov P(Volt) | Strand P(Strand) | Castellan P(Cast) |
|-----------------|---------------|-----------------|-------------------|
| L20 | 0% (New) | — | — |
| L22 | 0% (New) | — | — |
| L24 | 0% (Ver) | — | — |
| L26 | 0% (Ver) | 0% (Kessler) | — |
| L28 | 0% (Vol) | 0% (Kessler) | — |
| **L29** | **93.7%** | 3.9% (James) | 0% (a) |
| **L30** | **99.4%** | **94.9%** | **89.1%** |
| L31 | 99.4% | 95.9% | 92.0% |
| L32 | 97.9% | 89.4% | — |
| L33 | 97.0% | 41.0% | 29.1% |

**Razor-sharp transition.** One layer takes P(target) from 0% to >89%.
Optimal transplant layer: **L30–L31** (highest P across all probes).

## Experiment 4 — Generation Sustainability

| Probe | First token | Continuation | Full text |
|-------|------------|-------------|-----------|
| Zarkov | Volt ✓ | ara ✓ | "Voltara, a city known for its advanced technology..." |
| Strand | Strand ✓ | man ✗ | "Strandman, for his work on the Kessler-Strandman method..." |
| Castellan | Cast ✓ | ine ✗ | "Castine. Specifically, it was **Castine**..." |

**Finding:** Single-token answers survive generation (Volt→ara works).
Multi-token answers get first token right but continuation diverges.
The KV cache lacks context for proper name completion.

**Implication:** For production, either:
1. Use single-token answer targets
2. Inject residual at EVERY generation step
3. Build KV cache from transplanted state before generating

## Experiment 5 — Dimensionality

| Method | Dims | Zarkov | Strand | Castellan |
|--------|------|--------|--------|-----------|
| Token 1D (answer token) | 1 | 95.4% ✓ | 0% ✗ | 0% ✗ |
| Token 8D (context tokens) | 8 | — | 0% ✗ | 0% ✗ |
| PCA R1 | 1 | — | 0% ✗ | — |
| PCA R3 | 3 | — | 0% ✗ | 0% ✗ |
| PCA R5 | 5 | — | 0% ✗ | 4.3% ✗ |
| PCA R9 (full span) | 9 | 99.4% ✓ | 94.9% ✓ | 89.1% ✓ |
| Full 2560D | 2560 | 99.4% ✓ | 94.9% ✓ | 89.1% ✓ |

**Not compressible.** The answer signal is distributed across the full
variation between donor and recipient. PCA R5 (82% variance) misses
the answer entirely. Only the full 9D span (100% variance among prompts)
works — but this doesn't generalise to unseen probes.

**Exception:** Novel single-fact entities (Zarkov) compress to 1D along
the answer token embedding — consistent with the 12-byte injection result.

**Cost:** 10 KB per passage (2560 × float32). No compression shortcut.

## Experiment 6 — Passage Routing

| Query → | Zarkov donor | Strand donor | Castellan donor | Correct? |
|---------|-------------|-------------|----------------|----------|
| Zarkov | **0.9859** | 0.9754 | 0.9705 | ✓ |
| Strand | 0.9707 | **0.9826** | 0.9654 | ✓ |
| Castellan | 0.9758 | **0.9764** | 0.9723 | ✗ |

**Routing: 2/3.** Raw cosine at L30 insufficient. All similarities >0.96,
margins ~0.01. Castellan misroutes to Strand (0.9764 vs 0.9723).

Consistent with prior finding: dark space = GPS, not topic index.
External routing (keyword index) needed.

## Experiment 7 — Store Format

| Document | Windows | Store size |
|----------|---------|-----------|
| 370K tokens | 725 | 7.25 MB |
| 1M tokens | 1,953 | 19.5 MB |
| 10M tokens | 19,531 | 195 MB |

Per passage: 10 KB (2560D × float32).

## Key Findings

### What works
1. **Crystallisation is real.** The answer appears in the residual at L29–L30
   in a single sharp transition from 0% to >89%.
2. **Transplant works.** Replacing the bare query's residual at L30 with the
   donor's produces the correct first token at 89–99% probability.
3. **Amplification.** The transplant often gives HIGHER P(target) than the
   original donor pass, because the clean KV cache doesn't fight the signal.
4. **L30 is the universal crystallisation layer** for in-context retrieval.

### What doesn't work
1. **Multi-token generation diverges.** First token correct, continuation
   confabulates (Strandman, Castine). KV cache lacks context.
2. **Compression fails.** The answer requires the full 2560D residual.
   No PCA or token-subspace shortcut works for general-purpose retrieval.
3. **Cosine routing fails.** L30 residuals are 97%+ similar across all
   prompts. Margin for correct routing is ~0.01. Not reliable.

### Architecture implication

The transplant gives **one correct token** from 10 KB. That's the answer
direction. For full answer generation, you need either:

- **Persistent injection:** Inject the crystallised residual at L30 on
  every autoregressive step (not just the first). Cost: one transplant
  per generated token.
- **KV cache prefill:** Run the transplanted residual through L30→L33
  to build proper KV entries, then generate with those cached. This
  essentially replays 4 layers instead of 34.
- **Single-token answer design:** If you only need the first token
  (entity identification, classification), the transplant is sufficient.

### The residual is a computation pointer, not a fact buffer

The crystallised residual at L30 contains the answer DIRECTION but not
the full answer SURFACE. It tells the unembedding "the answer starts with
Volt/Strand/Cast" but doesn't contain the KV cache entries that would
allow downstream layers to complete "ara"/""/""ellan" correctly.

This is consistent with the Markov property at the TOKEN level: the
residual at L30 determines the NEXT TOKEN but not the full continuation.
For full continuation, you need either the full KV cache (replay) or
persistent injection (transplant at every step).

### Comparison with prior approaches

| Method | Cost per passage | First token | Full answer | Routing |
|--------|-----------------|-------------|-------------|---------|
| Replay (512 tokens) | 56 KB KV | 100% | 100% | Keyword |
| 12-byte injection | 12 bytes | 100% (novel) | 100% (novel) | Template match |
| **Crystallisation transplant** | **10 KB** | **89–99%** | **First only** | **Needs external** |
| Full Markov (all positions) | Full KV | 100% | 100% | N/A |

The crystallisation transplant sits between 12-byte (novel only, template
matched) and replay (full answer, any content). It gives the first token
from any content but can't sustain generation. The 5× cost reduction over
replay (10 KB vs 56 KB) may not be worth the loss of full generation.
