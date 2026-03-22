# Archived — Crystallisation Experiments

Experiments testing whether a single crystallised residual vector (10 KB) can replace full window replay for fact retrieval, and whether geometric cosine similarity can route queries to passages at scale. Conducted March 2026 on `google/gemma-3-4b-it` (34 layers, 2560D).

The crystallisation transplant was the conceptual precursor to the knowledge store's persistent injection. The production system uses 12-byte injection (token ID + coefficient at L30 with agreement gating) instead of full 10 KB residual transplants.

## What's Here

| File | Experiment ID | Verdict | One-line summary |
|------|--------------|---------|------------------|
| `answer_crystallisation_results.md` | ed84135b | **Partial** | Transplant gives correct first token at 89-99% but can't sustain multi-token generation; amplification effect (clean KV > donor KV); 10 KB per passage |
| `geometric_routing_at_scale_results.md` | 45ecac84 | **Negative** | Cosine at L30 fails intra-document routing (correct passage ranks 36/50); works cross-domain only; keyword index is mandatory |

## Key Findings

**Crystallisation is real but insufficient.** The answer appears in the residual at L29-L30 in a sharp one-layer transition from 0% to >89%. Transplanting this vector into a bare query produces the correct first token. But the KV cache lacks context for continuation — "Volt" completes to "Voltara" correctly (single novel entity), but "Strand" becomes "Strandman" (confabulation). Full generation requires either persistent injection at every step or KV cache rebuild.

**Amplification effect.** Transplanted P(target) often exceeds the donor's own P(target): Strand goes from 40.1% (donor) to 94.9% (transplant). The clean KV cache environment doesn't fight the injected signal. This insight carried forward into the knowledge store design.

**The residual is not compressible.** PCA R5 (82% variance) misses the answer entirely. Only the full 2560D or the exact answer token direction (for single novel entities) works. No general-purpose compression shortcut.

**Geometric routing fails intra-document.** Within the Apollo 11 transcript, all windows share structural patterns (speaker labels, timestamps, call signs) that dominate the L30 residual (~0.97 cosine between all pairs). Content-specific signal is negligible. Cross-domain works (Zarkov vs Strand vs Apollo, 3/3) because domain separation >> intra-document separation.

## What This Led To

The crystallisation experiments established three things that shaped the knowledge store:

1. **L30 is the injection layer** — sharp crystallisation boundary, universal across probes
2. **Clean KV amplifies injected signals** — motivated agreement gating (inject only when the model's own circuit agrees)
3. **Keyword routing is mandatory** — geometric cosine cannot replace it at any tested scale

The production knowledge store compresses the 10 KB transplant down to 12 bytes (token ID + scalar coefficient) by exploiting the finding that novel single-entity facts compress to 1D along the answer token embedding. The full 2560D transplant is unnecessary when the injection targets a known token direction.

## Related Core Documents

- `knowledge_map_needs_results.md` / `knowledge_the_map_needs_results.md` — the 1D-per-fact and geometric hash findings that enabled 12-byte compression
- `compass_range_results.md` — L14 compass stability (the routing mechanism that does work for same-format comparison)