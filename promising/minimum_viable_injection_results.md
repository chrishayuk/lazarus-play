# Minimum Viable Injection — Experiment Results
**Experiment ID:** 2bd41b18-03b1-4d81-87fa-00fae3279733
**Model:** google/gemma-3-4b-it
**Date:** 2026-03-17

## Setup

- **3-fact prompt:** "Zarkov Industries was founded in the city of Voltara in 1987. The crew reported that the audio quality was scratchy during descent. Joe Namath agreed to sell his restaurant and report to the New York Jets training camp."
- **Queries:** "Zarkov Industries was founded in the city of" | "The crew reported that the audio quality was" | "Joe Namath agreed to"
- **Target tokens:** " Volt" | " scratch" | " sell"
- **Baseline:** Bare query has ~0% for correct answers; full-context has 96.9% / 84.8% / 28.1%

----

## Experiment 1 — Head Attribution at L29

**Head DLA for " Volt" at L29:**

| Head | DLA | Fraction | Top token |
|------|-----|----------|-----------|
| H0 | −0.011 | −0.6% | " " |
| H1 | +0.023 | +1.2% | " " |
| H2 | −0.060 | −3.1% | " версию" |
| H3 | −0.049 | −2.5% | " Visiting" |
| **H4** | **+1.977** | **+100.6%** | **" Volt"** |
| H5 | +0.076 | +3.9% | " Vol" |
| H6 | +0.013 | +0.6% | " " |
| H7 | −0.002 | −0.1% | \<unused\> |
| **Layer total** | **+1.964** | | |

H4 alone = 100.6% of the layer's contribution to " Volt". All other heads cancel out or contribute noise.

**Component interventions (zeroing each component at L29):**

| Zeroed | P(" Volt") | Δ | KL | Conclusion |
|--------|------------|---|-----|------------|
| H4 | 78.1% | −18.8pp | 0.165 | Main copy head |
| H5 | 96.1% | −0.8pp | 0.0004 | Structural only |
| H3 | 97.7% | **+0.8pp** | 0.0 | Slightly harmful — zeroing helps |
| H2 | 97.7% | **+0.8pp** | 0.0 | Slightly harmful — zeroing helps |
| FFN | **98.4%** | **+1.6pp** | 0.007 | **FFN is anti-correlated with fact signal** |
| All attention | 85.2% | −11.7pp | 0.086 | Residual from L0-L28 carries 85% |

**Key insight:** The FFN at L29 is slightly *hurting* factual retrieval. Zeroing it improves confidence. The 78.1% that survives H4 ablation means the signal is already accumulated from L0-L28 by earlier copy heads. H4 adds the final 18.8pp sharpening step.

---

## Experiment 2 — Injection Method Comparison

### Full Residual Injection (Baseline)

| Method | P(" Volt") | KL from donor | Storage/fact |
|--------|------------|---------------|--------------|
| Full L29 residual | 97.65% | 0.0019 | 5,120 bytes |

Confirms the V-injection result. The full 2560D residual works.

### Subspace Injection — The Surprise

Instead of injecting the full residual, inject only the component of the donor residual that lies in the *answer token's embedding direction*.

| Method | P(" Volt") | KL | Energy used | Storage |
|--------|------------|-----|-------------|---------|
| Full residual L29 | 97.65% | 0.0019 | 100% (2560D) | 5,120 bytes |
| 3-token subspace L29 | **99.85%** | 0.106 | **0.17%** | 24 bytes |
| **1-token subspace L29** | **99.53%** | 0.077 | **0.05%** | **12 bytes** |
| 3-token subspace L25 | 6.9% | 7.82 | — | FAILS |
| 3-token subspace L23 | 7.2% | 10.83 | — | FAILS |

**The subspace injection is BETTER than full residual injection** (99.85% vs 97.65%).

Why: the structural context in the bare query's residual (99.95% of its energy) is *better suited* for L31-L33 than the donor's structural context. Only the factual direction needs updating.

**Layer scan for 1D subspace injection:**

| Layer | donor_fraction | P(" Volt") | KL | subspace_cosine | Works? |
|-------|---------------|------------|-----|-----------------|--------|
| L27 | 0.009% | 7.2% | 8.45 | +1.0 | ✗ |
| L28 | 0.004% | 6.9% | 5.94 | +1.0 | ✗ |
| **L29** | **0.05%** | **99.53%** | 0.077 | **−1.0** | ✓ |
| **L30** | **0.08%** | **99.96%** | 0.136 | **−1.0** | ✓ |

**Phase transition at L29:** H4 fires and writes the answer into the vocab direction. At L28, the donor has a tiny same-direction signal (subspace_cosine = +1.0) — essentially nothing to inject. At L29, the signal jumps 14× and flips anti-aligned with the bare query (subspace_cosine = −1.0). The injection both removes the bare query's anti-signal AND installs the donor's pro-signal — double contribution.

**L30 is the best injection layer** because cleanup heads at L30 (H0: +1.36, H3: +1.88 DLA from prior experiments) pre-amplify the signal before injection.

---

## Experiment 3 — Three-Fact Validation at L30

| Fact | Bare query top-1 | Donor | L30 1D/3D injection | Override? |
|------|-----------------|-------|---------------------|-----------|
| Zarkov → Voltara | " New" 6.5% | " Volt" 97.2% | **99.96%** | ✓ |
| Audio → scratchy | " poor" 32.5% | " scratch" 84.8% | **97.6%** | ✓ |
| Namath → sell | " play" 40.8% | " sell" 28.1% | **56.3%** | ✓ |

All three facts work. Including the hard case (Namath), where:
- The donor itself is uncertain (only 28.1% " sell")
- The bare query has a strong parametric prior (" play" 40.8%)
- A single scalar injected at L30 flips it to " sell" 56.3%

---

## Experiment 4 — Final Architecture

### Minimum Viable Storage Unit

**12 bytes per fact:**
- 4 bytes: answer token ID
- 8 bytes: projection coefficient `c = dot(R_donor_L30, embed(answer_token)) / norm(embed(answer_token))`

The subspace basis (the answer token's embedding direction) is part of the model weights — zero per-fact cost.

### Inference Algorithm

**Prefill (once per document):**
```
for each fact in document:
    run document through L0→L30
    answer_token_id = tokenize(fact_answer)[0]
    e = embed(answer_token_id) / norm(embed(answer_token_id))
    c = dot(R_L30_last_position, e)
    store(answer_token_id, c)  # 12 bytes
```

**Inference (per query):**
```
route query → matched fact(s) via Q·K
run bare_query through L0→L29
at L30:
    e = embed(answer_token_id) / norm(embed(answer_token_id))
    R_patched = R_bare + (c - dot(R_bare, e)) * e
continue L31→L33 → generate
```

### Compression Table

| Method | Storage/fact | 3,625 facts | Notes |
|--------|-------------|-------------|-------|
| Full KV cache | ~28 MB | ~100 GB | Standard RAG |
| Full L29 residual | 5,120 bytes | 18.6 MB | V-injection baseline |
| 3D subspace L30 | 24 bytes | 87 KB | Better accuracy |
| **1D subspace L30** | **12 bytes** | **43.5 KB** | **Minimum viable** |

**1D subspace at L30 = 2,400,000× smaller than KV cache per fact.**
**43.5 KB for 3,625 facts.**

---

## Summary of Discoveries

1. **The answer lives in 0.05% of residual energy.** The remaining 99.95% is structural context that belongs to the query, not the fact. Injecting the full residual wastes 99.95% and slightly degrades accuracy.

2. **Subspace beats full residual.** The model's downstream layers (L31-L33) perform *better* when given the bare query's structural context with only the answer direction updated. Structural context of the donor is noise for the recipient.

3. **H4 is the only causally necessary head** (100.6% of layer DLA). FFN at L29 is anti-correlated with factual content and should not be included in any injection scheme.

4. **Phase transition at L29 is sharp.** One layer before (L28), essentially no signal. At L29, H4 writes the answer with a ×14 amplitude jump and flips the direction anti-aligned with the bare query.

5. **L30 is the optimal injection point.** Cleanup heads pre-amplify and the donor_fraction doubles from L29 (0.05%) to L30 (0.08%). Especially important for weak-signal or parametrically-conflicted facts.

6. **Works against strong parametric priors.** Namath: model's world knowledge predicts " play" at 40.8%. A 12-byte injection overrides it to " sell" 56.3%.

---

## Open Questions

- Does multi-fact superposition work? (inject multiple 1D components simultaneously — theory says yes, token embeddings are nearly orthogonal in 2560D, but untested)
- Does it scale to multi-token answers? (need to inject at multiple generation steps)
- What's the routing precision requirement? (wrong routing → wrong answer at high confidence — Q·K routing is the safety gate)
- Is 12 bytes sufficient for all fact types or do ambiguous/rare tokens need 3D?
