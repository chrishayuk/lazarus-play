# Fact Markers in the Residual: Can We Detect Facts Without a Query?

**Experiment ID:** 218ab701-f2bc-44be-a16b-82a56f963f6c
**Model:** google/gemma-3-4b-it (34 layers, 2560 hidden dim, 8 heads)
**Date:** 2026-03-16

## Summary

**The model does NOT mark facts during prefill.** Seven independent experiments searched for geometric signatures that distinguish fact-bearing tokens from filler during raw prefill (no query). Six returned negative. One (local outlier) found a partial signal that correlates with semantic novelty/entity-specificity, not fact-bearing status.

The fact only becomes addressable when a query activates the retrieval circuit. During prefill, the model encodes each token's relationship to context (next-token prediction), not its potential usefulness for future retrieval.

---

## Test Windows

Three passages with labeled token roles:

- **W1 (scratch):** "Apollo 11 transcript. Houston, this is Eagle. The sound quality was scratchy during the lunar module descent. Roger, copy that. We had some interference on the channel. Over."
- **W2 (Jets):** "The 1969 Super Bowl was won by the New York Jets. Joe Namath guaranteed victory before the game. The final score was sixteen to seven. It was played at the Orange Bowl in Miami. The weather was clear and warm that day."
- **W3 (Hornet):** "After separation from the command module, the lunar module Hornet descended to the surface. Splashdown occurred in the Pacific Ocean on July 24th. Recovery ship USS Hornet retrieved the crew safely. The mission lasted eight days and included two moonwalks."

---

## Experiment 1 — Residual Norm

**Question:** Do fact tokens have unusual L2 norms at any layer?

**Method:** `residual_decomposition` at 14 positions across W1, layers L0-L33.

**Result: NEGATIVE**

| Token | Role | L29 norm | L33 norm |
|-------|------|----------|----------|
| sound | CONTEXT | 8192 | **26624** |
| Over | FILLER | 5792 | 20608 |
| Houston | ENTITY | 7040 | 14464 |
| Apollo | ENTITY | 2448 | 12672 |
| copy | FILLER | 5248 | 12096 |
| **scratch** | **FACT** | **5888** | **10240** |
| Roger | FILLER | 6112 | 7424 |
| interference | CONTENT | 4224 | 5152 |

"sound" (non-fact context word) has the highest norm at L33 (26624). "scratch" (FACT) is mid-pack at 10240. Norms track next-token prediction dynamics (how confident/specific the prediction is), not fact-bearing status.

---

## Experiment 2 — Layer-to-Layer Deltas

**Question:** Do fact tokens cause larger residual changes at retrieval layers (L23, L29)?

**Method:** Full 34-layer `residual_decomposition` for scratch (FACT), Roger (FILLER), Houston (ENTITY), sound (CONTEXT).

**Result: NEGATIVE**

All positions follow the same general delta trajectory:
- Small deltas L0-L6 (embedding adjustment)
- Spike at L4/L7-L10 (entity/attention processing)
- Moderate L11-L22
- Growing L22-L33 (output preparation)

No systematic difference at retrieval layers. Houston (ENTITY) has a huge delta at L11 (6400) and L13 (4768) — entity-specific processing — but scratch (FACT) does not show equivalent spikes at L23 or L29.

---

## Experiment 3 — V Vector Content (Decode Residual)

**Question:** Does the residual encode fact identity at retrieval layers during prefill?

**Method:** `decode_residual` at L7/L14/L23/L29 for fact, filler, and entity positions.

**Result: NEGATIVE**

All positions have identical raw top tokens — the dark space attractors (ꗜ, 𒉶, PLDNN, etc.) that dominate the unnormalized residual. Mean energy fraction < 0.07% at all layers = >99.93% dark space.

The normalized view (after layer norm) shows standard next-token prediction:
- "scratch" at L29: predicts "y" at 99.3% (morphological completion)
- "Roger" at L29: predicts "understood" at 89% (contextual)
- "Houston" at L29: predicts "control" at 39.6% (entity-contextual)

The model encodes "what comes next at this position," not "I am a fact."

---

## Experiment 4 — Per-Layer Surprise

**Question:** Is the surprise signal (prediction error) at L29 a better fact detector than at L33?

**Method:** `decode_residual` at the position PRECEDING each fact token to measure P(fact_token).

**Result: NEGATIVE — Surprise decorrelates from fact status**

| Before → After | P(fact) at L29 | Surprise | Outlier Score |
|----------------|---------------|----------|---------------|
| "was" → "scratch" | <<0.05% | VERY HIGH | 0.025 |
| "York" → "Jets" | **64.0%** | **LOW** | **0.042** |
| "was" → "sixteen" | <<0.1% | VERY HIGH | 0.050 |
| "." → "Roger" | ~0.7% | HIGH | 0.022 |

"Jets" is well-predicted (64% probability) but has a high outlier score. "scratch" is very surprising but has a moderate outlier score. Surprise and geometric displacement measure different things — surprise is prediction error; outlier is semantic novelty.

**Key insight from "was" → next predictions:**
- W1: "quality was" → model predicts "poor" (67.6%), "terrible" (22%), "bad" (5.6%). "scratch" not in top 20.
- W2: "score was" → model predicts " " (89%), "New" (10.6%). "sixteen" not in top 20.

The model expects quality adjectives and numeric formats, not the specific fact tokens. But this doesn't make the fact tokens geometrically special — it just makes them surprising.

---

## Experiment 5 — FFN Neurons at L29

**Question:** Are there neurons that fire specifically for fact-bearing tokens across different windows?

**Method:** `top_neurons` at L29 for fact positions across all three windows.

**Result: NEGATIVE**

| Token | Window | Top Neuron | Contribution |
|-------|--------|-----------|-------------|
| scratch | W1 | 5683 | +0.512 (→ "ier") |
| Jets | W2 | 5754 | -0.085 (→ "of") |
| Hornet | W3 | 5754 | +0.204 (→ "of") |
| Roger (filler) | W1 | 4466 | +0.039 (→ "،") |
| clear (filler) | W2 | 5754 | +0.119 (→ "of") |

Neuron 5754 appears for Jets, Hornet, AND clear (filler). Neuron 5683 is unique to "scratch" (morphological -y completion). No neuron fires consistently for facts and only facts.

---

## Experiment 6 — Dark Space Clustering (Feature Dimensionality)

**Question:** Do fact-ending and filler-ending prompts separate in a subspace?

**Method:** `feature_dimensionality` with prompts truncated to end at fact vs filler tokens.

**Result: NEGATIVE — Apparent signal was a confound artifact**

| Layer | Sample Size | 1D Accuracy | 10D Accuracy | PC1 Interpretation |
|-------|-------------|------------|-------------|-------------------|
| L7 | 6+6 | **91.7%** | 91.7% | Dark space norm |
| L14 | 6+6 | **91.7%** | 91.7% | ꗜ (dark attractor, 85.7% var) |
| L29 | 6+6 | 83.3% | 91.7% | Dark space norm |
| **L29** | **8+8** | **50.0%** | 87.5% | **Collapsed to chance** |

With 6+6 samples, PC1 separated at 91.7% — but this was classifying by dark space magnitude (content words vs function words), not fact-ness. When expanded to 8+8 with more diverse prompts, 1D accuracy dropped to **50% (chance)**. The 10D accuracy of 87.5% with 16 samples is likely overfitting (10 dimensions for 16 data points).

---

## Experiment 7 — Local Outlier (Geometric Displacement)

**Question:** Are fact tokens geometrically unusual compared to their neighbours?

**Method:** `compare_activations` with prompts truncated to end at consecutive positions. Computed cosine similarity between adjacent last-position residuals at L29.

**Result: PARTIAL — Signal exists but confounded with semantic novelty**

### Outlier Scores (1 − mean cosine to neighbors)

| Token | Role | L29 Outlier Score |
|-------|------|------------------|
| sixteen | FACT (specific number) | **0.050** |
| Jets | FACT (entity name) | **0.042** |
| Ocean | FACT (location) | ~0.035 |
| scratch | FACT (adjective) | 0.025 |
| interference | Non-fact content | 0.022 |
| Roger | Filler | 0.022 |
| clear | Filler | 0.020 |
| copy | Filler | 0.017 |
| during | Function | 0.013 |
| the | Function | 0.008 |

### Layer specificity
- **L29:** Clear gradient (facts 0.025-0.050 vs filler 0.008-0.022)
- **L14:** All cosines >0.997, no separation. Centroid distance = 0.0017 (vs 0.023 at L29)

The local outlier effect is **strongest at L29 (retrieval layer)** and absent at L14. But it correlates with **semantic novelty and entity-specificity**, not fact-bearing status:

- "Jets" (entity name, 64% predicted) → HIGH outlier despite LOW surprise
- "sixteen" (specific number, <<0.1% predicted) → HIGHEST outlier
- "scratch" (common adjective, very surprising) → moderate outlier
- "interference" (informative but expected) → same as "Roger" (filler)

The signal is a gradient of **how much new semantic content the token introduces**, which partially overlaps with but does not cleanly separate facts from non-facts.

---

## Conclusions

### What doesn't work (6 negative results)

1. **Residual norm** — Norms track prediction confidence, not fact status
2. **Layer deltas** — No preferential processing at retrieval layers for fact tokens
3. **V vector / decoded residual** — All positions are 99.93%+ dark space; normalized view shows next-token prediction only
4. **Per-layer surprise** — Prediction error decorrelates from geometric displacement
5. **FFN neurons** — No universal fact-marking neurons; neurons are context-specific
6. **Dark space clustering** — Apparent 91.7% accuracy was content-word vs function-word confound; collapsed to chance with more samples

### What partially works (1 result)

7. **Local outlier at L29** — Fact tokens do tend to have higher geometric displacement from neighbors (0.025-0.050) than filler (0.008-0.022). But this measures **semantic novelty**, not fact-ness. It's a noisy proxy at best: entity names that are NOT facts (like "Houston" as a callsign) would score similarly to entity names that ARE facts (like "Jets" as a team).

### Theoretical Implications

**The model does not pre-index facts during prefill.** There is no geometric "tag" that says "this token carries retrievable information." The retrieval circuit (L29 H4 = 44.9% attention to fact position during generation) is purely **query-driven** — it activates when a query arrives and uses attention to locate the relevant position.

This makes architectural sense: during prefill, the model doesn't know what questions will be asked. Marking all potentially-useful tokens would require anticipating all possible queries. Instead, the model stores ALL tokens in the KV cache with equal status, and the retrieval head selects the right one at query time.

**The closest thing to a marker is semantic novelty** — tokens that introduce new information cause larger geometric displacements at retrieval layers (L29). This is a byproduct of how the model processes information (novel tokens require more representational adjustment) rather than an intentional indexing signal.

### Practical Implications

For fact extraction systems:
- **Cannot use model internals** to identify fact positions during prefill
- **External methods** (NER, POS tagging, semantic role labeling) remain necessary for pre-indexing
- The **local outlier heuristic at L29** could serve as a weak pre-filter (rank positions by geometric displacement, top quartile enriched ~2x for facts), but precision would be poor
- The **surprise heuristic** at L33 measures a different signal (prediction error) and does not cleanly identify facts either
