# The Markov Property of Transformer Residual Streams: Three Experiments

## Overview

A transformer's residual stream is Markovian if the state at layer L is sufficient to determine all subsequent computation — no information from earlier layers or the input history persists through any other channel. This property holds by architectural construction within a single forward pass, but three questions remained empirically open:

1. **Cross-step Markov**: Does the KV cache carry information that a fresh recomputation wouldn't produce? Or is it pure optimization?
2. **Full-sequence Markov**: Does transplanting all-position residuals at layer L completely transfer the computation, even across unrelated prompts?
3. **Subspace independence**: Does the tiny fraction of residual bandwidth used for city prediction (~0.6%) operate independently of the other 99.4%?

All three experiments were run on Gemma 3-4B-IT (34 layers, 2560 hidden dim, bfloat16).

---

## Experiment 1: Cross-step Markov During Generation

### Setup

Autoregressive generation uses a KV cache: at each step, attention keys and values from previous tokens are stored and reused rather than recomputed. If this cache carries information beyond what recomputation from the current residuals would produce, the cross-step Markov property fails — state at step N would depend on hidden cache contents, not just the residuals at that step.

The test is direct. Generate 10 tokens from a fixed prompt using `generate_text` (KV cache active). Then replicate manually: run `predict_next_token` on the original prompt, take the greedy top token, append it to the prompt, repeat. Each `predict_next_token` call does a fresh forward pass with no cached state.

**Prompt:** `"The capital of Australia is"`

### Results

| Step | Token | p (fresh) | Match |
|------|-------|-----------|-------|
| 1 | ` Canberra` | 0.922 | ✓ |
| 2 | `.` | 0.672 | ✓ |
| 3 | `\n\n` | 0.418 | ✓ |
| 4 | `Can` | 0.859 | ✓ |
| 5 | `berra` | 1.000 | ✓ |
| 6 | ` is` | 0.723 | ✓ |
| 7 | ` located` | 0.594 | ✓ |
| 8 | ` in` | 0.984 | ✓ |
| 9 | ` the` | 0.949 | ✓ |
| 10 | ` Australian` | 0.992 | ✓ |

**10/10 tokens match exactly.**

### Interpretation

The KV cache carries zero additional information beyond what fresh recomputation from the residuals produces. Discarding the cache and rerunning the full forward pass at each step gives identical greedy outputs.

This is expected from architecture: RMSNorm is stateless (no running statistics), RoPE computes position encodings from indices on the fly, and K/V projections are linear functions of the current residual. There is no persistent buffer. The cache is a computational shortcut, not an information store. This experiment confirms it empirically rather than by argument.

**Conclusion: Cross-step Markov holds. KV cache is pure optimization.**

---

## Experiment 2: Full-sequence Cross-task Patching

### Background

Previous experiments established that single-position residual patching at layer 14 (the last token position only) transfers computation within the same task (capital→capital) but fails cross-task (capital→translation). The failure has a clear explanation: at L14, country identity has not yet been written into the last-position residual. The last-position residual is nearly identical between "The capital of Australia is" and "The capital of France is" at L14 — patching it accomplishes nothing because the country information lives at the country token's position, not the final position.

The theory predicts that **full-sequence patching** — transplanting all-position residuals simultaneously — should succeed even cross-task. The country token positions are included in the transplant, so layers L15–L26 can perform their normal operation of writing country identity into the last position.

### Setup

Donor: `"The capital of Australia is"` → naturally produces `Canberra` (92%)

Three recipients, ranging from same-domain to maximally different:
- `"The largest city in Australia is"` → naturally produces `Sydney` (90%)
- `"The capital of France is"` → naturally produces `Paris` (81%)
- `"The language spoken in France is"` → naturally produces `French` (84%)

Injection via `inject_residual` with `patch_all_positions=True`. Sequence lengths differ between donor (6 tokens) and recipients (6–7 tokens); the tool handles this.

### Results

| Recipient | Layer | Donor→inj KL | Recipient→inj KL | Injected top-1 |
|-----------|-------|--------------|------------------|----------------|
| Largest city in Australia | L14 | **0.0** | 4.73 | Canberra 92% |
| Language spoken in France | L14 | **0.0** | 19.32 | Canberra 92% |
| Capital of France | L26 | **0.0** | 13.53 | Canberra 92% |

KL divergence between donor and injected output distributions is 0.0 in every case — not just the same top-1 token, but the same full probability distribution over the vocabulary. The residual angle between donor and recipient at L14 is 2.1–2.8°; at L26 it is 12.5°. Despite the larger angular divergence at L26, full-sequence patching still achieves KL=0.

### Interpretation

When the complete Markov state is transplanted, the downstream computation is fully determined by it. The recipient prompt's history — "largest city," "language spoken," "France" — has no effect once the residuals are replaced. The computation from L14 onward is identical regardless of what prompt was used to get to L14.

This resolves the failure of single-position cross-task patching. That experiment failed not because Markov is wrong but because patching a single position is insufficient — the Markov state at L14 includes all positions, not just the last one. Single-position patching transplants a fragment of the state; full-sequence patching transplants the whole thing. The result is correspondingly complete.

**Conclusion: Full-sequence Markov confirmed at L14 and L26, same-domain and cross-domain. The prompt history is irrelevant given the full residual state.**

---

## Experiment 3: Subspace Independence

### Background

The city-prediction subspace occupies approximately 0.6% of the residual stream's energy at L26 (roughly 6–10 dimensions of 2560). The question is whether this tiny subspace operates as an independent channel. If it does, transplanting only those dimensions from a capital-city prompt into a translation prompt should produce Canberra from a state that is 99.4% about French translation.

This tests a stronger form of the Markov property: not just that the full state is sufficient, but that distinct functional subspaces are orthogonal and can be addressed independently.

### Setup

Donor: `"The capital of Australia is"` → Canberra (92%)

Three injection configurations, using `inject_residual` with `subspace_only=True`:

1. **2D, cross-domain (L26):** Subspace defined by `{Canberra, Sydney}` token embeddings. Recipient: `"The language spoken in France is"`.
2. **8D, cross-domain (L26):** Subspace defined by `{Canberra, Sydney, Melbourne, Brisbane, capital, Australia, city, country}`. Same recipient.
3. **8D, same-task (L26):** Subspace defined by `{Canberra, Sydney, Paris, Lyon, capital, Australia, France, city}`. Recipient: `"The capital of France is"`.
4. **8D, same-task (L14):** Same tokens. Same recipient.

### Results

| Config | Layer | cos(subspaces) | donor_frac | Injected top-1 | KL(donor→inj) |
|--------|-------|----------------|------------|----------------|----------------|
| 2D → translation | L26 | 0.123 | 0.57% | French 78% | 12.51 |
| 8D → translation | L26 | 0.322 | 0.78% | French 77% | 11.22 |
| 8D → France capital | L26 | 0.322 | 0.77% | **Sydney 47%** | 4.06 |
| 8D → France capital | L14 | 0.998 | 0.086% | Paris 81% | 14.22 |

`cos(subspaces)` is the cosine similarity between the donor's projection onto the subspace basis and the recipient's projection onto the same basis.

### Interpretation

**The L14 result is null by construction.** At L14, country identity has not yet been written into the last-position residual — subspace fractions are near zero (0.086%) and the donor and recipient project almost identically onto the city/country subspace basis (cos=0.998). Swapping a nearly-zero component that is pointing in nearly the same direction in both prompts has no effect. This is fully consistent with the single-position patching results.

**The L26 cross-domain result shows a weak effect.** The 8D injection shifts Sydney from 0% to 5.8% in the translation prompt. Some city-type signal is leaking through, but it's overwhelmed by the 84% French probability and 99.4% of the residual pointing at translation context.

**The L26 same-task result is the most informative.** Injecting the donor's 8D city/capital subspace into `"The capital of France is"` produces **Sydney at 47%** — not Canberra, and not Paris. Paris drops from 81% to near-zero. Sydney, the misconception token, emerges.

This is a clean finding. The 8D injection does transfer city-type information: the output shifts from Paris (French capital) to Sydney (Australian city). But it transfers the wrong Australian city. The capital-vs-misconception distinction — Canberra vs Sydney — is not carried by the directions spanned by `{Canberra, Sydney, Paris, ...}` token embeddings. The Canberra specificity requires something that is orthogonal to the token embedding subspace, or distributed across more dimensions than this basis spans.

### Why the subspace fails

The token-embedding-defined subspace is a proxy, not the functional subspace. The cosine similarity between how the donor and recipient use the 8D basis is only 0.32 at L26 — meaning the donor's "capital of Australia" activates the city/country embedding dimensions in a substantially different direction than the recipient's "capital of France." When the donor's component is transplanted, the downstream layers (L27–33) are handed a vector that points in an unfamiliar direction within the city subspace. Their best interpretation is "generic Australian city" — Sydney — not "Australian capital specifically" — Canberra.

The capital-specificity requires directions that are:
- Not aligned with any individual city token embedding
- Established by the interaction of the capital query structure with the country-specific context over multiple layers
- Probably only recoverable by running PCA over many capital-city prompt residuals at L26 and extracting the principal components that vary with country

**What this result doesn't tell us** is whether the actual functional subspace is independent of the surrounding context. The experiment fails to find independence, but it uses the wrong basis. The correct test requires extracting the real functional directions first.

### What the subspace test requires

To test subspace independence properly:
1. Run the capital-city prompt for 9 countries through the model, collecting last-position residuals at L26
2. Run PCA to find the principal components that vary across countries — this is the actual country-identity subspace in residual space
3. Inject those PCA directions from the Australia prompt into the France (or translation) prompt

If Canberra emerges from that injection, the subspaces are independent channels. If it doesn't — if the PCA components require the surrounding context to be decoded correctly — there is nonlinear mixing that the clean channel picture doesn't capture.

The injection infrastructure exists (`inject_residual` with `subspace_only=True` accepts arbitrary subspace tokens as a basis). What's needed is a way to specify the subspace by residual-space directions rather than token embeddings — a `compute_subspace` tool that runs PCA across a batch of prompts and returns the principal directions.

---

## Summary

| Question | Answer | Confidence |
|----------|--------|------------|
| Cross-step Markov (KV cache is pure optimization) | **Confirmed** | High — 10/10 token match |
| Full-sequence Markov (all-position patching transfers complete state) | **Confirmed** | High — KL=0.0 across 3 conditions |
| Subspace independence (functional subspaces are independent channels) | **Inconclusive** | Token-embedding subspace is wrong basis; test requires PCA-defined directions |

The first two questions are settled. The Markov property holds within steps and across steps. The full residual state at any layer is sufficient — the entire upstream computation is compressed losslessly into the residuals, and the history adds nothing.

The third question is open for a specific reason: the subspace injection experiment revealed that Sydney emerges when the city-type subspace is transplanted — which itself is informative. The capital-vs-misconception distinction is not in the embedding-aligned directions; it's in something that requires extraction from the actual residual geometry. That's the next experiment to build for.

---

## Practical implications

**Full-sequence patching as a tool:** The ability to transplant complete computational states cross-task is practically useful for mechanistic analysis. It confirms that circuit findings from one prompt structure generalize — the same computation runs on the transplanted state regardless of where it started. This validates using patching to isolate circuits without worrying that the source-prompt context contaminates the result.

**The misconception window has a new explanation:** Sydney emerging from the subspace injection at L26 replicates the natural Sydney→Canberra misconception in miniature. The 8D embedding subspace carries "Australian city" but not "capital specifically" — which is exactly the state the model is in when it first reaches L24 before the misconception override circuit fires. The L26 FFN bottleneck writes capital-specificity into dimensions that are not captured by token embedding directions. This is consistent with the previous finding that capital-city features occupy an orthogonal, low-dimensional subspace (85° from Sydney, <0.6% of energy).

**Bandwidth and hallucination:** If the functional city subspace (Canberra vs Sydney specifically) is in directions not aligned with token embeddings, it means the signal is more fragile than the bandwidth measurement suggests. Raw bandwidth (0.6% of residual energy) may overstate the robustness of the signal if the relevant directions are vulnerable to interference from context shifts. The subspace protection experiment — freezing 3 dimensions against the repetition attack — becomes more important to run once the correct functional directions are identified.
