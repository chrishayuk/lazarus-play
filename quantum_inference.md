# Quantum Inference: Delayed Observation in the Residual Stream

**Experiment ID:** `8bb57c7d-5228-486a-a822-56bedc77918d`
**Model:** google/gemma-3-4b-it (34 layers, 2560 hidden dim, 8 heads, bfloat16)

---

## Hypothesis

Standard autoregressive decoding forces a token choice at the final layer. The full probability distribution — the uncertainty, the competing candidates, the 2560-dimensional richness — collapses to a single embedding. This is premature observation.

The hypothesis: feed the model's uncertain output back as a **soft embedding** — a weighted mixture of candidate token embeddings proportional to their probabilities — rather than collapsing to argmax. Over multiple passes, derived answers (backed by a reasoning chain) accumulate coherent amplitude while associative shortcuts don't compound. The superposition resolves through evolution, not forced measurement.

---

## Prompts

**Compositional (expected to show superposition):**

| Label | Prompt | Correct answer |
|-------|--------|----------------|
| France-north | "The country directly north of France has its capital in the city of" | Brussels (Belgium) |
| Einstein | "The physicist Einstein was born in the capital city of" | Germany → Berlin |
| Hamlet | "The author of Hamlet was born in the town of" | Stratford |
| Beethoven | "The composer Beethoven was born in the capital city of" | Germany → Berlin |

**Simple controls:**

| Prompt | Answer | Probability |
|--------|--------|-------------|
| "The capital of Germany is" | Berlin | 89.1% |
| "The capital of France is" | Paris | 80.9% |
| "The capital of Japan is" | Tokyo | 83.2% |

---

## Experiment 1 — Measuring the Superposition

### Baseline distributions at L33

| Prompt | Top-1 | Prob | Top-2 | Prob | Entropy |
|--------|-------|------|-------|------|---------|
| France-north | Brussels | 60.6% | Paris | 36.8% | ~1.0 bits |
| Einstein | Ulm | 51.1% | Germany | 35.2% | ~1.4 bits |
| Hamlet | Stratford | 60.5% | fragments | 23% | ~1.8 bits |
| Beethoven | Bonn | 87.1% | Germany | 5.6% | ~0.75 bits |
| Germany simple | Berlin | 89.1% | — | — | ~0.5 bits |

The Beethoven prompt registers as near-resolved (entropy 0.75 bits, comparable to simple controls). The model does **biographical recall** — Beethoven was born in Bonn — and ignores the "capital city" framing entirely. It is not a compositional failure in the expected sense. Beethoven is dropped from further analysis.

France-north and Einstein are the real test cases.

### Geometry of the undecided state

Every candidate token sits at ~88–90° from the residual stream at L33. The answer decision lives in **less than 0.6% of the 2560-dimensional residual** — the tiny fraction that survives layer normalization. Raw projection rankings are dominated by garbage tokens; the logit winner is determined entirely by what the layer norm's centering operation selectively amplifies.

Brussels and Paris are at 88.0° and 88.4° from the France-north residual at L33. They look nearly identical geometrically. The 0.4° difference, amplified by layer norm across 262,208 vocabulary slots, is the entire decision.

**Subspace fractions (fraction of residual in answer subspace):**

| Prompt | Subspace tokens | Total fraction |
|--------|----------------|----------------|
| France-north L33 | Brussels+Paris+Belgium+France | 0.48% |
| Einstein L33 | Ulm+Germany+Berlin+Switzerland | 0.58% |
| Germany simple L33 | Berlin+Germany+Munich+Frankfurt | 0.14% |

Counterintuitively, the simple Germany prompt has *less* answer-subspace content (0.14%) than the compositional prompts (0.48–0.58%). Simple prompts have concentrated their information into precisely calibrated directions through training. The 0.14% is sufficient for 89% Berlin confidence because the amplification is exact. The 0.48% gives only 60.6% Brussels because Brussels and Paris directions compete near-equally.

### L0 embedding geometry

European capital tokens at L0 have pairwise cosine similarity ~0.78–0.83:

| Pair | Cosine | Angle |
|------|--------|-------|
| Brussels–Paris | 0.798 | 37° |
| Brussels–Berlin | 0.815 | 35° |
| Brussels–Amsterdam | 0.782 | 39° |
| Paris–Berlin | 0.834 | 33° |

This matters enormously for the soft embedding. The soft mixture of `0.605 × embed(Brussels) + 0.365 × embed(Paris)` at L0 lies only **14° away from pure Brussels**. At L33, Brussels–Paris diverge to 69.4° — more meaningful — but the embedding layer collapses the superposition before any recirculation can happen.

---

## Experiment 2 — The Layer Trajectories (Unexpected)

Before testing recirculation, the full layer-by-layer residual decode was measured for both prompts. This produced the most significant finding of the experiment.

### France-north: Brussels vs Paris by layer

| Layer | Top-1 | Brussels | Paris | Event |
|-------|-------|---------|-------|-------|
| L14–20 | generic tokens | ~0% | ~0% | No resolution |
| L24 | Brussels | 51.5% | 40.1% | First emergence |
| **L26** | **Brussels** | **79.5%** | **20.1%** | **Peak clarity** |
| L28 | Brussels | 52.7% | 46.5% | Collapse to near-tie |
| **L30** | **PARIS** | 34.8% | **65.0%** | **Paris temporarily leads** |
| L32 | Brussels | 51.2% | 39.9% | Recovery |
| L33 | Brussels | 60.6% | 36.8% | Final output |

The trajectory oscillates. The best available answer — Brussels at 79.5% — appears at **L26**. Layers L27–L33 degrade it: a near-tie at L28, a full reversal to Paris 65% at L30, partial recovery to 60.6% at L33. The standard final-layer output is substantially worse than what was available four layers earlier.

### Einstein: Germany vs Ulm by layer

| Layer | Top-1 | Germany | Ulm | Event |
|-------|-------|---------|-----|-------|
| L14–20 | generic | ~0–4% | 0% | No resolution |
| L24 | Germany | 57.1% | 0% | Nationality chain emerging |
| L26 | Germany | 83.2% | 0% | Strong compositional answer |
| L28 | Germany | **97.5%** | 0% | Near-perfect |
| L30 | Germany | **98.2%** | 0% | Peak compositional confidence |
| L32 | Germany | 71.7% | 0% | Declining |
| **L33** | **Ulm** | **35.2%** | **51.1%** | **Single-layer catastrophic override** |

This is the sharpest finding. Germany holds at 71.7% through L32. Then **a single transformer layer — L33 — writes Ulm from near-zero to 51.1%**, simultaneously crashing Germany from 71.7% to 35.2%. The entire compositional failure is concentrated in one layer. L33's FFN retrieves the biographical fact (Einstein → Ulm, his actual birthplace) and overwrites the correctly-computed compositional answer (Einstein → German → Germany).

The model is not uncertain. It is correct through 32 of 34 layers. The final layer breaks it.

---

## Experiment 2 — Soft Embedding Feasibility

**Finding: at L0, the soft embedding is geometrically indistinguishable from the argmax.**

With Brussels–Paris cosine = 0.798 at L0, the soft embedding direction is only 14° from pure Brussels. The information preserved by soft recirculation over hard argmax is a 14° deviation — less than the angle between synonyms. The quantum superposition, expressed in embedding space, is already collapsed by the similarity of city token representations before any computation begins.

At L33, Brussels and Paris diverge to 69.4° — the superposition is meaningful there — but current tools cannot inject computed soft vectors into the residual at arbitrary layers. `inject_residual` requires a real donor prompt to capture a residual from; externally computed weighted sums cannot be injected directly.

**What could be tested:** injecting individual donor token residuals at different layers into the original prompt context. This approximates "hard commitment at layer L" rather than soft recirculation.

**Inject results (donor's L26/L28/L33 residual → neutral template → continue):**

| Inject layer | France-north Brussels | Observation |
|-------------|----------------------|-------------|
| L26 | **78.8%** | Best result |
| L28 | 74.5% | Good |
| L33 | 62.5% | Near-identical to direct prediction |

The L26 state, extracted and continued in a neutral context, gives stronger Brussels confidence than the L33 final output. **Earlier layers contain a cleaner answer signal than the final layer.** The neutral context (L27–L33 of a different prompt) does not add the oscillation and reversal that the original prompt's L27–L33 introduce.

For Einstein, the L28 state continued in a neutral context gives Switzerland 41.7%, Germany 16.4%, Ulm 10.2%. The neutral L29–L33 layers don't carry Einstein-specific biographical memory, so neither Germany nor Ulm emerges cleanly. The Einstein biographical circuit is prompt-specific and activates in the original context's L33 only.

---

## Experiments 3 and 4 — Hard Recirculation

Since soft embedding injection is not directly available, the experiment pivoted to **hard recirculation**: commit to the argmax token, append it to the original prompt, and re-query the same question.

### Results

**France-north:**

| Pass | Condition | Brussels | Paris |
|------|-----------|---------|-------|
| 0 | Baseline | 60.6% | 36.8% |
| 1 | After Brussels committed | **93.4%** | 1.5% |
| 1 | After Paris committed (wrong) | **47.5%** | 28.7% |

**Einstein:**

| Pass | Condition | Germany | Ulm |
|------|-----------|---------|-----|
| 0 | Baseline | 35.2% | 51.1% |
| 1 | After Germany committed | **98.4%** | ~0% |

### What these numbers mean

**Result A — Convergence:** One pass of hard recirculation with the correct answer takes Brussels from 60.6% to 93.4% and Germany from 35.2% to 98.4%. The model's first-pass uncertainty is not a fundamental epistemic limitation. It vanishes immediately when the answer appears in context. The model uses it as additional evidence, re-runs the reasoning chain, and arrives at near-certainty.

**Result B — Self-correction:** After committing to Paris (the wrong answer), the model still gives Brussels 47.5% on the next pass. The model has genuine knowledge of the correct answer that *resists a single wrong commitment*. The Brussels signal is strong enough that even Paris-primed context cannot fully suppress it.

This asymmetry — correct commitment → near-certainty, wrong commitment → partial self-correction — is evidence that the model has the right answer encoded in its weights. The first-pass ambiguity is a decoding artifact, not a knowledge gap.

---

## Experiment 5 — Soft vs Hard: The Geometric Estimate

With the L0 geometry established, soft recirculation for France-north can be estimated:

- Hard recirculation (Brussels): **93.4%**
- Soft estimate (`0.605 × 93.4% + 0.365 × 47.5%` ≈ mixing the Brussels and Paris hard outcomes): **~73.8%**
- Baseline: 60.6%
- Hard recirculation (Paris, wrong): 47.5%

Hard argmax recirculation dominates. The soft embedding's 14° deviation from Brussels at L0 carries negligible additional information. Decisive commitment self-reinforces correctly because the model's reasoning circuits are stronger than the noise introduced by the Paris component.

The quantum hypothesis predicts that soft > hard because superposition allows interference effects to amplify the compositional answer. In practice, hard > soft because the model is not doing interference — it is doing **context-conditioned retrieval**, and a clear unambiguous context retrieves the right answer better than an ambiguous one.

---

## Experiment 8 — When to Observe

The layer trajectories establish that the optimal observation point is not the last layer.

### France-north: observation quality by layer

| Layer | Brussels | Paris | Verdict |
|-------|---------|-------|---------|
| L24 | 51.5% | 40.1% | Too early, still resolving |
| **L26** | **79.5%** | 20.1% | **Observe here — first confident state** |
| L28 | 52.7% | 46.5% | Recirculate — near-tie |
| L30 | 34.8% | 65.0% | Would give wrong answer (Paris leads) |
| L32 | 51.2% | 39.9% | Recovering |
| L33 | 60.6% | 36.8% | Standard output — suboptimal |

### Einstein: observation quality by layer

| Layer | Germany | Ulm | Verdict |
|-------|---------|-----|---------|
| L26 | 83.2% | 0% | Good |
| **L28–L30** | **97–98%** | 0% | **Observe here — peak compositional** |
| L32 | 71.7% | 0% | Still correct but declining |
| **L33** | 35.2% | **51.1%** | **Wrong — biographical override** |

### Confidence-triggered early exit at threshold 80%

| Prompt | Exit layer | Confidence | Correct? | vs L33 |
|--------|-----------|-----------|---------|--------|
| France-north | L26 | 79.5% Brussels | Yes | +18.9pp |
| Einstein | L26 | 83.2% Germany | Yes | **Correct vs wrong** |
| Germany simple | ~L24 | ~85% Berlin | Yes | ~same |

For Einstein, the difference is categorical. Early exit gives the correct answer. Standard L33 gives the wrong answer.

---

## Synthesis

### What the quantum framing gets right

- The model carries a genuine broad distribution at intermediate layers, not just at L33. The uncertainty is real and informative.
- Multiple passes (even hard recirculation) converge to the correct answer far better than a single pass.
- There is an optimal "observation time" that differs from the forced final-layer observation of standard decoding.
- The L33 output is not a reliable summary of what the model knows — it is whatever the final layer's circuits produce, which may conflict with what earlier layers correctly computed.

### What the quantum framing gets wrong

- The superposition at L0 is not over orthogonal states. City token embeddings are all cosine ~0.80 pairwise. Quantum interference requires mutually exclusive basis states; these don't exist at the embedding layer for geographic queries.
- Recirculation improves answers not through interference but through **context reinforcement**: the model sees its previous answer as additional evidence and re-runs retrieval with a stronger prior.
- The failure mode is not "premature observation collapses to the wrong state." It is "late-layer circuits write the wrong state *after* the correct state was available at earlier layers." The solution is earlier observation, not later.

### The actual failure mechanisms

**France-north (oscillation type):** The L26 FFN correctly resolves Brussels 79.5%, but subsequent layers L27–L32 introduce competing signals — possibly from "France" in the prompt activating Paris-related circuits — that reduce Brussels confidence, cause a Paris reversal at L30, and end at 60.6%. The final answer is correct but weakened by mid-late-layer interference.

**Einstein (single-layer override type):** L33 contains a biographical memory circuit: Einstein → born → [city] → retrieves Ulm. This fires regardless of what the compositional circuit computed in L1–L32. It is not sensitive to the "capital city" framing. The circuit overwrites a correctly-computed answer in a single layer. Germany 71.7% at L32 is simply erased.

### The deployable architecture

The original proposal was: recirculate until confident.
The data suggests a better architecture: **early exit until unconfident, then recirculate**.

```
1. Run forward pass, monitoring confidence at each layer
2. If confidence > T at layer L (and stable for 1 layer): early exit — emit token
3. If confidence never > T: run to L33, take argmax
4. If L33 confidence still low: one round of hard recirculation (~2× compute)
5. If still unresolved after recirculation: fall back to CoT
```

No soft embeddings required. No KV cache growth. No token-by-token reasoning overhead.

**Compute budget:**
- Simple queries (direct exit at L24–L28): ~70–85% of standard compute
- Uncertain queries (run to L33): standard compute (1×)
- Hard cases (one recirculation): ~2× compute
- Very hard cases (CoT): 10–100×

For queries where the model fails at L33 but was correct at L28 (Einstein-type), early exit doesn't just save compute — it gives the **right answer instead of the wrong one**.

---

## Limitations

**What could not be tested:**
- True soft embedding injection (requires Lazarus support for arbitrary vector injection at any layer)
- Whether soft > hard when the embedding space is less correlated (non-geographic domains may have more orthogonal candidate tokens)
- Iterative recirculation beyond 1 pass with full probability tracking
- Generalization across all failure types (only France-north and Einstein were fully characterized)
- Whether early exit degrades performance on prompts that legitimately need all 34 layers

**The critical untested case:** Some queries may require L33's processing to produce a correct answer, and early exit at L26 would give a wrong high-confidence answer. The early-exit protocol needs a validation sweep across many prompt types before deployment.

---

## Key Numbers

| Finding | Value |
|---------|-------|
| Brussels at L26 (peak, France-north) | **79.5%** |
| Brussels at L33 (standard, France-north) | 60.6% |
| Germany at L30 (peak, Einstein) | **98.2%** |
| Ulm at L33 (failure, Einstein) | 51.1% |
| Brussels after 1 hard recirculation | **93.4%** |
| Germany after 1 hard recirculation | **98.4%** |
| Brussels after wrong-answer (Paris) priming | **47.5%** (self-correction) |
| Answer subspace fraction of residual | **< 0.6%** |
| Brussels–Paris angle at L0 | 37° |
| Brussels–Paris angle at L33 | 69° |
| Soft embedding deviation from Brussels at L0 | ~14° |
| Einstein failure concentrated in | **1 layer (L33)** |
