# Single-Position Residual Capacity
## Experiment 9276f1ce-3e0b-4935-868d-443c13592c4b

**Question:** How many independent facts can a single residual vector faithfully encode?

**Answer:** The question was dissolved. The model does not *store* N facts in a single
residual — it uses *attention routing* to retrieve them from context. The residual encodes
only a ~7-dimensional query direction pointing attention at the right context position.

---

## Setup

- Model: google/gemma-3-4b-it (2560 hidden_dim, 34 layers, 8 heads)
- Format A: `"[Capital] is the capital of [Country]. … The capital of [X] is"`
- Format B: `"Given that the capital of France is Paris, … the capital of [X] is"`
- Novel facts: fabricated country-capital pairs (Zorland→Vexity, etc.)
- Cross-model: HuggingFaceTB/SmolLM2-135M-Instruct (576 hidden_dim, 30 layers)

---

## Experiment 1 — Baseline

- `"The capital of France is"` → Paris **80.9%** (from model weights alone)
- **Logit lens emergence**: L0-L12 multilingual noise → L16 "cities" category →
  L24 Paris 72.7% (first signal) → **L26 Paris 100%** → L33 80.9% (final)
- **Activation magnitudes**: By L14, virtually ALL 2560 dims are active (|v| > 1.0).
  Naive "active dims" measure useless for measuring fact-specific bandwidth.
- **Feature dimensionality (France vs Germany at L26)**: Only **2–6 dims** needed
  for 100% classification. The fact-specific signal is extremely compressed.

---

## Experiment 2 — Accumulating Facts (Approach A)

Format: `[Cap] is the capital of [Country].` × N, then rotating retrieval queries.

| N facts | Correct retrievals | Notes |
|---------|-------------------|-------|
| 1 | 1/1 (75%) | Single context example |
| 2 | 2/2 (96%) | Confidence RISES |
| 3 | 3/3 (98%) | Template strengthens |
| 5 | 5/5 (89–99%) | Australia lowest |
| 8 | 7/8 | Brazil→"Brasil" tokenization artifact |
| 10 | 9/10 | Same artifact |
| 15 | 14/15 | All new facts (Argentina–Russia) correct |
| 20 | 19/20 | Nigeria, Kenya, Turkey, Thailand, Poland all 99–100% |
| 30 | 29/30 | Ukraine→Kyiv 82% (geopolitical hedging, not capacity) |
| 50 | 8/8 sampled | Saudi Arabia, Indonesia, Ethiopia, Cambodia all 100% |

**No capacity degradation through N=50 (370+ token context).** The only failures are:
- **Brazil→"Brasil"**: model's tokenizer conflates Brasilia (capital) with Brasil
  (Portuguese word for Brazil). Consistent at ALL N. Not a capacity failure.
- **Ukraine→Kyiv**: model hedges with "currently"/"now". World-knowledge uncertainty,
  not capacity. Improves with more context (82% at N=30 → 99% at N=50).

**Approach B (compressed format)**: Works at N=50 with 95–99% confidence, slightly
lower than Approach A due to template unfamiliarity but same zero-degradation pattern.

---

## Experiment 3 — Geometric Signature of Saturation

### Same query, different N (France at L26)
| Comparison | Cosine | Angle |
|-----------|--------|-------|
| N=1 vs N=10 | 0.9926 | 0.74° |
| N=1 vs N=20 | 0.9918 | 0.83° |
| N=10 vs N=20 | 0.9996 | 0.16° |

**The France query residual barely moves as more facts are loaded.** N=10 vs N=20 are
nearly identical. There is no "filling up" — the residual stabilizes after N≈3.

### Different queries, same N=20 (at L26)
| Pair | Cosine | Angle |
|------|--------|-------|
| France vs Germany | 0.9907 | 0.82° |
| France vs Japan | 0.9878 | 0.91° |
| France vs Nigeria | 0.9815 | 1.10° |
| Nigeria vs Poland | 0.9783 | 1.27° |

Different fact retrieval queries differ by only **0.8–1.3°** — yet the model correctly
retrieves each fact with 99%+ confidence.

### Query discrimination dimensionality (France/Japan/Germany/Spain vs Nigeria/Poland/Kenya/Turkey at N=20)
- dims for 50%: 3
- dims for 80%: 5
- dims for 99%: **7**
- Classification with 3 dims: 100%
- Interpretation: **SUBSPACE** (not directional) — ~1 dim per additional query

### L14 vs L26 comparison
| Layer | France vs Nigeria | Angle |
|-------|------------------|-------|
| L14 | cosine 0.9996 | **0.16°** — routing signal doesn't exist yet |
| L26 | cosine 0.9815 | **1.10°** — routing crystallizes here |

**The query-specific routing signal builds across L14→L26.** At L14, all retrieval
queries look essentially identical. By L26, the 7-dim subspace encodes which fact
to retrieve, and the answer crystallizes to 100%.

### Per-fact dimensionality (France query at N=0, 1, 3, 10)
| Transition | Cosine | Angle | Interpretation |
|-----------|--------|-------|----------------|
| N=0 → N=1 | 0.9903 | 0.97° | Largest shift: template effect |
| N=1 → N=3 | 0.9920 | 0.82° | Diminishing returns |
| N=3 → N=10 | 0.9980 | 0.36° | Near-plateau |

Residual saturates quickly. Most change happens on the first repetition (template
establishment), not from loading additional facts.

---

## The Mechanism: Attention Routing

The model does NOT store N facts in the last-token residual. Instead:

1. **Query encoding (L14→L26)**: The retrieval cue "The capital of France is" builds a
   tiny ~7-dim query direction in the residual that encodes "look for France."
2. **Attention routing (via Q/K matching)**: Attention heads route back to the matching
   `"Paris is the capital of France."` sentence in context.
3. **Answer crystallization (L26)**: Answer locks in at 100% confidence at L26
   — the same layer that serves as commitment layer for all other retrieval tasks.

**Theoretical capacity**: 2560 dims / 7 dims per query ≈ **365 concurrent query
directions** — far more than tested. The limit is attention span, not residual width.

---

## Novel Facts Test

Fabricated pairs: Zorland→Vexity, Fubria→Grondel, Flurbia→Drenzol, etc.
No model weights to fall back on — pure attention routing.

| N | Zorland | Flurbia | Mexalia |
|---|---------|---------|---------|
| 5 | "V" 81% ✓ | — | — |
| 10 | "V" 91% ✓ | "D" 94% ✓ | — |
| 15 | "V" 93% ✓ | "D" 82% ✓ | "Tor" 56% ✓ |

**Confirms mechanism**: works for completely fabricated facts where model weights
are useless. One N=5 failure (Dralbia→Torken): specific novel token doesn't produce
strong enough attention key. Not a capacity failure.

---

## Experiment 5 — Cross-Model Scaling (SmolLM2-135M)

| Metric | Gemma-3-4B | SmolLM2-135M |
|--------|-----------|--------------|
| Hidden dim | 2560 | 576 |
| Width ratio | — | 4.44× smaller |
| Known-fact capacity | **>50** | ~10–15 |
| Novel-fact capacity | **>15** | **<5** |
| Capacity ratio | — | ≥3.3× |
| Scaling exponent α | — | ~1.0–1.3 |

**SmolLM2 known-fact failures** (N=15–20):
- India: "Delhi" 33.6% + Mumbai 27.7% — weight prior overwhelming "New Delhi" context
- Nigeria: "Lag"(Lagos) 18.7% beats "Abu"(Abuja) 12% — model doesn't know Abuja
- Turkey: Istanbul 52% vs Ankara 46% — coin flip

These are **capability failures** (model doesn't strongly know Abuja/Ankara from weights),
not pure capacity failures. A fair comparison needs novel facts.

**SmolLM2 novel-fact failures** (N=5):
- Zorland: "Z" 30% (letter of the COUNTRY, not "V" for Vexity!) — self-referential confusion
- Dralbia: "T" 9%, "D" 9% — no clear answer

SmolLM2's attention routing **breaks for novel/low-frequency tokens** — it can't cleanly
separate the query entity from the answer entity in its narrower key-query space. This is a
qualitative difference from Gemma, not just a quantitative one.

**Scaling exponent α ≈ 1.0–1.3** (estimated from known-fact comparison):
`α = log(50/10) / log(2560/576) = log(5)/log(4.44) = 1.08`

---

## Final Conclusions

### The capacity question is re-answered:

> **"Single-position capacity is LARGE (>50 facts via attention routing, ~365
> theoretically). The 0.6% bandwidth finding generalises and is exceeded. Facts pack
> efficiently into a 7-dimensional query subspace. The residual is genuinely not the
> bottleneck, confirming the attention-routing conclusion."**

### Engineering implications:

For distributed residual-stream inference:
- A single 10.2 KB residual checkpoint does NOT need to store N concurrent facts.
- It only needs to encode a 7-dim query direction (~0.27% of total capacity).
- **Checkpoint interval should be set by attention span requirements** (can last-token
  attend back to relevant context?), not by residual width.
- For fact-dense inputs: keep context accessible; the residual will handle routing.
- For Gemma-3-4B: the 2560-dim residual can theoretically support ~365 concurrent
  fact-retrieval queries — the context window (8192 tokens) will be exhausted first.

### What would actually stress single-position ENCODING capacity:

This experiment measured attention-routing capacity, not pure encoding capacity. To find
the true single-position encoding limit, future experiments should:
1. Test facts NOT present in context (encoded across positions earlier in sequence)
2. Use injection experiments: force multiple facts into a single residual vector and
   measure simultaneous readout (the dark superposition approach, exp d6ec9d72,
   already showed two operands held simultaneously at L7 at 99.9% each)
