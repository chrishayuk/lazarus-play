# Unified Stratigraphy Report: Gemma 3-4B-IT Translation (EN→FR)

**Model:** `google/gemma-3-4b-it` · 34 layers · 2560 hidden dim · 8 heads · 1.34B params · vocab 262K
**Prompt:** `"Translate ... English: The sky is blue.\nFrench:"`
**Output:** ` Le ciel est bleu.` · ` Le` predicted at **99.2% confidence**

---

## Method 1 — Logit Lens

**Result: Complete failure across all 34 layers.**

` Le` never appears in the top-5 predictions at any layer. The logit lens shows noise tokens (multilingual fragments, code artifacts, random subwords) throughout:
- Layers 0–3: `aseti`, `eterm`, `感想`, `格` — random multilingual garbage
- Layers 10–14: Romanian `către`, Cyrillic `поведение` — geographically adjacent to French?
- Layers 23–29: Telugu `నీ`, `eligible`, `Jakub` — no French signal
- Layer 33 final: `emb` (not ` Le`)

**Interpretation:** This is an architectural finding. Gemma 3's 262K vocabulary combined with its specific normalization scheme (likely RMSNorm applied before the unembedding, but not the same norm used by the logit lens projection) makes intermediate hidden states not cleanly decodable through the standard unembedding matrix. The model computes in a high-dimensional space that only becomes readable at the actual output head. The logit lens is blind here.

**Methodological consequence:** We cannot use logit lens to determine *when* ` Le` emerges in the forward pass. We must rely on other methods.

---

## Method 2 — Probe Stratigraphy

**Language identity is readable from the very first layer — but blurs dramatically in the middle.**

| Layer band | Val accuracy | Interpretation |
|---|---|---|
| Layer 0 | 92% | Language in raw embeddings |
| Layer 1 | **100%** | Peak early — perfectly separable |
| Layers 2–16 | **83–88%** | **Language blur zone** — middle layers abstract away language-specific features |
| Layer 17+ | **100%** | Stable consolidation — language identity re-established |

**Key finding:** There are two phases of language encoding — an early embedding phase (layers 0–1) where language is trivially readable because different languages use different vocabularies, and a late consolidated phase (17+) where the model has fully committed to producing French output. The dip in between (83% at layers 2–16) is not noise — it reflects the model genuinely blurring language identity while performing semantic abstraction.

**Crossover layer:** The probe identifies layer 1 as peak early, but the *stable* crossover (where language identity is locked in for the final computation) is layer 17.

---

## Method 3 — Causal Stratigraphy

**One layer is causally indispensable. The other 33 are not.**

| Layer | Effect after ablation | Status |
|---|---|---|
| Layer 2 | 0.043 (from baseline 0.992) | **CRITICAL — destroys prediction** |
| All other layers | ~0.992 (unchanged) | Non-critical |

Ablating layer 2 drops ` Le` probability from **99.2% → 4.3%**. No other layer comes close. This is a clean single-point-of-failure structure.

**Paradox:** Layer 2 is critical, yet the logit attribution shows layer 2 has a *negative* net contribution to ` Le` (cumulative logit drops from 14.4 to 8.9 after layer 2). How can a layer be causally necessary but logit-negative?

**Resolution:** Layer 2 is doing **critical information routing, not direct promotion**. It structures the residual stream in a way that all downstream computation depends on. Ablating it doesn't just remove its own contribution — it collapses the information scaffold that layers 3–33 build on. Causal necessity ≠ direct promotion.

---

## Method 4 — Direction Stratigraphy

**Both methods agree: language direction is *cleaner* in early layers than late ones.**

| Layer | diff_means sep. | diff_means acc. | LDA sep. | LDA acc. |
|---|---|---|---|---|
| Layer 1 | **3.43** | 95.8% | **3.76** | 91.7% |
| Layer 2 | 2.87 | 87.5% | 3.55 | 91.7% |
| Layer 19 | 1.73 | 83.3% | 1.45 | 83.3% |

**Surprise:** Late-layer linear separability is *worse* than early-layer, despite the probe reaching 100% at layer 19. The LDA vector norm tells the story:
- Layer 1 LDA: `||v|| = 0.755`
- Layer 2 LDA: `||v|| = 0.651`
- Layer 19 LDA: `||v|| = 0.0078` — **nearly degenerate**

At layer 19, French and English are so distributed across the 2560-dimensional space that no single linear direction captures the distinction well. The probe succeeds at 100% because it's a *trained* linear classifier (optimally oriented), but an axis-aligned direction from diff_means or LDA becomes nearly meaningless. **Late-layer language identity is non-linearly, holographically distributed.** It's everywhere and nowhere in particular.

Both methods **strongly agree** on the trend: early layers have geometrically cleaner language separation. Late layers have functionally perfect language representation but geometrically diffuse.

---

## Method 5 — Neuron Stratigraphy

**Layer 2 (causal bottleneck):** 20 discriminative neurons identified. Top neuron 1633 (sep=2.62) suppresses French. Neuron 605 is extreme: English mean=36.8, French mean=−3.25 — a neuron that fires ~40× higher for English text. Neuron 889 inverts: French mean=4.4, English mean=0.26.

**Layer 19 (late plateau):** Overlapping set — neurons 1633 and 889 reappear, but with ~20× larger magnitude and altered polarity. The same neurons carry language information across depths but at different scales and orientations.

**For ` Le` prediction specifically (top_neurons DLA):**

| Layer | Top neuron contribution | Notable neuron | Total MLP→` Le` |
|---|---|---|---|
| Layer 2 | 0.063 | N5245 (`top_token=" bonjour"`) | **+0.42** |
| Layer 19 | 0.015 | N747 (abstract) | +0.14 |

Layer 2 has neuron 5245 with `top_token=" bonjour"` — a **French greeting neuron** that directly promotes ` Le`. It's the only semantically interpretable French neuron in the top 20. But its contribution (0.020) is tiny.

**Key finding:** The MLP at layers 2 and 19 contribute almost nothing directly to the ` Le` logit. The massive FFN contributions seen in logit attribution come from **late layers 27–33** — not from the middle layers where language identity is established. There are two separate computations: *language identity representation* (middle layers, MLPs) and *actual token prediction* (final layers 27–33, dominant FFN).

---

## Method 6 — Decomposition Stratigraphy

**A clean regime transition at layer 12: from attention-dominated routing to FFN-dominated knowledge retrieval.**

### Attention vs FFN dominance by layer band:

| Band | Layers | Dominant | Attention % | Total norm range |
|---|---|---|---|---|
| Early | 0–11 | **Attention** | 65–90% | 87–4064 |
| Transition | 12 | FFN | 64% FFN | 4000 |
| Middle | 13–18 | Mixed | ~50/50 | 1672–2864 |
| Late-early | 19–24 | **FFN** | 29–52% FFN | 2224–4832 |
| Late-late | 25–33 | **FFN strongly** | 15–32% attn | 5248–**17152** |

The residual stream norm explodes 22× from layer 0 (780) to layer 33 (17,152). The final FFN at layer 33 has norm 17,792 — larger than the entire residual stream. It's doing the last massive computation.

### Logit Attribution — The Full Story of ` Le`:

The embedding contributes **+59.25** raw logit for ` Le`. The colon token at end of `French:` is in a context where the embedding strongly primes French-article-like outputs. Then the network does complex work:

```
Embedding:  +59.25  ← Context already primes French article
Layer 0:    −47.0   ← Attention aggressively suppresses naive prediction
Layers 1–13:  −9.5  ← Continued suppression (processing input)
Layers 14–24: +9.4  ← Gradual recovery begins; language committed (layer 17+)
Layer 27:   +2.4    ← FFN begins amplifying
Layer 28:   +2.3    ← FFN continues
Layer 29:   +6.5    ← Major FFN push ←─ top_token first points to " sky" (content!)
Layer 30:   +1.5    ← FFN top_token first becomes " Le"!
Layer 31:   +0.25
Layer 32:   +0.875
Layer 33:   +9.1    ← LARGEST single contribution; FFN top_token " Le"
Final:      +32.75  (= 99.2% probability)

Net attention: −90.4  ← Attention hurts " Le" across ALL 34 layers
Net FFN:       +63.9  ← FFN helps " Le" across all layers
```

**Grand finding from logit attribution:** The network is waging an internal war. Attention systematically suppresses the naive embedding prediction throughout the entire forward pass. FFN systematically promotes it. The FFN wins — but only barely (embedding + FFN > attention losses), and only because of the massive final kick from layers 29 and 33.

Layer 30 is when the FFN's `top_token` first becomes ` Le` — this is where the decision crystallizes in logit space.

---

## Unified Layer-Band Analysis

### Early Layers (0–4): The Input Reader / Language Eraser

| Method | What it shows |
|---|---|
| Logit lens | Noise — uninterpretable |
| Probe | Language 92–100% decodable |
| Causal trace | Layer 2 is the single critical node |
| Direction | Strongest linear language separation (sep ~3.5–3.8) |
| Neurons | Moderate separation, neuron 605 is extreme English detector |
| Decomposition | Attention dominant (67–88%); layer 2 spike in total norm (334) |
| Logit attr. | Layer 0 attention DESTROYS ` Le` logit (−66.2); net: −47.0 |

**Summary:** Early layers read the input and set up the information scaffold. Language is trivially decodable from embeddings (different languages = different tokens). Layer 2's attention does the most important structural work in the network — not by promoting ` Le` but by routing information that all later computation depends on. The embedding's naive French-article prior gets aggressively suppressed.

### Middle Layers (5–16): The Semantic Abstractor / Language Blur Zone

| Method | What it shows |
|---|---|
| Logit lens | Noise continues |
| Probe | 83–88% — **lowest accuracy** — language identity blurred |
| Causal trace | Non-critical (ablating any single layer: no effect) |
| Direction | N/A (not tested here) |
| Neurons | Moderate discrimination, decreasing separation |
| Decomposition | Transition at 12 to FFN dominance; layer 12 biggest early FFN |
| Logit attr. | Hovering near 0 cumulative logit (1–5); small negative attention, tiny positive FFN |

**Summary:** The model is doing abstract semantic processing. Language-specific features dissolve into a more language-neutral representation of *meaning*. No single layer matters causally. The logit for ` Le` is at its lowest — the model has suppressed its naive prediction and hasn't yet committed to the new one. The transition at layer 12 (first strong FFN dominance) marks the beginning of the "knowledge application" phase.

### Late Layers (17–33): The Language Re-Encoder / Decision Crystallizer

| Method | What it shows |
|---|---|
| Logit lens | Still noise... until layer 30 FFN's `top_token` = ` Le` |
| Probe | Stable 100% — language identity locked in |
| Causal trace | Non-critical individually (but collectively doing all the logit work) |
| Direction | Worst linear separability (sep ~1.5) — holographically distributed |
| Neurons | Large-magnitude activations; same neurons as early but 20× bigger |
| Decomposition | FFN increasingly dominant (85% at layer 33); norm explosion |
| Logit attr. | Steady build (+0.3 to +2.6 per layer); layer 29 (+6.5) and layer 33 (+9.1) decisive |

**Summary:** The model re-establishes language identity as a stable commitment, then spends the final layers converting that commitment into a specific token prediction. FFN layers 29 and 33 are the "output writers" — they add the biggest direct contributions to ` Le`. Attention remains net-negative throughout (still suppressing competitors), while FFN accumulates the winning logit.

---

## Agreements Between Methods

1. **Early language encoding is strong.** Probes (100% at layer 1) and direction stratigraphy (sep 3.8 at layer 1) both agree: language is most geometrically accessible in early layers.

2. **Middle layers blur language.** Probes (83–84% at layers 2–16) and direction stratigraphy (declining separation) both confirm the blur zone.

3. **Late layers consolidate.** Probes (100% stable at 17+) and logit attribution (building ` Le` logit from layer 17 onward) both confirm late-layer commitment.

4. **FFN dominates the final prediction.** Decomposition (85% FFN at layer 33) and logit attribution (FFN top contributor at layers 29, 33) agree.

---

## Disagreements and Surprises

1. **Logit lens fails completely.** Every other method reveals clear structure; logit lens is blind. This is a Gemma 3 architectural property that should be documented.

2. **Causal critical layer (2) is logit-negative.** The most causally important layer *hurts* ` Le` prediction in direct logit attribution. These methods measure complementary things: causal trace measures *structural necessity*, logit attribution measures *direct promotion*. They disagree about what "important" means.

3. **Direction separability DECREASES with depth** despite probe accuracy increasing to 100%. Late-layer language information is non-linearly distributed — better represented but less geometrically accessible in a single direction. The LDA vector at layer 19 is nearly degenerate (`||v||=0.008`).

4. **The answer emerges from the embedding, not the computation.** The colon in `French:` carries an embedding logit of +59.25 for ` Le` — already knowing the answer. But the network then spends 33 layers suppressing that answer (attention: net −90.4) before the FFN wins and re-establishes it (+63.9). This is profound: the model "knows" the answer from context but must process through it fully before committing.

5. **Neuron 5245 with `top_token=" bonjour"`** is a semantically clean French activation neuron visible at layer 2 — but its logit contribution is tiny (0.020). The actual prediction is driven by a distributed, diffuse FFN computation in late layers, not by a single "French neuron."

---

## Central Question: Do Representations Lead Predictions?

**Yes — by approximately 16 layers.**

Language identity is established in representations at layer 1 (probe: 100%). The direct logit prediction for ` Le` only begins meaningfully accumulating at layer 17. The representation "knows" French 16 layers before the prediction "says" French.

The middle layers (2–16) are a decoupled zone: language identity is present in representations but not yet driving the output. The model uses this space for semantic processing that is language-agnostic. The translation computation itself — the cross-lingual meaning mapping — appears to happen in this blur zone, invisible to both logit lens and probe.
