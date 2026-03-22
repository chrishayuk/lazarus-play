# The Embedding War: A Complete Mechanistic Picture
## Gemma 3-4B-IT — Logit Attribution Across Prediction Types

---

## Experiment Summary

Logit attribution (`normalized=true`) run across 10 prediction instances on Gemma 3-4B-IT
(34 layers, 2560 hidden dim, 8 attention heads). Covers correct confident facts, three types
of hallucination, and an ambiguous completion.

The central finding: the "embedding war" is one of at least three mechanistically distinct
prediction regimes. And hallucination itself splits into three subtypes with completely
different internal signatures — only one of which is detectable from output probabilities.

---

## I. The Full Data Table

| Prompt → Target | Category | Embed | Attn | FFN | Final | Prob | \|Attn\|/FFN |
|---|---|--:|--:|--:|--:|--:|--:|
| capital of France → **Paris** | high-conf correct | +22.5 | −34.6 | +22.5 | +21.4 | 80.9% | 1.54 |
| Four score…our → **fathers** | high-conf correct | +47.8 | −59.0 | +28.9 | +32.0 | 97.7% | 2.04 |
| Marie Curie…city of → **Warsaw** | high-conf correct | −16.4 | +8.7 | +41.1 | +27.0 | 98.4% | 0.21 |
| capital of Australia → **Canberra** | high-conf correct | −5.4 | −6.1 | +33.9 | +21.6 | 84.4% | 0.18 |
| capital of Australia → **Sydney** | wrong (not predicted) | +22.6 | −15.4 | +7.5 | +18.0 | 2.2% | 2.06 |
| Thanatos Syndrome is → **Robert** | hallucination type 1 | +47.0 | −69.7 | +29.5 | +16.4 | 7.7% | 2.36 |
| 2009 Nobel Physics → **Andre** | hallucination type 2 | +14.2 | −33.7 | +51.1 | +37.8 | 94.1% | **0.66** |
| "Correction" 1975 → **Toni** | hallucination type 3 | ≈0 | +13.1 | −6.3 | +21.0 | 40.8% | n/a† |
| She opened door and → **stepped** | ambiguous/creative | +6.0 | −19.9 | +23.8 | +20.9 | 61.3% | 0.84 |

*Attn and FFN are net sums across sampled layers (~18 of 34). †Type 3 ratio inverted: FFN negative.*

---

## II. Three Prediction Mechanisms

### Mechanism 1 — The Embedding War
**Tokens:** Paris, fathers, Sydney (wrong), Robert (uncertain hallucination)
**Embedding logit:** +14 to +48

High-frequency tokens in this syntactic context carry a strong embedding prior. Attention
suppresses it; FFN partially recovers. The |Attn|/FFN ratio is the war balance.

### Mechanism 2 — Constructive Retrieval
**Tokens:** Warsaw, Canberra
**Embedding logit:** −16 to −5

Rare proper nouns have no default affinity. Attention and FFN *cooperate* to build the
prediction from scratch. Warsaw: net attention is **positive** (+8.7) — actively routing
toward the correct answer. This is the mechanistic signature of genuine knowledge retrieval.

### Mechanism 3 — Creative/Contextual Generation
**Token:** stepped
**Embedding logit:** +6

FFN dominates (+23.8 vs −19.9), constructing the continuation from narrative patterns.

---

## III. Three Types of Hallucination

This is the most important finding. Hallucination is not one thing mechanistically.

### Type 1 — Uncertain Hallucination (Diffuse Distribution)
**Example:** "The Thanatos Syndrome" → " Robert" (7.7%)

The model doesn't know the answer. Attention suppresses all candidate tokens uniformly
because nothing specific can be routed to. FFN supplies only generic signal ("this is an
author name"). The result is a diffuse distribution over plausible-sounding tokens (Robert
7.7%, Stephen 6.0%, James 4.6%...). Greedy decode picks the mode arbitrarily.

```
Structure: High embed (+47) → massive attn suppression (−69.7) → moderate FFN (+29.5)
|Attn|/FFN = 2.36  (highest in dataset)
Detection: trivial — top token probability < 10%, flat distribution
```

### Type 2 — Confident Confabulation (Wrong Fact, Stored with Certainty)
**Example:** "2009 Nobel Prize in Physics" → " Andre" (94.1%)

The model has stored a specific, wrong fact: Andre Geim = Nobel Physics winner. (He won in
2010, not 2009; the 2009 prize went to Charles Kao, Willard Boyle, and George Smith.)
The mechanism is **structurally identical to a correct high-confidence prediction**:

```
Structure: Moderate embed (+14.2) → attn suppression (−33.7) → strong FFN (+51.1)
|Attn|/FFN = 0.66  (same range as Warsaw, Canberra — "looks correct")
Key tell: layer 33 FFN spike = +20.1, top token " Andre"
Detection: IMPOSSIBLE from output probability alone (94.1% confidence)
```

The model retrieved a real fact (Geim won Nobel Physics) but with an off-by-one year error.
These year-shift confabulations are particularly dangerous because the model is genuinely
confident and the retrieved person is plausibly correct.

### Type 3 — Attention-Driven Error (Wrong Association)
**Example:** "Correction" published 1975 → " Toni" (40.8%)

Thomas Bernhard wrote *Korrektur* (Correction) in 1975. The model produces Toni Morrison.
The mechanism is unique: **attention is net-positive while FFN is net-negative.**

```
Structure: Near-zero embed (≈0) → attention PROMOTES (+13.1) → FFN opposes (−6.3)
Dominant component: ATTENTION (first time in dataset)
Detection: moderate confidence (40.8%), attention dominance is the structural flag
```

The attention mechanism latches onto a wrong association: "literary novel, 1975, female
author" → Toni Morrison (won Nobel 1993, wrote *Beloved* 1987). The FFN weakly opposes
this but is overruled. At layer 33, even the FFN capitulates and pushes " Toni" (+7.4).

| Type | Embed | Attn | FFN | Prob | Detection |
|---|---|---|---|---|---|
| 1 — Uncertain | High | Strongly neg | Moderate pos | **< 10%** | Easy (flat distribution) |
| 2 — Confabulation | Moderate | Neg | **Strongly pos** | **> 90%** | Impossible without attribution |
| 3 — Attention error | Zero | **Pos** | Neg | **~40%** | Attention-dominance flag |

---

## IV. The Misconception Race — Sydney vs. Canberra

Attributing both Sydney and Canberra simultaneously on the same prompt reveals the model
thinking the wrong answer, then correcting itself:

```
Layer:      embed    L16     L22     L24     L26     L28     L32     L33 (final)
Canberra:   −5.4     3.3     9.4    20.6   [36.0]   35.0    18.8    21.6
Sydney:    +22.6     3.4    12.4    23.0   [36.0]   32.8    17.6    18.0
                                             ↑ TIE at layer 26 ↑
                             Sydney leads           Canberra wins
```

Sydney's embedding advantage (+28 points) sustains its lead through layers 16–26.
At layer 26, both tokens reach exactly +36.0 — literally tied.
Layers 28–33 are the fact-checking phase. The final arbiter is **layer 33, head 5**,
whose own top token is " Canberra" (contribution: +2.97, concentration 228%).

The model thinks Sydney from layers 16–26, then self-corrects via a single head in layer 33.

---

## V. Key Layer Roles

| Layer range | Role |
|---|---|
| **0** | Embedding compression. All 8 heads negative for high-embed tokens. For rare tokens, immediate positive contributions. The "prior rejection" layer. |
| **2–14** | Semantic confusion / context integration. Cumulative logits low and noisy. |
| **16–22** | Common associative knowledge activates. Sydney, Toni Morrison, generic names. |
| **24–26** | **World knowledge retrieval.** Specific facts: Warsaw (head 1, " Polish"), Canberra (attn 24 + FFN 26). Neuron 3593 [intermediate dim] fires for both. |
| **28–33** | **Fact-checking override.** Corrects misconceptions. Layer 33 head 5 = Canberra arbiter. For confabulations (Nobel), this is where wrong facts are *confidently asserted*: FFN spike +20.1 at layer 33. |

---

## VI. Capital Neurons Are Frame Detectors, Not Knowledge Detectors

`discover_neurons` at layer 26 found neurons 1948, 1197, 445 as top discriminators between
"capital of X is" prompts vs. other prompts. But per-prompt activation analysis reveals
they detect **syntactic frame**, not the semantic concept of "capital city":

| Prompt | Neuron 1948 | Neuron 1197 | Neuron 445 |
|---|--:|--:|--:|
| The capital of France is | −63.3 | +55.5 | +50.0 |
| The capital of Japan is | −71.5 | +52.3 | +56.0 |
| The capital of Germany is | −59.0 | +46.5 | +50.3 |
| The capital city of Australia is | −55.8 | +33.8 | +48.8 |
| The capital of Canada is | −60.3 | +79.0 | +69.5 |
| The capital of Brazil is | −69.5 | +67.0 | +64.0 |
| **Marie Curie was born in the city of** | −21.9 | **−59.0** | **−11.9** |
| The author of Hamlet was | +14.4 | −2.4 | −3.2 |
| She opened the door and | +25.5 | +1.2 | +13.7 |
| Thanatos Syndrome is | +9.0 | +15.6 | +15.8 |
| 2009 Nobel Prize in Physics | +29.6 | −26.6 | **−31.8** |

**Critical result:** The Marie Curie → Warsaw prompt (which predicts a capital city correctly
at 98.4%) has neuron 1197 = −59, strongly anti-capital. These neurons are responding to the
prompt *asking for a capital*, not to the concept that the answer is a capital.

Real capital-city knowledge is encoded in the attention patterns (head 1, layer 24 for Warsaw)
and in distributed FFN activations, not in these frame-detector neurons.

---

## VII. The War Balance Ratio as a Confidence Signal

```
Ratio < 1.0  →  No war; FFN or attention dominant
               Warsaw 0.21, Canberra 0.18, stepped 0.84
               Also: Nobel confabulation 0.66 ← DANGER ZONE (indistinguishable)

Ratio 1.0–2.0  →  Moderate war; correct or borderline
               Paris 1.54, fathers 2.04

Ratio > 2.0  →  Heavy war; uncertain hallucination
               Sydney 2.06 (rejected), Robert 2.36 (uncertain hallucination)
```

The ratio distinguishes uncertain hallucination from correct predictions, but **fails
completely** for confident confabulation (type 2, ratio 0.66) — which is structurally
identical to Warsaw-style genuine retrieval.

---

## VIII. Practical Implications for Hallucination Detection

Three detection strategies, matched to hallucination type:

**Type 1 (Uncertain) — already solved:**
Top-token probability < 15% on a factual question template. The flat distribution is
detectable without any mechanistic analysis.

**Type 2 (Confabulation) — mechanistically hard:**
The model is 94% confident. The only structural tell is a *late-layer FFN spike* (layer 33,
+20 logit) on a prompt where the fact involves similar entities across time (Nobel years,
Olympic venues, election results). Cross-validation heuristic: ask the adjacent question
("who won in 2010?") and check if the model gives the same name.

**Type 3 (Attention error) — mechanistically detectable:**
Attention-dominant predictions on factual question frames are suspect. If `total_attention >
total_ffn` on a "The [PROPERTY] of [ENTITY] was" prompt structure, flag for review. The FFN
opposing attention on a factual question is a weak suppression signal.

**General insight:**
The FFN stores *facts*. Attention *routes* to them. When facts are well-stored (Warsaw,
Canberra, Andre-for-2010), both components cooperate or FFN dominates constructively.
When facts are absent (Thanatos), attention's suppression creates a diffuse distribution.
When facts are mis-stored (Nobel year confusion), FFN retrieves confidently but wrongly.
When attention misfires (Correction/Morrison), it routes to the wrong associative cluster
and FFN can't fully stop it.

---

*Experiment run on Gemma 3-4B-IT using the Lazarus MCP interpretability server.*
*logit_attribution normalized=true, head_attribution and discover_neurons/analyze_neuron at selected layers.*
*Note: top_neurons uses intermediate MLP dim (10240); analyze_neuron uses hidden dim (2560) — different index spaces.*
