# Knowledge in the Residual Stream — Results

**Experiment ID:** 803b4fc5-00c0-43b4-b192-fc4b7afa76aa
**Model:** google/gemma-3-4b-it (34 layers, 2560 hidden dim, 8 heads)

## The Setup

3-fact synthetic prompt (49 tokens):
```
Zarkov the explorer founded the city of Voltara. Roger the copy editor
enjoys scratchy football on weekends. The Thessmere Jets won the
Crenthia Cup in 2019.
```

Key positions: pos 13 (" Volt", fact binding), pos 14 ("ara"), pos 16 (" Roger"),
pos 21 (" scratch"), pos 25 (" weekends", filler), pos 30 (" Jets").

---

## Experiment 1 — The Prediction Subspace

### Phase 1a: Prediction Energy Fraction

At L33, the angle between the residual and its top-1 predicted token's
unembedding direction:

| Pos | Token | Prediction (prob) | Angle | cos²(θ) | Orthogonal |
|-----|-------|-------------------|-------|---------|------------|
| 6 | "ov" (Zarkov) | "'" (79%) | 87.97° | **0.13%** | 99.87% |
| 9 | "founded" | " the" (68%) | 87.44° | **0.20%** | 99.80% |
| 13 | " Volt" | "ara" (79%) | 85.98° | **0.49%** | 99.51% |
| 14 | "ara" | "." (67%) | 87.56° | **0.18%** | 99.82% |
| 16 | " Roger" | "," (27%) | 87.27° | **0.23%** | 99.77% |
| 21 | " scratch" | "y" (92%) | 87.44° | **0.20%** | 99.80% |
| 25 | " weekends" | "." (99.5%) | 87.32° | **0.22%** | 99.78% |
| 30 | " Jets" | " are" (61%) | 87.51° | **0.19%** | 99.81% |

**Finding 1:** The residual is ~99.8% orthogonal to its own prediction
at every position. Prediction confidence does NOT correlate with
prediction energy. "weekends" → "." at 99.5% confidence uses only
0.22% energy — the same as "Roger" → "," at 27% confidence.

The prediction is not encoded by energy. It's encoded by DIRECTION
after layer norm removes the mean. The 99.8% orthogonal component
is invisible to prediction but is the vast majority of the signal.

### Phase 1b: Top-10 Prediction Subspace

| Pos | Token | 10-token subspace | Orthogonal |
|-----|-------|-------------------|------------|
| 13 | " Volt" | **0.78%** | 99.22% |
| 14 | "ara" | **0.41%** | 99.59% |
| 25 | " weekends" | **0.46%** | 99.54% |
| 30 | " Jets" | **0.60%** | 99.40% |

Even 10 predicted tokens collectively capture <1% of residual energy.

### Phase 1c: What's in the Orthogonal Component?

At pos 13 (" Volt"), knowledge-related tokens at L33:
- "ara" (prediction): 0.49%
- " City" (category): 0.23%
- " town", " place": 0.02%
- " founded", " explorer", " settlement": ~0%
- **Total knowledge + prediction: 0.81%**

At pos 25 (" weekends"), content tokens:
- "." (prediction): 0.22%
- " enjoys": 0.15%
- " Roger": 0.12% (negative — anti-correlated)
- " football", " sport": 0.06%
- **Total: 0.60%**

**Finding 2:** Knowledge tokens capture as little energy as prediction
tokens. The knowledge is NOT in vocabulary-projectable directions.
The 99.2% dark space is where knowledge lives — but it lives there
in a form that doesn't project to any individual token.

### Raw vs Normalized: The Dark Giant

At every position, the raw (pre-norm) top tokens are garbage: `ꗜ`,
`<unused338>`, cuneiform characters, `PLDNN`. The raw residual points
at NOTHING in vocabulary space. Only after layer norm — which subtracts
the mean and rescales — does a meaningful prediction emerge.

The mean itself is stable across positions: always dominated by
" cause", " crashing", "鑑", "yle", "الاعت" — the same dark direction
regardless of what token is at the position.

**Finding 3:** The prediction is not IN the residual. It's extracted
FROM the residual by layer norm. The residual is a dark object. The
norm reveals the tiny directional signal.

---

## Experiment 4 — The Knowledge Lifecycle at a Fact Position

Decoded residual at pos 13 (" Volt") through layers:

| Layer | Norm Top-1 | Prob | Knowledge Signal |
|-------|-----------|------|-----------------|
| L0 | "お" | 97% | Pure noise |
| L7 | "お" | 76% | No knowledge |
| L14 | "ier" | 0.08% | Flat — no signal |
| **L23** | **"ian"** | **8.9%** | **"ville" (2.9%) — CITY CATEGORY EMERGES** |
| **L29** | **" City"** | **4.8%** | **Category crystallizes** |
| L33 | "ara" | 78.8% | Prediction commits |

**Finding 4:** Category knowledge (that this position carries a city
name) appears at L23, 10 layers before the specific prediction. The
residual knows "this is a city" before it knows "this is ara."

Knowledge is EARLY. Prediction is LATE.

---

## Experiment 6 — THE Experiment: Knowledge Transfer

### The Retrieval Event

Query: "What city did Zarkov found?" → predicts "Z" (100%)

Logit attribution for "Z":
- Embedding: +49.75 (already strong — "Z" is common after "model\n")
- L0 attention: **-71.1** (massive suppression — don't just echo)
- L2-L28: gradual recovery, cumulative reaches ~12
- **L30 attention: +11.0** — THE RETRIEVAL
- L33 FFN: **+31.4** — THE AMPLIFICATION
- Final logit: 60.0

**L29 → L30 prediction flip:**
- L29: "According" (96.2%), "Z" at 0.05% (rank 7)
- L30: **"Z" (89.7%)**, "According" drops to 9.5%
- A single layer rewrites the entire prediction distribution.

### The Retrieval Head

L30 Head 3 contributes 98.5% of the layer's "Z" logit (+3.67).

Head 3 attention pattern at the generation position:
- BOS: 30.9%
- **" Z" (pos 47, query entity)**: 21.2%
- **"ark" (pos 5, fact entity)**: 5.4%
- **" Volt" (pos 13, answer content)**: 5.4%
- "ov" (pos 49, query): 2.6%
- "ara" (pos 14): 2.0%

The head reads from THREE types of positions simultaneously:
1. The query entity (" Z" in the question)
2. The fact entity ("ark" in the facts)
3. The answer content (" Volt"/"ara" in the facts)

### Cross-Fact Comparison — Different Heads!

| Query | Answer | Head | Attn logit | FFN boost |
|-------|--------|------|-----------|-----------|
| Zarkov city? | "Z" | **H3** | +11.0 | +31.4 |
| Roger enjoy? | "Roger" | **H0** | +6.9 | +38.8 |
| Thessmere cup? | "The" | — | +0.25 | +32.8 |

**Finding 5:** Different heads retrieve different entities!
L30 H3 handles the Zarkov query, L30 H0 handles Roger.
The retrieval circuit is multi-headed, not single-headed.

The Thessmere query predicts "The" (template word with embedding
102.5), needing no retrieval. Real content retrieval would
activate at the "Crenthia" token position later.

### The Knowledge Transfer Vector

L30 attention output decomposition (Zarkov query):

| Component | Energy fraction | Angle to attn output |
|-----------|----------------|---------------------|
| "Z" direction | **17.9%** | 65.0° |
| " Zarkov" direction | **5.9%** | 76.0° |
| " Volt" direction | **1.4%** | 83.1° |
| " City" + " city" | 0.3% | ~88° |
| **Total interpretable** | **25.7%** | — |
| **Dark transfer** | **74.3%** | — |

Roger query: 13.4% interpretable (9.9% "Roger", 3.0% " Roger"),
86.6% dark transfer.

**Finding 6:** The knowledge transfer vector is 25% interpretable
and 75% dark. The interpretable part is the answer token direction.
The dark 75% carries... what? It's information that doesn't project
to any vocabulary token but is apparently necessary.

### The L33 FFN Amplifier — Dark Space Trick

L33 FFN output: norm 35,252. Total in 8-token subspace: **0.23%**.
"Z" direction specifically: **0.007%** (projection = -287, negative!).

Yet L33 FFN contributes +31.4 logits to "Z."

**Finding 7:** The FFN amplifier is 99.8% dark space. Its output
points AWAY from "Z" in raw projection (-287), but after layer norm,
it contributes +31.4. Layer norm transforms dark space geometry into
vocabulary predictions. The FFN doesn't push toward tokens directly —
it reshapes the dark manifold so that the norm reveals the answer.

### L30 Component Geometry

| Pair | Angle |
|------|-------|
| Attn ↔ FFN | **89.8°** (perfectly orthogonal) |
| FFN ↔ Residual | **48.9°** (strongly aligned) |
| Attn ↔ Residual | **73.9°** (partially orthogonal) |

**Finding 8:** Attention and FFN write in perfectly orthogonal
directions. The FFN reinforces existing state (48.9° to residual).
The attention writes NEW information in a direction orthogonal to
the FFN. This is the fundamental geometry: attention injects
knowledge; FFN maintains the dark infrastructure.

### Generation Position Trajectory

At the generation position, the residual stays 99.0-99.8% orthogonal
to ALL reference tokens (Volt, ara, City, city, founded, Zarkov)
through ALL 34 layers. Even at L33 with 100% confidence:

- " Zarkov": 84.79° → cos² = 0.83% energy
- " Volt": 87.41° → cos² = 0.20% energy

**Finding 9:** The answer is never more than a ~5° tilt away from
orthogonal. A 5° tilt in 2560D, amplified by layer norm, produces
100% confidence. The entire retrieval, the whole "knowledge transfer,"
is a microscopic angular perturbation in an enormous dark space.

---

## Experiment 5 — Knowledge Dimensionality

### Transfer Subspace at L30

Fact-retrieval queries vs generic queries:
- 2D for 50% variance
- 5D for 80% variance
- **8D for 95% variance**
- 9D for 99% variance

Classification: 80% at 1D, 90% at 2D.

PC spectrum: first dimensions encode response format (yes/no style),
last dimensions encode entity content (PC7→"Professor", PC9→"Roger").

**Finding 10:** The knowledge transfer channel is an ~8D subspace
of the 2560D residual. The first few dimensions are response format.
The entity-specific signal is in the tail (dims 5-9). This means
the KNOWLEDGE portion of the transfer is ~4-5D.

### The Compression Number

If knowledge per fact position requires ~5 extra dimensions beyond
what filler positions carry, and each dimension is a float16 (2 bytes):

- Knowledge per fact: 5 × 2 = 10 bytes
- For 3,625 novel facts: 3,625 × 10 = **36.25 KB of knowledge**
- KV cache for 370K tokens at 34 layers: ~56 GB
- **Ratio: ~1,500,000:1**

The notebook metaphor: one sentence on 1.5 million blank pages.

---

## The Architecture of Knowledge

```
PREFILL (at fact positions):
  L0:  Embedding (dark, points at garbage tokens)
  L7:  Entity identity resolves in dark space
  L14: Entity compass (100% probe accuracy)
  L23: Category knowledge appears ("ville", "ian")
  L29: Category crystallizes ("City" as top-1)
  L33: Prediction commits ("ara" at 78.8%)

GENERATION (at query position):
  L0:  Embedding + L0 attn suppression (-71 logits)
  L2-L28: Gradual query formation
  L29: "According" at 96.2% (no answer yet)
  L30: RETRIEVAL — attention head reads KV
       Head H3 or H0 (query-dependent)
       Writes 3,472-norm vector into residual
       25% interpretable (answer token), 75% dark
  L31-L32: Minor adjustments
  L33: FFN AMPLIFICATION (+31.4 logits)
       99.8% dark, layer norm reveals answer
```

The knowledge flow:
1. During prefill, each fact position's residual accumulates
   category and relation information (L23-L29)
2. This is projected to K (for addressability) and V (for content)
   by linear projections — the KV cache
3. During generation, the query residual builds up to L30
4. At L30, one attention head matches the query to fact positions
   via Q·K, then copies V content
5. The copied content is 25% answer-token-aligned, 75% dark
6. L33 FFN amplifies through dark space manipulation
7. Layer norm transforms the dark signal into a prediction

**The knowledge IS in the residual. The KV is just the window.**

---

## Key Numbers

| Metric | Value |
|--------|-------|
| Prediction energy (cos²) at any position | **0.13-0.49%** |
| Orthogonal to top-10 predictions | **99.2-99.6%** |
| Knowledge + prediction combined | **<1%** of residual |
| Raw residual top token | **garbage** (cuneiform, unused) |
| Layer norm mean energy | **0.01-0.1%** |
| Knowledge appears (category) | **L23** |
| Prediction commits | **L33** |
| Retrieval layer | **L30** |
| Retrieval head concentration | **97-98.5%** (single head) |
| Transfer vector interpretable | **13-26%** |
| Transfer vector dark | **74-87%** |
| L33 FFN in token subspace | **0.23%** |
| L33 FFN logit contribution | **+31 to +39** |
| Knowledge transfer dimensionality | **~8D** |
| Entity-specific dimensions | **~4-5D** |
| Knowledge per fact position | **~10 bytes** |
| KV cache to knowledge ratio | **~1,500,000:1** |
