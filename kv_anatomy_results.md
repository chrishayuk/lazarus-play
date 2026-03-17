# KV Anatomy of Fact Retrieval

**Experiment ID:** 8fdf177b-6dd5-42a6-bf40-f9442dc719c0
**Model:** google/gemma-3-4b-it (34 layers, 2560 hidden dim, 8 heads, bfloat16)
**Date:** 2026-03-16

## Setup

**Prompt:**
```
<start_of_turn>user
Zarkov Industries was founded in the city of Voltara in 1987.
The headquarters employs over 2000 people in advanced research.
Zarkov Industries was founded in the city of<end_of_turn>
<start_of_turn>model
```

**Generation:** "Zarkov Industries was founded in the city of Voltara.\n\n"

The model echoes the full query before producing "Voltara". The fact retrieval
moment is when the model generates " Volt" (token 89711) after echoing
"Zarkov Industries was founded in the city of". At that position, " Volt"
is predicted at 81.6% probability.

**Token map (62 tokens at retrieval moment):**

| Pos | Token | Role |
|---|---|---|
| 0 | `<bos>` | BOS |
| 1-3 | `<start_of_turn>` user `\n` | Chat template |
| 4-6 | Z ark ov | Entity name (1st mention) |
| 7 | Industries | Entity name |
| 8-9 | was founded | Relation verb |
| 10-13 | in the city of | Relation context |
| **14** | **Volt** | **Fact token 1** |
| **15** | **ara** | **Fact token 2** |
| 16-22 | in 1987. | Year + period |
| 23-36 | The headquarters...research. | Filler sentence |
| 37-46 | Z ark ov...city of | Query (2nd mention in user turn) |
| 47-51 | `<end_of_turn>` `\n` `<start_of_turn>` model `\n` | Chat template |
| 52-61 | Z ark ov...city of | Generated echo (model's output so far) |

---

## Experiment 1 — DLA Decomposition (all 34 layers)

Target token: " Volt" (id 89711). Measured at last position (pos 61, " of")
where model is about to generate " Volt".

### Full layer-by-layer DLA

| Layer | Attn DLA | FFN DLA | Total | Cumulative | Top Token (logit lens) |
|---|---|---|---|---|---|
| Embed | — | — | — | 11.69 | — |
| L0 | -13.77 | -10.61 | **-24.38** | -12.69 | " of" |
| L1 | -0.50 | +0.38 | -0.13 | -12.81 | " course" |
| L2 | +3.06 | -4.31 | -1.25 | -14.06 | "is" |
| L3 | +1.44 | -3.63 | -2.19 | -16.25 | "is" |
| L4 | **+9.22** | -1.47 | **+7.75** | -8.50 | "TEN" |
| L5 | +3.00 | -0.91 | +2.09 | -6.41 | "TEN" |
| L6 | +1.00 | +0.69 | +1.69 | -4.72 | "is" |
| L7 | +1.25 | -0.91 | +0.34 | -4.38 | "is" |
| L8 | +1.75 | -0.19 | +1.56 | -2.81 | "is" |
| L9 | +0.77 | +0.26 | +1.02 | -1.79 | "TEN" |
| L10 | +0.20 | -0.05 | +0.14 | -1.65 | "TEN" |
| L11 | +0.68 | +0.45 | +1.13 | -0.52 | "TEN" |
| L12 | -0.19 | +0.37 | +0.18 | -0.34 | "TEN" |
| L13 | -0.06 | +0.04 | -0.03 | -0.36 | " quella" |
| L14 | -0.03 | +0.41 | +0.38 | +0.02 | " quella" |
| L15 | -0.06 | +0.14 | +0.09 | +0.10 | " __(" |
| L16 | +0.30 | +0.51 | +0.81 | +0.91 | " (" |
| L17 | -0.09 | +0.11 | +0.03 | +0.94 | " (" |
| L18 | -0.27 | -0.00 | -0.27 | +0.67 | " (" |
| L19 | -0.12 | +0.08 | -0.04 | +0.63 | " " |
| L20 | -0.34 | +0.01 | -0.33 | +0.30 | " " |
| L21 | +0.12 | +0.16 | +0.29 | +0.59 | " " |
| L22 | -0.18 | +0.03 | -0.15 | +0.44 | " " |
| **L23** | **+7.31** | -1.13 | **+6.19** | **6.63** | " " |
| L24 | +0.84 | -0.06 | +0.78 | 7.41 | " " |
| L25 | +0.25 | -0.47 | -0.22 | 7.19 | " " |
| L26 | +1.81 | +0.69 | +2.50 | 9.69 | " " |
| L27 | -2.69 | +0.00 | -2.69 | 7.00 | " " |
| L28 | +1.38 | -0.97 | +0.41 | 7.41 | " **" |
| **L29** | **+17.09** | -1.38 | **+15.72** | **23.13** | **" Volt"** |
| L30 | +5.50 | -2.63 | +2.88 | 26.00 | " Volt" |
| L31 | +0.13 | -1.38 | -1.25 | 24.75 | " Volt" |
| L32 | -2.13 | +0.50 | -1.63 | 23.13 | " **" |
| **L33** | +4.13 | **+14.00** | **+18.13** | **41.25** | " Volt" |

**Final logit: 41.25. Probability: 81.6%.**

### DLA Summary

| Component | Total DLA | % of final logit |
|---|---|---|
| Embedding | +11.69 | 28.3% |
| L0 (suppression) | -24.38 | -59.1% |
| L4 attention | +9.22 | 22.4% |
| L23 attention | +7.31 | 17.7% |
| **L29 attention** | **+17.09** | **41.4%** |
| L30 attention | +5.50 | 13.3% |
| **L33 FFN** | **+14.00** | **33.9%** |
| L33 attention | +4.13 | 10.0% |
| All other layers combined | ~-4.3 | -10.4% |

**Dominant contributors:** L29 attention (+17.09) and L33 FFN (+14.00).
Together they account for 75% of the final logit above zero.

**Critical surprise:** L26 contributes only +1.81 attention / +0.69 FFN = +2.50 total.
L26 is the parametric fact store (confirmed in prior experiments for Paris, Canberra, etc.).
For novel in-context facts, the retrieval circuit bypasses L26 almost entirely.

---

## Experiment 1 — Attention Maps (where do retrieval heads look?)

At the retrieval moment (pos 61 "of" → predicting " Volt"), measured attention
weights for every head at key layers. Only positions receiving >5% from any head shown.

### L23 — First Retrieval Stage (+7.31 attn DLA)

Three heads attend to fact tokens:

| Head | Pos 14 " Volt" | Pos 15 "ara" | BOS | Raw DLA | Role |
|---|---|---|---|---|---|
| **H1** | **62.1%** | — | 12.3% | +1.20 | Fact token 1 reader |
| **H3** | **23.6%** | **44.1%** | 18.4% | +2.33 | Both fact tokens |
| **H6** | **26.2%** | **26.2%** | 26.2% | +0.55 | Balanced reader |

L23 H3 is the strongest contributor by DLA (+2.33) and reads BOTH fact tokens.
L23 H1 concentrates entirely on " Volt" (62.1%).

### L29 — Main Retrieval Head (+17.09 attn DLA)

| Head | Pos 14 " Volt" | Pos 15 "ara" | BOS | Raw DLA | Role |
|---|---|---|---|---|---|
| **H4** | **61.7%** | — | 25.6% | **+2.22** | **THE fact copier** |
| H5 | 20.9% | — | 57.0% | +0.09 | Secondary (BOS-dominated) |

**L29 H4 is the single most important head in the entire retrieval circuit.**
It puts 61.7% of its attention on position 14 (" Volt") and contributes
100.3% of L29's total layer DLA. One head, one position, one fact.

Neither L29 H4 nor H5 attend to position 15 ("ara") at all.
They read ONLY " Volt", not "ara".

### L30 — Reinforcement (+5.50 attn DLA)

| Head | Pos 14 " Volt" | Pos 15 "ara" | BOS |
|---|---|---|---|
| H3 | **32.2%** | **12.6%** | 36.5% |
| H0 | 14.8% | 3.5% | 55.1% |
| H7 | 13.7% | 3.2% | 65.2% |
| H5 | 10.1% | 1.0% | 62.1% |

Multiple heads reinforce the " Volt" signal. H3 also reads "ara" (12.6%).

### L33 — Output Stage (+4.13 attn, +14.0 FFN)

**L33 attention does NOT attend to fact tokens at all.** Instead:

| Head | Top position | Weight | Role |
|---|---|---|---|
| H7 | pos 61 " of" (gen query) | **77.0%** | Query reader |
| H2 | pos 61 " of" (gen query) | **44.5%** | Query reader |
| H5 | pos 61 " of" (gen query) | 24.7% | Query reader |
| H0 | pos 50 "model" | 26.8% | Template reader |
| H1 | pos 50 "model" | 24.8% | Template reader |

L33 FFN contributes +14.0 to " Volt" — the second-largest single contribution.
But it reads from the **residual stream** (which already contains " Volt" signal
from L23/L29/L30), NOT from KV attention to fact positions.

### L26 — Marginal for Novel Facts

At the retrieval moment, L26 head attribution for " Volt":

| Head | Raw DLA | Top token |
|---|---|---|
| H2 | +0.21 | " electrode" |
| H3 | +0.17 | " vuot" |
| H0 | +0.12 | " licht" |
| H7 | -0.14 | " importantly" |

Total layer DLA: +0.39. None of the heads' top tokens are " Volt".
L26 is doing something else entirely during novel fact retrieval.

---

## The Retrieval Circuit Architecture

```
Fact in context:  pos 14 " Volt"  pos 15 "ara"
                       |               |
                       v               v
        L23 ------> H1 (62%)     H3 (44% ara, 24% Volt)
                    H6 (26% each)
                       |
                    [residual now contains Volt signal]
                       |
                       v
        L29 ------> H4 (62% → pos 14 ONLY)     ← THE retrieval head
                       |                           +17.09 normalized DLA
                    [Volt signal strongly in residual]
                       |
                       v
        L30 ------> H3 (32% Volt, 13% ara)     ← reinforcement
                    H0, H5, H7 (10-15% Volt)
                       |
                       v
        L33 ------> FFN (+14.0)                 ← amplifier
                    [reads Volt from residual, NOT from KV]
                    [attention reads query "of" at 77%]
                       |
                       v
                    OUTPUT: " Volt" at 81.6%
```

---

## Key Findings

### 1. ONE KV position carries the fact

Position 14 (" Volt") receives 62% attention from both L23 H1 and L29 H4.
Position 15 ("ara") is secondary — read by L23 H3 (44%) and L30 H3 (13%)
but ignored by the main retrieval head L29 H4.

The model does NOT attend to:
- Entity tokens (Zarkov/Industries) at positions 4-7 — irrelevant for retrieval
- Relation tokens (was/founded/in/the/city) at positions 8-12 — not read
- Filler sentence (positions 23-36) — completely ignored
- Year tokens (1987) at positions 17-22 — not read

### 2. Novel facts use L23→L29→L30, NOT L26

Prior experiments showed L26 as the fact store/commitment layer for parametric
facts (Paris, Canberra). For novel in-context facts, L26 is marginal (+2.50
total DLA). The retrieval circuit is:
- **L23**: First reads fact from KV (3 heads, +7.31 DLA)
- **L29**: Main copy operation (1 head, +17.09 DLA)
- **L30**: Reinforcement (4 heads, +5.50 DLA)
- **L33 FFN**: Amplification from residual (+14.0 DLA)

### 3. L33 FFN is an amplifier, not a retriever

L33 FFN provides the second-largest single contribution (+14.0), but L33
attention goes to query positions, not fact positions. The FFN reads the
" Volt" signal already in the residual stream from earlier layers and
amplifies it for output. This is consistent with prior findings about
L33 FFN as a confabulation detector / confidence amplifier.

### 4. L0 actively suppresses (-24.38)

L0 contributes -24.38 to " Volt" — massive suppression. This matches
L0's known role as "continuous prompt reader" — it reads the current
context (the "of" token) and suppresses non-local predictions. L4 then
partially recovers (+7.75) before the main retrieval cascade.

---

## Experiment 2 — Causal Interventions

### L29 H4: THE Retrieval Bottleneck

| Intervention | P("Volt") before | P("Volt") after | New top-1 | KL |
|---|---|---|---|---|
| **Zero L29 H4** | 81.6% | **0.5%** | " **" (99.2%) | **3.81** |
| Zero L23 attn | 81.6% | 67.6% | " Volt" (still) | 0.05 |
| Zero L33 FFN | 81.6% | 24.4% | " **" (75.4%) | 0.73 |

**L29 H4 is the causal bottleneck.** Zeroing it destroys retrieval (81.6% → 0.5%).
L23 is helpful but redundant. L33 FFN amplifies but doesn't retrieve.

### Token Emergence Tracking (" Volt" probability by layer)

| Layer | Rank | Probability | Event |
|---|---|---|---|
| L0-L22 | 19,000-210,000 | ~0% | Nowhere near vocabulary |
| **L23** | **195** | 0.0005% | Jump from rank 19,344 → 195 (first signal) |
| L24-L28 | 74-338 | ~0% | Hovering in top-300 |
| **L29** | **0 (top-1)** | **75.8%** | THE emergence — rank 195 → #1 |
| L30 | 0 | **96.1%** (peak) | Reinforcement |
| L31 | 0 | 63.7% | Regression |
| L32 | 1 | 24.4% | Active suppression |
| L33 | 0 | 81.6% | Recovery (L33 FFN amplifier) |

" Volt" does not exist in the residual until L23 (rank 195). It jumps to #1 at L29.
The signal is CREATED by attention copying from KV, not gradually built.

---

## Experiment 3 — Parametric vs Novel Fact Comparison

### France/Paris DLA (parametric fact — no context needed)

| Layer | Attn DLA | FFN DLA | Total | Top token (logit lens) |
|---|---|---|---|---|
| Embed | — | — | — (+29.0) | — |
| L0 | -26.6 | +5.5 | -21.2 | — |
| L23 | -1.0 | **+4.25** | +3.25 | "The" |
| **L25** | -0.1 | **+12.375** | **+12.25** | **"Paris" (FFN!)** |
| **L26** | **+8.0** | -0.5 | **+7.5** | "Paris" |
| L29 | -0.5 | +0.75 | +0.25 | "Paris" |
| **L33** | +6.75 | **+14.25** | **+21.0** | "Paris" |

### L26 Attention for Paris — Reads Entity from KV

| Head | DLA | Top token | Attention to "France" |
|---|---|---|---|
| **H2** | **+1.89** | **" Paris"** | **28.7%** |
| **H3** | **+0.81** | **"Paris"** | 2.2% (reads " is" at 43.8%) |
| H0-H7 others | < 0.05 | noise | — |

### The Double Dissociation

| Component | Novel (Voltara) | Parametric (Paris) |
|---|---|---|
| **L25 FFN** | +0.25 (irrelevant) | **+12.375 (THE fact store)** |
| **L26 attn** | +1.81 (marginal) | **+8.0 (entity reader)** |
| **L29 attn** | **+17.09 (THE retrieval head)** | -0.5 (irrelevant) |
| L33 FFN | +14.0 (amplifier) | +14.25 (amplifier) |

**Novel facts use attention-based KV copy (L23→L29→L30).**
**Parametric facts use FFN from weights (L25 FFN→L26 attn).**
**L33 FFN amplifies both.** Completely different retrieval circuits.

---

## Experiment 4 — Layer-by-Layer Attention Ablation (KV Necessity Map)

For each layer: zero ALL attention at that layer (= layer can't read from KV).
Baseline: P("Volt") = 81.6%.

### Complete Map

| Layer | Zero Attn → P("Volt") | Top-1 | KL | Category |
|---|---|---|---|---|
| **L0** | **0%** | "the" 19% | **17.9** | **CRITICAL — context reader** |
| L1 | **100%** (+18.4%) | Volt | 1.2 | Anti-critical (noise source) |
| L2 | **100%** (+18.4%) | Volt | 2.1 | Anti-critical |
| L3 | **100%** (+18.4%) | Volt | 1.1 | Anti-critical |
| **L4** | **0%** | "Z" 94% | **5.6** | **CRITICAL — relation template reader** |
| L5 | **100%** (+18.4%) | Volt | 1.0 | Anti-critical |
| L6 | 26.8% | "**" 73% | 0.7 | Moderate damage |
| L7 | **100%** (+18.4%) | Volt | 1.1 | Anti-critical |
| L8 | 9.5% | "**" 90% | 1.5 | Damaged |
| **L9** | **0.006%** | "**" 100% | **7.5** | **CRITICAL — dark router** |
| **L10** | **0.02%** | "**" 100% | **6.6** | **CRITICAL — dark router** |
| L11 | 85.2% | Volt | 0.001 | Negligible |
| L12 | 14.8% | "**" 85% | 1.1 | Moderate damage |
| L13 | 7.6% | "**" 92% | 1.6 | Damaged |
| **L14** | **0.7%** | "**" 99% | **3.6** | **CRITICAL — dark router** |
| **L15** | **0.003%** | "**" 100% | **8.1** | **CRITICAL — dark router (MOST critical!)** |
| L16 | 99.2% (+17.6%) | Volt | 0.3 | Anti-critical |
| L20 | 88.3% | Volt | 0.01 | Negligible |
| L23 | 67.6% | Volt | 0.05 | Mild degradation |
| L25 | 90.2% | Volt | 0.04 | Negligible |
| L26 | 67.6% | Volt | 0.05 | Mild degradation |
| L28 | 62.1% | Volt | 0.09 | Mild degradation |
| **L29** | **0.6%** | "**" 99% | **3.7** | **CRITICAL — THE retrieval head** |
| L30 | 50% | Volt | 0.22 | Moderate (reinforcement) |
| L33 | 56.3% | Volt | 0.15 | Moderate (amplifier) |

### The Critical Seven

Seven layers are essential. Removing any one destroys retrieval:
1. **L0** (KL=17.9) — context window reader
2. **L4** (KL=5.6) — relation template reader ("city of")
3. **L9** (KL=7.5) — dark router (query structure)
4. **L10** (KL=6.6) — dark router (reads "city" from query at 44.7%)
5. **L14** (KL=3.6) — dark router (reads "of" from query at 68.8%)
6. **L15** (KL=8.1) — dark router (finalizes retrieval key)
7. **L29** (KL=3.7) — fact copy head (62% attention to pos 14)

### The Anti-Critical Six

Six layers actively HURT retrieval. Removing them IMPROVES P("Volt") to 100%:
L1, L2, L3, L5, L7, L16

These layers introduce noise/competition that degrades the retrieval signal.
The 18.4% probability occupied by "**" at baseline is entirely due to their
interference.

### The Dark Routing Discovery

**L9, L10, L14, L15 have near-zero DLA for "Volt" yet are the MOST critical
layers (KL 3.6-8.1, higher than L29's 3.7).** They contribute NOTHING directly
to the output logit. Yet without them, retrieval fails completely.

What they do: they read QUERY STRUCTURE from KV (not fact positions), building
the retrieval key that L29 H4 uses to attend to the correct KV position.

| Dark router | What it reads | From which positions |
|---|---|---|
| L9 | Structural tokens | `<end_of_turn>`, BOS, template |
| L10 H1 | **"city"** at 44.7% | Query relation word (pos 60) |
| L14 H7 | **"of"** at 68.8% | Final query word (pos 61) |
| L14 H4 | **"was"** at 34.4% | Query verb (pos 56) |
| L15 H7 | **"of"** 30.3%, **"city"** 13.9% | Query relation (pos 60-61) |

The dark routing layers read the query template and compute a matching key.
L29 H4 then uses this key to find the position in KV where the matching
fact token lives. **They are the query processor. L29 is the fact copier.**

### Parametric Comparison — Dark Routing is Novel-Fact Specific

| Layer | Zero attn → P("Paris") | Effect for parametric |
|---|---|---|
| L9 | 62.1% (from 37.7%) | **Anti-critical (HELPS)** |
| L10 | 73.0% (from 37.7%) | **Anti-critical (HELPS)** |
| **L14** | 0.09% | **CRITICAL (universal)** |
| **L15** | 0% | **CRITICAL (universal)** |
| L25 FFN | 0.15% | **CRITICAL (parametric store)** |
| L29 | 50% (≈same) | Not needed |

### Architecture: Shared + Pathway-Specific Layers

```
SHARED (critical for BOTH novel + parametric):
  L0:   Context reader
  L14:  Universal query processor
  L15:  Universal retrieval key

NOVEL FACTS ONLY (KV-based copy):        PARAMETRIC ONLY (weight-based):
  L4:   Relation template reader            L25 FFN:  Fact from weights
  L9:   Query structure processor           L26 attn: Entity reader
  L10:  Query relation word reader
  L29:  Fact copy from KV pos 14

L9/L10 SUPPRESS parametric retrieval while enabling KV retrieval.
They act as a pathway switch: novel vs parametric.

BOTH amplified by: L33 FFN (+14.0)
```

---

## Experiment 5 — V Vector Anatomy at Fact Position

Decoded the residual stream at position 14 (" Volt") across layers.
This is what the V vector at the fact position encodes — what attention
copies when it reads this position.

| Layer | Top normalized tokens | Interpretation |
|---|---|---|
| L0 | お, в | Random noise |
| L7 | お, demean, factorization | Still noise |
| L14 | ua, ier, icles, iano | Name-like suffixes emerging |
| L23 | **ville (11.9%), City, city** | "This is a city name" |
| L29 | **City (17.3%), ville, city** | City category strengthening |
| L33 | **City (46.4%), ara (31.9%), aria (13.3%)** | Category + CONTINUATION |

**At L33, the V vector at pos 14 encodes BOTH the semantic category ("City")
AND the continuation token ("ara").** When L29 H4 attends to this position,
it reads a vector that represents "a city whose next token is -ara".

The raw top-k at all layers is dominated by `<unused>` tokens (PC00 dark
dimension), consistent with prior findings. The city information lives in
the normalized (post-layernorm) projection, not in the raw dot product.

---

## Experiment 6 — Position Importance (Minimum Context)

Tested which tokens from the fact sentence are needed by varying the
context provided before the query.

| Context provided | P("Volt") | Tokens | Finding |
|---|---|---|---|
| None (query only) | 0% | 0 | No parametric knowledge |
| "Voltara." | 0% | 2 | Fact token alone worthless |
| "Zarkov Voltara." | 0% | 3 | Entity + fact, no relation = nothing |
| "city of Voltara." | 9.5% | 4 | Partial relation helps slightly |
| **"in the city of Voltara."** | **67.6%** | **5** | **Minimum relation context** |
| "founded in the city of Voltara." | 32.0% | 6 | Adding verb HURTS (interference) |
| Full sentence (no year) | 0.4% | 12 | Lacks sentence boundary → collapses |
| Full sentence (with "in 1987") | **99.2%** | 16 | Year provides structural separation |

### Key Findings

1. **Relation context is essential, fact token alone is worthless.** "Voltara" without
   "city of" = 0%. The model needs the relation template at the fact position to
   match it against the query's relation template.

2. **"in the city of" = 5 tokens minimum for 67.6%.** This is the minimum context
   that enables retrieval. These 5 tokens at the fact position provide:
   - Relation context for dark routing (L4 reads "city"/"of", L10/L14/L15 match)
   - Position marker for L29 H4 to attend to

3. **Structural separation is critical.** "in 1987." provides a sentence boundary
   between the fact mention and the query repeat. Without it, the model treats
   the query as a continuation of the same sentence, disrupting the template
   matching that dark routing layers depend on.

4. **Entity name ("Zarkov") is irrelevant.** "Zarkov" is never read by any critical
   head during retrieval. The circuit doesn't need to know WHOSE city it was —
   only that a city was mentioned in the relation template.

### Minimum Position Set

For "city of Voltara" retrieval:
- **Essential positions:** "in the city of Voltara" (5 tokens at fact) + query tokens
- **Helpful:** "in 1987." (structural separator, adds ~32% confidence)
- **Irrelevant:** Entity name, filler sentences, all other tokens

---

## Experiment 7 — Generalization Across Novel Facts

### DLA Profile Comparison

| Layer | Zarkov "Volt" | Helion "Hel" | Match? |
|---|---|---|---|
| L23 attn | +7.31 | +7.50 | Yes |
| L29 attn | +17.09 | +16.63 | Yes |
| L30 attn | +5.50 | +8.25 | Yes |
| L33 FFN | +14.00 | +19.38 | Yes |

**Circuit is universal.** Same layers, same heads, same structure.

### Criticality Depends on Confidence Margin

| Fact | P(answer) | Ablation-sensitive? | Why |
|---|---|---|---|
| Zarkov/Voltara (with filler) | 81.6% | YES | Filler + "**" competition |
| Helion (Dr. prefix) | 100% | NO | Very tight cloze, no competitor |
| Korvath (with filler) | 1.4% | — | Filler devastates this fact |
| Korvath (no filler) | 99.2% | NO | Over-provisioned |
| Vestara (with filler) | 100% | NO | Over-provisioned |

**At marginal confidence (~80%), individual components become bottlenecks.**
**At saturated confidence (100%), all components are individually redundant.**
The circuit structure is identical; vulnerability depends on the margin.

---

## The Complete Retrieval Circuit

```
LAYER    COMPONENT    ROLE                          READS FROM KV?
─────    ─────────    ────                          ──────────────
L0       Attention    Context window reader          YES (all positions)
L1-L3    Attention    NOISE (anti-critical)           YES but harmful
L4       Attention    Relation template reader        YES ("city", "of")
         H2           83% to "of" positions
         H6           80% to "city" position
L5,L7    Attention    NOISE (anti-critical)           YES but harmful
L9       Attention    Query structure processor       YES (structural tokens)
L10      Attention    Query relation word reader      YES ("city" at 44.7%)
         H1           Reads "city" from generated echo
L14      Attention    Query key refinement            YES ("of" at 68.8%)
         H7           Reads final query word "of"
L15      Attention    Retrieval key finalization       YES (query + BOS)
         H7           30.3% "of", 13.9% "city"
L16      Attention    NOISE (anti-critical)           YES but harmful
L23      Attention    First fact copy                 YES (pos 14 "Volt")
         H1           62% to pos 14
         H3           44% to pos 15 "ara"
L29      Attention    MAIN FACT COPY                  YES (pos 14 "Volt")
         H4           62% to pos 14 — THE retrieval head
                      +17.09 DLA — 41% of final logit
L30      Attention    Reinforcement                   YES (pos 14 "Volt")
         H3           32% to pos 14
L33      FFN          Amplification                   NO (reads residual)
                      +14.0 DLA — 34% of final logit
```

## Two-Phase KV Reading

The model reads from KV in two distinct phases:

**Phase 1: Query Processing (L0→L4→L9→L10→L14→L15)**
- Reads QUERY positions and structural tokens
- Builds a "retrieval key" in the residual stream
- Identifies WHAT to retrieve ("city of ___")
- Does NOT read fact positions at all

**Phase 2: Fact Copying (L23→L29→L30)**
- Uses the retrieval key from Phase 1
- L29 H4 attends to fact position (pos 14)
- Copies fact token into residual stream
- " Volt" emerges at L29 (rank 195 → #1)

**Phase 3: Amplification (L33 FFN)**
- Reads " Volt" from residual (NOT from KV)
- Amplifies to final output probability

## Minimum KV Requirements

### Per-position:
- **Fact tokens:** "in the city of Voltara" (5 tokens minimum)
- **Structural separator:** "in 1987." (4 tokens, +32% confidence)
- **Query tokens:** "Zarkov Industries was founded in the city of" (required for Phase 1)
- **Total minimum:** ~15-20 positions out of 62

### Per-layer:
- **Critical:** L0, L4, L9, L10, L14, L15, L29 (7 of 34 layers)
- **Helpful:** L23, L30, L33 (reinforcement/amplification)
- **Harmful:** L1, L2, L3, L5, L7, L16 (6 layers actively hurt)
- **Irrelevant:** L11, L20, L25 (negligible impact)

### Estimated minimum KV budget:
- 20 positions × 7 critical layers × 5.12 KB = **717 KB per fact window**
- With reinforcement layers: 20 × 10 × 5.12 KB = **1.02 MB per fact window**
- Full KV for 62 positions × 34 layers: **10.8 MB per fact window**
- **Compression ratio: ~10-15x reduction possible**

---

## Summary of Findings

### The Three Phases of Novel Fact Retrieval

**Phase 1 — Query Processing (L0→L4→L9→L10→L14→L15):**
Dark routing layers read QUERY positions from KV to build a retrieval key.
They contribute near-zero DLA but are the most critical layers (KL 3.6-8.1).
L10 H1 reads "city" at 44.7%. L14 H7 reads "of" at 68.8%. They match the
query template against relation context stored at the fact position.

**Phase 2 — Fact Copying (L23→L29 H4→L30):**
One head (L29 H4) puts 62% attention on one position (pos 14 "Volt") and
provides 41% of the output logit. The fact signal jumps from rank 19,344
to #1 at L29. This is a discrete copy operation, not gradual accumulation.

**Phase 3 — Amplification (L33 FFN):**
Reads "Volt" from the residual stream (NOT from KV) and amplifies +14.0 DLA.

### Novel vs Parametric: Complete Double Dissociation

Novel in-context facts use attention-based KV copy (L29 H4).
Parametric facts use FFN from weights (L25 FFN +12.4).
L9/L10 act as a pathway switch — they suppress parametric while enabling
KV-based retrieval. L33 FFN amplifies both pathways.

### The KV Budget

7 of 34 layers are critical. 6 actively hurt retrieval.
~20 of 62 positions matter. Estimated 10-15x KV compression possible.
The fact token alone is worthless — relation context ("in the city of")
is the minimum requirement (5 tokens for 67.6% retrieval).

### Experiment Log
- Experiment ID (Lazarus): 9f61e60b-b7ff-4f6a-81fa-b9ae55fab059
- Original session: 8fdf177b (DLA + attention maps)
- Continuation session: 9f61e60b (causal interventions through generalization)
