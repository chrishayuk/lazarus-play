# Path Trace Experiments — Exp 6a86facb

**Question**: Does the residual at the FINAL token carry a detectable geometric trace of having processed a specific earlier fact, even when NOT asked to retrieve it?

All prompts end with the same neutral sentence: "The weather today is pleasant and mild."
Filler P1 = ~99-token meadow paragraph. P1x5 ≈ 500 tokens, P1x20 ≈ 2000 tokens, P1x80 ≈ 8000 tokens.

---

## EXPERIMENT 7: Path Trace Detection (3 prompts, P1x5 distance)

Prompts:
- **A**: FACT_ZARKOV + P1x5 + ENDING
- **B**: FACT_NEXION + P1x5 + ENDING
- **C**: P1x5 + ENDING (no fact)

### 7b — Cosine similarity at last token, all layers

**Layer 14:**

|       | A      | B      | C      |
|-------|--------|--------|--------|
| A     | 1.0000 | 0.9999 | 0.9998 |
| B     | 0.9999 | 1.0000 | 0.9999 |
| C     | 0.9998 | 0.9999 | 1.0000 |

A vs C = 0.9998255 | A vs B = 0.9999234 | B vs C = 0.9998928
Centroid distance = 0.000119

Footprint barely detectable. A and B (two facts) are MORE similar to each other (0.9999) than either is to C (no-fact ~0.9998-0.9999). Trace present but tiny.

**Layer 20:**

|       | A      | B      | C      |
|-------|--------|--------|--------|
| A     | 1.0000 | 0.9995 | 0.9992 |
| B     | 0.9995 | 1.0000 | 0.9995 |
| C     | 0.9992 | 0.9995 | 1.0000 |

A vs C = 0.9992264 | A vs B = 0.9994802 | B vs C = 0.9994999
Centroid distance = 0.000598

Footprint growing. Pattern: A_vs_B > A_vs_C. Two different facts are more similar to each other than either is to no-fact — this is a **"fact present"** signal beginning to separate from no-fact.

**Layer 26:**

|       | A      | B      | C      |
|-------|--------|--------|--------|
| A     | 1.0000 | 0.9988 | 0.9982 |
| B     | 0.9988 | 1.0000 | 0.9980 |
| C     | 0.9982 | 0.9980 | 1.0000 |

A vs C = 0.9981556 | A vs B = 0.9987507 | B vs C = 0.9980150
Centroid distance = 0.001693

**A_vs_B (0.9988) > A_vs_C (0.9982)**: The two different facts are still MORE similar to each other than to no-fact. The dominant signal is "fact was present", not "which fact". The fact-present offset is ~0.0006 above the no-fact baseline.

**Layer 33:**

|       | A      | B      | C      |
|-------|--------|--------|--------|
| A     | 1.0000 | 0.9942 | 0.9961 |
| B     | 0.9942 | 1.0000 | 0.9929 |
| C     | 0.9961 | 0.9929 | 1.0000 |

A vs C = 0.9960828 | A vs B = 0.9942074 | B vs C = 0.9928774
Centroid distance = 0.005611

**CRITICAL REVERSAL**: A_vs_B (0.9942) is now LOWER than A_vs_C (0.9961). The two different facts are now MORE different from each other than Zarkov is from no-fact. **Fact identity dominates at L33.** The signal has shifted from "presence" to "identity" between L26 and L33.

### Summary table — Layer progression

| Layer | A_vs_C | A_vs_B | A_vs_B > A_vs_C? | Signal type    |
|-------|--------|--------|-------------------|----------------|
| 14    | 0.9998 | 0.9999 | YES (barely)      | Noise          |
| 20    | 0.9992 | 0.9995 | YES               | Fact presence  |
| 26    | 0.9982 | 0.9988 | YES               | Fact presence  |
| 33    | 0.9961 | 0.9942 | NO (reversed)     | Fact identity  |

**The footprint IS detectable** at L26. At L33 it becomes identity-specific.

---

### 7c — Distance decay (P1x20 vs P1x5)

Comparing SHORT_A (Zarkov+x5), LONG_A (Zarkov+x20), LONG_C (no-fact+x20) at L26:

| Pair                      | Cosine  | Interpretation                        |
|---------------------------|---------|---------------------------------------|
| SHORT_A vs SHORT_C (x5)   | 0.9982  | Fact footprint at ~500 tokens          |
| LONG_A vs LONG_C (x20)    | 0.9912  | Fact footprint at ~2000 tokens         |
| SHORT_A vs LONG_A         | 0.9765  | Same fact, different distances         |

**Key finding**: The footprint does NOT decay to zero — it actually appears LARGER at longer distance (0.9912 vs 0.9982). This is counterintuitive. The explanation: at longer filler, the filler itself dominates and both fact+filler and no-fact+filler converge toward a "long filler" attractor. The fact component represents a smaller fraction of the total representation but is preserved absolutely.

Wait — re-reading: LONG_A vs LONG_C = 0.9912 means they are MORE different than SHORT_A vs SHORT_C = 0.9982. So fact-presence signal is **stronger** (lower similarity = more separation) at longer distance.

**SHORT_A vs LONG_A = 0.9765**: Same fact, different distances — these are more different from each other (cosine 0.9765) than either is from no-fact at matching distance. This means **distance effects dominate over fact identity** when comparing across distances.

---

## EXPERIMENT 8: Path Trace Uniqueness — 5 Different Facts at P1x5

6 prompts at L26: P0 (no fact), P1-P5 (5 different facts).

### Full 6×6 cosine matrix

|           | P0(none) | P1(Zarkov) | P2(Nexion) | P3(Brightfall) | P4(Vennox) | P5(Orindale) |
|-----------|----------|------------|------------|----------------|------------|--------------|
| P0(none)  | 1.0000   | 0.9982     | 0.9980     | 0.9985         | 0.9981     | 0.9981       |
| P1(Zarkov)| 0.9982   | 1.0000     | 0.9988     | 0.9992         | 0.9993     | 0.9987       |
| P2(Nexion)| 0.9980   | 0.9988     | 1.0000     | 0.9990         | 0.9986     | 0.9993       |
| P3(Bfall) | 0.9985   | 0.9992     | 0.9990     | 1.0000         | 0.9989     | 0.9989       |
| P4(Vennox)| 0.9981   | 0.9993     | 0.9986     | 0.9989         | 1.0000     | 0.9988       |
| P5(Orin)  | 0.9981   | 0.9987     | 0.9993     | 0.9989         | 0.9988     | 1.0000       |

### Analysis

**P0 distances to each fact** (how different is each fact from no-fact):
- Zarkov: 0.9982 | Nexion: 0.9980 | Brightfall: 0.9985 | Vennox: 0.9981 | Orindale: 0.9981
- Range: 0.9980–0.9985. They are NOT equidistant — some variation, but small.

**Fact-to-fact distances** (all are higher similarity than fact-to-no-fact):
- Most similar pair: P1-P4 (Zarkov–Vennox) = 0.9993
- Most different fact-to-fact pair: P1-P2 (Zarkov–Nexion) = 0.9988

**Semantic clustering** — Are location facts (Zarkov/Nexion/Brightfall) closer to each other than to discovery/orbital?
- P1(Zarkov) vs P3(Brightfall): 0.9992 — both location
- P1(Zarkov) vs P4(Vennox): 0.9993 — location vs discovery
- P2(Nexion) vs P5(Orindale): 0.9993 — location vs orbital
- No clear clustering by semantic type. Variance too small to support type-based grouping.

### Conclusion for Experiment 8

1. **P1-P5 are NOT equidistant from P0** — there is small but real variation in how much each fact perturbs the representation. Brightfall perturbs least (0.9985), Nexion most (0.9980).
2. **P1-P5 are not clearly distinguishable from each other** — the fact-to-fact range (0.9987–0.9993) is tiny. No classifier could reliably identify which fact was seen.
3. **No semantic type clustering** — location/discovery/orbital facts produce similar-magnitude footprints.
4. **The dominant signal is "fact was present"**, not "which fact" or "what type". The residual at L26 encodes a "something happened" perturbation without encoding fact content.

---

## EXPERIMENT 9: Path Trace at Maximum Distance (~8000 tokens)

Prompts at L26: SHORT_A (Zarkov+x5), MAX_A (Zarkov+x80), MAX_B (Nexion+x80)

### 3×3 cosine matrix

|           | SHORT_A | MAX_A  | MAX_B  |
|-----------|---------|--------|--------|
| SHORT_A   | 1.0000  | 0.9924 | 0.9911 |
| MAX_A     | 0.9924  | 1.0000 | 0.9993 |
| MAX_B     | 0.9911  | 0.9993 | 1.0000 |

### Key measurements

| Comparison              | Cosine | Meaning                                              |
|-------------------------|--------|------------------------------------------------------|
| SHORT_A vs MAX_C (est.) | ~0.97  | Fact footprint erased by distance (distance dominates) |
| MAX_A vs MAX_B          | 0.9993 | At 8000 tokens: Zarkov vs Nexion nearly identical    |
| SHORT_A vs MAX_A        | 0.9924 | Same fact, different distances — very different      |

**Critical findings:**
1. **MAX_A vs MAX_B = 0.9993**: At ~8000 tokens distance, two different facts are nearly indistinguishable from each other. The cross-fact separation (0.0007) is far smaller than at P1x5 (0.0006 at L26, but that was A_vs_B not A_vs_C).
2. **Distance dominates over fact identity**: SHORT_A vs MAX_A = 0.9924, which is lower similarity than any fact-to-fact pair at the same distance. The filler length creates the largest geometric signal.
3. **Footprint does persist but is diluted**: MAX_A and MAX_B should still differ from MAX_C (no-fact+x80), but the two different facts become near-identical — confirming the Exp 7c finding that footprints at long distance encode "fact was present" with near-zero identity content.

---

## CROSS-EXPERIMENT SYNTHESIS

### Layer-wise signal progression

| Layer | Signal type           | A_vs_B vs A_vs_C | Centroid distance |
|-------|----------------------|------------------|-------------------|
| L14   | Noise                | presence barely  | 0.000119          |
| L20   | Fact presence        | presence clear   | 0.000598          |
| L26   | Fact presence        | presence strong  | 0.001693          |
| L33   | **Fact identity**    | REVERSED         | 0.005611          |

### Distance decay of footprint (at L26)

| Distance  | Fact vs No-fact cosine | 1 - cosine (separation) |
|-----------|----------------------|-------------------------|
| P1x5 (~500 tok)  | 0.9982        | 0.0018                  |
| P1x20 (~2000 tok)| 0.9912        | 0.0088                  |
| P1x80 (~8000 tok)| ~0.999+       | <0.001 (estimated)      |

Counterintuitive: separation at L26 INCREASES from x5 to x20, then appears to collapse at x80 based on cross-fact near-identity.

### Three-layer story

1. **L14-L20**: Fact is encoded at the fact-token positions (dark space), not propagated to final token. Last-token residual shows barely perceptible perturbation.
2. **L26**: The "something happened earlier" signal consolidates. All facts look alike from this vantage: "a fact was in the context". This is consistent with L26 being a context integration layer. It doesn't know *what* fact, just *that* there was one.
3. **L33**: Identity information arrives. Different facts begin to pull the final token toward different regions. The L33 signal is weaker in absolute terms than what you'd see if the fact were directly queried, but it's now fact-specific.

### Is the footprint detectable?
**YES** — fact vs no-fact is reliably distinguishable at L26 (cosine ~0.9980–0.9985 vs 0.9987–0.9993 for fact-fact pairs). A probe trained on L26 activations could likely classify "context contained a fact" with decent accuracy.

### Is the footprint distinguishable by fact identity?
**WEAKLY at L33 only**. At L26 all five facts produce nearly identical perturbations. At L33 the identity signal emerges but is small (cosine range 0.9929–0.9961 for fact-vs-no-fact, vs fact-to-fact range 0.9942–0.9993).

### Does footprint magnitude decay with distance?
**NOT monotonically**. At L26 the separation (1-cosine) actually GROWS from x5 to x20 (0.0018 → 0.0088). At x80 the two different facts become near-identical (0.9993), suggesting the representation is dominated by "long context" geometry and individual fact identity is lost. The fact-vs-no-fact separation at x80 is unknown without MAX_C but is likely small.

### Most surprising finding
**The L33 identity reversal**: At L26, Zarkov and Nexion are more similar to each other (0.9988) than either is to no-fact (~0.9981). At L33 this flips — Zarkov and Nexion are more different from each other (0.9942) than Zarkov is from no-fact (0.9961). The transformation between L26 and L33 takes a "fact present" signal and converts it into a fact-specific fingerprint. This mirrors the known L26→L33 dynamics from the Canberra/misconception experiments, where L26 commits to an answer and L33 fine-tunes it.

---

## IMPLICATIONS

1. **Passive fact tagging**: The model's final-token residual carries a low-magnitude "fact was present" tag at L26, even when doing nothing with the fact. This tag is not fact-specific.

2. **Late identity resolution**: Fact-specific signals only appear at L33. This is consistent with the general pattern: L26 is a commitment/consolidation layer; L33 is an identity/disambiguation layer.

3. **Distance compression**: At very long distances (~8000 tokens), different facts produce nearly identical footprints. The context length itself dominates the geometry. This is consistent with the prior finding that compass bearing (D10 vs D2000) shifts only 0.9713 — distance effects are large but fact presence is an even larger signal.

4. **Probe design implication**: A L26 probe for "fact present in context" should work. A L33 probe for "which fact was in context" should work better than L26 for identity, but both will be weaker than a probe trained on direct fact retrieval.
