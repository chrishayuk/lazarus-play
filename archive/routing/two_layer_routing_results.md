# Two-Layer Routing: L26 Compass + L29 K-Vector

**Experiment ID:** 8fa47623-3dc7-4184-a33b-8fe847768c43
**Model:** google/gemma-3-4b-it (34 layers, 2560 hidden dim, 8 attention heads)

---

## Executive Summary

**The two-layer hypothesis is partially confirmed but the architecture is different from predicted.** L26 and L29 are NOT complementary routers (map + index). Instead, they are **redundant** for factual queries and **both fail** for tonal queries. The model routes through the full multi-layer attention stack, not through any two isolatable layers. However, K-vectors DO carry content-type information in a 3D dark subspace — a surprising finding that opens a different path.

----

## Key Findings

### 1. L26 and L29 Find the SAME Targets (Experiment 1)

| Query | L26 Best Content | L29 Best Content | Same? |
|---|---|---|---|
| Sports | baseball(p42) 8.9% H2 | baseball(p42) 9.2% H4 | **YES — same position** |
| Amusing | NOTHING (BOS 71%) | NOTHING (BOS 47-98%) | YES — both fail |
| Worried | unexpected(p212) 5.7% | Buzz(p262) 4.5% | Same section, different entry |

**For factual queries:** Both layers converge on the same token position. Two layers adds confirmation, not coverage. The hypothesis that L26 finds windows while L29 finds positions is **wrong** — both find the same positions.

**For tonal queries:** Both layers fail completely. No attention routes to Section B (Czar joke, Laughter) for the "amusing moments" query. BOS token dominates with 47-98% attention across all heads at both layers.

**For emotional/reasoning queries:** Both find the correct section (C — crew worry) but through different entry points. L26 H0 finds "unexpected" (adjective, p212), L29 H4 finds "Buzz" (entity, p262). Mild complementarity within the same section.

### 2. K-Vectors Carry Content-Type Information (Experiment 3)

**3D subspace at L29 achieves 100% classification between interesting content and routine comms.**

| Dimension | Variance | Cumulative | Top Token |
|---|---|---|---|
| PC1 | 29.1% | 29.1% | dark (unused) |
| PC2 | 17.9% | 47.0% | Roger |
| PC3 | 12.4% | 59.5% | dark |
| | | **100% accuracy at 3D** | |

Feature dimensionality analysis: positive class = interesting content (sports, humor), negative class = routine comms (radio checks, trajectories). The K-vector space has genuine content-type structure:

- **PC2 projects to "Roger"** — the routine comms marker. High PC2 = routine.
- After RMSNorm at position 42 ("baseball"): biggest gainers are **bats, standings, softball, headlines** — all sports domain
- The content signal is REAL but lives in dark dimensions, not in vocab-projectable angles

**However:** Raw residual-to-token angles show all positions nearly orthogonal (~88-92°) to all content tokens. The 1.2° self-token advantage is noise in 2560D. The Q·K match exploits the full 256D head space including dark components that direction_angles cannot measure.

### 3. L29 Is MORE Discriminative Than L26 for Query Intent (Experiment 4)

| Query Pair | L26 Angle | L29 Angle | More Discriminative |
|---|---|---|---|
| Amusing vs Sport | 11.85° | 14.07° | **L29** (+2.22°) |
| Amusing vs Tense | 8.15° | 10.30° | **L29** (+2.15°) |
| Sport vs Timeline | 10.58° | 12.37° | **L29** (+1.79°) |
| Worried vs Surprised | 4.36° | 4.29° | L26 (+0.08°) |
| Worried vs Wrong | 6.38° | 6.28° | L26 (+0.10°) |

**Overall:** L29 mean angle = 10.97° vs L26 mean angle = 9.78°. L29 provides 12% more angular spread. L29 is the more discriminative layer overall, not L26.

**The hypothesis that L26 discriminates tonal queries better is REJECTED.** Neither layer clusters queries by TYPE (tonal/factual/emotional). Within-cluster distance equals between-cluster distance. Queries separate by semantic content, not abstract category.

### 4. Routing is Distributed, Not Two-Layer (Experiment 2)

Zeroing L29 H4 for the sports query: KL = 0.012 (weak effect). The model still predicts "The" → "baseball" correctly.

For mixed queries, attention at both L26 and L29 shows **zero content attention** — all attention is on query tokens and structural positions. The full multi-layer stack does the routing collectively.

This is consistent with prior work (geometric_routing_results.md): routing is distributed across L24-L28, with different heads at different layers handling different queries.

### 5. Section Labels Are Not Needed (Experiment 5)

| Condition | Sports Quality | Amusing Quality |
|---|---|---|
| With section labels | 5/5 | 4/5 |
| Without section labels | 5/5 | 4.5/5 |

The model routes via **surface-text features**: "baseball", "Orioles", "(Laughter)", "Czar", "brushing his teeth" — not via [SECTION A - SPORTS NEWS] labels. Removing labels doesn't degrade quality. In fact, amusing query quality slightly *improves* without labels (model finds "turn blue" humor it missed with labels).

### 6. Mixed Queries: Phrasing Is the Bottleneck

| Query Phrasing | Quality |
|---|---|
| "What sport and was there anything funny about how it was reported?" | 1.5/5 — confabulates |
| "Describe both the sport AND the funniest moment. Address both parts." | 5/5 — perfect |

The implicit compound question ("X and Y?") causes the model to anchor on X and try to find Y *within the same section*. Explicit two-part instruction ("Describe X AND Y. Address both.") correctly routes to both sections independently.

---

## Architecture Revision

### The Hypothesis Was Wrong

```
PREDICTED:  L26 (window selection) → L29 (position-level retrieval)
            Complementary: L26 finds WHERE, L29 finds WHAT

ACTUAL:     L24-L28 (distributed multi-head stack)
            Redundant at each layer. Different heads for different queries.
            No single layer or head pair is the bottleneck.
```

### What Actually Routes

**Factual queries (sports, facts, entities):**
- L26 H2: section headers + content keywords (SPORTS 7.4%, baseball 8.9%)
- L29 H4: content keywords directly (baseball 9.2%)
- BOTH converge on same positions. L29 slightly stronger but not by much.
- Content signal carried in 3D dark subspace within K-vector space

**Tonal queries (amusing, mood, emotion):**
- Neither L26 nor L29 routes via attention
- Model uses surface-text features: "(Laughter)", "Czar", absurdity cues
- No geometric routing needed — keyword matching works

**Emotional/reasoning queries (worry, surprise):**
- Both L26 and L29 find correct section
- Different entry points: L26 → adjectives, L29 → entities
- Mild complementarity but same section identified

**Mixed queries:**
- Success depends on query phrasing, not routing architecture
- Explicit multi-part instructions succeed; implicit compounds confabulate

### The Corrected Architecture

```
Factual query:   Full multi-layer attention (L24-L28) → distributed routing
                 Any single layer is sufficient but not necessary
                 K-vector has 3D content-type subspace — exploitable

Tonal query:     Surface-text matching (Laughter, Czar, joke markers)
                 No geometric routing involved
                 The model IS the reader — comprehension not routing

Mixed query:     Explicit decomposition in prompt → sequential routing
                 "Do X. Then do Y." not "X and Y?"
```

---

## Implications for Practical Routing

### 1. Two-Layer Cascade Adds Nothing

For factual queries at this scale (~600 tokens), the model's full attention stack already routes perfectly. Adding an external two-layer pipeline (L26 → L29) would duplicate what the model does internally. The routing is **built into the forward pass**.

### 2. K-Vector Content Subspace Is Exploitable

The 3D subspace finding (100% interesting-vs-routine classification) means K vectors from fact positions could be pre-filtered by content type before Q·K matching. This would be useful at SCALE (725+ windows) where the K-vector library is too large for exhaustive Q·K:

```
Step 1: Project all stored K vectors into 3D content-type subspace
Step 2: Filter by query type (sports → filter to "interesting" cluster)
Step 3: Run Q·K only against filtered K vectors
```

This is a **pre-filter**, not a cascade. It reduces the search space before the main routing operation.

### 3. Tonal Routing Needs a Different Mechanism

Neither geometric compass nor K-vector routing finds humor. The signal is in surface text. For a practical system:

```
Tonal queries:   BM25 with expansion (Laughter, joke, funny, Czar, etc.)
                 → top-10 windows → full replay → generate
                 No forward pass needed for routing
```

### 4. Query Phrasing > Routing Architecture

The mixed query experiment shows that **how you ask** matters more than **how you route**. Decomposing a compound question into explicit parts ("Do X. Then do Y.") is more valuable than any routing optimization.

---

## Summary Table

| Question | Answer |
|---|---|
| Do L26 and L29 find different windows? | **No** — same positions for factual, both fail for tonal |
| Do K vectors carry content-type info? | **Yes** — 3D subspace, 100% classification |
| Is L26 more discriminative for tonal? | **No** — L29 is 12% more discriminative overall |
| Does two-layer cascade improve generation? | **No** — model already routes perfectly at this scale |
| Can mixed queries use both circuits? | **Not via routing** — query decomposition works better |

---

## The Pattern

L26 is NOT the map and L29 is NOT the index. They're both part of the same distributed multi-layer routing stack. The "two circuits" from prior work (parametric L26 FFN vs novel L29 attention) reflected **different knowledge sources** (trained weights vs KV cache), not different routing strategies.

For a practical external routing system:
1. **K-vector Q·K** for factual position-level retrieval (proven at 6/6)
2. **3D content-type pre-filter** to reduce K-vector search space
3. **BM25 keyword expansion** for tonal/subjective queries
4. **Query decomposition** for compound questions
5. No need for L26 compass — the compass measures content type which is better captured by the 3D K-vector subspace
