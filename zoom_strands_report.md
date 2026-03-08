# Hierarchical Zoom: Mapping the Resolution Strands

**Model:** google/gemma-3-4b-it (34 layers, 2560 hidden dim, 8 heads, bfloat16)
**Experiment:** f87eb7ba-03a0-4591-8a4a-fb912b5cf4a8
**Prompts:** 27 across simple capitals, named birthplaces, compositional retrievals, and failures

---

## The Central Question

How does the residual stream resolve hierarchical answers? Does it proceed through FORMAT → CULTURE → COUNTRY → CITY → TOWN as distinct sequential stages, or do multiple resolution strands coexist in parallel, projecting their signals simultaneously until one wins?

The answer: **both, and neither cleanly.** Zoom levels exist, but they are not sequential stages — they are simultaneous projections that shift in relative amplitude across depth. The "merge point" is not a geometric event (strands converging in shared dimensions) but a computational one (a single FFN write at layer 26 that amplifies one projection over another by 4.6×).

---

## The Seven Zoom Levels

Tracking residual projections onto answer tokens across all 34 layers and 27 prompts reveals a consistent seven-phase structure:

| Level | Layers | Dominant Signal | Notes |
|---|---|---|---|
| **ARTIFACT** | L0–7 | "wolves" / multilingual tokens at 97–100% | BOS embedding dominates. Universal across all prompts. |
| **COLLAPSE** | L8 | Artifact eviction | Semantic computation begins. Residual mean shifts. |
| **DOMAIN** | L13–18 | "famously", "located", "city", "born" at <1% | Query type resolved, nothing content-specific. |
| **CRYSTALLIZATION** | **L24** | Shakespeare/Poland/Dublin/Paris/Sydney at 30–90% | **Universal jump. Single most important layer.** |
| **DISCRIMINATION** | L25–26 | Denmark vs Stratford; Canberra vs Sydney | **L26 FFN writes the decisive answer. All failures determined here.** |
| **CONSOLIDATION** | L27–31 | Stratford (77%), Canberra (98%), Dublin (97%) | Successes lock in; failures oscillate. |
| **FINAL** | L33 | Stratford (89% R&J) vs "a" (18% Hamlet) | Format template competition at L32. Recovery for capitals. |

The crystallization at L24 is the most reliable signal in the model. Across all 27 prompts — every success and every failure — layer 24 is the layer where the residual first projects strongly onto semantically specific tokens. Before L24: format and domain. After L24: content.

---

## Strand Structure

The residual atlas (PCA across all 27 prompts) reveals the strand geometry at three key layers.

### At L24 — Cultural crystallization

The atlas effective dimensionality: **50% variance in 3D, 80% in 9–11D, 90% in 14–15D.** This is the same at every layer — the effective rank does not change across depth, meaning no compression occurs. The same number of independent dimensions carry information from L16 to L33.

Key strands at L24:
- **PC07 (3.7%):** Shakespeare/Elizabethan/Stratford(+) vs German(−) — the AUTHORSHIP strand
- **PC10 (2.1%):** Valladolid/Spain(+) — the Cervantes strand

The Shakespeare cluster at L24 has **feature dimensionality 7** (containing both Warwickshire and Denmark/Danish within the same feature). The correct birthplace and the wrong dramatic setting are already bound into the same low-dimensional feature at crystallization.

### At L26 — Literary geography competition

- **PC07 (4.0%):** German/Goethe(+) vs Warwickshire/Verona/Stratford(−) — LITERARY GEOGRAPHY COMPETITION

This is the critical observation: Stratford (correct birthplace) AND Verona (Romeo & Juliet setting) appear on the **same negative side** of the dominant strand, opposed to German literary geography. The model has partially co-represented birthplace and Italian setting — they share the same subspace. The FFN discrimination at L26 operates within this shared space.

- **PC15 (1.7%):** Denmark/Belgium on negative — the setting strand (tiny, despite its causal power)

### At L28 — English geography consolidation

- **PC06 (5.0%):** Gloucester/Stratford(+) vs Goethe/Deutschland(−) — English birthplace persists
- **PC08 (3.4%):** Bonn(+) — Beethoven strand appears

By L28, for successful prompts, the correct city token has consolidated into its own strand. For failures, the wrong geography continues to dominate.

---

## The Merge Point Is Computational, Not Geometric

The naive expectation: strands merge when their token directions become less orthogonal across depth. The measurement contradicts this.

**Token-to-token angles in embedding space are constant across all 34 layers:**

| Token pair | Angle |
|---|---|
| Denmark ↔ Stratford | 81.6° (constant all layers) |
| Verona ↔ Stratford | 79.1° (constant) |
| Setting direction ↔ Stratford | 90.4° (perfectly orthogonal) |

These angles do not change from L0 to L33. The geometry of the answer tokens in the residual basis is static. There is no geometric merge event.

What actually happens at L24:

| Prompt | Denmark projection | Stratford projection |
|---|---|---|
| Hamlet (L24) | 994 | 990 |

They are **tied at crystallization.** Both strands are equally present at L24. The L26 FFN then writes:

- Hamlet L26 FFN: **+829 Denmark, +181 Stratford** → 4.6× asymmetry favoring the wrong answer
- R&J L26 FFN: **+610 Stratford, +397 Verona** → 1.5× favoring the correct answer

The merge is not a geometric event. It is the FFN learned association: "Hamlet → Denmark" more strongly than "Hamlet → Stratford → Shakespeare birthplace." The strands don't collapse into each other — one gets written louder.

---

## The Raw/Normalized Paradox

The most surprising finding from tracing residual trajectories to L33:

**For Hamlet at L33 RAW projection:**

| Token | Raw projection |
|---|---|
| Stratford | **2706** (HIGHEST) |
| Shakespeare | 2388 |
| Denmark | 1847 |

The correct answer has the highest raw dot product at the final layer. Stratford overtakes Denmark in raw space at L31 and reaches 2706 by L33 — higher than any other token.

**But the normalized output gives: "a" at 18.3%, Stratford at 5.9%.**

The bottleneck is not the residual content. It is the layer norm.

Layer normalization subtracts the residual mean before projecting onto the vocabulary. The accumulated Denmark-frame content from L24–L28 biases the residual mean. When common-mode rejection is applied, the mean includes strong Denmark-direction signal. Stratford, which is partially aligned with this mean (since both Denmark and Stratford crystallized together at L24), gets more subtracted than format tokens like "a", which are more orthogonal to the accumulated Denmark-frame.

After normalization, format tokens dominate because they are more orthogonal to the biased mean. Stratford's high raw projection is partially masked by the common-mode.

**The bottleneck is layer norm application, not residual content, not unembedding geometry.**

Full L26 injection (replacing the Hamlet L26 residual with the R&J residual) confirms this: Stratford reaches **85.9%** (KL from donor = 0.014). The normalization reads out the correct answer cleanly once the L24–L28 mean bias is removed.

---

## The Zoom Transition: 46.2° Partial Rotation

Between cultural crystallization (L24) and city consolidation (L28), the residual undergoes a partial rotation:

- **Culture direction** (Russia vs France, derived at L24): axis 1
- **City direction** (Moscow vs Paris, derived at L28): axis 2
- **Angle between them: 46.2°**

This is neither:
- Pure refinement (0°) — the same dimensions carrying finer information
- Pure rotation (90°) — entirely new dimensions constructed from scratch

The 46.2° rotation means approximately **70% of the cultural information carries forward** into the city representation; **30% is newly constructed** in the L24–L28 band. The partial overlap creates a window where culture-level strands can project into city-level dimensions, enabling the zoom transition to occur continuously rather than as a discrete switch.

---

## The Blind Intervention Fails

Given that L26 FFN is the discrimination layer, one attempted intervention: extract the direction separating setting-heavy prompts (Hamlet, Othello, A Midsummer Night's Dream) from birthplace-heavy prompts (Romeo & Juliet, Twelfth Night, Merchant of Venice), then subtract the setting direction to suppress the wrong signal.

The setting direction at L26 is **90.4° from Stratford** — perfectly orthogonal to the birthplace signal.

Steering at α = −20 (subtracting the setting direction): produces "Verona Verona Verona..." for **both** Hamlet AND R&J. The anti-pole of the setting direction is Verona (Italian-setting Shakespeare), not birthplace. Suppressing setting reveals Italian geography, not English birthplace. They are in perpendicular subspaces.

Two operations are needed:
1. Suppress setting direction (removes "Verona" pressure)
2. Amplify birthplace direction (requires knowing the correct answer class)

The second operation requires supervised knowledge of the answer. The blind intervention cannot recover Stratford without already knowing Stratford is the target.

---

## Four Failure Signatures

Across the 27-prompt library, four distinct failure types emerge:

| Type | Example | L24 | L26 | Mechanism |
|---|---|---|---|---|
| **Wrong write** | Hamlet → Denmark | Shakespeare(30%) | Denmark wins(44%) | L26 FFN writes setting 4.6× over birthplace |
| **Zoom ceiling** | Faust, Marie Curie | Germany(72%), Poland(89%) | Country(95–98%) | Country strand saturates; city-specific neurons absent |
| **Missing strand** | Moby Dick | "located"(50%) | Boston via weak association | Author-identity hop fails; Melville never bound to context |
| **Ambiguous entity** | Symphony 9 | "located"(45%) confused | 3-way Germany/London/Vienna | Multiple composer candidates, none dominant |

### Wrong write
The model knows the correct answer — it's present in the residual at L24 (994 vs 990, tied). The FFN discrimination overwrites it. A targeted subspace injection at L26 corrects to 85.9%.

### Zoom ceiling
The granularity is stuck at country level. Germany reaches 99% at L28 for Faust. No strand ever resolved to Weimar (Goethe's city). The city-level feature for this entity was never learned strongly enough to appear in the atlas. Country-level knowledge is complete; sub-country knowledge is absent.

### Missing strand
"The birthplace of the author of Moby Dick" returns "located" (50%) at L24 — no geographic signal at all. The Melville → Nantucket/New Bedford binding was never formed as a crystallizable strand. No injection fix is available because there is no donor residual that carries the correct binding.

### Ambiguous entity
Symphony No. 9 activates multiple composer frames simultaneously (Beethoven, Schubert, Dvorak all have famous 9th symphonies). No single entity dominates at L24, producing a three-way competition between Germany, London, and Vienna at L26. The ambiguity is upstream of the discrimination layer.

---

## Architectural Synthesis

### The zoom is amplitude, not stage

All resolution levels coexist in the residual simultaneously. What changes across depth is the relative amplitude of each level's signal:
- FORMAT tokens dominate L0–L8 (BOS drives ~97% of projection)
- DOMAIN tokens have a brief window at L13–L18
- CULTURE/COUNTRY tokens crystallize abruptly at L24
- CITY tokens resolve at L26–L28 for successes, never for zoom-ceiling failures

There is no discrete transition from one stage to the next. Each level fades as the next rises.

### The FFN at L26 is the resolution gate

The single most causally important computation in this model's factual retrieval is the L26 FFN write. For all prompts in the 27-prompt library:
- Correct capitals: L26 FFN writes the correct city to dominance
- Wrong-write failures: L26 FFN writes the wrong association
- Zoom-ceiling failures: L26 FFN has no city-level neuron to write

Zeroing L26 FFN flips Australia → Sydney (correct to wrong). Replacing L26 residual for Hamlet flips Denmark → Stratford (wrong to correct). The layer is both necessary and sufficient for discrimination.

### Strand separation is impossible without supervised access

The blind intervention shows the fundamental limit: setting and birthplace strands are orthogonal to each other. You cannot subtract one to reveal the other. Separating them requires either:
- Knowing the correct answer class (to amplify the right strand)
- A classifier gate that identifies which strand should dominate for a given prompt

The geometry does not provide a free lunch. Orthogonality means independence, but it also means you can't use one axis to navigate the other.

### The correct answer is always there

Across all successful prompts and most failures, the correct answer token's raw projection rises monotonically from L24 onward, often overtaking the wrong answer in raw space by L31–L33. The failure is not that the model lacks the knowledge — it is that:

1. The wrong answer's mean bias contaminates the normalization
2. The layer norm's common-mode rejection suppresses the correct answer relative to format tokens
3. The discrimination at L26 amplified the wrong direction, setting up the normalization to fail

**The geometry is not the barrier. The knowledge binding is.**

The model learned "Hamlet → Denmark" as a stronger association than "Hamlet → Shakespeare → Stratford." The strand for Stratford exists in the residual. It simply loses the normalization competition because the residual mean was biased by the stronger (wrong) association during the L24–L28 window.

Fix the FFN write at L26 and the layer norm reads out the correct answer at 85.9%.

---

## Key Numbers

| Measurement | Value |
|---|---|
| Answer subspace fraction of residual norm | 0.44–0.76% |
| All answer tokens to residual angle | ~87–89° (near-orthogonal) |
| Denmark ↔ Stratford embedding angle | **81.6°** (constant all layers) |
| Verona ↔ Stratford embedding angle | **79.1°** (constant) |
| Setting direction ↔ Stratford | **90.4°** (perfectly orthogonal) |
| Culture (L24) to City (L28) transition angle | **46.2°** (partial rotation) |
| Atlas effective dims: 50% / 80% / 90% | 3D / 9–11D / 14–15D (all layers) |
| L26 FFN asymmetry (Hamlet) | 4.6× Denmark over Stratford |
| L26 FFN asymmetry (R&J) | 1.5× Stratford over Verona |
| L24 Denmark/Stratford tie | 994 vs 990 (delta = 4) |
| L33 raw: Stratford (Hamlet) | 2706 (highest — correct answer present) |
| L33 normalized: Stratford (Hamlet) | 5.9% (suppressed by layer norm) |
| Full L26 injection result | 85.9% Stratford (KL = 0.014 from donor) |
| Shakespeare feature dimensionality | 7D (contains both Warwickshire AND Denmark) |

---

*Experiment conducted on google/gemma-3-4b-it using Lazarus MCP interpretability server. All measurements via residual_trajectory, residual_atlas, subspace_decomposition, feature_dimensionality, extract_direction, steer_and_generate, and inject_residual tools.*
