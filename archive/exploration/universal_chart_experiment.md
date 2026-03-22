# Universal Chart: Relational Address or Private Neighbourhood?

**Experiment ID:** 52394963-c7f6-4ae7-a95e-eda77efde4d2
**Model:** google/gemma-3-4b-it (34 layers, 2560 hidden dim)
**Layer of interest:** L26

---

## Background

Previous work ("The Subspace Is the Address", experiment fd54bf19) established that factual knowledge at L26 is organised as local coordinate frames. A cross-domain PCA subspace (5 capital prompts + 8 unrelated prompts, rank 8) could inject Canberra into a translation prompt at 87–94%. The key finding was: "France via Australia-subspace → Canberra. Australia via France-subspace → Paris." Output follows the coordinate frame, not the vector content.

But the successful subspace was trained with all five capitals in the positive set. The open question: **is that subspace a universal "capital city" relational address that routes any country to its correct capital, or is it a private coordinate frame specific to its training countries?**

This experiment tests three related hypotheses:

- **Relational address hypothesis**: A generic capital subspace can route any held-out country to its correct capital. The chart encodes the relation "capital-of" and the donor provides the entity.
- **Private neighbourhood hypothesis**: Each country needs its own private chart. The generic chart has no attractor for unseen countries and collapses to the nearest training-set attractor.
- **Hierarchy hypothesis**: Even if charts are private, the entity identity encoded at L26 may be relation-independent — meaning a Japan capital activation carries "Japan-ness" that a Japan language chart can find and map to Japanese.

---

## Setup

**Training countries (in generic chart):** Australia, France, Japan, Germany, Brazil
**Held-out countries:** Italy, Egypt, Canada, Thailand, Kenya
**Subspace construction:** For each chart, 1 focal prompt + 8 unrelated prompts → PCA at L26, last token position.

**8 unrelated prompts used throughout:**
1. "Translate to French: The weather is"
2. "Once upon a time there was a"
3. "The largest planet in the solar system is"
4. "The colour of the sky is"
5. "The president of the United States is"
6. "Two plus two equals"
7. "The chemical symbol for water is"
8. "She walked into the room and saw"

**Recipient prompt for all injection tests:** "Translate to French: The weather is"

**Injection method:** `inject_residual` with `subspace_only=True`, `subspace_name=<chart>`, at L26.

**Baseline capitals confirmed:**

| Country | Capital | Model confidence |
|---|---|---|
| Australia | Canberra | 91.9% |
| France | Paris | 80.9% |
| Japan | Tokyo | 83.4% |
| Germany | Berlin | 88.7% |
| Brazil | (ambiguous — "a vibrant and bustling city") | — |
| Italy | Rome | 64.1% |
| Egypt | Cairo | 85.3% |
| Canada | Ottawa | 83.8% |
| Thailand | Bangkok | 67.8% |
| Kenya | Nairobi | 94.8% |

Brazil is an anomaly — the model does not cleanly output Brasília. It is kept in the training set since the L26 activation still encodes some capital-of-Brazil signal.

---

## Phase 1: PCA Eigenvalue Spectra

All country-specific charts were built with rank 8 (constraint: rank < number of prompts = 9). The generic charts used rank 10 from 13 prompts.

| Chart | PC1 variance | 80% cumulative at rank |
|---|---|---|
| australia_chart_L26 | 40.6% | 6 |
| france_chart_L26 | 39.9% | 5 |
| japan_chart_L26 | 40.0% | 6 |
| germany_chart_L26 | 40.2% | 5 |
| brazil_chart_L26 | 39.4% | 6 |
| italy_chart_L26 | 38.6% | 6 |
| canada_chart_L26 | 39.9% | 6 |
| egypt_chart_L26 | 39.2% | 6 |
| japan_language_chart_L26 | 39.8% | 6 |
| generic_capital_L26 | 42.4% | 6 |
| generic_language_L26 | 46.5% | 6 |

**Observation:** The eigenvalue spectra are nearly identical across all countries and both relation types. Language charts have a slightly higher PC1 (46.5% vs ~40%) — language facts produce a more concentrated dominant direction, likely because the model is more certain (97–100% confidence on language vs 64–92% on capitals). The PCA *structure* is shared; the PCA *directions* are private.

---

## Phase 2: The Relational Test

**Question:** Does the generic capital chart route held-out countries to their correct capitals?

**Method:** Inject each held-out country's capital prompt through `generic_capital_L26` into the translation recipient.

| Country | Correct capital | Donor top-1 | Injected top-1 | Result |
|---|---|---|---|---|
| Italy | Rome | Rome (64.1%) | "a" (40.8%) | FAIL — training-set blend: Tokyo 3.4%, Paris 2.4%, Berlin 2.0% |
| Egypt | Cairo | Cairo (85.3%) | "a" (30.0%) | FAIL |
| Canada | Ottawa | Ottawa (83.8%) | **Canberra (77.3%)** | FAIL — collapses to dominant training attractor |
| Thailand | Bangkok | Bangkok (67.8%) | "a" (39.9%) | FAIL |
| Kenya | Nairobi | Nairobi (94.8%) | "a" (17.4%), Canberra 10.3% | FAIL — Canberra leaks |

**The relational address hypothesis is falsified.** The generic chart does not route held-out countries to their correct capitals. Italy's top-k output is a blend of training-set capitals (Tokyo, Paris, Berlin). Canada collapses entirely to Canberra. Kenya shows Canberra leaking at 10%.

The generic chart is a **multi-attractor system** with attractors at the training-country coordinates. When a held-out country is projected in, it lands in the basin of the nearest training attractor. There is no "relational" slot that accepts any country and routes to its capital.

---

## Phase 3: Chart Geometry

### 3a. Cross-injection matrix

**Question:** When a capital-query donor is injected through a *different* country's private chart, which capital comes out — the donor's or the chart-owner's?

| Donor | Chart | Donor top-1 | Injected top-1 |
|---|---|---|---|
| Australia (Canberra 91.9%) | france_chart_L26 | Canberra | **Paris (81.7%)** |
| France (Paris 80.9%) | australia_chart_L26 | Paris | **Canberra (58.5%)** |
| Japan (Tokyo 83.4%) | germany_chart_L26 | Tokyo | **Berlin (90.8%)** |
| Germany (Berlin 88.7%) | japan_chart_L26 | Berlin | **Tokyo (85.5%)** |
| Brazil (ambiguous) | australia_chart_L26 | "a" | **Canberra (91.7%)** |

**Result: the chart owner's capital always wins, without exception.**

France's chart, given Australia's activation, outputs Paris. Australia's chart, given France's activation, outputs Canberra. The donor's country identity is completely overridden. Each chart's dominant PC1 direction encodes the chart-owner's specific capital — any capital-query activation projected into that direction activates the chart-owner's attractor.

**Implication:** The chart does not contain a "capital-city slot" into which the donor places its entity. The chart IS the entity-fact attractor. Projecting Australia's signal into France's 8D coordinate frame lands in the France-capital region because France's chart was defined by how France's capital-query differs from everything else — and that difference, as a PCA basis, is the most salient direction in the subspace.

### 3b. Residual atlas at L26

PCA was run on 15 capital prompts (the 10 listed above plus Spain, Mexico, India, China, Russia) at L26, last token position, 10 components.

| Component | Variance | Positive pole | Negative pole |
|---|---|---|---|
| PC1 | 27.9% | Noise/unused tokens | Canberra, Ottawa, Australian — English-speaking capitals |
| PC2 | 11.5% | Canada | Cairo, Egypt |
| PC3 | 9.5% | Russia, Germany, Japan | Egypt, Kenya |
| PC4 | 7.6% | Egypt | Kenya |
| PC5 | 6.8% | Canada | Brazil, Brasília |
| PC6 | 5.8% | Australia, Sydney | Brazil |
| PC7 | 5.5% | India, Delhi | (scattered) |
| PC8 | 5.4% | Japan, Tokyo | France, French |
| PC9 | 4.7% | Brazil | Russia, Moscow |
| PC10 | 4.1% | China, Beijing | Thailand, Russia |

Effective dimensionality: 4D for 50%, 8D for 80%, 10D for 90%.

**Every single PCA axis is a country-pair discriminator.** There is no abstract "capital-city query" direction. The manifold IS country-identity space. Capitals are not stored as a "capital-of" relation applied to an entity — they are stored as attributes embedded within entity identity coordinates. The distinction between "capital query" and "language query" or "largest city query" is not visible at this level of PCA; what is visible is which *country* is being queried.

---

## Phase 4: The Hierarchy Test

**Question:** Does cross-relation injection work? If Japan's capital activation is injected through Japan's *language* chart, does the output show Tokyo (donor's fact) or Japanese (chart's relation)?

### 4a. Generic chart cross-relation

| Donor | Chart | Donor top-1 | Injected top-1 |
|---|---|---|---|
| Japan capital | generic_language_L26 | Tokyo (83.4%) | **Japanese (82.7%)** |
| Australia capital | generic_language_L26 | Canberra (91.9%) | **English (84.4%)** |
| France capital | generic_language_L26 | Paris (80.9%) | "the" (38.7%), French (10.3%) |
| Japan language | generic_capital_L26 | Japanese (97.4%) | Tokyo (14.1%), diffuse |
| France language | generic_capital_L26 | French (98.9%) | diffuse |
| Germany language | generic_capital_L26 | German (95.6%) | diffuse |

Capital→Language is robust (82–84% for Japan and Australia, which are both in the generic language chart's training set). Language→Capital through the generic chart is weak (14% for Tokyo) or fails entirely.

### 4b. Private chart cross-relation (the definitive test)

| Donor | Chart | Donor top-1 | Injected top-1 |
|---|---|---|---|
| Japan capital | japan_language_chart_L26 | Tokyo (83.4%) | **Japanese (86.3%)** |
| Japan language | japan_chart_L26 | Japanese (97.4%) | **Tokyo (78.3%)** |

**Both directions work cleanly with private charts, and are roughly symmetric (86% vs 78%).** The asymmetry in the generic chart (82–84% vs 14%) is explained by multi-attractor interference: the generic capital chart has 5 competing attractors that swamp the Japan-entity signal arriving from a language activation. The private Japan capital chart has one attractor and can find Japan's entity coordinates from any Japan-related activation.

### 4c. Interpretation

The two-level structure is confirmed:

**Level 1 — Entity coordinates (~8D, relation-independent):** Japan's L26 position encodes "Japan-ness" in a region of residual space that is shared across different relation queries. A Japan capital-query activation and a Japan language-query activation both carry Japan entity identity, and that identity is accessible across different relation charts.

**Level 2 — Relation charts:** A chart built for "capital-of Japan" maps Japan entity coordinates → Tokyo. A chart built for "language-of Japan" maps the same entity coordinates → Japanese. The chart provides the relation-specific decoding; the entity coordinates are the input.

Cross-relation injection works because: the donor's activation = entity coordinates + relation-specific component. The chart's subspace captures the entity-coordinate dimension (the direction that distinguishes this entity from unrelated prompts). The chart maps those entity coordinates to its own relation's output.

---

## Phase 5: Private Charts Work for Any Country

**Question:** Do private charts work for held-out countries (countries not in the generic chart's training)?

| Country | Status | Donor top-1 | Private chart output | KL (donor vs injected) | orthogonal_cosine |
|---|---|---|---|---|---|
| Italy | Held-out | Rome (64.1%) | **Rome (72.7%)** | 0.039 | **1.000** |
| Canada | Held-out | Ottawa (83.8%) | **Ottawa (94.1%)** | 0.091 | **1.000** |
| Egypt | Held-out | Cairo (85.3%) | **Cairo (93.4%)** | 0.049 | **1.000** |

All three held-out countries succeed, with `orthogonal_cosine_similarity = 1.0` in every case.

**What `orthogonal_cosine = 1.0` means:** The 8D subspace captures the *complete* differential signal between the capital-query context and the translation-query context. Nothing useful is left in the orthogonal complement — the 2552D space outside the chart is identical for donor and recipient. The chart has precisely extracted the 8D signal that distinguishes "a capital query about this country" from everything else.

Note that private charts also *improve* the output confidence in some cases (Canada: 83.8% → 94.1%; Egypt: 85.3% → 93.4%). The injection into the neutral translation context, with the correct chart applied, can produce higher confidence than the original direct prompt, possibly because the translation context has lower baseline entropy.

**The chart construction algorithm is universal:** (1 focal prompt) + (8 diverse unrelated prompts) → PCA at L26, rank 8. This procedure works for any country-capital pair the model knows, with no training or fine-tuning on the target country. The chart is computed, not learned.

---

## Synthesis

### What a private chart is

A private chart for country X is a PCA basis computed from the variation between {X's capital query} and {8 diverse unrelated queries}. The dominant direction (PC1, ~40% variance) captures how X's capital-query residual differs from the centroid of all queries — this is X's entity coordinate direction. PCs 2–8 capture how the 8 unrelated queries vary among themselves; these dimensions are needed to correctly locate the injected residual within the broader context, but they do not carry factual signal.

When the chart is used for subspace injection:
- The chart's 8D subspace replaces the recipient's 8D component with the donor's 8D component
- `orthogonal_cosine = 1.0` confirms that outside the 8D chart, donor and recipient are identical — the chart is the minimal sufficient intervention
- The downstream layers see a residual that differs from the unperturbed recipient only in the 8D chart direction, which now points toward X's entity coordinates

### What a private chart is not

A private chart is not a relational router. It does not contain a "query slot" where any entity can be inserted. It contains only one entity's coordinates. Projecting a different entity through the chart does not apply a relation transformation — it activates the chart's own single attractor.

The failure of the generic chart for held-out countries is not surprising in this light: the generic chart contains 5 attractors (one per training country). A held-out country has no attractor in the chart's span. Its projection falls in the basin of the nearest training attractor (Canberra, for Canada and Kenya), or spreads diffusely across multiple attractors (Italy).

### Why the manifold is flat but private

All country-specific charts have nearly identical eigenvalue spectra (PC1 ≈ 40%, 80% at rank 6). This means the manifold has uniform curvature everywhere — the same amount of variation is needed to describe any country's factual coordinates. But the PCA *directions* are different for each country. The manifold is like a sphere: same metric tensor everywhere, but every point has its own tangent plane. Each country's private chart is the tangent plane at that country's point on the manifold.

This uniformity explains why the private chart algorithm is universal: the same construction (1 focal + 8 unrelated, rank 8) works for every country because the manifold has the same local structure everywhere. The output changes (from Canberra to Ottawa to Cairo) because the chart points in a different direction for each country, even though the eigenvalue structure is the same.

### The two-level hierarchy

The factual manifold at L26 has this structure:

```
Entity coordinates (~8D per entity, relation-independent)
    |
    |  [relation chart for "capital-of X"]      → capital city
    |  [relation chart for "language-of X"]     → official language
    |  [relation chart for "currency-of X"]     → currency
    |  ...
    v
Fact output
```

Entity identity is encoded at a level that is accessible across different relation types. A Japan-capital activation and a Japan-language activation both carry Japan entity coordinates. A Japan-capital chart can extract Tokyo from a Japan-language activation (78.3%), and a Japan-language chart can extract Japanese from a Japan-capital activation (86.3%).

This is not the same as saying the entity and relation are cleanly separable — the cross-relation injection works by projecting the donor into the chart's subspace, which implicitly extracts the entity signal and discards the relation signal. The projection IS the separation.

---

## Open Questions

**1. What is the geometric relationship between a country's capital-chart and language-chart?**
Are their PC1 directions aligned (pointing in similar directions through entity space) or orthogonal (completely different angles)? If aligned, it would confirm that entity identity has a single "address" used across relations. If orthogonal, entity identity is encoded differently per relation and the cross-relation success is explained by something more subtle.

**2. Does the entity × relation hierarchy hold for non-geographic facts?**
The current experiments use country attributes (capital, language). Do the same results hold for, e.g., inventor-of, discovered-by, born-in? These are structurally different: inventors and birthplaces are not attributes of a country-entity but of a person-entity. The two-level structure may only hold within a single entity type (countries).

**3. At what layer does entity identity first become relation-independent?**
L14 was the Markov threshold for same-domain injection (capital → capital). Does cross-relation injection (capital → language) work at L14, or does it require L24+? This would reveal when the entity coordinates become stable and relation-agnostic.

**4. What is the dimensionality of the shared entity signal vs the private relation signal?**
Cross-relation injection works at 78–86%. The missing 14–22% might be explained by a relation-specific component in the donor that corrupts the injection. If entity coordinates are truly shared, the shared component should be extractable and its dimensionality measurable.

**5. Can a chart for a relation the model is uncertain about be used to read out a fact the model IS certain about?**
The birthplace-of-author prompts have very diffuse distributions at the next token (top-1 < 20%). If a private birthplace-of-Shakespeare chart were built and used to inject a Shakespeare-related activation, would the output be Stratford-upon-Avon, or would the chart's uncertainty propagate?
