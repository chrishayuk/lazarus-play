# Experiment: Entity Manifold Displacement — Three-Stage Architecture Diagnostic

**Experiment ID:** `1d3b4353-8bb0-4b0b-9442-9bad6b6a8bba`
**Model:** google/gemma-3-4b-it (34L, 2560D, 8 heads, bfloat16)

---

## The Central Finding

**The entity compass coordinate at L14 is never displaced by contamination.**

For all four failure cases tested, the 8D PCA entity chart position at L14 is virtually identical between clean and contaminated prompts:

| Case | L14 subspace cosine (clean vs contaminated) |
|------|---------------------------------------------|
| Shakespeare/Hamlet-birthplace | **0.9997** |
| Einstein/Relativity | **0.9992** |
| Dostoevsky/Crime&Punishment | **0.9998** |
| Nabokov/Lolita | **0.9993** |

The entity is in the right place on the manifold. The readers are broken.

---

## Background

Three prior experiments converge on this question.

**From the Entity Compass experiment (0164bd2a):** Factual retrieval has three stages. Stage 1 (L0–L14): entity identity accumulates into a dark, compact signal — 1–2D, FFN-aligned, attention-opposed, orthogonal to all vocabulary directions, invisible to logit lens. Stage 2 (L14–L25): entity signal expands from 1–2D to 5D, relation-specific structure accumulates, cross-relation symmetry breaks at L22. Stage 3 (L26): FFN reads entity+relation state and explodes it into vocabulary space at ~100% confidence.

**From multi-hop branching:** Head 1 at L24 fires on "Danish" for the Hamlet prompt instead of "Elizabethan" — extracting the play's setting rather than the playwright's era. The circuit works perfectly on the wrong entity.

**From the failure taxonomy:** Type 1 (late overwrite) has the correct chain at 87.5% before L32 destroys it. Type 2 (early shortcut) never builds a chain. These map onto different levels of damage — but the WHERE in the three-stage architecture had never been measured directly.

The question: for a multi-hop prompt that fails, at which stage does the damage first appear?

---

## Phase 1: Entity Chart Variance Profiles

PCA charts built with 4 entity-variant + 5 unrelated prompts, rank-8. This construction (cross-domain) is required — within-entity-only charts capture universal high-energy directions and fail to isolate the entity signal.

| Layer | Shakespeare PC1 | Einstein PC1 | Dostoevsky PC1 | Structure |
|-------|----------------|-------------|----------------|-----------|
| L14 | **90.2%** | **87.8%** | **87.3%** | 1D dominant — compact dark signal |
| L18 | 81.9% | — | — | Transitioning |
| L22 | 32.2% | 35.8% | — | 5D for 80% — Stage 2 expansion |
| L24 | 29.3% | — | — | Fully expanded |
| L26 | 24.9% | 38.1% | 34.6% | 4–5D for 80% |

Pattern is entity-independent. Confirms the three-stage architecture at every entity tested.

---

## Phase 2: Full-Space Displacement — The Resolution Limit

`residual_match` computes cosine similarity between last-position residuals at a given layer. Angles from clean Romeo to all candidates:

| Prompt | L14 | L18 | L22 | L24 | L26 |
|--------|-----|-----|-----|-----|-----|
| Hamlet-birthplace (contaminated, fails) | 2.9° | 3.6° | 5.3° | 8.0° | 10.9° |
| Hamlet-century (clean-with-Hamlet, succeeds) | 3.8° | 5.8° | 9.7° | 13.1° | 16.2° |
| Joyce/Ulysses (different entity) | **2.3°** | 3.2° | 5.5° | 9.9° | 13.1° |
| Einstein (different entity) | **2.3°** | 3.2° | 5.9° | 11.3° | 14.7° |
| "Shakespeare was born in..." (clean) | 4.1° | 5.8° | 8.1° | 9.8° | 11.4° |

At L14, the contaminated Hamlet prompt (2.9°) is only 0.6° farther from clean Romeo than Joyce/Ulysses (2.3°) — a completely different entity. The 1° cross-entity signal from the Entity Compass experiment is buried in format-structure noise at this scale. Full-space measurement cannot detect entity displacement because the entity-relevant signal occupies only ~8D out of 2560D (0.3%).

A 45° displacement within the 8D chart would shift the full-space angle by only ~2.3°. Indistinguishable from structural variation.

---

## Phase 3: Logit Lens — When the Damage Detonates

Running logit_lens at L0/L8/L14/L18/L22/L24/L26/L32/L33 for all prompts:

| Case | L14 | L24 | L26 | L33 |
|------|-----|-----|-----|-----|
| Romeo (clean ✓) | dark generic: " supposed", " famously" | Shakespeare 26.9% | Stratford 25%, Verona 22% | Stratford **88%** |
| **Hamlet-birthplace (✗)** | dark generic: " famously", " supposed" | **Shakespeare 30.3%** ✓ | **Denmark 43.8%** ✗ | " a" 18.3% (diffuse wrong) |
| Einstein clean (✓) | dark generic: " famously", " supposed" | Germany 26.9% | **Germany 85.9%** | Ulm **55.5%** |
| **Relativity (✗)** | dark generic: ":", " alors" | **Germany 42.4%** ✓ | **Switzerland 57.4%** ✗ | Switzerland 45.7% |
| Dostoevsky clean (weak ✗) | dark generic | Russia 53.9%, Moscow 3.2% | Russia 73.8%, Moscow 12.8% | " the"/" a" 23% — loses signal |
| **Crime&Punishment (✗)** | dark generic: " supposed" | Russia 46.1%, **Moscow 24.7%** | **Moscow 73%** ✓ | " in" → St. Petersburg |
| **Lolita (✗)** | **"ইংরেজি"** (Bengali "English"!) | **English 93.4%** ✗ | English 98.4% | ":" 54.3% |
| Nabokov clean (✗) | " Germany" as 2nd token (not fully dark) | Russia 36.1%, Ukraine 16% | Russia 65.2%, Moscow 16.4% | "Chi" 67.9% → Chișinău (also wrong) |
| Joyce/Ulysses (control ✓) | dark generic | Dublin 73%, Ireland 16.3% | Dublin **91.4%** | Dublin 35.7% |

Three distinct patterns emerge:

1. **Shakespeare and Einstein:** Dark signal at L14. Entity correct at L24. L26 FFN flips to wrong country.
2. **Dostoevsky:** Dark signal at L14. Entity correct at L24 AND L26. L33 overwrites correct answer.
3. **Nabokov/Lolita:** Bengali "English" at L14 — NOT dark. Committed to English by L24 at 93.4%.

---

## Phase 7: The Dark Signal Diagnostic

`direction_angles` at L14 measures the structural signature of the entity compass mechanism:

| Prompt | Res→FFN | **Res→Attn** | FFN→Attn | Interpretation |
|--------|---------|-------------|----------|----------------|
| Einstein clean ✓ | 13.6° | **163.8°** | 160.1° | Purest dark signal |
| Romeo clean ✓ | 30.3° | **164.6°** | 147.5° | Pure dark signal |
| Relativity contaminated ✗ | 18.1° | **151.8°** | 149.9° | Slightly degraded — Stage 3a |
| Nabokov clean ✓ | 19.7° | **142.1°** | 140.9° | Normal range |
| Hamlet contaminated ✗ | 29.1° | **120.1°** | 120.7° | Moderately degraded — Stage 3a |
| **Lolita contaminated ✗** | **44.4°** | **46.9°** | 66.7° | **BROKEN — attn constructive — Stage 1** |

The dark signal pattern (Entity Compass experiment): residual ≈ FFN (14–30°), residual anti-parallel to attn (142–165°), FFN anti-parallel to attn (141–160°).

For Lolita: all three angles collapse into the 44–67° range. Attention is nearly aligned with the residual instead of opposed. The entity compass building mechanism has switched mode — from suppressive (dark) to constructive (explicit). Yet the entity chart coordinate remains 0.9993 identical to clean Nabokov.

### Diagnostic Thresholds

| L14 attn-residual angle | Stage | Meaning |
|------------------------|-------|---------|
| ≥ 140° | Stage 3 (L26 or L33) | Dark signal intact — entity correctly encoded |
| 90–140° | Stage 3a degraded | Dark signal partially disrupted |
| **< 90°** | **Stage 1 — mechanism break** | Attention constructive — task-type interference at L14 |

A single `direction_angles` call at L14 predicts which stage is damaged without logit lens sweeps or injection tests.

---

## Phase 5: The Three-Stage Failure Taxonomy

### Stage 3a — L26 FFN Relation Conflation

**Cases:** Shakespeare/Hamlet-birthplace, Einstein/Relativity-country

**L14:** Dark signal intact (120–165°). Entity compass correctly building.

**L24:** Head 1 bridge correctly routes to the entity. Shakespeare appears at 30.3%; Germany at 42.4%. The relay works.

**L26:** FFN performs a biased lookup: **entity + surrounding context tokens → wrong fact**. For Hamlet, the context carries Denmark (the play's setting) and the FFN conflates that with the authorial birthplace. For Relativity theory, the context carries Switzerland (Einstein's place of work and citizenship) and the FFN outputs that instead of Ulm (birthplace). The entity coordinate is correct; the readout operator is reading the wrong relation.

**L33:** Wrong output, no recovery.

### Stage 3b — L33 Late Format Overwrite

**Case:** Dostoevsky/Crime&Punishment

**The paradox:** The contaminated C&P prompt shows **more** correct information at L26 than the clean Dostoevsky prompt:
- Clean Dostoevsky at L26: Russia 73.8%, Moscow 12.8%
- C&P contaminated at L26: **Moscow 73.0%**, Russia 23.7%

The novel title helps the entity circuit achieve city-level precision at L26. Then L33 cultural association (Crime and Punishment is set in St. Petersburg, where most of the novel's action takes place) overwrites the correct Moscow signal. The format template generates " in" as the first token, then the model completes "the city of St. Petersburg."

Two independent processes on the same computation: Stage 2–3a succeeds (finding Moscow), then a late-stage cultural template destroys it.

**Fix:** L26 injection does nothing (entity already correct at L26). Requires L31–33 intervention.

### Stage 1 — Mechanism Break (not Coordinate Displacement)

**Case:** Nabokov/Lolita

**L14:** "ইংরেজি" (Bengali for "English") appears in logit lens — not the dark generic signal seen for other prompts. The entity compass mechanism is not dark.

**Direction angles:** Residual→attn = 47° (constructive) vs normal 142–165° (suppressive). FFN→attn = 67° (nearly orthogonal) vs normal 141–160° (opposed). Both attention and FFN are partially constructive — the dark signal dynamic has completely inverted.

**Mechanism:** The "language spoken in the country where the author of X was born is" query structure activates a language-retrieval circuit at L14. The attention, instead of suppressing the FFN signal to build a compact dark entity address, fires constructively on the word "language" — locking the circuit into language-output mode. The model never enters entity-addressing mode; it enters task-type mode immediately, and the task type (language query) pulls toward "English" via Lolita's strong association with the English language (the novel was famously written in English).

**Critical finding:** Despite the mechanism breaking, the entity chart **coordinate** at L14 is still 0.9993 identical to clean Nabokov. Stage 1 failure is mechanism failure, not coordinate failure. The entity address is approximately correct by inertia, but the computation that should be building it has switched to a different mode.

Note: clean Nabokov ("Nabokov was born in") also fails at L33 (generates Chișinău rather than St. Petersburg, possibly confusing Nabokov with another figure or misremembering his biography). This is a separate factual retrieval problem independent of the Lolita query contamination.

---

## Phase 6: Injection Efficacy

### Shakespeare/Hamlet — Subspace Injection at L14, L24, L26

Donor: "The birthplace of the author of Romeo and Juliet was" (→ Stratford 88.5%)
Recipient: "The birthplace of the author of Hamlet was" (→ " a" 18.3%, Stratford 5.9%)

| Injection layer | Injected Stratford | KL from donor | Subspace cosine |
|----------------|-------------------|---------------|----------------|
| None | 5.9% | — | — |
| L14 subspace | 20.9% | 1.46 | **0.9997** |
| L24 subspace | 29.0% | 1.09 | **0.9956** |
| **L26 subspace** | **35.2%** | **0.79** | **0.9890** |

Three findings from this gradient:

1. **Proximity principle:** Later injection = stronger correction. Confirms L26 as the damage site.
2. **Subspace cosine degrades with layer:** The entity chart is nearly identical at L14 (0.9997) but diverges slightly by L26 (0.989). The entity position gradually differentiates through the pipeline even before the final L26 flip.
3. **Full fix is unreachable:** "Hamlet" remains in the prompt context. The word continues to influence the L26 FFN through other token positions. Single-layer subspace surgery partially corrects but cannot override the distributed contextual signal.

### Einstein/Relativity — Injection Fails

The clean "The birthplace of Einstein was" and contaminated "The country where the scientist who developed the theory of relativity was born is" have:
- Different output types (city vs country)
- Different last tokens ("was" vs "is")
- Donor-recipient KL = 10.47 (vs Shakespeare's 2.59)

L14 injection → ":" (54%), Switzerland 7.9% — disrupts but doesn't fix.
L26 injection → "the United States of America" — completely wrong.

The entity chart captures structural and format features alongside entity-specific content. When prompts have incompatible structures, the injected chart conflicts with the recipient's query structure rather than correcting the entity address.

**Structural compatibility between donor and recipient is a prerequisite for injection to work.**

### Dostoevsky/Crime&Punishment — Nothing to Fix at L26

L26 injection: subspace_cosine = 0.9998, KL from recipient = 0.137. The injection changes almost nothing because the entity is already correctly placed at L26 (Moscow 73%). The damage is at L33, where the injection has no reach.

### Nabokov/Lolita — Nothing to Inject

L14 injection: subspace_cosine = 0.9993. The two prompts have nearly identical entity chart positions — there is no correction signal to transfer. Additionally, the donor (clean Nabokov) generates Chișinău itself, so even a perfect transfer would produce wrong output.

---

## Phase 4: Cross-Domain Residual Angles

### Einstein at L14 and L26

| Candidate vs Einstein clean | L14 angle | L26 angle |
|----------------------------|-----------|-----------|
| Dostoevsky clean (different entity) | **1.67°** | 13.56° |
| Hamlet contaminated | 2.95° | 15.7° |
| Relativity contaminated | 3.33° | 15.1° |

At L14, a completely different entity (Dostoevsky) is **closer** to clean Einstein (1.67°) than the contaminated Einstein variant (3.33°). The dominant signal at L14 is the shared "factual query" format direction, not entity identity. By L26, prompts have separated to 13–16° as entity-specific content accumulates.

### Nabokov at L14

| Candidate vs Nabokov clean | L14 angle |
|---------------------------|-----------|
| Einstein clean (different entity) | 3.18° |
| Lolita contaminated | 3.89° |
| Hamlet contaminated | 4.00° |

All within 4° at L14 — entity identity, failure type, and failure stage are all invisible in full-space measurement.

---

## Key Theoretical Revisions

### What the experiment predicted

The design posited three possible outcomes:
- **Stage 1 damage:** Dark entity compass displaced. Wrong coordinates before relation machinery engages.
- **Stage 2 damage:** Entity correct at L14 but Head 1 misfire corrupts the relation mapping at L24.
- **Stage 3 damage:** Both entity and relation correct through L25; L26 FFN or L32–33 overwrites.

### What was found

All four failure cases show **Stage 3 damage** (or a Stage 1 mechanism break that is not coordinate displacement). Stage 2 damage was not observed — the Head 1 bridge at L24 correctly identifies the entity for both Stage 3a failures. The "Danish" Head 1 misfire from the 4-hop boundary test was not reproduced here; for these 2-hop prompts, the bridge fires correctly.

**Revised taxonomy:** The entity compass coordinate at L14 is robust to contamination across all tested cases. What varies across failure types is **where in the downstream pipeline the contaminating context signal is first misread as the output fact:**

| Stage | Layer | Mechanism | Cases |
|-------|-------|-----------|-------|
| **3a** | L26 | FFN reads entity+context → wrong fact | Hamlet-birthplace, Relativity |
| **3b** | L33 | Cultural/format template overwrites correct L26 readout | Crime&Punishment |
| **1\*** | L14 | Query-type interference inverts attention mechanism | Lolita |

\*Stage 1 failure is mechanism failure, not coordinate failure. The entity chart position remains correct.

---

## Connection to Prior Work

| Prior finding | Connection |
|--------------|------------|
| **Compositional Override Failures** (a2321302): Hamlet→Denmark = "Entity competition at L26 FFN" | Confirmed as Stage 3a. Same circuit, now causally mapped. |
| **Head 1 L24 Contextual Attribute Bridge** | Correctly identifies entity in both Stage 3a cases. The bridge works; the L26 FFN misreads downstream. |
| **L26 FFN as fact store** (layer26_ffn.md) | Extended: L26 reads entity+context, not just entity. Context within the prompt window creates a conflation window when it carries strong geographic associations. |
| **L33 late overwrite** (recirculation.md) | Dostoevsky Stage 3b matches the Beethoven/Vienna pattern (L31 FFN overwrite). Stage 3b is a recurring failure mode. |
| **Dark signal structure** (entity_compass.md) | FFN-aligned (14–30°), attn-opposed (142–165°) pattern confirmed for all clean prompts. Contamination degrades the opposition angle: 164° → 152° → 120° → 47° as failure stage moves earlier. |
| **Subspace correction** (trajectory_geometry.md) | Injection efficacy gradient (L14→L24→L26: 20.9%→29%→35.2%) is consistent with each failure living in 0.3–0.8% of the residual at its damage layer. |
| **4-hop boundary test** (markov_bandwidth.md) | That experiment showed Head 1 firing "Danish" for Hamlet (entity disambiguation failure). Here, in 2-hop prompts, Head 1 correctly routes to Shakespeare. Disambiguation capacity depends on prompt structure, not just hop count. |
