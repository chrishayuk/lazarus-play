# The Dark Dimension Atlas: Mapping the Space Where Transformers Think

**Model:** google/gemma-3-4b-it (34 layers, 2560 hidden dim)
**Experiment ID:** 446690dc-5503-4628-aa17-ee3f750b8432
**Entity library:** 42 prompts spanning 13 countries, 12 people, 5 literary works, multiple relations

---

## Background

Layer 14 of Gemma 3 4B encodes entity identity in a direction orthogonal to all 262,000 vocabulary embeddings. The logit lens reads "cities" for both Japan and France. Shakespeare and Dante are invisible. Yet the signal is causally complete — patching it at L14 changes the output with KL divergence of zero.

We have proved the dark space exists and measured its causal power. This experiment enters it and maps what's inside.

---

## Experiment 1 — The Dark Atlas

### PCA of 42 entity prompts at L14

**Effective dimensionality:**
- 50% of variance: **1 dimension**
- 80% of variance: **3 dimensions**
- 90% of variance: **5 dimensions**
- 99% of variance: **28 dimensions**

### The dominant axis is completely dark

**PC00 — 69.9% of all variance — is orthogonal to the vocabulary.**

Positive tokens: `<unused687>`, `<unused1077>`, `<unused201>`, `<unused515>` — tokens never assigned meaning during training.
Negative tokens: `ꗜ`, `𒉶`, `<unused338>`, `𒅤`, ` PLDNN` — Unicode garbage from obscure scripts.

No shadow vocabulary. PC00 does not have "faint projections" onto semantically meaningful tokens. It is genuinely alien to vocabulary space — encoding something with no partial projection onto any real token. This is the **factual query format attractor**: the signal common to all 42 prompts that registers "I am processing a structured entity query." Every country-capital, person-birthplace, and work-author prompt in the library shares this dark signal.

**Total dark variance: ~75–80%.** Total lit (vocabulary-decodable) variance: ~20–25%.

### The lit PCs are smaller and entity-specific

| PC | Variance | Vocabulary decode | Interpretation |
|----|----------|-------------------|----------------|
| PC01 | 8.2% | neg: *playwright, literary, writer*; pos: *Agriculture, Munic* | Entity TYPE separator — literary vs geographic |
| PC02 | 7.0% | pos: Russian verbs (*проведен, проводи*) | Russian/language signal |
| PC04 | 2.2% | pos: *जाप, Vladimir* | South Asian / Eastern European |
| PC09 | 0.5% | pos: *Yorkshire, Chichester* | British places |
| PC11 | 0.4% | pos: *Eisen* (Einstein fragment); neg: *egyptian* | Person-specific fragments |
| PC18 | 0.2% | pos: *German, Queensland, Vermont* | Geographic/national |
| PC26 | 0.1% | neg: *অস্ট্রেল* (Australia in Bengali) | Australia-specific |
| PC27 | 0.1% | neg: *Japan, Tokyo, Osaka, Japanese* | Japan-specific |
| PC33 | 0.08% | pos: *Tunisia, Thailand, Thai* | Thailand-specific |
| PC34 | 0.07% | neg: *üsseldorf* (Düsseldorf) | German city |

The entity-specific lit signals are 0.1% each — **700× smaller** than the dark PC00. Individual country identity lives in tiny vocabulary-accessible dimensions, while the dominant computational work happens in the dark.

### The shadow vocabulary question

PC00 has **no shadow vocabulary.** It is not "approximately orthogonal" to all tokens with some tokens at 88° and others at 92°. The decode shows only unused tokens and garbage Unicode — the unembedding has essentially no opinion on the PC00 direction at all. This is dark not because the concepts are encoded in near-null directions, but because the concepts encoded have no vocabulary-space correlate whatsoever.

---

## Experiment 2 — The Dark Manifold Has Structure

The dark space is not shapeless. It organizes entities along clear axes.

### Three-way entity type separation

The primary topology in the dark space at L14 separates entity **type**:

| Comparison | Cosine similarity | Angle |
|-----------|-------------------|-------|
| Country vs Country (same relation) | 0.9997–0.9999 | 0.6–1.5° |
| Person vs Person (same relation) | 0.9993–0.9997 | 1.5–2.0° |
| Work vs Work (same relation) | 0.9995–0.9997 | 1.5° |
| Country vs Person | 0.9983–0.9985 | **3.2–3.5°** |
| Country vs Work | 0.9983–0.9986 | **3.0–3.4°** |
| Person vs Work | 0.9983–0.9985 | **3.2–3.5°** |

The cross-type angle (3°–3.5°) is 2–3× the within-type angle (0.6°–2°). A 2D PCA of L14 residuals places countries in one quadrant, birthplace-of-person in another, and author-of-work in a third — clean separation across all three domains.

Centroid distances confirm the structure:
- Countries only: **0.000203** (tightest)
- People (birthplace): **0.000478** (2.4× more spread)
- Mixed entity types: **0.001260** (6× more spread)

### Relation creates more variance than entity identity

At L14, the dark space is more sensitive to **what you're asking** than **who you're asking about**:

- Same entity, different relation (Japan-capital vs Japan-language): **2.9°** apart
- Different entity, same relation (Japan vs France, both capital): **1.1°** apart

The relation template leaves a larger footprint in the dark space than entity identity does. This is consistent with PC01 (the first vocabulary-readable axis) separating literary from geographic entities — the *kind of question* being asked is the dominant vocabulary-legible signal.

### No geographic clustering at L14

All country pairs are roughly equidistant:

| Pair | Distance at L14 |
|------|----------------|
| France–Germany | 0.000127 |
| Japan–China | 0.000106 |
| France–Japan | 0.000153 |
| France–Australia | 0.000272 |

Geography does not exist in the dark space. France is not closer to Germany than to Japan. The dark manifold encodes ontological category (what kind of entity) but not semantic content (which specific entity, where it is, what it does).

### No cultural clustering at L14

Shakespeare–Goethe (both European literary figures): 0.9996. Shakespeare–Kafka: 0.9994. No meaningful clustering by cultural affiliation, language, or historical era in the dark space.

---

## Experiment 3 — The Shadow Vocabulary

As established above, PC00 has **no shadow vocabulary** — its unembedding decode is pure garbage with no semantic pattern. The dark PC00 is not encoding familiar concepts in dim directions; it is encoding something with no vocabulary-space representation whatsoever.

The lit PCs with vocabulary-decodable content (PC27 for Japan, PC26 for Australia, PC33 for Thailand) are genuine vocabulary-space signals — small dimensions where the model's country-specific knowledge projects onto recognizable tokens. These are not dark. They are simply small.

The architecture at L14 thus has two layers:
1. **One large dark attractor** (PC00, 70%) — the "I am processing an entity query" signal, shared universally, invisible to output
2. **Many small lit signals** (0.1% each) — the country/person-specific vocabulary projections, entity-distinguishing but tiny

---

## Experiment 4 — The Viewport Is a Ramp, Not a Door

The dark-to-lit transition is **continuous** across layers, not a discrete event at L26:

| Layer | Centroid spread (mixed group) | Amplification from L14 |
|-------|-------------------------------|------------------------|
| L14 | 0.001260 | 1.0× |
| L18 | 0.003338 | 2.6× |
| L22 | 0.010178 | 8× |
| L24 | 0.022852 | 18× |
| L26 | 0.038328 | 30× |

For individual entity types: countries amplify 89× from L14 to L26; people amplify 48×.

**The three-way entity type topology (country / person / work) is structurally identical from L14 through L26** — just 30× smaller at L14. Countries are upper-left, birthplace-people are right, works-author are a third quadrant at every layer from 14 to 26. The dark geometry is not replaced by the lit geometry; it is amplified into it. The viewport does not translate dark to lit — it *scales* dark into lit.

---

## Experiment 5 — Two Different Geometries

The rank ordering of entity distances **changes** from dark (L14) to lit (L26):

| Pair | L14 distance | L26 distance | Structure at L26 |
|------|-------------|-------------|-----------------|
| France–Germany | 0.000127 | **0.0077** | Neighbors — closest European pair |
| France–Italy | ~0.0002 | **0.0082** | Neighbors — close |
| Japan–China | 0.000106 | **0.0128** | Regional — medium |
| France–Japan | 0.000153 | **0.0123** | Intercontinental — medium |
| France–Australia | 0.000272 | **0.0236** | Most distant |

At L14: all country pairs approximately equidistant, no geographic structure.
At L26: France–Germany is the closest pair; Japan–China cluster in one region; Australia is the largest outlier.

The L26 PCA 2D shows a recognizable geography: East Asian cluster (Japan, China upper-right), European cluster (France, Germany, Italy, Russia left), with Australia and Brazil as distinctive outliers.

**The viewport reorganizes entity relationships.** The dark geometry encodes entity type topology. The lit geometry encodes entity content topology — geographic proximity, cultural similarity. These are two genuinely different organizations of the same entities in the same ambient space. The viewport doesn't just amplify — it imposes a new semantic structure that was absent at L14.

---

## Experiment 6 — Navigating in the Dark

**The residual streams of Japan-capital and Australia-capital at L14 are 1.16° apart.**

They are virtually indistinguishable to the unembedding. Both read as the same dark PC00 attractor signal. Yet:

**All-position injection of Japan's L14 residual into Australia's forward pass:**
- Recipient baseline: Canberra **92.0%**
- After injection: Tokyo **83.4%**
- KL(donor, injected) = **0.0** — injected output is identical to pure Japan output

A 1.16° rotation in the dark — invisible to the output layer — **completely overwrites 20 layers of subsequent computation**. L15 through L33 faithfully read the new dark coordinates and output Tokyo. This is dark navigation: enter the space, rotate 1.16°, exit with a different answer.

**Last-position-only injection at L14:** Canberra 82.3% (essentially unchanged). Cross-entity navigation requires all-position Markov state replacement. The entity identity information at L14 lives in the context-wide residual, not just the last token's dark signal.

**Steering vector (alpha=5, 30):** Garbage output at both scales. Adding a single directional perturbation to the stream at every generation step disrupts coherence without achieving entity substitution. Dark navigation works through all-position context replacement, not through additive perturbation in a single direction.

**What this means:** The entity coordinates at L14, across all token positions, are causally complete. Once the dark coordinates are established, the downstream computation has no escape — it reads what the dark space tells it. The 1.16° rotation is not a hint or a nudge; it is a complete rewrite of the entity's computational identity, processed faithfully by all 20 subsequent layers.

---

## Experiment 7 — The Dark Compass for Failures

### Fictional-character birthplace queries

| Prompt | Type probe (L14) | Identity probe (L14) | Model output |
|--------|-----------------|---------------------|-------------|
| "The birthplace of Hamlet was" | person (93.4%) | **Shakespeare (99.96%)** | *a / the / not / never* (diffuse) |
| "The birthplace of Faust was" | person (99.7%) | Goethe (42.9%) | *the / a / not / Germany (2.4%)* |
| "The birthplace of Don Quixote was" | person (99.7%) | Cervantes-adjacent | *a / the / Madrid (4.3%) / Spain (2.5%)* |
| "The birthplace of Shakespeare was" | person (100%) | Shakespeare (100%) | Stratford (clean) |
| "The author of Hamlet was" | work (100%) | Shakespeare (91.1%) | Shakespeare (clean) |

**The dark entity coordinates for "The birthplace of Hamlet was" encode Shakespeare — not Hamlet — with 99.96% confidence.**

The dark compass fires on the *relevant entity*, not the surface token. Hamlet is a literary work; the relevant entity for any birthplace question about it is its author. The L14 dark space correctly identifies this as Shakespeare.

The entity **type** probe classifies it as "person" (93.4%) — correct for Shakespeare, triggered by the "birthplace of" relation template regardless of whether the subject is fictional.

**probe_at_inference at L14:** Before any generation begins, the probe reads Shakespeare at 99.96%. The dark entity compass is active on the initial prompt. As the model generates confused output, the probe degrades — but the initial dark state is correct.

**Why the output hedges instead of confidently hallucinating:** The model output distribution is diffuse (*a, the, not, never, once*) rather than a confident wrong city. This is because the training distribution includes corrections and hedges for fictional birthplace queries — the model has encountered "Hamlet is a fictional character" in training. The dark space did its job correctly; the output mechanism is uncertain about how to respond, not systematically wrong.

**For Faust and Don Quixote:** Faint author-country signals leak through (Germany 2.4% for Faust; Spain 2.5% and Madrid 4.3% for Don Quixote). The dark entity coordinates for these fictional-character birthplace queries partially route to the correct author's geography.

### Depth of the failure

**The dark coordinates are intact.** The dark entity compass correctly encodes the author (Shakespeare, Goethe, Cervantes-adjacent) for all fictional-character birthplace queries. The failure is entirely in the output mechanism — the vocabulary output cannot locate a crisp city because the training distribution provides no confident "birthplace of Hamlet" answer.

This is distinct from Stage 3a failure (L26 FFN conflation), where the dark coordinates are correct but the viewport produces a confident wrong city. Here the dark coordinates are correct AND the output refuses to commit. The dark space succeeded; the output mechanism hedged.

---

## Experiment 8 — Reading the Dark Directly

### Can an external classifier read entity identity from L14?

**Entity type probe at L14** (country / person / work):

| Metric | Value |
|--------|-------|
| Training accuracy | 100% |
| Validation accuracy | **97.1%** |
| Probe weight norm | 0.027 |

Held-out results: capital of Greece → country (100%), birthplace of Cervantes → person (100%), author of Romeo and Juliet → work (100%), author of Brothers Karamazov → work (99.99%).

Failure pattern: "The birthplace of [fictional character]" → person (93–99.7%). The probe classifies by relation template ("birthplace of" = person-type query) rather than by entity ontology. This is not a probe failure — it is the correct reading of the dark state, which itself is template-driven.

**Entity type probe at L26**: 100% validation accuracy, probe weight norm **0.004** (6× smaller than L14). The signal is more amplified at L26, making it easier to classify with a shorter probe direction.

---

**Entity identity probe at L14** (19 specific entities — countries, people, authors):

| Metric | Value |
|--------|-------|
| Training accuracy | 100% |
| Validation accuracy | **100%** |
| Classes | 19 (Australia, France, Japan, Germany, Brazil, Italy, Egypt, Russia, Shakespeare, Einstein, Beethoven, Darwin, Kafka, Curie, Dante, Dostoevsky, Goethe, Joyce, Cervantes) |
| Probe weight norm | 0.145 |

Held-out results:
- "The author of Romeo and Juliet was" → **Shakespeare (91.1%)** — novel entity not in training, generalized correctly from "The author of Hamlet was"
- "The birthplace of Hamlet was" → **Shakespeare (99.96%)** — encodes the author, not the fictional character
- "The birthplace of Faust was" → Goethe (42.9%) — weakly but correctly

### The bypass hypothesis — confirmed

**An external linear probe at L14 classifies specific entity identity across 19 distinct entities with 100% accuracy.**

The same 2560-dimensional activations. The model's unembedding reads them as `<unused687>` and Unicode garbage. A linear probe reads the same vectors and says "Japan," "Shakespeare," "Einstein" with certainty.

The dark dimension is dark **only to the model's own decoder.** An external observer with a trained linear probe has full read access to entity identity at L14. The asymmetry:

| Decoder | Reads entity identity? | Notes |
|---------|----------------------|-------|
| Model unembedding | No | PC00 reads as unused tokens |
| Linear probe (trained) | Yes, 100% accuracy | Same activations |

**14 layers is sufficient for complete entity identification.** The viewport (L15–L33) is not needed for this. It adds geographic topology and confidence amplification — but the identity was already there, written in the dark. All of the model's failure modes (Stage 3a L26 FFN conflation, Stage 3b L33 late overwrite, template competition) occur in the viewport. The dark space is reliable.

---

## Summary: What the Dark Dimension Contains

| Signal | Location | Who can read it | Notes |
|--------|---------|-----------------|-------|
| Factual query format | PC00 (70%, DARK) | Probe only | "I am an entity query" — shared by all 42 prompts |
| Entity TYPE | PC01 (8.2%, LIT) | Probe + unembedding | Literary vs geographic at L14; 30× amplification to L26 |
| Entity IDENTITY | Scattered dark+lit PCs | **Probe only** — 100% accuracy | Unembedding sees garbage |
| Country-specific lit signals | PC27/PC33/PC26 (0.1%, LIT) | Unembedding (barely) | Japan → PC27, Thailand → PC33, Australia → PC26 |
| Geographic proximity | Absent at L14 | Unembedding at L26 only | Viewport imposes it; not in dark |

---

## The Structure of the Dark Dimension

The dark dimension at L14 is not noise. It is not residual artifact from training. It is the **primary computational medium of entity processing** — the space where the model thinks before it speaks.

Its structure:

**One dominant attractor (PC00, 70%)** — the factual query format, shared universally across all entity types, invisible to the output layer. Every entity query the model has ever seen converges to this direction. It is the "I am processing a question about an entity" state, written in dimensions the vocabulary cannot touch.

**Three-way type topology (entity type, 3–4°)** — countries, people, and works are separated in the dark space. The separation is real, consistent from L14 to L26, and detectable by a probe at 97%+ accuracy. This ontological structure is present before any specific entity identity is resolved.

**Fine-grained identity information (19 entities, 100% probe accuracy)** — specific entity identity is encoded in the L14 dark residual with enough precision for a linear probe to distinguish 19 entities perfectly. The probe generalizes across relations (it learns "Shakespeare" from Hamlet-authorship and correctly identifies Romeo-and-Juliet-authorship). The dark entity compass points to the *relevant entity*, not the surface token — Shakespeare for Hamlet-birthplace, Goethe for Faust-birthplace.

**No geography (absent at L14)** — geographic proximity does not exist in the dark space. France–Germany are not closer to each other than France–Japan. Geography is a property of the lit space (L26), imposed by the viewport, absent from the dark.

**Causal completeness** — a 1.16° rotation in the dark space, spanning 25,000-norm residuals in 2560D, completely overwrites 20 layers of subsequent computation. The viewport reads the dark coordinates faithfully. Enter the dark space, rotate slightly, exit with a different answer.

Dr Strange found cities and landscapes in the dark dimension. We found entity type topology, specific identity information at 100% linear probe accuracy, and a dominant format attractor encoding the act of factual reasoning itself — all invisible to the model's own output layer, all readable by a trained observer, all causally sufficient for determining what the model says. The dark dimension has structure. It has geometry. And it can be navigated.
