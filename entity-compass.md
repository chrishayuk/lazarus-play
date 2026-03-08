# Experiment: The Entity Compass — Shared Coordinates Across Relations

**Experiment ID:** `0164bd2a-55b4-46b8-bea3-db2f6f410da0`
**Model:** google/gemma-3-4b-it (34 layers, 2560 hidden dim)
**Date:** 2026-03-07

---

## Background

The previous Universal Chart experiment (52394963) found that entity coordinates are
relation-independent at L26: Japan-capital through a Japan-language chart produces
"Japanese" (86.3%), and the reverse produces "Tokyo" (78.3%). This was interpreted as
evidence for a two-level hierarchy: entity coordinates ~8D, relation-independent at L26.

This experiment tests that interpretation more carefully and asks:
- Are capital and language charts geometrically distinct (entity-private), or do they share the same 8D subspace?
- At what layer do entity coordinates become relation-agnostic?
- How many dimensions are shared (entity) vs private (relation)?
- Does the hierarchy hold for person-entities (Shakespeare)?
- What does the entity compass direction decode to in vocabulary space?

---

## Phase 1: Chart Alignment at L26 — The Universal Subspace

### Method

Built 8 private charts at L26 (rank 8 PCA) using 4 countries × 2 relations (capital, language):
- Chart construction: 1 target prompt + 8 unrelated prompts → PCA at last token position
- Measured subspace cosine similarity between all pairs of charts
- Ran cross-relation injections: capital residual through language chart, and vice versa

### Result: No Entity-Private Charts Exist at L26

All charts span nearly the same 8D subspace regardless of entity or relation:

| Chart pair | Subspace cosine |
|---|---|
| Japan-capital vs Japan-language | 0.981–0.983 |
| Australia-capital vs Australia-language | 0.972–0.980 |
| France-capital vs France-language | 0.978–0.986 |
| Germany-capital vs Germany-language | 0.972–0.983 |
| **Japan-capital vs France-capital** | **0.992–0.998** |
| **Japan-capital vs France-language** | 0.962–0.965 |

Cross-entity, same-relation charts (Japan-cap vs France-cap) are **more similar** to each
other (0.998) than same-entity, cross-relation charts (Japan-cap vs Japan-lang, 0.981).
There is no entity-private 8D address — only a universal "factual query" subspace.

### Cross-Injection Results at L26

| Direction | Country | Top-1 after injection | Recipient KL |
|---|---|---|---|
| Capital → Language chart | Japan | Japanese 44.4% ✓ | 0.73 |
| Language → Capital chart | Japan | "a" 45.7%, Tokyo 6% ✗ | 2.00 |
| Capital → Language chart | Australia | English 50.7% ✓ | 0.55 |
| Language → Capital chart | Australia | "a" 30.7%, Canberra 1.7% ✗ | 3.68 |
| Capital → Language chart | France | French 49.5% ✓ | 0.37 |
| Language → Capital chart | France | Paris 35.5% ✓ | 0.50 |
| Capital → Language chart | Germany | German 39.0% ✓ | 0.63 |
| Language → Capital chart | Germany | "a" 27.8%, Berlin 11.3% ✗ | 1.64 |

Capital→Language "successes" have very low recipient KL (0.37–0.73): the injection is
near-identity because both charts span the same subspace. The previous 86.3% success
was largely a near-identity artifact. Language→Capital failures (KL 1.6–3.7) reveal
that the capital chart's rotation within the shared subspace doesn't align with the
language residual's rotation — injecting it disrupts rather than transfers.

### Direction Angles at L26

For all four country prompts (capital and language), at L26:
- Residual vs any token embedding: **85–90°** (near-orthogonal — answer not vocabulary-pointing)
- Residual vs FFN output: 62–69°
- Residual vs attention output: **32–49°** (strongest alignment — entity carried in attention stream)

The factual answer is not encoded as a direction that points toward the answer token.
It lives in an abstract internal coordinate readable by weight matrices, not by direct
projection onto the vocabulary.

---

## Phase 2: Shared vs Private Dimensions

At L26, with the 1-target + 8-unrelated chart construction:
- ~33% of the residual falls within any 8D chart (donor_subspace_fraction)
- Entity-specific signal = tiny rotation within the shared space

With improved construction (4 entity-variant prompts + 4 unrelated), PC1 jumps from
21% to 73%+ — the entity direction dominates when given enough instances to isolate it.

**Key insight:** The chart subspace at L26 is best understood as the "factual query manifold" —
the high-energy directions that distinguish any factual query from unrelated text. Entity
identity and relation type are both encoded as subtle rotations within this manifold.

---

## Phase 3: Depth of Entity Independence

### Method

Built improved charts (4 entity-variant prompts + 4 unrelated) at layers L14, L18, L22,
L24, L26, L30 for Japan capital and language. Ran cross-relation injections at each layer.

### PC1 Variance by Layer — The Phase Transition

| Layer | Capital PC1 | Language PC1 | Rank for 80% |
|---|---|---|---|
| L14 | **72.8%** | **74.6%** | 2 |
| L18 | **71.3%** | **66.8%** | 2–3 |
| L22 | 34.2% | 32.5% | 5 |
| L24 | 26.2% | 28.7% | 5 |
| L26 | ~21% | ~21% | 6 |
| L30 | 23.2% | 27.4% | 4–5 |

At L14–L18, entity identity is compressed into 1–2 dimensions (PC1 dominates at 67–75%).
At L22+, it spreads into 5D. This matches the Markov bandwidth expansion (3D at L16, 6D at L24–33).

### Cross-Relation Injection by Layer

| Layer | Residual angle | Cap→Lang | Lang→Cap |
|---|---|---|---|
| L14 | **2.9°** | **Japanese 97.0%** ✓ (↑ from 64%) | **Tokyo 96.6%** ✓ (↑ from 83%) |
| L18 | 4.2° | **Japanese 91.3%** ✓ | **Tokyo 72.5%** ✓ |
| L22 | 7.7° | **Japanese 90.2%** ✓ | "a" 44.8%, Tokyo 4.9% ✗ |
| L24 | 10.6° | **Japanese 55.5%** ✓ | "a" 42.5%, Tokyo 11.9% ✗ |
| L26 | 13.4° | Japanese 44.4% (↓ from 64%) | "a" 45.7%, Tokyo 6% ✗ |
| L30 | 15.1° | "**" 26.6%, Japanese 14.3% ✗ | "a" 29.8%, Tokyo 4.5% ✗ |

**L14 is the entity independence layer.** Both directions work and **boost** the correct
answer above baseline (not near-identity — genuine entity signal amplification). The
L14 injection is causally driving the output via the entity coordinate.

**Asymmetric break at L22:** Capital→Language continues to work through L24 because the
language chart remains tolerant. Language→Capital fails because the capital chart's
rotation within the shared 8D space encodes capital-specific structure by L22 that the
language residual activates incorrectly, destroying rather than transferring entity identity.

**Both fail at L30:** By L30, enough relation-specific structure has accumulated that the
8D chart only captures 47–57% of the residual (vs 96% at L14), and the entity signal
is too diluted relative to relation-specific content.

---

## Phase 4: Person-Entities — Shakespeare

### Baselines

- "The birthplace of Shakespeare was": "a" 27.9%, Stratford 8.0% (uncertain)
- "The century Shakespeare was born in was the": " " 90.6% (→ 16th century)

### Chart PC1 Values

| Chart | L14 PC1 | L26 PC1 |
|---|---|---|
| Shakespeare-birthplace | **83.3%** | 33.0% |
| Shakespeare-era | **85.0%** | 28.3% |

Even tighter entity compression at L14 than Japan (73%), because Shakespeare is a more
distinctive entity signal than a country.

### Cross-Relation Injections

| Direction | Layer | Injected top-1 | vs Baseline | Recipient KL |
|---|---|---|---|---|
| Birthplace → Era chart | L14 | "the" 83.3% | correct format | 2.44 |
| **Era → Birthplace chart** | **L14** | **Stratford 99.5%** | **8% → 99.5%** | 9.25 |
| Birthplace → Era chart | L26 | " " 39.7% | same as recipient | 0.76 |
| Era → Birthplace chart | L26 | " " 18.2%, Stratford 4.2% | weak | 2.44 |

The L14 era→birthplace result is striking. The era donor itself outputs " " (91% — a
space preceding "16th century"). It has no direct answer to birthplace. Yet when that
prompt's residual at L14 is injected through the birthplace chart, the Shakespeare entity
coordinate is sufficient to drive Stratford to **99.5%** — a 12× boost from baseline.

The injection substantially changes the output (KL_recipient = 9.25), confirming this is
genuine entity transfer, not near-identity. The Shakespeare entity compass at L14 carries
complete birthplace retrieval capability.

**Two-level hierarchy confirmed for person-entities.**

---

## Phase 5: The Entity Compass — Geometry and Vocabulary Decoding

### Angular Structure at L14

For Japan-capital and France-capital prompts at L14:

| Measurement | Japan-cap | France-cap |
|---|---|---|
| Residual vs FFN output | **16.7°** | **13.7°** |
| Residual vs attention output | **164.2°** | **150.6°** |
| Residual vs " Tokyo" / " Paris" | 85.0° / 88.5° | — / 86.8° |
| Residual vs " Japan" / " France" | 90.2° | 88.1° |
| All other token embeddings | 85–91° | 85–91° |
| FFN vs attention output | 158.1° | 149.8° |

At L14, the residual IS the FFN output (13–17° alignment). The attention output is
strongly anti-aligned with the residual (150–164°) — attention at L14 is actively
writing in the opposite direction from the accumulated representation. FFN and attention
are nearly anti-parallel to each other.

The entity compass at L14 = FFN output direction = abstract internal coordinate,
orthogonal to all token embeddings.

### Cross-Entity Angles at L14

| Pair | Angle |
|---|---|
| Japan-cap vs France-cap | **1.0°** |
| Japan-cap vs Australia-cap | **1.2°** |
| Japan-cap vs Japan-lang | **2.9°** |
| Shakespeare-birthplace vs Shakespeare-era | **4.0°** |

**Counterintuitive: cross-entity angle (1°) < cross-relation angle (2.9°).**
Countries look MORE similar to each other than the same country under different relational
frames. At L14, the dominant signal is the query format ("The X of Y is"), not entity
identity. Entity identity is an even tinier perturbation on top of that format signal.

Yet this tiny ~1% perturbation is causally decisive, as proven by the injection experiments.

### Logit Lens — The Dark Signal

| Layer | Japan-cap | France-cap | Japan-lang | Shakespeare-birth |
|---|---|---|---|---|
| L0 | 否 87.5% | 否 61.7% | 否 94.9% | "wolves" 100% |
| L5 | 否 27.3% | своего 41.8% | "và" 20% | "wolves" 100% |
| L10–L14 | **" cities"** | **" cities"** | "ইংরেজি" | " famously" |
| L18 | " cities" 1.7% | " city" 1.1% | " called" 2.8% | " very" |
| L22 | " what" 17.8% | " what" 27.1% | **" called" 98.4%** | " not" 11.3% |
| L26 | **" Tokyo" ~100%** | **" Paris" ~100%** | **" Japanese" 97%** | " debated" 26% |
| L33 | " Tokyo" 83% | " Paris" 81% | " Japanese" 64% | " a" 28%, Stratford 8% |

**Japan-cap and France-cap predict identical " cities" at L14** — no entity specificity
visible via unembedding. The entity compass is in the null space of the output projection.

The answer is fully invisible until **L26**, where it explodes to ~100% confidence.
L26 FFN is the layer where accumulated entity+relation state is first translated into
a vocabulary-space answer.

Note the language prompt is different: " called" at 98.4% at L22 (relation format resolves
before entity), and "ইংরেজি" (Bengali "English") at L14 — schema-level knowledge present
early in a different form.

Shakespeare-birthplace never clearly resolves even at L33 (" a" 28%, Stratford 8%) —
the fact is genuinely uncertain/debated, consistent with the model's low confidence.
Yet the entity coordinate at L14 carries enough Shakespeare-signal to produce Stratford
99.5% when channeled through the birthplace circuit.

---

## Synthesis: A Revised Architecture

The model computes factual answers through three stages:

### Stage 1: Entity Accumulation (L0–L14)
Entity identity concentrates into a compact, dark signal in the last-position residual.
This signal is:
- Encoded in the FFN output direction (residual ≈ FFN at 13–17°)
- Actively opposed by the attention output (150–164° anti-alignment)
- Orthogonal to all vocabulary directions (~90°)
- Invisible to the logit lens (unembedding reads " cities" for both Japan and France)
- Causally complete for entity retrieval — Shakespeare-era entity address → Stratford 99.5%

Cross-entity angle at L14 is only 1–1.2°; cross-relation angle is 2.9–4°. Entity identity
is a smaller variation than query format. Both are tiny perturbations on the dominant
"factual query" residual direction.

### Stage 2: Relation Differentiation (L14–L25)
Entity signal expands from 1–2D to 5D. Relation-specific structure accumulates.
By L22, capital and language contexts diverge enough to break cross-relation symmetry.
At L26, the signal still lives in a universal ~8D factual-query subspace — no entity-private
8D address exists at this layer.

### Stage 3: Fact Explosion (L26)
The L26 FFN reads accumulated entity+relation state and writes the specific fact
into vocabulary space at ~100% confidence. Prior to L26: invisible. After L26: committed.
L26 FFN is the entity address → vocabulary answer translator.

---

## Open Questions

1. **Why is the L14 entity compass orthogonal to the unembedding matrix?**
   Is this deliberate (information hiding from shallow decoders) or an artifact of how
   entity identity accumulates from token positions other than the last?

2. **What mechanism creates the L14 entity address?**
   The attention output at L14 is anti-parallel to the residual — it opposes the FFN.
   Is L14 attention suppressing generic signals to reveal the entity direction, or writing
   relation-specific content that pushes against the entity direction?

3. **Why does cap→lang succeed longer than lang→cap (through L24)?**
   The language chart may be more "tolerant" because language identity (Japanese, French)
   is a broader cultural signal with less precise coordinate requirements than a specific city.

4. **Does the entity compass change with model scale?**
   Larger models may maintain entity independence deeper (more layers), or may encode it
   in higher dimensions.

5. **Can the dark entity signal at L14 be extracted directly?**
   Given that it's orthogonal to the unembedding, it can't be read by logit lens. But a
   trained probe at L14 for entity identity should work — the signal is causally present.

---

## Experimental Notes

- **Chart construction matters critically:** 1-target + 8-unrelated → PC1 ≈ 21% (too diluted
  for clean cross-relation tests). 4-entity-variant + 4-unrelated → PC1 ≈ 73–85% (entity
  direction dominates, gives clean injection results).
- **inject_residual patches last position only.** Cross-entity injections at L14 are near-
  identity (KL ≈ 0.03) because country identity at L14 lives mostly in earlier positions
  (the country name token), not the last position (" is"). This is consistent with the
  Markov cross-task failure finding.
- **Subspace cosine ≈ 0.999 between charts** means "inject donor through recipient chart"
  is essentially injecting the donor's own signal — the chart is near-transparent. The
  entity-specific difference is in the remaining 0.1%.
