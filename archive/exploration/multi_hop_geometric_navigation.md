# Multi-Hop Geometric Navigation: Can Residual Hop-Chaining Replace Chain of Thought?

**Experiment ID:** 25f2177a-f68a-44a4-8a5e-7fbd64eaf493
**Model:** google/gemma-3-4b-it (34 layers, 2560 hidden dim, 8 heads, bfloat16)
**Date:** 2026-03-06

---

## Executive Summary

We tested whether chaining geometric operations on the residual stream can solve
multi-hop compositional problems without generating reasoning tokens. Eight experiments,
starting from the hypothesis that "portal jumps" (cached residual injections) could
replace CoT's multiple forward passes.

**What we found:**

1. No shared "hop vector" exists. Each hop is entity-specific and irreducibly so.
2. Last-position injection fails for multi-hop because Head 1 reads entity identity
   from all token positions, not just the last.
3. Full-sequence injection at L14 (patch_all_positions) always succeeds — but this
   is a query transplant, not a hop chain. It answers the donor's question, not the
   recipient's.
4. A true Geometric CoT pipeline works with an oracle planner, producing correct
   answers in ~33 layers vs 34 for (incorrect) direct inference. The savings are
   minimal: 1 layer with a cached donor library.
5. The model already partially performs relational hops inside Head 1 at L24. Type B
   failures are not misdirection — they are insufficient confidence. Amplifying Head 1
   × 2 flips the output from wrong to right.
6. The planner cannot be built from internal model representations. The probe achieves
   90% accuracy at layer 0 — detecting surface syntax, not computational depth.

**The central finding:** Geometric CoT works. It is not faster than CoT. CoT generates
intermediate results as tokens, making them available at all attention positions.
Geometric injection can only replace wholesale forward passes, not extract and re-use
partial results from within a single forward pass. The token-generation mechanism in
CoT provides something that residual manipulation cannot: context-agnostic intermediate
results that can be re-read by all attention heads in the next hop.

---

## Experiment 1 — Single Hop Catalogue

### Setup

Ten prompts, all answered correctly by Gemma-3-4b-it:

| Prompt | Answer | Type |
|--------|--------|------|
| The capital of Germany is | Berlin (89%) | geo-capital |
| The capital of Belgium is | Brussels (58%) | geo-capital |
| The capital of Japan is | Tokyo (83%) | geo-capital |
| The capital of Australia is | Canberra (92%) | geo-capital (contested) |
| The capital of France is | Paris (81%) | geo-capital |
| The capital of Italy is | Rome | geo-capital |
| Beethoven was born in | Bonn (98%) | biographical |
| Shakespeare was born in | Stratford (99%) | biographical |
| The language spoken in Brazil is | Portuguese | lang-query |
| The country north of France is | Belgium | geo-relational |

### L23 → L26 Geometry

The most important structural fact about single hops:

**Capital prompts at L23 are nearly identical** (cos similarity 0.994–0.999,
centroid distance = 0.0026). They all share the template "The capital of X is" and
have not yet differentiated. After the hop:

**Capital prompts at L26 have diverged** (cos similarity 0.970–0.993, centroid
distance = 0.0164 — 6.3× larger). Each prompt now encodes its specific answer.

The hop is a **differentiation process**: entity-generic → entity-specific. The shared
template information at L23 remains but entity-specific answer content is injected.

Non-capital prompts at L23 are already more differentiated (centroid 0.026) and diverge
further at L26 (centroid 0.056). These prompts have distinct templates even before
the hop.

### Answer Subspace at L26

The answer token occupies an extremely small fraction of the residual:

| Prompt | Answer fraction of residual | Angle to residual |
|--------|----------------------------|-------------------|
| Capital of Germany | 0.138% (Berlin) | 87.9° |
| Capital of France | 0.307% (Paris) | 86.8° |
| Capital of Japan | 0.756% (Tokyo) | 85.0° |

Total answer-token subspace across 4 basis tokens: 0.35%–0.81%. The answer lives at
~85–88° to the residual — nearly perpendicular. The remaining 99%+ of the residual
encodes query context, template structure, and position.

### Answer Emergence Layer

Capital queries crystallise sharply and early:

- Berlin: rank 15 at L23 (1.1%) → **rank 0 at L24 (83.6%)** → 100% at L26
- Tokyo: rank 0 at L23 (16.2%) → rank 0 at L24 (89.5%) → 100% at L26

Biographical queries emerge late and gradually:

- Bonn: rank 556 at L22 → rank 52 at L24 → rank 5 at L26 → **rank 0 at L28 (47.1%)**
  → top-1 at L33 (98.4%)

The hop layer is not fixed at L24. It is prompt-type-dependent. Capital queries rely
on Head 1's one-shot relational hop at L24. Biographical queries require late FFN work
through L28–L33.

### Attention / FFN Split at L24–L26

FFN dominates consistently across the key layers:

| Prompt | L24 Attn | L24 FFN | L25 Attn | L25 FFN | L26 Attn | L26 FFN | FFN/Attn ratio |
|--------|----------|---------|----------|---------|----------|---------|----------------|
| Germany → Berlin | +5.69 | +3.25 | −1.13 | +4.38 | +2.38 | +2.13 | 1.40× |
| France → Paris | +5.56 | +1.38 | −1.13 | +8.00 | +2.25 | +0.63 | 1.50× |
| Japan → Tokyo | +5.25 | +0.88 | −1.38 | +7.25 | +2.75 | +4.13 | 1.85× |
| Beethoven → Bonn | +1.13 | +1.31 | −0.94 | +5.25 | +1.63 | +1.50 | 4.45× |

L24 attention (~+5.5 for capital templates) is Head 1's contribution — the relational
bridge hop. L25 attention is consistently slightly negative (−1.0 to −1.4) across all
prompts. L25 FFN is the single largest per-layer contributor (+4.4 to +8.0) — it
amplifies the answer after Head 1 has identified it.

Beethoven's lower L24 attention (+1.1 vs +5.5) confirms that Head 1 fires weakly when
the answer is a rare biographical fact vs a common geo-capital.

---

## Experiment 2 — The Hop Vector

### Hypothesis

If all capital retrievals produce a similar displacement direction (L26 − L23), that
direction is a reusable "capital retrieval" hop vector. Adding it to a new prompt's L23
residual could produce the correct capital without running L24–L26.

### Direction Extraction

`extract_direction` at L26 (capital prompts vs non-capital prompts) achieves 100%
separation accuracy with score 4.49.

**Critical finding:** The direction at L23 (before any hop has occurred) achieves even
higher separation (score 5.02, accuracy 100%). The direction captured at L26 primarily
encodes the **template type** ("The capital of X is" vs other templates), not the
entity-specific hop result. The model already "knows" it's a capital query at L23 before
any computation has occurred.

### Portal Jump Tests (Last-Position, L26)

| Donor | Recipient | Injected | KL(donor,inj) |
|-------|-----------|----------|--------------|
| Germany → Berlin | Canada | **Berlin** 85.5% | 0.005 |
| France → Paris | Japan | **Paris** 79.3% | 0.002 |
| Australia → Canberra | Germany | **Canberra** 89.8% | 0.003 |
| Germany → Berlin | Poland | **Berlin** 87.7% | 0.001 |

All four inject the **donor's answer**, not the recipient's. Portal jumps at L26
are entity transplants: the entire answer state of "Germany → Berlin" overwrites
"Poland → Warsaw" with Berlin. The L26 residual IS the committed answer.

### Steering Test

Applying `capital_retrieval_L26` as a steering vector to "Beethoven was born in":
produces "located located located located located" — pathological repetition.
No usable hop direction exists.

### Conclusion

**The hop vector hypothesis fails.** No shared "capital retrieval" direction exists
that can be added to an arbitrary L23 residual to produce the correct capital.
Each hop is entity-specific. The answer configuration at L26 is not decomposable
into a shared "retrieval move" plus an entity-specific offset.

---

## Experiment 3 — Two-Hop Chaining

### Compositional Prompts (Direct Inference)

| Prompt | Direct Output | Correct Answer |
|--------|--------------|----------------|
| The country where Beethoven was born has its capital in | Vienna (46%) | **Berlin** |
| The country north of France has its capital in | Paris (44.5%) | **Brussels** |
| The birthplace of the author of Hamlet is | evasive (generic) | **Stratford-upon-Avon** |
| The language spoken in the country where Shakespeare was born is | format error | **English** |
| The language spoken in the capital of Australia is | **English** ✓ | English |
| The continent where Shakespeare was born is | **Europe** ✓ | Europe |

The model succeeds on easy 2-hop problems (where one hop is trivial) and fails on
harder ones. The "Beethoven capital" failure is a Type 2 confabulation — the model
confidently outputs Vienna (Austria, where Beethoven died) rather than Berlin.

### Method A — Last-Position Injection at L23

Inject "The capital of Germany is" L23 residual (last position) → Beethoven compositional prompt.

**Result: FAILS.** Injected output = Vienna (26.4%), not Berlin.
KL(donor, injected) = 1.09 — large divergence from donor.
KL(recipient, injected) = 0.998 — injected closely matches the original wrong answer.

**Why it fails:** Head 1 at L24 attends primarily to BOS (74%) and the entity token
in OTHER positions (~12%). The OTHER positions still contain "Beethoven" and "born",
which drive Vienna. Replacing only the last-position residual at L23 does not remove
the wrong entity from the positions that Head 1 actually reads.

### Method A — Last-Position Injection at L26

Inject Beethoven@L26 → Beethoven-compositional at L26: gives **Bonn** (99.6%).

This copies the first-hop answer (Bonn), not the second-hop answer (Berlin). Injecting
the sub-query result at the hop-completion layer just reproduces that sub-query's output.

Inject Germany@L26 → Beethoven-compositional at L26: gives **Berlin** (83.5%) ✓

This works — but it requires knowing Berlin is the answer. Not a hop chain; a cached
answer transplant.

### Full-Sequence Injection (patch_all_positions)

| Donor | Recipient | Layer | Injected | KL |
|-------|-----------|-------|----------|-----|
| Capital of Germany | Beethoven-compositional | L23 | **Berlin** 88.7% | 0.0 |
| Capital of Belgium | North-France-compositional | L23 | **Brussels** 57.6% | 0.0 |
| Capital of Germany | Beethoven-compositional | L14 | **Berlin** 88.7% | 0.0 |
| Capital of Belgium | North-France-compositional | L14 | **Brussels** 57.6% | 0.0 |

All four succeed with KL=0.0 — perfect Markov transfer.

**But this is not hop chaining.** Full-sequence injection replaces ALL token positions
with the donor's state. From L14 onward, the model is computing "The capital of Germany
is" — it has discarded the Beethoven prompt entirely. The recipient's token sequence is
irrelevant; only the injected state matters.

### Core Failure Mode

Last-position injection cannot chain hops because it cannot override entity information
at the OTHER token positions that Head 1 attends to. The architecture requires the
correct entity to be present at positions 1–N of the context, not just at the last
position. CoT achieves this by generating the intermediate result as a token, making
it available at all positions in the next forward pass.

---

## Experiment 4 — The Intermediate Residual

### The Three-Stage Race in "Beethoven was born in"

| Layer | Germany | Vienna | Bonn |
|-------|---------|--------|------|
| L20 | 1.6% (rank 7) | 0.2% | <0.01% |
| L22 | 4.0% (rank 2) | 2.3% (rank 7) | <0.01% |
| L24 | **14.1% (rank 1)** | 11.7% (rank 2) | 0.08% |
| L26 | **51.9% (top-1)** | 31.4% (rank 1) | 1.1% |
| L27–L29 | declining | declining | **47–48% (top-1)** |
| L30 | **62.9% (top-1)** | 0.2% | 33.6% (rank 1) |
| L33 | 0.05% | <0.01% | **98.4% (top-1)** |

The model resolves Beethoven → Germany at L26 (first hop completing), then Germany →
Bonn competes from L27–L29, Germany resurges at L30, and Bonn wins decisively at L33.
The intermediate hop result (Germany) exists transiently as a logit-lens signal
between L24 and L30.

### Geometric Reality of the Intermediate Result

At L26, "Beethoven was born in" is **equidistant** from "capital of Germany" and
"capital of Austria":

| Pair | Cosine similarity |
|------|------------------|
| Beethoven vs Germany-capital | 0.9442 |
| Beethoven vs Austria-capital | 0.9458 |
| Germany-capital vs Austria-capital | 0.9865 |

Despite Germany being top-1 in the logit lens at L26 (51.9%), the residual geometry
shows Beethoven is geometrically CLOSER to Austria-capital than Germany-capital.

On the Germany-Austria discriminant direction:
- Germany: cos = −0.152 (98.75°)
- **Beethoven: cos = −0.214 (102.37°)** — between them, 0.062 from Germany vs 0.082 from Austria
- Austria: cos = −0.296 (107.23°)

The signal exists but is tiny: a 0.062 difference in cosine similarity over a 0.144
total Germany–Austria gap. Biographical query context (Beethoven/born) dominates 99%+
of the residual geometry, masking the country-identity subspace.

### Why Injection Fails

The "Germany" intermediate result in "Beethoven was born in" at L26 is **not
interchangeable** with the "Germany" encoded in "The capital of Germany is" at L26.
Their cosine similarity is only 0.9442 — much lower than the 0.978–0.989 between
different capital queries. The Beethoven residual holistically encodes biographical
context that overwhelms the Germany signal.

To chain hops, you would need to extract just the "Germany" entity component and
re-embed it in a capital-query context. The residual provides no mechanism for this
decomposition. It is a holistic encoding, not a factored one.

---

## Experiment 5 — Attention Steering (Type B Failures)

### "The country north of France has its capital in"

Head 1 attention weights at L24 (last token, "in"):

| Position | Token | Weight |
|----------|-------|--------|
| 5 | **France** | 37.3% |
| 6 | **has** | 37.3% |
| 3 | **north** | 8.3% |
| 9 | in (self) | 5.4% |
| 4 | of | 4.2% |
| 0 | BOS | 3.7% |

Head 1 attends to both **France** (the entity) and **north** (the relational token).
This is not misdirection — it is reading the correct relational structure.

### Head 1 Attribution

| Target | Head 1 logit contribution | Layer total |
|--------|--------------------------|-------------|
| Brussels | +3.22 (top token: France) | +2.67 |
| Paris | +3.63 (top token: France) | +3.51 |

Head 1 contributes slightly **more** to Paris (+3.63) than Brussels (+3.22) in raw
DLA. But the probability effects tell a different story.

### Intervention Results

| Intervention | Brussels | Paris |
|-------------|----------|-------|
| Baseline | 44.5% | 44.5% (tied) |
| Zero Head 1 | **6.0%** ↓ | 26.8% ↓ |
| Zero all L24 attention | 8.2% ↓ | 32.6% ↓ |
| Zero L24 FFN | 35.5% ↓ | **58.6%** ↑ |
| Scale Head 1 × 2 | **64.5%** ↑ | 30.5% ↓ |
| Scale Head 1 × 3 | **73.0%** ↑ | 23.7% ↓ |

Zeroing Head 1 hurts Brussels (−38.5 pp) far more than Paris (−17.7 pp). Head 1 is
the **primary Brussels-supporting component**, despite its top attended token being
France. Scaling Head 1 × 2 flips the model to correct output (Brussels 64.5%).

The L24 FFN is also pro-Brussels — zeroing it makes Paris win (58.6%). Both Head 1
and the FFN are performing the correct relational computation (France + north →
Belgium → Brussels), but the combined signal is insufficient to commit.

### Revised Failure Classification

This is **not** a Type B (misdirection) failure. It is an **amplitude failure**:
the circuit is computing correctly but producing insufficient confidence. Head 1 reads
both the entity (France) and the relation (north) and outputs a Brussels-favoring
signal. The model ties at 44.5%/44.5% and greedy decoding selects Paris due to
(slightly) higher raw logit.

**Fix:** Amplify Head 1 (scale × 2) rather than redirect it. The attention weights
are already correct.

---

## Experiment 6 — Geometric CoT Pipeline

### Protocol

For each failing 2-hop problem:
1. Identify the hop structure (manual/oracle planner)
2. Identify the correct final-hop donor sub-query
3. Run donor through L0–L14 (14 layers)
4. Full-sequence inject at L14 into the compositional recipient
5. Run L15–L33 (19 layers)
6. Record output, KL, compute

### Results

| Problem | Direct Output | Geo CoT Output | KL | Layers |
|---------|--------------|----------------|-----|--------|
| Beethoven → country → capital | Vienna ✗ | **Berlin 88.7%** | 0.0 | 33 |
| Hamlet → author → birthplace | evasive ✗ | **Stratford 99.3%** | 0.0 | 33 |
| Shakespeare → country → language | format error ✗ | **English 83.8%** | 0.0 | 33 |
| Einstein → country → capital | format error ✗ | **Berlin 88.7%** | 0.0 | 33 |
| Belgium language → N-France 3-hop | format error ✗ | **Dutch/French** | 0.0 | 33 |

All five succeed. KL = 0.0 in every case — perfect transplant of the donor's state.
The residual angle between donor and recipient at L14 is consistently small (2.6°–3.9°)
regardless of recipient prompt length (5–18 tokens). At L14, all prompts share a
near-identical "query intent" structure.

### Compute Accounting

| Method | Layer count | Accuracy |
|--------|-------------|----------|
| Direct inference | 34 | Wrong on 4/5 problems |
| Geometric CoT (with cache) | 14 (sub-query) + 19 (run) = **33** | Correct on 5/5 |
| Geometric CoT (no cache) | 14 + 14 + 19 = **47** | Correct on 5/5 |
| Full CoT (N reasoning tokens × 34) | ~100–3400 | Correct |

With a precomputed cache of sub-query states at L14, Geometric CoT uses 33 layers —
**1 layer fewer than direct inference** — while being correct. Without cache, it costs
47 layers (38% more than direct).

### The Circular Dependency Problem

To run Geometric CoT, you need to:
1. Parse the problem into hops (requires understanding the question)
2. Identify the intermediate result (requires solving hop 1)
3. Look up or compute the donor state for the final hop
4. Inject and run

Step 2 requires running the hop-1 sub-query. If hop-1 takes 14+ layers to identify the
intermediate result (Germany, Shakespeare, etc.), then:

**Total compute = layers_for_hop1 + 19 ≥ 14 + 19 = 33 layers minimum**

This is never worse than ~33 layers but never dramatically better than direct inference
(34 layers). The "savings" are marginal: 1 layer with cache, −13 layers without.

The mechanism works — but the architecture doesn't create significant savings because
intermediate results emerge late (L26 for Beethoven → Germany). The model needs to
process most of its depth before the first hop completes.

---

## Experiment 7 — Hop Capacity

### Direct Inference on 3+ Hop Problems

| Problem | Hops | Direct Output | Correct? |
|---------|------|--------------|----------|
| Language of capital of Beethoven's country | 3 | **German** (66.5%) | ✓ (entity shortcut) |
| Language of capital of country north of France | 3 | format error | ✗ |
| Language of capital of Hamlet-author's country | 4 | format error + "Danish" | ✗ |
| Currency of Beethoven's country | 2 | **Euro** | ✓ (entity shortcut) |
| Continent containing Shakespeare's country | 2 | **Europe** | ✓ |
| Birthplace of the spouse of the author of Hamlet | 3 | "county of Norfolk" | ✗ (wrong) |

The successful 3-hop cases do not reflect true compositional chaining. They rely on
**entity association shortcuts**: the model associates Beethoven directly with Germany →
German, bypassing Berlin entirely. The model is not doing Beethoven → Germany → Berlin →
German. It is doing Beethoven → German (one shortcut step).

Evidence: "The capital of Beethoven's country" → Vienna (wrong), while "The language
of the capital of Beethoven's country" → German (correct). If the model were truly
chaining, it would get the capital right first and the language second. Instead it
shortcuts to the language and bypasses the capital.

### Geometric CoT on 3+ Hop Problems

| Problem | Donor Used | Geo CoT Output | Correct? |
|---------|-----------|----------------|----------|
| Belgium-language → N-France 3-hop | Belgium-language @L14 | Dutch/French | ✓ (ambiguous) |
| Capital-of-Germany → Beethoven-language 3-hop | Germany-capital @L14 | **Berlin** | ✗ (wrong donor) |
| Shakespeare → Hamlet-author-country-language 4-hop | Shakespeare@L14 | **Stratford** | ✗ (wrong donor) |

**Key finding:** Geometric CoT always answers the donor's question. If the donor encodes
"capital of Germany," the output is Berlin — regardless of whether the recipient asks
for the language. The injection at L14 completely overwrites the recipient's question
structure.

To answer a 3-hop question correctly via injection, you must provide the FINAL hop's
donor (the sub-query that directly produces the target answer), not an intermediate
donor. Providing "The language spoken in Belgium is" as donor → 3-hop N-France problem
correctly produces Dutch/French. Providing an intermediate state (Germany's capital)
into a language query produces the wrong answer for the wrong question.

**Error accumulation:** Not observed when the correct donor is used. KL remains 0.0
regardless of chain depth. The failure mode is donor mismatch, not compound error.

**Conclusion on hop capacity:** Geometric CoT has unlimited depth in principle — any
number of hops can be solved by injecting the correct final donor. But each hop requires
knowing the intermediate results, which requires running the previous hops. The mechanism
provides no computational shortcut over running sequential sub-queries.

---

## Experiment 8 — The Planner

### Probe Results: Single-Hop vs Multi-Hop Classification

| Layer | Train accuracy | Val accuracy |
|-------|---------------|-------------|
| 0 | 100% | **90%** |
| 4 | 100% | 90% |
| 8 | 100% | 85% |
| 12 | 100% | 80% |
| 16 | 100% | 80% |

The probe achieves its **peak accuracy at layer 0** — the token embedding layer, before
any transformer computation. Accuracy decreases monotonically with depth.

### Interpretation

The probe is detecting **surface syntactic structure**, not any computational property
internal to the model. Multi-hop prompts are structurally longer and contain nested
clauses ("the country where X was born has its capital in") while single-hop prompts
are short and flat ("The capital of X is"). This distinction is fully present in token
embeddings.

The model does NOT develop a richer internal representation of "compositional
difficulty" as it processes the prompt. Layer 16 is actually less reliable at detecting
multi-hop structure than layer 0, suggesting the model begins to process and blur the
structural distinction as it computes.

### Confounds

- Multi-hop prompts trigger a different output format. The model emits ": \n\n **" for
  complex queries, treating them as open-ended questions rather than completions. This
  is detectable from token embeddings alone and is a surface artifact.
- A truly useful planner probe would need to detect compositional complexity
  **independent of prompt length and format**, which would require controlled stimuli
  with matched length. In the current dataset, length and hop count are confounded.

### Conclusion

**The planner hypothesis is falsified.** No layer between 0 and 16 provides a
non-trivial improvement over surface features for detecting multi-hop complexity. The
model does not internally represent "this requires compositional reasoning" as a
detectable signal that could trigger an alternative processing strategy. The planner
must be external — a rule-based parser or an LLM pre-pass — not derived from the
model's internal representations.

---

## Synthesis

### What Geometric CoT Actually Is

Full-sequence injection at L14 is **equivalent to running the donor sub-query from L15
onward**. The recipient's token sequence is completely discarded. L14 of "The capital
of Germany is" and L14 of "The capital of Belgium is" have nearly identical last-position
residuals (angle ~3.5°) because the template structure is common. The donor state fully
specifies what the model will compute in subsequent layers.

This is why it works perfectly (KL=0.0) and why it can't chain hops: it doesn't
incorporate any information from the recipient's tokens after injection. It is a query
substitution, not a reasoning step.

### Why CoT Provides Something Geometry Cannot

Chain-of-thought generates intermediate results as tokens ("Germany"). In the next
forward pass, "Germany" exists as an actual token at a specific position. Head 1 can
attend directly to the Germany token (at ~12% concentration) along with all other
positions. The intermediate result is:

1. **Position-indexed** — it has a definite location in the context that attention can
   target
2. **Context-agnostic** — "Germany" as a token carries Germany's identity independent
   of how it was derived (from a Beethoven question or a geographic query)
3. **Available to all heads** — every head at every subsequent layer can attend to it

Residual injection cannot achieve this because:

1. The residual encodes the **full query context**, not just the entity identity. The
   "Germany" signal in "Beethoven was born in" at L26 is geometrically equidistant
   from both Germany-capital and Austria-capital residuals (cos 0.9442 vs 0.9458).
   The biographical context overwhelms the entity signal.
2. Injection replaces ALL information at the injected position. You cannot selectively
   inject "just the Germany component" without the surrounding context.
3. Head 1 reads entity identity from OTHER token positions (the context tokens), not
   from the last-position residual. Replacing only the last position leaves the wrong
   entity tokens in place at positions that Head 1 actually attends to.

### The Compute Picture

The only meaningful compute advantage of Geometric CoT over direct inference is:

- **2-hop with pre-cached donors:** 33 vs 34 layers (1 layer saved)
- **vs. Full CoT:** potentially 100–3400 vs 33 layers — a large saving

But: the cache must be populated. Populating the cache for all possible intermediate
results requires running those sub-queries anyway. For novel facts (rare entities,
one-off factual chains), the cache is empty and no savings are possible.

The real comparison is:
- **Geometric CoT** = external parser + cached sub-query states + fast final-hop computation
- **CoT** = the model generates its own parser, sub-queries, and final computation

CoT is fully self-contained. Geometric CoT requires external infrastructure that CoT
does not.

---

## Theorems Established

**T1 (No hop vector).** There is no reusable "capital retrieval" direction in the
residual stream. Each hop is entity-specific and encodes the full query context. The
diff-means direction at L26 (capital vs non-capital prompts) achieves 100% separation
but captures template type, not the hop itself. The separation score is higher at L23
(5.02) than L26 (4.49), confirming that template type is a pre-hop property.

**T2 (L14 is the Markov threshold for full-sequence injection).** Full-sequence
injection at L14 achieves KL=0.0 for cross-type same-question pairs (capital of Germany
→ Beethoven's country's capital, etc.) and across prompt lengths from 5 to 18 tokens.
The residual angle between donor and recipient at L14 is 2.6°–3.9° regardless of
recipient token count. The query structure is encoded holistically before L14.

**T3 (Intermediate results are geometrically entangled).** The intermediate result
"Germany" in "Beethoven was born in" at L26 cannot be cleanly extracted for use in a
second hop. At L26, the Beethoven residual is equidistant from Germany-capital and
Austria-capital (0.9442 vs 0.9458 cosine similarity). The country-identity signal is
real but occupies a tiny subspace (<0.6% of the residual), drowned in biographical
query context.

**T4 (Head 1 at L24 partially performs relational hops).** For "The country north of
France has its capital in," Head 1 attends to both the entity (France, 37.3%) and the
relational token (north, 8.3%) and produces a Brussels-favoring output. Amplifying Head
1 × 2 flips the output from Paris (wrong) to Brussels (correct, 64.5%). The circuit is
correct but under-confident. This is an amplitude failure, not a misdirection failure.

**T5 (Planner cannot be built from internal representations).** A linear probe trained
on last-position residuals achieves 90% accuracy at layer 0 (token embeddings), with
no improvement at deeper layers (decreasing to 80% at L16). The model does not develop
an internal "compositional complexity" signal beyond surface syntactic structure.

---

## Summary Table

| Exp | Hypothesis | Result |
|-----|-----------|--------|
| 1 | Single hop catalogue | Hop emerges L24 for capitals, L28–33 for biographical. FFN dominant. Answer = <1% of residual. |
| 2 | Shared hop vector exists | **Falsified.** Direction captures template type, not entity-specific hop. |
| 3 | Last-position injection enables 2-hop | **Falsified.** Head 1 reads entity from other positions. Full-sequence injection works but is query substitution. |
| 4 | Intermediate result is injectable | **Falsified.** Germany in Beethoven@L26 is geometrically equidistant from Germany-capital and Austria-capital. |
| 5 | Attention steering fixes Type B failures | **Revised.** Type B failure is amplitude, not direction. Scaling Head 1 × 2 fixes it. |
| 6 | Geometric CoT saves compute vs CoT | **Partially confirmed.** Correct answers with ~33 layers. 1 layer cheaper than wrong direct inference; much cheaper than CoT. Requires oracle planner. |
| 7 | Hop capacity degrades with depth | **Nuanced.** No error accumulation when correct donor is used. Failure is donor mismatch. Direct 3-hop shortcuts via entity association, not true chaining. |
| 8 | Model internally plans hop count | **Falsified.** Probe peaks at L0. Model does not represent compositional difficulty internally. |
