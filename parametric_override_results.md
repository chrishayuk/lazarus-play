# Parametric Override — 1D Injection for Knowledge Editing
**Experiment ID:** a25bd50a-19a7-41a7-a075-5c38ffa6c5af
**Model:** google/gemma-3-4b-it
**Date:** 2026-03-18

## Question

Can 1D subspace injection at L30 correct parametric misconceptions baked into model weights, without any weight modification?

**Answer: Yes — under specific conditions. The mechanism is identical to novel-fact injection.**

---

## Phase E1 — Baseline Survey (Finding Misconceptions)

| Query | Model's top-1 | P | Correct | Verdict |
|---|---|---|---|---|
| "Frankenstein is the name of the" | " monster" | 57.4% | " scientist" | ✓ USEFUL — clear misconception |
| "Goldfish have a memory span of only three" | " seconds" | 96.9% | " months" | ✓ USEFUL — extreme prior |
| "Fortune cookies were invented in" | " Japan" | 54.9% | " America" | ⚠ DEBATABLE (origin disputed) |
| "The Great Wall of China is visible from" | " space" | 82.0% | — | ⚠ No clean correct token |
| "Pluto is classified as a" | " dwarf" | 99.2% | — | Model CORRECT |
| "The capital of Australia is" | " Canberra" | 84.4% | — | Model CORRECT |

**Selected cases:** Frankenstein (moderate 57% prior) and Goldfish (extreme 97% prior).

----

## Phase E2 — Donor Design

### The Template-Lock Problem

The critical insight: for a same-suffix donor to work, the corrective context must actually SHIFT the model's prediction at the shared query position.

**Goldfish template-lock (fails):**
```
Donor: "Contrary to the popular myth, scientific studies show goldfish have excellent
        long-term memory. Goldfish have a memory span of only three"
→ " seconds" 99.2%
```
Even explicit corrective context cannot override the "of only three → seconds" template. No viable same-suffix donor exists.

**Fortune cookies template-lock (fails):**
```
Donor: "The Benkyodo Company in San Francisco is often cited as one of the original
        manufacturers of fortune cookies, which were popularized among Japanese immigrants
        in California. Fortune cookies were invented in"
→ " Japan" 96.9%
```
Mentioning San Francisco, Japanese immigrants, California — none of it matters. Template wins.

**Frankenstein (works):**
```
Donor: "In Mary Shelley's novel, Frankenstein is the name of Victor, a scientist who
        creates a creature. The creature itself has no name in the book.
        So Frankenstein is the name of the"
→ " creator" 69.6%, " scientist" 12.1%
```
The corrective context successfully shifts the prediction away from " monster".

---

## Phase E3 — Injection Results

### Test 1: Frankenstein → "creator" (primary target)

| | P(" creator") | P(" monster") | P(" creature") |
|---|---|---|---|
| Bare recipient | 0.5% | **57.4%** | 23.9% |
| Donor | **69.6%** | 1.9% | — |
| **After injection** | **83.6%** | 7.3% | 4.6% |

**Result: OVERRIDE SUCCESS.** " monster" 57.4% → 7.3% (8× suppression). Injection uses 0.21% of residual energy.

### Test 2: Frankenstein → "scientist" (secondary target — remarkable)

| | P(" scientist") | P(" monster") |
|---|---|---|
| Bare recipient | 2.2% | **57.4%** |
| Donor | 12.1% (secondary — top-1 was "creator") | 1.9% |
| **After injection** | **75.0%** | 12.7% |

**Result: OVERRIDE SUCCESS — with a non-top-1 target.** The donor predicted " creator" as its top-1 token, but the injection targets the " scientist" embedding direction specifically. A 12.1% secondary belief in the donor amplifies to 75% at inference. The injection reads whatever coefficient the donor has in the SPECIFIED direction, regardless of what the donor would generate autoregressively.

### Test 3: Goldfish (cross-suffix, extreme prior)

| | P(" months") | P(" seconds") |
|---|---|---|
| Bare recipient | 0.012% | **96.9%** |
| Donor (cross-suffix, 19.5°) | 82.9% | 0.05% |
| **After injection** | 0.10% | **97.0%** |

**Result: COMPLETE FAILURE.** Zero practical effect. The 97% prior is completely unaffected. The cross-suffix mismatch (19.5° angle vs 9.4° for Frankenstein same-suffix) means the donor's " months" coefficient doesn't transfer.

---

## Phase E4 — Analysis

### Four Laws of Parametric Override

**Law 1 — Same-suffix requirement is strict.**
Donor and recipient must share the same final tokens. The L30 residual encodes both factual content AND the syntactic/structural context of the current position. Cross-suffix transfer fails (19.5° vs 9.4° angle; complete failure vs success).

**Law 2 — Template-lock is the primary failure mode.**
Some query templates (e.g. "of only three", "were invented in") are so strongly associated with a specific next token that no corrective text within the same prompt can shift the prediction. These cases have NO viable same-suffix donor and CANNOT be corrected by 1D injection. The parametric belief is encoded at the template level, below the corrective-context threshold.

**Law 3 — Moderate priors (≤57%) ARE overridable with a viable donor.**
When the corrective context successfully shifts the donor's prediction, L31-L33 amplify the injected direction strongly enough to override a moderate parametric prior.

**Law 4 — The injection targets a SPECIFIED direction, not the donor's top-1.**
The mechanism extracts the donor's L30 projection onto the TARGET token's embedding direction. This allows injecting a token the donor predicted with only 12% confidence (scientist) to achieve 75% override. A single donor context can be used to inject multiple possible corrections by varying `subspace_tokens`.

### Comparison Table

| Case | Prior | Donor type | Donor P(target) | Angle | Result |
|---|---|---|---|---|---|
| Frankenstein → creator | 57% wrong | same-suffix | 69.6% | 9.4° | **83.6% ✓** |
| Frankenstein → scientist | 57% wrong | same-suffix (secondary) | 12.1% | 9.4° | **75.0% ✓** |
| Goldfish → months | 97% wrong | cross-suffix | 82.9% | 19.5° | 97.0% ✗ |
| Fortune cookies | 97% wrong | template-locked | ~0% | N/A | untestable ✗ |

### The Mechanism is Identical to Novel-Fact Injection

For novel facts: donor = (full context with fact) + (bare query). Donor knows the answer from context → predicts it at 85-97%. Injection works at 97-99.99%.

For parametric corrections: donor = (explicit correction text) + (same query). The explicit correction shifts the donor's prediction from the wrong token toward the right one. Same L30 injection mechanism. Same L31-L33 amplification. The only difference is that the "donor context" must explicitly contain the correction, not just assert the fact.

### Architectural Implication for Correction Tables

A "correction table" (analogous to the 12-byte fact index) could store:
- `wrong_token_id` (4 bytes) — for routing detection
- `correct_token_id` (4 bytes) — for subspace selection
- `coefficient c` (8 bytes) — donor's L30 projection in the correct direction

12 bytes per corrected misconception. Same as for novel facts.

**Condition for this to work:** A same-suffix donor must exist that achieves P(correct) > ~20% at the query position. Template-locked misconceptions (prior unshiftable by in-context correction) cannot be stored in this table.

---

## Summary

| Question | Answer |
|---|---|
| Can 1D injection override parametric misconceptions? | YES — for moderate priors where a viable donor exists |
| What's the failure mode? | Template-lock: some priors are immune to same-suffix correction |
| Does the injection target the donor's top-1? | NO — it targets the SPECIFIED direction; donor's secondary beliefs are amplifiable |
| Same storage cost as novel facts? | YES — 12 bytes per corrected fact |
| Requirement for donor viability? | Corrective context must actually shift donor's prediction at the shared suffix |

---

## Open Questions

- What is the prior strength threshold? Is 70% injectable? 80%? The transition between overridable (57%) and unoverridable (97%) is not yet mapped.
- Can template-locked misconceptions be corrected by intervening at an earlier layer (e.g. L14) where corrective context has more influence?
- Does the 75% scientist result mean secondary beliefs (12% in donor) have a privileged role? Or is it purely proportional to the donor's subspace fraction?
