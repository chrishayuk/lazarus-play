# L29 H4 Fact Detection — Validation Results

## Experiment: Step 1 from Engineering Spec

**Question:** Does L29 H4 attention during raw prefill (no query) flag
fact-bearing tokens in real Apollo 11 transcript windows?

**Answer: No.** L29 H4 is a retrieval head, not a salience head. It only
concentrates on fact tokens when a query drives the retrieval. During raw
prefill, no head at any layer gives a usable fact-detection signal.

## Setup

- Model: google/gemma-3-4b-it (34 layers, 8 heads)
- Windows tested: W654 (scratchy), W77 (football/Namath), W723 (Hornet/splashdown)
- Tool: `attention_pattern` at L29 (and L7/L14/L20/L25/L33 for comparison)

## Key Token Positions

| Window | Token       | Position | Content                     |
|--------|-------------|----------|-----------------------------|
| W654   | ` scratch`  | 11       | Audio quality descriptor    |
| W77    | ` Jets`     | 126      | Football team               |
| W77    | ` football` | 150      | Sport reference             |
| W723   | `splash`    | 4-5      | Splashdown event            |
| W723   | `HOR`/`NET` | 67,96... | Recovery ship (Hornet)      |
| W723   | ` Hornet`   | 179,188  | Recovery ship (full token)  |

## Result 1: L29 H4 at last position (no query) — W654

L29 H4 at position 511 (last token in window):

| Position | Token     | Weight |
|----------|-----------|--------|
| 0        | `<bos>`   | 58.6%  |
| 509      | ` ENTER`  | 9.6%   |
| 500      | ` VER`    | 3.1%   |
| 489      | ` PTC`    | 2.7%   |
| 470      | ` Houston`| 2.0%   |

**" scratch" at pos 11: 0.0% (not in top 20)**

Attention goes to BOS + local context (VERB 24, ENTER). Zero attention to
the fact token "scratchy" 500 tokens earlier.

## Result 2: L29 H4 at position 50 (no query) — W654

Reading from a position closer to "scratchy":

| Position | Token      | Weight |
|----------|------------|--------|
| 0        | `<bos>`    | 85.9%  |
| 50       | `CMP`      | 9.1%   |
| 30       | ` votes`   | 1.7%   |
| 1        | ` sound`   | 0.9%   |
| **11**   | **` scratch`** | **0.6%** |

**0.6% — well below the 2% threshold proposed in the spec.**

## Result 3: L29 H4 at last position (no query) — W723

| Position | Token     | Weight |
|----------|-----------|--------|
| 0        | `<bos>`   | 41.8%  |
| 485      | `'`       | 12.7%  |
| 127      | ` h`      | 9.3%   |

**No Hornet, splashdown, crew, or chutes in top 20.**
Only 1 head (H7) at L29 had any fact token in top 20: pos 294 "Hor" at 0.02%.

## Result 4: L29 H4 WITH query — W723

When prepending the question "what ship recovered the Apollo 11 crew?":

| Position | Token        | Weight | Note        |
|----------|--------------|--------|-------------|
| 0        | `<bos>`      | 46.7%  |             |
| 12       | ` recovered` | 4.3%   | Query term  |
| 24       | ` splash`    | 2.3%   | Context     |
| 81       | ` HOR`       | 2.2%   | **ANSWER**  |
| 11       | ` ship`      | 1.7%   | Query term  |
| 108      | ` HOR`       | 1.6%   | **ANSWER**  |
| 149      | ` SWIM`      | 1.5%   | Related     |
| 178      | ` Hornet`    | 1.3%   | **ANSWER**  |
| 187      | ` splash`    | 1.3%   | Context     |
| 110      | ` Hornet`    | 1.1%   | **ANSWER**  |

**With a query, L29 H4 attends to the answer.** Multiple Hornet positions
(2.2%, 1.6%, 1.3%, 1.1%) + splash, SWIM, crew. Total fact attention ~12%.

## Result 5: Multi-layer scan at position 100 (no query) — W723

Checked L7, L14, L20, L25, L29, L33. Best non-local fact attention per layer:

| Layer | Best Head | Non-local Fact Attn | Details                          |
|-------|-----------|---------------------|----------------------------------|
| L7    | H6        | **10.2%**           | crew 3.9%, crew 3.5%, HOR 2.9%  |
| L7    | H5        | 4.4%                | HOR 2.2%, crew 1.2%, crew 1.0%  |
| L14   | H3        | 5.1%                | HOR 5.1%                        |
| L14   | H4        | 2.3%                | HOR 2.3%                        |
| L29   | H4        | 3.9%                | splash 1.8%, crew 1.4%          |
| L20   | H1        | 2.2%                | HOR 2.2%                        |
| L33   | H0        | 1.0%                | HOR 0.3% (below threshold)      |

**L7 H6 is the strongest fact-attention head during prefill** at 10.2%,
but even this is driven by repeated entity tokens (HOR appears 5× in the
window). The signal is still too weak and too entity-repetition-dependent
to serve as a reliable fact detector.

## Why the Spec's Hypothesis Fails

**Experiment 993f5f93** measured L29 H4 during *generation* on a synthetic
prompt (" Volt" in the Voltara experiment). That's a retrieval scenario —
the model is actively looking for a fact to copy into the output.

**During processing (raw prefill),** there is no query. L29 H4 has nothing
to retrieve for, so it attends to:
1. `<bos>` (42-86%) — the sink token
2. Local context (tokens within ~20 positions of readout)
3. Formatting/structural tokens (timestamps, speaker labels)

The 45:1 signal-to-noise ratio from 993f5f93 was measured in retrieval
mode, not in indexing mode. The two are fundamentally different.

## The Core Problem

The spec assumed: "what L29 H4 attends to during prefill is what it will
need during retrieval." This is backwards. L29 H4 doesn't know what it
will need until there's a query. During prefill, it's doing standard
language modeling — predicting the next token in the transcript, which
requires attending to local context, not distant facts.

## What Might Work Instead

### Option A: Query-time attention extraction (already how it works)
During generation with KV replay, L29 H4 naturally attends to the right
windows. The sparse index just needs to get the query to the right windows.
This is the existing architecture — improve the keyword extraction, not
the detection mechanism.

### Option B: Activation norms / residual spikes during prefill
Instead of attention, look at whether fact-bearing tokens produce unusual
activation patterns (e.g., high residual norms) during prefill. This
would be position-independent and query-independent.

### Option C: Self-supervised fact marking
Run each window with a generic "summarize key facts" query appended, then
read L29 H4 attention. Expensive (requires generation per window) but
would activate the retrieval circuit.

### Option D: Entity/noun extraction (NLP-based)
Use a simple NER or POS tagger to extract entities and nouns. "Hornet",
"splashdown", "Namath", "football" are all nouns/entities that a tagger
would catch. No model introspection needed. Complements surprise heuristic.

## Conclusion

**Step 1 validation fails.** L29 H4 does not flag fact positions during
raw prefill on real Apollo 11 windows. The spec should not proceed to
Steps 2-5 as written.

The retrieval head insight from 993f5f93 is correct — L29 H4 IS the
retrieval head during generation. But retrieval and indexing are different
operations. The model retrieves facts when asked, it doesn't pre-mark
them during encoding.
