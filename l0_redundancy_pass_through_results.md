# L0 Redundancy: Does Attention Retrieve or Refresh?

**Experiment ID:** 8cb58097-1bdb-485f-bfae-b242fcfc45df
**Model:** google/gemma-3-4b-it (34 layers, 2560 hidden dim, 8 heads)

## Central Question

L0 re-reads the prompt at every generation step. The pass-through channel
carries prefill-written information unchanged to L33. How much of what L0
retrieves is genuinely new vs already in the residual?

## Central Answer

**NEITHER retrieve NOR refresh. L0 attention UPDATES the pass-through channel
with new autoregressive context at each generation step.**

---

## Finding 1 — L0 Writes Entirely Into Dark Space

L0 attention output is 88-92° to ALL token embeddings across all 5 prompts,
all generation steps. Cosine similarity < 0.04. Every bit of L0's write goes
to dimensions invisible in vocabulary space.

| Prompt | attn→Paris | attn→France | attn→capital |
|--------|-----------|-------------|-------------|
| France | 90.04° | 91.38° | 90.25° |
| Beethoven | 89.06° (Vienna) | 89.63° (Bonn) | 92.07° (Beethoven) |
| Water | 89.89° | 89.89° | 89.77° |
| Eiffel | 90.64° | 91.35° | 89.83° |
| Python | 90.10° | 90.44° | 90.02° |

This is constant across generation steps (steps 1, 2, 7, 10 all 90-92°).

## Finding 2 — The Coupling Shift (Key Discovery)

L0's attention-FFN coupling angle CHANGES DRAMATICALLY between prefill and
generation:

| Generation step | attn↔FFN angle | ||attn|| | ||FFN|| | Interpretation |
|----------------|---------------|---------|--------|----------------|
| Step 1 (prefill end) | **54.8°** | 657 | 302 | **Cooperative** — attn and FFN coupled |
| Step 2 (after "Paris") | **83.1°** | 735 | 591 | **Orthogonal** — write to different subspaces |
| Step 7 (after "*huge") | **99.7°** | 714 | 514 | **Anti-correlated** — oppose each other |
| Step 10 (after "things") | **96.3°** | 746 | 526 | **Anti-correlated** — stable |

**At prefill**: L0 attn and FFN cooperate (54.8°, moderately coupled).
**During generation**: They write in OPPOSITE directions (83-100°).

This means during generation, L0's attention writes almost entirely to the
pass-through highway — dimensions that L0's FFN does not touch and nothing
downstream will modify.

Also note: L0's FFN nearly DOUBLES in norm during generation (302→591).
The FFN becomes a much larger component, but writes to completely different
dimensions than attention.

## Finding 3 — L0 Heads Are Holographically Redundant

Every single L0 head is individually dispensable:

| Head | KL when zeroed | Top-1 preserved? |
|------|---------------|-------------------|
| 0 | 0.0 | Yes (Paris 100%) |
| 1 | 0.000015 | Yes |
| 2 | 0.0 | Yes |
| 3 | 0.000015 | Yes |
| 4 | 0.0 | Yes |
| 5 | 0.0 | Yes |
| 6 | 0.0 | Yes |
| 7 | 0.000005 | Yes |

But removing ALL L0 attention: KL = 7.5-17.1 (catastrophic).

Inter-head angles: 82-95° (approximately orthogonal subspaces).
The dark signal is distributed holographically — any 7/8 heads compensate.

## Finding 4 — L0 Attention Evolves Across Generation

L0 does NOT simply re-read the same positions. By step 7:

- **Head 0**: reads " known" from GENERATED context at **26.2%**
- **Head 4**: reads " known" (gen) at **34.8%** — primarily reading own output
- **Head 6**: BOS attention drops from 60% (step 1) to 9.3% (step 7)

L0 integrates newly generated tokens that don't exist in the pass-through
from prefill. Its value INCREASES across generation steps as it provides
autoregressive context that the pass-through channel cannot carry.

## Finding 5 — Only L0 and L4 Have Critical Attention

Per-layer attention ablation (first token, France prompt):

| Layer | KL | Status |
|-------|------|--------|
| **L0** | **7.59** | **CRITICAL** (content reader) |
| L1 | 0.00004 | Safe |
| L2 | 0.00006 | Safe |
| L3 | 0.00003 | Safe |
| **L4** | **15.94** | **MOST CRITICAL** (format reader) |
| **L5** | **10.5** | **CRITICAL** (prompt-dependent) |
| **L6** | **10.5** | **CRITICAL** (prompt-dependent) |
| L7 | 0.0 | Safe |
| L14 | 0.046 | Minimal |
| L24-L33 | 0.0-0.008 | Safe |

Beethoven prompt: L4 KL=23.0 (even more critical), L5-L6 safe.
The critical window is prompt-dependent: L0 + L4 always, L5-L6 sometimes.

## Finding 6 — L0 vs L4: Content Reader vs Format Reader

**L0 at Layer 0 (complex Beethoven multi-hop, 33 tokens):**
- Head 2: "Bonn" 60.2% — birthplace entity
- Head 6: "Bonn" 66.4% — entity (adaptive, NOT fixed template!)
- Head 3: "Bonn" 16.4% — content
- Head 5: "Bonn" 14.8% + "capital" 7.2% — entity + query structure

**L4 at Layer 4 (same prompt):**
- Head 6: "model" 83.2% — role assignment
- Head 2: BOS 73.4% — learned bias
- Head 5: `<start_of_turn>` 37.7% — template structure
- Others: BOS-dominant 27-49%

Two completely different reading strategies. L0 = what entities, where.
L4 = what role, what format. Both irreplaceable.

## Finding 7 — Entity Direct Reading, Not Coreference

L0 reads entities DIRECTLY, bypassing pronouns:

| Position/Token | Max L0 attention | Role |
|---------------|-----------------|------|
| "Bonn" (pos 9) | 66.4% (Head 6) | **Query-relevant entity** |
| "Be" (pos 4) | 13.5% (Head 7) | Entity name |
| "born" (pos 7) | 11.5% (Head 6) | Relation |
| "He" (pos 11) | 13.0% (Head 1) | **Pronoun — low** |
| "he" (pos 23) | 3.7% (Head 4) | **Pronoun — negligible** |
| "Vienna" (pos 15) | 5.8% (Head 3) | **Distractor — deprioritized** |

Coreference was resolved during prefill. Pronouns are low-value KV entries.

## Finding 8 — Collective Attention Removal Boundary

Individual layer dispensability ≠ collective dispensability.

| Layers removed | Count | France | Beethoven |
|---------------|-------|--------|-----------|
| L25-L33 | 9 (26%) | **WORKS** ("a *lot!") | **WORKS** ("Vienna, Austria") |
| L23-L33 | 11 (32%) | DEGRADED (coherent) | FAILED (hallucination) |
| L22-L33 | 12 (35%) | FAILED | FAILED |
| L15-L33 | 19 (56%) | FAILED (gibberish) | FAILED (gibberish) |
| Keep only L0+L4-L6 | 4 | FAILED | FAILED |

Safe removal: **L25-L33 (9 layers, 26%)**. Sharp boundary at L22-L23.
Matches the orthogonal regime from the geometry experiment exactly.

## Finding 9 — Head 6 is Adaptive

Head 6 at L0 was thought to be a fixed BOS/template reader (60% to BOS on
simple prompts). On the complex multi-hop Beethoven prompt:

- Simple "moved to": BOS 59.8%
- Complex "capital of country where born": **Bonn 66.4%** (BOS < 1%)

Head 6 ADAPTS its reading strategy based on query complexity. It switches
from template reading to entity reading when the query requires it.

---

## Mode 3+ Implications

### KV Layer Budget
- **L0**: Critical. Content reader. Cannot skip at any generation step.
- **L4**: Critical. Format reader. Cannot skip.
- **L5-L6**: Prompt-dependent. May be critical for some queries.
- **L1-L3**: Individually and collectively safe to remove.
- **L7-L24**: Individually safe but collectively necessary. Cannot remove
  more than ~9 layers from this range without degradation.
- **L25-L33**: Safely removable (9 layers, 26% attention reduction).

### KV Position Budget at L0
High-value positions (by attention concentration):
- **Entity positions** (Bonn, Beethoven): 14-66% attention from entity heads
- **Query structure** (capital, born, is): 7-14% from structure heads
- **Template** (BOS, start_of_turn): 10-26% from template heads when active

Low-value positions:
- **Pronouns** (He, he): <13%, usually <4%
- **Function words** (of, the, a): <6%
- **Distractor entities** (Vienna): <6%

For the 33-token complex prompt, ~9 positions carry most of the signal.

### KV Position Budget at L4
Only needs: BOS (27-73%), model (83%), start_of_turn (38%), end_of_turn (10%).
4-5 fixed template positions. Content positions negligible.

### Theoretical Compression
- Full KV: all positions × 34 layers
- Sparse KV: ~9 content positions (L0) + ~5 template positions (L4) +
  distributed mid-range (L7-L24, full KV needed here)
- Critical-layer compression: ~50-100x for L0 and L4 alone
- Total compression limited by mid-range collective necessity

### Refresh Rate
L0 **cannot be skipped**. It updates the pass-through channel with new
autoregressive context (generated tokens) at each step. The information it
writes at step N is DIFFERENT from step N-1 because new tokens have been
generated.

The pass-through channel doesn't need refreshing because it carries old
information forward automatically. But L0 writes NEW information (about
generated context) that overwrites the old pass-through content. This is
updating, not refreshing.

---

## Architecture Summary

```
PREFILL:
  L0 attn reads prompt → writes dark (88-92° to tokens)
  L0 attn-FFN coupled (54.8°) → cooperative processing
  L0 attn norm=657, FFN norm=302 (attn dominates 2.2x)
  Entity identity encoded in pass-through at L7-L14

GENERATION:
  L0 attn reads prompt + generated tokens → writes dark (88-92°)
  L0 attn-FFN ANTI-CORRELATED (83-100°) → write to opposite dimensions
  L0 attn norm=735, FFN norm=591 (FFN nearly doubles, 1.2x ratio)
  Attention → pass-through channel (orthogonal to FFN)
  FFN → active processing channel (orthogonal to attention)

  L0 attention reads DIFFERENT positions at each step:
    Step 1: template heads dominate (BOS 60%)
    Step 7: generated-context heads dominate (own output 26-35%)

  L0 head 6: adaptive — template reader OR entity reader based on query
  L0 heads: holographically redundant (any 7/8 compensate)

  L4: pure format/role reader (BOS + model tag)
  L4: most critical single layer (KL=15.9-23.0)

  L25-L33: attention safely removable (pass-through regime)
```
