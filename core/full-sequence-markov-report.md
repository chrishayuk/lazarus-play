# Full-Sequence Markov Property — Experimental Report

**Model:** google/gemma-3-4b-it (34 layers, 2560 hidden dim)
**Experiment ID:** 929cdef0-26e9-494f-bdc1-9e772f0519d9
**Date:** 2026-03-06

---

## Theory

The refined Markov hypothesis: the complete computation state at layer L is the residual
vector at **every** token position. If you transplant all of them from prompt A into prompt B
at layer L, downstream layers (L+1 through 33) must produce prompt A's output, because:

1. Each layer recomputes its K and V matrices fresh from the current residuals at all positions
2. There is no persistent state that carries history beyond those residuals
3. The output head reads only the final-layer residual at the last position

The prior experiment established that patching the **last position only** works for same-task
prompts (threshold L14) but fails for cross-task prompts (translation vs capital-city).
The explanation: attention at every subsequent layer reads all positions, and the unpatched
positions still carry the original task context.

This experiment tests that explanation directly.

---

## Diagnostic: What does `patch_activations` actually patch?

The tool description ("replace the hidden state at a layer") doesn't specify position.
Before designing experiments, this needed to be determined empirically.

**Test design:** France → Australia same-structure patch

Prompts (both 7 tokens, identical tokens at positions 0–4 and 6):
- Source: `The capital city of France is` → Paris
- Target: `The capital city of Australia is` → Canberra

The final token (" is") is **identical** in both prompts. The only difference is position 5:
" France" vs " Australia". This is a clean discriminator:

- If `patch_activations` patches **all positions**: France context overrides Australia → all
  generated tokens should be Paris-related
- If it patches **only the last position**: the last " is" carries France's accumulated context
  (which attended to " France" during layers 0–L), but positions 0–5 still have Australia's
  residuals. Only the first generated token is affected; subsequent tokens revert.

**Result at Layer 26:**
```
Patched output: " Paris.\n\nThe capital city of Australia is Canberra.
                 \n\nThe capital city of Australia is Canberra...."
```

**Confirmed: `patch_activations` patches last position only.**

- Token 1: "Paris" — driven by the France " is" residual at the last position
- Token 2+: Immediately reverts to Australia — the unpatched positions 0–5 carry Australia
  context which reasserts via attention in all subsequent autoregressive steps

Cross-task confirmation (same length, completely different task):
```
Source: "Translate to French the word blue"  → "."
Target: "The capital city of Australia is"   → "Canberra"

Patched output: ".\n\nWhat is the largest state in Australia by area?..."
```
- Token 1: "." (source's first token)
- Token 2+: Australian quiz content — target positions reassert

**Single-position patching controls exactly one generated token.**

---

## Experiment 1: Country Transfer Threshold

With the tool confirmed as last-position-only, the question becomes: at what layer does
the " is" residual carry sufficient country-specific information to flip the first predicted
token from Canberra to Paris?

| Layer | Output (first token) | Result |
|-------|---------------------|--------|
| 14    | Canberra            | FAIL   |
| 20    | Canberra            | FAIL   |
| 22    | Canberra            | FAIL   |
| 23    | Canberra            | FAIL   |
| **24** | **Paris**          | **SUCCESS** |
| 26    | Paris               | SUCCESS |

**Exact threshold: Layer 24.**

This aligns perfectly with the previously established **Head 1 contextual attribute bridge**
(documented in prior experiments). Head 1 at layer 24 is the component that reads the country
token and injects its cultural/contextual attribute into the computation. The attention pattern
data (below) shows why this is the threshold.

**Comparison with prior thresholds:**
- L14: sufficient to transfer "capital vs largest" distinction (query *type*)
- L24: sufficient to transfer "France vs Australia" distinction (specific *country*)

Different types of information enter the last-position residual at different layers, reflecting
a staged processing hierarchy: syntactic structure → query type → semantic entity → specific fact.

---

## Experiment 2: Attention Pattern — Which Positions Does the Last Token Read?

Running `attention_pattern` on "The capital city of Australia is" at the last position (" is")
across layers 14, 20, 24, 26, 33 gives the complete picture of how the last-position residual
is assembled.

### Average attention weight across all 8 heads

```
Tokens: [<bos>, The, ' capital', ' city', ' of', ' Australia', ' is']
         pos 0   1      2          3        4        5             6

LAYER 14 — Diffuse, broad reading:
  pos0 <bos>       : 0.323  ###################
  pos2  capital    : 0.121  #######
  pos5  Australia  : 0.204  ############
  pos6  self (is)  : 0.186  ###########
  [others ~6% each]

LAYER 20 — BOS beginning to dominate:
  pos0 <bos>       : 0.500  #############################
  pos5  Australia  : 0.151  #########
  pos6  self (is)  : 0.168  ##########

LAYER 24 — Head 1 fires on Australia:
  pos0 <bos>       : 0.742  ############################################
  pos5  Australia  : 0.116  ######
  pos6  self (is)  : 0.077  ####
  [others <3% each]
  → Head 1 specifically: 59.8% on ' Australia'

LAYER 26 — Country info already embedded, BOS near-total:
  pos0 <bos>       : 0.875  ####################################################
  pos5  Australia  : 0.026  #
  pos6  self (is)  : 0.070  ####
  [others <2% each]

LAYER 33 — Final assembly, Australia re-read:
  pos0 <bos>       : 0.380  ######################
  pos5  Australia  : 0.114  ######
  pos6  self (is)  : 0.335  ####################
  → Head 6 specifically: 33.6% on ' Australia'
  → Head 2 specifically: 16.6% on ' Australia'
```

### What this reveals

**The BOS token is the dominant attention sink mid-computation (L24–26 at 74–87%).** BOS
carries minimal semantic content — it's the "this is a sequence start" signal. The model
channels most of its attention there as a way of attending to nothing in particular while
still being numerically well-behaved.

**Australia's attention weight peaks at L14 (20%), then falls as information is absorbed.**
The trajectory:
- L14: 20.4% — being read broadly
- L20: 15.1% — still being read
- L24: **11.6%** average, but **Head 1 fires at 59.8%** — the contextual bridge fires
- L26: **2.6%** — already embedded in residual; further attention not needed
- L33: **11.4%** — re-read for confidence restoration by Head 6

This is the mechanism of the L24 threshold: Head 1 fires its concentrated Australia-attention
at L24, writing country identity into the last-position residual. After L24, fresh attention
to " Australia" drops dramatically because the information is already embedded.

**At L33, Head 6 fires on Australia (33.6%)** — a different head from Head 1. This is
the confidence restoration mechanism that was already established in prior work.

### The effective Markov state

The attention distribution answers the key question for Experiment 4 (progressive position
patching, which cannot be run directly with current tools):

To faithfully transfer a prediction via patching, you need to patch positions that account
for most of the attention weight. At L24:

| Position | Avg attention | Semantic content |
|----------|--------------|-----------------|
| BOS (0)  | 74.2%        | Low (sequence start marker) |
| Australia (5) | 11.6%   | HIGH (country identity) |
| self/is (6) | 7.7%      | Medium (last-position context) |
| Others   | 6.5% total   | Low |

**Patching only position 6 (last):** Works for 1 token because the " is" residual has
absorbed Australia/France identity through Head 1's L24 attention. Fails for token 2+
because position 5 still says " Australia".

**Patching positions 5+6:** Would patch the primary semantic content (Australia) plus
the last-position residual. Should work for 2+ tokens since position 5 would now say
" France". Position 0 (BOS, 74% attention) would still anchor Australia context — might
still revert.

**Patching positions 0+5+6:** Would cover ~93% of attention weight at L24. Almost
certainly sufficient for complete transfer.

**Patching all 7 positions:** Theoretically guarantees complete transfer. By construction,
each subsequent layer recomputes KV fresh from the transplanted residuals.

---

## Experiment 3: KV Cache — Does Persistent State Exist Beyond Residuals?

This experiment addresses whether anything *other* than the residual vectors persists
across layers.

**Theoretical answer:** No. In a standard transformer:
- Each attention layer computes Q, K, V from the current residual at all positions
- Uses them within that layer's attention operation
- The resulting output is added to the residual stream
- The *next* layer computes its own fresh Q, K, V from the updated residuals
- There is no persistent KV cache that carries information from layer L to layer L+2
  (KV caching is an inference optimization that caches across *generation steps*, not
  across *transformer layers*)

**Empirical verification approach:** Compare logit distributions between:
1. Running source prompt normally
2. Transplanting all-position residuals from source into target at layer L, then running
   layers L+1–33

If the logit distributions match to floating-point precision, the residual IS the complete
state. Any mismatch would indicate additional state. However, this requires a tool that
can transplant all-position residuals simultaneously — which `patch_activations` cannot do.

**Conclusion:** KV cache across layers does not exist by architecture. The residual vectors
at all positions are the complete state at any layer. **The full-sequence Markov property
holds by construction**, not as an empirical question.

---

## Experiment 4: Progressive Position Patching (Inferred)

Cannot be run with current tools (`patch_activations` patches last position only, and
looping multiple calls would each be separate forward passes — cannot patch multiple
positions in a single forward pass). The following is inferred from the attention pattern data.

**Prediction (based on L24 attention weights):**

| Positions patched | Expected result | Reasoning |
|------------------|----------------|-----------|
| pos6 only (current capability) | Token 1 = Paris, token 2+ = Australia | Only last-position residual affected |
| pos5 + pos6 | First 2–3 tokens = Paris then reverts | Australia token replaced, but BOS (74%) still anchors Australia semantics |
| pos0 + pos5 + pos6 | Probably full transfer | ~93% of attention weight covered |
| All 7 positions | Complete transfer guaranteed | Full Markov state replaced |

The model would need approximately **3 positions patched** (BOS + country token + last) to
achieve reliable cross-task transfer from L24 onward.

---

## Experiment 5: Attention Weight Summary

**How concentrated is last-position attention?**

At the critical computation layers (L24–26), attention from the last position is:
- L24: 74% BOS, 12% Australia, 8% self, 6% spread across 4 other positions
- L26: 87% BOS, 7% self, 3% Australia, 3% others

**It is highly concentrated but NOT semantically concentrated.** Most attention weight (74–87%)
goes to BOS which carries minimal semantic content. The semantic action is in the 12–20%
that goes to the country/topic token.

This explains why single-position patching transfers 1 token but not more: the last-position
residual is semantically determined by the 12% Australia attention (plus its accumulated
history from earlier layers), but BOS (74%) anchors the sequence-level frame that subsequent
generation inherits.

---

## Summary: Full-Sequence Markov Theory

### What was confirmed

**1. `patch_activations` patches last position only.**
The tool's effect propagates for exactly one generated token. Subsequent tokens are generated
autoregressively reading all unpatched earlier positions.

**2. Different information types enter the last-position residual at different layers:**
- L14: Query type (capital vs largest, differing at positions 2–3)
- L24: Country identity (France vs Australia, differing at position 5) — via Head 1 firing 60% attention

**3. Attention is highly non-local for the first 20% of semantic weight:**
The country token at position 5 gets 12–20% of last-position attention. Patching only the
last position transfers country identity for 1 token because the last position has absorbed
it. Patching also the country position would extend transfer to multiple tokens.

**4. Full-sequence Markov property holds by construction, not as an empirical claim.**
Each layer recomputes KV from current residuals. There is no persistent cross-layer state
beyond the residuals. Full-position transplant must produce the source output.

### What cannot be tested with current tools

Full-sequence patching requires injecting a [seq_len × hidden_dim] matrix into a forward
pass at a specific layer. The current `patch_activations` tool only replaces a [1 × hidden_dim]
vector at the last position.

To test full-sequence Markov empirically, the tool would need a `patch_all_positions`
parameter that replaces the full residual matrix, not just the last row.

### The Markov state structure

At layer L, the complete Markov state is:

```
state[L] = {residual[L, pos] for pos in 0..seq_len-1}
```

For next-token prediction, the relevant components weighted by approximate importance:
- BOS residual (pos 0): 74–87% attention weight, sequence-frame anchor
- Topic/entity token (position varies): 12–20%, semantic content carrier
- Last position (pos -1): 8–34%, accumulated context from all prior layers
- Other positions: collectively <10% at L24–26

The "Markov bandwidth" for next-token prediction is effectively the **BOS + topic_token +
last_position** triple, not the full sequence — but altering any other position that feeds
into those three would propagate upstream.

---

## Layered Information Encoding Timeline

The staged processing hierarchy revealed across all experiments:

| Layer | What enters the last-position residual |
|-------|---------------------------------------|
| 0–13  | Token embeddings and positional encodings. Syntactic structure. |
| 14    | Query type (capital vs largest vs author of). Structural meaning. |
| 15–23 | Semantic context enrichment. Country/topic token information building. |
| **24** | **Country/entity identity via Head 1 (60% attention spike).** Critical layer. |
| 25–26 | Fact retrieval (L26 FFN = capital fact store). Prediction crystallizes. |
| 27–31 | Consolidation. |
| 32    | Possible active disruption (unknown mechanism). |
| 33    | Final assembly. Head 6 re-reads country token (34%). Confidence restored. |

The residual stream accumulates a progressively richer representation as computation proceeds.
The "Markov state" at each layer is complete for that layer's computation — but the information
it encodes has different granularity at each depth.

---

*Experiment saved to Lazarus experiment store.*
*Related: markov-bandwidth-report.md (prior experiment)*
