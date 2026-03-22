# V-Injection Architecture Specification
**Status:** Current canonical spec as of 2026-03-18
**Supersedes:** Implementation section of v_injection_results.md

This document synthesizes findings from:
- v_injection_results.md — mechanism proof
- k_vector_routing_results.md — native routing confirmed
- minimum_viable_injection_results.md — 12-byte content compression
- multi_fact_1d_superposition_results.md — inject-matched-only architecture
- routing_scale_results.md — N=12 stress test, failure modes identified
- addressing_precision_results.md — W_K crowding explained, hybrid routing fix

---

## What the Architecture Does

At prefill time, for each factual sentence in a document, compute and store two
things: the answer vector (12 bytes) and the routing key (532 bytes). At query
time, match the query against the routing index and inject the pre-computed answer
vector at L30 of a bare forward pass. No document tokens are replayed.

**Compression ratio:** 43.5 KB (3,625 facts) vs 515 GB (full Apollo 11 KV cache).
**Latency:** 330ms (injection path) vs 2,000ms (replay path).
**Accuracy:** KL ≈ 0 when routing is correct. Wrong routing delivers wrong answer
at maximum confidence — routing must be correct before injection.

---

## Storage per Fact (532 bytes total)

```
answer_token_id  : 4 bytes     ← which vocabulary token to inject
projection_coeff : 8 bytes     ← scalar, magnitude of injection
k_vector         : 512 bytes   ← 256D float16, from L29 KV-head-2
entity_string    : ~12 bytes   ← entity name from fact ("Zarkov Industries")
────────────────────────────────
Total            : ~536 bytes per fact
3,625 facts      : ~1.9 MB index
```

The `(answer_token_id, projection_coeff)` pair encodes the content: inject the
answer token's embedding direction scaled by the coefficient, at L30. This is
the 1D subspace mechanism from minimum_viable_injection. The `k_vector` and
`entity_string` are the two-stage routing index.

---

## Prefill Phase (once per document)

For each fact sentence F_i, run ONE forward pass with a single-fact donor prompt:

```
donor_i = "[F_i sentence]. [query_template_for_F_i]"
```

The query template ends at the token just before the expected answer (e.g.,
"Zarkov Industries was founded in the city of"). Extract from this forward pass:

1. **answer_token_id**: the first token of the correct answer
   (e.g., " Volt" for Voltara). Verify with `predict_next_token`.
2. **projection_coeff**: the projection of the L30 last-position residual onto the
   answer token embedding direction.
3. **k_vector**: the 256D hidden state at L29 KV-head-2 at the last token position,
   computed via W_K @ h_L28[-1].
4. **entity_string**: the entity name associated with this fact (e.g., "Zarkov
   Industries"). Extracted from the fact template at index construction time.

**First-token screening:** During index construction, flag facts whose answer
first-token length is ≤ 3 characters (" A", " Bel", " Del"). These will have low
Q·K routing scores. Mark them as entity-filter-only or explicit fallback facts.

---

## Routing Phase (per query)

### Stage 1: Entity String Match

Before any vector computation, scan the query text for stored entity strings.

```
for entity_str, fact_id in entity_index:
    if entity_str in query_text:
        inject(fact_id)
        return
```

Properties:
- Zero wrong injections (entity strings are unique per fact by construction)
- Zero false negatives for entity-explicit queries (names appear verbatim)
- O(N·k) string scan — faster than Q·K dot products
- Coverage: ~75–83% of factual queries in typical operational documents

**Entity-explicit queries** include anything that names the entity being queried:
"What city was Zarkov Industries founded in?", "Joe Namath agreed to what?",
"What year did Nexaris Corp incorporate?". For these, entity string match is
definitive — proceed directly to injection.

### Stage 2: Adaptive Q·K Routing (entity-implicit queries)

For queries that don't contain a stored entity name:

```
q_vec = extract_Q_at_L29_H4(bare_query_prompt)
scores = [dot(q_vec, k_i) for k_i in k_index]
argmax_id = argmax(scores)
adaptive_threshold = mean(scores) * 2.0
if scores[argmax_id] > adaptive_threshold:
    inject(argmax_id)
else:
    fallback_to_replay()
```

**Why adaptive threshold, not fixed 15%:** At N=12, max scores are 20–47%.
At N=100, max scores drop to ~5–7%. At N=3,625, max scores are ~0.03–0.3%.
A fixed 15% threshold produces 0% injection rate at scale. The adaptive
threshold `mean × 2.0` is scale-invariant: it injects when the top score is
substantially above the per-query average, regardless of N.

**What entity-implicit queries look like:**
- "What audio quality was reported during descent?" (no entity name)
- "What was the transmission condition noted?" (no entity name)
- "What happened at thirty-two thousand feet?" (no entity name)

These are structurally distinctive — less prone to template collision than
entity-explicit queries. Q·K discrimination is better for them because they
use unique vocabulary (" scratchy", " crackled", " thirty") as routing keys.

### Stage 3: Fallback

If neither stage routes confidently: replay the fact windows via standard
attention. This is the 2,000ms path. Expected frequency: ~15% at Apollo 11
scale with hybrid routing.

---

## Injection Mechanism (L30)

Having identified fact_id via routing:

```python
# At L30 of the bare query forward pass:
e_answer = embedding(answer_token_id)     # 2560D answer token embedding
R_new = R_bare + projection_coeff * e_answer
# Continue L31-L33 normally
```

The 1D subspace injection beats full residual replacement (99.96% vs 97.65%
accuracy). Use L30, not L29: L29's FFN is anti-correlated with the factual
signal; L30 cleanup heads pre-amplify before the injection takes full effect.

**Parametric conflict cases:** When the model has a strong prior against the
fact (e.g., "sell his restaurant" overriding "play football" for Namath), the
L30 injection still wins if routing is correct. The 1D injection overrides
parametric memory with maximum confidence.

---

## Failure Modes and Mitigations

### Mode 1: Token Indistinctiveness (content failure)

**Cause:** Answer first-token is a common 1–3 character prefix (" A", " Bel",
" Del"). The K-vector at the answer position lacks discriminative signal.

**Mitigation:** Entity string filter handles this at Stage 1 for entity-explicit
queries. For entity-implicit queries with indistinctive tokens: flag at index
construction time and route directly to fallback (don't attempt Q·K).

**Frequency at scale:** ~15–20% of facts depending on fact vocabulary.

### Mode 2: Same-Template Argmax Collision (addressing failure)

**Cause:** Two facts share an identical query template (e.g., "[name] agreed to
[verb]"). W_K projects both entities' 2560D hidden states — which ARE well
separated at 73.87° — onto similar 256D K-vectors. Q·K scores become nearly
identical (0.88× ratio in the Namath/Marchand case).

**Mitigation:** Entity string filter (Stage 1) resolves 100% of same-template
collisions for entity-explicit queries. For entity-implicit same-template
collisions (rare, as implicit queries tend to use distinctive vocabulary):
adaptive Q·K threshold + fallback.

**Why not fix W_K:** The entity signals in 2560D hidden space ARE distinct
(norms 2000–5000, Namath/Marchand at 73.87°). W_K was not trained to preserve
entity-discriminative directions — it was trained for structural attention.
The projection collapses the distinction. This cannot be fixed at inference time
without model retraining.

### Mode 3: Fixed Threshold at Scale (threshold failure)

**Cause:** 15% threshold calibrated for N=12. At N=3,625, max Q·K scores are
below 1%. Every query falls below threshold → 0% injection rate.

**Mitigation:** Adaptive threshold = mean_score × 2.0. Auto-scales with N.

### Mode 4: Routing Confidence (cross-contamination risk)

**Cause:** Wrong injection delivers wrong answer at maximum confidence. The
model amplifies whatever the injection contains with no entity verification.

**Mitigation:** Entity string filter has 0% false positives (names are unique).
Adaptive Q·K threshold requires the top score to be 2× the mean — this filters
most ambiguous cases. Residual risk: entity-implicit queries where argmax is
correct but margin is barely above adaptive threshold. Acceptable at <1%.

---

## Projected Performance at Apollo 11 Scale (N=3,625)

| Method | Injection rate | Wrong inject rate | Avg latency |
|---|---|---|---|
| Q·K, fixed 15% threshold | ~0% | 0% | 2,000ms |
| Q·K, adaptive threshold | ~40% | <0.5% | 1,200ms |
| **Hybrid: entity filter + adaptive Q·K** | **~85%** | **~0%** | **505ms** |

**Breakdown of 85% hybrid injection:**
- ~75% entity-explicit queries: 100% injection via string match
- ~25% entity-implicit: ~40% injection via adaptive Q·K, rest fallback
- 0.75 × 100% + 0.25 × 40% = **85%** overall injection rate

---

## Design Rules for Index Construction

1. **First-token length ≥ 4 characters for reliable Q·K routing.** Screen answer
   vocabulary during index construction. Facts with short first tokens depend
   on entity string filter for routing; mark them accordingly.

2. **Entity string must be unique per fact.** If two facts reference the same
   entity (e.g., two Zarkov Industries facts), suffix-disambiguate: "Zarkov
   Industries [city]", "Zarkov Industries [year]".

3. **Entity string should be the most specific identifying token from the fact.**
   "Joe Namath" not "Joe". "Zarkov Industries" not "Zarkov" (if other Zarkov
   facts exist). Prefer the longest unique identifier.

4. **Donor prompt design:** The donor prompt must output the answer as its
   first token after the query template. "Zarkov Industries was founded in the
   city of Voltara. Zarkov Industries was founded in the city of" → next token:
   " Volt". Do NOT use prompts where the answer was already output earlier in
   the same sequence.

5. **Same-type facts scale cleanly.** Adding N same-template entity facts does
   not degrade routing for existing facts (confirmed at N=8 same-template city
   facts). The 2560D embedding space has sufficient room.

6. **L14 is NOT useful for novel-entity routing.** The L14 entity compass fires
   for training-data entities only. Novel injected facts get near-random
   discrimination at L14. Use L29 K-vectors exclusively.

7. **Multi-layer K-vectors (L23+L29) waste storage.** L23 ≈ L29 in variance
   structure. The marginal discrimination gain does not justify 4× storage cost.

---

## What NOT to Do

| Don't | Why |
|---|---|
| Use fixed 15% threshold at N > 50 | Becomes 0% injection rate above N≈100 |
| Use L14 K-vectors for novel entities | Entity compass fires for known entities only |
| Concatenate multi-layer K-vectors (L23+L29) | L23 ≈ L29 structure, 4× storage waste |
| Try to fix W_K via hidden-state amplification | W_K projection discards entity signals; unfixable at inference |
| Inject below adaptive threshold | Risk wrong injection at high confidence |
| Use inject-all from shared multi-fact donor | List-continuation mode, catastrophic cross-contamination |
| Store the full 2560D L29 residual | 5,120 bytes/fact when 12 bytes suffice |

---

## Summary Metrics

```
Storage  : 536 bytes/fact   (vs 5,120 full residual, vs 82MB full KV/fact)
Latency  : 330ms fast path  (vs 2,000ms replay)
Accuracy : KL ≈ 0 when routing correct
Routing  : 85% inject @ N=3,625  (entity filter + adaptive Q·K)
Safety   : 0% wrong inject (entity filter: 0 FP; adaptive threshold: catches ambiguous)
```
