# A* Pathfinding over the Residual Stream
### Can geometric search replace sequential transformer inference?

**Model**: google/gemma-3-4b-it (34 layers, 2560 hidden dim, 8 attention heads, bfloat16)
**Experiment ID**: 4fe30d12-fdab-4f3b-a169-02eee7313b03
**Tools**: Lazarus MCP interpretability server

---

## Motivation

The residual stream of a transformer is a proven Markov state: `P(output | residuals_at_L) = P(output | full_history)`. No hidden channel exists. Every layer reads from and writes to a shared 2560-dimensional vector, and the entire computational history is captured in the current residual at any layer.

This raises a question: if the residual is the complete state, can we treat inference as a search problem — finding paths through residual space that reach the correct answer, potentially skipping or correcting layers along the way? This is the A* framing: treat each layer as a move, the residual at each layer as a node, and the target token's logit as the heuristic.

Eight experiments tested this hypothesis.

---

## Experiment 1: Terrain Map

**Question**: What does each phase of the network actually do?

Five diverse prompts were run through `residual_decomposition` and `compare_activations` to build a phase map.

**Three-phase architecture confirmed:**

| Phase | Layers | Dominant component | Function |
|-------|--------|--------------------|----------|
| 1 | L0-9 | Attention | Syntactic encoding, embedding compression |
| 2 | L10-23 | FFN | Slow semantic buildup |
| 3 | L24-33 | Mixed | Explosive answer crystallization |

**Convergence at L16**: Cosine similarity ~0.994-0.997 across all 5 prompts — the network processes them as generic factual queries, indistinguishable.

**Divergence at L24**: Cosine drops to 0.952-0.979. Content-specific computation begins. Head 1 at L24 fires on the salient entity token and writes the contextual attribute (nationality, cultural association) into the last-position residual.

Three distinct mechanisms:
- *Paris*: L24 attention fires directly (+5.56 logits)
- *Gold*: L25-26 FFN dominant
- *Warsaw*: Two-stage — L24 attention writes nationality, L25-26 FFN writes city

---

## Experiment 2: Heuristic Evaluation

**Question**: What signal is useful for A* to know where it is in the search?

Three candidates were tested:

**Logit lens (probability)**: Unreliable for compositional prompts. Example: for "Romeo and Juliet was written by _", the logit lens probability for " William" drops from L23→L26 (0.06→0.001%) while the cumulative logit rises monotonically. The FFN at L23-28 routes through " Shakespeare" (surname) before resolving to " William" (first name). Softmax distortion from competing surname tokens makes probability a misleading heuristic.

**Angular distance in 2560D**: Useless. All token embeddings sit ~88-90° from the residual in high-dimensional space. " Shakespeare" is always geometrically closer to the residual than " William", yet the model outputs " William". Zero discriminative power.

**Best heuristic**: Raw cumulative logit of the target token, with awareness of competing name clusters. Neither probability nor angle works reliably.

**Finding**: The A* heuristic problem is partially unsolved for non-trivial prompts.

---

## Experiment 3: Layer Skip Test

**Question**: Can we skip redundant layers?

Each of the 34 layers was ablated individually for "The capital of France is" using `ablate_layers`. A layer is "skippable" if the correct answer Paris remains top-1.

**Individual results**:
- Skippable: 29/34 layers (85.3%)
- Critical: L0, L4 (universal), L1, L7, L10 (France-specific)

**But multi-layer ablation fails catastrophically**:
- Ablating all 29 "safe" layers simultaneously → gibberish output
- Ablating just L11-L23 (13 layers) → full disruption across all 5 prompts

**Conclusion**: The 85% individual skip rate is an artifact of distributed compensation. Each layer is individually skippable because all others compensate. Simultaneously skipping cooperating layers destroys the computation entirely. Layer skipping is NOT viable as an A* strategy.

---

## Experiment 4: Portal Jump

**Question**: Can we bypass early layers by injecting an intermediate residual from a cached donor?

The portal layer is defined as the last layer before entity identity crystallizes into the last-position residual. From prior work, this is **L23** — Head 1 fires at L24, absorbing the entity.

**Results using `inject_residual`**:

| Scenario | Residual angle | Result |
|----------|---------------|--------|
| France → Japan (same type, last-pos) | ~3° | Tokyo 84-88% ✓ |
| France → Italy (same type, last-pos) | ~3° | Rome 88-92% ✓ |
| France → Marie Curie (cross-type, last-pos) | ~12° | Warsaw (degraded 98%→59%) |

**Injection at L24 vs L23**: Injecting the France residual at L24 (after entity absorption) → donor's Paris (78.9%) hijacks the output. Injecting at L23 (before absorption) → recipient's own entity dominates.

**Efficiency**: Injecting a cached L23 residual bypasses 24/34 layers = **67.6% compute savings** for same-type queries.

**Constraint**: Portal works only when residual angle < ~5°. Cross-type queries (angle 12-14°) cause the donor to hijack the recipient.

---

## Experiment 5: Single-Layer A* Routing on Compositional Failures

**Question**: Can we identify and zero a single misfiring component to recover the correct answer?

Four compositional failure cases were analyzed: crossing-point identification via `track_token`, attribution via `head_attribution`, and intervention via `component_intervention`.

### Case 1: Beethoven/Berlin (Type A — FFN wrong association)

**Prompt**: "The country where Beethoven was born has its capital in"
**Model output**: Vienna 45.7% (wrong — career city)
**Correct**: Berlin (Germany, Beethoven's birthplace country)

`head_attribution` at L26: Head 2 fires for " Beethoven" → +0.43 to Vienna, +0.32 to Berlin. Attention barely distinguishes. The FFN does the heavy lifting.

**Intervention results**:

| Component zeroed | Vienna | Berlin | Flips? |
|-----------------|--------|--------|--------|
| L26 FFN | 38.5% | **49.6%** | YES |
| L26 attention | 40.8% | 24.7% | NO |

Zeroing L26 FFN flips from wrong to correct. Zeroing L26 attention barely moves anything. Surgical precision matters — the FFN is the misfiring component; attention is not.

**False positive rate** — does L26 FFN zeroing break correct prompts?

| Prompt | Original | After | Breaks? |
|--------|----------|-------|---------|
| France/Paris | 80.9% | 62.9% | No |
| Germany/Berlin | 89.1% | 81.6% | No |
| Japan/Tokyo | 83.2% | 77.7% | No |
| Australia/Canberra | 92.2% | 57.8% | No |
| Marie Curie/Warsaw | 56.3% | 49.6% (tie) | No |

**0/5 false positives.** Simple prompts have redundant attention pathways that survive FFN ablation. Compositional prompts do not.

### Case 2: France-north/Brussels (Type B — attention entity misfiring)

**Prompt**: "The capital of the country that borders France to the north is"
**Model output**: Paris 26.0% (wrong — France's own capital)
**Correct**: Brussels (Belgium)

`head_attribution` at L24: Head 1 contributes +5.22 to Paris and +3.91 to Brussels. Head 1 reads "France" (the most salient entity token in the prompt) and routes to France's capital. It is performing correctly for direct France queries — the problem is that the prompt asks for a *relational* result ("neighbor of France"), not France itself.

**Intervention**: Zeroing Head 1 at L24 → Paris drops to 6%, Brussels reaches 6.9% (top city, but ":" wins at 23.9%). No clean flip. Head 1 zeroing removes the wrong answer but doesn't provide the right one — Brussels signal has no source in the residual.

**Conclusion**: Type B failures cannot be fixed by single-component zeroing.

### Case 3: Hamlet/English (Type C — format template)

**Prompt**: "The language in which Hamlet was originally written is"
**Model output**: ":" 40.2%, English 4.8%

`head_attribution` at L32: ALL 8 heads positive for English (+0.38 total). ALL 7 heads negative for Danish (-0.36 total). L32 attention is helping English. L32 FFN is also positive for English (+0.5 delta). The collapse from L24 (English ~100% in logit lens) to final (4.8%) is not caused by a single misfiring component — it is a distributional effect where the format template ":" captures probability mass across L30-33.

**Intervention**: Zeroing L32 FFN removes English from top-10 (it was helping). No addressable component.

**Conclusion**: Type C format failures are not correctable by component surgery.

---

## Experiment 6: Multi-Layer Lookahead

**Question**: When single-layer intervention fails, can a pre-conditioning injection fix France-north?

**Strategy**: Inject from "The capital of Belgium is" at L23 into the France-north prompt. The hypothesis: Belgium's residual state pre-conditions the last position before Head 1 fires, causing Head 1 to route to Brussels.

**Results**:

| Injection mode | Output top-1 | KL vs donor | KL vs recipient |
|----------------|-------------|-------------|-----------------|
| Last-position at L23 | " a" (generic) | 0.60 | 3.31 |
| Last-position at L20 | " a" (generic) | 2.42 | 3.61 |
| All-positions at L23 | **Brussels 57.6%** | **0.0** | 6.22 |

**Last-position injection fails** because Head 1 attends to ALL token positions. "France" appears explicitly in the recipient prompt; even with Belgium's last-position residual, Head 1 attends to "France" in other positions and routes to Paris.

**All-positions injection succeeds** — but this is equivalent to just running "The capital of Belgium is" from scratch. The recipient prompt is completely overwritten. KL=0.0 means the output is indistinguishable from the pure Belgium prompt.

**Finding**: All-positions injection requires already knowing the answer (Belgium). It is circular — you need to solve the compositional problem to provide the intervention. Multi-layer lookahead degenerates to oracle injection for Type B failures.

---

## Experiment 7: Move Ordering

**Question**: Given that L24-L26 FFN all contribute to the wrong answer in the Beethoven case, which is the optimal intervention point?

| Layer zeroed | Berlin % | Vienna % | Flips? | KL |
|-------------|---------|---------|--------|-----|
| L24 FFN | **61.7%** | 22.7% | YES | 0.297 |
| L25 FFN | **51.9%** | 10.3% | YES | 0.480 |
| L26 FFN | **49.6%** | 38.5% | YES | 0.136 |
| L27 FFN | 26.9% | 39.1% | NO | 0.077 |
| L28 FFN | 20.0% | 25.8% | NO | 0.238 |
| L26 attention | 24.7% | 40.8% | NO | 0.046 |

**Move ordering law**: Earlier is always better. Zeroing L24 FFN produces Berlin at 61.7% — the cleanest correction. Each subsequent layer compounds the Vienna advantage further into the residual, making later interventions progressively weaker.

**Circuit revealed**: Three consecutive FFN layers (L24→L25→L26) cooperatively install Vienna. This is a cascade — once L24 FFN fires and adds Vienna signal to the residual, L25 FFN reads that signal and amplifies it, and so does L26. After L26, the advantage is baked in and zeroing L27-L28 cannot recover it.

**A* principle confirmed**: Identify the EARLIEST misfiring component in the cascade, not the latest. The optimal move is at the root of the error chain.

---

## Experiment 8: A* vs Chain of Thought

**Question**: How does mechanical circuit surgery compare to the natural alternative — chain-of-thought prompting?

### Beethoven

| Method | Output | Confidence |
|--------|--------|------------|
| Direct prompt | Vienna | 45.7% |
| A* (zero L24 FFN) | **Berlin** | **61.7%** |
| CoT: "Beethoven was born in Bonn. Bonn is in Germany. The capital of Germany is" | **Berlin** | **100%** |

### France-north

| Method | Output | Confidence |
|--------|--------|------------|
| Direct prompt | Paris | 26.0% |
| A* (zero L24 H1) | ":" (confused) | — |
| CoT: "Belgium borders France to the north. The capital of Belgium is" | **Brussels** | **100%** |

**CoT achieves certainty on both cases.** Mechanism (consistent with prior work): CoT is input simplification, not circuit change. The compositional chain is resolved in the prompt text, leaving Head 1 with a single unambiguous entity to route from. The same circuit executes correctly when the intermediate entity is provided explicitly.

**A* achieves correctness only on Type A failures**, and at lower confidence than CoT.

---

## Summary: Failure Taxonomy and A* Viability

Three failure types, three different A* outcomes:

### Type A: FFN Wrong Association (Beethoven → Vienna over Berlin)
- **Cause**: L24-26 FFN stores career/cultural association that overrides birthplace-country reasoning
- **A* fix**: Zero earliest FFN in misfiring cascade (L24 optimal). False positive rate 0/5.
- **CoT fix**: Decompose into single-hop chain. Achieves 100%.
- **A* verdict**: ✓ Works, with lower confidence than CoT

### Type B: Attention Entity Misfiring (France-north → Paris over Brussels)
- **Cause**: Head 1 reads most salient entity ("France") instead of the relational result
- **A* fix**: None — head zeroing demotes wrong answer but doesn't provide correct signal. All-positions injection requires oracle (circular).
- **CoT fix**: Provide intermediate entity explicitly. Achieves 100%.
- **A* verdict**: ✗ Fails

### Type C: Format Template Competition (Hamlet → ":" instead of "English")
- **Cause**: L30-33 distributional shift toward colon-answer format overtakes content tokens
- **A* fix**: None — both attention and FFN at L32 are helping English. Regression is distributional, not component-level.
- **CoT fix**: Rephrase using geographic template; avoid "is" completion triggers.
- **A* verdict**: ✗ Fails

---

## Conclusions

**What A* over the residual stream can do**:
1. **Portal jumps** — Cache a donor residual at L23 and inject into same-type queries, bypassing 24/34 layers (67.6% savings). Requires <5° residual angle between queries.
2. **Type A FFN correction** — Identify the earliest FFN layer in a misfiring cascade and zero it. Works with 0 false positives. Requires interpretability tools to identify the misfiring layer.

**What A* cannot do**:
3. **Layer skipping in general** — Distributed cooperative redundancy means individually safe layers fail collectively.
4. **Type B attention errors** — The salient entity token in the prompt dominates Head 1's attention across ALL positions. You can't excise it without replacing the entire context.
5. **Type C format errors** — Distributional, not localized to any single component.

**Core limitation**: A* requires knowing the target state to construct a useful path. For portal jumps, the target state is a cached same-type residual — this is genuinely useful. For compositional failures, the correct target state requires solving the problem first. The method is circular for the hardest cases.

**Relationship to CoT**: Chain-of-thought is input simplification that achieves 100% accuracy by reducing multi-hop reasoning to single-hop completion. A* is surgical but achieves lower accuracy and only on a subset of failure types. CoT dominates on accuracy; A* provides mechanistic interpretability but not a reliable inference replacement.

**Architectural insight**: The bottleneck is not bandwidth or layer depth — it is Head 1's entity disambiguation capacity. Head 1 performs exactly one relational hop per forward pass. Multi-hop compositional reasoning requires chaining hops sequentially (CoT), which the single-pass architecture cannot do internally.
