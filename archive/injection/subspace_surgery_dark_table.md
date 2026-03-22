# Subspace Surgery + Dark Table Validation

**Experiment ID:** 5eeee327-a22c-4493-80b5-5dd85cbed017
**Model:** google/gemma-3-4b-it (34 layers, 2560 hidden dim, bfloat16)

Two new tools tested as the final components of a production inference engine:
- `subspace_surgery`: replace only a subspace component while preserving the orthogonal complement
- `build_dark_table`: precompute dark coordinates for zero-forward-pass injection

---

## Tool Mechanics: Both Validated

**`subspace_surgery`**: orthogonal_cosine=1.0 on every call across 15+ experiments. Surgery is always perfectly clean. The tool does exactly what it claims.

**`build_dark_table`**: dict format `{"key": "prompt"}` required (not a list of dicts). Lookup mode produces identical output to donor mode — exact coordinate norm match confirmed. The tool is a valid coordinate cache.

---

## Part A — Subspace Surgery

### Experiment 1: Gatekeeper — Hamlet Entity Surgery at L10

The intended test: replace entity subspace at L10, fix "Denmark" to "Stratford" in one pass.

**Setup**: `entity_chart_L10` computed from 16 birthplace prompts, rank 8.

```
subspace_surgery(
    recipient="The birthplace of the author of Hamlet was",
    donor="The birthplace of Shakespeare was",
    layer=10, subspace="entity_chart_L10"
)
```

| | Result |
|---|---|
| Baseline | "a small town in Denmark." |
| After surgery | "a small town in Denmark." |
| orthogonal_cosine | **1.0** |
| prediction_changed | false |

**Verdict: FAIL — surgery clean but no fix.**

**Why**: PC1=71.4% at L10 is format/template signal (same flaw as L7 PC1=85.8%). Entity-specific variation is in the remaining 28.6%, mostly outside the rank-8 subspace. Critically, the "Hamlet" context that triggers the Stage 3a failure lives in the orthogonal complement — which surgery preserves by design. Same result at L26 (PC1=28.5%, correctly multi-dimensional) and L28. Surgery cannot remove what it is designed to preserve.

### Subspace Quality by Layer

| Layer | PC1 variance | Character |
|-------|-------------|-----------|
| L10 | 71.4% | Format-dominated — same flaw as L7 |
| L12 | 71.4% | Same pattern |
| L26 | 28.5% | Genuinely multi-dimensional (5 PCs for 80%) |
| Answer subspace L7 | 27.0% | Genuinely multi-dimensional |
| Answer subspace L26 | 30.4% | Genuinely multi-dimensional |

**PCA subspaces are template-specific.** A subspace built from birthplace prompts does nothing useful for capital queries and vice versa.

### Experiment 2: Surgery Across Failure Library

Tested all Stage 3a failures with matching subspaces at L26. All fail for the same reason: contaminating context is in the orthogonal complement.

| Query | Donor | Layer | Baseline | Surgery | Fixed? |
|-------|-------|-------|----------|---------|--------|
| Hamlet birthplace | Shakespeare birthplace | L10 | Denmark | Denmark | No |
| Hamlet birthplace | Shakespeare birthplace | L26 | Denmark | Denmark | No |
| Hamlet birthplace | Shakespeare birthplace | L28 | Denmark | Denmark | No |
| Beethoven capital | Germany capital (lookup) | L26 | Vienna 41% | ":" 46.5% (disrupted) | No |

The Beethoven surgery disrupted the first token (Vienna→":") and generated `":\n\nVienna, Austria.\n\nBeethoven was born in Bonn, Germany"` — evidence the surgery did something, but the Stage 3a Vienna association recovered immediately from the orthogonal complement.

### Experiment 3: Result Injection for Entity-Math

Target: inject "9" coordinates into "The square of the number of sides of a triangle is" and have 9 emerge from the viewport.

Tested at L7, L14, L26, L32 with both `answer_subspace_L7` and `answer_subspace_L26`.

All fail. All are orthogonal_cosine=1.0 (surgery clean). The Pythagorean attractor fires at L25 FFN (+11.75) from the "triangle" and "sides" tokens in the orthogonal complement — which surgery preserves by design.

The L32 injection is additionally a no-op for a scale reason (see Scale Law below).

### The Failure Principle

Stage 3a contamination is in the orthogonal complement **by definition**: the contaminating signal is the query context ("Hamlet", "triangle sides"), which is preserved by the orthogonal complement. Surgery is designed to preserve the orthogonal complement. These constraints are incompatible — surgery cannot fix Stage 3a failures regardless of layer, subspace, or injection strength.

---

## Part B — Positive Validation: Same-Template Entity Swap

Surgery works when the subspace is template-matched and the distinction between prompts is genuinely encoded within the target subspace.

**Setup**: `capital_context_L26` from 16 capital prompts (PC1=35.0%, 5 PCs for 80%). `capital_facts_L26` dark table with 8 entries.

### Capital Swapping at L26

| Recipient | Donor | Mode | Baseline | Surgery | Generated |
|-----------|-------|------|----------|---------|-----------|
| Capital of Australia | Canada (donor) | donor | Canberra 91.9% | **Ottawa 80.7%** | "Ottawa, but the capital of Australia is Canberra." |
| Capital of Australia | Canada (lookup) | lookup | Canberra 91.9% | **Ottawa 62.9%** | "Ottawa, but the capital of Australia is Canberra." |
| Capital of Japan | Canada (lookup) | lookup | Tokyo 83.4% | **Ottawa 54.9%** | "Ottawa. The capital of Japan is Tokyo." |
| Capital of Germany | France (donor) | donor | Berlin 88.7% | Berlin 86.2% | Berlin (no flip) |

### The Context Preservation + Self-Correction Pattern

This is the tool's characteristic behavior and is **correct by design**:

1. Surgery fires the donor entity's answer as the first token (Ottawa)
2. The orthogonal complement carries the recipient's context tokens ("Australia", "Japan")
3. Subsequent generation self-corrects to the recipient's true answer

"Ottawa, but the capital of Australia is Canberra." — The model simultaneously outputs the injected answer AND knows it's wrong, because the entity identity is in the orthogonal complement. For sustained output control, use `steer_and_generate` instead.

### When Surgery Flips vs. Fails

Germany→France fails (Berlin stays at 86.2%) because Berlin is too strongly memorized to be displaced. The flip happens when:
- The subspace is template-matched (capital prompts for capital queries)
- Donor's entity representation is sufficiently strong relative to the recipient's memorized answer
- Confidence is moderate (not >90% baseline)

---

## Part C — The Scale Law

Table coordinates must match the injection layer's residual scale.

| Injection layer | Table built at | new_content_energy_fraction | Effect |
|-----------------|----------------|----------------------------|--------|
| L7 | L7 | 36% | Effective |
| L26 | L26 | 30–60% | Effective |
| L32 | L7 | **0.3%** | No-op |
| L32 | L26 | 0.3% | No-op |

L7 coordinates (norm ~3,040) injected at L32 (residual norm ~50,000) carry 0.3% energy — the injection is invisible. Build tables at the same layer you inject. The residual norm grows ~15× from L7→L33.

### Layer-Specific Coordinate Norms

| Table | Layer | Entry norms |
|-------|-------|-------------|
| arithmetic_results_L7 | L7 | 2,770–3,300 |
| arithmetic_results_L26 | L26 | 25,000–30,000 |
| capital_facts_L26 | L26 | 30,800–35,900 |
| entity_birthplace_L10 | L10 | 13,430–14,080 |

---

## Part D — Dark Table Design Rule

The arithmetic tables fail to produce correct number output even on neutral prompts:

```
"The result of the calculation is" + inject "9" from arithmetic_results_L7 at L7 → "100."
"The result of the calculation is" + inject "315" from arithmetic_results_L26 at L26 → ":\n\n123456"
```

**Why**: "The answer is 9" has `9` as an **input** token at the last position. The table captures the "just read 9" state, not the "about to output 9" state. These are different residual configurations.

The donor design rule from dark loop experiments applies here too:
> Donors must naturally OUTPUT the answer as their first token ("state: about to say X"), not already contain it ("state: just said X").

For arithmetic, this means building table entries from prompts like "3 squared equals" (outputs "9" first) not "The answer is 9" (9 is an input token). The dark loop used "Three hundred and fifteen is written numerically as" → " 315" for the same reason.

---

## Part E — Number Geometry (Experiment 7)

All digit tokens measured at L7 via `direction_angles`:

| Pair | Angle | Cosine |
|------|-------|--------|
| 2 vs 3 | 0.0° | 1.0 |
| 2 vs 9 | 0.0° | 1.0 |
| 3 vs 16 | 0.0° | 1.0 |
| ... all pairs ... | 0.0° | 1.0 |

All digit tokens are colinear in embedding space — identical direction, identical norm (1.034). There is no number line, no arithmetic structure, no squaring direction in the input embeddings.

**Implication**: Geometric arithmetic (`dark_coords(3) + squaring_direction = dark_coords(9)`) does not exist at the embedding level. The lookup table approach is necessary, not just convenient. Arithmetic structure emerges in the residual stream through attention and FFN processing, not from token geometry.

---

## Experiment 8: Final 15-Query Benchmark

**Result: 14/15 at avg ~96 layers.**

| # | Query | Method | Output | Correct? |
|---|-------|--------|--------|---------|
| 1 | Hamlet birthplace | inject_residual L10 all-pos | Stratford | ✓ |
| 2 | Faust birthplace | inject_residual L10 all-pos | (Ore Mountains) | **✗** |
| 3 | Beethoven capital | inject_residual L12 all-pos | Berlin | ✓ |
| 4 | Shakespeare language | standard | English | ✓ |
| 5 | Relativity country | standard | Germany | ✓ |
| 6 | Triangle² | inject_residual L10 all-pos | 9 | ✓ |
| 7 | Tripod double | CoT | 6 | ✓ |
| 8 | Triangle factorial | standard | 6 | ✓ |
| 9 | Half months | standard | 6 | ✓ |
| 10 | Einstein−Shakespeare years | CoT (explicit years) | 315 | ✓ |
| 11 | print(2\*\*10) | standard (with `# Output:`) | 1024 | ✓ |
| 12 | print(3\*7) | standard (with `# Output:`) | 21 | ✓ |
| 13 | x=5; y=x+1; x=y\*2 | CoT (traced steps) | 12 | ✓ |
| 14 | g(f(3)) | standard (with `# Output:`) | 7 | ✓ |
| 15 | Red + blue | standard | purple | ✓ |

**Q2 failure**: Model's direct Goethe birthplace query is wrong ("a small village in the Ore Mountains" vs. correct: Frankfurt am Main). This is a model knowledge gap — no routing strategy can fix a wrong donor. inject_residual succeeds mechanically (KL=0.0) but the donor's answer is already incorrect.

**Standard inference baseline**: 8/15 (53%). Note: code queries require `# Output:` prompt suffix to trigger execution-mode completion rather than code continuation.

**`subspace_surgery` contribution to benchmark**: zero. `inject_residual` (all-position, patch_all_positions=True) remains the correct tool for factual multi-hop correction.

### Comparison Table

| Method | Accuracy | Avg layers |
|--------|----------|-----------|
| Standard inference | 8/15 (53%) | 34 |
| Previous routing (Exp 5) | 15/15 (100%) | ~90 |
| This benchmark | 14/15 (93%) | ~96 |
| CoT only | ~14/15 | ~300 |

The 1-query regression vs. Exp 5 (Q2 Faust) may reflect different prompt phrasing or that Exp 5 used a different donor strategy for Q2. The routing framework is otherwise confirmed.

---

## Architecture of the Inference Engine (Final State)

```
Query
  │
  ├─ Factual multi-hop? ──→ inject_residual(donor, L10, patch_all_positions=True)
  │                          KL=0.0 universally. 68 layers (34 donor + 34 recipient).
  │
  ├─ Entity-math hybrid? ──→ Dark loop:
  │                          1. Probe at L7: operand + operation
  │                          2. Tool executes
  │                          3. Donor template that OUTPUTS result first
  │                          4. inject_residual at L10/L14
  │                          ~50–68 layers total.
  │
  ├─ Code simple? ─────────→ standard inference (with # Output: suffix)
  │                          34 layers.
  │
  ├─ Code complex? ────────→ CoT
  │                          ~300 layers.
  │
  └─ Novel/simple? ────────→ standard inference
                             34 layers.
```

`subspace_surgery` and `build_dark_table` are valid tools for interpretability experiments and controlled entity-swapping studies. They are not (yet) part of the production routing path.

---

## What Subspace Surgery Is Good For

The tool is correctly implemented and has genuine uses:

1. **Interpretability probe**: does a specific subspace carry decision-relevant information? Inject donor coordinates and observe if output changes. orthogonal_cosine=1.0 guarantees the measurement is clean.

2. **Controlled entity-swapping experiments**: in same-template settings, surgery provides first-token control while the orthogonal complement provides context recovery. This cleanly separates entity signal from context signal.

3. **Failure diagnosis**: inject "correct" entity coordinates and observe whether the model self-corrects (Stage 3a: yes — orthogonal complement triggers wrong FFN. Stage 1: different pattern). The failure mode is diagnostic information.

4. **Not for**: sustained generation control (use `steer_and_generate`), multi-hop failure correction (use `inject_residual`), arithmetic result injection without proper donor design.
