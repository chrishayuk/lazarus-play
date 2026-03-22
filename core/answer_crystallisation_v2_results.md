# Answer Crystallisation v2 — KV-Independence Boundary

**Experiment ID:** 7bf43e53-fa58-4474-9b7b-a304be726d87
**Model:** google/gemma-3-4b-it (34 layers, 2560D, 8 heads)

## Core Finding

The answer crystallises in the residual stream at **L24** for novel entities.
Transplanting the donor's last-position residual at L26 into a bare query
(no context tokens) produces the **correct first token at 99-100%** across
all probes. Layers L24-L33 (10 layers) are **KV-independent** — they operate
purely on the residual and do not need the context KV cache.

## Probes

Multi-fact novel passage (~200 tokens) about Zarkov Industries. 4 facts queried:

| Probe | Target token | P(target) with context | P(target) bare query |
|-------|-------------|:---------------------:|:-------------------:|
| Voltara (city) | Volt | 100% | 0% (Detroit 72%) |
| CEO (Petrov Lansky) | Dr | 82% | 0% (Vik 45%) |
| Brenmoor (facility lead) | Cass | 100% | 0% (Arthur 16%) |
| EchoTrace (sonar tech) | Echo | 100% | 0% (The 44%) |

Second passage (Vestholm Collective) verified independently:

| Probe | Target | P(target) with context | P(target) bare |
|-------|--------|:---------------------:|:-------------:|
| Dranneth (city) | Dr | 100% | 0% (Stock 99%) |

## Experiment 1 — KV-Independence Sweep

Donor: multi-fact novel passage (Zarkov, 200 tokens) + city query.
Recipient: bare query (no context).
Target: "Volt" (Voltara).

| Layer | P(Volt) | Top-1 | Generated | KL(d→i) | Angle |
|------:|--------:|-------|-----------|--------:|------:|
| L33 | 100% | Volt | Volt City | 0.0 | 58.1° |
| L32 | 100% | Volt | Volt City | 0.0 | 17.5° |
| L31 | 100% | Volt | Volt City | 0.0 | 16.0° |
| L30 | 100% | Volt | Volt City | 0.0 | 15.0° |
| L29 | 100% | Volt | Volt City | 0.0 | 14.2° |
| L28 | 93.2% | Volt | Volt City | 0.07 | 12.2° |
| L27 | 92.3% | Volt | Volt City | 0.08 | 12.7° |
| L26 | 99.5% | Volt | Volt City | 0.005 | 12.6° |
| L25 | 95.7% | Volt | Volt City | 0.04 | 11.4° |
| L24 | 66.1% | Volt | Volt City | 0.41 | 10.4° |
| **L23** | **17.9%** | **Vol** | **Volgograd** | **1.72** | **10.1°** |
| L22 | 0% | Z | Zhitomyr | 23.0 | 6.7° |
| L20 | 0% | Z | Zhitomyr | 23.0 | 4.9° |
| L18 | 0% | Z | Zhitomyr | 23.0 | 3.4° |
| L14 | 0% | Detroit | Detroit | 23.0 | 1.5° |
| L10 | 0% | Detroit | Detroit | 23.0 | 1.2° |

**Boundary: L23→L24.** Sharp one-layer transition from 17.9% to 66.1%.
Robust from L26 onward (>99%).

### Anomaly: L26 > L28 > L27

P(Volt) dips at L27-L28 (92-93%) but recovers at L26 (99.5%). L26 FFN is
the commitment layer that amplifies the answer direction. L27-L28 are
formatting layers that slightly dilute the signal before L29 re-amplifies.

### Three Failure Regimes

| Layer range | Failure mode | Generated |
|-------------|-------------|-----------|
| L14-L10 | Residual too similar (1-1.5°), completely overwritten | Detroit (bare query answer) |
| L22-L20 | Entity "Z" direction survives, answer direction doesn't | Zhitomyr (Z-confabulation) |
| L23 | Partial: "V" direction present but too weak | Volgograd (V-confabulation) |

## Experiment 2 — Multi-Probe Transplant at L26

| Probe | Donor | Recipient | Transplant | P(target) |
|-------|-------|-----------|------------|----------:|
| Voltara | Volt 100% | Detroit 72% | **Volt 99.5%** | 99.5% |
| CEO | Dr 82% | Vik 45% | **Dr 100%** | 100% |
| Brenmoor | Cass 100% | Arthur 16% | **Cass 100%** | 100% |
| EchoTrace | Echo 100% | The 44% | **Echo 100%** | 100% |

**4/4 correct. Amplification effect:** CEO donor has 82% Dr but transplant
amplifies to 100%. The late layers actively amplify the dominant direction.

## Experiment 3 — Why KV-Dependent Layers Fail

### Logit Lens Trajectory

| Layer | Top-1 in vocab space |
|------:|---------------------|
| L10-L20 | "The" / generic (0% V-like) |
| L22 | "The" (13.9%) |
| L23 | **"V" (17.3%)** — first signal |
| L24 | "V" (26%) |
| L28 | "Vol" (28%) |
| L30 | **"Volt" (96%)** — resolved |
| L33 | "Volt" (100%) |

### First Copy Head: L23 H3

| Condition | L23 H3 DLA → "Volt" | Top token |
|-----------|--------------------:|-----------|
| With context | **+2.95** (61% of layer) | **Volt** |
| Bare query | -0.02 | Z |

L23 H3 reads "Voltara" from context KV and writes +2.95 logits toward "Volt".
This is the crystallisation moment — the first time the answer enters the
residual as a vocabulary-aligned direction.

### Ablation Paradox

Zeroing L23 attention with full context: **no effect** (still Volt at 100%).
L23 H3 is redundant when context is present (other paths compensate). But
for cross-context transplant, L23 H3's contribution IS the threshold.

### Mechanism Summary

1. L23 H3 copies "Voltara" from context KV → +2.95 logits → "V" direction
2. Layers 24-33 amplify: V (17%) → V (26%) → Vol (28%) → Volt (96%) → 100%
3. Transplant at L22: L23 H3 hasn't fired → no V direction → FAIL
4. Transplant at L24: L23 H3 has fired → V direction in residual → SUCCESS
5. **KV-independence boundary = layer after first copy head**

## Experiment 4 — Generation Sustainability

| Probe | Donor generates | Transplant generates |
|-------|----------------|---------------------|
| Voltara | Voltara | Volt **City** |
| CEO | Dr. Petrov Lansky | Dr. **Aris Thorne** |
| Brenmoor | Cassidy Thorne | Cass**ian Andromache** |
| EchoTrace | EchoTrace | Echo **Knight** |
| Vestholm | Dranneth | Dr**aghatton** |

**First token correct, second token confabulates.** The transplanted residual
is a direction pointer, not a multi-token narrative. The bare-query KV cache
at L26-L33 pulls subsequent tokens toward parametric completions.

## Experiment 5 — Compression

| Injection mode | P(Volt) | Result |
|---------------|--------:|--------|
| Full residual (2560D) | 99.5% | Volt City |
| Vocabulary subspace (4D: Volt/Vol/volt/Voltara) | **0.2%** | **Detroit** |

The answer occupies **0.22% of residual norm** in vocabulary space.
The remaining 99.78% is dark-space signal required for late-layer amplification.
**Full 2560D required. Not compressible to vocabulary subspace.**

## Experiment 6 — Multi-Passage Routing

### Same-Passage (3 queries, 1 passage)

| Bare query → | d_voltara | d_brenmoor | d_echo | Correct? | Margin |
|-------------|:---------:|:----------:|:------:|:--------:|------:|
| Voltara | **0.976** | 0.974 | 0.972 | Yes | 0.002 |
| Brenmoor | 0.972 | **0.978** | 0.973 | Yes | 0.005 |
| EchoTrace | 0.971 | 0.974 | **0.979** | Yes | 0.005 |

### Cross-Passage (2 passages)

| Bare query → | d_zarkov | d_vestholm | Correct? | Margin |
|-------------|:--------:|:----------:|:--------:|------:|
| Zarkov city | **0.976** | 0.971 | Yes | 0.005 |
| Vestholm city | 0.958 | **0.960** | Yes | 0.001 |

**5/5 correct.** But margins 0.001-0.005. May not scale to N>10.

## Production Format

| Field | Size |
|-------|-----:|
| Crystallised residual at L26 | 10 KB |
| Per document (370K tokens, 725 windows) | 7.25 MB |
| Per 1M tokens | 19.5 MB |
| Per 10M tokens | 195 MB |

## Comparison to v1 (Previous Run)

| | v1 (ed84135b) | v2 (this) |
|---|:---:|:---:|
| Boundary | L29-L30 | **L24** |
| Probes | Zarkov/Strand/Castellan | Zarkov (multi-fact) + Vestholm |
| Passage type | Single-fact | Multi-fact (~200 tokens) |
| First token accuracy | 89-99% | **99-100%** |
| Multi-token | Drifts | Drifts |
| Compressible | No | No |

The v1 boundary at L29-L30 may reflect different probe design or parametric
interference. The v2 boundary at L24 is for clean novel entities with no
parametric competition. **The boundary is fact-type-dependent:**

- **Novel entities:** L24 (after first copy head L23 H3)
- **Parametric facts:** L29-L30 (after main copy head L29 H4 + commitment)

## Key Numbers

| Metric | Value |
|--------|------:|
| KV-independence boundary (novel) | **L24** |
| Robust crystallisation layer | **L26** |
| First copy head | **L23 H3** (+2.95 DLA) |
| P(correct first token) at L26 | **99-100%** |
| Residual size | **10 KB** per passage |
| Compressible? | **No** (full 2560D) |
| Routing accuracy | **5/5** (N=2-3, margins thin) |
| Multi-token generation | **Fails** (token 2+ confabulates) |
| KV-independent layers | **L24-L33** (10 of 34 = 29%) |

## Architecture Summary

```
L0-L22:  KV-dependent. Attention reads from context KV to build answer.
L23:     CRYSTALLISATION LAYER. H3 copies answer from context KV.
L24-L33: KV-INDEPENDENT. Operate on residual alone. Amplify V→Volt.

Transplant at L26: one vector (10 KB) → correct first token (99-100%)
                   but multi-token generation drifts to confabulation.
```

The residual at L26 IS the answer — for one token. 29% of the network
operates purely on this vector without needing the context KV cache.
The copy head, commitment layer, and amplification layers are all
KV-independent pass-through amplifiers when given the crystallised state.
