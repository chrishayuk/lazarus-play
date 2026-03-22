# Copy Head Calibration — Experiment Results
**Experiment ID:** 71541b5d-81d8-4dc5-8205-35857974c03a
**Date:** 2026-03-19

## Summary

The current calibrate_arch.py uses cosine similarity between head output and answer embedding.
This is wrong — it finds cleanup/amplifier heads instead of the originating copy head.

**Two correct methods validated:**
1. **DLA through unembedding** (head_attribution) — fast scan, shortlists candidates
2. **Causal ablation** (component_intervention zero) — confirms the true copy head

**Combined method:** DLA top-5 → ablation confirmation → injection validation.
Validated on both Gemma 4B and 1B.

---

## Method Comparison on Gemma 4B

| Method | Rank-1 result | Correct? | Notes |
|--------|--------------|----------|-------|
| Cosine proxy (current) | L30 H7 | **NO** | Cleanup head |
| Raw DLA alone | L31 H7 | **NO** | Amplifier (fights L31 H6) |
| Ablation alone (mean) | L30 H7 | **NO** | Outlier-driven (2/6 probes) |
| Ablation (consistency) | **L29 H4** | **YES** | 4/6 probes affected |
| **DLA top-5 + ablation** | **L29 H4** | **YES** | In DLA top-5 (#5), wins ablation consistency |

### Why each method fails alone:
- **Cosine proxy**: measures alignment with answer embedding, picks up cleanup heads that amplify existing signals
- **Raw DLA**: amplifier heads at L30-L31 have higher absolute DLA than the originating copy head at L29
- **Ablation (mean)**: L30 H7 has outlier effects on 2 probes but zero effect on 4 others

### Why the combined method works:
1. DLA shortlists the right neighborhood (L29 H4 is in top-5)
2. Ablation consistency metric (# probes with ΔP < -1pp) correctly identifies the originating copy head
3. Amplifier heads have high DLA but low/zero ablation effect (their signal is redundant)

---

## Gemma 4B Results

### Architecture
- 34 layers, 2560 hidden dim, 8 attention heads

### Copy head: L29 H4 (85% through model)

| Metric | L29 H4 | L30 H7 | L31 H7 | L30 H0 | L30 H3 |
|--------|--------|--------|--------|--------|--------|
| Mean DLA | +0.833 (#5) | +1.607 (#2) | +3.123 (#1) | +1.043 (#3) | +0.855 (#4) |
| Layer dominance | 100.2% | 84.7% | fights H6 | 72.8% | 56.2% |
| Ablation mean ΔP | -5.2pp | -8.1pp | -0.3pp | -2.2pp | -0.9pp |
| Ablation consistency | **4/6** | 2/6 | 0/6 | 2/6 | 2/6 |

### Injection validation at L30

| Probe | P(answer) injected | KL | |
|-------|-------------------|-----|---|
| Zarkov/Volt | 99.8% | 0.006 | ✓ |
| Nexaris/Cer | 96.5% | 0.009 | ✓ |
| Helion/Dra | 98.5% | 0.027 | ✓ |
| Keltara/Sol | 98.6% | 0.001 | ✓ |
| Namath/endorse | 93.8% | 0.005 | ✓ |
| Marchand/sell | 93.1% | 0.010 | ✓ |

**Mean: P=96.7%, KL=0.010**

### Final config (4B)
```
retrieval_layer = 29
query_head = 4
injection_layer = 30
```

---

## Gemma 1B Results

### Architecture
- 26 layers, 1152 hidden dim, 4 attention heads

### Copy head: L23 H1 (88% through model)

**DLA top-5:**

| Rank | Layer | Head | Mean DLA |
|------|-------|------|----------|
| 1 | L20 | H2 | +1.715 |
| 2 | L17 | H0 | +1.551 |
| 3 | **L23** | **H1** | **+1.393** |
| 4 | L20 | H3 | +1.012 |
| 5 | L18 | H3 | +1.006 |

**Ablation confirmation:**

| Rank | Layer | Head | Mean ΔP | #Affected |
|------|-------|------|---------|-----------|
| 1 | **L23** | **H1** | **-32.5pp** | **4/4** |
| 2 | L18 | H3 | -22.5pp | 4/4 |
| 3 | L17 | H0 | -11.2pp | 4/4 |
| 4 | L20 | H2 | -4.9pp | 3/4 |
| 5 | L20 | H3 | -0.3pp | 0/4 |

**Current cosine proxy found L18 H3** — rank 2 by ablation. Not terrible but suboptimal.

### Injection validation at L24

| Probe | P(answer) injected | KL | |
|-------|-------------------|-----|---|
| Zarkov/Volt | 97.2% | 0.0004 | ✓ |
| Keltara/Sol | 83.7% | 0.002 | ✓ |
| Namath/endorse | 93.0% | 0.0001 | ✓ |
| Marchand/sell | 91.2% | 0.0003 | ✓ |

**Mean: P=91.3%, KL=0.0007**

### Final config (1B)
```
retrieval_layer = 23
query_head = 1
injection_layer = 24
```

---

## Cross-Model Scaling Law

| Property | Gemma 1B | Gemma 4B | Ratio |
|----------|----------|----------|-------|
| Copy head layer | L23 | L29 | — |
| % through model | 88% | 85% | ~constant |
| Copy head head | H1 (of 4) | H4 (of 8) | — |
| Injection layer | L24 | L30 | copy + 1 |
| Injection P(answer) | 91.3% | 96.7% | scales with capacity |
| Injection KL | 0.0007 | 0.010 | both excellent |

**The copy head lives at ~85-88% through the model.** This is a positional scaling law
that should generalize to other Gemma sizes and potentially other architectures.

---

## New Insight: Copy Head Multiplicity

On Gemma 4B, the copy LAYER is L29 but different heads handle different token types:
- **H4**: novel/rare entity tokens (Volt, Cer, Dra) — PRIMARY
- **H3**: common tokens (Sol)
- **H5**: common verbs (sell)

On 1B, the circuit is more distributed with multiple important layers (L17, L18, L23),
suggesting smaller models spread the copy function across more layers.

---

## Recommended calibrate_arch.py Algorithm

```python
def calibrate(model, tokenizer, probes):
    n_layers = model.config.num_hidden_layers
    n_heads = model.config.num_attention_heads

    # Scan top third of model
    scan_start = int(n_layers * 0.65)
    scan_end = n_layers - 1

    # Phase 1: DLA scan (one forward pass per layer per probe)
    dla_scores = {}
    for layer in range(scan_start, scan_end):
        for probe in probes:
            result = head_attribution(probe.prompt, layer, probe.target_token)
            for head in result.heads:
                key = (layer, head.index)
                if key not in dla_scores:
                    dla_scores[key] = []
                dla_scores[key].append(head.logit_contribution)

    # Rank by mean DLA, take top-5
    mean_dla = {k: sum(v)/len(v) for k, v in dla_scores.items()}
    top5 = sorted(mean_dla, key=mean_dla.get, reverse=True)[:5]

    # Phase 2: Ablation confirmation (5 candidates × N probes)
    ablation = {}
    for layer, head in top5:
        drops = []
        for probe in probes:
            result = component_intervention(
                probe.prompt, layer, "head", head, "zero"
            )
            delta = result.original_p - result.intervened_p
            drops.append(delta)
        ablation[(layer, head)] = {
            'mean_drop': sum(drops) / len(drops),
            'consistency': sum(1 for d in drops if d > 0.01),
        }

    # Winner: most consistent drops, tiebreak by mean drop
    best = max(ablation, key=lambda k: (
        ablation[k]['consistency'],
        ablation[k]['mean_drop']
    ))

    # Phase 3: Injection validation
    inject_layer = best[0] + 1
    # ... validate injection works ...

    return ArchitectureConfig(
        retrieval_layer=best[0],
        query_head=best[1],
        injection_layer=inject_layer,
    )
```

## Key Answers

1. **Does causal ablation find L29 H4 on 4B?** YES, when using consistency metric (4/6 probes).
2. **Does DLA find L29 H4?** Not alone (rank 5), but it shortlists it.
3. **Does the combined method work?** YES — DLA top-5 + ablation consistency = L29 H4 on 4B, L23 H1 on 1B.
4. **Does injection work on 1B?** YES — 4/4 probes, mean P=91.3%, KL=0.0007.
5. **Is the cosine proxy wrong for 1B?** YES — it found L18 H3 (rank 2). The correct answer is L23 H1 (rank 1).
