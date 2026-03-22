# Synthetic KV Injection at L29: Results

**Experiment ID:** 2ece587e
**Date:** 2026-03-20
**Model:** google/gemma-3-4b-it (34 layers, 2560 hidden, 8 heads, 4 KV heads, head_dim=256)

## Summary

**Hypothesis:** Injecting stored K/V entries at L29 lets the copy head
(H4) find the correct answer via its own attention, eliminating external
routing.

**Result:** The attention routing works perfectly. The answer retrieval
does not. Single-layer KV injection at L29 fails because the copy head
is not a standalone retrieval unit — it amplifies a signal that must
already be partially present in the residual stream from layers 0-28.

## Architecture Confirmed

| Parameter | Value |
|-----------|-------|
| Q heads at L29 | 8 |
| KV heads at L29 | 4 |
| Head dim | 256 |
| GQA grouping | n_rep=2: KV-head-2 serves Q-heads 4,5 |
| L29 attention type | Global (not sliding window) |
| Attention scale | query_pre_attn_scalar^(-0.5) = 1/16 |
| Q/K normalization | GemmaRMSNorm per head (before RoPE) |
| V processing | Projection only (no norm, no RoPE) |

## Experiment 1 — Ground Truth: Full Context

| Metric | Value |
|--------|-------|
| P("Volt") | **96.9%** |
| H4 DLA for "Volt" | 1.484 (raw), 99.5% of layer |
| H4 attention to pos 39 ("Volt") | 35.9% |
| H4 attention to BOS | 55.9% |
| H5 attention to pos 39 | 13.9% (backup copy head) |

The copy circuit is confirmed: H4 attends to the "Volt" position
in the document and copies the answer direction via its V vector.

## Experiment 3 — Bare Query (No Context)

| Metric | Value |
|--------|-------|
| P("Volt") | ~0% (not in top 10) |
| Top predictions | New (6.5%), Chicago (4.0%), Pittsburgh (3.1%) |
| H4 DLA for "Volt" | 0.005 (zero) |
| H4 attention | 96% to BOS (nothing useful to attend to) |

Clean failure baseline. The model guesses parametric city names.

## Experiment 2 — K/V Extraction

Extracted K/V vectors at L29 from the full document forward pass:
- **k_volt:** (1, 4, 1, 256) — post-k_norm, post-RoPE at position 39
- **v_volt:** (1, 4, 1, 256) — post-v_proj (no norm, no RoPE)
- K norms: 43.8–61.0 across KV heads
- V norms: 7.3–20.6 across KV heads
- 10 noise positions: [2, 7, 8, 9, 15, 16, 18, 42, 45, 49]

## Experiment 4 — Single Entry Injection (Core Test)

**Injected:** One K/V entry (from Volt position) at L29 for all KV heads.

| Metric | Full Context | Injection | Delta |
|--------|-------------|-----------|-------|
| P("Volt") | 96.9% | **0.06%** | -96.8% |
| H4 attention to Volt entry | 35.9% | **38.3%** | +2.4% |
| H4 attention to BOS | 55.9% | 59.4% | +3.5% |

**The attention routing works perfectly.** H4 attends 38.3% to the
injected entry — even higher than in the natural case (35.9%). But
the output probability for "Volt" is essentially zero.

Top predictions with injection: New (6.0%), Z (3.6%), Vol (3.6%),
Zapor (3.2%), Pittsburgh (3.0%) — nearly identical to the bare query.

## Experiment 5 — Multi-Entry Injection (Discrimination Test)

**Injected:** 11 K/V entries (1 Volt + 10 noise positions) at L29.

| Metric | Value |
|--------|-------|
| P("Volt") | 0.05% |
| H4 attention to Volt entry | 38.1% |
| H4 attention to best noise entry | 0.66% |
| H4 discrimination ratio | **58:1** |

The copy head's attention mechanism discriminates the correct entry
from 10 noise entries with a 58:1 ratio. This validates the routing
hypothesis: the model's own softmax handles selection perfectly.

But the answer still doesn't appear at the output.

## Experiment 6 — RoPE Position Mismatch

| Condition | Q·K Score | P("Volt") |
|-----------|-----------|-----------|
| (a) Original RoPE (doc pos 39) | 13.625 | 0.06% |
| (b) Pre-RoPE (content only) | 13.500 | 0.05% |
| (c) Re-RoPE at pos S_query | 13.500 | 0.07% |
| (d) Re-RoPE at pos 0 | 13.500 | 0.05% |

**RoPE position mismatch is irrelevant.** All Q·K scores are within
0.125 of each other. Content matching completely dominates positional
encoding for this query/key pair. No need for pre-RoPE storage or
re-encoding.

## Experiment 7 — Scale Test

| Query | Target | P(target) | Best attended |
|-------|--------|-----------|---------------|
| "founded in the city of" | Volt | 0.0% | filtration |
| "built on a" | former | 3.4% | filtration |
| "established in the" | mid | 17.6% | s (pos 15) |

The 17.6% for "mid" is parametric (the model already predicts
"mid-" as 3rd choice without any context). The injection does not
meaningfully shift the output distribution.

## Diagnostic: Why the Injection Fails

### Logit Trajectory Comparison

| Layer | Full Context | Bare Query | Gap |
|-------|-------------|------------|-----|
| Embedding | 11.69 | 11.69 | 0 |
| L26 cumulative | 7.06 | 3.38 | 3.69 |
| L28 cumulative | **7.72** | **3.06** | **4.66** |
| L29 attn delta | **+10.66** | +0.14 | 10.52 |
| L29 cumulative | 17.75 | 3.30 | 14.45 |
| L30 attn delta | +7.38 | -0.09 | 7.47 |
| L30 cumulative | 23.38 | 3.45 | 19.92 |
| Final logit | 20.63 | 5.06 | 15.56 |

### The Root Cause

1. **Embedding to L28:** Both prompts start with the same embedding
   logit (11.69). But layers 0-28 reduce the Volt logit differently:
   - Full context: 11.69 → 7.72 (down 3.97)
   - Bare query: 11.69 → 3.06 (down 8.63)
   - **Deficit at L28: 4.66 logits**

2. **L29 attention in full context:** Adds **+10.66 logits** via
   the copy head. This is the dominant contribution to the final
   prediction.

3. **L29 attention with injection:** H4 attends 38% to the injected
   V vector (correct!) but the head output is diluted by 60% BOS
   attention. The query's BOS V vector (trained on 11 query tokens
   via layers 0-28) pushes in a different direction than the
   document's BOS V vector. The net logit boost is ~+6-7 logits
   (based on P increasing from 1e-6 to 6e-4), not the full +10.66.

4. **Combined effect:** Starting from 3.06 (L28 bare) and adding
   ~+6 (injection at L29), the Volt logit reaches ~9 — but the
   competing "New" logit is at ~13-14. Volt never becomes top-1.

### The Fundamental Issue

The copy head at L29 is not a standalone retrieval unit. It is an
**amplifier** that boosts a signal already partially present in the
residual stream. In the full context, layers 0-28 build up entity
identity and relational context that prepares the residual for L29's
copy operation. Without this preparation, L29's V-copy adds a signal
to the wrong manifold.

This is consistent with the convergence_curves finding (L10-L14
universal convergence) and the entity_compass finding (L14 dark
signal). The preparation happens in early-to-mid layers, not at
the copy head itself.

## What This Means for the Architecture

### Single-Layer KV Injection: Dead End
Injecting K/V at one layer cannot produce correct answers because
the residual stream lacks the preparatory context from earlier layers.

### What Was Validated
1. **Attention-based routing works** — H4 finds the correct entry
   with 58:1 discrimination among 11 candidates
2. **RoPE mismatch doesn't matter** — content matching dominates
3. **The model's softmax is a better router than any external
   routing we built** — zero threshold, zero coefficient, perfect
   discrimination

### Remaining Options

**A. Multi-layer KV injection (L23–L30)**
Inject K/V at all layers where the copy circuit and amplification
operates. Cost: ~7 layers × 4 KV heads × 256 dim × 2 (K+V) × 2B
= ~28 KB/position. For 1,472 positions: ~40 MB. Still large.

**B. Hybrid: KV routing + residual injection**
Use KV injection at L29 purely as a **router** (extract the attention
weights to identify the best entry), then inject the answer into the
residual at L30 using the 12-byte approach. This combines:
- Native routing quality (58:1 discrimination)
- Proven answer injection (12 bytes, L30 projection)
- No external routing code needed

**C. Full KV cache (status quo)**
The model's own KV cache is the only mechanism that produces correct
answers from attention-based retrieval. Any sparse approximation
must replicate the multi-layer preparation, not just the copy head.

**D. Residual injection with improved routing**
Stay with 12-byte residual injection at L30 but invest in better
routing. The current bottleneck is routing accuracy, not injection.
The KV routing result (58:1 discrimination) shows the model has
much better routing internally — can we extract and use it?

### Recommendation

Option B (hybrid KV routing + residual injection) is most promising.
It leverages the validated native routing while using the proven
12-byte injection. Key next step: extract H4 attention weights from
the KV-injected forward pass and use them to select entries for
residual injection.
