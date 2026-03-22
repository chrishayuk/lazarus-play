# Token Frequency Copy Circuit — Ablation Matrix (L29)
# Model: google/gemma-3-4b-it
# Multi-entity template (with ** competition)

## Template
```
<start_of_turn>user
Several kingdoms have sacred materials. Pyrethis values silver. Korvathi holds stone sacred. Zenthari reveres obsidian. Volthari prizes gold. Draventh considers water holy. What does [Entity] [verb]?
<end_of_turn>
<start_of_turn>model
[Entity] [verb]
```

## Individual Head Ablation — Obsidian (very rare, ID 226956)
Baseline P(obsidian) = 97.7%

| Head | P(obsidian) | ΔP | KL | target_delta |
|------|------------|-----|------|-------------|
| H0 | 96.9% | -0.8pp | 0.002 | -1.0 |
| H1 | 98.4% | +0.7pp | 0.0 | +0.25 |
| H2 | 97.7% | 0.0pp | 0.0 | -0.5 |
| H3 | 92.6% | -5.1pp | 0.024 | -1.0 |
| **H4** | **73.0%** | **-24.7pp** | **0.227** | +4.75 |
| H5 | 97.7% | 0.0pp | 0.0 | -0.75 |
| H6 | 96.9% | -0.8pp | 0.002 | -0.5 |
| H7 | 96.9% | -0.8pp | 0.002 | -2.0 |
| **ALL attn** | **37.7%** | **-60.0pp** | **0.856** | -9.75 |

Sum of individual ΔPs: -33.0pp
Full attention ΔP: -60.0pp
Non-linear cooperation: -27.0pp (45% of total effect)

## Individual Head Ablation — Gold (common, ID 5122)
Baseline P(gold) = 50.0%

| Head | P(gold) | ΔP | KL |
|------|---------|-----|------|
| **H4** | **62.1%** | **+12.1pp** | 0.033 |
| **H5** | **4.8%** | **-45.2pp** | **0.848** |
| **ALL attn** | **18.3%** | **-31.7pp** | 0.256 |

Note: Zeroing H4 HELPS gold — H4 is actively pushing toward obsidian/other rare tokens.
H5 is the sole causal copy head for common tokens.

## Cross-Frequency Causal Summary

| Token | Band | Token ID | H4 ΔP | H5 ΔP | All attn ΔP |
|-------|------|----------|--------|--------|-------------|
| obsidian | 1 (very rare) | 226956 | -24.7pp | 0.0pp | -60.0pp |
| gold | 5 (v.common) | 5122 | +12.1pp | -45.2pp | -31.7pp |

## Key Findings
1. H4 is causally critical for rare tokens (-24.7pp), irrelevant/harmful for common tokens (+12.1pp)
2. H5 is causally critical for common tokens (-45.2pp), irrelevant for rare tokens (0.0pp)
3. Non-linear cooperation: individual ablations sum to -33pp but collective is -60pp
4. The copy circuit has massive redundancy — cooperative heads invisible to individual ablation
5. H3 shows mild contribution to obsidian (-5.1pp) — partial backup role for rare tokens
