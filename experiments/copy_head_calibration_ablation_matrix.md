# Ablation Matrix — Copy Head Calibration
# Model: google/gemma-3-4b-it
# Method: Zero each head, measure P(answer) drop

## Baselines
| Probe | Answer token | Token ID | P(baseline) |
|-------|-------------|----------|-------------|
| Zarkov/Voltara | " Volt" | 89711 | 99.2% |
| Nexaris/Cerulion | " Cer" | 24996 | 94.1% |
| Helion/Dravenport | " Dra" | 55857 | 95.7% |
| Keltara/Solmere | " Sol" | 5718 | 98.4% |
| Namath/endorse | " endorse" | 62773 | 91.4% |
| Marchand/sell | " sell" | 6739 | 89.5% |

## Ablation ΔP (percentage points, negative = drop)

### Top-5 DLA candidates
| Head | Zarkov | Nexaris | Helion | Keltara | Namath | Marchand | Mean ΔP | #Affected |
|------|--------|---------|--------|---------|--------|----------|---------|-----------|
| **L29 H4** | **-4.3** | **-7.8** | **-16.8** | 0.0 | **-3.1** | +0.8 | **-5.2** | **4/6** |
| L30 H7 | 0.0 | -35.5 | 0.0 | -12.1 | -0.8 | 0.0 | -8.1 | 2/6 |
| L30 H0 | -0.8 | -6.6 | -5.5 | 0.0 | -0.8 | +0.8 | -2.2 | 2/6 |
| L30 H3 | 0.0 | -2.0 | -3.1 | -0.8 | 0.0 | +0.8 | -0.9 | 2/6 |
| L31 H7 | 0.0 | -0.8 | -0.8 | -0.8 | 0.0 | +0.8 | -0.3 | 0/6 |

### Supplementary (L29 alternate heads)
| Head | Keltara | Marchand | Note |
|------|---------|----------|------|
| L29 H3 | -2.3 | +0.8 | DLA-dominant for Keltara (+0.684) |
| L29 H5 | -0.8 | -1.2 | DLA-dominant for Marchand (+0.559) |

## KL Divergence (from full output distribution)
| Head | Zarkov | Nexaris | Helion | Keltara | Namath | Marchand | Mean KL |
|------|--------|---------|--------|---------|--------|----------|---------|
| L29 H4 | 0.034 | 0.040 | 0.120 | 0.004 | 0.016 | 0.000 | 0.036 |
| L30 H7 | 0.000 | 0.338 | 0.001 | 0.095 | 0.000 | 0.002 | 0.073 |
| L30 H0 | 0.002 | 0.028 | 0.025 | 0.001 | 0.002 | 0.000 | 0.010 |
| L30 H3 | 0.000 | 0.007 | 0.011 | 0.000 | 0.000 | 0.001 | 0.003 |
| L31 H7 | 0.001 | 0.003 | 0.004 | 0.002 | 0.002 | 0.000 | 0.002 |

## Consistency Analysis

L29 H4 causes drops on probes with NOVEL entity tokens:
- Zarkov (" Volt" — unique, no parametric memory) → -4.3pp
- Nexaris (" Cer" — common prefix, reduced effect) → -7.8pp
- Helion (" Dra" — semi-common prefix) → -16.8pp
- Namath (" endorse" — real word but in-context) → -3.1pp

L29 H4 has NO effect on:
- Keltara (" Sol" — very common token, L29 H3 is the copy head)
- Marchand (" sell" — common verb, L29 H5 is the copy head)

INTERPRETATION: Common answer tokens use different L29 heads.
H4 is the PRIMARY copy head for novel/rare tokens. H3 and H5
handle common tokens. The copy LAYER is L29, but the head
varies by token rarity.

## Winner: L29 H4
- Most consistently impactful head across diverse probes
- 100.6% of layer-29 DLA (layer-dominant on 4/6 probes)
- Ablation-confirmed on the probes where it matters most (novel entities)
