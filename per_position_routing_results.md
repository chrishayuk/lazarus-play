# Per-Position Contextual Routing Results

**Experiment:** 524e5b2e | **Model:** google/gemma-3-4b-it | **Layer:** 29

## Hypothesis

Interior positions encode **content state** ("the fact here") while last positions encode **navigational state** ("finished reading"). The format gap between query and document residuals might not exist at content positions. Comparing the query's L29 residual against per-position residuals (K-norm sampled) rather than window summaries (last position) might enable model-native routing.

## Result: NEGATIVE

The content-state hypothesis is **wrong**. The format gap exists at ALL positions, not just window boundaries. Per-position routing does not solve the routing problem.

## Experiment 1 — Per-Position vs Last-Position Cosine (3 windows)

| Query | Window | Best position cos | Mean position cos | Last-position cos | Target? |
|-------|--------|:-:|:-:|:-:|:-:|
| Landing | W370 | 0.9283 | 0.9229 | **0.9315** | ← |
| Landing | W170 | 0.9295 | 0.9237 | **0.9348** | |
| Landing | W169 | 0.9236 | 0.9192 | **0.9354** | |
| Porridge | W170 | **0.9545** | 0.9491 | 0.9541 | ← |
| Weather | W169 | 0.9447 | 0.9407 | **0.9556** | ← |

**Finding:** Last-position cosines are systematically HIGHER than per-position cosines. The navigational state has more shared structure across query-window pairs. Interior positions are noisier, not more content-specific.

Critical observation: W169 has the highest last-position cosine for the *landing* query (0.9354 > W370's 0.9315). The wrong window wins even at last-position level. The problem isn't navigational vs content — it's that raw cosine in 2560D doesn't discriminate content.

## Experiment 2 — Novel Entity Routing (N=3)

| Query | Per-pos winner | Correct? | Margin | Last-pos winner | Correct? |
|-------|:-:|:-:|:-:|:-:|:-:|
| Zarkov city | kelvara | ✗ | 1.002× | kelvara | ✗ |
| Strand city | kelvara | ✗ | 1.002× | kelvara | ✗ |
| Kelvara town | kelvara | ✓ | 1.003× | kelvara | ✓ |

**Finding:** Kelvara passage wins ALL queries — both per-position and last-position. This is structural bias (longest passage = highest baseline cosine), not content matching. Margins are negligible (0.2-0.3%).

## Experiment 3 — Apollo Scale (N=50)

| Query | Last-pos rank | Per-pos MAX rank | Per-pos MEAN rank | Improved? |
|-------|:-:|:-:|:-:|:-:|
| Porridge | 14 | **6** | 8 | ✓ |
| Baseball | **17** | 33 | 40 | ✗ |
| Landing | 45 | **26** | **15** | ✓ |
| Weather | **11** | 47 | 38 | ✗ |
| News | **14** | 45 | 40 | ✗ |

**Finding:** Mixed. Landing improved from rank 45→26 (MAX) or 45→15 (MEAN), but still nowhere near top-3. Three queries got substantially WORSE. The same windows appear in top-5 for almost all queries (W465, W195, W285, W420) — structural dominance, not content matching.

Cosine spreads are tiny: the entire range across 50 windows is ~0.003 for per-position MAX. This is noise-level discrimination.

## Experiment 4 — What Position Residuals Encode

L29 position residuals decoded through the unembedding matrix:

| Position | Token | Top predictions |
|----------|-------|----------------|
| 85 | 'Ll' | ……., hain, Lok, അഭി, ovne |
| 216 | 'MODE' | Enquiry, advantage, 说了, setInterval, jim |
| 263 | 'h' | decimal, symbol, FOX, 累计, اولة |
| LAST | (end) | 缛, ⅙, Spare, BOAT, మై |

**Finding:** ALL positions — interior and last — decode to random multilingual garbage at L29. Both are deep in dark space. The residuals encode something orthogonal to vocabulary. The hypothesis that interior positions encode "the fact here" in a query-comparable way is wrong. They encode dark-space computation state, not vocabulary-adjacent content.

## Experiment 5 — Centred Cosine

| Query | Raw per-pos rank | Centred rank | Last-pos rank |
|-------|:-:|:-:|:-:|
| Porridge | 6 | 21 | 14 |
| Baseball | 33 | 19 | 17 |
| Landing | 26 | **20** | 45 |
| Weather | 47 | 17 | 11 |
| News | 45 | 22 | 14 |

**Finding:** Centring removes shared structure but introduces new bias. W540 and W360 dominate all queries — probably high-variance outlier windows. Landing's best rank across all methods is 20/50. Not usable.

## Experiment 6 — Full Scale (SKIPPED)

Best landing rank at N=50 was 20 (centred) or 26 (raw). Neither approaches top-3. Scaling to N=724 would only worsen discrimination. Not warranted.

## Why Per-Position Routing Fails

1. **L29 residuals are in dark space.** Both query and document residuals at L29 are orthogonal to vocabulary. Logit lens produces garbage. There's no interpretable "content signal" to match on.

2. **Raw cosine is dominated by shared structure.** All L29 residuals share ~92-95% cosine similarity. The discriminative signal is in the remaining 5-8%, which is below the noise floor of structural variation between windows.

3. **The full KV cache works differently.** Q·K in attention operates in a 256D projected subspace with RoPE positional encoding. This is NOT the same as cosine in the full 2560D residual space. The attention mechanism projects out the dark-space structure and matches in a content-specific subspace. Raw residual cosine doesn't do this.

4. **Structural bias dominates content signal.** Certain windows (W465, W345, W540) systematically have high cosine to all queries. This is window-level variance (text length, token distribution), not content matching.

## Implications

- The format gap is **not a boundary artifact** — it exists at all positions
- Dark-space residuals cannot be compared with raw cosine for routing
- The attention mechanism's Q·K projection is essential — it creates a content-discriminative subspace that doesn't exist in the full residual
- **Keywords remain optimal:** 800 bytes, 5/5 correct, zero generation cost, vocabulary-space matching that works because it operates in the space where content IS discriminative

## Architecture Confirmed

```
keyword index → persistent inject
```

Per-position residuals add 29 MB of model-native representations that don't discriminate content. Keywords add 800 bytes of vocabulary-space tokens that do.
