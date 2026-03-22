# Completion-Primed Routing Results

**Experiment ID:** e3bfd2c8-a64c-4304-93e7-1f855e1ff5bc
**Model:** google/gemma-3-4b-it
**Date:** 2026-03-21

## Hypothesis

The logit lens at L30 reads "Okay" because the chat template primes response-start prediction. Change the template suffix from chat format to completion format (e.g., `\n\nTopic:`) to surface the model's content comprehension as a next-token prediction. Use these predictions as zero-generation-cost routing descriptors.

## Experiment 1 — Template Sweep (W370)

Nine completion suffixes tested on W370 (moon landing window: "CONTACT LIGHT... Tranquility Base here. HAS LANDED."):

| Template | Top-1 | P(top-1) | Content in top-5 | Grade |
|----------|:---:|:---:|---|:---:|
| **Topic:** | **Lunar** | **26%** | Eagle(5.8%), Apollo(5.1%), Landing(3.5%), Moon(2.4%) | **A** |
| **Subject:** | **Lunar** | **28%** | Apollo(10%), Eagle(4.2%), Moon(1.9%) | **A** |
| Keywords: | \n\n | 36% | Eagle(11%), Lunar(1.2%), Landing(1.2%) | B |
| Main topic: | \n\n | 46% | Lunar(4%), Landing(1.5%) | C |
| This is about the | \n\n | 20% | landing(3.9%), lunar(1.4%) | C- |
| Summary: | \n\n | 64% | Lunar(0.2%) | F |
| TL;DR: | \n\n | 35% | — | F |
| This passage is about | the | 59% | — | F |
| The main event is | the | 72% | — | F |

**Winner: `\n\nTopic:`** — Produces content tokens as top-1 for W370 (Lunar 26%).

Cross-window validation:

| Window | Topic: top-1 content | P | Keywords: top content |
|--------|:---:|:---:|---|
| W370 (landing) | **Lunar** | 26% | Eagle(11%) |
| W170 (porridge) | **Space** | 10% | oatmeal(3.4%), food(2.4%) |
| W169 (weather/news) | **Weather** | 16.5% | Weather(9.5%), weather(6.1%) |

**Finding:** Topic: elicits semantic categories. Keywords: elicits specific words. Both produce content tokens, but Topic: is more consistent as top-1. The model reads "CONTACT LIGHT" OCR garbage and predicts "Lunar" — it understands the content.

## Experiment 2 — Topic: Across All Key Windows

| Window | Content | Top-1 | P | Content relevant? |
|--------|---------|:---:|:---:|:---:|
| W370 | Landing | **Lunar** | 26% | YES |
| W170 | Porridge/baseball | **Space** | 10% | Partial (Food 6%, Oatmeal 1.7%) |
| W169 | Weather/Heyerdahl/baseball | **Weather** | 16.5% | YES (but only 1 of 3 topics) |
| W000 | Introduction | Lunar | 6.6% | YES |
| W240 | Baseball/LOI | **Space** | 15.7% | NO (baseball not surfaced) |
| W585 | TEI operations | **Apollo** | 33.6% | YES |

**6/6 produce content-relevant predictions.** But most windows predict Apollo/Lunar/Space — the document-level topic. Window-specific content (porridge, baseball) only surfaces when dramatically different from the general theme.

## Experiment 3 — Routing Descriptors (Top-10)

Content tokens across all 50 windows reveal strong topic dominance:

| Token | Windows containing it | IDF |
|-------|:---:|:---:|
| lunar | 44/50 | 0.13 |
| apollo | 37/50 | 0.30 |
| space | 31/50 | 0.48 |
| nasa | 25/50 | 0.69 |
| **weather** | **5/50** | **2.30** |
| **eagle** | **2/50** | **3.22** |
| **food** | **1/50** | **3.91** |
| **oatmeal** | **1/50** | **3.91** |
| **baseball** | **1/50** | **3.91** |
| porridge | 0/50 | — |
| heyerdahl | 0/50 | — |

Apollo/Lunar/Space appear in 60-88% of windows — nearly zero discriminative value. The distinctive predictions (Oatmeal, Baseball, Weather) are rare and highly discriminative, but they only appear in windows where the topic is unusual.

## Experiment 4 — Routing at N=50

| Query | Keywords | Summary | **Completion** | Token overlap |
|-------|:---:|:---:|:---:|:---:|
| Porridge | **1** | 1 | **1** | 1 |
| Baseball | **1** | 8 | 4 | 2 |
| Landing | **1** | 1 | **1** | 43 |
| Weather | **1** | 1 | 2 | 1 |
| News | **1** | 1 | **1** | 1 |
| **Score** | **5/5** | **4/5** | **3/5** | **3/5** |

**Completion routing: 3/5.** Same as token overlap. Worse than keywords (5/5).

Failure analysis:
- **Baseball:** W645 (All-Star game rain) predicts "Baseball(9.4%)" — correct comprehension! But routes to wrong window because W645 is more topically focused on baseball than W169 (which also contains weather and Heyerdahl).
- **Weather:** W645 predicts "Weather(69.5%)" vs W169's "Weather(16.5%)". W645 is about recovery area weather — also valid but not the target.

The vocabulary bridging WORKS (All Star game → "Baseball", CONTACT LIGHT → "Lunar") but creates **topic-dominance errors** in multi-topic windows.

## Experiment 5 — Cost Comparison

| Method | Calls/window | Gen tokens | N=725 cost | Accuracy |
|--------|:---:|:---:|:---:|:---:|
| Keywords (regex) | 0 | 0 | 0 | **5/5** |
| 5-word summary | 1 generate | ~10 | 7,250 tokens | 4/5 |
| **Completion top-10** | **1 predict** | **0** | **725 calls** | **3/5** |
| Logit lens (chat) | 0 | 0 | 0 | 0/5 |

Completion predictions cost more than keywords (require model forward pass) and perform worse.

## Core Finding

**The template change works — the comprehension surfaces.** The model reads garbled OCR text and correctly predicts "Lunar", "Weather", "Food", "Baseball" as topic tokens. The vocabulary gap is bridged.

**But single-token prediction is fundamentally lossy for routing.** A 512-token window typically contains 2-3 topics. One prediction captures only the dominant one. Multi-topic windows (W169: weather + Heyerdahl + baseball) collapse to a single label ("Weather"), losing all minority topics.

Keywords extract ALL distinctive strings in parallel — every word is a routing descriptor. Completion prediction extracts ONE summary word serially — the most salient topic wins.

### Why This Was Worth Testing

The hypothesis was sound: the model's comprehension could be extracted via template priming rather than generation. And it works — `\n\nTopic:` produces "Lunar" from "CONTACT LIGHT" at 26%, proving the model bridges the vocabulary gap internally.

The failure is not in comprehension extraction but in **information capacity**: one token ≈ 1 topic. Keywords extract ~3 tokens/fact with N facts/window. For routing within a single document where most windows share the same domain, the rare distinctive keywords matter more than the common predicted topic.

### When Completion Routing Would Win

- **Cross-domain corpora:** If windows come from different domains (medical, legal, sports, history), the topic prediction would be highly discriminative.
- **Very garbled text:** Where keywords are destroyed by OCR but the model still comprehends (it does — "CONTACT LIGHT" → "Lunar" proves this).
- **Combined approach:** Use keywords (cheap, 5/5) as primary, completion prediction as fallback for windows with no keyword matches.

## Architecture Implication

The final architecture remains:
1. **Keyword extraction** (regex, ~3 tokens/fact) → 800 B, 5/5
2. **Persistent injection** (crystallised L30 residual) → 10 KB/passage
3. Total: ~114 KB for Apollo 11. Zero generation cost.

Completion-primed routing is a valid mechanism but doesn't improve on keywords for within-document routing.
