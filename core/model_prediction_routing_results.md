# Model Prediction Routing Results

**Experiment:** e800db17
**Model:** google/gemma-3-4b-it
**Date:** 2025-03-21

## Hypothesis

The model's own next-token predictions bridge the vocabulary gap between query language and transcript language. After reading "CONTACT LIGHT. ENGINE STOP. THE EAGLE HAS LANDED," the model should predict tokens like "moon", "landing", "Apollo" — which match the query "What did they say when they landed on the moon?"

## Experiment 1 — What Does the Model Predict?

### Single next-token prediction: FAILS

| Method | W370 top prediction | Useful for routing? |
|--------|:---:|:---:|
| Raw text (no chat template) | "\n\n" (95%) | No — structural token |
| Chat template | "Okay" (100%) | No — assistant acknowledgment |

The model's single next-token prediction is structural (formatting) or social (acknowledgment), never semantic. predict_next_token is fundamentally wrong for this task.

### Short generation with topic prompt: WORKS

Prompt: "What is this transcript about? Answer in exactly 5 words."

| Window | 5-word summary | Bridging tokens |
|--------|---------------|-----------------|
| W370 (landing) | "Apollo 11 moon landing." | **moon**, landing, Apollo |
| W170 (porridge) | "Sports scores and porridge win." | Sports, scores, porridge |
| W169 (weather) | "Weather, travel, and sports." | Weather, travel, sports |

**Critical:** "moon" (token 16254) appears in BOTH W370's summary and the landing query. Exact token match.

### Prompt type matters enormously

| Prompt | W370 output | Routing value |
|--------|------------|:---:|
| "Extract keywords" | CDR, LMP, CONTACT, LIGHT, ENGINE, STOP | Zero — literal extraction |
| "What is this about? 5 words" | Apollo 11 moon landing | **Perfect** — semantic bridging |

Keyword extraction parrots. Topic summarization comprehends. The model bridges the vocabulary gap through COMPREHENSION, not prediction.

## Experiment 2 — Combined Routing at N=50

50 windows. Each gets: content words (case-insensitive from text) + summary words (from 5-word generation). TF-IDF weighted overlap.

### All 50 summaries

| Window | Summary | Key additions |
|--------|---------|--------------|
| W000 | Apollo 11 lunar communication transcript | transcript |
| W015 | Apollo 11 communication test | test |
| W030 | Apollo 11 lunar landing | (generic — not the landing window) |
| W060 | Aircraft monitoring and system checks | monitoring, checks |
| W120 | Storms are above the area | storms |
| W169 | Weather, travel, and sports | **weather, travel** |
| W170 | Sports scores and porridge win | **sports, scores, win** |
| W240 | Baseball game, playoff situation | **baseball, playoff** |
| W270 | Observing a lunar crater | crater |
| **W370** | **Apollo 11 moon landing** | **moon** (UNIQUE) |
| W375 | Lunar mission status report | (generic lunar) |
| W615 | Sports news and events | sports, news, events |
| W645 | Rain caused All Star game | rain, caused |

### Key discrimination

- **"moon"** appears in W370 summary ONLY (df=1). IDF=3.91.
- **"lunar"** appears in 11 summaries (df=11). IDF=1.51. Terrible discriminator.
- The model chose "moon" for the actual landing, "lunar" for routine mission windows.

### Routing results

| Query | Content-only | + Summaries | Target |
|-------|:---:|:---:|:---:|
| Porridge | ✅ #1 | ✅ #1 | W170 |
| Baseball | ❌ #2 | ❌ #8 | W169 |
| **Landing** | **❌ #7** | **✅ #1** | **W370** |
| Weather | ✅ #1 | ✅ #1 | W169 |
| News | ✅ #1 | ✅ #1 | W169 |

**4/5.** Landing fixed (was #7 → #1, score 6.73 from "moon"+2.81 + "landed"+3.91). Baseball worsened (#2 → #8) because "baseball" IDF diluted from 3.91 to 2.81 (now in 3 windows via summaries).

## Experiment 3 — Landing Deep Dive

### Only W370 produces "moon"

| Window | Summary | Contains "moon"? |
|--------|---------|:---:|
| W345 | Flight data processing report | No |
| W360 | Spacecraft communication, signal loss | No |
| **W370** | **Apollo 11 moon landing** | **Yes** |
| W375 | Lunar mission status report | No |

The model distinguishes "about the moon landing" from "related to a lunar mission" at the comprehension level.

### 20-keyword extraction also works

When asked for 20 keywords from W370 (cleaned transcript), the model lists literal tokens (CDR, EAGLE, CONTACT LIGHT...) BUT in the appended summary paragraph mentions: "Apollo 11 lunar module, 'Eagle,' during its historic landing on the **Moon**."

## Experiment 4 — N=725 Prediction

Only 50 sampled windows available. Cannot run.

**Prediction:** "moon" IDF at N=725 = log(725/1) = 6.59 (even more discriminative). Landing should route at rank 1 with large margin. Baseball uncertain — 4+ windows would each get "baseball" in summaries.

## Experiment 5 — Store Format

### Per-window entry: 84 bytes

| Field | Size | Purpose |
|-------|-----:|---------|
| Content token IDs (8) | 32 B | K-norm sampled, for injection |
| Content coefficients (8) | 32 B | Injection magnitudes |
| Summary word tokens (5) | 20 B | Routing descriptors from model generation |
| **Total** | **84 B** | |

### Storage budget

| Document | Windows | Store size |
|----------|---------|-----------|
| Apollo 11 | 725 | **59.5 KB** |
| 1M tokens | 1,953 | 160 KB |
| 10M tokens | 19,531 | 1.56 MB |

### Generation cost

~10 tokens generation + 40 token prompt per window = 50 tokens/window.
725 windows = 36,250 tokens total. One-time cost at store build time.

### vs. keyword index

| Approach | Size | External NLP? | Accuracy (N=50) |
|----------|-----:|:---:|:---:|
| Keyword + inject | 25 KB | Yes (regex) | 5/5 |
| Prediction routing | 60 KB | **No** | 4/5 |

Prediction routing is 2.4× larger and less accurate, but completely model-native.

## Experiment 6 — End-to-End Injection

### Landing (parametric topic)

| Method | Output | First token |
|--------|--------|:---:|
| Baseline | "one small step for mankind" | The (97%) |
| Full residual L30 | "According to... Eagle... one small step" | According (68%) |
| Gold (with context) | "CONTACT LIGHT... ENGINE STOP..." | Here (99%) |

**Parametric override.** "Apollo 11 moon landing" has such strong parametric knowledge that injection at L30 can't override it. Full residual shifts style (mentions "Eagle") but not content. Known constraint from prior experiments.

### Porridge (novel fact)

| Method | Output | First token |
|--------|--------|:---:|
| Baseline | "trick question! won by a bowl of porridge" | This (99%) |
| **Full residual L30** | **"John McGregor won the World Porridge Championships"** | **John (99%)** |
| Gold (with context) | "John Coyle won the porridge eating championship" | John (92%) |

**Novel fact injection works.** First token "John" at 99% (KL=0.09 from donor). Multi-token drift: "McGregor" instead of "Coyle" — the persistent_transplant problem (solved: inject every generation step → 100%).

## Architecture Summary

```
BUILD TIME (one-time per document):
  For each 512-token window:
    1. Prefill → K-norm sample 8 content tokens + coefficients (64 B)
    2. Generate 5-word topic summary → 5 summary tokens (20 B)
    3. Store: 84 bytes/window

QUERY TIME:
  1. Tokenize query
  2. TF-IDF overlap: query tokens vs (content + summary) tokens
     → Best window (pure set intersection, no model computation)
  3. Persistent inject crystallised L30 residual from best window
     → Generate answer
```

## What This Means

### The model bridges its own vocabulary gap — through generation, not prediction

Single next-token prediction is structural. But when asked "what is this about?", the model generates semantic descriptors that match query language. The transcript says "CONTACT LIGHT. ENGINE STOP." The model says "moon landing." The query says "landed on the moon." Token overlap succeeds.

### Summary choice is highly discriminative

"moon" appears in only W370's summary. "lunar" appears in 11 others. The model knows the difference between THE moon landing and routine lunar mission windows. This discrimination is more precise than keyword extraction.

### The tradeoff

| | Keywords | Model predictions |
|---|---|---|
| Accuracy (N=50) | 5/5 | 4/5 |
| External NLP | Yes (regex) | No |
| Store size | 25 KB | 60 KB |
| Build cost | Zero | 36K tokens generation |
| Mechanism | Concepts → strings | Model comprehension → strings |

Keywords are simpler, smaller, and more accurate. Model predictions are fully model-native but slightly worse and more expensive.

### When to use which

- **Novel entities** (Zarkov, Voltara): Token overlap IS the routing. No keywords needed. No summaries needed. The entity name is the routing key.
- **Known topics, natural language**: Keywords work. Predictions add bridging but dilute rare terms.
- **Transcript/technical → natural language**: Predictions bridge vocabulary gaps that keywords can't (CONTACT LIGHT → moon landing). But the parametric override makes injection impossible for well-known topics anyway.

### The irony

Prediction routing's strongest advantage (bridging vocabulary gaps for well-known topics) is exactly where injection fails (parametric override). For novel facts, where injection works perfectly, token overlap already works without predictions.

**The keyword index remains the cleanest architecture.** 800 bytes. 5/5 routing. No generation cost. Predictions are an interesting mechanism but don't improve the practical architecture.
