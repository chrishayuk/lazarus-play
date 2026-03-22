# Full Vocabulary Routing Results

**Experiment ID:** ab1d1ab9-77b8-4b76-a346-8ade2d2c9816
**Model:** google/gemma-3-4b-it
**Date:** 2026-03-21

## Hypothesis

Store ALL unique token IDs per 512-token window (~163 tokens).
The transcript's own vocabulary is the routing index — no keyword
extraction, no generation, no external NLP. TF-IDF weighted
overlap for routing. Expected: 5/5 with case normalisation.

## Critical Discovery: SentencePiece Token ID Instability

**Token IDs are context-dependent.** SentencePiece assigns
DIFFERENT IDs based on whether a word has a leading space:

| Word | Standalone | With space | In-context |
|------|-----------|-----------|------------|
| porridge | [2152, 24187] | [133909] | [133909] |
| scores | [32504] | [14900] | [14900] |
| baseball | [62372] | [21637] | [21637] |
| landed | [1591, 524] | [30473] | [30473] |

**Impact:** Raw token-ID overlap between query tokens and document
tokens is BROKEN by design for SentencePiece models. The same word
gets different IDs depending on position.

**Fix:** Decode each token to string, strip whitespace, compare
by normalised string. This works but adds a decode step.

## Experiment 1 — Vocabulary Coverage

| Metric | Value |
|--------|------:|
| Total tokens | 370,778 |
| Windows (512-tok) | 724 |
| Mean unique/window | 163 |
| Median | 161 |
| Min | 77 (W322) |
| Max | 280 (W76) |
| Dedup ratio | 3.15× |
| Total unique across all windows | 10,987 |
| Storage per window (uint16) | 326 bytes |

**163 unique tokens per window, not 200.** The 3.15× dedup ratio
means each token repeats ~3 times in a 512-token window.

### Coverage Table

| Token | In K-norm 8? | In full vocab? | Notes |
|-------|:---:|:---:|---|
| " porridge" (133909) | ✅ | ✅ in W170 | Single token with space |
| " scores" (14900) | ❌ | ❌ NOWHERE | Word "scores" never appears in transcript |
| " baseball" (21637) | ✅ | ✅ in W169 | Also in W590, W614, W671 |
| " moon" (16254) | ❌ | ❌ in W370 | Transcript uses EAGLE/CONTACT, not "moon" |
| " landed" (30473) | ❌ | ❌ in W370 | "LANDED" → "LAND"+"ED", not "landed" |
| " Minneapolis" (42384) | ✅ | ✅ in W169 | |
| " Thor" (29255) | ✅ | ✅ in W169 | |

## Experiment 2 — TF-IDF Routing at N=50

Decoded-string matching (the SentencePiece fix). 53 sampled windows.

| Query | Case-sensitive | Case-insensitive | Correct? |
|-------|:---:|:---:|:---:|
| Porridge | #1 | #1 | ✅ |
| Baseball | #1 | #1 | ✅ |
| Landing | #43 | #44 | ❌ |
| Weather | #1 | #1 | ✅ |
| News | #1 | #1 | ✅ |
| **Total** | **4/5** | **4/5** | |

N=50 works for 4/5 regardless of case sensitivity.
Landing has ZERO content overlap with W370.

## Experiment 3 — Landing Diagnosis

**Query tokens (CI):** did, landed, moon, on, say, the, they, what, when
**W370 overlap:** on, the (structural only)

**W370's actual vocabulary:** EAGLE, CONTACT, LIGHT, Tranquility,
Base, Houston, ENGINE, STOP, DETENT, ACA, COMMAND, Roger, copy,
breathing, bunch, blue

**The vocabulary gap is total.** The query asks about "landing on
the moon" — W370 describes it as "CONTACT LIGHT / ENGINE STOP /
THE EAGLE HAS LANDED / Tranquility Base here." Zero shared
content words.

### BPE Split Issue

| Transcript word | BPE tokens | Lowercase | Matches query? |
|----------------|-----------|-----------|:-:|
| LANDED | LAND + ED | land, ed | ❌ ("landed" ≠ "land") |
| EAGLE | EAG + LE | eag, le | ❌ |
| CONTACT | CONTACT | contact | ❌ (not in query) |

### Alternative Queries

| Query | W370 rank | Why |
|-------|:---------:|-----|
| "Houston, Tranquility Base here" | **#1** | Exact transcript words |
| "When did the Eagle land?" | #8 | "eagle" + "land" stem |
| "CONTACT LIGHT - what happened?" | #6 | Distinctive W370 token |
| "What did they say when they landed on the moon?" | #616 | Zero content overlap |

**Conclusion:** The gap is concept-to-vocabulary, not case or
coverage. Only queries using the transcript's own words can route.

## Experiment 4 — Full Scale N=724

| Query | Case-sensitive | Case-insensitive | Target |
|-------|:---:|:---:|---|
| Porridge | **#1** | **#1** | W170 |
| Baseball | #3 | #4 | W169 |
| Landing | #558 | #616 | W370 |
| Weather | #7 | **#1** | W169 |
| News | **#1** | **#1** | W169 |
| **Total** | **2/5** | **3/5** | |

### Scale Degradation

Baseball drops from #1 (N=50) to #3-4 (N=724) because "baseball"
appears in 4 different news broadcasts (W169, W590, W614, W671).
More windows = more competition from similar content.

Weather drops from #1 to #7 (case-sensitive) because "weather"
is a common word. Case-insensitive fixes it because "Minneapolis"
is distinctive when case is preserved.

### Baseball Diagnosis

The word **"scores" does not appear anywhere in the 370K-token
transcript.** The news segment lists individual game results
("Philadelphia 3; Montreal 5") but never uses the word "scores."
Full vocabulary cannot route to a word that doesn't exist.

## Experiment 5 — Case Normalisation

| Method | Porridge | Baseball | Landing | Weather | News | Total |
|--------|:---:|:---:|:---:|:---:|:---:|:---:|
| CI string match (N=724) | #1 | #4 | #616 | #1 | #1 | 3/5 |
| Stem matching (N=724) | #1 | #5 | #166 | #1 | #2 | 2/5 |

**Case normalisation fixes weather** (Minneapolis case sensitivity).
**Case normalisation does nothing for landing** — the gap is
word-level, not case-level.

**Stem matching helps landing** (616→166 via "land" prefix) but
**degrades baseball** (4→5) and **news** (1→2) from stem expansion
noise. Net negative.

## Experiment 6 — Store Format

### Actual Measurements

| Field | Size | Purpose |
|-------|-----:|---------|
| Routing strings (~163 unique) | ~326 B | TF-IDF string matching |
| Injection entries (8 × 8 B) | 64 B | Persistent injection |
| **Per window total** | **~390 B** | |

### Storage Budget

| Document | Windows | Store | vs Keyword | vs KV cache |
|----------|---------|-------|-----------|-------------|
| Apollo 11 | 724 | **319 KB** | 12.8× larger | 175,000× smaller |
| 1M tokens | 1,953 | 762 KB | | |
| 10M tokens | 19,531 | 7.6 MB | | |

### Comparison

| Method | Size | Routing accuracy | Extraction cost |
|--------|------|:---:|---|
| Keyword index | 25 KB | **5/5** | ~3 tok/fact generation |
| Full vocabulary | 319 KB | 3/5 | Zero (pure tokenisation) |
| KV cache | 56 GB | N/A | N/A |

## What We Learned

### The hypothesis was wrong

Full vocabulary routing achieves **3/5, not 5/5.** The two failures:

1. **Landing (vocabulary gap):** The query and transcript use
   different words for the same event. "landed on the moon" vs
   "CONTACT LIGHT / THE EAGLE HAS LANDED / Tranquility Base."
   No amount of vocabulary expansion fixes this — the words are
   simply different.

2. **Baseball (scale dilution + missing word):** "scores" never
   appears in the transcript. "baseball" appears in 4 different
   windows. At N=724, other sports windows outscore the target.

### SentencePiece is a real problem

Token-ID matching is fundamentally broken for SentencePiece
tokenizers. The same word gets different IDs depending on space
context. Any token-ID routing scheme must decode to strings first.
This eliminates the "zero-cost pure tokenisation" advantage — you
need a decode + normalise step.

### Keywords bridge the vocabulary gap; tokens cannot

Keywords work because they extract *concepts* ("moon landing",
"baseball scores") that bridge the gap between query vocabulary
and document vocabulary. Token sets store *strings* that can only
match identical strings.

The vocabulary gap is the fundamental limit of token-based routing.
It's not fixable by:
- More tokens (full vs 8 samples) — same 3/5
- Case normalisation — fixes weather only
- Stem matching — net negative (adds noise)
- BPE reconstruction — helps "LANDED"→"land" but not "moon"

### The right architecture remains keyword-based

| Component | Method | Rationale |
|-----------|--------|-----------|
| Routing | Keyword index (800 B) | Bridges vocabulary gap |
| Injection | Persistent 12-byte at L30 | Model-native |
| Store | 25 KB total | Minimal |

Full vocabulary is a larger, less accurate version of keywords.
The extra 294 KB per document buys zero additional correct routings.

### One genuine contribution

The SentencePiece finding is architecturally important: **any system
that stores token IDs for later matching against query tokens will
fail silently** unless it decodes to strings. This affects K-norm
sampling, attention-score routing, and any future token-ID scheme.
The fix is trivial (decode + strip) but the failure mode is invisible
(wrong IDs, zero matches, no error).
