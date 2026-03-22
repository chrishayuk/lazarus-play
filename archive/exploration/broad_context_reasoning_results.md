# Broad Context Reasoning: Can the Model Find Amusing Moments?

**Experiment ID:** e96e2860-f545-49e7-9b7d-d800ce51e2c8
**Model:** google/gemma-3-4b-it (34 layers, 2560 hidden dim, 8 attention heads)
**Archive:** apollo11_ctx_512 (725 windows × 512 tokens, 370,778 total tokens)

---

## Executive Summary (in progress)

Geometric routing (contrastive + compass) fails completely for subjective/tone queries
like "find amusing moments." The amusing windows have no geometric signature — humour
is a surface-text phenomenon. Keyword search finds them trivially. The 4B model CAN
identify genuine humour when given enough context (≥256 tokens/window), but OCR
artifacts in the transcript degrade quality

---

## Phase 1 — What's in the Router-Selected Windows?

### 1a: Contrastive-Selected Windows (top-5)

| Window | Content | Amusing? |
|--------|---------|----------|
| W4 | Launch phase GO calls, booster safe, orbital coast | No |
| W118 | PTC/gyro drift updates with Charlie, PIPA biases | No |
| W600 | PTC attitude, P52 torque, tracking data with FIDO Jay Green | No |
| W605 | Post-EVA seismic temp questions, dust cloud discussion, Kapton debris | No |
| W724 | Splashdown error coordinates (fragment, only 90 tokens) | No |

**Contrastive: 0/5 amusing.**

### 1a: Compass-Selected Windows (top-5)

| Window | Content | Amusing? |
|--------|---------|----------|
| W321 | Simplex comm configuration, time hack, T-EPHEM load | No |
| W480 | EVA sample packing, ladder climbing, "Adios amigo" | Slightly informal |
| W550 | Alignment torquing, radar circuit breakers, X or Y | No |
| W605 | (same as above) | No |
| W613 | Apollo 8 sphere-of-influence joke, Dave Reed burying head, press banter, news | **Yes** |

**Compass: 1/5 amusing (W613).**

### 1b: Genuinely Amusing Windows (keyword search + manual decode)

Searched all 725 windows for: Laughter, czar, kidding, funny, chuckle, happy birthday,
joke, music, song, hey, you guys, beautiful, far out, gee, cool, congratulat.
257 windows had keyword hits. Decoded 14 most promising. Found 11 genuinely amusing:

| Window | Rating | Content |
|--------|--------|---------|
| **W244** | 5/5 | **"The Czar is brushing his teeth, so I'm filling in for him."** Collins fills in for Armstrong. Houston plays along: "If you don't get in the way of the Czar..." |
| **W672** | 5/5 | **Baby named "Module"** — parents compromised from "Lunar Module McGhee." Crew laughs. Also All-Star game 9-3. |
| **W653** | 5/5 | **"Music Out of the Moon"** — crew plays scratchy album for Houston. "(Laughter)." "The czar likes it." "That's what we figured." |
| W37 | 4/5 | **Happy birthday from space** — Armstrong wishes Dr. Mueller happy birthday. "California is 200 years old... I don't think he is that old." |
| W613 | 4/5 | **Apollo 8 joke** — spacecraft "jumped" through lunar sphere of influence. Dave Reed burying his head. Press conference banter. |
| W634 | 4/5 | **Strange noises on downlink** — "You sure you don't have anybody else in there with you?" CDR: "Where do the White Team go during their off hours?" |
| W694 | 4/5 | **Sound of train / ergometer joke** — Collins teasing Neil about exercise. "(Laughter)" |
| W308 | 3/5 | "You guys wake up early" / "sawing them away" (snoring banter) |
| W706 | 3/5 | "I'm right in the middle of my orange juice." Collins has explanation but Buzz doesn't buy it. |
| W644 | 3/5 | Weather from space — crew reports clouds, Houston: "We can't quite see that far" from MOCR. |
| W156 | 3/5 | Team banter — "How's the old Green Team?" Black Team complaining. Ron Evans as CAP COMM. |

### 1c: Did Geometric Routing Find Them?

| Router | Top-5 | Amusing hits | Recall |
|--------|-------|-------------|--------|
| Contrastive | W4, W118, W600, W605, W724 | 0 | 0/11 (0%) |
| Compass | W321, W480, W550, W605, W613 | 1 (W613) | 1/11 (9%) |
| Combined (9 unique) | — | 1 | 1/11 (9%) |

**Verdict: Geometric routing FAILS for subjective/tone queries.** The amusing windows
have no distinguishing geometric signature at L14 or L26. Humour is encoded in surface
tokens (Laughter, Czar, funny) not in residual stream geometry. The router selects
windows with conversational structure but misses the actual amusing content.

---

## Phase 2 — Can the 4B Model Find Humour?

### 2a-b: Three-Condition Generation Test

Prompt: 7 windows × ~150 tokens each (~1200 total tokens). "Find 3 amusing moments."
Temperature 0, max 400 tokens.

**Oracle (known amusing windows: W244, W672, W653, W37, W613, W634, W694):**
- Quality: **3/5**
- Found W672 laughter + W694 ergometer as lighthearted (2 grounded)
- Misread W653 music exchange as OCR mishearing (1 hallucinated)
- Missed Czar joke entirely — truncated from excerpt at 150 tokens
- Verdict: Model can find humor but excerpt truncation loses punchlines

**Contrastive (router-selected: W118, W600, W605, W321, W480, W4, W550):**
- Quality: **1/5**
- No genuinely amusing content in any window
- Model fabricated humor from technical jargon ("PTC attitude and torque" called amusing)
- Repeated same excerpt for 2/3 picks
- Verdict: Complete failure — garbage in, garbage out

**Random baseline (W654, W114, W25, W281, W250, W100, W400):**
- Quality: **2.5/5**
- Lucky hit on W100: Collins' "cooking, and sweeping, and almost sewing, the usual housekeeping things" — genuinely funny
- W114 monitor distortion called amusing — forced
- Missed W654 Earth/Moon TV gaffe (actually funny, model skipped it)
- Verdict: Random sometimes beats the router by chance

### 2c: Comparison

| Routing | Windows | Quality | Grounded | Hallucinated |
|---------|---------|---------|----------|-------------|
| Oracle | Best 7 | 3/5 | 2/3 | 1/3 |
| Random | 7 random | 2.5/5 | 1/3 | 1/3 |
| Contrastive | Router top-7 | 1/5 | 0/3 | 3/3 |

**Oracle > Random >> Contrastive.** Routing quality IS the bottleneck, but oracle
is only marginally better than random at 150 tok/window because truncation obscures
the punchlines. More context per window should widen the oracle advantage.

---

## Phase 3 — Context Density (in progress)

Using the 3 best oracle windows (W244 Czar, W672 Module, W653 Music) at varying
tokens per window.

### 3a: Density Sweep

| Tok/window | Total tokens | OCR | Quality | Grounded | Notes |
|------------|-------------|-----|---------|----------|-------|
| 30 | 154 | raw | 1.5/5 | 1/3 | Model invents context it can't see. MOCR joke detected without punchline. |
| 100 | 366 | raw | 2.5/5 | 2/3 | Baseball incongruity found. Music partially understood. MOCR/MOCH confused as typo. Still no Czar visible. |
| 150 | ~517 | raw | ~3/5 | 2/3 | (Phase 2 oracle at similar density.) Czar truncated. |
| **256** | **834** | **clean** | **4/5** | **3/3** | **Czar joke found for first time. All-Star baseball. MOCR. All 3 grounded, 0 hallucinated.** |
| 512 | 1604 | raw | 3.5/5 | 2/3 | Czar found + linked across excerpts. But repeated Czar for 2/3 picks. Missed Module baby (OCR garbled). Missed Buzz lawn joke. |

### 3b: Quality Curve and the OCR Confound

```
Quality
  5 |
  4 |                  * (256 clean)
3.5 |                              * (512 raw)
  3 |              * (150 raw)
2.5 |          * (100 raw)
  2 |
1.5 |  * (30 raw)
  1 |
    +--+-----+-----+-----+------+---
      30    100   150   256    512  tok/window
```

**The knee is at 256 tokens/window** — but with an important caveat:

1. **Clean 256 (4/5) > Raw 512 (3.5/5).** OCR cleaning matters MORE than raw context
   volume. The garbled characters in "Lunar Mod_ie McGh_e" and fragmented dialogue
   lines actively confuse the 4B model. More raw tokens doesn't help if they're noisy.

2. **The OCR penalty is ~0.5-1.0 quality points.** At 512, the model has all the
   content but can't parse the Module baby story through the artifacts. At 256 clean,
   it finds 3 distinct moments with zero hallucination.

3. **Practical implication for Mode 7:** Either (a) serve 256 clean tokens per window
   (requires OCR post-processing), or (b) serve 512 raw and accept the artifact
   penalty. Option (a) is better — OCR cleanup is cheap, context window is expensive.

4. **Buzz's lawn joke was missed at 512** despite being clearly readable: "I wish we
   could find out when the last time my lawn was cut." The model fixated on the Czar
   joke (used it for 2/3 picks) rather than exploring other excerpts. Possible
   attention/diversity issue with the 4B model on long contexts.

---

## Phase 4 — Query Types

10 windows spanning the full mission (launch→transit→EVA→return), ~200 clean tokens
each, ~2150 total. Same context for all 8 queries. Temperature 0, max 300 tokens.

### 4a: Results by Query Type

| Query | Quality | Category | Key observation |
|-------|---------|----------|-----------------|
| Find 3 amusing moments | 3.5/5 | **Works well** | Czar, baseball, housekeeping correctly found |
| What was the mood? | 3.5/5 | **Works well** | Lighthearted tone, "magnificent desolation", camaraderie |
| Describe crew relationships | 3/5 | **Works well** | Casual friendly atmosphere, teasing, teamwork. Confused CC as crew. |
| What surprised the crew? | 2.5/5 | Partial | Comfort of Moon walking correct. Thin but grounded. |
| What worried the crew? | 2/5 | Partial | Fabricated generic "time and efficiency." No real worries in excerpts. |
| Summarize the timeline | 2/5 | **Fails** | Confused who=what (CMP=Armstrong wrong). Can't reconstruct arc from 10 windows. |
| What went wrong? | 1.5/5 | **Fails** | Called routine waste valve venting a "malfunction." Hallucinated problems. |
| Most tense moment? | 1/5 | **Fails** | Called sandwich exchange "tense." No tense content in excerpts → fabricated. |

### 4b: Pattern — What Predicts Success?

**Works well (3+ /5): TONE and SOCIAL queries**
- Mood, relationships, amusing moments
- These work because tone/social dynamics are PRESENT IN EVERY conversational exchange
- Any 10 windows with crew dialogue contain enough signal

**Partially works (2-2.5/5): INFERENTIAL queries**
- Worried, surprised
- Directionally plausible but thin — the model infers from limited evidence
- Would improve significantly with query-specific window selection

**Fails (1-2/5): SPECIFIC EVENT queries**
- Timeline, what went wrong, most tense
- These REQUIRE windows containing the relevant events (landing alarms, re-entry,
  communication blackouts) which aren't in the amusing-window selection
- 10 windows cannot represent the full mission arc for timeline queries
- When no relevant content exists, the model HALLUCINATES — fabricates problems
  from routine operations, calls lighthearted exchanges "tense"

### 4c: Implication for Mode 7

**Query-aware routing is essential.** The same windows cannot serve all query types:

| Query category | Required content | Router |
|---------------|-----------------|--------|
| Tone/social | Any conversational windows | Contrastive or random works |
| Amusing/informal | Windows with (Laughter), jokes | BM25 keyword expansion |
| Factual (who/what/when) | Windows with specific facts | K-vector routing |
| Events (problems, tension) | Windows with alarms, anomalies | BM25 "alarm warning problem" |
| Timeline/summary | Representative windows per phase | Uniform temporal sampling |

Mode 7 needs a **query classifier** → route to the right window selection strategy.

## Phase 5 — Routing Coverage & Query Expansion

Searched all 725 windows via decoded text. Four strategies tested.

### 5c: Strategy Comparison (Strict Match)

| Strategy | top-5 | top-10 | top-20 |
|---|---|---|---|
| 1: Literal BM25 ("amusing" "humor" "funny") | **0/11 (0%)** | 0/11 | 0/11 |
| 2: Query Expansion (20 terms) | 1/11 (9.1%) | 2/11 (18.2%) | 4/11 (36.4%) |
| 3: Indicator Search ((Laughter) (Music)) | **3/11 (27.3%)** | 4/11 (36.4%) | 5/11 (45.5%) |
| 4: Hybrid (2+3) | **3/11 (27.3%)** | 4/11 (36.4%) | 5/11 (45.5%) |
| *Contrastive Router* | *0/11 (0%)* | *n/a* | *n/a* |
| *Compass Router* | *1/11 (9.1%)* | *n/a* | *n/a* |

With ±2 window tolerance, hybrid reaches **9/11 (81.8%)** at top-20.

### 5c: Indicator Search — Top-20 Detail

| Rank | Window | Score | Terms | Amusing? |
|------|--------|-------|-------|----------|
| 1 | W653 | 3 | laughter×1, music×2 | **Yes** |
| 2 | W654 | 3 | music×3 | No |
| 3 | W652 | 2 | laughter×2 | No |
| 4 | W672 | 2 | laughter×1, chuckle×1 | **Yes** |
| 5 | W694 | 2 | laughter×1, music×1 | **Yes** |
| 8 | W156 | 1 | laughter×1 | **Yes** |
| 14 | W634 | 1 | laughter×1 | **Yes** |
| 10 | W239 | 1 | laughter×1 | No (near W244) |
| 12 | W303 | 1 | laughing×1 | No (near W308) |
| 15 | W645 | 1 | laughter×1 | No (near W644) |
| 16 | W707 | 1 | laughter×1 | No (near W706) |

### 5d: Key Findings

1. **Literal BM25 = total failure.** "Amusing", "humor", "lighthearted" never appear in
   the transcript. "Funny" appears 6 times but means "anomalous" in NASA-speak ("funny
   indications on the O2 flow indicator").

2. **Indicator search is the strongest strategy.** Editorial annotations like `(Laughter)`
   are direct physical evidence of humor. Beats both neural routers at every K.

3. **Query expansion adds almost nothing.** Only 8/20 expanded terms appear at all.
   "Crew" (47 hits) is pure noise. The signal comes from laughter/chuckle annotations.

4. **Two amusing windows are UNREACHABLE by any keyword method:**
   - **W37** (happy birthday from space) — humor is in the absurdity of wishing someone
     happy birthday from orbit. No lexical signal.
   - **W613** (Apollo 8 sphere-of-influence joke, Dave Reed burying head) — humor
     requires understanding social context. No laughter annotation.
   - These require **pragmatic understanding** that no lexical method can provide.

5. **BM25 indicator search beats both neural routers** (27.3% vs 0-9.1% at top-5),
   but the ceiling is 81.8%. The remaining 18.2% needs semantic understanding.

6. **The hybrid adds nothing over indicators alone.** Query expansion contributes noise
   keywords that don't discriminate. Indicator search is sufficient as the sole strategy.

---

## Key Findings So Far

1. **Geometric routing fails for subjective queries.** Contrastive 0%, compass 9%
   recall on amusing windows. No geometric signature for "humour" in residual space.

2. **Keyword search works trivially.** All 11 amusing windows contain obvious surface
   markers: (Laughter), "Czar", "kidding", "funny", "happy birthday", "Music".
   BM25 with query expansion would find them all.

3. **The 4B model CAN identify humor** when given the right content at sufficient
   density. The Czar joke, baseball incongruity, MOCR joke, and Collins housekeeping
   were all correctly identified as amusing from transcript excerpts.

4. **Context density knee at 256 tokens/window.** At 30-100 tokens, punchlines are
   truncated (Czar invisible at 100). At 256 clean, all exchanges visible, 3/3
   grounded, 0 hallucinated. Beyond 256 (raw 512), diminishing returns due to OCR.

5. **OCR artifacts are a bigger bottleneck than context length.** Clean 256 (4/5)
   outperforms raw 512 (3.5/5). The garbled characters actively confuse the model.
   OCR post-processing is cheap; context window is expensive. Clean and compress.

6. **Routing quality > model capacity** for this task. Oracle 3-4/5, random 2.5/5,
   contrastive 1/5. The model succeeds with the right windows and fails with the
   wrong ones. Selection is the bottleneck, not comprehension.

7. **The 4B model fixates on strong signals.** At 512 raw, it used the Czar joke
   for 2/3 picks instead of exploring the Module baby or lawn joke. Diversity of
   response may require prompting changes or temperature >0.

8. **Query type determines success, not just routing.** Tone/social queries (mood,
   relationships, humor) work with any conversational windows (3-3.5/5). Event
   queries (what went wrong, most tense) fail completely (1-1.5/5) because the
   excerpts don't contain the relevant events. The model hallucinates when asked
   about content that isn't in the context.

9. **Query-aware routing is essential for Mode 7.** A query classifier must route
   to different window selection strategies: K-vector for facts, BM25 for events,
   temporal sampling for timeline, any conversational windows for tone/social.

10. **Indicator search (transcript annotations) beats everything for humor.** `(Laughter)`
    markers are direct physical evidence. 27.3% recall at top-5 vs 0% (contrastive) and
    9.1% (compass). With ±2 tolerance, 81.8% at top-20. Keyword expansion adds noise.

11. **Two windows are fundamentally unreachable by keywords.** W37 (happy birthday) and
    W613 (sphere-of-influence joke) require pragmatic understanding. Maximum keyword recall
    is 81.8% — the remaining 18.2% is the semantic gap that only a reasoning model can bridge.

## Implications for Mode 7

### Routing Architecture

| Query type | Router | Recall | Example |
|-----------|--------|--------|---------|
| Factual (who/what/when) | K-vector Q·K | 6/6 (100%) | "What city was Zarkov from?" |
| Humor/informal | Indicator search ((Laughter)) | 5/11 strict, 9/11 tolerant | "Find amusing moments" |
| Events/problems | BM25 "alarm warning problem" | TBD | "What went wrong?" |
| Tone/mood/social | Any conversational windows | ~3.5/5 | "What was the mood?" |
| Timeline/summary | Uniform temporal sampling | TBD | "Summarize the mission" |

**Query classifier → router selection → window retrieval → clean context → generate.**

### Context Budget

- 256 clean tokens/window × 10 windows = **2560 tokens**
- Well within 8K context
- ~2s generation time at 4B parameters
- OCR cleaning required (cheap, ~10ms per window)

### What NOT to Build

- **Query expansion for humor.** Adds noise, not signal. Indicator search alone is better.
- **Geometric routing for subjective queries.** 0% recall. Don't try.
- **Raw OCR context.** Clean 256 > raw 512. Always clean.
- **Single router for all query types.** Different queries need fundamentally different
  window selection strategies.

### The Semantic Gap (18.2%)

Two of 11 amusing windows are unreachable by ANY lexical method:
- W37: Happy birthday from space (pragmatic humor — the absurdity of the situation)
- W613: Sphere-of-influence joke (social humor — understanding personnel dynamics)

These require a model that can READ windows and JUDGE whether they're funny — essentially
a two-stage pipeline: (1) coarse retrieval via indicators, (2) fine reranking via a
reasoning model. This is the gap between 81.8% and 100%.

### Summary

The pattern holds: **measure in Lazarus, then engineer.**

Broad context reasoning works for the 4B model when:
1. The **right windows** are selected (routing quality is the bottleneck)
2. With **enough context** (≥256 tokens per window)
3. That is **cleaned of OCR artifacts**
4. For **query types that match the content** (tone/social yes, events no)

Mode 7 is viable. The routing is the hard part, and different routers for different
query types is the architecture.
