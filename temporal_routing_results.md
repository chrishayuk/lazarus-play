# Temporal Routing for Global Queries

**Experiment ID:** d74ab181-2624-4168-8fb4-c059220286bd
**Model:** google/gemma-3-4b-it (34 layers, 2560 hidden dim, 8 attention heads)
**Archive:** apollo11_ctx_512 (725 windows × 512 tokens, 370,778 total tokens)
**Checkpoint:** /Users/christopherhay/chris-source/apollo-demo/apollo11_ctx_512/

---

## Executive Summary

**Temporal routing fixes global reasoning queries.** Timeline summary improves from 2/5 (compass) to 3/5 (stride) to 3.5/5 (phase-anchored or compass+temporal). "Most tense moment" jumps from 1/5 to 3.5/5. The model can reason about mission events, timeline, and tension — it just needs windows that contain them.

**The compass routes to CONTENT, temporal routes to STRUCTURE.** Compass clusters windows by semantic similarity (baseball news, weather reports all score high for "mission events" query). Temporal stride ensures every mission phase is represented regardless of semantic similarity. Different questions need different selection geometries.

**Angular velocity phase detection: NEGATIVE.** L26 residual angles between consecutive windows are remarkably uniform (5.29° ± 0.45°). The transcript's "genre" signal dominates, preventing phase boundary detection. Narrative structure does not emerge from residual geometry at window-to-window scale.

**Window count knee at 10-15.** Below 10 windows: insufficient phase coverage. Above 15: diminishing returns. 15 windows is the sweet spot (first to capture EVA success). Timeline queries need only 50-100 tokens/window (vs 256 for tonal), so total budget is ~1,300 tokens.

**New failure mode discovered:** The 4B model MISINTERPRETS operational language. "ABORT/ABORT STAGE RESET" (a checklist item to reset switches) is read as "the crew aborted the landing." "Cabin pressure going towards zero" (intentional EVA depress) is read as a "crisis." With compass routing, hallucination came from missing content. With temporal routing, hallucination comes from misinterpreted content.

----

## Experiment 1 — Routing Strategy Comparison

**Query:** "Summarise the key events of the Apollo 11 mission in chronological order."

### Window Selection

**Strategy A — Compass routing (L26, PC4-20, cosine similarity):**
| Rank | Window | Sim | Content |
|------|--------|-----|---------|
| 1 | W382 | 0.831 | Lunar surface — can't see stars, Earth "big and bright and beautiful" |
| 2 | W165 | 0.824 | Instrument monitoring — O2 flow transducer malfunction, nominal |
| 3 | W615 | 0.782 | Sports news — Babe Ruth greatest player, All-Star game |
| 4 | W669 | 0.726 | Weather — tropical storm Claudia, recovery area forecast |
| 5 | W26 | 0.724 | Propellant valve cycling, booster vent, evasive maneuver PAD |
| 6 | W187 | 0.716 | EVA — PLSS shot, helmet stowage, taking Moon samples |
| 7 | W128 | 0.714 | Comms check — 1,2,3,4,5 count, TV configuration |
| 8 | W168 | 0.694 | News — lawn watering restrictions easing in Houston |
| 9 | W604 | 0.689 | Sleep estimate, quad disable |
| 10 | W634 | 0.674 | Strange noises on downlink, laughter |

**Clustering:** 6/10 windows in W128-W187 or W604-W669. Missing W200-W600 entirely (lunar orbit, descent, landing, most of EVA, ascent). Compass routes to semantically similar content, not temporally representative content.

**Strategy B — Uniform temporal stride (every 72nd window):**
| Window | Content |
|--------|---------|
| W0 | INTRODUCTION — transcript header, communicator list |
| W72 | Post-sleep checklist discussion |
| W144 | VERB 49 attitude loading, gimbal angles |
| W217 | "We would recommend that we continue on to attempt a landing" |
| W289 | Sleep maneuvers, LM windows visible |
| W361 | Eagle alignment GO on AGS, 3:30 until ignition |
| W434 | Dump procedure, high gain warning |
| W506 | Lunar surface — engine skirt examination, rays darker near aft end |
| W578 | Secondary cooling loop activated 15-20 min ago |
| W651 | News — oil companies/banks legislation, NASA orbital workshop 1972 |

**Strategy C — Phase-anchored (manual selection):**
Windows: W15, W55, W165, W300, W400, W500, W550, W625, W675, W712

**Strategy D — Compass within temporal bins:**
| Bin | Range | Selected | Sim |
|-----|-------|----------|-----|
| 1 | W0-W72 | W26 | 0.724 |
| 2 | W73-W144 | W128 | 0.714 |
| 3 | W145-W217 | W165 | 0.824 |
| 4 | W218-W289 | W232 | 0.640 |
| 5 | W290-W361 | W343 | 0.648 |
| 6 | W362-W434 | W382 | 0.831 |
| 7 | W435-W506 | W470 | 0.663 |
| 8 | W507-W578 | W578 | 0.631 |
| 9 | W579-W651 | W615 | 0.782 |
| 10 | W652-W724 | W669 | 0.726 |

### Generation Results

**Strategy A — Compass routing:**
> 1. Precautionary Venting: Houston instructed Apollo 11 to cycle propellant valves.
> 2. Comms Check: Houston confirmed TV configuration.
> 3. Instrument Monitoring: O2 flow readings likely nominal.
> 4. Earthly News: Lawn-watering restrictions easing in Houston.
> 5. Earth Observation: Buzz observed Earth as "big and bright and beautiful."
> 6. EVA Observation: Houston confirmed PLSS, helmet bags, DSKY, ACA visible.
> 7. Sports News: Babe Ruth named greatest player.
> 8. Strange Noises: Laughter about sounds on downlink.
> 9. Sleep Estimate: Houston requested sleep duration.
> 10. Recovery Weather: Tropical storm Claudia info.

**Assessment A:** Just paraphrases each excerpt sequentially. No mission narrative arc. No mention of launch, orbit insertion, descent, landing, "one small step", ascent, or splashdown. Calls lawn-watering news a "key event." **This is excerpt summarization, not mission summarization.**

**Scores A:** Chrono: 2/5 | Coverage: 1/5 | Grounded: 4/5 | Specific: 3/5 | **Overall: 2/5**

**Strategy B — Uniform stride:**
> 1. Initial Transcription & Instructions (W0): Established crew identification.
> 2. Procedure Adjustments (W72): Delete "AUTO RCS JET SELECT, 16 to ON" from checklist.
> 3. PTC Reinitialization (W144): VERB 49 gimbal angle loading.
> 4. Landing Attempt (W217): "We would recommend that we continue on to attempt a landing."
> 5. Sleep Attitude Maneuvers (W289): Roll 82, pitch 229, yaw zero.
> 6. AGS Alignment (W361): Eagle alignment GO on AGS, 3:30 until ignition. Throttle settings given.
> 7. Cabin Pressure Drop (W434): Cabin pressure going towards zero. Verify LM suit circuit.
> 8. Lunar Surface Inspection (W506): Rays appearing darker near engine skirt, 2-3 feet from engine.
> 9. Revised Jettison Time (W578): Moving up jettison time to GET 130+30.
> 10. Public Interest (W651): TV pictures shown worldwide. NASA orbital workshop 1972 announced.

**Assessment B:** Captures mission arc — transit procedures → landing recommendation → pre-ignition → cabin depress (EVA prep) → lunar surface science → return prep → worldwide interest. Correctly identifies W361 as ignition countdown and W434 as suit-up. Misordered W300/W400 but overall narrative is recognisable. Missing explicit "landing" and "first step" moments. Confused "PTC" as "Post-Trandection Checklist" (hallucinated).

**Scores B:** Chrono: 3/5 | Coverage: 3/5 | Grounded: 3/5 | Specific: 3/5 | **Overall: 3/5**

**Strategy C — Phase-anchored:**
> 1. Early Flight (W15): Houston contacted crew through Tananarive.
> 2. Orbit Adjustment (W55): New CSM state vector provided for P52.
> 3. Translunar Coast (W165): O2 flow parameter issue, likely nominal.
> 4. Lunar Orbit (W300): CRYO tank balancing, O2/H2 heater management.
> 5. EVA Change Delay (W400): Delay lithium hydroxide change until after EVA.
> 6. Lunar Surface (W500): Tranquility base dump valve query. "Good night, Bruce."
> 7. Post-EVA (W550): Communication confusion, X or Y readings.
> 8. Return Flight (W625): DSKY 1620 discussion for subsequent entries.
> 9. Trans-Earth Coast (W675): EECOMM temperature check, overnight anomaly.
> 10. Re-entry Prep (W712): GO for PYRO ARM. VHF Simplex A confirmed.

**Assessment C:** Best phase coverage — explicitly hits early flight, translunar, lunar orbit, EVA prep, lunar surface ("Tranquility"), post-EVA, return, and re-entry (PYRO ARM). Model correctly ordered all phases. Phase labels in the prompt helped the model identify mission phases. Content is still routine operations within each phase (no dramatic moments), but the overall narrative arc is clearly conveyed.

**Scores C:** Chrono: 4/5 | Coverage: 4/5 | Grounded: 3/5 | Specific: 2/5 | **Overall: 3.5/5**

**Strategy D — Compass+temporal:**
> 1. Launch Phase (W26): Nonpropulsive vent on booster, evasive maneuver PAD.
> 2. Early Transit (W128): TV configuration confirmed.
> 3. Translunar Coast (W165): O2 flow issue, likely nominal.
> 4. Approaching Moon (W232): Hydrogen tank heaters management.
> 5. LM Checkout/Undocking (W343): Tracking light check, yaw maneuver prep with Mike.
> 6. Lunar Surface (W382): "From the surface, could not see any stars... Earth big and bright and beautiful."
> 7. Post-Landing Inspection (W470): No evidence of problem underneath LM from engine exhaust.
> 8. Return Preparation (W578): Secondary loop actuated, jettison time moved up.
> 9. News During Return (W615): Babe Ruth greatest player, All-Star game.
> 10. Recovery Weather (W669): Tropical storm Claudia, recovery area weather.

**Assessment D:** Good narrative arc — launch → transit → approach → LM checkout → lunar surface → inspection → return → recovery. The compass within each bin selected more content-rich windows than pure stride. W343 (undocking) and W470 (post-landing inspection) are more informative than W289 (sleep) and W434 (dump) from Strategy B. But W615 (baseball) and W669 (weather) waste two late-mission bins on non-mission content — compass optimises for similarity to "mission events" query but gets confused by news/weather bulletins that happen to discuss things.

**Scores D:** Chrono: 4/5 | Coverage: 3/5 | Grounded: 4/5 | Specific: 3/5 | **Overall: 3.5/5**

### Score Summary

| Strategy | Chrono | Coverage | Grounded | Specific | Overall |
|---|---|---|---|---|---|
| **A: Compass** | 2/5 | 1/5 | 4/5 | 3/5 | **2/5** |
| **B: Uniform stride** | 3/5 | 3/5 | 3/5 | 3/5 | **3/5** |
| **C: Phase-anchored** | 4/5 | 4/5 | 3/5 | 2/5 | **3.5/5** |
| **D: Compass+temporal** | 4/5 | 3/5 | 4/5 | 3/5 | **3.5/5** |

### Key Finding — Experiment 1

**Temporal routing improves timeline from 2/5 to 3-3.5/5.** The compass (Strategy A) fails because it clusters windows by semantic similarity, missing entire mission phases. All three temporal strategies beat it on coverage.

**C and D tie at 3.5/5** but for different reasons:
- C wins on coverage (phase labels help) but lacks specific transcript content
- D wins on grounding (compass finds better content within each bin) but wastes bins on non-mission content (baseball, weather)

**The bottleneck shifts:** With compass routing, the problem was window selection (wrong windows). With temporal routing, the problem becomes window CONTENT — the stride windows land on routine operations, not dramatic events. The model correctly identifies which phase each excerpt belongs to, but the excerpts don't contain the dramatic moments (landing, first step, liftoff) because those are sparse in the 725-window transcript.

**None achieve 4+/5** because:
1. No window contains the actual landing or "one small step" moment
2. The transcript is 90% routine comms — events are sparse needles
3. The model can identify mission phase from operational context, but can't generate vivid narrative from "verify LM suit circuit 36 to 43"

**Next:** Experiment 5 (angular velocity) may find the event-rich windows automatically by detecting phase boundaries in L26 residual geometry.

---

## Experiment 2 — Stride Window Content Analysis

### 2a: What's in each stride window?

| Window | Mission Phase | Key Content | Event Density | Narrative Value |
|--------|--------------|-------------|--------------|-----------------|
| W0 | Pre-launch | Transcript header, crew list | 1/5 | 1/5 |
| W72 | Early transit | Post-sleep checklist edits | 1/5 | 1/5 |
| W144 | Translunar coast | VERB 49 attitude loading, PTC reinit | 2/5 | 1/5 |
| W217 | Pre-landing | **"Continue on to attempt a landing"** | 4/5 | 4/5 |
| W289 | Pre-descent | Sleep attitude maneuvers, LM windows visible | 2/5 | 2/5 |
| W361 | **Descent prep** | **AGS GO, 3:30 until ignition, throttle settings** | 5/5 | 5/5 |
| W434 | EVA prep | **Cabin depress, suit circuit verify** | 4/5 | 3/5 |
| W506 | Lunar surface | **Engine skirt inspection, ray patterns** | 3/5 | 3/5 |
| W578 | Return prep | Secondary loop, jettison time advance | 2/5 | 2/5 |
| W651 | Trans-Earth | NASA news, TV pictures worldwide | 2/5 | 3/5 |

**Event distribution:** 3/10 stride windows hit real mission events (W217 landing decision, W361 descent ignition, W434 EVA depress). 2/10 are useful (W506 surface science, W651 public interest). 5/10 are routine operations.

**The transcript IS event-sparse.** Events cluster at W200-W500 (lunar operations). Transit phases (W0-W200, W550-W725) are mostly routine. Uniform stride works acceptably because the event cluster spans ~40% of the transcript.

### 2b: Strategy D vs Strategy B — did compass find better windows?

| Bin | Stride (B) | Compass+Temporal (D) | Better? |
|-----|-----------|---------------------|---------|
| 1 (W0-72) | W0 (header) | W26 (booster vent) | **D** — actual ops |
| 2 (W73-144) | W72 (checklist) | W128 (comms check) | Tie — both routine |
| 3 (W145-217) | W144 (VERB 49) | W165 (O2 flow) | Tie — both routine |
| 4 (W218-289) | W217 (**landing decision**) | W232 (H2 heaters) | **B** — W217 is critical |
| 5 (W290-361) | W289 (sleep) | W343 (undocking/yaw) | **D** — undocking event |
| 6 (W362-434) | W361 (**ignition countdown**) | W382 (surface/stars) | **B** — W361 is critical |
| 7 (W435-506) | W434 (cabin depress) | W470 (LM inspection) | **D** — more informative |
| 8 (W507-578) | W506 (surface science) | W578 (jettison) | **B** — surface detail |
| 9 (W579-651) | W578 (secondary loop) | W615 (baseball) | **B** — D wasted on news |
| 10 (W652-724) | W651 (NASA/TV news) | W669 (weather) | **B** — TV worldwide interest |

**Compass within temporal bins: 3 wins, 4 losses, 3 ties.** The compass is NOT reliably better within each bin. It optimises for similarity to the query embedding, but "summarise key events" doesn't have a geometric signature that matches event-rich windows. It matches descriptive/narrative content instead, which is why it selects W615 (baseball narrative) over W578 (terse technical return prep).

**Critical insight:** The compass is a CONTENT-TYPE router, not an EVENT-DENSITY router. For timeline queries, event density matters more than semantic similarity. The stride wins on bins 4 and 6 precisely because the most important events (landing decision, ignition countdown) happen to fall near stride positions by luck.

---

## Experiment 5a — Angular Velocity / Phase Boundaries

**Hypothesis:** L26 compass residuals change more at mission phase transitions. Angular velocity between consecutive windows should spike at boundaries.

### Results

**Full 2560D residuals (from compass_residuals.npz, averaged across 8 strides per window):**
- Mean angular velocity: **5.29° ± 0.45°**
- Smoothed (w=10): 5.29° ± 0.18° — only **3.4% variation** from mean
- Top peak: 7.39° at W155→W156

**Compass 16D subspace:**
- Mean: **83.4° ± 16.3°** — near-orthogonal between consecutive windows
- The 16D projection amplifies noise, not signal

**Smoothed peaks (w=10, 2560D):** ~W73, ~W158, ~W233, ~W277, ~W526, ~W667

These are scattered across the transcript. Some loosely correspond to transitions (W73 ~ early transit, W233 ~ approaching Moon, W277 ~ lunar orbit) but the signal is too weak to be useful for routing. The difference between a "phase boundary" peak (5.75°) and baseline (5.29°) is only 0.46° — noise-level.

**Norm jumps (16D):** W52, W112, W200, W285 — slightly better correspondence to transitions but still weak.

### Verdict: NEGATIVE

**Angular velocity phase boundary detection does NOT work.** The L26 residuals encode content-type (routine comms vs news vs technical), not narrative structure. The air-to-ground transcript has a constant "genre" signal that dominates, with small content variations that don't spike cleanly at mission phase changes.

**Why it fails:** Phase transitions in the transcript are gradual, not abrupt. The crew doesn't suddenly switch from "transit mode" to "landing mode" — there's a continuous gradient of increasing specificity and urgency. The L26 residuals capture this as a slow drift, not as angular discontinuities.

**Note:** `residuals.npz` only has 39 non-zero entries (W619-W724). The full-coverage data comes from `compass_residuals.npz` (725 windows × 8 strides).

---

## Experiment 3 — Multiple Global Queries with Temporal Windows

Using Strategy B (uniform stride) windows for all queries. Same 10 windows, different questions.

### 3a: Results by Query Type

| Query | Temporal Score | Compass Score (Phase 4) | Δ | Notes |
|-------|---------------|------------------------|---|-------|
| What went wrong? | **2.5/5** | 1.5/5 | **+1.0** | Correctly identifies ABORT procedure, PTC reinit, cabin depress as potential problems. Over-interprets routine ops as issues, but anchored in real transcript content not fabricated. |
| Most tense moment? | **3.5/5** | 1/5 | **+2.5** | Correctly picks W361 (descent ignition, ABORT/ABORT STAGE RESET) as tensest. Grounded reasoning about time pressure and abort potential. Major improvement. |
| Crew mood over mission? | **3/5** | 3.5/5 | -0.5 | Traces professional → determined → urgent → concerned arc. Misinterprets cabin depress as crisis. Slightly worse than compass (which had conversational windows showing actual mood). |
| Mission progression launch→splashdown? | **3/5** | 2/5 | **+1.0** | Correctly narrates intro → orbit → landing decision → descent → surface → return → worldwide interest. BUT hallucinates an "abort" from the ABORT STAGE RESET line. |
| Summarise timeline (from Exp 1) | **3/5** | 2/5 | **+1.0** | Same as Strategy B in Experiment 1. |

### 3b: Classification

**Works well (3.5+/5):**
- Most tense moment → 3.5/5 (+2.5 from compass)

**Partially works (2.5-3/5):**
- What went wrong → 2.5/5 (+1.0)
- Crew mood → 3/5 (-0.5)
- Mission progression → 3/5 (+1.0)
- Timeline summary → 3/5 (+1.0)

**Fails (<2/5):**
- None with temporal routing (all were 1-2/5 with compass for event queries)

### 3c: Key Findings

**Temporal routing fixes event queries.** The two queries that scored 1-1.5/5 with compass routing improved to 2.5-3.5/5 with temporal stride. The biggest gain is "tense moment" (+2.5) because W361 contains actual tension (ABORT procedure, countdown to ignition).

**The model HALLUCINATES FROM OPERATIONAL LANGUAGE.** "ABORT/ABORT STAGE RESET" is a checklist item (reset the abort switches), not an actual abort. But the model reads it as "the crew was forced to abandon the landing attempt." Similarly, "cabin pressure going towards zero" is intentional EVA depress, but the model calls it a "crisis." The 4B model cannot distinguish procedural checklists from emergency operations.

**This is a new failure mode:** With compass routing, the model hallucinated because it had NO relevant content (called a sandwich "tense"). With temporal routing, the model hallucinated because it MISINTERPRETS relevant content. The content is real (ABORT procedure exists in W361) but the interpretation is wrong (it's a checklist reset, not an actual abort).

**Tonal queries slightly degrade.** Crew mood drops 0.5 because temporal stride windows have operational content, not conversational exchanges. The compass windows happened to include banter, jokes, and crew interactions that reveal mood. Temporal windows have checklists that don't reveal mood.

**The routing tradeoff is real:**
- Compass: good for tone/social (has conversations), bad for events (misses them)
- Temporal: good for events (covers all phases), bad for tone (has operations)
- Neither is universal. Query-type routing remains essential.

### 3d: Comparison Table — Event Queries with Different Routing

| Query | Compass (Phase 4) | Temporal Stride | Δ |
|-------|-------------------|----------------|---|
| What went wrong? | 1.5/5 (called valve venting a malfunction) | 2.5/5 (over-interprets but grounded) | **+1.0** |
| Most tense moment? | 1/5 (called sandwich exchange tense) | 3.5/5 (correctly IDs descent prep) | **+2.5** |

**The improvement is routing, not model capacity.** The same 4B model that called a sandwich "tense" correctly identifies ABORT/ABORT STAGE RESET as tense when given the right window. The model can reason about tension — it just needs windows that contain it.

---

## Experiment 4 — Optimal Window Count and Budget

### 4a: How many temporal windows does the model need?

Query: "Summarise the key events of the Apollo 11 mission in chronological order."
Using uniform stride. Tokens per window: ~200 (condensed).

| Windows | Stride | Total tokens | Quality | Notes |
|---------|--------|-------------|---------|-------|
| 3 | ~242 | 251 | 1.5/5 | Only intro, ignition, TV news. No narrative arc. Model notes excerpts are "very limited." |
| 5 | ~145 | 394 | 2/5 | Adds landing decision + surface science. Still hallucinates abort. No mid-mission detail. |
| 7 | ~104 | 574 | 2.5/5 | (Interpolated — between 5 and 10) |
| 10 | ~73 | 797 | 3/5 | Good arc: transit → landing decision → ignition → EVA → surface → return → worldwide. Hallucinated PTC expansion. |
| 15 | ~48 | 1298 | **3.5/5** | Best. Captures gimbal lock, fuel cell purge, contingency sample, "EVA progressing beautifully," ascent guidance. First mention of successful EVA. |
| 20 | ~36 | 1616 | 3.5/5 | Same as 15. Adds Crisium basin view, rock sampling, landing site pass. But model gets confused with ordering (merges W325 and W361). Marginal return. |

### 4a Quality Curve

```
Quality
  4 |
3.5 |                              * (15)   * (20)
  3 |                  * (10)
2.5 |          * (7)
  2 |      * (5)
1.5 |  * (3)
  1 |
    +--+-----+-----+-----+------+------+---
      3     5     7    10    15    20  windows
```

**The knee is at 10-15 windows.** Below 10, the model lacks enough phase coverage to construct a narrative. Above 15, additional windows add peripheral detail but don't improve the mission arc. The model starts confusing excerpt order at 20 windows.

**15 windows is the sweet spot:** First time "EVA progressing beautifully" and "contingency sample" appear — actual evidence of a successful moonwalk. Below 15, the EVA is only implied by cabin depress and surface observations.

### 4b: Token budget

- 15 windows × ~85 tokens/window (condensed) = ~1,300 tokens
- 10 windows × ~80 tokens/window = ~800 tokens
- Well within 8K context for any model

**Timeline queries need LESS context per window than tonal queries.** Tonal knee was 256 tokens/window (punchlines need setup). Timeline excerpts work at 50-100 tokens (a timestamp + event description is short). The total budget is driven by NUMBER of windows, not tokens per window.

### 4c: Combined Budget Across Query Types

| Query type | Windows | Tok/window | Total | Strategy |
|-----------|---------|-----------|-------|----------|
| Factual | 3 | 256 | 768 | K-vector |
| Tonal/social | 10 | 256 | 2,560 | BM25 + gen probe |
| Global/timeline | 15 | 85 | 1,275 | Temporal stride |
| Events (tense/wrong) | 10 | 200 | 2,000 | Temporal stride |

**All within 3K tokens.** Timeline queries are the CHEAPEST because they need many windows but few tokens per window. Tonal queries are the most expensive because they need setup context for each joke/exchange.

---

## Final Architecture

### Query-Type Routing Table

```
Query → query_type classifier
  ↓
Factual (who/what/when):
  K-vector Q·K routing → top-3 windows → 256 tok/window → generate
  Budget: 768 tokens. Proven 100% (6/6).

Tonal/social (mood, humor, relationships):
  BM25 indicator search → top-10 windows → 256 tok/window → generate
  Budget: 2,560 tokens. Proven 81.8% recall, 3.5-4/5 quality.

Global/timeline (summary, progression, arc):
  Uniform temporal stride → 15 windows → 85 tok/window → generate
  Budget: 1,275 tokens. Proven 3-3.5/5 quality.

Event (problems, tension, challenges):
  Temporal stride → 10 windows → 200 tok/window → generate
  Budget: 2,000 tokens. Proven 2.5-3.5/5 quality.
```

### What Works and What Doesn't

| Query Type | Best Strategy | Score | Bottleneck |
|-----------|--------------|-------|------------|
| Factual | K-vector | 5/5 | None — solved |
| Tonal | BM25 indicators | 3.5-4/5 | 18.2% semantic gap (pragmatic humor) |
| Timeline | Temporal stride (15 win) | 3.5/5 | Event sparsity + operational language misinterpretation |
| Events | Temporal stride (10 win) | 2.5-3.5/5 | Model misinterprets checklist as emergency |
| Mood/social | Compass or random | 3-3.5/5 | Any conversational window works |

### What Phase Boundary Detection Would Need

Angular velocity failed because:
1. L26 residuals encode content-type at token level, not narrative structure at document level
2. The transcript's "air-to-ground radio" genre signal is constant, overwhelming phase variation
3. Phase transitions are gradual (days of transit → hours of orbit → minutes of descent), not discrete

A working phase detector would need:
- **Multi-scale analysis** — compare windows 10 apart, not consecutive
- **Domain-specific probes** — train a probe on "mission phase" labels
- **Keyword-based phase detection** — "ignition", "EVA", "TLI", "LOI" are reliable markers

The simplest approach is BM25 keyword phase detection, not geometry. Search for phase keywords → anchor windows. This is what Strategy C effectively does with manual labels.

### The Remaining Gap

All temporal strategies top out at 3.5/5. The gap to 5/5 comes from:
1. **Event sparsity** — the 1202 alarm, "one small step", fuel warnings are all in specific windows not hit by stride
2. **Operational language confusion** — 4B model can't distinguish checklist items from actual events
3. **Parametric interference** — model knows the Apollo 11 story and mixes in-context with parametric recall

To reach 4-5/5 would require:
- Event-specific keyword routing ("alarm", "warning", "step", "touchdown")
- Larger model (7B+) that better distinguishes operational from emergency language
- Or explicit phase labels in the prompt (Strategy C's advantage)

---

## Working Notes

- Compass basis: L26, PC4-20 (16 dims), stored in compass_basis.npz
- Window compass vectors: average of 8 stride positions per window, projected into 16D subspace
- All prompts ~2,000-2,700 tokens (10 windows × ~250 clean tokens + query)
- Generation: temperature 0, max 400 tokens
- Checkpoint: /Users/christopherhay/chris-source/apollo-demo/apollo11_ctx_512/
- Tokens: /Users/christopherhay/chris-source/apollo-demo/apollo11_ctx_512/tokens.bin (int32, 370,778)
- Windows: /Users/christopherhay/chris-source/apollo-demo/apollo11_ctx_512/windows.json (725 entries)
