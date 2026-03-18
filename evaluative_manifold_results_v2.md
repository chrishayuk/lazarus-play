# Evaluative Manifold — Full Experiment Results (v2)

**Model:** google/gemma-3-4b-it (34 layers, 2560 hidden dim)
**Date:** 2026-03-17
**Checkpoint:** apollo11_ctx_512 (725 windows, 370,778 tokens)

**Status: IN PROGRESS**

---

## Ground Truth Windows

### Tense (7 windows)
| Window | Content | Keyword? |
|--------|---------|----------|
| W364 | GO to continue powered descent, PGNS lock-on, DELTA-H | IMPLICIT (no alarm word, but descent) |
| W367 | GO for landing, 3000ft, 1201/1202 alarms | KEYWORD (1201, 1202) |
| W369 | 60 seconds fuel, 120ft, 100ft, descent rates | KEYWORD (60 seconds) |
| W370 | CONTACT LIGHT, ENGINE STOP — touchdown moment | IMPLICIT (clipped commands) |
| W281 | LOI-2 prep, about to go behind the Moon | IMPLICIT (situational) |
| W339 | Pre-DOI, K-factor, preparing for descent orbit | IMPLICIT (situational) |
| W347 | Ignition countdown, 7 minutes, thruster stabilization | IMPLICIT (countdown) |

### Surprising (7 windows)
| Window | Content | Keyword? |
|--------|---------|----------|
| W450 | "Magnificent sight out" — lunar surface close up | KEYWORD (magnificent) |
| W85 | Mediterranean through sextant, "fantastic sight" | KEYWORD (fantastic) |
| W97 | "View is just beautiful, out of this world" | KEYWORD (beautiful) |
| W257 | Lunar transient events, earthshine, spacecraft darkness | IMPLICIT (unexpected observation) |
| W461 | EVA paces on lunar surface, "progressing beautifully" | IMPLICIT (first time activity) |
| W371 | Post-touchdown, MASTER ARM ON, first moments on Moon | IMPLICIT (historic moment) |
| W452 | Walking on surface, losing balance, light-footed | IMPLICIT (novel experience) |

### Social (7 windows)
| Window | Content | Keyword? |
|--------|---------|----------|
| W238 | Sports scores, Houston astrologer Ruby Graham | KEYWORD (astrologer) |
| W239 | Astrologer continued, antenna switching problems | KEYWORD (astrologer) |
| W421 | Collins: "I haven't heard anything fairly lately" | KEYWORD (haven't heard) |
| W462 | EVA walking styles, discussion between crew | IMPLICIT (interpersonal) |
| W532 | Backup crew congratulations, "prayers with you" | KEYWORD (congratulations) |
| W576 | "Not kidding" exchange | KEYWORD (kidding) |
| W653 | "Highlight of my day," music playing | IMPLICIT (personal) |

### Routine (7 windows, shared negative set)
| Window | Content |
|--------|---------|
| W7 | Star check, P52 data |
| W10 | Abort PAD numbers, sextant star data |
| W67 | Block data readback, flight plan update |
| W80 | M-line alignment, state vector discussion |
| W158 | Battery charge, waste water dump |
| W204 | Waste water dump procedure |
| W246 | P52 option, TEI PAD copy |

---

## Experiment 1 — Probe Training

**Status: STARTING**

Training protocol per probe:
1. Replay window text (256 tokens)
2. Append assessment prompt
3. Generate 20 tokens (temperature 0)
4. Extract L26 at last generated token
5. Train/test split: 5+5 train, 2+2 held-out

### Phase 1a — Tension Probe
*Pending....*

### Phase 1b — Surprise Probe
*Pending....*

### Phase 1c — Social Dynamics Probe
*Pending...*
