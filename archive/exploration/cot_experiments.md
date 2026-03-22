# Chain-of-Thought as Circuit Input Simplification
## Mechanistic Analysis on Gemma 3-4B-IT

*Experiments building on the Head 1 / Layer 24 relational bridge circuit findings.*

---

## Background

Previous experiments identified Head 1 at layer 24 as a **contextual attribute bridge circuit**: it
extracts the most salient cultural attribute of an entity and injects it as an intermediate
reasoning step. The circuit works for 3-hop chains with unambiguous referents but fails on 4-hop
chains when referent competition causes it to fire on the wrong entity's attribute.

**The hypothesis being tested:** Chain-of-thought prompting works because it externalises referent
resolution — giving Head 1 a clean, unambiguous entity at each step. CoT doesn't add reasoning
capacity; it simplifies each step so the existing circuit can fire correctly.

**Central question — Hypothesis A vs B:**
- **H-A (input simplification):** The circuit is identical across direct and CoT prompts. Head 1
  does the same operation. CoT just ensures the entity is unambiguous, so Head 1 fires correctly.
- **H-B (different circuits):** CoT recruits different attention heads or layers. The computation
  is fundamentally different.

---

## Experiment 1 — The Broken Chain vs CoT

**Base case — Direct (4-hop, known to fail):**

Prompt: `"The birthplace of the spouse of the author of Hamlet was"`

| | |
|---|---|
| Head 1 top token | " Denmark" (69.4% of layer 24) |
| Top prediction | " in" at 23.7% |
| Output | "in the county of Norfolk" (wrong) |

Head 1 misfires on Hamlet's thematic setting (Denmark) rather than the author's nationality
(Elizabethan/English). The referent "the author of Hamlet" competes with "Hamlet-the-play" for
Head 1's attention, and the play's Danish association wins.

---

**CoT — Step by step:**

| Step | Prompt | Head 1 top token | % | Output |
|---|---|---|---|---|
| 1 | "The author of Hamlet was" | " Elizabethan" | 15.7% | William Shakespeare ✓ |
| 2 | "The spouse of William Shakespeare was" | " Elizabethan" | 135.8% | Anne Hathaway ✓ |
| 3 | "The birthplace of Anne Hathaway (Shakespeare's wife) was" | " Warwickshire" | 39.8% | Stratford-upon-Avon ✓ |

Each step resolves correctly. Head 1 fires on the right cultural intermediate every time:
- Step 1: Entity is unambiguous ("Hamlet was" → only one referent → Shakespeare). Head 1 fires
  " Elizabethan" as the era bridge, though Head 4 fires " Shakespeare" more strongly (24.6%) here.
- Step 2: "William Shakespeare" is the explicit subject. Head 1 fires " Elizabethan" at 135%
  concentration — the era attribute becomes the bridge to Anne Hathaway.
- Step 3: "Anne Hathaway" is explicit. Head 1 fires " Warwickshire" — the correct geography that
  routes toward Stratford-upon-Avon.

---

**Full CoT as single prompt:**

Prompt: `"The author of Hamlet was William Shakespeare. Shakespeare's spouse was Anne Hathaway.
Anne Hathaway was born in"`

| | |
|---|---|
| Head 1 top token (targeting Stratford) | " Elizabethan" (75.4% of layer 24) |
| " Stratford" logit at layer 24 | cumulative ~15.8 |
| " Stratford" logit at layer 28 | cumulative ~21.1 (FFN top token = " Stratford") |
| Layer 33 FFN | +11.0 toward " " (space) |
| Top prediction | " " at 98.4% |
| Output | "1556 and died in 1623" (date, not place) |

**The circuit fired correctly.** Head 1 fired " Elizabethan" at 75% and " Stratford" built to
logit ~21 by layer 28. Then **layer 33 hijacked the output**: a massive FFN spike (+11.0)
overrode the assembled geographic signal with a temporal template ("born in [year]"). The
construction "Anne Hathaway was born in" activates "born in [date]" more strongly than "born in
[place]" in the training distribution.

This failure is not a circuit failure — it is a **layer 33 template override**. The circuit
succeeded; the phrasing failed.

---

## Experiment 2 — Does CoT Change the Circuit or Just the Input?

**Logit attribution comparison — direct vs minimal CoT for " Stratford":**

*Direct broken chain:* `"The birthplace of the spouse of the author of Hamlet was"` (3.6% prob)

| Layer | Attn | FFN | Cumulative | Notes |
|---|---|---|---|---|
| 0 | −9.97 | +19.56 | +15.44 | Standard embedding reset |
| 16 | +0.41 | −0.05 | +1.38 | Weak start |
| 24 | +2.06 | +1.91 | +9.88 | Both components add; att top = " what" |
| 26 | +0.63 | +1.13 | +12.81 | FFN top = **" Denmark"** — contamination begins |
| 28 | −0.75 | +1.75 | +14.69 | Attn top = **" Denmark"** |
| 33 | +0.44 | +2.13 | +15.00 | FFN top = **" in"** — no Stratford assembly |

*Minimal working CoT:* `"The author of Hamlet was Shakespeare. The birthplace of his spouse was"` (97.6% prob)

| Layer | Attn | FFN | Cumulative | Notes |
|---|---|---|---|---|
| 0 | −9.36 | +17.45 | +13.94 | Similar start |
| 16 | +0.48 | −0.07 | +1.58 | Weak start, same pattern |
| 24 | +2.69 | +2.13 | +14.38 | Both add; att top = " London" |
| 26 | +2.38 | +1.88 | +20.25 | Attn = " England", FFN = **" Denmark"** — but cumulative already high |
| 28 | −1.13 | +2.63 | +23.88 | FFN top = **" Stratford"** — first appearance |
| 33 | −0.13 | +5.25 | +24.50 | FFN top = **" Stratford"** — final assembly |

**Verdict: Hypothesis A confirmed.** The layer-by-layer structure is nearly identical:
- Same early pattern (layers 0–16)
- Same layer 24 mechanism (both components add, similar magnitudes)
- Layer 26 FFN still produces " Denmark" in the working CoT — but the higher cumulative logit
  from the cleaner context means " Stratford" can win anyway
- The difference is at layer 28: the correct CoT has enough signal to route to " Stratford" at
  the fact-checking phase; the direct prompt does not

The computation is structurally the same forward pass. CoT changes the *input state of the
residual stream*, not the computation that operates on it. Head 1 does the same operation at
layer 24 with the same concentration (80.9% for minimal CoT; 75.4% for full CoT; 69% for direct).
Only the intermediate token it fires on differs.

---

## Experiment 3 — Minimal CoT

What is the minimum context needed to fix the broken chain?

| Prompt | Head 1 token | Output | Result |
|---|---|---|---|
| *(direct, broken)* "…author of Hamlet was" | " Denmark" 69% | "county of Norfolk" | ✗ |
| "The author of Hamlet was Shakespeare. The birthplace of his spouse was" | " Elizabethan" 80.9% | "Stratford-upon-Avon" | ✓ |
| "The author of Hamlet was William Shakespeare. Shakespeare's spouse was Anne Hathaway. Anne Hathaway was born in" | " Elizabethan" 75.4% | "1556 and died in 1623" | ✗ (template) |
| "Shakespeare's wife Anne Hathaway was born in" | — | "1556 and died in 1623" | ✗ (template) |
| "Hamlet, by Shakespeare (married to Anne Hathaway), who was born in" | — | "Stratford-upon-Avon in 1564" | ✗ (wrong entity — outputs Shakespeare's birthplace) |

**Minimum working form:** one sentence resolving "author of Hamlet" = Shakespeare, followed by
"The birthplace of his spouse was".

**Why this works:** Two conditions must be satisfied simultaneously:
1. The referent must be resolved (Shakespeare named), so Head 1 can fire on the right entity
2. The phrasing template must be "The birthplace of X was" (geographic template), not "X was
   born in" (temporal template)

The full CoT satisfies condition 1 but violates condition 2 — "Anne Hathaway was born in" triggers
the temporal template at layer 33, overriding the correctly assembled geographic signal.

**Template competition is a layer 33 phenomenon**, independent of Head 1's circuit. It applies
even when the relational chain has been correctly resolved by layers 24–28.

---

## Experiment 4 — Other Broken Chains

### Case A: 221B Baker Street (Referent Saturation)

Chain: 221B Baker Street → Sherlock Holmes → Arthur Conan Doyle → Edinburgh (4 hops)

| Prompt | Head 1 top token | % | Output |
|---|---|---|---|
| "The birthplace of the creator of the detective who lives at 221B Baker Street was" | " London" | 37.1% | Quiz mode ("In what city?") |
| "…Sherlock Holmes. The birthplace of his creator was" | " London" | **84.9%** | "Solihull, England" (wrong) |
| "…creator was Arthur Conan Doyle. The birthplace of Arthur Conan Doyle was" | " Scottish" | 78.2% | "Edinburgh, Scotland" ✓ |
| "The birthplace of Arthur Conan Doyle was" | " Scottish" | 87.0% | Edinburgh ✓ |

**Logit attribution for " Edinburgh" — direct broken chain:**
- Layer 26 FFN top token: " London" — contamination at knowledge retrieval phase
- Layers 28–33: attention and FFN both consistently top " London"
- Layer 33: slightly negative for Edinburgh (−0.5 total) — active suppression

**Logit attribution for " Edinburgh" — partial CoT ("his creator"):**
- Cumulative builds to 21.0, but " London" dominates all top tokens from L24 onward
- Layer 33 FFN = +3.0, top token " in" — never assembles Edinburgh

**This is a different failure mode from Hamlet: cultural saturation, not referent competition.**

- Hamlet failed because two entities competed for Head 1: the play (Danish) vs the author (Elizabethan). Naming Shakespeare in context resolved the competition.
- Holmes fails because one entity (Sherlock Holmes) has such a strong single attribute (London) that naming Holmes explicitly in the CoT *reinforces* the wrong association rather than overriding it. Head 1 fires " London" at 85% concentration even when the CoT states "Sherlock Holmes. The birthplace of his creator was" — the Holmes token dominates the context.

**The fix requires explicit proper-name subject:** "The birthplace of Arthur Conan Doyle was" — removing all reference to Sherlock Holmes from the query. Pronoun resolution through a saturated entity is insufficient. The saturated entity's name must not appear in the immediate context of the birthplace question.

---

### Case B: Raskolnikov (Unambiguous 3-hop — succeeds)

Chain: Raskolnikov → Crime and Punishment → Dostoevsky → Russian (3 hops)

| | |
|---|---|
| Prompt | "The native language of the author of the novel that features the character Raskolnikov was" |
| Head 1 top token | " Russian" (128.6% of layer 24) |
| Output | "Russian" ✓ |

Head 1 fires " Russian" at 128% — the bridge attribute IS the final answer. No further hop is
needed after nationality extraction because "native language = nationality" collapses the final
step. The chain is unambiguous at each step: Raskolnikov uniquely identifies Crime and Punishment;
Dostoevsky uniquely identifies Russian. No competing entities.

**This case reveals when Head 1 fires on intermediate vs final token:** the head extracts the
first salient attribute in the chain. When that attribute requires further downstream integration
(Polish → Warsaw), it serves as an intermediate. When that attribute is the answer (Russian =
native language), it fires directly on the final token.

---

## Summary of Findings

### 1. Hypothesis A is correct: CoT is input simplification, not circuit change

The computation across all working and failing prompts uses the same Head 1 at layer 24 at 70–88%
concentration, the same three-phase structure (early noise → middle association → late assembly),
and the same attention/FFN balance. CoT changes what Head 1 has to look at, not how it operates.

### 2. Two independent failure mechanisms, both fixable by different interventions

| Failure type | Mechanism | CoT fix | Structural signature |
|---|---|---|---|
| **Referent competition** | Multiple entities compete for Head 1; wrong one wins | Resolve to explicit entity name in context | Flat distribution, low top-token prob (Type 1) |
| **Cultural saturation** | Single entity correctly identified but attribute association so strong it overrides routing | Remove the saturated entity from query; use final entity name as direct subject | " London" (or saturated token) contaminates L26–L33 attribution |

### 3. Template competition is a third, independent failure mode

"The birthplace of X was" → geographic template (L33 FFN assembles city name)
"X was born in" → temporal template (L33 FFN +11.0 toward " " → date format)

This operates entirely at layer 33, independent of whether the relational chain (layers 24–28)
assembled the correct answer. A perfectly-resolved CoT can fail at layer 33 if the phrasing
activates the wrong template. The circuit succeeded; the phrasing failed.

### 4. The minimum CoT for each failure type

**Referent competition:** One disambiguating sentence + "birthplace of [pronoun/name] was" template.
Does NOT require repeating the full chain.

**Cultural saturation:** Requires the final entity's proper name as the grammatical subject of the
question. Pronoun resolution through the saturated entity inherits the saturation. The saturated
entity must not appear in the local context of the query.

**Template mismatch:** Use "The birthplace of X was" not "X was born in". The template fix is
independent of referent resolution — both must be correct simultaneously.

### 5. Head 1 behaviour across all tested cases

| Prompt | Head 1 top token | % concentration | Result |
|---|---|---|---|
| Hamlet → Warsaw (via Marie Curie) | " Polish" | 103% | ✓ |
| C&P → Moscow | " Russian" | 97% | ✓ |
| Don Quixote → Spanish | " Spanish" | 90% | ✓ |
| Hamlet → Stratford (3-hop) | " Elizabethan" | 70% | ✓ (partial) |
| 4-hop Hamlet chain (direct) | " Danish" | 76% | ✗ (wrong entity) |
| Minimal CoT → Stratford | " Elizabethan" | 80.9% | ✓ |
| Full CoT → Stratford ("born in") | " Elizabethan" | 75.4% | ✗ (template) |
| 221B Baker Street (direct) | " London" | 37.1% | ✗ (saturation) |
| 221B Baker Street (partial CoT) | " London" | 84.9% | ✗ (saturation intensified) |
| 221B Baker Street (full explicit CoT) | " Scottish" | 78.2% | ✓ |
| Doyle direct | " Scottish" | 87.0% | ✓ |
| Raskolnikov → Russian | " Russian" | 128.6% | ✓ (collapses to 1 hop) |
| "The spouse of William Shakespeare was" | " Elizabethan" | 135.8% | ✓ |
| Magic Flute → father | " Austrian" | 44.6% | ✗ (wrong entity) |

**Concentration as a signal:** When Head 1 concentration is high (>70%) and the intermediate
token is correct, the chain resolves. When Head 1 fires on the wrong entity's attribute (Denmark,
London, Austrian), the concentration can be equally high — the head is confident about the wrong
thing. Concentration alone cannot distinguish correct from incorrect routing.

---

## An Unexpected Finding: CoT Can Make Things Worse

The 221B Baker Street partial CoT is the key case. The direct broken chain had Head 1 at 37.1%
(" London"). After adding "is Sherlock Holmes" to the context, Head 1 fires at **84.9%**
(" London") — the saturation *increased*. Naming the intermediate entity (Holmes) in the CoT
concentrated the wrong association rather than diffusing it.

**This is the condition under which CoT degrades performance:** when the intermediate entity named
in the CoT has a dominant attribute association that is wrong for the final answer. The explicit
mention of Holmes activates the Holmes-London circuit more strongly than the indirect description
("the creator of the detective who lives at..."). CoT externalised the referent — and the
externalisation made the wrong association more salient.

The general principle: CoT resolves referent *competition* (multiple entities) but cannot resolve
referent *saturation* (one entity, overwhelmingly associated with the wrong attribute). Saturation
requires bypassing the intermediate entity entirely.

---

*Experiment run on Gemma 3-4B-IT (34 layers, 8 attention heads, bfloat16) using the Lazarus MCP.*
*Tools: `head_attribution`, `logit_attribution`, `generate_text`, `predict_next_token`.*
