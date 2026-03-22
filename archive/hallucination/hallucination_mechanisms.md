# Inside the Hallucinating Model
## Three Mechanistically Distinct Failure Modes in Gemma 3-4B-IT

*Mechanistic interpretability experiment using logit attribution, head attribution, and neuron analysis.*

---

## Background

A previous experiment on Gemma 3-4B-IT found an "embedding war" during translation: the
embedding starts with a strong prior for the correct token, attention systematically suppresses
it across all 34 layers, and the FFN fights back and barely wins. This raised a question worth
investigating directly:

**When the model gets something wrong, is it because the FFN lost the war?**

The answer turns out to be: *sometimes*, but the more interesting answer is that hallucination
is not one thing. It has at least three mechanistically distinct subtypes, only one of which
is detectable from output probabilities, and only one of which is the "FFN lost" story.

The experiment ran `logit_attribution` across ten prediction instances on Gemma 3-4B-IT
(34 layers, 2560 hidden dim, 8 attention heads, bfloat16), covering high-confidence correct
predictions, all three hallucination types, and an ambiguous completion. Head attribution and
neuron analysis were run at key layers to identify the specific circuits responsible.

---

## The Full Data

All values from `logit_attribution (normalized=true)`. Attn and FFN are net sums across
sampled layers (~18 of 34).

| Prompt → Target | Category | Embed | Attn | FFN | Final | Prob | \|Attn\|/FFN |
|---|---|--:|--:|--:|--:|--:|--:|
| capital of France → **Paris** | correct | +22.5 | −34.6 | +22.5 | +21.4 | 80.9% | 1.54 |
| Four score…our → **fathers** | correct | +47.8 | −59.0 | +28.9 | +32.0 | 97.7% | 2.04 |
| Marie Curie…city of → **Warsaw** | correct | −16.4 | **+8.7** | +41.1 | +27.0 | 98.4% | 0.21 |
| capital of Australia → **Canberra** | correct | −5.4 | −6.1 | +33.9 | +21.6 | 84.4% | 0.18 |
| capital of Australia → **Sydney** | wrong (2.2%) | +22.6 | −15.4 | +7.5 | +18.0 | 2.2% | 2.06 |
| "The Thanatos Syndrome" → **Robert** | hallucination — type 1 | +47.0 | −69.7 | +29.5 | +16.4 | 7.7% | 2.36 |
| 2009 Nobel Physics → **Andre** | hallucination — type 2 | +14.2 | −33.7 | **+51.1** | +37.8 | **94.1%** | 0.66 |
| "Correction" (1975) → **Toni** | hallucination — type 3 | ≈0 | **+13.1** | −6.3 | +21.0 | 40.8% | n/a† |
| She opened door and → **stepped** | ambiguous | +6.0 | −19.9 | +23.8 | +20.9 | 61.3% | 0.84 |

*† Type 3 has attention net-positive and FFN net-negative — the ratio is inverted.*

---

## Part I: Three Prediction Mechanisms

Before getting to hallucination, it's worth establishing that even correct predictions don't
all work the same way. Token rarity determines which mechanism fires.

### Mechanism 1 — The Embedding War

**Examples:** Paris, fathers
**Embedding logit:** high (+22 to +48)

Common tokens that fit the syntactic context ("X is ___") have a strong embedding prior —
they appear frequently in similar positions in the training corpus. The model must suppress
this default to route toward the contextually correct answer. Attention is net-negative,
suppressing the prior; FFN is net-positive, rebuilding the logit based on context.

The "war" is real and visible layer by layer. For *fathers* on the Gettysburg prompt, the
cumulative logit starts at +47.8, gets hammered to +7 by layer 8, slowly recovers through
the middle layers, and ends at +32.0 (97.7% confidence). Layer 0 alone contributes −46 from
attention — all 8 heads firing negatively, their top tokens pointing to garbage strings,
doing nothing but compressing the embedding prior.

The war balance ratio (|Attn| / FFN) measures how completely the FFN compensates:

- Paris: 1.54 — moderate war, FFN nearly matches the suppression
- Fathers: 2.04 — heavy war, but embedding was large enough to survive

### Mechanism 2 — Constructive Retrieval

**Examples:** Warsaw, Canberra
**Embedding logit:** low or negative (−16 to −5)

Rare proper nouns don't have a meaningful embedding prior for "X is ___" contexts.
The model can't fall back on frequency. Instead, attention and FFN *cooperate* to build the
prediction from near-zero.

Warsaw is the clearest case: net attention is **+8.7** — attention is actively routing toward
Warsaw, not suppressing it. This is the mechanistic signature of genuine knowledge retrieval.
The entire logit is constructed by the network from context, not recovered from a prior.

The war balance ratio is 0.21 — there is no war. Both sides are on the same team.

This matters for interpretation: constructive retrieval requires the model to actually *know*
the fact. The "fact-checking override" in late layers (28–33) is available as a second
correction pass if the early retrieval is wrong.

### Mechanism 3 — Contextual/Creative Generation

**Example:** stepped
**Embedding logit:** weak (+6)

Open-ended continuations have a weak embedding prior and are driven almost entirely by the
FFN (+23.8 net) reconstructing the narrative pattern. Attention mildly suppresses (+6 embed,
−19.9 net attention) but the FFN creates the prediction from learned continuation patterns.
No specific fact is being retrieved — this is the model's pattern-completion mode.

---

## Part II: Three Types of Hallucination

The central finding. Hallucination is not one mechanism — it has three distinct structural
signatures that call for different detection strategies.

### Type 1 — Uncertain Hallucination

**Example:** "The author of the 1987 novel *The Thanatos Syndrome* is" → **Robert** (7.7%)

*The Thanatos Syndrome* was written by Walker Percy. The model produces " Robert" via greedy
decode, but with only 7.7% confidence. The full top-15 distribution tells the real story:

```
Robert  7.7%   Stephen  6.0%   James  4.6%   John  4.4%
Peter   4.1%   Michael  4.1%   **      2.8%   Philip 2.3% ...
```

The model has no idea who wrote this book. It is sampling from a prior over plausible author
first names and greedy-decoding the mode of a nearly-flat distribution.

**Structural signature:**

```
Embed: +47.0 (high — " Robert" is common in "X is [author name]" contexts)
Attn:  −69.7 (most extreme suppression in the dataset)
FFN:   +29.5 (moderate — only generic "this is an author name" signal)
Final: +16.4 (7.7% probability)
|Attn|/FFN: 2.36
```

Attention is working correctly: it suppresses every candidate token uniformly because there is
no specific evidence to route to any one of them. The FFN provides only categorical signal —
"author name goes here" — with no specific person. The result is a diffuse distribution where
no token wins cleanly.

The hallucination fingerprint: layer 0 suppresses the +47 embedding to −4 (cumulative logit
goes negative before recovering), and the subsequent 33 layers gradually rebuild to only +16.
The model spends the entire forward pass trying to find a specific answer and never finds one.

**Detection:** Trivial. Top-token probability < 10% on a factual question is a reliable flag.

---

### Type 2 — Confident Confabulation

**Example:** "The 2009 Nobel Prize in Physics was awarded to" → **Andre** (94.1%)

Andre Geim won the Nobel Prize in Physics for graphene research — in 2010. The 2009 prize
went to Charles Kao (fiber optics) and Willard Boyle and George Smith (CCD sensor). The model
has stored a real fact but with a year-shift error.

The output probability is **94.1%**. The model is extremely confident. And it is wrong.

**Structural signature:**

```
Embed: +14.2 (moderate)
Attn:  −33.7 (suppression present)
FFN:   +51.1 (DOMINANT — nearly 1.5× the attention suppression)
Final: +37.8 (94.1% probability)
|Attn|/FFN: 0.66
```

The layer 33 FFN contributes **+20.1 logit**, with " Andre" as its top token. This is a
massive, targeted spike — the exact signature of the Warsaw and Canberra constructive
retrieval cases, which are correct at 98%+. The war balance ratio (0.66) falls squarely in
the "correct prediction" range.

**This is structurally identical to a correct high-confidence prediction. There is no
distinguishing feature in the output probability, the war balance, or the layer trajectory.**

The confabulation is a genuine retrieval — of a real person, a real Nobel Prize, a real
association — with an off-by-one year error. The FFN has stored "Andre Geim → Nobel Physics"
without the year constraint, and the year in the prompt doesn't override the retrieval.

**Detection:** Not possible from output probability alone. The structural tell is the
late-layer FFN spike — see Part V for the full dataset establishing the threshold. However,
Part V also reveals that **not all Type 2 confabulations produce a spike**. The spike is
specific to one submechanism (late-layer assertion), and a second submechanism (mid-layer
routing) produces wrong answers that are structurally indistinguishable from correct ones.

#### Type 2a — Late-Layer Assertion (the "spike" subtype)

The full answer is assembled at layer 33 FFN in a single massive positive contribution. The
cumulative logit is relatively low through the middle layers and then jumps sharply at layer
33. The model has a high-confidence person-prize association (e.g., "Eric Betzig → Nobel
Chemistry") stored with imprecise temporal or categorical binding. When the prompt's year or
category doesn't match, the binding fails silently and the late-layer FFN asserts its strongest
association anyway.

This subtype fires across Nobel Physics year-shifts (five examples), Nobel Chemistry year-shifts
(2013 → Eric Betzig, L33 FFN = +16.9), and Nobel Chemistry category bleeds (2016 → Yoshinori
Ohsumi who won Medicine, L33 FFN = +18.1). The common thread is a high-confidence
person-Nobel association with an unbound or misbound constraint.

#### Type 2b — Mid-Layer Routing (the "invisible" subtype)

**Example:** "The 43rd President of the United States was" → **Barack** (67.6%)

Barack Obama was the 44th President. The 43rd was George W. Bush. The model produces Barack
at 67.6% confidence, wrong — and with no layer 33 FFN spike whatsoever (L33 FFN = −1.0).

Attribution on both the confabulation (43rd) and the correct case (44th) for the same token
" Barack" shows near-identical layer profiles:

```
                   43rd (WRONG)   44th (correct)
Embed:             +14.6           +14.6
L24 cumulative:    +25.5           +23.6
L26 cumulative:    +35.0           +32.8
L28 cumulative:    +35.0           +32.0
L33 FFN:           −1.0            −0.6
Final:             +20.5           +20.9
```

The same token, built by the same mechanism, at nearly the same confidence. The model has no
ordinal-specific binding for Obama — it stores "Obama = President" and retrieves it equally
for 43rd and 44th. The ordinal number in the prompt does not constrain the retrieval.

Further evidence: for "The 41st President was" → " Bill" (75.4%, wrong — should be George H.W.
Bush), attribution shows the FFN retrieving " George" at layer 28 (top token "+4.6 logit") as
a competing correct signal — but the accumulated "Clinton" logit from layers 22–26 is too
large to overcome. The correct answer fights back mechanistically but loses.

**Detection of Type 2b:** Impossible from output probability or from layer 33 FFN. The model
genuinely doesn't have tight ordinal-to-president bindings. The only reliable detection is
external verification (check the adjacent ordinal) or training-time intervention.

---

### Type 3 — Attention-Driven Error

**Example:** "The author of the novel *Correction* published in 1975 was" → **Toni** (40.8%)

*Korrektur* (Correction) was written by Thomas Bernhard in 1975. The model produces Toni
Morrison — a different female literary author from the same era (she published *Sula* in 1973,
*Song of Solomon* in 1977, won the Nobel in 1993).

**Structural signature:**

```
Embed: ≈0 (near-zero — "Toni" has no default prior here)
Attn:  +13.1 (NET POSITIVE — attention PROMOTES the wrong answer)
FFN:   −6.3  (net negative — FFN weakly opposes)
Final: +21.0 (40.8% probability)
Dominant component: ATTENTION
```

This is the only case in the dataset where attention is net-positive *and wrong*. Attention
is routing toward a plausible associative cluster — literary novel, 1975, female author — and
latching onto Toni Morrison, a canonical member of that cluster. The FFN produces a weak
opposing signal, as if it has some registration that this is incorrect, but loses at layer 33
where the FFN's top token also becomes " Toni" (+7.4 logit), ceding the contest.

The early layers tell the story: the cumulative logit starts at −24 after layer 0 (both
attention and FFN oppose the token initially), then attention begins pulling it positive
through layers 2–8 (+8.2, +7.5, +1.9...) until the context resolves. By layer 33 both
components are aligned on the wrong answer.

**Detection:** Attention-dominant predictions on factual "The [property] of [entity] was"
frames are a structural flag. When `net_attention > net_ffn` on a factual retrieval prompt,
the prediction is relying on associative routing rather than stored knowledge. This is
measurable at inference time by comparing layer-level residual contributions.

---

### Summary

| Type | Mechanism | Prob | L33 FFN | Detection |
|---|---|---|---|---|
| 1 — Uncertain | FFN absent; attn suppresses everything | < 10% | small | Trivial (flat dist.) |
| 2a — Late assertion | FFN layer 33 asserts wrong fact | any | **> +16** | L33 FFN spike |
| 2b — Routing error | Mid-layer retrieval without ordinal binding | any | small | **Impossible** |
| 3 — Attention error | Attn routes to wrong associative cluster | ~40% | small | Attention-dominant flag |

The critical implication: a hallucination detector based only on output probability will catch
type 1 reliably, miss type 2 entirely, and catch type 3 partially. Types 2 and 3 require
mechanistic access — either full attribution or a proxy signal derived from layer-level
residual norms — to detect reliably.

---

## Part III: A Shared Neuron

During `top_neurons` analysis at layer 26, the same neuron appeared in both the Warsaw and
Canberra predictions despite their prompts being structurally different:

| Prompt | Neuron | Logit contribution | Top token |
|---|---|---|---|
| Marie Curie was born in the city of → Warsaw | **3593** | +0.011 | " magyar" |
| The capital city of Australia is → Canberra | **3593** | +0.055 | " magyar" |

Neuron 3593's top token — the vocabulary token its down-projection vector most points toward
— is " magyar", the Hungarian word for "Hungarian." It fires for Polish Warsaw and
Australian Canberra with the same top token. This could be a genuinely relational neuron
encoding something like "national capital of country" — a concept that, interestingly, might
be represented in the model's internal vocabulary via Hungarian rather than English. Or it
could be a coincidence: a neuron that incidentally activates on both prompts for unrelated
reasons.

Its contributions are small (0.011 and 0.055 logit against a total layer contribution of
+1.6 and +10.25 respectively). It would not survive a strict significance filter. But the
consistency of the top token across two structurally different prompts is suggestive.

A direct test was blocked by a tool limitation: `top_neurons` indexes into the intermediate
MLP dimension (10,240 neurons in SwiGLU's gate×up product), while `analyze_neuron` and
`discover_neurons` index into the hidden dimension (2,560). Neuron 3593 from `top_neurons`
is not the same neuron 3593 in `analyze_neuron`. The correct test — activating neuron 3593
across a diverse range of capital and non-capital prompts — requires accessing the intermediate
dimension directly, which the current tooling doesn't expose. Whether this is a genuine
capital-of-country neuron or an artifact remains open.

---

## Part IV: The Misconception Race

Running attribution for both Sydney and Canberra simultaneously on "The capital city of
Australia is" produces one of the clearest mechanistic pictures in the experiment.

```
                embed    L16     L22     L24     L26     L28     L32     L33
Canberra:       −5.4     3.3     9.4    20.6   [36.0]   35.0    18.8    21.6
Sydney:        +22.6     3.4    12.4    23.0   [36.0]   32.8    17.6    18.0
```

Sydney starts with a 28-point embedding advantage and leads from the beginning through
layers 16–26. This is the **misconception phase**: the common associative link
"Australia → Sydney" is encoded in the middle layers, and for a stretch it looks like Sydney
will win.

At layer 26 both tokens reach exactly **+36.0** — the model is tied.

Layers 28–33 are the **fact-checking phase**. They break the tie consistently for Canberra.
The final margin is ~3.6 logits, which corresponds to the 84.4% vs 2.2% confidence split.

The final arbiter is **layer 33, head 5**. Its own top token is " Canberra". Its contribution
(+2.97 logit) accounts for 228% of its layer's total — it more than compensates for the other
heads at that layer. Head 4 opposes at −1.98, top token " pyramid" (incoherent). The contest
is decided by a single head in the final layer.

This gives a mechanistic picture of how the model "knows better" than its own associations:
the common link is stored and activated (Sydney, layers 16–26), but specific factual knowledge
in late attention heads overrides it (Canberra, layers 28–33). The margin is not large.
Harder or less-trained facts would lose this late-layer contest.

### When the Fact-Checking Phase Fails

The Canberra case shows the late layers succeeding. The Nobel 2009 confabulation shows them
failing — or more precisely, doing exactly what they're designed to do, but on the wrong fact.

In the Canberra case, the late layers override an incorrect associative activation (Sydney)
with a correct stored fact (Canberra). The override mechanism is an attention head — layer 33,
head 5 — whose learned weight points specifically toward " Canberra" for this context.

In the Nobel 2009 case, the late layers don't encounter an associative activation to override.
There is no "Sydney-equivalent" competing answer trying to win. Instead, the late-layer FFN
simply asserts " Andre" with enormous confidence (+20.1 logit at layer 33). The fact-checking
phase is running, but what it retrieves is the wrong year's laureate. The confabulation *is*
the late-layer retrieval — not a failure to correct, but a confident retrieval of a
mislabeled memory.

This suggests the late-layer circuits (28–33) in Gemma serve two distinct roles:

1. **Override**: When a strong associative prior has accumulated from middle layers, late
   attention heads can assert a competing specific fact. Success depends on whether a
   specific-enough weight exists (head 5 for Canberra) and whether it's stronger than the
   accumulated prior (it was, barely — 3.6 logit margin).

2. **Assert**: When the middle layers produce a weak or diffuse signal, late FFN layers
   fire their strongest stored association for the context. If that association is correct
   (Warsaw: +5.9 at layer 33), confidence is warranted. If that association is wrong
   (Nobel 2009: +20.1 at layer 33), the confabulation is expressed at maximum confidence.

The difference between a Canberra-style correct answer and a Nobel-style confabulation is
not detectable from the structure of the late-layer spike. Both look like confident retrieval.
What differs is whether the retrieved association was accurately stored in training — which
is not mechanistically visible at inference time.

---

## Part V: The Layer 33 FFN Spike as a Confabulation Signal

The confabulation detection problem (Type 2) appeared to have one structural tell: the Nobel
2009 → Andre case showed a layer 33 FFN contribution of **+20.1 logit**, versus +1.6 and +5.9
for correct predictions (Canberra and Warsaw). To test whether this spike magnitude reliably
separates confabulations from correct retrievals, attribution was run on 23 predictions: 18
confirmed correct and 5 confirmed confabulations.

### The Dataset

**Correct predictions — layer 33 FFN contribution:**

| Prompt → Target | Prob | L33 FFN |
|---|---|---|
| capital of France → **Paris** | 80.9% | −1.1 |
| Four score…our → **fathers** | 97.7% | +8.1 |
| Marie Curie born in city of → **Warsaw** | 98.4% | +5.9 |
| capital of Australia → **Canberra** | 84.4% | +1.6 |
| 2010 World Cup winner → **Spain** | 61.7% | +6.0 |
| 2006 World Cup winner → **Italy** | 55.1% | +3.6 |
| capital of South Korea → **Seoul** | 78.1% | +1.75 |
| capital of Egypt → **Cairo** | 85.2% | +1.6 |
| capital of Japan → **Tokyo** | 83.2% | −0.25 |
| capital of Germany → **Berlin** | 89.1% | −0.25 |
| capital of Canada → **Ottawa** | 83.6% | +1.25 |
| capital of China → **Beijing** | 83.6% | +1.75 |
| capital of Mexico → **Mexico** [City] | 20.9% | +2.4 |
| capital of India → **New** [Delhi] | 80.5% | 0.0 |
| capital of Argentina → **Buenos** [Aires] | 63.7% | −1.0 |
| capital of South Korea → **Seoul** | 78.1% | +1.75 |
| capital of Italy → **Rome** | 64.5% | −0.5 |
| capital of Spain → **Madrid** | 7.6% | +2.0 |
| capital of Russia → **Moscow** | 68.4% | −0.5 |

**Correct prediction range: −1.1 to +8.1 logit. Median: ~+1.6**

**Confabulations — layer 33 FFN contribution:**

| Prompt → Target | Prob | L33 FFN | Actual winner |
|---|---|---|---|
| 2009 Nobel Physics → **Andre** [Geim] | 94.1% | +20.1 | Kao, Boyle, Smith |
| 2010 Nobel Physics → **Alain** [Aspect] | 28.3% | +19.1 | Geim & Novoselov |
| 2011 Nobel Physics → **Leon** [Lederman] | 36.1% | +16.5 | Perlmutter, Schmidt, Riess |
| 2013 Nobel Physics → **Saul** [Perlmutter] | 64.8% | +16.0 | Higgs & Englert |
| 2017 Nobel Physics → **Arthur** [Ashkin] | 86.3% | +20.25 | Weiss, Barish, Thorne |

**Confabulation range: +16.0 to +20.25 logit. Median: ~+19.1**

### Extended Dataset (including cross-domain confabulations)

**Additional confabulations tested:**

| Prompt → Target | Type | Prob | L33 FFN |
|---|---|---|---|
| 2013 Nobel Chemistry → **Eric** [Betzig] | Year-shift (Chem) | 93.4% | **+16.875** |
| 2016 Nobel Chemistry → **Yosh**[inori Ohsumi] | Category bleed | 67.2% | **+18.125** |
| 43rd President → **Barack** [Obama] | Ordinal off-by-one | 67.6% | **−1.0** |
| 41st President → **Bill** [Clinton] | Ordinal off-by-one | 75.4% | **+0.75** |

**Controlled comparisons (same token, correct context):**

| Prompt → Target | Correct? | Prob | L33 FFN |
|---|---|---|---|
| 44th President → **Barack** | Yes | 68.0% | −0.625 |
| 42nd President → **Bill** | Yes | 48.8% | +0.75 |

### The Result — Threshold Holds for One Submechanism, Not Both

```
Correct predictions:           max = +8.1    (n=19)
Type 2a (late assertion):      min = +16.0   (n=7: 5 Physics + 2 Chemistry)
Type 2b (ordinal routing):     max = +0.75   (n=2 — IN THE CORRECT RANGE)

Nobel year-shift threshold ~+12: separates Type 2a from everything else with zero errors.
President ordinal: confabulation is INDISTINGUISHABLE from correct prediction.
```

The threshold generalizes across Nobel domains (Physics and Chemistry) and across confabulation
subtypes within that domain (year-shift AND category bleed both spike). But it fails completely
for president ordinal confabulations, which show the same layer 33 FFN profile as correct
predictions.

### Two Distinct Submechanisms

The data reveals that "confident confabulation" is not one mechanism:

**Type 2a — Late-layer assertion:** The cumulative logit is low through the middle layers,
then a single massive FFN contribution at layer 33 asserts the answer. The spike (+16 to +20)
is detectable. This fires when the model has a strong person-domain association (Betzig =
Nobel Chemistry, Ohsumi = Nobel 2016) that is retrieved without properly binding the
year/category constraint. The constraint in the prompt is not strong enough to override the
stored association at inference time.

**Type 2b — Mid-layer routing:** The wrong answer builds up through layers 20–26 via normal
constructive retrieval, identical to how correct answers are built. No late spike. The model
simply lacks precise ordinal-to-president bindings — "Obama = President" is stored without
the ordinal "44th", so the ordinal in the prompt doesn't discriminate. For the 41st president,
the FFN at layer 28 even retrieves "George" (the correct answer) as a competing signal, but
the accumulated wrong logit from earlier layers is too large to be overridden.

### What the Spike Represents

The spike measures *where in the network the final answer confidence is assembled*. In correct
predictions and in Type 2b confabulations, the answer resolves in layers 24–26 and layer 33
merely trims. In Type 2a confabulations, layer 33 FFN is doing the primary work — it is the
source of the answer, not the finishing pass.

This maps onto a distinction in memory structure: Type 2a involves associations with imprecise
temporal/categorical bindings that "leak" across years or categories; Type 2b involves
associations that were never bound to the indexing dimension at all (no ordinal binding).
Late-layer assertion is the signature of a leaky binding; no spike is the signature of an
absent binding.

### Scope of the Detector

The layer 33 FFN spike at threshold ~+12 reliably detects **Type 2a confabulations only** —
cases where the model has a specific high-confidence person-attribute association with an
imprecise constraint. Confirmed across 7 examples (5 Nobel Physics, 2 Nobel Chemistry) with
no false positives in 19 correct predictions and no false negatives on this subtype.

It does **not** detect Type 2b (absent binding) confabulations. Those require external
verification to catch.

---

## Part VI: Capital Neurons Are Frame Detectors

`discover_neurons` at layer 26, comparing "The capital of X is" prompts against a diverse
set of other prompts, identified neurons 1948, 1197, and 445 as top discriminators. The
expectation was that these are "capital city knowledge" neurons. They are not.

Per-prompt activation analysis reveals they detect **prompt structure**, not semantic content:

| Prompt | N-1948 | N-1197 | N-445 |
|---|--:|--:|--:|
| The capital of France is | −63 | +56 | +50 |
| The capital of Japan is | −72 | +52 | +56 |
| The capital of Germany is | −59 | +47 | +50 |
| The capital city of Australia is | −56 | +34 | +49 |
| The capital of Canada is | −60 | +79 | +70 |
| The capital of Brazil is | −70 | +67 | +64 |
| **Marie Curie was born in the city of** | −22 | **−59** | **−12** |
| The author of Hamlet was | +14 | −2 | −3 |
| She opened the door and | +26 | +1 | +14 |
| Thanatos Syndrome is | +9 | +16 | +16 |
| 2009 Nobel Prize in Physics | +30 | −27 | −32 |

The Marie Curie → Warsaw prompt predicts a national capital correctly at 98.4%. Neuron 1197
fires at −59 — the most anti-capital value in the table. These neurons are not tracking
whether the answer will be a capital; they are tracking whether the question *asks for* a
capital.

Real geographic knowledge lives elsewhere: in attention head 1 at layer 24 (top token
" Polish", contributing +4.9 logit — 103% of the layer's total attribution for Warsaw), and
in distributed late-layer FFN activations. The frame-detector neurons are probably useful for
the model's syntactic processing of the query type, but they carry no information about
whether the retrieval will be correct.

This is a useful methodological note: `discover_neurons` finds what's *different* between
prompt classes. For two prompt classes that differ primarily in surface form ("The capital of
X is" vs everything else), it finds surface-form detectors. Finding knowledge neurons requires
a more carefully constructed contrastive set — same surface form, varying factual content.

---

## Part VII: What This Means for Hallucination Detection

The practical upshot of three mechanistically distinct hallucination types:

**On the easy end:** Type 1 (uncertain) is already detectable. Check whether the top-token
probability on a factual question falls below ~15% and the distribution is spread across a
semantic category (all author names, all scientist names). No mechanistic analysis needed.

**On the hard end:** Type 2 confabulation splits into two distinct submechanisms with
different detection profiles:

**Type 2a (late-layer assertion):** Detectable via the layer 33 FFN spike. Threshold ~+12
separates 7 confabulations (Nobel Physics and Chemistry, both year-shift and category bleed)
from 19 correct predictions with zero errors. The spike indicates the final answer was
assembled at layer 33 rather than arriving from layers 24–26 as in correct predictions. Not
probability-mediated — low-confidence (28%) and high-confidence (94%) confabulations both
spike equally.

*Practical proxy*: monitor (layer 33 FFN residual norm) / (cumulative residual norm at
layer 32). A ratio spike without being in the correct-prediction range flags the pattern
without full attribution.

**Type 2b (mid-layer routing / absent binding):** Not detectable from layer 33 FFN or from
output probability. The model lacks precise ordinal-to-answer bindings (e.g., no "Obama = 44th
specifically"), so ordinal prompts retrieve the associated answer via normal mid-layer
constructive retrieval — producing a layer profile identical to a correct prediction. The
confabulation (43rd → Barack, L33 FFN = −1.0) and the correct case (44th → Barack, L33 FFN =
−0.6) are mechanistically the same forward pass.

*Detection requires external knowledge*: cross-validate with adjacent ordinals/years or use
a retrieval-augmented system. There is no internal signal to exploit.

Additional heuristics applicable to both subtypes:
- Cross-validate with temporal/ordinal neighbors ("who won in 2010?", "who was 42nd?")
- For claims involving specific indexed facts (year, ordinal, term number), treat absence
  of tight constraint as a known model failure mode

**In the middle:** Type 3 (attention error) is structurally flaggable. When attention is the
dominant net-positive component on a factual retrieval prompt, the prediction is driven by
associative routing rather than stored knowledge. This is computable from a lightweight probe
on intermediate-layer residual norms without full attribution.

---

## Open Questions

**Is neuron 3593 a genuine capital-of-country relational neuron?**
It fires for both Warsaw and Canberra at layer 26, with " magyar" as its top token in both
cases. A direct test would activate it across a range of capital prompts (London, Tokyo,
Washington, Beijing), non-capital city prompts (Sydney, New York, Shanghai), and control
prompts with no city at all, then check whether activation correlates with the capital
relationship rather than with city mentions generally. The blocker is a tooling issue:
`top_neurons` and `analyze_neuron` use different index spaces (10,240-dim intermediate vs
2,560-dim hidden). The test requires access to the intermediate dimension directly.

**What determines whether the late-layer fact-checking phase succeeds or fails?**
Canberra shows the phase succeeding: layer 33 head 5 asserts the correct answer and
overrides the mid-layer Sydney misconception. Nobel 2009 shows it failing: the same phase
asserts the wrong answer with even higher confidence (+20.1 logit). The question is whether
there is a measurable precursor — in the trajectory of cumulative logits, in the distribution
of mid-layer attention heads, or in the FFN activation patterns before layer 28 — that
predicts which outcome will occur. A systematic test would take matched pairs of correct
predictions and year-adjacent confabulations and compare their layer 0–27 profiles.

**Do the three hallucination types hold up across more examples, or are there further subtypes?**
The type 2 dataset was extended to 5 examples (all Nobel Physics year-shift errors), and the
layer 33 FFN spike pattern held consistently. Type 1 (uncertain) is the most likely to
replicate cleanly, because its signature — diffuse distribution over a semantic category,
high |Attn|/FFN ratio — is robust to the specific domain. Type 2 (confabulation) probably
has substructure: year-shift errors (Nobel 2009) may look different from geographic
confusions ("The Amazon river flows through Argentina"), category errors ("Darwin's famous
book was *The Descent of Man*"), or person-attribute confusions ("Einstein was awarded the
Nobel for relativity"). Type 3 (attention error) is the most fragile hypothesis, resting on
one example. A second instance where attention promotes a wrong associative cluster while FFN
weakly opposes would confirm the signature; a counter-example where FFN is positive and
dominant would complicate it.

**What does the correct answer look like for the Correction/Toni case?**
We ran attribution for " Toni" (Morrison) but not for " Thomas" (Bernhard). If the FFN has
some representation of the correct author, it should appear as a weak positive contribution
on " Thomas" that attention overrides. If the FFN has no representation at all — because
Thomas Bernhard is too obscure for Gemma 3-4B-IT's training data — then " Thomas" would show
a purely constructive retrieval pattern starting from near-zero, and the competition between
" Toni" and " Thomas" would be between associative routing (attention-driven, wrong) and
absence-of-signal (FFN-absent, correct-but-unlucky). That would be a fourth subtype: *the
model doesn't know, but picks the wrong associative answer over the correct unknown.*

**Can a lightweight probe on intermediate residuals flag hallucination type in real time?**
Layer 16 is where common associative knowledge activates. A probe trained on layer 16
activations with labels (confident-correct, type-1, type-2, type-3) could serve as a
generation monitor. The hypothesis is that by layer 16 the trajectory is set: type 1 has
already produced a flat residual stream with no specific factual activation; type 2 has
already loaded the wrong specific fact into the residual; type 3 has already committed to an
associative cluster. If this is true, a linear probe at layer 16 could flag predicted
hallucination type before the late layers express it — without requiring full attribution.

---

## Part VIII: Head 1, Layer 24 — A Dedicated Relational Bridge Circuit

A secondary experiment tested whether the layer 24 attention finding from the Warsaw case
generalises. The hypothesis: attention heads fire on *intermediate* tokens in a relational
chain rather than the final answer, performing a hop rather than a lookup.

The test set: three prompts requiring a two-step relational inference, plus one requiring
three steps. Head attribution was run at layer 24 for each.

### The Data

| Prompt | Chain | Head 1 top token | Head 1 % of layer |
|---|---|---|---|
| Marie Curie was born in the city of → **Warsaw** | Curie → Polish → Warsaw | " Polish" | 103% |
| The author of Crime and Punishment was born in → **Moscow** | C&P → Dostoevsky → Russian → Moscow | " Russian" | 97% |
| The native language of the author of Don Quixote is → **Spanish** | Quixote → Cervantes → Spanish | " Spanish" | 90% |
| Hamlet was written by a playwright who was born in → **Stratford** | Hamlet → Elizabethan → Shakespeare → Stratford | " Elizabethan" | 70% |

**The same head (Head 1) at the same layer (24), every time. Contribution: 90–103% of the
layer's total attribution.**

Head 1's top token is never the final answer. It is always the **most prominent cultural or
contextual attribute** of the entity that serves as a bridge toward the answer:

- Marie Curie → **Polish** (nationality) → Warsaw
- Dostoevsky → **Russian** (nationality) → Moscow
- Cervantes → **Spanish** (nationality = answer for this question)
- Shakespeare → **Elizabethan** (era) → [English] → Stratford

The token for Shakespeare is " Elizabethan" rather than " English" because in the context of
"playwright," Elizabethan playwright is the stronger training co-occurrence. The head finds
the highest-salience attribute of the entity for the current context — not a fixed
nationality lookup, but a contextual attribute extractor that happens to fire on nationality
most of the time because nationality is usually the strongest relational bridge.

### The Hamlet Chain — Three Hops Across Nine Layers

The Hamlet case reveals that multi-hop inference can be distributed across multiple layers
and multiple components:

```
Layer 24, Head 1 (attn):   top token " Elizabethan"  → era-of-playwright bridge
Layer 24, FFN:              top token " Shakespeare"  → author retrieval (parallel to Head 1)
Layer 26, Head 2 (attn):   top token " Hamlet"        → routes stored birthplace info
                             from the "Hamlet" token position back to last position
Layer 33, Head 5 (attn):   top token " Shakespeare"  → author still present at final layer
Layer 33, FFN:              +5.125 logit, top token " Stratford" → final assembly
```

" Stratford" does not appear in logit space until layer 33 FFN. The answer is never resolved
in the middle layers — it requires all three intermediate representations (Elizabethan,
Hamlet, Shakespeare) to be integrated before the FFN can produce the specific city.

### What This Reveals

**Head 1 at layer 24 is not a nationality head — it is a contextual attribute bridge.** It
fires on whichever attribute of the entity-in-context is the strongest intermediate link
toward the expected answer type. For person-birthplace questions, that attribute is usually
nationality. For works-of-art questions, it may be the era, the language, or the author.

**The hop is not single-layer magic.** The Warsaw case looked like a two-step hop resolved
in one layer. The Hamlet case shows that harder inference chains are distributed: layer 24
finds the cultural intermediate, layer 26 routes the source-text association, layer 33
integrates both into the final answer. The depth of inference correlates with how many
separate associations need to be composed.

**The intermediate token in Head 1 is a prediction, not a lookup key.** The head doesn't
look up "Marie Curie → Polish" as a stored pair. Its output in vocabulary space *predicts*
"Polish" as the most likely next token given the residual stream at that position — and this
prediction, injected back into the residual stream, provides the information that later
layers use to route toward Warsaw. The hop mechanism is: predict an intermediate, inject it
as signal, let downstream layers use it.

This is qualitatively different from the FFN's retrieval mechanism (pattern-match key →
retrieve associated value). Attention heads do *relational prediction* — they predict what
would follow from the context, and that prediction serves as an intermediate reasoning step.

### The Failure Boundary — Entity Attribution Ambiguity

The relational bridge circuit has a characteristic failure mode, and it is not what a
hop-depth account would predict. Testing a 4-hop chain reveals that the circuit breaks on
*entity ambiguity*, not *chain length*.

**Test prompt:** "The birthplace of the spouse of the author of Hamlet was the village of"

Required chain: Hamlet → (author) Shakespeare → (spouse) Anne Hathaway → (birthplace) Shottery

**Result:** The model produces "Elmswell, Suffolk" — entirely wrong. Top token " El" at 5.1%
probability. The distribution is flat (Type 1 signature): " Stratford" appears at 2.3%,
"Shottery" is not in the top candidates. Head attribution at layer 24:

| Head | Top token | % of layer attribution |
|---|---|---|
| Head 1 | **" Danish"** | 76% |
| Head 4 | " Shakespeare" | 21% |
| Others | noise | ~3% |

Head 1 fired on " Danish" — the **thematic setting of Hamlet the play** — rather than the
nationality of its author. The chain required Head 1 to resolve "the author of Hamlet" as
an entity (Shakespeare = English/Elizabethan), but the 4-hop framing created referent
competition: "Hamlet" activates both Hamlet-the-play (strongly associated with Denmark/Danish)
and the-author-of-Hamlet (associated with Elizabethan/English). Head 1 extracted the wrong
entity's attribute.

The 3-hop Hamlet case (Part VIII data table) had no such ambiguity: "Hamlet was written by
a playwright who was born in" points unambiguously to the author. Head 1 fired " Elizabethan"
at 70% concentration and the chain resolved partially (though requiring 9 layers of distributed
integration). The 4-hop prompt didn't make the chain longer — it introduced a competing referent
that captured Head 1 before it could resolve the right entity.

**The failure mode is referent competition, not hop depth.** The circuit can navigate
3-hop chains when entity references are unambiguous. It fails on 4-hop chains when an
intermediate clause activates two competing entity readings and Head 1 latches onto the
wrong one. At that point the injected intermediate (" Danish") is wrong and all downstream
layers inherit the contaminated signal — the chain collapses to a flat distribution.

**Contrast — Crime and Punishment → Russian Ruble:**

To probe whether 4-hop chains can ever succeed, the prompt "The currency used in the setting
of the novel Crime and Punishment is the" was tested.

Required chain (apparent): C&P → Dostoevsky → Russia → Russian → Ruble

Result: " Russian" at 59%. Head 1 fires " Russian" at 122% of layer 24 attribution. But
this is not a successful 4-hop — the chain effectively collapsed to 2 hops. " Russian" is
the model's top prediction for the question (not " Ruble"), because "Russian" co-occurs with
both Dostoevsky and "Russian currency" in the training distribution. The late-layer FFN
(layer 33 FFN = −4.0) was actively trying to push toward " Ruble" against the accumulated
" Russian" signal, but couldn't overcome it. The model produced the intermediate step as
its output, not the final answer.

This demonstrates a second failure mode distinct from referent competition: **chain
short-circuiting**, where a salient intermediate token accumulates too much logit mass to
be overridden by the final-step FFN retrieval.

**Summary of circuit failure modes:**

| Failure mode | Mechanism | Signature |
|---|---|---|
| Referent competition | Head 1 misfires on wrong entity's attribute | Flat output distribution, wrong intermediate |
| Chain short-circuiting | Intermediate logit too large for final-step FFN to override | Intermediate token as output (e.g., "Russian" instead of "Ruble") |
| (No failure — 3-hop) | Unambiguous entity, distributed integration across 9 layers | Final token assembled at layer 33 FFN |

The mechanistic limit of the relational bridge circuit is not a specific hop count. It is
the **capacity for entity disambiguation under referent competition**. Head 1 at layer 24
is a contextual attribute extractor, not a full entity resolution system — and it has only
one shot. Once it fires on the wrong intermediate, there is no correction mechanism; the
wrong signal propagates through all subsequent layers.

---

## Conclusion

The embedding war framing — attention vs. FFN fighting over the correct token — is real but
describes only a subset of predictions, and a subset of hallucinations. The more complete
picture has three correct-prediction regimes determined by token rarity, and three
hallucination regimes determined by *why* the model doesn't know or doesn't know correctly.

The most important finding for practical systems is the existence of Type 2 confabulation: a
model can be 94% confident, produce a fluent specific answer, and be mechanistically similar
to a correct high-confidence prediction. The error is in the stored association, not the
retrieval process. The mechanism succeeded — it retrieved the wrong thing.

The follow-up experiment refines this. Type 2 splits into two submechanisms. Type 2a
(late-layer assertion) is detectable via the layer 33 FFN spike: this fires for Nobel laureate
confabulations across both Physics and Chemistry, both year-shifts and category bleeds (7/7
examples show spikes >+16; 19/19 correct predictions show ≤+8.1). The spike identifies
confabulations where a high-confidence person-domain association is retrieved without correctly
binding the temporal or categorical constraint. Type 2b (mid-layer routing) is not detectable
— the model lacks precise ordinal-to-answer bindings, and the wrong answer emerges via the
same constructive mechanism as a correct answer. The two president ordinal confabulations
(43rd → Obama, L33 FFN = −1.0; 41st → Clinton, L33 FFN = +0.75) are mechanistically identical
to their correct-context counterparts.

The layer 33 FFN spike reliably detects one important class of confident confabulation, not
all. The distinction is whether the error reflects a *leaky binding* (Type 2a, detectable: the
constraint was stored but imprecisely, and fires under the wrong year/category) or an *absent
binding* (Type 2b, undetectable: the constraint was never stored at all, so no internal signal
encodes the mismatch).

The relational bridge circuit (Head 1, layer 24) shows a structural limit that has nothing
to do with hop count and everything to do with entity disambiguation. The circuit can
navigate 3-hop chains when entity references are unambiguous, distributing work across
multiple layers. It fails on 4-hop chains when the prompt introduces referent competition
between two entities that share a strong contextual attribute. Head 1 has one shot at
extracting the contextual bridge — when it fires on the wrong entity's attribute, the
contaminated signal propagates through all subsequent layers with no correction mechanism.
A second failure mode (chain short-circuiting) occurs when a salient intermediate token
accumulates too much logit mass for the final-step FFN to override, producing the
intermediate as the output rather than the answer.

The finding about frame-detector neurons adds a methodological caution: contrastive neuron
searches return what distinguishes the prompt classes, not what encodes the knowledge they
elicit. Real knowledge neurons require a contrastive set that holds surface form constant
while varying factual content. The distinction matters for any interpretability work that
tries to localize "where facts are stored."

---

*Experiment run on Gemma 3-4B-IT using the Lazarus MCP interpretability server.*
*Tools used: `logit_attribution`, `head_attribution`, `top_neurons`, `discover_neurons`,*
*`analyze_neuron`, `predict_next_token`, `generate_text`.*
*Note: `top_neurons` uses the intermediate MLP dimension (10,240); `analyze_neuron` and*
*`discover_neurons` use the hidden dimension (2,560) — these are different index spaces.*
