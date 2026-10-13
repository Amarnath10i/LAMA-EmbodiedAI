# LAMA — Lifelong Affordance learning through counterfactual verificAtion

An embodied agent that learns what objects afford by **imagining** possible
interactions inside a world model, **verifying** only the hypotheses it is most
uncertain about through real interaction, and **keeping** only what verification
confirms.

## The research question

> Given a limited interaction budget, can an agent choose which of its imagined
> affordance hypotheses are worth testing in reality, and accumulate a knowledge
> base whose entries are both correct and durable?

The claim is about **interaction efficiency and knowledge retention**, not raw
task success. An agent that tries everything will eventually learn everything;
the question is whether targeted counterfactual verification gets there on a
fraction of the interactions, and whether what it stores stays right.

## The loop

```
observe
   |
   v
imagine        for each visible object, roll out every candidate action
   |           inside the world model -> predicted outcome + uncertainty
   v
select         rank hypotheses by epistemic uncertainty / expected information
   |           gain; pick the ones worth spending a real interaction on
   v
verify         execute the interaction; compare reality to the prediction
   |
   v
adjudicate     confirmed / refuted / inconclusive
   |
   v
remember       write only confirmed knowledge to the affordance bank, with
   |           provenance and confidence; revise on contradiction
   v
plan           downstream tasks consult the bank instead of re-deriving
```

The hard part is `select`. Uncertainty alone is not enough: some outcomes are
irreducibly random, and an agent that cannot separate "I do not know" from
"nothing could know" will burn its budget on noise. Making that distinction pay
off is the contribution.

## Status

Early. The environment is the current work.

| Component | State |
|---|---|
| Environment (hidden affordances, object-directed actions) | working |
| Perception encoder | not started |
| World model with calibrated uncertainty | not started (bank doubles as one, see below) |
| Counterfactual imagination | working |
| Verification selection | working |
| Verification execution and adjudication | working |
| Lifelong affordance bank | working |
| Planner over confirmed knowledge | working (backward-chaining regression search) |
| Evaluation harness and baselines | working |

## The environment

A numpy warehouse of objects with hidden mass and hidden affordance profiles.
The agent sees geometry and a noisy appearance descriptor; it never sees an
object's kind, and object ids are opaque.

Twelve verbs: `approach, push, pull, lift, press, rotate, open, close, grasp,
release, place_on, tip`. Each costs budget, including when it fails.

Doors start locked, so they open only through a mechanism. Buttons, levers and
valves latch. The pressure plate does not: standing on it opens its door only
while you stand there, which is useless when you need to walk through. Putting
something heavy on it opens the door for good -- an affordance that cannot be
found by interacting with the plate alone.

Which brings in the trap. A `crate` and a `block` have **identical**
appearance descriptors. The block holds the plate down; the crate is too light.
Nothing the agent can see will separate them, and their one observable
difference is that the crate moves further when shoved. Two more pairs
(`lever`/`switch`, `barrel`/`drum`) sit at about 15% confusion, so appearance
is a useful prior there but not a reliable one.

About 2,600 interactions per second on one core, so 10^5 interactions takes
under a minute. That number is the reason this backend exists rather than a
physics one: the prior work in `files/EmbodiedAI.zip` produced no signal at 186
interaction records and concluded volume was the bottleneck.

A second backend, `lama/env/isaac_warehouse.py`, implements the exact same
contract on top of Isaac Lab -- real PhysX rigid bodies, so hidden mass drives
push/pull/rotate/tip displacement through actual physics rather than a
formula. It was written unverified (no RTX-class GPU was available while
writing it) and has since run on the target RTX 4060 machine -- `outputs/`
holds real per-episode results from 10- and 50-episode runs against it,
strong evidence it works, though that data predates the fixes in
`docs/DECISIONS.md` D7, so it confirms the backend runs, not the current code
specifically. `scripts/isaac_smoke_test.py` and `scripts/run_lama_isaac.py`
are the way to check the code as it stands now.

## Memory

The agent never observes an object's kind, and object ids reset every episode
-- so lifelong memory cannot be keyed on either. Instead:

- **Concepts** (`lama/memory/concepts.py`) form online from appearance alone:
  each observation merges into the nearest existing concept if close enough,
  or starts a new one. This is the agent's own stand-in for "kind", built
  entirely from what it can see, and it inherits the catalogue's traps: the
  crate and block are visually identical, so they are guaranteed to share one
  concept, forever.
- **The affordance bank** (`lama/memory/bank.py`) holds a Bayesian belief per
  `(concept, verb, tool concept)`, calibrated so roughly 8 clean successes
  confirm a reliable affordance, roughly 27 clean failures refute one, and a
  genuinely 50%-reliable affordance still confirms once enough trials pin its
  rate down. Only `bank.confirmed()` is meant to be acted on; everything else
  stays internal as working evidence.

Run end to end against the real environment -- fetch a crate half the time and
a block the other half, place it on the plate, log what happens -- the bank
settles on exactly the honest answer: confident that placing something on the
plate works, at a remote-effect (door-opens) rate of about 50%. It cannot do
better than that, because appearance genuinely cannot tell the two apart, and
the numbers say so instead of hiding it.

That is not the end of the story, though. A `STUCK` status fires when a
belief's continuous evidence (how far a push actually moved something) looks
bimodal -- the statistical signature of two blended kinds -- and
`ConceptCodebook.split_concept` then tries to separate them. For crate/block
specifically (identical appearance, zero true separation) that stays at
chance, honestly: no algorithm recovers a signal that is not there. For the
catalogue's other two look-alike pairs, `lever`/`switch` and `barrel`/`drum`
(small but nonzero true separation), splitting measurably works -- about
73/27 and 88/12 correct sorting in testing, up from 50/50.

## Imagination and verification

The loop from observation to memory is closed and runs end to end:

- **Imagine** (`lama/imagination/`) enumerates every verb attemptable on every
  reachable object right now, and attaches whatever the bank already believes
  about it -- the bank itself stands in for a world model, since a
  nearest-concept Beta posterior is already a real prediction with a real
  uncertainty. Only *observable* preconditions (gripper state) gate what gets
  proposed; hidden ones are never consulted here.
- **Select** (`lama/verification/select.py`) scores each hypothesis by how
  much a real test would still teach the bank, divided by what it costs.
  Settled beliefs score zero, so budget stops going toward what is already
  known.
- **Verify and remember** (`lama/verification/loop.py`) turns the top choice
  into a real environment step and folds the outcome back into memory.

**A known, measured limitation.** Refuting a belief takes about 27 clean
failures (see the bank's calibration), and the acquisition function scores
per key, not across the whole reachable set. In practice this means a cheap,
persistently-failing verb on a single object can consume most of an episode's
budget before the loop ever moves on -- confirmed by a pinned regression test,
not just suspected. This is no longer papered over by a dummy fallback: when
nothing reachable is worth testing, `lama/exploration/` picks the
not-yet-reachable object with the best expected-information-gain per unit of
walking distance, not simply the nearest one -- but it still cannot escape a
single object's budget-grinding once the agent is already standing next to it,
which is exactly what the pinned regression test above is there to catch if a
future change fixes it.

## Planning and self-improvement

`lama/planner/` does real STRIPS-style backward chaining over the bank's
*confirmed* operators: given a goal ("some object of concept X should reach
effect E"), it searches backward for a chain of verbs that reaches it,
discovering genuine multi-step structure -- the project's own flagship case
requires GRASPing the block before `PLACE_ON` opens the door, and the planner
finds that ordering, not just the direct step. Its output feeds
`Hypothesis.relevance`, which `select.py` folds in multiplicatively alongside
a safety discount for irreversible outcomes and a mild curiosity bonus for
broadly-uncertain objects. A goal reorders what gets tried first; it never
restricts what CAN be tried, and leaving it unset changes nothing.

`scripts/run_lifelong.py` is the concrete, honest form "self-improvement"
takes given this architecture: there is no neural network in the working core
to gradient-train, so getting better over time means running many episodes on
one persistent memory and checking whether later episodes need less budget per
unit of newly-settled knowledge than earlier ones. Goals are derived from the
agent's own confirmed beliefs, never the oracle. Run for real over 100
episodes: total interactions per 20-episode block fell from 982 to 455, goal
pursuit switched on automatically once anything was confirmed to open, and two
real concept splits fired during ordinary play.

## Evaluation and results

`lama/evaluation/` measures interactions per confirmed affordance, precision
and recall against hidden ground truth (broken out by `PRIMARY`/`SECONDARY`/
`INCIDENTAL` role -- the headline split, see below), retention across tasks,
and transfer to held-out kinds. `scripts/run_baseline_comparison.py` runs
LAMA's full selection policy against three baselines -- random,
novelty-driven (Plan2Explore-style), and uncertainty-only (LAMA's core
acquisition function with the safety/curiosity/goal bonuses removed) -- plus
a targeted safety-discount ablation, through the identical environment and
adjudication.

**Full account: `docs/RESEARCH_FINDINGS.md`.** The short version, from a real
15-seed x 40-episode run: LAMA's full system clearly wins on overall recall,
primary-affordance recall, and interactions-per-confirmed-affordance (115.3
vs 184.5 for random). On the one metric this project's design is actually
organised around -- **secondary**-affordance recall -- it does not win: the
ordering *inverts* (uncertainty-only 0.122 > LAMA 0.103), and the reason is
identifiable rather than noise: secondary affordances live disproportionately
behind `PLACE_ON` and `TIP`, and LAMA's curiosity and safety refinements --
which correctly improve the aggregate case -- specifically deprioritise
exactly those. That is reported as a real, measured trade-off, not smoothed
into "LAMA wins", because the prior work this project grew from
(`files/EmbodiedAI.zip`) is worth citing again for modelling exactly that
standard: it reported InfoNCE scoring *below* random initialisation, plainly.

## Repository layout

```
lama/                            the LAMA system
scripts/isaac_smoke_test.py      isolates an Isaac Lab install problem from this project's code
scripts/run_lama_isaac.py        runs the verification loop against the Isaac backend
scripts/run_lifelong.py          runs many episodes on one persistent memory, reports improvement over time
scripts/run_baseline_comparison.py  LAMA vs random/novelty/uncertainty-only, across seeds
docs/DECISIONS.md                why the project is built this way
docs/RESEARCH_FINDINGS.md        gaps addressed, experiments run, results including where LAMA does not win
docs/ISAAC_LAB_SETUP.md          Isaac Lab backend setup, pinned versions, current status
outputs/                         real experiment results from both backends and the baseline comparison
aura_warehouse_sim/         prior work: AURA navigation-only warehouse prototype
files/AURA_Project_Plan.pdf prior work: the AURA research plan this project grew from
files/EmbodiedAI.zip        prior work: an earlier object-function-discovery project
```

Prior work is retained deliberately. `files/EmbodiedAI.zip` in particular
contains a documented negative result — no unlabelled function-learning
objective beat random initialisation at 186 interaction records — that directly
motivates LAMA's emphasis on interaction efficiency.

## Setup

```bash
pip install -r requirements.txt
streamlit run app.py      # interactive demo
pytest                    # 210 tests; the Isaac backend's logic tests run when torch is installed
```

CPU is sufficient for everything on the numpy backend, and the demo also runs in
the browser (NumPy only). It lets you:

- run any selection policy for up to 60 episodes on one persistent memory and
  watch confirmed knowledge accumulate;
- inspect the affordance bank (status, posterior mean, attempts per
  `(concept, verb, tool)`) and the step log of the last episode;
- rerun the controlled policy comparison live, and read the published
  15-seed x 40-episode results.

To run the Isaac Lab backend, see `docs/ISAAC_LAB_SETUP.md` -- it needs a
separate Python 3.10 environment and an RTX-class GPU, and pins specific
Isaac Lab / Isaac Sim versions chosen for compatibility with 8GB-class cards,
since the current latest release requires 16GB.
