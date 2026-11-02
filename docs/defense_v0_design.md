# Defense v0: minimum game and evaluation contract

This note fixes the interface expectations for Defense development.
It is not a claim that official 8-ball/9-ball rules are implemented.

## Existing, working foundation

- FastFiz: one cue ball + two object balls, physical transition and pocket states.
- `GameState`: positions, pocket flags, two-player scores, `current_player`.
- `SimplifiedGameRules`: pot -> same player, miss -> next player, scratch -> terminal penalty,
  both object balls pocketed -> terminal.
- Macro Attack v0: DIRECT/POSITION candidates, local FastFiz reward,
  Candidate Q, optional Candidate Policy, MCTS.
- SAFETY candidate/goal encoding and goal-position checks are available, but
  **SAFETY cannot yet be executed by `MicroBilliardEnv`**.

## Problem discovered before Defense

The old candidate-Q target and MCTS treated the second shot's reward as
a gain for the original player even if `SimplifiedGameRules` switched turns.

The value contract for opponent-aware search is now:

- `Q(s,c)` and `V(s)` are from the perspective of `s.current_player`.
- An edge reward is earned by the acting player before the edge.
- The child continuation is negated if its `current_player` is different.
- At every node, the current player chooses actions to maximize their own value.

In formulas:

```
Q(s,c) = r(s,c) + gamma * sign(s,s') * V(s')

sign(s,s') = +1  if s.current_player == s'.current_player
            -1  otherwise
```

The two-shot supervised target uses the same sign:

```
target(s,c) = r1 + gamma * sign(s,s') * max_c2 r2(s',c2)
```

The max is taken by the player to move in `s'`, **before** negating the
opponent's best outcome. This is a two-shot approximation, not full minimax
over a complete match.

## Compatibility boundary

- The existing Attack v0 default remains `turn_aware=False`.
- The turn-aware behavior is opt-in in MCTS and in two-step target generation.
- Datasets include a `turn_aware` metadata flag. Value training copies it
  to the checkpoint. Old datasets/checkpoints are treated as legacy.
- The full integration smoke rejects a candidate-Q checkpoint whose
  metadata does not match the chosen evaluation mode.
- The existing learned Candidate Policy is Attack-v0-only; turn-aware smoke
  uses the uniform Policy until a compatible Policy is trained.
- **Do not overwrite** the Attack v0 dataset or checkpoints when generating
  a turn-aware variant.
- A changed reward/turn contract implies retraining Candidate Q and any
  Policy trained from those targets. Reusing old weights is not valid.

## Defense v0 steps (in order)

1. **Turn-aware value contract and tests**: same-turn continuation positive,
   opponent continuation negative, terminal transitions no lookahead;
   preserve the legacy tests.
2. **Opponent opportunity/threat evaluation**: given post-shot `GameState`,
   estimate the opponent's best executable DIRECT shot using the same
   FastFiz/Micro model. Separate geometric feasibility from actual shot
   success. Compare this with simply counting possible shots.
3. **Minimal DEFENSE shot interface**: select a target object ball/aiming
   reference for actual FastFiz contact. Current `MicroBilliardEnv`
   requires a target ball and target pocket and only executes ATTACK.
   Decide and document a safe, physically meaningful shot
   parameterization before training a Defense Micro policy.
4. **Defense candidate generation + execution**: connect SAFETY goals to
   executable shot candidates and define goal/reward based on avoiding
   fouls and reducing opponent opportunity. The current fixed cue-region
   anchors are placeholders, not actual safety shot plans.
5. **Small paired benchmark**: same boards, Attack-only versus
   Attack+Defense, with outcome metrics including scratch, next-player
   opportunities, and selected strategy.

## Explicitly deferred

- Official 8-ball/9-ball rules, full 15-ball table, foul variations.
- A production-level learned Defense Micro policy.
- Major restructuring of the project while these interfaces are changing.
- Full re-training or tuning of Attack Candidate Policy.

## Refactoring checkpoints

1. **Small cleanup after the turn-aware boundary is tested**:
   consolidate duplicated state/turn utilities, remove stale comments,
   and document interfaces without changing behavior.
2. **Broader refactor after Defense v0 has an end-to-end smoke test**:
   consolidate dataset/training/eval CLI setup, separate scenario
   construction from search, organize test fixtures and reduce duplicate
   helper code. Run the associated regression tests before and after.

The rule of thumb is to refactor at stable interface boundaries, not to
postpone all cleanup nor to reorganize the entire codebase mid-feature.


## Opponent DIRECT threat diagnostic (stage 2)

`OpponentDirectThreatEvaluator` accepts an opponent-to-move,
nonterminal post-shot `GameState`. It generates the opponent's
feasible DIRECT candidates and executes each candidate through the
existing deterministic Micro + FastFiz + game rules + local evaluation
transition.

It separately reports:

- `geometric_candidate_count`: geometry-filtered DIRECT candidates;
- `successful_pot_count`: actually potted target, without scratch;
- `best_success_reward`: best local score among successful outcomes;
- `best_candidate`: corresponding candidate, if any.

Important limitations:

- These are deterministic outcomes under the current Micro policy,
  **not probabilities** of winning or of the opponent succeeding.
- No opponent model, strategic probability estimate, or other-player
  training is introduced.
- The diagnostic does not modify MCTS reward or candidate-Q targets.
- The implementation assumes a nonterminal state; scratch is terminal
  under the current `SimplifiedGameRules`.
- A state where the defending player retains the turn is not an
  opponent-turn state and is rejected explicitly.
- `scripts/eval_opponent_direct_threat.py` finds a nonterminal
  turn-changing DIRECT shot on each sampled board and examines the
  resulting opponent's DIRECT opportunities. This is a diagnostic
  board selection, **not an unbiased sample of play under a trained
  Macro strategy**.

The next Defense step is to define an executable physical shot
interface, after comparing geometric versus executable opponent shot
counts in this diagnostic.


## Defense shot execution v0 (stage 3)

A separate `DefenseShotExecutor` uses existing `ShotAction(angle, power)`
as a *physical* shot command. The pure `plan_defense_shot` routine
computes an absolute angle toward the center of a designated on-table
object ball, optionally with a bounded offset, and a physical power.

- No ghost-ball/pocket-target geometry is required.
- The unchanged FastFiz `step(target_ball, power, angle)` computes
  the physical outcome. In the current C++ wrapper `target_ball`
  chiefly selects the ball checked for its **legacy reward**;
  the physical ShotParams depend on power and angle.
- `SimplifiedGameRules` updates score/turn/terminal status.
- This is explicitly separate from ATTACK Micro SAC and from the
  currently unexecutable SAFETY MacroCandidates.
- Scratches and newly pocketed object balls are observable.
  First-contact event exposure is added in the next stage, but full
  legality still requires foul-rule handling.
- No learned Defense Micro and no MCTS integration is claimed.
- A small deterministic physical probe and scoped tests are included.

Only after this executor is verified should SAFETY candidates be
augmented with feasible *shot proposals* and compared by their
opponent DIRECT threat on the resulting state.


## Optional tip offsets and first contact (stage 3b)

The FastFiz C++ wrapper exposes previously fixed cue-tip contact
coordinates via optional `tip_x_mm`/`tip_y_mm` values.
They map directly to FastFiz `ShotParams(a,b,theta,phi,v)`
where `a,b` are **millimeters**, `theta=10 deg` is still fixed,
`phi` is the absolute angle and `v` is the cue speed.
A conservative tip-offset radius of at most 12 mm is enforced,
followed by FastFiz's own physical-possibility check.

`ShotAction.tip_x_mm`/`tip_y_mm` are distinct from the older
unused `spin_x`/`spin_y` placeholders. **No existing SAC action
dimension changes**. ATTACK still supplies the center hit at (0,0),
so prior checkpoints and training datasets retain their contract.

C++ now records the *first cue/object BALL_COLLISION event* per shot:

- `-1`: cue did not hit either object ball;
- `0`: object ball 0 first;
- `1`: object ball 1 first.

`DefenseShotResult.first_contact_verified` checks whether that
first ball matches the intended defensive target, without claiming
full official foul legality. Movement alone is no longer used as a
first-contact proxy.

**The C++ extension must be rebuilt after updating this wrapper.**
This stage does not yet connect DEFENSE to MacroCandidate or MCTS,
and does not claim spin-aware Micro policy learning.


## Outcome-guided Safety shot search (stage 4a)

Before joining the MacroCandidate / MCTS graph, explore a small,
explicit grid of physical DEFENSE shot commands using
`OutcomeGuidedSafetyShotGenerator`:

1. Select each on-table object as an intended first-contact ball.
2. Generate physical `ShotAction`s with center-line angle offsets,
   speeds and optional tip-y offsets (0 and +5 mm by default).
3. FastFiz-execute from the **same original GameState** for every
   candidate.
4. Screen for **first contact on intended object**, **no scratch**,
   **no newly pocketed object**, **nonterminal**, and **turn change**.
5. For surviving post-shot states, run the deterministic opponent
   DIRECT threat evaluator. Rank by the fewest actually successful
   opponent DIRECT shots, then geometrically feasible shots.

This is **not** the earlier arbitrary cue-region-anchor generator and
not a trainable MacroCandidate. It is an outcome-guided *physical shot
proposal* that will later help define Safety TacticalGoals and
candidate encodings. Do not mix its results into the existing
Attack-trained Candidate Q or Policy before new action/goal feature
encoding and retraining are implemented.

`scripts/eval_safety_shot_proposals.py` also compares the best
searched Safety action with the first failed DIRECT baseline from
the **same initial board**. This is an optimistic *in-model*
counterfactual (because Safety candidates are scored using the same
lookahead model), not an unbiased gameplay evaluation or proof of
improvement over Attack.

Known remaining limitations:

- `SimplifiedGameRules` does not apply official foul rules.
  The candidate screen checks contact, no scratch and turn change,
  but not rail-after-contact requirements.
- For now opponent shot quality is measured as a deterministic count
  of successful DIRECT candidates, not a probability.
- Only a few powers, angles and cue-tip offsets are enumerated.
  It is possible to have no eligible Safety shot in the grid.
- Learned Defense Micro, candidate-dependent Policy/Q and mixed
  Attack/Defense MCTS are still pending.


## MacroCandidate representation for executable Safety proposals (stage 4b)

The physical safety search is useful, but its proposal cannot be fed to
Attack-v0 Candidate Policy/Q without a new candidate encoding. Both
the candidate goal AND the physical shot matter.

- `PhysicalShotSpec(angle,power,tip_x_mm,tip_y_mm)` is immutable and
  hashable. `MacroCandidate.physical_shot` is optional and defaults to
  None for all existing Attack candidates.
- `OutcomeGuidedSafetyMacroCandidateGenerator` converts selected,
  physically executed Safety proposals into
  `MacroCandidate(action=DEFENSE,target_ball=...,target_pocket=None,
  goal=SAFETY,physical_shot=...)`. It uses the model-predicted
  cue resting location as a provisional goal.
- Existing `encode_macro_candidate` remains **25-dimensional** and
  now explicitly rejects DEFENSE candidates to avoid silently dropping
  shot parameters into Attack-only pretrained Policy/Q models.
- `encode_macro_candidate_v2` is **31-dimensional**, using 25
  goal-level features plus six physical-shot features:
  present, power/3, sin(angle), cos(angle), tip_x/12, tip_y/12.
  Attack without an explicit physical shot has six appended zeros.
- The changed dimensionality means that **new mixed-strategy
  datasets and Q/Policy checkpoints must be trained** before combining
  Attack and Defense in the learned MCTS. Old weights/checkpoints
  remain valid for Attack-only v0.
- The current adapter is **representation-only**; it does not connect
  executable SAFETY to `PhysicsMacroTransitionModel` or MCTS and
  does not imply learned spin control.
- The legacy anchor-based `SafetyCandidateGenerator` is still an
  unexecutable placeholder and must not be used as a substitute for
  model-based physical shot proposals in a final mixed-strategy run.

The next implementation boundary is to execute `physical_shot`
inside a dedicated Defense transition branch, define a compatible
local reward, and build/train mixed-strategy Candidate Q before
enabling MCTS to choose Defense actions.


## Separate mixed-strategy transition dispatcher (stage 4c)

`DefenseMacroTransitionModel` accepts only an **executable**
`MacroCandidate(action=DEFENSE, goal=SAFETY, physical_shot=...)`.
It reconstructs a physical `ShotAction`, re-executes it from the
supplied state with `DefenseShotExecutor`, then returns
`MacroTransitionResult` with updated score/turn and an optional
`DefenseLocalEvaluation`.

- **Attack path unchanged.** `MixedStrategyMacroTransitionModel`
  dispatches ATTACK to the existing `PhysicsMacroTransitionModel`
  and DEFENSE to the new Defense transition. No shared Micro policy
  or action reinterpretation is introduced.
- Immediate Defense reward is the simplified-game reward
  (including pot and scratch consequences) minus a configurable
  penalty if the intended first object was not contacted (but avoid
  double-penalizing scratches).
- The Defense reward deliberately does **not** include the
  opponent's next-shot threat. With turn-aware MCTS or two-step
  Q targets, the opponent's continuation is negative for the
  defending player; including it in local reward would duplicate it.
- Eligibility still requires intended first contact, no scratch,
  no newly pocketed balls, nonterminal result and turn change.
  A shot that pockets an object ball can receive the ordinary game
  score but is NOT recorded as successful SAFETY.
- `goal_achieved` on deterministic replay only verifies that the
  shot reached the predicted cue region; the region was generated
  using the same FastFiz shot, so this is **not an independent
  measure of defensive effectiveness**.
- Anchor-only DEFENSE candidates without `physical_shot` are
  rejected; untrained 25-dimensional Candidate Q/Policy is still
  NOT used for mixed-strategy selection.
- `scripts/smoke_mixed_strategy_transition.py` runs a small
  physical smoke (Attack and replayed Safety) without invoking MCTS.

The next step is a new **31-dimensional mixed-candidate data/target
pipeline**, with turn-aware Q targets and a paired ATTACK/DEFENSE
benchmark. Mixed MCTS integration must wait for the compatible
candidate prior/Q networks or an explicit v2 uniform-prior baseline.


## Small refactor + 31-D mixed Candidate Q dataset (stage 5a)

After the 2/2 deterministic mixed-transition smoke, the Safety
adapter gained a `from_proposals(proposals)` method. This allows
shot search, candidate encoding and testing to reuse **one** expensive
set of evaluated physical proposals instead of repeating all FastFiz
opponent-response searches. The attack transition API was unchanged.

`scripts/build_mixed_candidate_value_dataset.py` now generates
**31-D**, **turn-aware**, combined ATTACK/DEFENSE supervision.

- Attack roots: existing DIRECT and POSITION candidates.
- Defense roots: top-K eligible outcome-guided physical Safety
  proposals, encoded as executable Safety MacroCandidates.
- Transition: mixed Attack/Defense dispatcher, with each strategy's
  own existing one-shot execution and local reward.
- Shared two-step teacher (acting-player perspective):
  `r1 + gamma * (+1 for same player, -1 for opponent) *
   max(next DIRECT immediate reward)`.
- Next-step choices deliberately use **DIRECT only** for both
  strategies as a cheap, controlled first baseline. Thus the
  teacher is **not** a full mixed-strategy minimax value.
- The Safety root action generator uses deterministic opponent
  lookahead to *select* the top-K Safety shots, while labels are
  subsequently computed from the same physics model. These
  training samples are selection-biased. Don't claim out-of-sample
  Strategy improvement from training-set targets.
- The dataset stores `candidate_encoding_version=2`,
  `candidate_dim=31`, `turn_aware=True`,
  `next_action_set=DIRECT_ONLY`, `strategy`, goal type, board
  grouping and target diagnostics.
- Existing Q training accepts 25 or 31 dimensions. It preserves
  old 25-D dataset behavior; 31-D data **must** identify version 2
  and turn-aware labels. The checkpoint records dimensionality.
- The old `load_macro_candidate_value_network` defaults to 25-D
  and refuses 31-D models unless the caller explicitly sets
  `expected_candidate_dim=31`. Old Attack-MCTS therefore
  cannot silently mix incompatible feature formats.
- Store mixed data/checkpoints under **new filenames**, never
  overwrite `macro_candidate_value.pt` or the old dataset.

This stage builds the learning-data pipeline only. Mixed Q inference
in a new MCTS path, Candidate Policy retraining, and an unbiased
paired multi-board evaluation remain future work.


## Mixed Candidate Q held-out evaluation (stage 5b)

The 3-board/70-sample smoke demonstrated the generator contract:
58 ATTACK and 12 SAFETY candidates, physical Safety proposals
39/72 eligible, 31-dimensional inputs and turn-aware teacher labels.
It is **not** enough data to claim learned generalization.

The mixed data generator now stores `seed_start` and
`requested_states` for reproducible, disjoint board ranges.
`scripts/eval_mixed_candidate_value.py` evaluates a trained
31-D checkpoint on a **separately generated dataset**:

- overall MSE / MAE / top1 against the two-step teacher;
- per-strategy target mean, prediction mean, MSE and counts;
- mean two-step ranking regret within each board;
- number of boards where predicted top choice is SAFETY;
- number of boards where SAFETY ties/beats the best teacher
  candidate (among boards that have SAFETY proposals).

It explicitly rejects the training dataset and overlapping
training/evaluation seed ranges when both datasets have the new
metadata. If metadata is unavailable, it prints that disjointness
could not be verified. These are **held-out teacher agreement
metrics**, not real-world Defense quality or game win-rate.

Suggested separate experiments (from repository root):

```bash
python -m scripts.build_mixed_candidate_value_dataset \
  --base-checkpoint goal_conditioned_micro_mixed_curriculum_100k.pt \
  --residual-checkpoint position_power_residual_mlp_2000.pt \
  --states 20 --seed 93000 --device cuda \
  --output macro_candidate_value_mixed_v2_train.npz

python -m scripts.train_macro_candidate_value \
  --dataset macro_candidate_value_mixed_v2_train.npz \
  --checkpoint macro_candidate_value_mixed_v2.pt \
  --epochs 80 --batch-size 128 --device cuda

python -m scripts.build_mixed_candidate_value_dataset \
  --base-checkpoint goal_conditioned_micro_mixed_curriculum_100k.pt \
  --residual-checkpoint position_power_residual_mlp_2000.pt \
  --states 8 --seed 94000 --device cuda \
  --output macro_candidate_value_mixed_v2_heldout.npz

python -m scripts.eval_mixed_candidate_value \
  --checkpoint macro_candidate_value_mixed_v2.pt \
  --dataset macro_candidate_value_mixed_v2_heldout.npz \
  --device cuda
```

The old Attack 25-D dataset and checkpoint remain untouched.
Candidate Policy v2 and mixed MCTS still have not been integrated.


## Mixed turn-aware MCTS integration smoke (stage 6a)

Held-out mixed Candidate Q v2 (20 train boards, 8 held-out boards)
reported: overall MSE 1.0842, top1 0.500 and 0/8 Safety
choices; the teacher oracle also preferred Attack for all 8 boards.
Mean teacher Q was Attack +0.607 and Safety -1.160. A well-
executed successful Attack should generally dominate a Safety shot
under this teacher, so 0/8 Safety does not prove a software bug
or that a Defense policy cannot exist. Importantly, there is
not yet evidence that the new Q selects Defense in difficult
Attack states. Validation 100% was based on very few boards.

`MCTS` now offers an **explicit opt-in** for mixed 31-D candidates:
`candidate_encoding_version=2`, `uniform_candidate_priors=True`,
`candidate_mode=True`, `turn_aware=True`.
The existing v1 defaults, networks and code path remain unchanged.

- Root: existing DIRECT+POSITION Attack plus a small set of
  outcome-guided, physically executable SAFETY candidates.
- Tree continuation: DIRECT-only on subsequent states (the
  teacher's cheap next-action baseline).
- Policy prior: uniform over available candidates; no untrained
  Candidate Policy v2 is invented or loaded.
- Value: learned **31D** mixed Candidate Q for unvisited edge
  predictions, combined with empirical returns as MCTS visits
  accumulate.
- Leaf state value: zero baseline, **not** an independently trained
  long-horizon state V.
- Opponent continuation: negated across real player changes
  by `turn_aware=True`.
- At root, `root_min_visits_before_puct=1` can ensure each
  candidate receives at least one physical transition **if**
  simulations exceed the number of root candidates.
- The smoke reports root candidate counts, visits allocated to
  each strategy, selected strategy, predicted Q, executed reward
  and next player. It is a wiring test, not a win-rate claim.

`scripts/smoke_mixed_candidate_mcts.py` uses the user's trained
`macro_candidate_value_mixed_v2.pt` checkpoint and leaves the
Attack-only run code untouched. Candidate Policy v2 training and
fair paired board-level strategy comparisons remain separate work.


## Attack-difficult board paired MCTS diagnostic (stage 6b)

The two-board mixed-MCTS smoke (seed 94000,30 simulations)
reported 20 Attack and 4 Safety candidates per board.
Root visits were 26 Attack / 4 Safety in each case; MCTS
selected POSITION_ATTACK with rewards 1.25 and 1.60.
This confirms that both strategies can be expanded and
executed, but **30 simulations for 24 root candidates** means
most budget only gives each branch one visit. It does not
prove deep strategic search quality or the superiority of Attack.

`src/macro/mixed_search.py` centralizes ZeroValue, fixed root
candidate snapshots and the small mixed root generator. These
helpers avoid copying identical setup between smoke/evaluation
scripts, without altering legacy Attack behavior.

`scripts/eval_mixed_mcts_difficult_boards.py` provides a
*controlled in-model* board-difficulty evaluation:

1. Sample new random boards from a separate seed range.
2. Execute **all** ATTACK root candidates under deterministic
   Micro+FastFiz (not just geometry) and count non-scratch
   successfully pocketed targeted balls.
3. Select boards where no more than K Attack candidates
   actually succeeded (K is configurable; K=0 is the strict
   no-success case).
4. On these boards only, generate physical Safety proposals.
5. Compare Attack-only root MCTS vs Attack+Safety root MCTS,
   using the **exact same Attack choices**, 31-D Q checkpoint,
   uniform priors, zero state-V, simulations, discount and
   DIRECT-only continuations.
6. Compare chosen candidates' turn-aware two-step teacher Q
   from the same board, their root visits, and Defense choice
   frequency. Report no qualifying boards explicitly.

This evaluates the *integration and short-horizon teacher*,
not a win-rate or an unbiased proof of Defense effectiveness.
Safety shot proposals themselves are selected with in-model
opponent lookahead. Also, including Safety increases the
mixed root's candidate count, so equal simulation budgets
leave fewer visits per candidate; the output includes visit
counts for interpretation.

A reasonable small first run is up to 20 sampled boards,
at most three selected hard cases, 60 simulations each,
with success threshold K=2. If too few cases qualify, widen
the threshold or seed scan. For a strict hard case, use K=0.


## Attack-difficulty screening and teacher-oracle diagnostics (stage 6c)

On seeds 95000..95019, the earlier threshold of at most two
successful ATTACK candidates selected only one board (board 12).
Its best immediate ATTACK reward was +1.250 and the selected
DIRECT_ATTACK 2-step teacher Q was +2.240 for both compared MCTS
variants. Their Safety alternatives each received only one root
visit. This is not evidence that Defense is ineffective.

`eval_mixed_mcts_difficult_boards.py` now reports, in addition
to the previous all-goal success count:

- successful DIRECT candidate count;
- number of distinct successful (target ball, target pocket)
  routes, collapsing DIRECT and POSITION representations
  that refer to the same route (not necessarily the same shot);
- histogram of total successful ATTACK candidates on scanned
  boards: 0, 1, 2, or 3+.

Optional `--oracle-diagnostic` also replays all Attack and Safety
root candidates to calculate the BEST two-step teacher Q for
each strategy, independent of the MCTS-selected action. It
reports `oracle_best_attack`, `oracle_best_safety`, their
difference and aggregate comparisons.

This diagnostic distinguishes:
1. **Teacher prefers Attack**: Defense selection would not be
   justified by the current two-step objective, even with
   perfect ranking.
2. **Teacher prefers Safety, MCTS still selects Attack**:
   investigate 31D Q predictions, priors, and search budget.
3. **Teacher and MCTS both prefer Safety**: a promising
   *within-model* Defense case to test under broader seeds,
   rules, and an independent gameplay evaluation.

Costs: `--oracle-diagnostic` executes all paired-board
root candidates plus DIRECT one-shot continuations and can
be substantially slower. Use only a few paired boards.
As before, the Safety proposal generator uses the same physics
lookahead as evaluation, and the game rules remain simplified.


## Post-SAFETY hard-state oracle diagnostic (stage 6d)

On the earlier random-board diagnostic (20 scanned boards,
threshold <=2 successful Attack candidates), only 1 board
qualified, with 2/20 successful ATTACK Micro outcomes.
The per-strategy **best** two-step teacher Q on that board was

- ATTACK: +2.240
- SAFETY: -1.238
- SAFETY minus ATTACK: -3.478

The mixed MCTS selected the same DIRECT_ATTACK as the
Attack-only MCTS and assigned only 4 of its 60 root visits
to the 4 SAFETY candidates. This is **not** a failure of
MCTS to discover a better SAFETY under the current teacher.

More fundamentally, random initial boards in the simple
3-ball setting rarely have no executable attack; 19/20 had
3+ successful ATTACK root candidates. Also the SAFETY shot
generator ranks by the NUMBER of opponent DIRECT candidates
which pot successfully, while the two-step Q teacher uses
the MAXIMUM next DIRECT local reward. If the next player
has any successful DIRECT candidate with reward 1.25,
then 1 versus 10 such candidates can yield the SAME teacher
continuation. This is an objective-alignment limitation,
not a reason to change reward weights merely to force SAFETY
choices. We need to measure it before revising objectives.

`scripts/eval_post_safety_hard_states.py` constructs a
**targeted stress-test**, not an unbiased board sample:

1. Sample a few ordinary seeded initial boards.
2. Use physical SAFETY search to produce valid post-shot
   opponent-turn GameStates. These use FastFiz outcomes, not
   manually invented geometric layouts.
3. Retain poststates with at most K successful DIRECT shots
   for the new player (usually K=0); deduplicate nearby
   states to limit repeated evaluation.
4. Reevaluate **all** Attack candidates using the existing
   Micro+FastFiz and retain states with at most a configurable
   number of successful Attack outcomes including POSITION.
5. Generate executable SAFETY moves for the new player,
   compare best two-step teacher Q over all Attack and Safety
   root candidates. Optional `--run-mcts` compares the MCTS
   choices using the same 31D checkpoint and fixed simulation
   budget.

Note that these root positions are selected *because* Safety
earlier made DIRECT attacks difficult. They are useful to
test whether the model can handle such positions, but
cannot establish broad generalization or win rate. The
two-step teacher's DIRECT-only continuation and the
simplified foul rules are unchanged.

Small first run (oracle only, no new model training):

```bash
python -m scripts.eval_post_safety_hard_states \
  --base-checkpoint goal_conditioned_micro_mixed_curriculum_100k.pt \
  --residual-checkpoint position_power_residual_mlp_2000.pt \
  --source-boards 5 --source-seed 90000 \
  --max-successful-direct 0 \
  --max-successful-attacks 2 \
  --max-compared 2 --device cuda
```

If it finds no paired states, report how many post-Safety
DIRECT-hard states were found and how many failed the
full Attack filter. Only then consider relaxing that filter
or broadening the physical search grid.


## Stage 6e: no Safety continuation in post-Safety stress test

The first 5-source-board stress test (seed 90000) produced 2
unique post-Safety states with 0 successful opponent DIRECT shots.
Both states passed the max-two successful ALL-ATTACK filter,
but **neither had an eligible subsequent Safety proposal** from
the fixed grid (powers 0.65/1.10, offsets -12/0/+12 degrees,
tip Y=0/+5 mm). Therefore 0 paired teacher comparisons were
possible. This is a candidate coverage issue, not a finding
about the relative strategy quality.

The stress-test script now prints full per-grid physical rejection
counts for every hard poststate: attempted, invalid physics,
no/wrong first contact, scratch, pot, other ineligible,
and eligible. They distinguish no geometric/physical contact
from foul/shot outcome filters.

An opt-in `--retry-expanded-safety` retries ONLY hard states
whose baseline Safety candidate list was empty. The wider
grid uses powers 0.55/0.8/1.3/1.8, angle offsets
-20/-10/0/+10/+20 degrees, tip Y -5/0/+5 mm
(max 120 shots with two balls). This does NOT change
the training dataset generator, the learned Micro, Macro
local rewards or standard MCTS. It only measures whether
a larger grid recovers physically valid Safety continuations.

Even an expanded grid with no valid candidates does not
establish that no safe shot exists: it only tests the
enumerated physical shot parameters and simplified first
contact/no-pot/no-scratch/turn-change conditions.

Small opt-in probe:

```bash
python -m scripts.eval_post_safety_hard_states \
  --base-checkpoint goal_conditioned_micro_mixed_curriculum_100k.pt \
  --residual-checkpoint position_power_residual_mlp_2000.pt \
  --source-boards 5 --source-seed 90000 \
  --max-successful-direct 0 --max-successful-attacks 2 \
  --max-compared 2 --retry-expanded-safety --device cuda
```


## Stage 6f: FastFiz physical rejection bitmask diagnosis

On five source boards (seeds 90000-90004), two post-SAFETY
opponent states had zero executable Attack outcomes.
Neither permitted a subsequent eligible SAFETY in the tested
grids. The observed rejection counts were:
- source board 1: 24/24 baseline and 120/120 expanded
  shots rejected by the native FastFiz precondition.
- source board 3: 24/24 baseline and 116/120 expanded
  shots rejected at precondition; remaining 4 failed
  first cue/object contact.

This strongly indicates that the physical shot precondition
is the limiting step. The native FastFiz `isPhysicallyPossible`
check uses a bitmask including invalid shot parameters
and `CUE_STICK_COLLISION=128`, which includes collision
of the cue stick with an object ball or rail before impact.
For a cue ball parked near a cushion, the fixed cue elevation
(theta=10 degrees in the wrapper) is one possible cause;
**actual rejection codes must be measured**, not inferred.

The native wrapper now retains the precondition code and exposes
the `isValidBallPlacement()` status separately. Python maps
the bitmask names, carries the **pre-shot** ball-placement code,
and raises a typed `FastFizPreconditionError` rather than
discarding the reason behind a rejected shot. The Safety
generator counts rejected-shot codes without changing
eligible Safety criteria, shot physics, rewards or learned
policies. The post-Safety probe prints the raw code histogram
and decoded names for baseline and expanded attempts, and
the hard-state cue position for identifying rail proximity.

The native `billiard_env_cpp` extension MUST be rebuilt:

```bash
git pull
cmake --build build -j4
python -m pytest tests/test_fastfiz_simulator.py tests/test_defense_shot.py tests/test_outcome_guided_safety_shots.py tests/test_post_safety_hard_states.py -v
```

If `CUE_STICK_COLLISION` dominates, do not expand power as
the first remedy: check the cue-to-rail distance, other-ball
obstruction, and the fixed cue elevation. No rule has been
relaxed to force artificially legal shots.


## Stage 6g: near-rail cue-stick obstruction sensitivity probe

Native precondition flags on the same source states now show:
- seed 90001: cue=(1.0040, 0.0337), all 24/24 baseline
  and 120/120 expanded attempts rejected as
  CUE_STICK_COLLISION (128), valid placement (0).
- seed 90003: cue=(0.2207, 2.1985), all 24/24 baseline
  and 116/120 expanded attempts rejected as
  CUE_STICK_COLLISION (128), valid placement (0).
  The remaining four shots failed target first-contact.

These cue positions are approximately 33.7 mm from the
south rail and 37.5 mm from the north rail (cue-ball radius
about 28.6 mm), so the cue stick is near the cushion.
In FastFiz, CUE_STICK_COLLISION includes cue-stick vs
rail and cue-stick vs other-ball obstruction, however. The
exact obstacle is still unknown.

`scripts/probe_cue_elevation_on_hard_states.py` explores
cue elevation only in a **separate physical diagnostic**:
`theta=[10,20,35,50,60]` degrees, and five offsets around
the target-ball center direction, each with fixed power 1.0
and tip offsets (0,0). It checks native precondition,
first contact, scratch and pocket state. It also reports
how many of the present ATTACK root candidates actually
passed the native shot precondition, which is different
from counting learned Attack success. Attack transitions
now retain their last shot's `info` for this diagnostic;
no Attack reward or invalid-action behavior is changed.

The C++ bridge and Python FastFizSimulator.step now take
an optional trailing `cue_elevation_deg=10.0`, preserving
the historical default *exactly*. The existing Attack Micro
and DefenseShotExecutor do **not** expose this parameter;
their action semantics, learned weights, datasets and
31D candidate encoding therefore remain unchanged.
Any future integration of elevation into learned mixed
Strategy evaluation would require an explicit new action
representation and retraining / new teacher datasets,
rather than silently reusing the old 31D checkpoint.

Run native rebuild first:

```bash
git pull
cmake --build build -j4
python -m pytest tests/test_fastfiz_simulator.py tests/test_probe_cue_elevation.py tests/test_post_safety_hard_states.py -v
```

Then try the standalone sensitivity probe:

```bash
python -m scripts.probe_cue_elevation_on_hard_states \
  --base-checkpoint goal_conditioned_micro_mixed_curriculum_100k.pt \
  --residual-checkpoint position_power_residual_mlp_2000.pt \
  --source-boards 5 --source-seed 90000 \
  --max-hard-states 2 --max-successful-direct 0 --device cuda
```

If larger elevation removes bit 128, that supports fixed
cue elevation as a bottleneck; if not, test more directions
or inspect other-ball/cue-stick geometry before assuming
no legal physical shot exists. This is only a feasibility
study, not Defense efficacy evidence.


## Stage 6h: cue elevation / first-contact sweep on hard post-Safety states

On the same two hard poststates (source boards 1, 3; nearest
rails 0.0337m and 0.0375m away), all 21 tested Attack
candidates with the trained fixed 10-degree action format
failed the FastFiz precondition with code 128
(`CUE_STICK_COLLISION`). Both also had no successful
DIRECT reply, by construction.

A controlled elevation probe showed the number of
physically permitted shot attempts (10 attempts per theta):
- board 1: 10/20/35/50/60 degrees -> 0/0/1/2/5 valid
- board 3: 10/20/35/50/60 degrees -> 0/2/5/10/10 valid

NONE of these passed the desired first-object-contact test.
This separates native precondition feasibility from making a
legitimate shot. Higher theta mitigates rail/cue obstruction,
but may also change the ball's initial trajectory (including
jumping) and cannot simply be assumed a valid Safety move.

The standalone `probe_cue_elevation_on_hard_states.py`
now optionally tries `--powers` and `--tip-y-mm` grids,
while preserving the old single `--power` diagnostic.
It reports any first object contact, correct first contact,
and non-pot/non-scratch outcomes after correct contact,
plus an example shot if a correct contact occurs.
This is NOT a new Defense MacroCandidate, trained Micro
action, learned Q representation, or official-rule validator.

Important interpretation: zero Attack successes on these
states may primarily reflect the **10-degree actuator
restriction**, rather than a strategically difficult board.
Do not train/relabel the 31D mixed Candidate Q on
high-elevation shots without updating executable shot
parameters, encoding, teacher generation, and downstream
evaluation consistently. Analyze the feasibility/contact
limits first.

Small controlled parameter sweep:

```bash
python -m scripts.probe_cue_elevation_on_hard_states \
  --base-checkpoint goal_conditioned_micro_mixed_curriculum_100k.pt \
  --residual-checkpoint position_power_residual_mlp_2000.pt \
  --source-boards 5 --source-seed 90000 \
  --max-hard-states 2 --max-successful-direct 0 \
  --elevations 20 35 50 60 \
  --angle-offsets -10 -5 0 5 10 \
  --powers 0.65 1.0 1.5 \
  --tip-y-mm 0 5 \
  --device cuda
```


## Stage 6i: first contact is possible but Defense value is still unknown

The 4 elevation x 5 heading offsets x 3 powers x 2 tip-Y grid
on two physical post-SAFETY rail-hard states found:
- Source board 1: 20/35/50/60 deg physical validity
  0/6/33/42 out of 60; **0** correct first contacts at all
  elevations, even when shots were physically allowed.
- Source board 3: validity 18/48/60/60 out of 60; correct
  first contacts **10/11/0/0**, respectively. All 21
  correct contacts were nonpotting and nonscratching.
- The earlier 10-deg trained Attack candidate set remained
  physically invalid (12/12 and 9/9) with code 128.

Therefore (a) 10-degree impossibility creates a spurious
hard-board label; (b) increasing elevation can unblock the
cue, but higher is not always better for object-ball contact;
(c) there are possible physical Safety-like contacts from
board 3, but no demonstration that they lower opponent
threat or pass official billiards legality.

An isolated diagnostic addition now checks the post-shot
state using SimplifiedGameRules and reports
`simplified_safety_eligible`. With
`--threat-samples-per-elevation N`, it also evaluates up
to N of those eligible shots per elevation under the
**existing fixed 10-degree opponent DIRECT Micro**, reporting
the minimum successful opponent DIRECT count over that
nonrandom, grid-ordered sample. This does not re-label
the 31D Q, create executable elevated MacroCandidates, or
establish any real-game success rate. The opponent can also
be blocked by the same 10-degree actuator restriction, so
zero observed DIRECT success may overstate tactical Safety.

First try a small threat sample:

```bash
python -m scripts.probe_cue_elevation_on_hard_states \
  --base-checkpoint goal_conditioned_micro_mixed_curriculum_100k.pt \
  --residual-checkpoint position_power_residual_mlp_2000.pt \
  --source-boards 5 --source-seed 90000 \
  --max-hard-states 2 --max-successful-direct 0 \
  --elevations 20 35 50 60 \
  --angle-offsets -10 -5 0 5 10 \
  --powers 0.65 1.0 1.5 --tip-y-mm 0 5 \
  --threat-samples-per-elevation 3 \
  --device cuda
```

For training or deployed action selection, elevation must
be carried consistently in the physical-shot representation,
the macro encoding, execution and next-state teacher. That
is deliberately deferred until physical and simplified-game
diagnostics justify it.


## Stage 6j: opponent-elevation-aware DIRECT physical diagnostic

The full post-SAFETY threat scan found 21 eligible elevated SAFETY
shots from source board 3 (10 with cue elevation 20 degrees; 11
with 35 degrees). Under the existing opponent 10-degree learned
DIRECT Micro, the best recorded result retained one successful
DIRECT candidate at each elevation (none reduced the count to 0).
Source board 1 still had no eligible Safety in the tested grid.

This does NOT mean elevated cue control would make a strategic
Safety better or worse: the defender can change theta while
the opponent in that metric is still restricted to theta=10.

A separate `src/macro/elevated_direct_threat.py` diagnostic
enumerates, for each geometrically feasible DIRECT candidate,
physical shots aimed at the ghost-ball position for that target
ball and pocket, with small heading offsets, power, cue-tip Y
and elevation options. A successful ROUTE requires:
- FastFiz precondition 0;
- first collision with the intended object ball;
- the intended ball pocketed in its specified pocket;
- no cue scratch.

The count is over **distinct (ball, pocket) routes**, not the
number of parameter combinations that made the same route.
The grid is neither a trained multi-elevation SAC policy nor
an exhaustive opponent action search. In particular, a failed
route may succeed using a shot outside the grid.

To avoid lengthy nested simulation, pass `--opponent-elevated-grid`
and a small `--threat-samples-per-elevation` (e.g. 2) to
`probe_cue_elevation_on_hard_states.py`. On each selected
post-Safety board, it now reports:
- legacy 10-degree Micro minimum successful DIRECT candidates;
- physical elevation-grid minimum and maximum successful DIRECT
  routes across exactly the same sampled Safety poststates;
- physical shot attempts and validity totals.

These two metrics use different action-selection mechanisms.
Interpret disagreement as a possible action-space limitation,
not as a fair learned-policy comparison. No Q targets, trained
checkpoints, MacroCandidate encodings, or game rules are altered.
