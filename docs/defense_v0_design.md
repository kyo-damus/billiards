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
