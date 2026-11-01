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
