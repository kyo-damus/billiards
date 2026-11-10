# Phase 12: 9-ball turn-aware selected-action Macro Q/V (2026-11-10)

Branch: `phase12-nineball-turn-aware-qv`, branched from
`phase11-nineball-macro-v3-bootstrap`. Existing checkpoints are not overwritten.

## Motivation / distinction from Phase 11

Phase 11 trained a **Macro Candidate Policy** from one-shot,
fixed-Micro physics teacher scores. It did not learn discounted game
returns. Phase 12 adds a deliberately separate **selected-action terminal
match-return** dataset and **behavior-policy Monte Carlo Q/V regression**.

It is *not* a counterfactual Q oracle for unexecuted candidates, a learned
optimal Q function, a MCTS value target, or an RL policy-iteration result.

## Data contract

- Only **finished race-to-1 single-rack matches** can contribute.
  Unfinished/truncated matches are skipped, never treated as terminal ties
  or zero-value games. Multi-rack matches are skipped until rack score is
  in the state representation and cross-rack returns are specified.
- For every chosen action, persist `chosen_action.goal_v3`, the exact
  36-dimensional candidate selected by the Macro agent. No guesses
  about POSITION stop targets from a reconstructed candidate list.
  Earlier self-play JSONs produced *before* this patch have no goal_v3
  and must be recollected for Macro Q training.
- `src/game/nine_ball_agents.ShotPlan` has an optional `macro_goal` field
  (geometry agents leave it None). `NineBallMicroSACAgent.plan_goal`
  attaches the chosen v3 goal. The match logger serializes it.
- v3 state features: 72D; chosen candidate: 36D; one label per executed
  shot; group by original match seed for validation.
- Terminal reward: +1 if the match winner is the acting player of the
  last step, else -1. Earlier immediate rewards are zero.
  `G_t = r_t + gamma * sign(p_t, p_(t+1)) * G_(t+1)`
  with `sign` +1 if the player stays the same, else -1.
  No post-terminal bootstrap. Equivalently each acting player receives
  `+/-gamma**remaining_shots` based on their eventual win/loss.
- Q(s,c): regression for **only** the actually selected candidate.
  V(s): on-policy state regression for the recorded behavior mixture.
  They have the same realized returns but different input representations.
- Metadata: v3, turn_aware=True, target_kind=
  `selected_action_terminal_match_return`, reward_kind=
  `terminal_match_win_loss`. Strict loaders in
  `src/game/nine_ball_value_v3.py` refuse incompatible checkpoints.
- There is no model for opponent skill yet: the result is valid for the
  self-play behavior mixture and current partial nine-ball rules.

## Added or changed code

- `scripts/collect_nine_ball_self_play.py`: record exact goal v3 in action.
- `src/game/nine_ball_learned_agent.py`: shot plan retains v3 Macro goal.
- `scripts/build_nine_ball_macro_returns_v3.py`: terminal Monte Carlo
  dataset from completed matches.
- `scripts/train_nine_ball_macro_qv_v3.py`: Q/V regression and match-heldout
  validation with constant-mean MSE baseline. Writes two v3 checkpoint types.
- `src/game/nine_ball_value_v3.py`: strict v3 Q/V model loaders for a
  future 9-ball MCTS integration.
- `tests/pipelines/macro/test_nine_ball_macro_qv_v3.py`: turn-sign,
  no truncated zero bootstrap, exact goal, train/validation, model loader.

## DGX quick smoke

Choose a free GPU and keep `CUDA_VISIBLE_DEVICES` to exactly one ID.

```bash
cd ~/research
conda activate bill_sub1
git status --short
git fetch origin
git switch -c phase12-nineball-turn-aware-qv --track origin/phase12-nineball-turn-aware-qv
export BILLIARDS_FASTFIZ_BUILD_DIR=/tmp/billiards-fastfiz-phase7

python -m pytest \
  tests/pipelines/macro/test_nine_ball_macro_qv_v3.py \
  tests/pipelines/macro/test_nine_ball_macro_v3_bootstrap.py \
  tests/unit/game/test_nine_ball_self_play.py \
  tests/unit/game/test_nine_ball_learned_agent.py \
  tests/integration/game/test_nine_ball_self_play_native.py -q
```

Collect *fresh* logs that include exact `goal_v3` (do not reuse older
phase11 outputs for selected-action Q training):

```bash
python -m scripts.collect_nine_ball_self_play \
  --matches 6 --race-to 1 --start-seed 440000 \
  --max-shots-per-rack 60 --agent-kind micro_sac \
  --micro-checkpoint artifacts/runs/phase8_micro5d_first/outputs/micro_sac_5d.pt \
  --macro-policy-checkpoint artifacts/runs/phase11_nineball_macro_policy_smoke_v1/outputs/macro_policy_v3.pt \
  --output-dir artifacts/runs/phase12_nineball_macro_selfplay_smoke_v1/outputs \
  --device cuda

python -m scripts.build_nine_ball_macro_returns_v3 \
  --rollout-dir artifacts/runs/phase12_nineball_macro_selfplay_smoke_v1/outputs \
  --output artifacts/runs/phase12_nineball_mc_dataset_smoke_v1/outputs/selected_mc.npz \
  --gamma 0.99

python -m scripts.train_nine_ball_macro_qv_v3 \
  --dataset artifacts/runs/phase12_nineball_mc_dataset_smoke_v1/outputs/selected_mc.npz \
  --output-dir artifacts/runs/phase12_nineball_qv_smoke_v1/outputs \
  --epochs 5 --device cuda
```

This is a pipeline smoke: 6 matches is **not enough** for game-strength,
Q generalization, or stable statistical findings. If fewer than two
matches complete, collect more; do not weaken the contract.

## Next

- Verify the three new runs and scoped tests on DGX; record failures
  and fix them before increasing scale.
- Improve opponent/SAFETY handling and obtain broader independent
  match data with policy-separated evaluation.
- Implement physics-based 9-ball MCTS with correct state/context
  restoration and turn-aware continuation; do not simply score
  arbitrary unexecuted goals using the selected-action Q as ground truth.
- Subsequently train Policy from visits and correct Q/V targets.
