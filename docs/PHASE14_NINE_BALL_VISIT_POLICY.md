# Phase 14: nine-ball v3 Macro Policy update from MCTS visits (2026-11-10)

Work branch: `phase14-nineball-mcts-visits-policy`, branched from
`phase13-nineball-mcts-v3`.

## Objective, not an outcome claim

In Phase 13, 9-ball PUCT selected an actual v3 Macro goal with 5D Micro
and logged the **per-root candidate vectors, prior probabilities, visit
counts, selection, and simulations**. Phase 14 implements the first
one-step MCTS visit imitation update to a **version-3 Candidate Policy**.

1. Read `match_seed_*.json` from MCTS self-play (must have
   `chosen_action.macro_search`).
2. Verify candidate count and length, integer nonnegative visits,
   `sum(root_visits)==simulations`, valid priors, chosen index's maximal
   visit count, and exact match to the executed 36D Macro Goal.
3. Encode **observed pre-shot** rack + foul context into 72D state using
   `read_state_and_goal`. Candidates come from the ORIGINAL stored MCTS
   root, not regenerated at a different state or with updated code.
4. Discard single-candidate roots by default (usually breaks) because
   their one-hot targets provide no information about tactical selection.
   `--include-single-candidate` can disable this filter.
5. Build concatenated `candidates(C,36)`, `visits(C,)` with
   `offsets(N+1,)`, `states(N,72)`, `match_seeds(N,)`,
   `chosen_indices(N,)`, and a versioned target marker.
6. Fine-tune the existing v3 Candidate Policy from its Phase11 checkpoint,
   minimizing **root-wise cross entropy** between normalized MCTS visit
   counts and per-root model softmax logits. No class quotas or
   direct/position/safety equalization.
7. Split train/validation by **source match seed**. Compare initial and
   trained validation CE/KL and max-visit top-choice agreement. Retain the
   initial model if training worsens held-out visit CE; checkpoint remains
   compatible with `NineBallMicroSACAgent` and `NineBallMCTSAgent`.

**Unfinished matches are allowed** for Policy MCTS visit targets because
the root search is already complete at each recorded shot. This differs
from Phase12 terminal Monte Carlo Q/V, which requires *finished* matches
and excludes truncations. No winner or post-shot state is used to label
visit Policy targets.

**Do not claim game strength improved.** Distillation imitates an MCTS
search that uses a weak Micro, one-step trained prior, small State V, and
limited simulations. Search is an in-model teacher, not a physical oracle
or ground truth optimal policy. No Q/V update and no actual multiple-
iteration cycle are implemented in this phase.

## Files

- `scripts/build_nine_ball_mcts_visit_dataset_v3.py`: dataset converter
- `scripts/train_nine_ball_macro_visits_v3.py`: supervised policy update
- `tests/pipelines/macro/test_nine_ball_mcts_visits_v3.py`: dataset integrity,
  incomplete game inclusion, mismatch detection, grouped split, checkpoint
  adapter compatibility

## DGX verification

Before branch switch ensure no local uncommitted changes. New output
paths do not overwrite existing artifact directories. Keep the GPU
selection as before (exactly one `CUDA_VISIBLE_DEVICES` ID).

```bash
cd ~/research
conda activate bill_sub1
git status --short
git fetch origin
git switch -c phase14-nineball-mcts-visits-policy \
  --track origin/phase14-nineball-mcts-visits-policy

export BILLIARDS_FASTFIZ_BUILD_DIR=/tmp/billiards-fastfiz-phase7

python -m pytest \
  tests/pipelines/macro/test_nine_ball_mcts_visits_v3.py \
  tests/unit/game/test_nine_ball_mcts_v3.py \
  tests/unit/game/test_nine_ball_self_play.py \
  tests/unit/game/test_nine_ball_learned_agent.py \
  tests/pipelines/macro/test_nine_ball_macro_qv_v3.py \
  tests/integration/game/test_nine_ball_self_play_native.py -q
```

Collect at least 3 *different matches* with MCTS; a Phase13 single-match
smoke cannot provide a match-disjoint validation split:

```bash
python -m scripts.collect_nine_ball_self_play \
  --matches 3 --race-to 1 \
  --start-seed 460000 --max-shots-per-rack 30 \
  --agent-kind micro_mcts \
  --micro-checkpoint artifacts/runs/phase8_micro5d_first/outputs/micro_sac_5d.pt \
  --macro-policy-checkpoint artifacts/runs/phase11_nineball_macro_policy_smoke_v1/outputs/macro_policy_v3.pt \
  --state-value-checkpoint artifacts/runs/phase12_nineball_qv_smoke_v1/outputs/state_v_v3.pt \
  --mcts-simulations 16 --mcts-max-depth 2 \
  --output-dir artifacts/runs/phase14_nineball_mcts_visits_smoke_v1/outputs \
  --device cuda

python -m scripts.build_nine_ball_mcts_visit_dataset_v3 \
  --rollout-dir artifacts/runs/phase14_nineball_mcts_visits_smoke_v1/outputs \
  --output artifacts/runs/phase14_nineball_mcts_visits_dataset_smoke_v1/outputs/visits.npz

python -m scripts.train_nine_ball_macro_visits_v3 \
  --dataset artifacts/runs/phase14_nineball_mcts_visits_dataset_smoke_v1/outputs/visits.npz \
  --base-policy-checkpoint artifacts/runs/phase11_nineball_macro_policy_smoke_v1/outputs/macro_policy_v3.pt \
  --output artifacts/runs/phase14_nineball_macro_visit_policy_smoke_v1/outputs/macro_policy_visits_v3.pt \
  --epochs 5 --device cuda
```

If the collector produces only 1 eligible match after filtering, keep
the integrity checks and collect new data with different seeds; do not
silently split roots from the same match.

## Next work

Conduct paired, identical-seed evaluation comparing:
(i) old Macro prior + MCTS vs (ii) updated visit Policy + MCTS,
with identical Micro, State V, and simulation settings.
Also assess policy-only vs MCTS before any reinforcement or advantage
claim. Then implement bounded multi-iteration data collection, visit
policy update, optional separately validated Q/V update, independent
evaluation seeds, stable checkpoint selection, and seed/dependency
metadata. Explore diverse training boards before increasing GPU budget.
