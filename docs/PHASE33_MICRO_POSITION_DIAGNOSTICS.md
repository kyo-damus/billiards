# Phase33: Micro POSITION_ATTACK failure decomposition

Read-only, development-only diagnostic. Does not change SAC learning,
Macro Policy / State V, or FastFiz physics.

## Observation from the initial Phase28a Micro 32-case audit

| Dataset | Checkpoint | Exact pot | Full Position success |
| --- | --- | ---: | ---: |
| train | 100k | 20/32 | 2/32 |
| heldout | 100k | 23/32 | 2/32 |
| train | 20k | 16/32 | 2/32 |
| heldout | 20k | 17/32 | 0/32 |

Full POSITION success requires a legal exact pot, continuation
(shooter keeps the table and next ball remains), and cue stop within
the requested goal radius. Failure after pot can reflect either of
the latter conditions. The goal generator normally uses 0.10 m radius.
Current Micro reward adds +2.0 for the exact pot and at most +0.5
for the stop component only inside that radius and with continuation.

## DGX: create separate Phase33 worktree

    cd ~/research
    git fetch origin phase33-micro-position-diagnostics-v1
    git worktree add -b phase33-micro-position-diagnostics-v1 \
      ~/research_phase33 origin/phase33-micro-position-diagnostics-v1
    ln -s ~/research/artifacts ~/research_phase33/artifacts
    cd ~/research_phase33
    source ~/miniconda3/etc/profile.d/conda.sh
    conda activate bill_sub1
    export BILLIARDS_FASTFIZ_BUILD_DIR=/tmp/billiards-fastfiz-wpa-phase22

    python -m pytest -q \
      tests/pipelines/evaluation/test_audit_nine_ball_micro_generalization.py

## Larger read-only evaluation on actual train/holdout rollouts

    micro=artifacts/runs/phase28a_micro_frozen_break_lr1e-4_100000_v1
    summary="$micro/summary.json"
    train_dir=$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["training_data"]["rollout_dirs"][0])' "$summary")
    holdout_dir=$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["training_data"]["holdout_rollout_dir"])' "$summary")
    python -m scripts.evaluation.audit_nine_ball_micro_generalization \
      --training-summary "$summary" --train-dir "$train_dir" \
      --holdout-dir "$holdout_dir" \
      --checkpoint "$micro/micro_sac_5d.pt" \
      --base-checkpoint "$micro/checkpoints/step_00020000.pt" \
      --goal-source candidates --outcome-position \
      --max-matches-per-split 64 --max-cases-per-goal 128 \
      --output "$micro/micro_position_decomposition_128_v2.json"

Inspect the per-model POSITION_ATTACK.position_decomposition field:
- legal_exact_pots
- continued_after_exact_pot
- within_goal_radius_after_exact_pot
- continued_and_within_goal_radius
- continuation_rate_given_exact_pot
- within_radius_rate_given_continued_pot
- stopping_distance_m_after_continued_pot (mean/median/p90)
- within_absolute_radius_m_after_continued_pot (0.10, 0.20, 0.30)

Rate denominators differ: overall success uses all attempted goals,
while conditional position rates use legally pocketed shots only.

If continuation is low, investigate the turn/next-ball failure.
If continuation is high but stopping precision low, investigate
the stop-reward gradient and goal feasibility. Train/heldout gaps
need careful matched-board interpretation: a small or nonuniform
holdout set cannot prove overfitting.

This change records additional measurements from the SAME physical
shots. It does not simulate extra shots, retrain, or overwrite models.
