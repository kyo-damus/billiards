# Phase32 Micro generalization audit

The Micro currently learns one-shot terminal goal rewards from a replay
buffer: each transition has terminal=True and a zero next-state.
This is an intended division of labor with long-horizon Macro, but
the learned physical execution should be evaluated independently.

Phase28a froze the BREAK actor. Compare non-break goal precision
independently from BREAK. Training checkpoints were saved every
10k accepted physical shots.

## Dataset integrity

Do **not** call Phase29 self-play the Micro training dataset: Micro
was frozen in Phase29. Read the real inputs from the Phase28a
training summary's training_data.rollout_dirs and
training_data.holdout_rollout_dir. The audit tool validates these
against the original summary and refuses sampled board overlap.
This prevents misleading train/holdout labels.

## Commands (Phase32 worktree)

    cd ~/research_phase32
    source ~/miniconda3/etc/profile.d/conda.sh
    conda activate bill_sub1
    export BILLIARDS_FASTFIZ_BUILD_DIR=/tmp/billiards-fastfiz-wpa-phase22

    micro=artifacts/runs/phase28a_micro_frozen_break_lr1e-4_100000_v1
    summary="$micro/summary.json"

    python - "$summary" <<'PY'
    import json,sys
    s=json.load(open(sys.argv[1]))
    print("training:",s["training_data"]["rollout_dirs"])
    print("holdout:",s["training_data"]["holdout_rollout_dir"])
    print("goal_source:",s["goal_source"])
    print("outcome_position:",s["outcome_position"])
    PY

    train_dir=$(python -c 'import json,sys;print(json.load(open(sys.argv[1]))["training_data"]["rollout_dirs"][0])' "$summary")
    holdout_dir=$(python -c 'import json,sys;print(json.load(open(sys.argv[1]))["training_data"]["holdout_rollout_dir"])' "$summary")

    ls "$micro/checkpoints/step_00020000.pt"

    python -m scripts.evaluation.audit_nine_ball_micro_generalization \
      --training-summary "$summary" \
      --train-dir "$train_dir" \
      --holdout-dir "$holdout_dir" \
      --checkpoint "$micro/micro_sac_5d.pt" \
      --base-checkpoint "$micro/checkpoints/step_00020000.pt" \
      --goal-source candidates --outcome-position \
      --max-matches-per-split 16 --max-cases-per-goal 32 \
      --output "$micro/micro_generalization_audit_20k_vs_100k.json"

If Phase28a summary has different goal-source or position settings,
use the recorded ones; the audit rejects mismatched provenance.

## Results and limitations

Metrics: BREAK, DIRECT_ATTACK, POSITION_ATTACK, SAFETY; success,
legal rate, fouls, physical rejection; 20k vs 100k on true
training and untouched heldout samples. Position success means exact
pot *and* legal continuation *and* landing within cue-stop radius.

Train-vs-holdout gaps alone do not prove overfitting: case difficulty
can differ across these splits. A 20k-to-100k drop on unchanged
heldout cases motivates a deeper learning-curve investigation.
Low train and heldout success points instead to underlearning,
data diversity, reward or decoder limitations.

No self-play results or model weights are modified. Preserve fresh
seeds/layouts for the final thesis evaluation.
