# Phase31 Policy-frozen State V continuation: experimental v1

This follows the 2x2 head-to-head development results:

- A (Phase28b Policy+V) vs B (Phase29 Policy, Phase28b V):
  B wins 38/64 = 59.4%.
- C (Phase28b Policy, Phase29 V) vs D (Phase29 Policy+V):
  C wins 39/64, D wins 25/64.
- A vs C: C wins 63/64.
- A vs D: D wins 58, A wins 5, and 1 unfinished.

These are *development* observations, not official independent
comparison results. They suggest a Policy-Value interaction, not proof
that the new Policy is intrinsically inferior.

## Change scope

Source branch starts from phase31-macro-ablation-v1. The training
program still executes the established self-play -> terminal-return
dataset -> Q/V sequence, but exposes two optional flags:

- --freeze-policy: skip visit imitation, reusing the exact initial
  Macro Policy checkpoint on every round.
- --warm-start-state-v: initialize each round State V from previous
  checkpoint, validate tags and dimensions, log its SHA-256. The
  optimizer is intentionally reset each round. Epoch 0 is an
  eligible validation checkpoint, so new data cannot *force* a worse
  within-round validation MSE relative to initialization.
- Macro Q remains trained from scratch on selected-action returns.
  It does NOT steer exploration; the MCTS uses State V. The Q model
  is retained for backwards-compatible pipeline outputs.
- The 5D Micro and its frozen break actor remain identical to Phase29.

This is **not an offline full replay of Phase29 trajectories**:
the next run collects its own on-policy matches with the strong C
configuration. One round starts from the Phase29 final State V.

## Setup in a new DGX worktree

    cd ~/research
    git fetch origin phase31-value-only-continuation-v1
    git worktree add -b phase31-value-only-continuation-v1 \
      ~/research_phase31_value origin/phase31-value-only-continuation-v1
    ln -s ~/research/artifacts ~/research_phase31_value/artifacts
    cd ~/research_phase31_value
    source ~/miniconda3/etc/profile.d/conda.sh
    conda activate bill_sub1
    export BILLIARDS_FASTFIZ_BUILD_DIR=/tmp/billiards-fastfiz-wpa-phase22

## Required focused tests before running GPU

    python -m pytest -q \
      tests/pipelines/macro/test_nine_ball_hierarchical_iteration_v3.py \
      tests/pipelines/macro/test_nine_ball_macro_qv_v3.py \
      tests/pipelines/macro/test_phase31_value_only_launcher.py \
      tests/pipelines/evaluation/test_eval_nine_ball_hierarchical_v3.py

## Non-writing plan preflight

    manifest=artifacts/runs/phase29_macro_fixed_micro100k_512x4_v1/run_manifest.json
    python -m scripts.experiments.run_phase31_value_only \
      --phase29-manifest "$manifest" \
      --iterations 1 --matches-per-iteration 8 \
      --qv-epochs 2 --start-seed 1600000 \
      --output-root artifacts/runs/phase31_value_only_smoke_8_v1 \
      --device cpu --plan-only

The preflight verifies the original Phase28a Micro and Phase28b
Policy hashes, Phase29 Micro identity, starting State V file, and
the recorded layout-split hash.

## Small on-policy smoke: eight matches, one iteration

    read -rp 'GPU番号: ' gpu
    export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES="$gpu"
    python -u -m scripts.experiments.run_phase31_value_only \
      --phase29-manifest "$manifest" \
      --iterations 1 --matches-per-iteration 8 \
      --qv-epochs 2 --start-seed 1600000 \
      --output-root artifacts/runs/phase31_value_only_smoke_8_v1 \
      --device cuda

No existing model or Phase29/30/31 run will be overwritten.
Resume an interrupted run by adding --resume to exactly
the same command. The manifest is fixed to one Git revision.

## After smoke checks

Inspect:
- run_manifest.json: status completed, round Policy path unchanged,
  new round State V different, Micro unchanged.
- qv/summary.json: train/validation MSE and best epoch (0 means
  the initial State V was retained).
- per-match completion, physics errors, and nonzero return count.

Next, train a medium candidate (e.g. 2x64 matches) on distinct new
training seeds (start_seed 1640000). Compare its final V against
the Phase29 starting C with the *normal* evaluator and a NEW
development evaluation seed (e.g. 1680000):

    python -u -m scripts.eval_nine_ball_hierarchical_v3 \
      --training-manifest artifacts/runs/<new_training_run>/run_manifest.json \
      --pairs 32 --start-seed 1680000 \
      --device cuda --output-dir artifacts/runs/<new_eval_run>

This evaluates C vs C' with the Macro Policy and Micro frozen.
Do not escalate immediately to large-scale final baseline comparisons.
Use unopened evaluation seeds/layouts for the final thesis evaluation.

## Notes

The FastFiz native overlap warnings observed at seed 1450180 on both
ablation logs, and other overlap output, did not stop those matches.
They should be investigated separately before final statistical
claims; avoid "fixing" a physics result by relabeling a foul or
fabricating a valid shot.
