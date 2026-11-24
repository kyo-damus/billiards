# Phase31 Macro Policy × State V ablation (development evaluation)

Goal: isolate the impact of Macro Policy vs State V while keeping
the Phase28a Micro, frozen BREAK actor, nine-ball rules, MCTS budget,
rack layout split, and race-to-9 configuration fixed.

These are **development tests**, not final independent comparisons.

| Variant | Macro Policy | State V |
| --- | --- | --- |
| A | Phase28b | Phase28b |
| B | Phase29 final | Phase28b |
| C | Phase28b | Phase29 final |
| D | Phase29 final | Phase29 final |

Existing seed 1450000 (32 pairs) evaluation:
- A vs D: D won 58, A won 5, one unfinished.
- A vs C: C won 63, A won 1, zero unfinished.

The results do *not* directly establish whether C or D is stronger.
Run A vs B and C vs D with breaker-balanced paired games.

## DGX: use a separate worktree

    cd ~/research
    git fetch origin phase31-macro-ablation-v1
    git worktree add -b phase31-macro-ablation-v1 \
      ~/research_phase31 origin/phase31-macro-ablation-v1
    ln -s ~/research/artifacts ~/research_phase31/artifacts
    cd ~/research_phase31

    source ~/miniconda3/etc/profile.d/conda.sh
    conda activate bill_sub1
    export BILLIARDS_FASTFIZ_BUILD_DIR=/tmp/billiards-fastfiz-wpa-phase22

    python -m pytest -q \
      tests/pipelines/evaluation/test_eval_nine_ball_macro_ablation.py \
      tests/pipelines/evaluation/test_eval_nine_ball_hierarchical_v3.py

### A vs B: Policy-only update

    read -rp 'GPU番号: ' gpu
    export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES="$gpu"
    python -u -m scripts.comparisons.eval_nine_ball_macro_ablation \
      --training-manifest artifacts/runs/phase29_macro_fixed_micro100k_512x4_v1/run_manifest.json \
      --base-variant A --final-variant B \
      --pairs 32 --start-seed 1450000 --device cuda \
      --output-dir artifacts/runs/phase31_A_vs_B_policy_only_32pairs_v1

### C vs D: effect of updated Policy when V is Phase29

Use the same command except these args:

    --base-variant C --final-variant D \
    --output-dir artifacts/runs/phase31_C_vs_D_policy_effect_32pairs_v1

For an interrupted run, repeat the same command plus --resume.
Each output contains evaluation_config.json and summary.json, and
one match record per seed+breaker; completed matches are read-only.

Interpret the results in the context of the completed self-play
training dataset and MCTS visit imitation. Unfinished matches are
neither wins nor losses. Do not reuse the development seed set as a
fresh final held-out benchmark.

### Follow-up choice (not automated)

- If C > D in direct head-to-head, consider learning State V while
  holding Macro Policy fixed in a later Phase31 training variant.
- If D > C, continue joint Policy/State V and evaluate new checkpoints.
- If inconclusive, inspect MCTS visit entropy, selected safety/position
  proportions, and source of terminal rewards before adding iterations.

No training data or checkpoints are modified by this evaluation.
