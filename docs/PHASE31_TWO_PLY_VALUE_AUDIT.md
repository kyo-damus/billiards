# Phase31: root Q versus first successor V and second-shot responses

## Why

The same 7 failure-focused recorded boards gave:

| Variant | Depth 1 fouls/pots | Depth 2 fouls/pots |
| --- | --- | --- |
| C = Phase29 State V | 1 / 3 | 0 / 4 |
| C-prime = Phase31 State V | 1 / 5 | 3 / 1 |

All other settings (frozen Macro Policy, 5D Micro, WPA-style rules,
FastFiz, positions and 24 simulations) were held fixed.

In two of the three C-prime scratch boards, depth 1 chose a legal
shot, and depth 2 selected a scratch. This is evidence of an
interaction between learned value and deeper search *on these boards*,
not proof that any depth, PUCT rule, or sign backup is incorrect.

This script inspects the three C-prime scratch boards from the earlier
root search and physically verified same-board evaluation. It
distinguishes:
- One-step successor Value, from the next player's perspective,
  and the corresponding discounted value from the root player's
  perspective.
- MCTS backed-up root Q (a mean over first-step and later search
  simulations; **not** a one-shot estimate).
- The actual second-ply responses explored below each of
  the two competing root candidate goals, including their visit
  counts, Q from the response player's perspective, and root-perspective
  translations.
- One-step FastFiz foul, pocketed balls, next player and ball-in-hand,
  using independent restores of the identical root state.

The historical C/C-prime depth-2 chosen goals, root visit counts
and candidate lists must exactly replay. Existing checkpoint hashes
and evaluation identities are checked before loading models.

## Commands on DGX

    cd ~/research
    git fetch origin phase31-two-ply-value-audit-v1
    git worktree add -b phase31-two-ply-value-audit-v1 \
      ~/research_phase31_two_ply origin/phase31-two-ply-value-audit-v1
    ln -s ~/research/artifacts ~/research_phase31_two_ply/artifacts
    cd ~/research_phase31_two_ply

    source ~/miniconda3/etc/profile.d/conda.sh
    conda activate bill_sub1
    export BILLIARDS_FASTFIZ_BUILD_DIR=/tmp/billiards-fastfiz-wpa-phase22

    python -m pytest -q \
      tests/pipelines/evaluation/test_phase31_two_ply_values.py \
      tests/pipelines/evaluation/test_phase31_depth_factorial.py

    read -rp 'GPU番号: ' gpu
    export CUDA_DEVICE_ORDER=PCI_BUS_ID
    export CUDA_VISIBLE_DEVICES="$gpu"

    python -u -m scripts.experiments.diagnose_phase31_two_ply_values \
      --evaluation-dir artifacts/runs/phase31_C_vs_Cprime_32pairs_v1 \
      --root-diagnostic-dir artifacts/runs/phase31_C_vs_Cprime_mcts_diag_16_v1 \
      --output-dir artifacts/runs/phase31_two_ply_value_audit_3_v1 \
      --device cuda

    cat artifacts/runs/phase31_two_ply_value_audit_3_v1/summary.json

Output:
- summary.json: compared board IDs, how many first successor Value
  rankings were reversed by deeper MCTS Q, by model.
- details.json: two candidate goals (original C and C-prime choices)
  under both C and C-prime value networks; physical successor status,
  root Q/visits, intermediate State V predictions, and visited
  second-shot replies. Specifically inspect
  models.final.C_choice and models.final.Cprime_choice.

Default three seed:row pairs:
- 1690522:197
- 1690144:40
- 1690324:190

To inspect another documented disagreement board, pass repeated
--seed-row SEED:ROW arguments; no source games are changed.

## Next decision

1. If deeper response-ply MCTS values reverse the ranking,
   inspect which second-shot responses and child leaf values cause it.
2. If both values and first-shot physical outcomes are already
   inconsistent, inspect calibration on counterfactual poststates,
   especially opponent ball-in-hand after a scratch.
3. If an exact-sign/discount/cue placement bug is observed,
   fix that specific bug in a separate branch with deterministic
   unit tests. Do **not** simultaneously alter network architecture,
   reward targets, Position Attack, or MCTS simulations.
4. If the tree is structurally correct but uncalibrated, do
   a held-out value-calibration and Micro-reliability study next,
   then evaluate one targeted improvement in fresh paired matches.

Important: These boards were selected from C-winning evaluation
matches and are intentionally biased toward disagreements. One-shot
outcomes and two-ply averages are not direct estimates of general
match win rate.
