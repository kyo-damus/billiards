# Phase31: audit the physical opponent replies after a scratch

The prior Phase31 2-ply value audit showed numerically consistent
root backups: for example, on seed 1690144 row 40 C-prime's scratch
Q is reconstructed from initial +0.59061 and five response values
+0.57881, +0.58023, +0.58706, +0.61782, +0.68607, averaging
to +0.60677. On seed 1690522 row 197, the initial root-view
of -1.0 and five responses of about -0.92425 average to -0.93688.

These are not signs of a directly reversed player-perspective sign.
The key unresolved question is whether the opponent's actually
simulated second shots *after receiving ball-in-hand* are poor,
or the learned State V over/under-values their resulting states.

This diagnostic enumerates the second-ply action edges that were
actually visited below C's selected first goal and C-prime's
selected first goal, in each of three original C-prime scratch boards.
It re-executes each second shot from the exact cue-placed
child board recorded by MCTS, with identical physics and rules.

It also reports the match score and the fair-rack baseline,
because a negative V for the opponent after ball-in-hand is
not automatically an error if the opponent is already far
behind in the match.

## Run on DGX

    cd ~/research
    git fetch origin phase31-second-ply-physics-audit-v1
    git worktree add -b phase31-second-ply-physics-audit-v1 \
      ~/research_phase31_replies origin/phase31-second-ply-physics-audit-v1
    ln -s ~/research/artifacts ~/research_phase31_replies/artifacts
    cd ~/research_phase31_replies

    source ~/miniconda3/etc/profile.d/conda.sh
    conda activate bill_sub1
    export BILLIARDS_FASTFIZ_BUILD_DIR=/tmp/billiards-fastfiz-wpa-phase22

    python -m pytest -q \
      tests/pipelines/evaluation/test_phase31_second_ply_physics.py \
      tests/pipelines/evaluation/test_phase31_two_ply_values.py

    read -rp 'GPU番号: ' gpu
    export CUDA_DEVICE_ORDER=PCI_BUS_ID
    export CUDA_VISIBLE_DEVICES="$gpu"

    python -u -m scripts.experiments.diagnose_phase31_second_ply_physics \
      --evaluation-dir artifacts/runs/phase31_C_vs_Cprime_32pairs_v1 \
      --root-diagnostic-dir artifacts/runs/phase31_C_vs_Cprime_mcts_diag_16_v1 \
      --output-dir artifacts/runs/phase31_second_ply_physics_3_v1 \
      --device cuda

    cat artifacts/runs/phase31_second_ply_physics_3_v1/summary.json

The full per-response details are in details.json. Find
models.final.Cprime_choice.second_replies to inspect
C-prime's model of the opponent's action after the first
shot that ultimately scratches.

## Interpretation

- Second-ply candidates may be the same opponent's alternatives
  from one board, so their counts are not independent test shots.
- Outcome quality is a one-shot physical indicator (pot, foul,
  retained turn, cue-ball position), NOT a whole-match win target.
- If an opponent who receives ball-in-hand repeatedly fails to pot
  or fouls, inspect Micro/ball-in-hand placement and whether the
  value function is simply predicting that weak opponent policy.
- If opponent shots are successful but Q still consistently
  predicts their match value decreasing, inspect leaf V calibration
  on search-generated successor states.
- A targeted fix must be independently evaluated with held-out
  match runs. Do not add an arbitrary foul penalty or grow the
  model merely because 2-ply search changes the action.
