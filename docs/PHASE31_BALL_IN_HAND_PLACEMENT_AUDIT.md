# Phase31 — first controlled repair candidate: ball-in-hand placement

## Evidence leading here

On three failure-focused boards where C-prime originally scratched:
- 1690522:197 (rack 5-2): second shooter was player 0, with
  placement (0.850, 2.116); selected reply missed and passed turn.
- 1690144:40 (rack 1-0): second shooter was player 1, with
  placement (0.266, 0.785); some direct-shot replies legally pocketed
  ball 4, another reply scratched.
- 1690324:190 (rack 3-6): second shooter was player 1, with
  placement (0.704, 1.783); selected reply missed, another scratched.

Existing MCTS player-perspective Q backup appears numerically
consistent. The immediate question is **not** "increase Value MLP
size" but whether its modeled opponent is artificially weak because
it receives a simplistic cue placement before its frozen Micro shot.

In existing live play and MCTS, the ball-in-hand candidate grid is
sorted by distance to the lowest-numbered target plus a weak center
preference; it does not consider legal pot pathways.

## This isolated experiment

The new code is intentionally *not wired into live matches or MCTS*.
This makes the results diagnostic and preserves Phase29 C exactly.
On each of three original recorded C-prime scratches:

1. Reconstruct the pre-shot state and original first-shot physical
   controls from the previously successful 7-board shot-quality audit.
2. Reproduce the original scratch and opponent ball-in-hand.
3. Enumerate the same 13 x 7-ish legacy placement grid.
4. Choose baseline = original first native-legal distance-ranked
   position; verify it reproduces the placement seen in the
   earlier second-ply search.
5. Choose alternative = first native-legal position after ranking by
   number of unobstructed target pockets, then shortest candidate
   cue-to-ghost + ball-to-pocket path.
6. Evaluate the SAME frozen trained Macro Policy and Micro SAC actor
   with **one deterministic policy-selected shot** at baseline and
   alternative, using the same FastFiz and WPA rules.
7. For context only, evaluate up to 17 positions, including sampled
   nearby and distant points. Label the best realized outcome
   sampled_oracle. This hindsight oracle uses the physical results
   and is NOT deployable as a policy or a fair real opponent.

Ranking does not guarantee best actual pot. The physical comparison
tests whether geometry ranking is a useful candidate.
The sampled oracle diagnoses **whether another placement available
to the same frozen Micro could do better**.

Do not infer full match strength, MCTS accuracy, or generalization from
three boards selected from C-winning matches.

## DGX steps

    cd ~/research
    git fetch origin phase31-ball-in-hand-placement-audit-v1
    git worktree add -b phase31-ball-in-hand-placement-audit-v1 \
      ~/research_phase31_bih origin/phase31-ball-in-hand-placement-audit-v1
    ln -s ~/research/artifacts ~/research_phase31_bih/artifacts
    cd ~/research_phase31_bih

    source ~/miniconda3/etc/profile.d/conda.sh
    conda activate bill_sub1
    export BILLIARDS_FASTFIZ_BUILD_DIR=/tmp/billiards-fastfiz-wpa-phase22

    python -m pytest -q \
      tests/pipelines/evaluation/test_phase31_bih_placement.py \
      tests/pipelines/evaluation/test_phase31_second_ply_physics.py

    read -rp 'GPU番号: ' gpu
    export CUDA_DEVICE_ORDER=PCI_BUS_ID
    export CUDA_VISIBLE_DEVICES="$gpu"

    python -u -m scripts.experiments.diagnose_phase31_bih_placement \
      --evaluation-dir artifacts/runs/phase31_C_vs_Cprime_32pairs_v1 \
      --root-diagnostic-dir artifacts/runs/phase31_C_vs_Cprime_mcts_diag_16_v1 \
      --shot-quality-dir artifacts/runs/phase31_C_vs_Cprime_shot_quality_7_v1 \
      --second-ply-dir artifacts/runs/phase31_second_ply_physics_3_v1 \
      --output-dir artifacts/runs/phase31_bih_placement_3_v1 \
      --max-placements 17 \
      --device cuda

    cat artifacts/runs/phase31_bih_placement_3_v1/summary.json

## Decision following this test

- Geometry improves the frozen policy's legal pots/fouls over
  baseline on these boards: prioritize integrating optional
  geometry placement in BOTH match play and MCTS, behind an
  experimental opt-in, then test on many unselected scenarios
  and paired matches.
- Sampled oracle improves but geometry doesn't: the frozen Micro
  is capable of useful shots from *some* other positions; refine
  placement ranking, without changing the network.
- Even sampled oracle fails: examine target/goal conditioning,
  Micro precision and local physical-shot search before scaling
  the Value model.
- Never treat oracle scores as real match outcomes, and don't
  simultaneously change MCTS horizon, Position Attack, Value
  model or reward targets.
