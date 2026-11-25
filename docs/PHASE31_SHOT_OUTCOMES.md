# Phase31: C vs C-prime one-shot physical outcome audit

## Motivation

The frozen-Micro / frozen-Policy Phase31 State V continuation C-prime lost
9-55 against C in 32 paired test matches (64 completed). On the earlier
16-board root-MCTS replay, 7 boards changed their chosen Macro goal, while
candidate lists and Policy priors remained equal. All seven changes
were accompanied by changes in root MCTS visit distributions; they are
not cases of identical visits resolved solely by a Q-value tie break.

This script tests the **physical one-shot consequences** of each chosen
shot from the exact same recorded pre-shot board. It does NOT infer the
causal match-win impact of an action, claim an independent test set, or
simulate the rest of the match.

## Contents

- Verify all checkpoint SHA256 identities against the already-completed
  evaluation's `evaluation_config.json`.
- Verify original root diagnosis reproducibility.
- For the **7 mismatched choices** in existing root diagnosis, restore
  a single original shot position and re-run both C and C-prime search.
- Verify each replay's chosen goal and candidate visit counts match the
  previous diagnosis, and that C-prime's chosen shot controls match its
  historical match log.
- Simulate each selected shot exactly once through `simulate_nine_ball_shot`,
  the *same* FastFiz model and WPA-style NineBallRules, restoring the
  initial board before each shot.
- Save physical first contact, fouls and reasons, potted balls, declared
  exact target-pocket success, next shooter and ball-in-hand, cue-ball
  stop location and requested POSITION_ATTACK region, and number of
  geometrically clear pockets for the next legal object ball.
- Position goal is counted successful only when the declared ball is
  legally potted in the declared pocket **and** the cue rests within
  the requested stop region. Direct-pocket count is a geometry proxy,
  not a probabilistic opponent/continuation value. A ball-in-hand
  successor has no pre-placement direct-pocket count.

## How to run on DGX

    cd ~/research
    git fetch origin phase31-shot-outcome-diagnostics-v1
    git worktree add -b phase31-shot-outcome-diagnostics-v1 \
      ~/research_phase31_shots origin/phase31-shot-outcome-diagnostics-v1
    ln -s ~/research/artifacts ~/research_phase31_shots/artifacts
    cd ~/research_phase31_shots

    source ~/miniconda3/etc/profile.d/conda.sh
    conda activate bill_sub1
    export BILLIARDS_FASTFIZ_BUILD_DIR=/tmp/billiards-fastfiz-wpa-phase22

    python -m pytest -q \
      tests/pipelines/evaluation/test_phase31_mcts_diagnostics.py \
      tests/pipelines/evaluation/test_phase31_shot_outcomes.py

    read -rp 'GPU番号: ' gpu
    export CUDA_DEVICE_ORDER=PCI_BUS_ID
    export CUDA_VISIBLE_DEVICES="$gpu"

    python -u -m scripts.experiments.diagnose_phase31_shot_outcomes \
      --evaluation-dir artifacts/runs/phase31_C_vs_Cprime_32pairs_v1 \
      --root-diagnostic-dir artifacts/runs/phase31_C_vs_Cprime_mcts_diag_16_v1 \
      --output-dir artifacts/runs/phase31_C_vs_Cprime_shot_quality_7_v1 \
      --device cuda

Outputs are summary.json and details.json in that fresh output directory.
Artifacts are never overwritten or relabeled. Some physical native
shots may be rejected; rejected pairs are recorded as **not compared**,
never as billiards fouls or losses.

For a summary:

    cat artifacts/runs/phase31_C_vs_Cprime_shot_quality_7_v1/summary.json

## Interpretation

- `base` means C (original Phase29 State V); `final` means C-prime
  (continuation State V). All other checkpoints are held fixed.
- Potting a ball is not synonymous with the best positional outcome.
  A same-board single shot is not a match-win estimate.
- Study per-board differences in `details.json`, especially differences
  in foul status, exact target pocket, retained turn, actual cue stop,
  and following shot options.
- On 7 **failure-focused, C-winning** source boards, counts are descriptive,
  not unbiased population frequencies.
- If some boards fail reproducibility or physics checks, diagnose them,
  do not count them as evidence of C's superior strategy.
- Follow-on work: for important disagreements, use *identical independent*
  follow-on policies to evaluate the post-shot states; never let C's
  own Value head be the judge of C's decisions.
