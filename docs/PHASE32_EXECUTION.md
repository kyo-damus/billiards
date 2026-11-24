# Phase32 handoff: baseline implementation + Micro audit

**Two independent tracks:** 
(1) PickPocket-inspired probabilistic physical Monte Carlo v0,
(2) read-only deterministic Micro generalization diagnostics.
No Phase29/31 checkpoint or source file is overwritten.

## Checkout in dedicated DGX worktree

    cd ~/research
    git fetch origin phase32-pickpocket-micro-audit-v0
    git worktree add -b phase32-pickpocket-micro-audit-v0 \
      ~/research_phase32 origin/phase32-pickpocket-micro-audit-v0
    ln -s ~/research/artifacts ~/research_phase32/artifacts
    cd ~/research_phase32

    source ~/miniconda3/etc/profile.d/conda.sh
    conda activate bill_sub1
    export BILLIARDS_FASTFIZ_BUILD_DIR=/tmp/billiards-fastfiz-wpa-phase22

## CPU unit tests (no full test-suite command)

    python -m pytest -q \
      tests/unit/game/test_pickpocket_adapted_v0.py \
      tests/pipelines/evaluation/test_audit_nine_ball_micro_generalization.py \
      tests/unit/game/test_nine_ball_comparison_baselines.py

## One-board physics integration smoke (CPU)

    python -m scripts.comparisons.smoke_pickpocket_v0 \
      --rollout-dir artifacts/runs/phase29_macro_fixed_micro100k_512x4_v1/iteration_003/rollouts \
      --micro-checkpoint artifacts/runs/phase28a_micro_frozen_break_lr1e-4_100000_v1/micro_sac_5d.pt \
      --samples 2 --max-candidates 4 --device cpu \
      --output artifacts/runs/phase32_pickpocket_one_board_smoke_v0.json

The result reports planning diagnostics, the selected shot type,
and whether the native physics engine executed it. It does not
claim a game win-rate, even if the shot is legal.

## Paired 1-pair smoke, only after unit and one-board tests pass

    read -rp 'GPU番号: ' gpu
    export CUDA_DEVICE_ORDER=PCI_BUS_ID
    export CUDA_VISIBLE_DEVICES="$gpu"
    python -u -m scripts.comparisons.eval_nine_ball_baselines \
      --training-manifest artifacts/runs/phase29_macro_fixed_micro100k_512x4_v1/run_manifest.json \
      --baseline pickpocket_mc \
      --pickpocket-samples 2 --pickpocket-max-candidates 4 \
      --pairs 1 --start-seed 1480000 --device cuda \
      --output-dir artifacts/runs/phase32_pickpocket_pair1_smoke_v0

This is a slow test: every physical decision samples several
independent alternatives, and a race-to-9 match can require
hundreds of shots. Unlike the one-board check it is a whole match.
Use --resume on exactly the same arguments if interrupted. If
full-rack runtime is impractical, keep the one-board result and
first optimize or budget the search before longer matches.

## Micro diagnostic

See PHASE32_MICRO_GENERALIZATION_AUDIT.md. Use the actual
Phase28a training summary to identify original train+heldout
sources; Phase29 data was collected *after* Micro freezing and
must not be mislabeled as Micro training.

## Scope and next actions

- PickPocket v0: root-level stochastic outcome sampling and basic
  safety; not Smith's full two-ply MC or precomputed difficulty table.
- Chen–Li (2019): source-verified mapping in
  docs/baselines/chen_li_2019.md, but independent networks and
  search-training pipeline remain UNIMPLEMENTED.
- CueCard: separate source audit and implementation after the above.
- If Micro audit reveals a consistent generalization gap or low
  precision even on training boards, revise reward/goal feasibility,
  data diversity and SAC objective before another large-scale train.
- Phase31 proposal model learning and evaluation remain independent.

All current PickPocket comparison outcomes are exploratory, not
a thesis-level reproduction of a published model.
