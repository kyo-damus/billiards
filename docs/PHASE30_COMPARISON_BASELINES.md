# Phase30: adapted nine-ball comparison baselines

This phase is **independent** of Phase29 training and does not alter
its checkpoints, model architecture, or self-play output.

These are **experimental adaptations, not faithful paper replications**.

## Agents

| Agent | Tactical choice | Learned components |
| --- | --- | --- |
| Phase29 final | Candidate Policy + MCTS + State V | 100k Micro, Macro Policy, State V |
| pot_greedy | Shortest geometrically clear pocket for lowest ball; if none, aim at lowest ball; always accept table; no defense or positioning | Exactly the same 100k Micro |
| uniform_mcts | Tactical candidates + MCTS with uniform priors and no State V | Exactly the same 100k Micro |

Uniform MCTS retains the tactical candidate generator: it isolates the
learned Policy and State V, not the entire Macro layer. Pot Greedy
removes tactical planning. Neither replicates PickPocket, CueCard,
PoolMaster, or another published system verbatim.

The same Micro includes its frozen-break actor. The physical simulator,
WPA-style break rules, 9-rack race, alternate break policy and held-out
rack layout split are shared. Greedy retries cue-stick clearance with
the same-aim correction used by MCTS, but never ranks alternative
pockets using simulation. Native non-execution is an unfinished
match, not an invented win or billiards foul.

## DGX: separate checkout during Phase29 evaluation

    cd ~/research
    git fetch origin phase30-comparison-baselines-v1
    git worktree add -b phase30-comparison-baselines-v1 \
      ~/research_phase30 origin/phase30-comparison-baselines-v1
    ln -s ~/research/artifacts ~/research_phase30/artifacts
    cd ~/research_phase30
    source ~/miniconda3/etc/profile.d/conda.sh
    conda activate bill_sub1
    export BILLIARDS_FASTFIZ_BUILD_DIR=/tmp/billiards-fastfiz-wpa-phase22

Focused CPU tests before GPU evaluation:

    python -m pytest -q \
      tests/unit/game/test_nine_ball_comparison_baselines.py \
      tests/unit/game/test_nine_ball_learned_agent.py \
      tests/unit/game/test_nine_ball_mcts_v3.py

2-pair smoke on one free GPU (Phase29 may use the other GPU):

    read -rp 'GPU番号: ' gpu
    export CUDA_DEVICE_ORDER=PCI_BUS_ID
    export CUDA_VISIBLE_DEVICES="$gpu"
    mkdir -p artifacts/terminal_logs
    for method in pot_greedy uniform_mcts; do
      out="artifacts/runs/phase30_"$method"_smoke_2pairs_v1"
      log="artifacts/terminal_logs/phase30_"$method"_smoke.log"
      python -u -m scripts.eval_nine_ball_comparison_baselines_v1 \
        --training-manifest artifacts/runs/phase29_macro_fixed_micro100k_512x4_v1/run_manifest.json \
        --baseline "$method" --pairs 2 --start-seed 1460000 \
        --device cuda --output-dir "$out" 2>&1 | tee -a "$log" || break
    done

If both smoke tests are satisfactory, repeat with 32 pairs and
--start-seed 1470000 using new output directories.
Results are saved in summary.json with base (baseline) and final
(Phase29) win counts and unfinished matches. Add --resume to rerun
an interrupted same-directory evaluation. Do not reuse smoke
output directories for the definitive 32-pair evaluation.

The 2-pair smoke checks integration only, not game strength.


## Code organization

    src/game/agents/
      proposed/
        hierarchical.py       # Public entry to established proposal MCTS/Micro
        README.md             # Map of the validated Phase29 components
      baselines/
        pot_only_greedy.py    # Real pot-only implementation
        uniform_mcts.py       # Explicit uniform-prior MCTS ablation
    scripts/comparisons/
      eval_nine_ball_baselines.py  # The comparison runner

The previous script entry, python -m scripts.eval_nine_ball_comparison_baselines_v1,
and the old pot-only import remain supported via compatibility shims.
Phase29 source, training entrypoints, checkpoints, and artifacts are unchanged.
