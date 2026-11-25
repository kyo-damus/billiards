# Phase31: C vs C-prime value-head diagnostics (development)

## Observed evidence (2026-11-25)

Both variants share the Phase28a Micro, frozen Phase28b Macro Policy,
nine-ball rules, FastFiz, search budget, and rack split. Only State V differs.

- C: Phase29 final State V, SHA256
  `890e1cd515d1972b04966caea39019bbb758ad57be1521365a7df0e69780396a`.
- C-prime: Phase31 value-only iteration_001 State V, SHA256
  `3743d650f09644ab7603133fdafbae7c733ec16da986c2b19b07f9f9fd7a9533`.
- C vs C-prime: 32 pairs, 64 complete race-to-N matches,
  C 55 wins vs C-prime 9; 24 both-for-C pairs,
  1 both-for-C-prime, 7 split.
- State V only, no Policy or Micro learning:
  iteration 0 best_epoch=0 (weights identical to original C);
  iteration 1 best_epoch=2 (weights changed).
- Per-set validation prediction MSE:
  Phase29 iter3: C 0.5291 vs C-prime 0.5464 (C-prime worse).
  Phase31 iter0: C 0.6376 vs C-prime 0.6643 (worse).
  Phase31 iter1: C 0.4989 vs C-prime 0.4790 (better).
- Fraction of predictions outside [-1,1]:
  Phase29 iter3: 0.070 vs 0.157;
  Phase31 iter0: 0.077 vs 0.164;
  Phase31 iter1: 0.057 vs 0.145.

**Interpretation:** C-prime gains on the selected recent validation split
but loses modestly on earlier sampled splits, while match strength
collapses. This is consistent with distribution dependence / possible
forgetting but does not prove it; nor can aggregate MSE explain why
match wins change so much. Clip-rate growth is associated, not proof
that clipping causes the loss.

## Same-state MCTS diagnostic

The source is the already-completed paired evaluation, NOT new games.
Replay identical logged C-prime acting states in C's and C-prime's
MCTS, with their exact stored checkpoints. This compares root visits,
empirical Q, prior, and the chosen Macro goal without playing additional
matches or modifying training artifacts.

On DGX:

    cd ~/research
    git fetch origin phase31-value-mcts-diagnostics-v1
    git worktree add -b phase31-value-mcts-diagnostics-v1 \
      ~/research_phase31_diag origin/phase31-value-mcts-diagnostics-v1
    ln -s ~/research/artifacts ~/research_phase31_diag/artifacts
    cd ~/research_phase31_diag

    source ~/miniconda3/etc/profile.d/conda.sh
    conda activate bill_sub1
    export BILLIARDS_FASTFIZ_BUILD_DIR=/tmp/billiards-fastfiz-wpa-phase22
    python -m pytest -q tests/pipelines/evaluation/test_phase31_mcts_diagnostics.py

    read -rp 'GPU番号: ' gpu
    export CUDA_DEVICE_ORDER=PCI_BUS_ID
    export CUDA_VISIBLE_DEVICES="$gpu"

    python -u -m scripts.experiments.diagnose_phase31_mcts_choices \
      --evaluation-dir artifacts/runs/phase31_C_vs_Cprime_32pairs_v1 \
      --max-states 16 --random-seed 42 \
      --winner base --acting-model final \
      --device cuda \
      --output-dir artifacts/runs/phase31_C_vs_Cprime_mcts_diag_16_v1

Files:
- `summary.json`: chosen disagreement count/rate, prior mismatch,
  visit-distribution total variation (normalized L1 / 2), whether the
  re-run reproduces the historical logged action.
- `details.json`: per-board candidate encoded IDs, visits, priors,
  empirical root Q values and chosen goals for both models.

## Validity and limits

- Sample one non-break, non-table-choice position from each of up to
  16 distinct games **won by C**, where **C-prime was acting**.
  Sampling is deterministic for the given random seed. This is a
  *failure-focused* diagnostic, not an unbiased test-state sample.
- A difference in selected shot does not prove C-prime's choice is
  worse in that same state. That requires an independent, common
  downstream outcome evaluation or larger paired tests.
- Prior differences should be negligible when candidate sets match,
  since Macro Policy is frozen. If they are not, investigate input
  state/feature or candidate-generation reconstruction.
- If the replayed C-prime action often differs from the originally
  logged action, treat the diagnostic as non-reproducible until the
  source is audited.
- This isolates **MCTS decision sensitivity** rather than
  exhaustive causal mechanisms. Source trajectories are observational
  and come from matches where C-prime lost.
- Existing trained checkpoints and evaluation artifacts are read-only.
  New diagnostic outputs use a fresh path and refuse overwrites.
- Development data from Phase29 were involved in the choice of C;
  do not treat these as untouched test observations.
