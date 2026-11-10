# Phase 11: 9-ball Macro candidate-policy bootstrap (2026-11-10)

Branch: `phase11-nineball-macro-v3-bootstrap`, based on
`phase10-rollout-analysis-20261110`. Older two-object-ball and v2 model
checkpoints are deliberately not reused as v3 weights.

## Scope and scientific meaning

This is the first **9-ball v3 Macro Candidate Policy** training pipeline.
It is a supervised, *one-shot physics-oracle* bootstrap:

1. Read only pre-shot observations and rules context from recorded match
   trajectories (including micro self-play); never read next_state as teacher.
2. Generate all version-3 Macro goals on each board with no artificial
   DIRECT/POSITION/SAFETY balancing.
3. Use an **unchanged, deterministic 5D Micro** checkpoint to decode each goal
   into a shot. FastFiz restores the same source board before every candidate.
4. Apply the same NineBallRules to obtain a one-step result.
5. Train a variable-candidate Macro Policy to rank these physical results.
   Hold out *whole source matches*, not candidates or individual boards.

This is not a learned Macro Q, State V, MCTS, discounted match return,
independent game-strength test, nor an RL policy-iteration improvement claim.

### Provisional teacher score

- Foul: -2; three-foul rack loss: -5
- Legal rack win: +5
- Target into desired pocket: +2
- Target pocketed elsewhere: +0.4
- Correct-pot POSITION cue-stop proximity bonus: +0 to +0.25
- Legal no-pot and legal SAFETY: 0

In particular, SAFETY has no opponent-threat, turn-aware continuation,
or true defensive-quality estimate yet. These scores are a *temporary local
teacher*, not a final game objective. Keep them separate from v2 Macro local
rewards and future Q/return targets.

## Added files

- `scripts/build_nine_ball_macro_oracle.py`: physical counterfactual
  dataset. Output `.npz` includes states (N,72), candidates (C,36),
  scores (C,), offsets (N+1), match seeds (N,), and per-candidate
  diagnostics (foul, exact target pot, rack win, turn change).
- `scripts/train_nine_ball_macro_policy_v3.py`: supervised, listwise
  cross-entropy to a softmax of scores; held-out match validation.
  Output checkpoint is tagged `nine_ball_macro_candidate_policy_v3`
  with state_dim=72, candidate_dim=36, format_version=3.
- `NineBallMicroSACAgent.plan_goal`: directly decode a given candidate
  without allowing the agent to choose a different goal.
- `tests/pipelines/macro/test_nine_ball_macro_v3_bootstrap.py`: scoped
  tests for score semantics, candidate isolation, match hold-out, and
  compatibility with the existing v3 learned-agent loader.

## Run on the DGX

Use an unused GPU, with exactly one `CUDA_VISIBLE_DEVICES` ID. Source
the local conda environment and retain prior artifacts.

### Tests (before GPU work)

```bash
cd ~/research
conda activate bill_sub1
python -m pytest \
  tests/pipelines/macro/test_nine_ball_macro_v3_bootstrap.py \
  tests/unit/game/test_nine_ball_learned_agent.py \
  tests/pipelines/micro/test_train_nine_ball_micro_sac.py \
  tests/integration/game/test_nine_ball_self_play_native.py -q
```

### Tiny smoke: physics labels

```bash
export BILLIARDS_FASTFIZ_BUILD_DIR=/tmp/billiards-fastfiz-phase7
# Set exactly ONE free GPU, e.g. export CUDA_VISIBLE_DEVICES=2
python -m scripts.build_nine_ball_macro_oracle \
  --rollout-dir artifacts/runs/phase8_micro5d_selfplay_smoke_v1/outputs \
  --micro-checkpoint artifacts/runs/phase8_micro5d_first/outputs/micro_sac_5d.pt \
  --max-states-per-match 2 \
  --output artifacts/runs/phase11_nineball_macro_oracle_smoke_v1/outputs/oracle.npz \
  --device cuda
```

This reads three matches from the last Micro self-play, at most two
valid non-break states per match. If fewer than two distinct source
matches remain, the training stage deliberately refuses to split the data.

### Tiny smoke: supervised Macro Policy

```bash
python -m scripts.train_nine_ball_macro_policy_v3 \
  --dataset artifacts/runs/phase11_nineball_macro_oracle_smoke_v1/outputs/oracle.npz \
  --output artifacts/runs/phase11_nineball_macro_policy_smoke_v1/outputs/macro_policy_v3.pt \
  --epochs 3 \
  --device cuda
```

### Connect learned Macro into 9-ball self-play

```bash
python -m scripts.collect_nine_ball_self_play \
  --matches 3 --race-to 1 \
  --start-seed 430000 --max-shots-per-rack 60 \
  --agent-kind micro_sac \
  --micro-checkpoint artifacts/runs/phase8_micro5d_first/outputs/micro_sac_5d.pt \
  --macro-policy-checkpoint artifacts/runs/phase11_nineball_macro_policy_smoke_v1/outputs/macro_policy_v3.pt \
  --output-dir artifacts/runs/phase11_nineball_macro_selfplay_smoke_v1/outputs \
  --device cuda
```

Compare this **only as an integration smoke** against the geometric
Macro with the same Micro, seeds, and rule/simulator settings. Three
matches are not enough for meaningful outcome claims. Next phase:
9-ball candidate Q and state V based on properly labeled returns and
turn-aware MCTS (do not treat the oracle scores as Q-values).

The ChatGPT-side source change is committed to GitHub. CPU tests and DGX
physical runs must still be performed on the local environment before
claiming verification.
