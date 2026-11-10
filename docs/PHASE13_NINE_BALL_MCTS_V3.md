# Phase 13: nine-ball version-3 physical Macro MCTS (2026-11-10)

Branch: `phase13-nineball-mcts-v3` (from `phase12-nineball-turn-aware-qv`).
This is an implementation/integration smoke boundary; no game-strength claim.

## Inputs and semantics

- State: `RackState` (full balls 0..9), `NineBallContext` (consecutive
  fouls, push-out flag). Candidate: exact `NineBallGoal` (36D encoding).
- Macro Candidate Policy v3: PUCT priors. No artificial class balancing or
  top-level Attack/Safety quotas.
- Micro: **fixed, deterministic 5D SAC**. Each candidate's action is decoded
  from its own goal. Break/Safety use current explicit heuristic shots.
- Dynamics: `MultiBallFastFizSimulator.restore(parent_state)` before *each*
  candidate transition, followed by the native shot and `NineBallRules.apply`.
  Failed shots and fouls are NOT silently filtered out.
- Model returns: terminal winner yields +1/-1 from the *acting parent's*
  perspective. No one-shot physics teacher score or Macro-local reward is
  added. Nonterminal edges have zero immediate reward.
  Continuation is `gamma * sign(current_player, next_player) * child_value`.
- Optional v3 **State V** from Phase 12 is evaluated ONLY at search leaves,
  clamped to [-1,1] to match terminal MC targets. Without a V, leaf=0.
  The small Phase12 V is uncalibrated on unseen boards and needs evaluation.
- Phase12 **selected-action Q is deliberately not loaded/used** to predict
  unexecuted candidate values. Its off-policy counterfactual accuracy is
  unknown. A later MCTS-focused critic can be trained on visits/returns.
- Tree: PUCT with bounded simulations and shot depth, first-visit physical
  expansion, optional minimum root candidate visits, visit-count root choice
  with empirical-Q/prior tie break. Root visit distribution and empirical
  values are logged with the exact selected Macro Goal.
- When simulating a foul, the next player's cue-ball placement uses the
  existing `NineBallRackSession.place_cue_ball` validation and
  `GeometryNineBallAgent.cue_placement_candidates` order, matching the live
  collector. **Cue placement is heuristic, not a learned decision**.
- This is deterministic in-model lookahead. Does not quantify simulator
  uncertainty, Micro shot execution noise, referee-only WPA rule cases,
  position quality beyond value-network predictions, or opponent skill.
  Uses the current single-rack win/loss objective, not future race-to-N
  match-state conditioning.

## Files

- `src/game/nine_ball_mcts_v3.py`: search implementation and diagnostics.
- `src/game/nine_ball_mcts_agent.py`: learned Micro + v3 PUCT agent.
- `src/game/nine_ball_agents.py`: optional `macro_search` shot metadata.
- `scripts/collect_nine_ball_self_play.py`: `--agent-kind micro_mcts`,
  `--state-value-checkpoint`, `--mcts-simulations`,
  `--mcts-max-depth`, `--mcts-c-puct`, `--mcts-root-min-visits`.
  JSON trajectory includes `chosen_action.macro_search`.
- `tests/unit/game/test_nine_ball_mcts_v3.py`: in-memory tree tests.
  `tests/unit/game/test_nine_ball_self_play.py`: visit serialization.

## DGX test and bounded native smoke

Before switching branches, check a clean working tree. Existing checkpoint
files are left untouched. Choose one free GPU; maintain the already selected
`CUDA_VISIBLE_DEVICES` environment variable.

```bash
cd ~/research
conda activate bill_sub1
git status --short
git fetch origin
git switch -c phase13-nineball-mcts-v3 --track origin/phase13-nineball-mcts-v3
export BILLIARDS_FASTFIZ_BUILD_DIR=/tmp/billiards-fastfiz-phase7

python -m pytest \
  tests/unit/game/test_nine_ball_mcts_v3.py \
  tests/unit/game/test_nine_ball_self_play.py \
  tests/unit/game/test_nine_ball_learned_agent.py \
  tests/pipelines/macro/test_nine_ball_macro_qv_v3.py \
  tests/integration/game/test_nine_ball_self_play_native.py -q
```

First **1 match / 8 PUCT simulations / depth 2**:

```bash
python -m scripts.collect_nine_ball_self_play \
  --matches 1 --race-to 1 \
  --start-seed 450000 --max-shots-per-rack 30 \
  --agent-kind micro_mcts \
  --micro-checkpoint artifacts/runs/phase8_micro5d_first/outputs/micro_sac_5d.pt \
  --macro-policy-checkpoint artifacts/runs/phase11_nineball_macro_policy_smoke_v1/outputs/macro_policy_v3.pt \
  --state-value-checkpoint artifacts/runs/phase12_nineball_qv_smoke_v1/outputs/state_v_v3.pt \
  --mcts-simulations 8 --mcts-max-depth 2 \
  --output-dir artifacts/runs/phase13_nineball_mcts_smoke_v1/outputs \
  --device cuda
```

`--max-shots-per-rack` truncation is recorded as unfinished, not a
winner. To examine data without rerunning:

```bash
python - <<'PY'
import json
from pathlib import Path
f = Path('artifacts/runs/phase13_nineball_mcts_smoke_v1/outputs/match_seed_450000.json')
d = json.loads(f.read_text())
rows = d['trajectory']
searched = [r['chosen_action']['macro_search'] for r in rows]
print('finished', d['match_finished'], 'winner', d['match_winner'])
print('shots', len(rows), 'fouls', d['fouls'])
print('MCTS diagnostics', len([x for x in searched if x is not None]))
for r in rows[:3]:
    a = r['chosen_action']
    s = a['macro_search']
    print(a['goal_label'], 'candidate count', s['root_candidates'],
          'chosen visits', s['root_visits'][s['chosen_index']])
PY
```

Further work: run matched-seed Macro-only vs Macro+MCTS experiments with
identical Micro, official-rule limitations, no accidental shared seed
training/validation overlap, and sufficiently many matches; then decide
how to train a search-specific Q/V/visit Policy. No performance claims
from a one-match smoke.
