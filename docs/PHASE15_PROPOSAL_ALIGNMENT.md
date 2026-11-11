# Phase15: 9-ball階層型RLの提案手法整合性監査とMicro継続学習

2026-11-11。安全作業ブランチ: `phase15-nineball-coupled-learning-audit-20261111`。
原Phase14は保持、旧Checkpointと既存`artifacts/runs`は変更しない。

## 監査

- 研究の核: Macroが戦術・TacticalGoalを選択し、5D Goal-conditioned SACが実ショットを実行、物理と9-ballルールで更新、自己対戦と探索結果で改善。
- Phase14はPolicy bootstrap、選択行動Q/V、手番考慮MCTS、訪問分布Policy蒸留を実装。候補比率の均等化なし、選択行動Qはoff-policy未実行候補に使わないのが正しい。
- 差: Microは固定で、Macro Policy改善後の目標や盤面分布がMicroの学習へ戻らない。Break/SAFETYは明示的な仮ショットでMicroによる制御学習なし。
- 確定バグ: 5D Microの`micro_reward`が失敗したPOSITION_ATTACKにも手球停止位置ボーナスを与え得る。POSITIONの本来条件（指定ポケット成功と指定手球位置）を破る。今回修正。

## Phase15で追加

1. `--init-checkpoint`: 厳格なv3/108D/5D SAC Actor/Critic/Target/Alphaを初期値に使用。optimizer/replayは新ラン開始でリセットし、完全な停止位置からのresumeとは呼ばない。保存した`trained_steps`は累積。
2. `--goal-source logged`: MCTS自己対戦ログの`chosen_action.goal_v3`を、同じ盤面の候補集合から**同一36D**を照合して取り出す。非BreakのDIRECT/POSITIONのみ対象（Safety/Breakは現状明示shotでMicroが関与しない）。一致しないデータは再ラベルせず拒否。従来通りのランダム候補事前学習は既定`uniform`。
3. 位置ボーナスはポケット成功時のみ付ける回帰テストと、継続学習Checkpoint/Goal照合のテスト。

## DGX 確認（GPU1枚設定済み想定）

```bash
cd ~/research
conda activate bill_sub1
git fetch origin
git switch -c phase15-nineball-coupled-learning-audit-20261111 --track origin/phase15-nineball-coupled-learning-audit-20261111
export BILLIARDS_FASTFIZ_BUILD_DIR=/tmp/billiards-fastfiz-phase7
python -m pytest tests/pipelines/micro/test_train_nine_ball_micro_sac.py tests/unit/game/test_nine_ball_learned_agent.py tests/unit/game/test_nine_ball_mcts_v3.py -v
```

完走後に以下の新出力へMicro fine-tune。既存v3モデルは別フォルダに保護：

```bash
python -m scripts.train_nine_ball_micro_sac \
  --rollout-dir artifacts/runs/phase14_nineball_mcts_visits_smoke_v1/outputs \
  --init-checkpoint artifacts/runs/phase8_micro5d_first/outputs/micro_sac_5d.pt \
  --goal-source logged \
  --output-dir artifacts/runs/phase15_micro_logged_update_v1/outputs \
  --steps 5000 --warmup-steps 0 --device cuda
```

データ不足で結果が不安定になり得る。ここでは学習経路を確認し、次は新MicroとMacroを固定して別seed自己対戦→更新Macro→新Microの再学習という循環を実装・検証する。


## Phase15 反復統合ランナー

\`scripts/iterate_nine_ball_hierarchical_v3.py\` で、各反復を
MCTS自己対戦→root訪問分布のMacro Policy更新→終端MCリターンの選択行動Q/V更新→
**実際にMacro選択されたDIRECT/POSITION目標**から5D Micro SACを継続学習、
という順で実施する。次反復は更新済みPolicy, State V, Microを使用する。
これはMacroとMicroの相互**データ依存・交互更新**であり、両者の同時勾配最適化ではない。

- race-to-1のみ（状態72Dにマッチスコアがない）。未完了試合はvisit教師に使用可だが、Q/Vには使用しない。完成試合が2未満なら修正せず停止。
- 各段階で実行コマンド・標準出力/エラーのlog、モデルの初期SHA-256、反復ごとのseedを\`run_manifest.json\`に保存。失敗しても既存成果物を消去しない。途中再開はまだなく、新run rootを指定する設計。
- GPUは指定済み\`CUDA_VISIBLE_DEVICES\`1枚だけを使用。旧Phase14の評価や実験結果、旧checkpointは不変。
- 方策変更後の独立評価・異種モデル間head-to-headは次の段階。各反復同一自己対戦の勝率を実力向上と扱わない。

\`\`\`bash
cd ~/research
export BILLIARDS_FASTFIZ_BUILD_DIR=/tmp/billiards-fastfiz-phase7
python -m pytest \
  tests/pipelines/micro/test_train_nine_ball_micro_sac.py \
  tests/pipelines/macro/test_nine_ball_hierarchical_iteration_v3.py -v

python -m scripts.iterate_nine_ball_hierarchical_v3 \
  --base-micro-checkpoint artifacts/runs/phase8_micro5d_first/outputs/micro_sac_5d.pt \
  --base-policy-checkpoint artifacts/runs/phase14_nineball_macro_visit_policy_smoke_v1/outputs/macro_policy_visits_v3.pt \
  --base-state-v-checkpoint artifacts/runs/phase12_nineball_qv_smoke_v1/outputs/state_v_v3.pt \
  --iterations 2 --matches-per-iteration 32 \
  --mcts-simulations 24 --micro-steps 5000 \
  --output-root artifacts/runs/phase15_hierarchical_iteration_v1 \
  --device cuda
\`\`\`

今後：独立seedで旧・新Policyの対戦、先攻交代、完全ルールとより妥当なSAFETY、Micro回収データの多様化を優先。
