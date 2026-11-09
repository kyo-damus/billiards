# テストの配置と実行方法

研究のテストは用途に応じて段階的に整理する（2026-11-09、Phase 3開始）。

- `tests/unit/pipelines/`: 学習・推論の共通処理、旧import互換性の局所的検証
- `tests/unit/tools/`: 実験管理・保全用ツールの局所的検証
- `tests/unit/game/`: ゲームルールのデータ型・インターフェース、簡易GameRules
- `tests/unit/env/`: ショット幾何・環境の局所契約
- `tests/unit/macro/`: Macro候補表現・Policy/Qネットワーク・局所評価
- `tests/unit/macro/position/`: Position候補生成、到達可能性・feasibility
- `tests/unit/micro/`: SAC・Position Residual、適応的Position Policy、ゲート処理
- `tests/unit/goal/`: TacticalGoal・Goal条件付け・報酬・Safety Goal
- `tests/unit/defense/`: 防御候補生成、相手脅威・仰角の局所評価
- `tests/test_*.py`: 未移動のテスト。依存関係・fixtureを確認してから段階的に分類
- `tests/integration/physics/`: FastFiz実物理とMCTSの結合動作を確認するテスト
- `tests/integration/agent/`: 階層型Agentのエンドツーエンドsmoke
- `tests/integration/defense/`: FastFizを伴うDefenseショット・Macro遷移
- `tests/pipelines/macro/`: Macro Candidate Policy/Q、State V、反復更新と2-step教師・Mixed候補dataset
- `tests/pipelines/rollout/`: Rollout収集/保存、MCTS訪問分布教師とMC return教師の生成
- `tests/pipelines/micro/`: Position Residual/Success・Probability Gateの教師・小規模学習契約
- `tests/pipelines/evaluation/`: Candidate Policy/Q・MCTS・Residual・Safety難盤面・Cue仰角の評価CLI契約

## 関連テストの実行例

`~/research` で次を実行する。

```bash
python -m pytest \
  tests/unit/pipelines/test_training_utils_compat.py \
  tests/unit/pipelines/test_checkpoint_loading_compat.py \
  tests/unit/pipelines/test_component_factories_compat.py \
  tests/unit/tools/test_prepare_run.py \
  tests/unit/game/test_game_rules_interface.py \
  -v
```

`python -m pytest tests/unit/pipelines/ -v` のように分類フォルダ単位の実行も可能。移動したファイルは旧パスで指定しても見つからないため**新パスを使うこと**。未移動のテストは従来のパスで実行する。FastFiz統合テスト4ファイルは上記の新しい場所を使い、従来通りネイティブ物理エンジンが必要。ファイルの内容・assertionは移動前から変更していない。

過去の検証ログや設計履歴のテストファイル名は、当時の記録としてそのまま保持する。

## Phase 3第2段階の実行例

```bash
python -m pytest \
  tests/unit/macro/ \
  tests/unit/micro/ \
  tests/unit/goal/ \
  -v
```

Macro5・Micro3・Goal3ファイルの内容は変更していない。元の単体テストは移動後のパスで参照する。既存のテストシナリオは増減していない。

## Phase 3第3段階の実行例

```bash
python -m pytest \
  tests/integration/physics/ \
  tests/integration/agent/ \
  -v
```

新配置の4ファイルは**内容未変更**。この分類は単体テストであることを主張するものではなく、FastFiz実物理や階層的な接続を検証する分類である。学習時間を要する本実験は含まれない。

## Phase 3第4段階の実行例

```bash
python -m pytest \
  tests/unit/macro/test_mcts.py \
  tests/unit/macro/test_candidate_mcts.py \
  tests/unit/macro/test_mixed_candidate_mcts.py \
  tests/pipelines/macro/ \
  tests/pipelines/rollout/ \
  -v
```

9ファイルは関数・assertionの変更なしに移動した。単体MCTSのテストと学習パイプラインのテストを、物理統合テストとは別に選択できるようにした。既存のパスを直接指定していたコマンド・資料は移動後のパスに読み替えること。

## Phase 3第5段階の実行例

```bash
python -m pytest \
  tests/unit/defense/ \
  tests/unit/goal/test_safety_goal.py \
  tests/integration/defense/ \
  tests/pipelines/micro/ \
  -v
```

10ファイルの内容は変更していない。`tests/integration/defense/` の実物理ケースはFastFizライブラリに依存する。既存の`tests/unit/goal/`全体は第2段階で検証済みのため、ここでは移した`test_safety_goal.py`だけを指定する。

## Phase 3第6段階の実行例

```bash
python -m pytest \
  tests/unit/macro/test_macro_action_generator.py \
  tests/unit/macro/position/ \
  tests/unit/env/ \
  tests/unit/game/test_simple_game_rules.py \
  tests/unit/micro/test_adaptive_position_policy.py \
  tests/unit/micro/test_position_blend_gate.py \
  tests/pipelines/micro/test_position_success_gate.py \
  tests/pipelines/micro/test_position_probability_gate.py \
  tests/pipelines/micro/test_outcome_witness_bc.py \
  tests/pipelines/evaluation/ \
  -v
```

上記15ファイルは**内容変更なし**の配置変更。元のルートのパスは廃止され、従来の `python -m scripts.*` の呼び出しは変更しない。

## Phase 3第7段階の実行例

```bash
python -m pytest \
  tests/pipelines/macro/test_build_mixed_candidate_value_dataset.py \
  tests/pipelines/macro/test_macro_candidate_value_targets.py \
  tests/pipelines/rollout/test_collect_mixed_rollouts.py \
  tests/pipelines/rollout/test_mixed_rollout_dataset.py \
  tests/pipelines/evaluation/ \
  -v
```

第7段階の13ファイルは内容変更なしの移動。これらの評価関数・データ生成関数のテストは、新しい分類先で実行する。旧パスは存在しない。残る9ファイルは次段階に分類する。
