# テストの配置と実行方法

研究のテストは用途に応じて段階的に整理する（2026-11-09、Phase 3開始）。

- `tests/unit/pipelines/`: 学習・推論の共通処理、旧import互換性の局所的検証
- `tests/unit/tools/`: 実験管理・保全用ツールの局所的検証
- `tests/unit/game/`: ゲームルールのデータ型・インターフェース、簡易GameRules
- `tests/unit/env/`: ショット幾何・環境の局所契約
- `tests/unit/macro/`: Macro候補表現・Policy/Qネットワーク・局所評価
- `tests/unit/macro/position/`: Position候補生成、到達可能性・feasibility
- `tests/unit/micro/`: SAC・Goal-conditioned SAC・Curriculum・Position Residual・適応的Position Policy、ゲート処理
- `tests/unit/goal/`: TacticalGoal・Goal条件付け・報酬・Safety Goal
- `tests/unit/defense/`: 防御候補生成、相手脅威・仰角の局所評価
- 2026-11-09のPhase 3第8段階で全76ファイルを分類。現在 `tests/test_*.py` の直下ファイルはなし
- `tests/integration/physics/`: FastFiz実物理、MicroBilliardEnv、PhysicsMacroTransitionとMCTSの結合動作
- `tests/integration/agent/`: 階層型Agentのsmoke、Mixed階層Rolloutの接続契約
- `tests/integration/defense/`: FastFizを伴うDefenseショット・Macro遷移
- `tests/integration/goal/`: MicroBilliardEnvを使用したGoal sampler、Outcome Goal sampler
- `tests/pipelines/macro/`: Macro Candidate Policy/Q、State V、反復更新と2-step教師・Mixed候補dataset
- `tests/pipelines/rollout/`: Rollout収集/保存、MCTS訪問分布教師とMC return教師の生成
- `tests/pipelines/micro/`: Position Residual/Gateの学習、Goal-conditioned SACのcheckpoint再開と並列Goal Sampling
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

`python -m pytest tests/unit/pipelines/ -v` のように分類フォルダ単位の実行も可能。移動したファイルは旧パスで指定しても見つからないため**新パスを使うこと**。第8段階の移動後、ルート直下の旧テストパスは存在しない。FastFiz統合テスト4ファイルは上記の新しい場所を使い、従来通りネイティブ物理エンジンが必要。ファイルの内容・assertionは移動前から変更していない。

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

## Phase 3第8段階の実行例（最後の9ファイル）

```bash
python -m pytest \
  tests/unit/micro/test_goal_conditioned_sac.py \
  tests/unit/micro/test_goal_curriculum.py \
  tests/pipelines/micro/test_goal_finetune.py \
  tests/pipelines/micro/test_parallel_goal_sampling.py \
  tests/integration/goal/ \
  tests/integration/physics/test_micro_billiard_env.py \
  tests/integration/physics/test_physics_transition.py \
  tests/integration/agent/test_mixed_hierarchical_rollout.py \
  -v
```

このコマンドは最終9ファイルの検証であり、全テストスイートの実行ではない。配置変更後の最終テスト完了は**ユーザー環境の結果を確認してから**判断する。

### 普段の用途別実行

- Macro候補・MCTSの小規模テスト: `python -m pytest tests/unit/macro/ -v`
- Macro Policy/Q/V学習・ターゲット: `python -m pytest tests/pipelines/macro/ -v`
- Microの学習・教師データ: `python -m pytest tests/pipelines/micro/ -v`
- 実物理での接続: `python -m pytest tests/integration/physics/ -v`

`tests/README.md` と `docs/PROJECT_STRUCTURE.md` の構成は2026-11-09現在。GitHub HEADとファイルツリーで実際の配置を確認すること。

## Mixed Rolloutの読み取り集計（2026-11-10追加）

学習や物理実行をせず、既存のMixed Rollout `rollout_seed_*.npz` から選択戦術、完了ゲーム、終了理由、ポケット・スクラッチ、手番交代、Macro局所報酬とゲームスコア差分、MCTS訪問教師として使えるroot数を確認する。

```bash
# 指定ディレクトリ直下のRolloutを集計
python -m scripts.analyze_mixed_rollouts --rollout-dir mixed_rollouts

# Policy反復出力のrollouts/を再帰的に検出
python -m scripts.analyze_mixed_rollouts \
  --rollout-dir macro_policy_iteration_smoke --recursive

# 指定済みRolloutだけを集計
python -m scripts.analyze_mixed_rollouts \
  --rollout mixed_rollout_smoke.npz \
  --json-output artifacts/runs/<run_id>/outputs/rollout_summary.json
```

`<run_id>` は実行時に固有のIDへ置き換える。`--json-output` は既存ファイルの上書きを拒否する。ディレクトリ検索は `rollout_seed_*.npz` のみを対象にし、その他の教師データNPZは除外する。

**指標の注意点:** `completed_games` は `rollout_terminated=True` と `stop_reason=game_terminated` を満たす件数（勝率ではない）。ポケット数は、物理状態の的球ポケットフラグが新しく立った回数（狙ったポケットへの成功率ではない）。スクラッチは手球のポケットフラグから算出。局所報酬とゲームスコア差分は別集計。新規の学習・勝敗評価・全pytestスイート実行は含まない。

実装本体: `src/pipelines/rollout_analysis.py`、CLI: `scripts/analyze_mixed_rollouts.py`、関連テスト: `tests/pipelines/evaluation/test_analyze_mixed_rollouts.py`。

### 候補が生成されたか・選ばれたかの分離（2026-11-10）

`analyze_mixed_rollouts` はMCTS rootの `root_candidate`、`root_visit_count`、`root_selected` を使い、DIRECT / POSITION / SAFETYについて候補数、候補が存在したroot数、訪問回数を集計する。**実際に選択された戦術**と区別して表示する。

- `MCTS root candidates`: rootに渡された31D候補の総行数。1rootに同種候補が複数存在できる。
- `MCTS roots offering each goal`: その種別が1つ以上現れたrootの数（生成回数ではない）。
- `MCTS visits by goal`: 候補へのroot訪問数の合計（勝率ではない）。
- `roots offering POSITION or SAFETY`: 非DIRECT候補があるroot数。
- `DIRECT chosen despite alternative`: 非DIRECT候補も存在したのにDIRECTを実行したroot数。

全rootが記録されている保証のない過去データでは `MCTS roots recorded` と実行ターン数の差にも注意。**候補0件なら生成器が候補を出さなかった/事前に除外された可能性**があるが、それだけで原因を確定できない。候補が存在してDIRECT選択なら、その盤面・そのモデル・そのMCTS設定でDIRECTになった事実を確認できる。学習による改善を示すものではない。
