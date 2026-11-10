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

### rootごとの最終決定・訪問数同点の確認（2026-11-10）

`src/macro/mcts.py` の `select_action` は**戦術ごとの訪問数合計ではなく、各候補の訪問数が最大の行動**を選ぶ。`max` の同点はrootの候補登録順で最初のものが選ばれる。未訪問候補の探索を優先する設定と少数simulationsの組み合わせでは、この同点が起こり得る。

```bash
python -m scripts.analyze_mixed_rollouts \
  --rollout-dir macro_policy_iteration_smoke --recursive --per-root
```

`--per-root` は各rootの選択戦術、選択候補のvisit数、個別最大visit数、最大値同点の候補数、選択候補が最初の最大候補か、戦術ごとの候補数・visit数を表示する。`selected-prior` と `selected-predicted-Q` はログに存在すれば表示（後者は**ニューラルネットワークの保存された予測値**であり、MCTSのback-up Qや実際の報酬ではない）。過去の一部データには予測値が存在しないこともある。

この診断は**read-only**で、行動の選択規則やPolicy/Q/Vの学習を変更しない。訪問数同点の実例があるかどうかは既存Rolloutの実出力を確認するまで断定しない。

## 2026-11-10: ルートの訪問数同点を比較するオプション

既存Mixed Rolloutでは5/5 rootが、探索10回で最大visit数1の10候補同点となり、どのrootでも候補リストの先頭のDIRECTが選ばれた。`src/macro/mcts.py` の選択規則は、通常は**個別候補の最大visit数、同点時は最初の候補**。候補群の合計訪問数とは無関係。

- `--root-tie-breaker first`（デフォルト）: 旧選択規則を厳密に維持。
- `--root-tie-breaker empirical_q`（実験用）: **個別最大visit数で同点、かつ訪問済みの候補同士だけ**、手番符号を考慮した `reward + gamma * sign * child.value` が高いものを選択。さらに同点ならPolicy prior、その後候補登録順。訪問数が異なれば従来どおりvisit数優先。
- この `empirical_q` は観測したFastFiz遷移と子状態価値による探索時の見積もりであり、保存されている学習済みCandidate-Q `predicted_q` とは別の量。手番交代時は子の価値を負符号にする。
- `scripts/smoke_mixed_hierarchical_rollout.py`、`scripts/collect_mixed_rollouts.py`、`scripts/iterate_mixed_macro_policy.py` にオプションを追加。**指定しなければ旧動作**であり、学習済みモデル・報酬・教師データ定義は変更しない。
- 比較時は同じcheckpoint・初期seed・探索回数・候補生成条件を指定し、**出力先は必ず別にする**。rootに同点がない場合、両方式の結果は一致する。特定の設定で選択が改善しても性能改善とは言わない。
- 対象テスト: `tests/unit/macro/test_mixed_candidate_mcts.py`（デフォルト互換・手番符号・visit優先・Q/ prior同点）＋既存MCTS契約テスト。長時間学習は不要。

## 2026-11-10: seedを揃えたMCTS方式の比較

既存の `scripts.collect_mixed_rollouts` で **同じseed集合・同じcheckpoint・同じsimulations・同じmax-turns** を使い、`--root-tie-breaker first` と `--root-tie-breaker empirical_q` のRolloutを**別々のディレクトリ**に保存する。

```bash
python -m scripts.compare_mixed_rollout_modes \
  --first-dir artifacts/runs/<run_id>/outputs/first \
  --empirical-dir artifacts/runs/<run_id>/outputs/empirical_q \
  --json-output artifacts/runs/<run_id>/outputs/paired_summary.json
```

比較CLIはread-onlyで、seed集合・NPZの内部seed・初期12D状態の一致を検証する。不一致ならエラーにし、欠損seedを勝手に除外しない。主な指標は、**同じ初期盤面における最初の実行Macro候補と初手の局所報酬差**、ゲーム完了件数、SimplifiedGameRulesのプレイヤー別累積スコア差分、スクラッチ回数。ペアごとの記録をJSONにも保存できる（既存ファイルは上書き拒否）。

**注意:** ゲーム完了率は勝率ではない（スクラッチでも簡易ルール上の終了となる）。異なる政策の軌跡長が違えば累積のMacro局所報酬を単純に優劣へ結びつけられない。探索回数10と50などの予算を変える比較では、予算ごとに分けて上記のseed対応比較を実行する。実行後の性能結論はユーザーDGXログを確認してから記録する。

対象テスト: `tests/pipelines/evaluation/test_compare_mixed_rollout_modes.py`。

補足: `scripts.compare_mixed_rollout_modes` は初手の物理的的球入球数に加え、Attack系候補のエンコード済みの目標球/目標ポケットと実際の次状態のポケットIDを照合し、**狙ったポケットへの入球**を区別する。SAFETYや目標ポケットのない候補は `N/A` とし、失敗には数えない。Macro局所報酬にはwrong-pocket減点、POSITION加点、次ショット可能性の加点が含まれるため、スコアだけでなくこの物理的結果も確認する。
