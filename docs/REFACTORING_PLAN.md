# リファクタリング計画と監査記録（2026-11-09開始）

## 基準・保全

- 研究目的・現行仕様の正本: [RESEARCH_CONTEXT.md](RESEARCH_CONTEXT.md)。この計画は研究方針を変更しない。
- 元の退避ブランチ: `phase9-defense-v0`、基準HEAD: `39cee14c277e5cd4eadff5811e375589872160a1`。このブランチには手を加えない。
- 作業ブランチ: `refactor-phase9-preserve-20261109`。作業前のコミットを保存し、機能ごとに小さくコミットする。
- 直近ユーザー報告の基準テスト: mixed policy iteration / hierarchical rollout / return dataset / visit policy / state V の5ファイル、27件すべて通過（2026-11-02）。**この記録は最新HEADの全テストを実行したという意味ではない。**
- 機能、入出力API、31次元混合Candidate Policy/Q、12次元State V、25次元旧形式との区別、チェックポイントメタデータ、報酬・手番符号・教師種別、FastFiz呼び出しの挙動を保持する。学習性能を改善したとは主張しない。
- `.pt` / `.npz` などローカルにしかない生成物はGitブランチでは復元できない。棚卸し・保管前に一切削除しない。

## GitHub上の初期棚卸し（基準HEAD）

- Git追跡ファイル: 204。内訳: `src/` 59、`scripts/` 50、`tests/` 74、`build/` 10、`docs/` 3、`outputs/` 3、その他5。
- `build/` に旧MuZero関連と思われるPythonファイルが存在する。**未使用・削除可能と確定したわけではない。**
- `outputs/micro/micro_sac_final.pt`、`micro_sac_step_2500.pt`、`micro_sac_step_5000.pt` はGit追跡中（`.gitignore` に`*.pt`があっても既存の追跡は継続する）。利用状況を確認するまで残す。
- `.gitignore` では`*.npz`の無視を明示していない。ローカルの実物を確認してから生成物の方針を決める。
- 過去ブランチの削除は**未実施**。後で祖先関係・残すべき独自コミット・使用中チェックポイントへの依存を確認する。

## 実行順序

1. **ローカル監査（読み取り専用）**: 作業ツリーの状態・未pushファイル・チェックポイント/学習データのパス、サイズ、更新日時、外部参照を収集。削除候補を「現在必要」「研究再現に必要」「要確認」「安全に削除可能」に分類。
2. **テストの基準固定**: 既存の27件と、Defense/Safety/FastFiz/チェックポイント互換性などを対象とするテストを実行。現在の失敗を記録し、リファクタリング起因か区別する。
3. **低リスクな責務分離**: `scripts/` 内にまたがるMacro混合コンポーネント構築、CLI、dataset、trainingの依存を整理。公開APIを残す薄い互換ラッパーを使い、挙動変更なしを検証。
4. **重複コード/テストの統合**: 振る舞いが重複するものを特定し、重要な境界条件（手番反転、完了/未完了、Q教師種別、Safety合法性、モデル次元）を失わない範囲で統合。**テスト件数の削減それ自体を成功指標にしない。**
5. **古いファイル・データの削除候補確定**: import、スクリプト呼び出し、Git履歴、ドキュメント、実験再現性を調べ、判断に迷う対象は所有者に確認してから削除。ローカル生成物は保管先と必要性を明記して扱う。
6. **完了確認**: 関連テストと最小smoke、変更差分、チェックポイント互換性を再検証。節目で`RESEARCH_CONTEXT.md`を更新し、コミットを確認する。

## 変更・削除のガードレール

- 基準ブランチの改変、既存ファイルの一括削除、Git履歴書き換え、`git clean -fdx`、ローカルの不可逆削除は行わない。
- 仕様変更と整理を同一コミットに混ぜない。仕様上の問題を見つけた場合は別件として提示する。
- ショットの物理パラメータやモデルの入力次元が変わる処理は、リファクタリングとは別に承認を得る。
- 安全に変更を進められる単位ごとに必要なテストコマンドを示し、ユーザーのDGX上での実行結果を確認して進む。
- ローカルとGitHubのブランチ名/HEADが食い違う場合は、現物の保全と差分確認を優先する。

## 監査結果と初回削除（2026-11-09）

- ローカル監査結果: Git追跡Python 194、テストファイル74・テスト関数300、AST構文エラー0。 `src/micro/policy.py` は未参照のProtocolだが、設計上のインターフェースとして残す（削除保留）。
- GitHub上の `build/` には過去のMuZero/SAC学習用Pythonスクリプト8件がある。これらは `src/`、`scripts/`、`tests/` の現在の実装から静的importされておらず、`CMakeLists.txt` のビルド対象でもない。Pythonスクリプト群は相互に依存する過去の別実装。
- **削除対象（Git追跡ソースのみ）:** `build/custom_env.py`, `build/eval_policy.py`, `build/fine_tune_policy.py`, `build/muzero_buffer.py`, `build/muzero_models.py`, `build/test_mcts.py`, `build/train_muzero.py`, `build/train_policy.py`。現在のMacro/Micro/FastFizの構築・学習・推論系には含まれないため、古い実装の混在防止を目的に作業ブランチで削除。過去コードは分岐元の `phase9-defense-v0` とGit履歴に残る。
- **削除しない:** `build/Makefile`（既存のローカルビルドに必要な可能性）、`build/.gitignore`、`CMakeLists.txt`、`cpp_simulator.cpp`、`libs/fastfiz`、全学習済みモデル（`.pt`/`.pth`）・データ（`.npz`）・ログ。ファイル削除後のテスト完走をユーザー環境で確認するまで、ほかの削除は進めない。
- **注意:** 過去の `python build/train_muzero.py` 等の起動は対象外。これらを使った歴史的な実験の再現性が必要な場合は元ブランチから復元する。現在の研究成果の再現に必要なローカル重みは保持。
- **テスト削減:** 1～2件しかテストを含まないファイルは23件あるが、それだけで不要とは判定しない。テストの意味・重複・境界条件を比較してから統合。

## テスト統合（2026-11-09・第1弾）

- `tests/test_candidate_mcts.py` の5件のテストは、同じ乱数seed・同じMCTS設定・同じ1回の探索を繰り返し、異なる性質（候補生成、Goal種別、Prior合計、MacroCandidateのTransition伝達、last_root）を個別に検査していた。
- これらを `test_candidate_mcts_root_contract` の1件に統合した。**5項目のassertionは全て保持**。MCTSの値推定・探索優先順位・手番反転など独立したテストはそのまま残す。
- 統合で `tests/test_candidate_mcts.py` は15件→11件、全テスト関数数は300→296件の見込み。**テスト結果はユーザーDGX上で要確認**。
- 1件のテストが複数条件を検証するようになるため、最初の失敗以降のassertionは実行されない点は留意。今回の統合対象は同一のroot契約に限定した。

## テストファイル統合（2026-11-09・第2弾）

- `tests/test_mcts_root_generator.py` は1テスト・137行で、現在の `tests/test_candidate_mcts.py` と同じMCTS、Candidate Policy/Value、GameState、ダミー遷移を独立に定義していた。
- 元のテストの重要契約（ルート候補専用Generatorが1回、深部候補Generatorが1回、ルートに深部用候補が混入しない）を `tests/test_candidate_mcts.py::test_mcts_uses_separate_root_generator` に統合。ルート/深部で異なるポケットの候補を明示し、同様の挙動を確認。
- テストファイル数74→73、テスト関数総数296→296（検証ケースの件数は維持）。ダミー実装を共通化したため重複コードを減らした。
- 旧テストファイルは作業ブランチ上では削除したが、`phase9-defense-v0` とGit履歴には保存される。変更後のテストはユーザーDGX上で未実施。

## 階層Agent単体テストの統合（2026-11-09・第3弾）

- `tests/test_hierarchical_agent.py` の3件は同一のDummyMCTS/DummyTransitionを用いて、`select_macro_action`、`execute_macro_action`、`step` をそれぞれ検証していた。
- `tests/test_mixed_hierarchical_rollout.py` に `test_hierarchical_agent_legacy_select_execute_and_step_contract` を追加し、**3種類の公開メソッド全てを維持して検証**。特に現行Mixed専用のMacroCandidateだけでなく、従来の`MacroAction`を通す経路を残す。
- 元の3件を1件にまとめることでテストファイル73→72、テスト関数296→294件（見込み）。遷移回数、入力の同一性も新たに確認する。旧テストファイルは作業ブランチ上では削除し、分岐元・履歴から復元可能。
- この変更はテストコードのみ。実装コード・チェックポイント・データは変更しない。テストはユーザーDGX上で未実施。

## データ整理・安全監査（2026-11-09・第1段階）

- ユーザー実施の最新確認: 2026-11-09、`tests/test_mixed_hierarchical_rollout.py` / `test_hierarchical_smoke.py` / `test_candidate_mcts.py` / `test_defense_macro_transition.py` / `test_collect_mixed_rollouts.py` の関連33件全件PASSED（6.03秒）。`b21b18a` 時点の範囲指定テストであり全スイートの合格は意味しない。
- ユーザーは `~/billiards_research_backup_20261109.tar.gz` (105MB) を作成し、`tar -tzf` の一覧読み出しで `Backup OK` を確認。GitHubに存在しないモデル/データも当時の研究ディレクトリ内に含む可能性が高いが、**ファイル単位のバイト一致は未検証**。
- 新設 `tools/audit_local_artifacts.py` はファイル名・SHA256・サイズ・Git追跡・過去のtarアーカイブ内の名前・Git文書/ソースの名前出現を一覧化する。**データを読み込む学習処理、削除・移動処理は一切実行しない。** バックアップとの照合はアーカイブ内の名前のみ。
- 保護対象の候補は `goal_conditioned_micro_mixed_curriculum_100k.pt`、`position_power_residual_mlp_2000.pt`、`macro_candidate_policy_mixed_v2.pt`、`macro_candidate_value_mixed_v2.pt`、`macro_state_value_mixed_v2.pt` と、実験反復・選択済み行動MCリターンの関連成果物。
- `build/results/`、`build/results_muzero/` の過去学習結果や同一SHA256の重複は、**削除確定ではなく移動・削減候補**。物理的なデータ削除は利用者による保存方針の確認とバックアップの照合の後だけ行う。
- `.gitignore` とデータ保存先の設計は、一覧を確認してから決定する（現時点では変更しない）。

## ローカル成果物157件の個別監査（2026-11-09）

- ユーザー添付 `local_artifacts_review.tsv`（Git非追跡、ユーザー環境）の157件を調査。合計79.62 MiB、SHA-256で完全一致する別ファイルは0組。157件とも2026-11-09のtarアーカイブに同じ名前が存在（**内容一致は未確認**）。
- 現行主要モデル5件/3.54MiB、Git追跡モデル3件/4.03MiB、他のモデル24件/42.50MiB、データセット12件/0.24MiB、反復実験8件/0.21MiB、旧build/results関連105件/29.10MiB。
- **初期退避候補107件/29.12MiB**: `build/results/sac_actor_ep1000.pth`〜`sac_actor_ep100000.pth`（100件、1000ごと）、旧Actor/Critic finalの2件、`build/results_muzero/*_net_latest.pth`の3件に加え、`build/mcts_experience.pt`、`build/policy_model.pth`の2件。すべて削除済みの旧build Pythonスクリプトの成果物で、現行研究への名前参照が見つからない。
- **退避保留**: 主要5モデル、反復Policy実験8件、Macro 25/31D教師データ、Residual/Goal-conditioned Micro学習済みモデル（比較実験・再現に使う可能性）。`build/` 以外の24モデルから旧2件を除いた22モデルは個別確認待ち。
- `tools/quarantine_legacy_build_artifacts.py` を追加。**デフォルトは読み取り専用**で、監査TSVのサイズ・SHA256、現物のサイズ・SHA256、tar内の同じファイルの実バイトSHA256、107件の厳密なホワイトリストを照合。すべて一致する場合に限り `--apply` によってリポジトリ外の `~/billiards_refactor_quarantine_20261109/` に移動可能。`--restore` で上書きせず復元可能。削除は行わない。
- **検証と退避は完了（2026-11-09、ユーザー実行ログ）:** 初回の読み取り専用照合で107件全てが監査TSV・ローカル現物・tarバックアップのSHA-256で一致。続く `python tools/quarantine_legacy_build_artifacts.py --apply` で107件/29.12MiBを `~/billiards_refactor_quarantine_20261109/` へ移動した（削除0件）。`_legacy_build_quarantine_manifest.tsv` の存在も確認。復元は `python tools/quarantine_legacy_build_artifacts.py --restore`。退避先フォルダや元tarを消さないこと。
- 残る成果物50件（約50.50MiB）：現行主要5件/3.54MiB、Git追跡3件/4.03MiB、比較候補の過去チェックポイント22件/約42.48MiB、Macro・Micro学習データセット12件/0.24MiB、Policy反復実験8件/0.21MiB。これら50件は**意図的に維持**（性能比較・再現性・モデル互換性の根拠）。
- 生成済み `.npz` と既知のローカル監査/評価テキストを `.gitignore` に追加し、Gitの作業状態を読みやすくする。Git無視はファイル削除ではない。必要なデータを後にGit管理したい場合は `git add -f` で明示的に追加する。
- **最終確認を完了（2026-11-09、ユーザー実行ログ）:** `tools/audit_local_artifacts.py --output local_artifacts_post_quarantine.tsv` で成果物 **50件**（主要5、Git追跡3、その他モデル22、データ12、反復8）、完全一致SHA256重複0組。`git status --short --branch` は追跡ファイルの未コミット変更も未追跡ファイルの表示もないクリーンな状態。
- **学習済み5モデルを使う実行smoke成功（CPU）:** `python -m scripts.smoke_mixed_hierarchical_rollout --base-checkpoint goal_conditioned_micro_mixed_curriculum_100k.pt --residual-checkpoint position_power_residual_mlp_2000.pt --candidate-value-checkpoint macro_candidate_value_mixed_v2.pt --candidate-policy-checkpoint macro_candidate_policy_mixed_v2.pt --state-value-checkpoint macro_state_value_mixed_v2.pt --max-turns 1 --simulations 1 --safety-top-k 1 --safety-powers 0.65 --safety-angles 0 --safety-tip-y-mm 0 --outcome-grid-size 3 --outcome-top-k 2 --device cpu`。seed=94000、手番0→0、選択 `attack/position_attack`（ball0、pocket5）、Macro local reward +1.250、score_delta=(1,0)、ポケット状態[-1,5,-1]、`stop reason=max_turns`、`game terminated=False`。これは**1盤面・1ターンの経路実行確認**。Defense候補の物理実行や性能向上の実証にはならない（Defenseは既存の単体・統合テストで確認）。
- **本フェーズの完了判定:** 旧コード・テスト重複の整理、旧モデルのSHA256検証付き退避、Git無視設定、現行5モデルの読み込みを含む最小実行確認まで完了。元の `phase9-defense-v0` は改変せず、作業ブランチを維持。**全テストスイート未実行、mainへのマージ未実施、研究性能は未評価**。必要に応じてブランチ差分のレビュー後にマージを判断する。

## フォルダ整理プロジェクト Phase 1（2026-11-09）

- 追加: `docs/PROJECT_STRUCTURE.md`、`configs/macro/`・`configs/smoke/` の実験記録JSONテンプレート、`tools/prepare_run.py`、その単体テスト。
- `artifacts/` をGit管理外の新規実験出力先とし、`artifacts/runs/<run_id>/` に設定・Git metadata・outputs/logsをまとめる。
- **未変更:** 現行 `src/`・`scripts/`・`tests/` の既存ファイル配置、現行・過去の全チェックポイントとデータ、旧buildの退避先、学習・推論ロジック。現行CLIはJSONを読み込まない。
- `python -m pytest tests/test_prepare_run.py -v` によるユーザー環境での検証待ち。合格後にPhase 2のコード移動へ進む。全スイートや実際のRL性能を再評価したことにはならない。

## Phase 2: scriptsとsrcの責務分離（2026-11-09・第1段階）

- `scripts/training_utils.py` の実装3関数（`resolve_device`、`make_group_split`、`make_groups`）を変更せず `src/pipelines/training_utils.py` に移す。
- 旧 `scripts.training_utils` は3関数をそのまま再公開する薄い互換ラッパーとして残す。既存のCLI、テスト、従来のインポートパスは維持する。
- 新設 `tests/test_training_utils_compat.py` は同一の関数オブジェクトの再公開、状態グループをまたがない分割、`make_groups`、明示的CPU指定を確認する。
- **他のスクリプト・モデル・dataset・設定の意味は変更しない。** `attack_components.py` をただちに移さない理由: `scripts.eval_position_residual` 内のモデル読み込みに依存しているため、依存ループを作らない抽出方法が必要。
- ユーザー環境で対象pytestが通った後、ファクトリの切り出しを進める。実際の動作確認は未実施。

## Phase 2: 評価CLIへの依存解消（2026-11-09・第2段階）

- ユーザー報告: 共通学習ユーティリティ移動後の関連23テストが **すべてPASSED（6.58s）**。
- `scripts/eval_position_residual.py` で定義していた `load_base_agent` と `load_residual_network` を **本文の処理を変えず** `src/pipelines/checkpoint_loading.py` へ移動し、元の評価CLIは旧名を再公開する。
- `scripts/attack_components.py` は旧評価CLIではなく新しいローダーモジュールをimportする。これにより次のファクトリ移動時に `src/` が `scripts/` に依存しないよう準備。既存の `scripts.eval_position_residual_sweep` などの旧importも維持。
- モデルの読み込み方法・チェックポイントキー・target critic optional処理・Residual checkpoint辞書の返し方を変えない。旧importのidentityと必須キー/optional critic targetの検証を追加。
- **実装コードの入出力が変わっていないことは、ユーザーDGXでのテストと5モデル統合smokeで要確認**。元の `phase9-defense-v0`、checkpoint、データは変更しない。

## Phase 2: Attack/Mixed構築ファクトリの移動（2026-11-09・第3段階）

- ユーザー報告: チェックポイントローダー移動後の関連16件のpytest **すべてPASSED（2.85秒）**。
- `scripts/attack_components.py` の構築処理本体を `src/pipelines/attack_components.py`、`scripts/mixed_components.py` の構築処理本体を `src/pipelines/mixed_components.py` へ移行。Attackのcheckpoint取得・Residualスケール・候補生成器・物理Transition、MixedのDefense/Safetyパラメータと5要素の戻り値は同じ。
- `src/pipelines/mixed_components.py` のAttack依存は `src.pipelines.attack_components` に変更し、`src/` から `scripts/` への逆向き依存を避ける。
- `scripts/attack_components.py` と `scripts/mixed_components.py` は、既存import元が使えるよう元の関数と公開クラス名を再公開する互換ラッパーに置換。既存の呼び出し側のimport行は変更せず、従来のCLIも維持。
- `tests/test_component_factories_compat.py` で旧/新import関数のidentity、Attackの3要素＋checkpoint metadataの契約、Mixedの5要素とSafety powers/angles/tip offsets等をモックで検証。既存モデル・学習済み重み・Dataset・前回退避データには触れない。
- **テスト未実行。** ユーザーDGX上で関連pytestと必要ならCPU 1ターンの統合smokeの再実行結果を確認してから、次のファイル整理に進む。

## Phase 2完了確認とPhase 3テスト配置整理（2026-11-09・第1段階）

- **ユーザー確認:** ファクトリ移行後の関連22テストが全件PASSED（3.41秒）。続くCPU 1ターン混合階層smoke（seed=94000、simulations=1、学習済み5モデル）は移行前と同一出力: `attack/position_attack`、ball=0、pocket=5、Macro local reward +1.250、score [1,0]、停止理由 `max_turns`、ゲーム非終了。これは限定した条件の互換性確認でありモデル強度の証明ではない。
- **Phase 2の主要実装を完了:** `src/pipelines/` に共通学習補助、モデルローダー、Attack/Mixed構築を移し、従来の `scripts/` importとCLIを維持。
- **Phase 3の初回はテスト内容を変えないパス移動のみ**。以下5ファイルを単体テスト階層へ整理。旧パスはGit上からなくなるため、`python -m pytest` のファイル指定は新パスを使用すること。
  - `tests/test_training_utils_compat.py` → `tests/unit/pipelines/test_training_utils_compat.py`
  - `tests/test_checkpoint_loading_compat.py` → `tests/unit/pipelines/test_checkpoint_loading_compat.py`
  - `tests/test_component_factories_compat.py` → `tests/unit/pipelines/test_component_factories_compat.py`
  - `tests/test_prepare_run.py` → `tests/unit/tools/test_prepare_run.py`
  - `tests/test_game_rules_interface.py` → `tests/unit/game/test_game_rules_interface.py`
- 相対importを持たず、root起動の `python -m pytest` で参照可能なファイルを選択。ファイル内容・テスト関数・pytestマーカー・物理/学習実装は無変更。旧テストはGit履歴から復元可能。
- `tests/README.md` に分類基準とテスト指定コマンドを明記。残るテストは未移動で、分類を継続する。現時点では新配置でのpytestは未実行。

## Phase 3: Macro/Micro/Goal単体テストの分類（2026-11-09・第2段階）

- ユーザー報告: 第1段階の `tests/unit/pipelines/`・`unit/tools/`・`unit/game/` と `test_iterate_mixed_macro_policy.py` の**27件全件PASSED（3.64秒）**、`git status` クリーン。
- Macro5件（candidate / networks / candidate_policy / candidate_value / local_evaluation）、Micro3件（sac / position_residual_policy / position_residual_scale_gate）、Goal3件（goal_reward / goal_conditioning / tactical_goal）をそれぞれ `tests/unit/macro/`、`unit/micro/`、`unit/goal/` に移動。計11ファイル。
- Git blob SHAを再利用した**テスト内容変更ゼロのパス移動**。ファイル内に他テストへの相対importや `__file__` / `sys.path` / chdirのパス依存がないことを確認。研究コード・テスト関数・成果物・チェックポイントは不変更。
- 次のユーザー検証では `python -m pytest tests/unit/macro/ tests/unit/micro/ tests/unit/goal/ -v` を使用。新ディレクトリでのpytestは未実行。FastFizの物理統合や大量の学習処理はこの分類変更に含めない。

## Phase 3: FastFiz・MCTS・階層Agent統合テストの分類（2026-11-09・第3段階）

- **ユーザー報告:** 第2段階で移した`tests/unit/macro/`・`unit/micro/`・`unit/goal/`の**49件全件PASSED（5.72秒）**。 `git status` はクリーン。
- 以下4ファイルをテスト内容を変えず配置変更（blob SHAを再利用、元ファイルの削除はGitのrenameとして追跡可能）。
  - `tests/test_fastfiz_simulator.py` → `tests/integration/physics/test_fastfiz_simulator.py`
  - `tests/test_mcts_physics_integration.py` → `tests/integration/physics/test_mcts_physics_integration.py`
  - `tests/test_candidate_mcts_physics_integration.py` → `tests/integration/physics/test_candidate_mcts_physics_integration.py`
  - `tests/test_hierarchical_smoke.py` → `tests/integration/agent/test_hierarchical_smoke.py`
- これらはFastFiz実物理／MCTS-物理接続／階層Agentの結合smokeに関するテスト。内部の相対import、`__file__`、`sys.path`、作業ディレクトリ変更への依存がないことを確認。**移動先でのpytestは未実行。** 実行環境は従来通りFastFizネイティブライブラリが必要。
- Phase 3累計は20ファイルを分類（単体16、統合4）。計76ファイル自体は減らしていない。研究ソース、モデル、データセット、チェックポイント、MCTSの仕様には手を触れていない。残り56ファイルは引き続きrootにあり、依存を確認して段階的に分類する。

## Phase 3: MCTS契約と学習・Rolloutパイプラインの分類（2026-11-09・第4段階）

- **ユーザー報告:** 第3段階の `tests/integration/physics/` と `tests/integration/agent/` の全14テストがPASSED（6.12秒）。 `git status` クリーン。
- Mock/DummyモデルでMCTSの選択・候補・31D mixed契約を検証する3ファイル（`test_mcts.py`, `test_candidate_mcts.py`, `test_mixed_candidate_mcts.py`）を `tests/unit/macro/` へ移動。
- 小規模のCandidate Policy/Q/State V学習とmixed policy iterationの4ファイルを `tests/pipelines/macro/` へ、実行済み軌跡からMCTS訪問分布・MC return教師を作成する2ファイルを `tests/pipelines/rollout/` へ移動。訓練方法・教師種別の違いは保持。
- **9ファイルともGit blob SHAを再利用した内容変更ゼロのパス移動**。各ファイルの相対import、`__file__`、`sys.path`、chdirなどのパス依存がないことを確認。元の古いパスは指定できなくなるので新パスを指定。
- Phase 3累計は29/76ファイルを分類（unit19、integration4、pipelines6）。残る47ファイルはrootの`tests/test_*.py`に残す。研究コード・学習済みモデル・checkpoint・データセットは未変更。**今回の移動後のpytestはまだ未実施**。

## Phase 3: Defense/Safety・Micro学習の分類（2026-11-09・第5段階）

- **ユーザー実行結果:** 第4段階で移動したMCTS/Macro/rolloutの関連51件が全件PASSED（6.25秒）、Git作業ツリーもクリーン。
- 物理実行の契約を含む`test_defense_macro_transition.py`と`test_defense_shot.py`を`tests/integration/defense/`へ移動。前者は実物理FastFiz統合ケースも含むので、丸ごとunit扱いしない。
- 相手DIRECT脅威・仰角診断・Safety候補生成の4ファイルを`tests/unit/defense/`へ、`test_safety_goal.py`を`tests/unit/goal/`へ移動。
- `test_train_position_residual.py`、`test_train_position_residual_scale_gate.py`、`test_position_residual_dataset.py`を`tests/pipelines/micro/`へ移動。Residualモデルの訓練・教師データ検証は実験本体ではなく小規模契約テスト。
- 今回の10ファイルは全てGitの既存blob SHAを再利用し、**テスト内容に変更なし**。ファイル内の相対import、`__file__`、`sys.path`、chdir等のパス依存がないことを確認済み。旧パスで指定せず新パスでpytestを実行すること。
- Phase 3累計**39/76ファイル**（unit24、integration6、pipelines9）。残る37ファイルは未分類。研究ロジック・訓練済みモデル・成果物・データの移動や削除はしていない。**移動後のpytestは未実行。**

## Phase 3: Position候補・Micro位置系・評価ヘルパーの分類（2026-11-09・第6段階）

- **ユーザー実行結果:** 第5段階のDefense/Safety・Micro学習48件が全件PASSED（4.14秒）、Git作業ツリークリーン。
- 位置関連の候補生成・到達可能性/feasibilityの5ファイルを `tests/unit/macro/position/`、Macro action generatorを `tests/unit/macro/`、Shot Geometryを `tests/unit/env/`、SimplifiedGameRulesを `tests/unit/game/` に移動。
- Adaptive Position Policy・Position Blend Gateを `tests/unit/micro/`、Position Success/Probability GateとOutcome Witness BCの小規模学習契約を `tests/pipelines/micro/`、Position Actor Blend・Position Goal Evalの評価CLIヘルパーを `tests/pipelines/evaluation/` に移動。
- 計15ファイル。各ファイルは `__file__` や相対import等に依存しないことを検査し、Gitの既存blob SHAを再使用した**内容変更ゼロの移動**。これらのテストは既存の関数とassertionをそのまま保持する。
- 累計**54/76ファイル**を分類（unit34、integration6、pipelines14）、未分類22ファイル。学習済みモデル、Macro/Micro/FastFiz実装、既存データ・成果物は移動も削除もしない。移動後の指定pytestは**ユーザーDGXで要確認**。
