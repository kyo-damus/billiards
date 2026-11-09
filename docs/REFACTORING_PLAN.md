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
