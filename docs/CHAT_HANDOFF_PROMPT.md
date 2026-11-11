# 新しいChatGPTチャットへの引き継ぎ用・最初のプロンプト

> 下の「コピーする文章」を、修士研究プロジェクト内で新しいチャットを開いた直後に送信してください。

## コピーする文章

このチャットは、これまで進めてきた**ビリヤードの階層型強化学習（修士研究）の実装・設計相談**の続きです。**最新は9-ballのPhase15実装監査（2026-11-11）、原Phase14は保護。古い2球の現状説明は履歴であり現在地ではありません。**チャットの長さ制限による移行なので、研究を最初から設計し直さず、既存の目的・決定・実装を正確に引き継いでください。

**最初にGitHubで以下の文書を読み、関連ソースやテストと照合してください。**

- リポジトリ: https://github.com/kyo-damus/billiards
- 最新安全作業ブランチ（2026-11-11時点）: `phase15-nineball-coupled-learning-audit-20261111`。原 `phase14-nineball-mcts-visits-policy` と `refactor-phase9-preserve-20261109` は保護。最新HEADはGitHubで確認する。
- 研究目的・全体設計・実装済み/未実装・研究方針・進捗・更新ルール: https://github.com/kyo-damus/billiards/blob/phase15-nineball-coupled-learning-audit-20261111/docs/RESEARCH_CONTEXT.md
- 入口のREADME: https://github.com/kyo-damus/billiards/blob/phase15-nineball-coupled-learning-audit-20261111/README.md
- 詳細な段階別実装ログ: https://github.com/kyo-damus/billiards/blob/refactor-phase9-preserve-20261109/docs/defense_v0_design.md

- リファクタリングの監査・保全・進行状況: https://github.com/kyo-damus/billiards/blob/refactor-phase9-preserve-20261109/docs/REFACTORING_PLAN.md

**必ず保持してほしい前提:**

1. 目的は戦術選択（Macro）とショットの連続制御（Micro）を統合した階層型RLシステム。Macroは可変個数の候補、Candidate Policy / Candidate Q / State V / 手番対応MCTS、MicroはGoal-conditioned SACとPositionのpower-only Residual、物理はFastFiz。学習済みMuZero/World Modelを導入済みと解釈しない。
2. **現在の主要対象は9-ball**（cue+1〜9球、部分的な9-ballルール）。旧2的球も回帰用に残る。v3 Macro状態72D・候補36D・Micro入力108D・動作5D、独立Checkpoint。SAFETY/BREAKはまだ明示的な仮ショットで、WPA完全ルール対応は未完成。
3. `Macro local reward`とゲームスコアは別。MCリターンは行動プレイヤー視点で、手番が変われば将来価値の符号を反転する。**完了ゲームだけ**がMCリターン教師の対象。MCTS訪問分布からのPolicy学習は未完了ゲームも利用可。2-step全候補Q教師、実行行動のみMC教師、訪問分布教師を混同しない。
4. 直近までにmixed multi-turn rollout、31次元Candidate Policy/Q、12次元State V、MCTSでの推論、複数ゲームの経験収集、訪問分布によるPolicy更新、完了ゲームによるQ/V更新（完了ゲームが2件以上のときのみ）が実装されている。smokeの小サンプルから性能向上を主張しない。
5. **優先は大規模自己対戦学習→独立評価→考察まで早期完走させること**。細かなアブレーションと既存研究比較は後回し。Phase15で交互更新の実装を追加したがDGX未検証。共有GPU8枚32GiB、原則可視1枚、必要なら空き確認後最大2枚。DGX実行とログ報告はユーザーが行う。
6. 既に実装されたものを再実装しない。ファイルとテストを読み、既存のAPIやチェックポイント互換性を保つ。ソース変更時はGitHubに反映し、関連するテストを指定してほしい。`.pt` / `.npz` は私のローカル環境にありGitHub上にあるとは限らない。

まず、確認できた最新コミットと資料をもとに**現在地・実装済み・未解決・次に着手すべき作業**を簡潔に整理してください。推測は推測と明示し、文書とコードに差異があれば最新コードを優先してください。その後、私の指示に沿って前回からの実装作業を続けてください。研究目的や設計上の前提を変更する場合は、変更理由と影響を説明してください。

今後、大きな機能追加、フェーズ完了、リファクタリング完了、研究方針の変更などの節目で `docs/RESEARCH_CONTEXT.md` を更新してください。毎回の小さなテスト・コミットごとの更新は不要です。自動で更新されたと仮定せず、改訂時はGitHubのコミットを確認してください。

## 補足

- チャット移行時にGitHubの参照ができなければ、`docs/RESEARCH_CONTEXT.md` のコピーを添付してください。
- 新しいブランチへ移った場合は、上のブランチ名とURLを置き換えてください。
- 最新ローカル実行ログが文書の更新日より新しい場合は、それも続けて貼るとスムーズです。

## 2026-11-09時点の引き継ぎ注意

- 現行作業は**仕様を維持した責務分離と不要コード/データ整理**。最新のリファクタリング計画・テスト結果・未完了のローカル成果物整理を `docs/REFACTORING_PLAN.md` で確認する。
- 旧 `build/` MuZero/SAC Python 8件は作業ブランチから削除済み。現在の研究はFastFiz物理を使うモデルベースMCTSであり、MuZeroを学習済みと解釈しない。
- ローカル現行主要5チェックポイントと関連Datasetは原則保護。旧build学習成果物107件/29.12MiBは、tarバックアップとSHA-256全件一致を確認して `~/billiards_refactor_quarantine_20261109/` に**退避完了**（削除0）。復元は `python tools/quarantine_legacy_build_artifacts.py --restore`。残る50件は現行・比較・再現性用として保全し、勝手に削除しない。
- 退避後の再監査で50件を確認し、学習済み5モデルを使う1ターンの統合smokeに成功（seed=94000、POSITION_ATTACK、score=[1,0]）。2026-11-09の段階で**今回の保守的リファクタリングは一区切り**。全スイート未実行・性能未評価・main未マージ。次はGitHubの最新HEADと `docs/REFACTORING_PLAN.md` を確認し、マージの要否と次の研究開発課題を判断する。

## 2026-11-09 Phase 3の構造整理（最新追記）

- `src/pipelines/` が共通処理の実装、`scripts/` がCLI・旧import互換用。旧CLIの呼び出しは変更していない。
- テスト76ファイルは `tests/unit/`・`tests/integration/`・`tests/pipelines/` に分類済み。個別の新ファイルパスは `tests/README.md`。**第8段階で移動した9ファイルの実行検証はまだ必要**。直下 `tests/test_*.py` を前提にコマンドを指定しない。
- Git管理下のテストファイル数とテスト関数件数は別の指標。個々のpytestにPASSEDがあっても、全テストスイートを実行済みとみなさない。
- Phase 4では `build/Makefile` 等の生成物と `outputs/micro/` の既存追跡3モデルを監査する。ローカルのモデル・バックアップ・退避先には勝手に手をつけない。

## 2026-11-09 Phase 3完了・Phase 4監査開始

- 第8段階の新配置9ファイルに対応する40テストも全件成功し、作業ツリークリーン。**テスト76ファイルの分類と各段階の対象テスト検証は完了**。全スイート一括PASSや学習性能向上の証拠ではない。
- Phase 4では `docs/PHASE4_ARTIFACT_AUDIT.md` に従い `build/Makefile`・`outputs/micro/` 3モデル・FastFiz拡張の状態を**読み取り専用で監査**する。Git追跡解除・削除・再ビルドは未実施。
- `build/Makefile` はCMake自動生成の絶対パス付きファイルだが、`src/env/fastfiz_simulator.py` は `build/` のネイティブ拡張を使う。ディレクトリごと消さないこと。`git rm --cached` のGitHubコミットをpullするとローカル現物が消え得ることにも注意。

## 2026-11-09 Phase 4完了（現行引き継ぎの優先情報）

- `build/Makefile` のGit追跡解除（`.gitignore`追加）がユーザーDGXに反映済み。Makefileは `~/billiards_refactor_safety_20261109/Makefile.original` から復元して現物を維持。追跡解除・無視設定を確認し、FastFiz最小回帰は **11 passed in 0.37s**。既存 `build/billiard_env_cpp*.so` は維持。
- FastFizの隔離CMake configure・ビルド・Python 3.11での拡張import/コンストラクタは成功済み。FastFiz旧C++にコンパイル警告はあるが、この検証ではエラーなし。
- 既存 `outputs/micro/` の3モデルは追跡維持かつtarバックアップとSHA-256一致。研究成果物を一括削除・モデル移動しない。
- **Phase 2/3/4の保全リファクタリングは一区切り。** ユーザーにリファクタリング作業の再実施を要求せず、研究開発・実験計画とmainへの統合の要否を検討する。全スイート一括PASS・性能改善・mainマージは未実施。

## 2026-11-11 Phase15最新引き継ぎ（旧記述より優先）

- 作業ブランチ `phase15-nineball-coupled-learning-audit-20261111`。旧Phase14は変更せず保持。
- 監査記録/実行手順: `docs/PHASE15_PROPOSAL_ALIGNMENT.md`。Phase11〜14は`docs/PHASE11...`等に段階別。研究上はMacro Policy+State V+MCTS、固定5D Microを使った9-ball自己対戦までDGX動作実績あり。選択行動Qは探索で未使用（無理にoff-policy候補Qを使わない）。
- **修正:** POSITION_ATTACKのMicro位置ボーナスは指定ポケット成功時のみ。Macro選択Goalを使った5D Micro checkpoint継続fine-tuneと交互学習ループを追加。Phase15のDGX実機テスト/反復実験は未確認。
- 新しいChatGPTチャットでは古い2球状態を「現状」と解釈せず、最新9-ballのソース、`docs/RESEARCH_CONTEXT.md`末尾、修正ブランチのHEADを確認してから実装すること。
