# 新しいChatGPTチャットへの引き継ぎ用・最初のプロンプト

> 下の「コピーする文章」を、修士研究プロジェクト内で新しいチャットを開いた直後に送信してください。

## コピーする文章

このチャットは、これまで進めてきた**ビリヤードの階層型強化学習（修士研究）の実装・設計相談**の続きです。チャットの長さ制限による移行なので、研究を最初から設計し直さず、既存の目的・決定・実装を正確に引き継いでください。

**最初にGitHubで以下の文書を読み、関連ソースやテストと照合してください。**

- リポジトリ: https://github.com/kyo-damus/billiards
- 作業ブランチ（2026-11-09時点）: `refactor-phase9-preserve-20261109`。元の `phase9-defense-v0` は保全済み。最新HEADはGitHubで確認する。
- 研究目的・全体設計・実装済み/未実装・研究方針・進捗・更新ルール: https://github.com/kyo-damus/billiards/blob/refactor-phase9-preserve-20261109/docs/RESEARCH_CONTEXT.md
- 入口のREADME: https://github.com/kyo-damus/billiards/blob/refactor-phase9-preserve-20261109/README.md
- 詳細な段階別実装ログ: https://github.com/kyo-damus/billiards/blob/refactor-phase9-preserve-20261109/docs/defense_v0_design.md

- リファクタリングの監査・保全・進行状況: https://github.com/kyo-damus/billiards/blob/refactor-phase9-preserve-20261109/docs/REFACTORING_PLAN.md

**必ず保持してほしい前提:**

1. 目的は戦術選択（Macro）とショットの連続制御（Micro）を統合した階層型RLシステム。Macroは可変個数の候補、Candidate Policy / Candidate Q / State V / 手番対応MCTS、MicroはGoal-conditioned SACとPositionのpower-only Residual、物理はFastFiz。学習済みMuZero/World Modelを導入済みと解釈しない。
2. 現在は**手球1・的球2、簡易ルール**による統合試験段階。Attack（DIRECT/POSITION）とSafety（物理グリッドで候補生成）はあるが、Safety用の汎用学習済みMicro、完全な反則・公式ルール、スピン/仰角の全面的な学習は未完成。
3. `Macro local reward`とゲームスコアは別。MCリターンは行動プレイヤー視点で、手番が変われば将来価値の符号を反転する。**完了ゲームだけ**がMCリターン教師の対象。MCTS訪問分布からのPolicy学習は未完了ゲームも利用可。2-step全候補Q教師、実行行動のみMC教師、訪問分布教師を混同しない。
4. 直近までにmixed multi-turn rollout、31次元Candidate Policy/Q、12次元State V、MCTSでの推論、複数ゲームの経験収集、訪問分布によるPolicy更新、完了ゲームによるQ/V更新（完了ゲームが2件以上のときのみ）が実装されている。smokeの小サンプルから性能向上を主張しない。
5. **いまは実装・責務分離・小規模テストを優先**。数十時間の大規模学習・性能比較・細かいパラメータ最適化は後の本実験に回す。ローカルCUDA/DGXのテストや学習は原則こちらが実行するので、必要なコマンドを示して結果を待って判断してほしい。
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
