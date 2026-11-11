# Billiards Hierarchical RL / ビリヤード階層型強化学習

本リポジトリは、ビリヤードにおける**戦術選択（Macro）**と**物理ショット制御（Micro）**を分離し、強化学習とモデルベース探索を統合する研究用プロトタイプです。

> **研究を引き継ぐ人・AIへのお願い：** 詳細は [docs/RESEARCH_CONTEXT.md](docs/RESEARCH_CONTEXT.md) を最初に読んでください。ここには、研究目的、設計上の意図、実装済み機能と未実装部分の区別、直近の進捗、作業時の注意事項、更新ルールを記録します。実装の正確な仕様は常に最新コード・テストを確認してください。

## 目標となる全体像

1. **Macro**: 盤面・手番・スコアに応じて、攻撃／守備、対象球・ポケット、次ショットに有利な手球位置などの戦術候補を生成する。
2. **Candidate Policy / Q / State V**: 可変個の戦術候補に事前確率と価値を与え、将来的に多数ターンの判断を学習する。
3. **MCTS**: Macro候補をMicroと物理シミュレータ（FastFiz）によって展開し、行動を選択する。
4. **Micro**: Goal-conditioned SACを基本に、連続パラメータ（方向・力など）から物理ショットを実行する。
5. **GameRules**: 実行後の状態、得点、手番、終了条件を更新する。

**2026-11-11時点**では従来の2的球簡易環境を残しつつ、**9-ball（cue+1〜9球）の物理シミュレーションと部分的ルール**に対応しました。9-ball用のMacro Policy/Q/V・MCTSと5D Micro SACによる自己対戦/学習の経路を実装しています。完全なWPAルール・十分な強さ・汎化性能は未達成です。

## 以前の簡易2的球経路（2026-11-02時点の履歴）

- DIRECT / POSITION 攻撃と、物理ショット候補に基づく SAFETY を扱う31次元のMacroCandidate経路を構築。
- Macro Candidate Policy、候補依存Q、状態価値V、MCTS、Micro、FastFiz、簡易GameRulesを接続。
- 複数ターンの軌跡保存、MCTS訪問分布によるPolicy学習、完了ゲームからの行動リターン学習、Q/Vの任意更新を実装。
- **小規模smokeが完走したことと、学習性能が良いことは区別する。** Macro Q/Vの経験学習、Defense選択、ゲーム勝率等について結論はまだ出ていない。
- 2026-11-02の最新smokeは2ゲーム中1ゲーム終了。Policyを更新し、完了ゲーム不足によりQ/Vを更新しないガードが機能。

## 現在の9-ball実装（2026-11-11時点、Phase15はDGX未検証）

- v3 Macro状態72D・候補36D・Micro入力108D・5D物理操作。旧12D/31D/2D学習済みモデルと非互換。
- Macro Policyは物理教師で初期学習し、MCTS root訪問分布で更新。Q/Vは完了race-to-1自己対戦の選択行動終端リターンから推定。選択行動Qを未実行候補の真値としない。
- 9-ball手番考慮MCTSはMacro Policy事前確率・固定Micro・FastFiz・State V葉評価。防御/ブレイクは依然として明示的な仮ショット。
- **Phase15で追加:** MicroのPOSITION成功条件に整合する報酬修正、Macro選択GoalからのMicro継続学習、自己対戦→Policy/Q/V→Microの反復ランナー。コードコミット済み、関連DGX実行は未確認。
- [Phase15監査・実行手順](docs/PHASE15_PROPOSAL_ALIGNMENT.md) および [研究の正本](docs/RESEARCH_CONTEXT.md) を参照。原Phase14は保存済み。共有GPUは原則可視1枚。

## コード・関連ドキュメント

- `src/macro/`: 戦術候補、Candidate Policy/Q、State V、MCTS、攻守遷移、局所評価
- `src/micro/`: Goal-conditioned SAC、Position Residual、守備ショット実行の補助
- `src/controller/`: 階層型エージェント、複数ターン実行、経験ログ
- `src/env/`: FastFiz連携、Micro環境
- `src/game/`: 簡易GameRules
- `scripts/`: データ生成、学習、診断、統合smoke（既存CLI互換性を維持）
- `src/pipelines/`: 共通学習補助、モデルローダー、Attack/Mixedファクトリの実装
- `configs/`: 新しい実験の記録用設定テンプレート（現行CLIはJSONを自動読み込みしない）
- `artifacts/`: 新しい実験の成果物保存先（Git管理外・準備ツール実行時に生成）
- `tools/prepare_run.py`: 実験ディレクトリと設定・Gitメタデータの保存（学習は実行しない）
- `tests/`: 単体・統合・学習パイプラインのテスト（`unit/`、`integration/`、`pipelines/`）
- [tests/README.md](tests/README.md): **機能別テスト配置・実行コマンド**
- [docs/defense_v0_design.md](docs/defense_v0_design.md): Defense v0と実装段階ごとの技術ログ
- [docs/RESEARCH_CONTEXT.md](docs/RESEARCH_CONTEXT.md): **研究方針と引き継ぎ資料（主に更新する文書）**
- [docs/PROJECT_STRUCTURE.md](docs/PROJECT_STRUCTURE.md): **実験データ・設定・フォルダ構成の管理ルール**
- [docs/PHASE4_ARTIFACT_AUDIT.md](docs/PHASE4_ARTIFACT_AUDIT.md): **CMakeビルド・追跡済み重み・保全条件の監査**
- [docs/CHAT_HANDOFF_PROMPT.md](docs/CHAT_HANDOFF_PROMPT.md): **新しいChatGPTチャットを開始するときのコピー用プロンプト**

実行方法・今後の優先事項は上記の引き継ぎ資料を参照してください。

## 更新について

引き継ぎ資料は**毎コミットではなく、研究上の節目**（戦略追加、学習経路完成、ルール・報酬・特徴量・行動空間の変更、週／ゼミ単位の大きな進捗）で見直します。変更時は更新日・実装済み／予定／未検証を明記し、過去の実験結果を新しい成果と混同しないようにします。
