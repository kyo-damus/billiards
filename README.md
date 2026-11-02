# Billiards Hierarchical RL / ビリヤード階層型強化学習

本リポジトリは、ビリヤードにおける**戦術選択（Macro）**と**物理ショット制御（Micro）**を分離し、強化学習とモデルベース探索を統合する研究用プロトタイプです。

> **研究を引き継ぐ人・AIへのお願い：** 詳細は [docs/RESEARCH_CONTEXT.md](docs/RESEARCH_CONTEXT.md) を最初に読んでください。ここには、研究目的、設計上の意図、実装済み機能と未実装部分の区別、直近の進捗、作業時の注意事項、更新ルールを記録します。実装の正確な仕様は常に最新コード・テストを確認してください。

## 目標となる全体像

1. **Macro**: 盤面・手番・スコアに応じて、攻撃／守備、対象球・ポケット、次ショットに有利な手球位置などの戦術候補を生成する。
2. **Candidate Policy / Q / State V**: 可変個の戦術候補に事前確率と価値を与え、将来的に多数ターンの判断を学習する。
3. **MCTS**: Macro候補をMicroと物理シミュレータ（FastFiz）によって展開し、行動を選択する。
4. **Micro**: Goal-conditioned SACを基本に、連続パラメータ（方向・力など）から物理ショットを実行する。
5. **GameRules**: 実行後の状態、得点、手番、終了条件を更新する。

現在は、**的球2個＋手球1個の簡易環境**でシステムの実装・接続を確認している段階です。公式競技ルールに準拠したゲームAIの完成や、実対戦での強さを示すものではありません。

## 現状（2026-11-02時点）

- DIRECT / POSITION 攻撃と、物理ショット候補に基づく SAFETY を扱う31次元のMacroCandidate経路を構築。
- Macro Candidate Policy、候補依存Q、状態価値V、MCTS、Micro、FastFiz、簡易GameRulesを接続。
- 複数ターンの軌跡保存、MCTS訪問分布によるPolicy学習、完了ゲームからの行動リターン学習、Q/Vの任意更新を実装。
- **小規模smokeが完走したことと、学習性能が良いことは区別する。** Macro Q/Vの経験学習、Defense選択、ゲーム勝率等について結論はまだ出ていない。
- 2026-11-02の最新smokeは2ゲーム中1ゲーム終了。Policyを更新し、完了ゲーム不足によりQ/Vを更新しないガードが機能。

## コード・関連ドキュメント

- `src/macro/`: 戦術候補、Candidate Policy/Q、State V、MCTS、攻守遷移、局所評価
- `src/micro/`: Goal-conditioned SAC、Position Residual、守備ショット実行の補助
- `src/controller/`: 階層型エージェント、複数ターン実行、経験ログ
- `src/env/`: FastFiz連携、Micro環境
- `src/game/`: 簡易GameRules
- `scripts/`: データ生成、学習、診断、統合smoke
- `tests/`: 関連機能の単体・結合テスト
- [docs/defense_v0_design.md](docs/defense_v0_design.md): Defense v0と実装段階ごとの技術ログ
- [docs/RESEARCH_CONTEXT.md](docs/RESEARCH_CONTEXT.md): **研究方針と引き継ぎ資料（主に更新する文書）**
- [docs/CHAT_HANDOFF_PROMPT.md](docs/CHAT_HANDOFF_PROMPT.md): **新しいChatGPTチャットを開始するときのコピー用プロンプト**

実行方法・今後の優先事項は上記の引き継ぎ資料を参照してください。

## 更新について

引き継ぎ資料は**毎コミットではなく、研究上の節目**（戦略追加、学習経路完成、ルール・報酬・特徴量・行動空間の変更、週／ゼミ単位の大きな進捗）で見直します。変更時は更新日・実装済み／予定／未検証を明記し、過去の実験結果を新しい成果と混同しないようにします。
