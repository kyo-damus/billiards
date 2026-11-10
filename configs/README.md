# 実験設定（記録用テンプレート）

- `macro/`: Macro Policy反復、Q/V学習などの設定例
- `smoke/`: 小規模な結合動作確認の設定例

現在の `scripts/*.py` は、これらJSONを `--config` で直接受け取りません。**設定内容は準備時の記録用**であり、実験のCLI引数はこれまでどおり明示して実行してください。

`python tools/prepare_run.py --config configs/smoke/mixed_one_turn.example.json --id 20261109_mixed_smoke_01` は、設定のコピーとGitのHEADを保存するだけです。実際に実行するときは、該当するCLIの `--output-dir` などに `artifacts/runs/<ID>/outputs/` を指定し、実行コマンドを `logs/` に記録してください。

モデル次元（旧25Dと混合31D）、`candidate_encoding_version`、`turn_aware`、Policy訪問分布教師とQ/Vの教師種別は、テンプレートの注記を信じるだけでなく、**実際のチェックポイント・データのメタデータで検証**してください。テンプレートを編集しても既存モデルが変更されることはありません。

## Root同点処理のpaired比較（2026-11-10）

`configs/macro/mcts_tie_paired_eval.example.json` は、`first` / `empirical_q` を新規seed 95000〜95023、探索予算10・50、最大8ターンで比較する**記録用テンプレート**。実行ツールは設定を自動で読まない。実行前に`tools/prepare_run.py`で独立runを作り、CLIにはテンプレートと同じチェックポイント・seed・探索予算を渡す。

`scripts.compare_mixed_rollout_modes` はseedごとの初期GameStateを照合して初手局所報酬・ゲームスコア・完了件数を比較する。**ゲーム完了は勝率ではない。** 比較を済ませるまで効果の断定はしない。
