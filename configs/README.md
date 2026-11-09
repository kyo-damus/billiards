# 実験設定（記録用テンプレート）

- `macro/`: Macro Policy反復、Q/V学習などの設定例
- `smoke/`: 小規模な結合動作確認の設定例

現在の `scripts/*.py` は、これらJSONを `--config` で直接受け取りません。**設定内容は準備時の記録用**であり、実験のCLI引数はこれまでどおり明示して実行してください。

`python tools/prepare_run.py --config configs/smoke/mixed_one_turn.example.json --id 20261109_mixed_smoke_01` は、設定のコピーとGitのHEADを保存するだけです。実際に実行するときは、該当するCLIの `--output-dir` などに `artifacts/runs/<ID>/outputs/` を指定し、実行コマンドを `logs/` に記録してください。

モデル次元（旧25Dと混合31D）、`candidate_encoding_version`、`turn_aware`、Policy訪問分布教師とQ/Vの教師種別は、テンプレートの注記を信じるだけでなく、**実際のチェックポイント・データのメタデータで検証**してください。テンプレートを編集しても既存モデルが変更されることはありません。
