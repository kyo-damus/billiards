# 研究リポジトリの構成・ファイル管理ルール

更新: 2026-11-09。対象ブランチ: `refactor-phase9-preserve-20261109`。

## 現状と目標

Phase 1開始前のGit管理ファイルは `src/` 59、`scripts/` 53、`tests/` 72、`docs/` 4、`tools/` 3、`outputs/micro/` 3など。Phase 3時点ではテストファイル76件（追加4件）の分類を進めている。過去のモデル107件はGit管理外の `~/billiards_refactor_quarantine_20261109/` に移動済みであり、変更・削除しない。

Phase 1では**新しい実験の保存方針と記録手段**を追加し、既存の`src/`、`scripts/`、`tests/`、学習済みモデル、既存データは移動・削除しなかった。Phase 2の第1段階では、Macro学習の共通ユーティリティの実装だけを`src/pipelines/training_utils.py`へ移し、旧`scripts`モジュールを再公開用ラッパーとして維持する。

```text
billiards/
├── src/                 # Macro/Micro/Goal/物理/Controller
│   └── pipelines/       # 学習補助・チェックポイント読み込み・構築ファクトリ
├── scripts/             # 現行の起動CLI（互換性維持）
├── configs/
│   ├── macro/           # 人が読む実験設定
│   └── smoke/           # 人が読む小規模検証設定
├── tests/               # テスト（Phase 3で76ファイル分類）
│   ├── unit/            # 局所的な契約の検証
│   │   ├── pipelines/   # 共通処理と旧import互換性
│   │   ├── tools/       # 実験準備・管理ツール
│   │   ├── game/        # ゲームルールのデータ型・簡易GameRules
│   │   ├── macro/       # Macro候補・Policy/Q・局所評価
│   │   │   └── position/ # Position候補・到達可能性
│   │   ├── micro/       # SAC・Residual・位置調整
│   │   ├── goal/        # TacticalGoalと目標条件付け
│   │   ├── defense/     # 防御候補・相手脅威の単体評価
│   │   └── env/         # ショット幾何などの純粋関数
│   ├── integration/     # 実物理・階層構成を通した接続確認
│   │   ├── physics/    # FastFiz・Micro環境・物理遷移
│   │   ├── agent/      # 階層AgentとMixed rollout
│   │   ├── defense/    # Defense物理ショットと遷移
│   │   └── goal/       # 実環境でのGoal sampler
│   ├── pipelines/       # Macro学習・rollout変換の処理契約
│   │   ├── macro/      # Macroモデルの小規模学習・反復
│   │   ├── rollout/    # 訪問分布・リターン教師生成
│   │   ├── micro/      # Residual・Gateの教師データ/学習
│   │   └── evaluation/ # 評価CLIヘルパー・集計
│   └── README.md        # テスト分類と実行方法
├── tools/               # 監査、退避、実験ディレクトリ準備
├── docs/                # 設計、引き継ぎ、保守記録
├── artifacts/           # Git管理外。新しい実験時に生成
│   └── runs/
│       └── <run_id>/
│           ├── config.json
│           ├── metadata.json
│           ├── outputs/
│           └── logs/
├── outputs/micro/       # 既存の追跡済みモデル3件。今は移動しない
└── build/               # CMake生成物。今は構成変更しない
```

将来的には `artifacts/checkpoints/current/`、`artifacts/checkpoints/archive/`、`artifacts/datasets/` を設けてもよいが、**既存モデルへのパス依存を検証する前には移動しない**。

## 新しい実験の記録

1. `configs/` のテンプレートを編集・複製する。設定内容は記録用であり、現行CLIはJSONを自動読み込みしない。
2. 一意の `run_id` を選ぶ。推奨 `YYYYMMDD_<目的>_<番号>`。例: `20261109_macro_visit_01`。日時、モデル版、seedを識別できる命名にする。
3. 準備コマンドを実行する。ファイル・ディレクトリが既存なら上書きを拒否する。

```bash
python tools/prepare_run.py \
  --config configs/smoke/mixed_one_turn.example.json \
  --id 20261109_mixed_smoke_01
```

4. `artifacts/runs/<run_id>/outputs/` にスクリプトの出力先を明示指定し、実行した**正確なコマンド、コンソール出力、例外、学習結果**を `logs/` に保存する。例: `python -m scripts.iterate_mixed_macro_policy ... --output-dir artifacts/runs/<run_id>/outputs`。ここでの `...` は説明用であり、実際には必要な引数をすべて指定する。
5. `metadata.json` には準備時点のGitコミット・ブランチ・作業ツリー変更有無、Python版、設定のSHA-256を自動保存する。これは**実験実行自体を記録するものではない**。実行直前に準備し、別途実行コマンドを残す。

区別すべきデータ契約: Macroの12D状態、Attack旧25DとMixed 31D候補、Candidate PolicyのMCTS訪問分布教師、Candidate Qのモデルベース2-stepと実行済みMC教師、State Vの完了ゲームのみの教師、`turn_aware`、`candidate_encoding_version`、`target_kind`。設定ファイルだけで互換性を保証しない。

## Gitの管理ルール

| 管理する | 管理しない（ローカルに保存） |
| --- | --- |
| `src/`、`scripts/`、`tools/`、`tests/` | `artifacts/` 以下の生成物 |
| `configs/` 内の設定テンプレート | `.pt`、`.pth`、`.npz` の新規生成物 |
| `docs/`、README、`CMakeLists.txt` | 実験ログ、生成キャッシュ、個人情報を含むローカルデータ |

旧 `outputs/micro/` にはGit追跡済みのモデル3件がある。 `.gitignore` は**既に追跡しているファイルを非追跡にはしない**ため、ここは別途判断する。監査済みの既存50件と退避済み107件を一括で移動・削除しない。

重要な実験は、設定・Gitコミット・seed・教師種別・チェックポイント出所・出力をセットで保管し、必要に応じて別媒体にバックアップする。 `artifacts/` はGitに含まれないため、GitHubへのpushだけではバックアップされない。

## 次の移行フェーズ（未実施）

- **Phase 2（主要部分完了）**: `src/pipelines/` に共通学習補助・チェックポイントローダー・Attack/Mixedファクトリの本体を移動。旧 `scripts/` importを維持。22件の関連pytestと、2026-11-09のCPU 1ターンsmokeで移動前と同一結果を確認。
- **Phase 3（分類・指定テスト完了）**: 全8段階でテスト76/76ファイルをunit36・integration11・pipelines29に配置。最終段階の指定40テストも2026-11-09に全件PASSED（8.91秒）、Git作業ツリークリーン。全スイート一括検証・モデル性能評価とは区別する。
- **Phase 4（監査開始）**: [PHASE4_ARTIFACT_AUDIT.md](PHASE4_ARTIFACT_AUDIT.md) にGit追跡状態・CMake/ネイティブ拡張依存・ローカル保全手順を記載。`build/Makefile` と旧 `outputs/micro/` 3モデルは**削除・移動・追跡解除しない**。DGXの現物を読み取り確認してから次の処置を判断する。

各フェーズで、関連テストと少量の統合smokeを実行し、研究モデルの学習条件・探索・FastFizの挙動を変更しない。
