# リポジトリのフォルダ構造とファイル管理ガイド

**更新日:** 2026-11-10（JST）  
**対象:** `kyo-damus/billiards` の `refactor-phase9-preserve-20261109` ブランチ  
**目的:** 初めて本研究を引き継ぐ人やChatGPTが、**どのファイルをどこに置き、何をGitで管理し、何を保全するか**を、この文書だけで把握できるようにする。

この文書を**フォルダ構造・ファイル管理の正本**とする。研究目的・学習仕様は [RESEARCH_CONTEXT.md](RESEARCH_CONTEXT.md)、リファクタリングの変更履歴は [REFACTORING_PLAN.md](REFACTORING_PLAN.md) に記録する。実際のファイル構成やCLI仕様が変わったら、必ず最新のGitツリーとコードを確認する。

## 1. 現行の構成（Phase 1〜4整理後）

```text
~/research/                         # ローカルのリポジトリルート
├── src/                             # Python実装本体
│   ├── common/                      # 共通の型・定義
│   ├── controller/                  # 階層Agent、実行・Rollout管理
│   ├── env/                         # FastFizラッパー、Micro環境
│   ├── game/                        # 簡易ゲームルール
│   ├── goal/                        # TacticalGoal・Goal関連
│   ├── macro/                       # 候補生成・Policy/Q/V・MCTS
│   ├── micro/                       # Goal-conditioned SAC、Residual制御
│   └── pipelines/                   # 共通処理をsrc側へ集約
│       ├── training_utils.py
│       ├── checkpoint_loading.py
│       ├── attack_components.py
│       ├── mixed_components.py
│       └── rollout_analysis.py       # 保存済みMixed Rolloutの統計集計
├── scripts/                         # 学習・評価・データ生成・smokeのCLI
│                                    # 一部は旧import互換ラッパー
│                                    # analyze_mixed_rollouts.py: NPZ読取集計
├── configs/                         # 実験設定の記録用テンプレート
│   ├── macro/
│   ├── smoke/
│   └── README.md
├── tests/                           # 計77テストファイル
│   ├── unit/                        # 36: 純粋/局所的な契約
│   │   ├── macro/position/
│   │   ├── micro/
│   │   ├── goal/
│   │   ├── defense/
│   │   ├── env/
│   │   ├── game/
│   │   ├── pipelines/
│   │   └── tools/
│   ├── integration/                 # 11: 実物理・Agent・Defenseなどの接続
│   │   ├── physics/
│   │   ├── agent/
│   │   ├── defense/
│   │   └── goal/
│   ├── pipelines/                   # 30: 学習/評価/教師データ生成の処理
│   │   ├── macro/
│   │   ├── micro/
│   │   ├── rollout/
│   │   └── evaluation/
│   └── README.md                    # テストの細かな指定方法
├── tools/                           # 実験準備・監査・旧成果物の安全な退避
│   ├── prepare_run.py
│   ├── audit_local_artifacts.py
│   ├── refactor_audit.py
│   └── quarantine_legacy_build_artifacts.py
├── docs/                            # 研究方針・設計・保全記録
│   ├── PROJECT_STRUCTURE.md         # ★この文書（構成と管理の正本）
│   ├── RESEARCH_CONTEXT.md          # 研究目的・実装状況の引き継ぎ
│   ├── CHAT_HANDOFF_PROMPT.md       # 新しいチャットへの移行文面
│   ├── REFACTORING_PLAN.md          # 整理の変更・テスト履歴
│   ├── PHASE4_ARTIFACT_AUDIT.md     # CMake・モデル保全の詳細
│   └── defense_v0_design.md
├── libs/fastfiz/                     # Gitサブモジュール（物理エンジン）
├── CMakeLists.txt                    # C++/pybind11のビルド設定
├── cpp_simulator.cpp                 # Python用C++バインディング
├── build/                            # CMake出力先（ディレクトリは維持）
│   ├── .gitignore                    # Git追跡中
│   ├── Makefile                      # ローカルのみ・Git無視
│   └── billiard_env_cpp*.so          # ローカルのみ・Git無視
├── outputs/micro/                    # 過去の追跡済みMicroモデル3件
│   ├── micro_sac_final.pt
│   ├── micro_sac_step_2500.pt
│   └── micro_sac_step_5000.pt
├── artifacts/                        # 新規実験の生成先（Git管理外）
│   └── runs/<run_id>/
│       ├── config.json
│       ├── metadata.json
│       ├── outputs/
│       └── logs/
├── .gitignore
├── .gitmodules
└── README.md
```

上の `build/Makefile`、`*.so`、`artifacts/` は**現行DGX上のローカル生成物/運用先**として示したもので、Gitのclone直後に必ず存在するとは限らない。 `artifacts/runs/<run_id>/` は `tools/prepare_run.py` を実行して初めて生成される。

## 2. 役割・新規ファイルの置き場所

| 内容 | 保存先 | 管理方針 |
| --- | --- | --- |
| Macro/Micro/Goal、ゲーム・物理との接続の実装 | `src/` | Git管理。研究処理の本体 |
| 共通ローダー・学習ユーティリティ・構築ファクトリ | `src/pipelines/` | Git管理。重複した本体を `scripts/` に作らない |
| 実行可能な学習・評価・dataset生成コマンド | `scripts/` | Git管理。既存 `python -m scripts.*` を維持 |
| 単体/統合/学習パイプラインのテスト | `tests/{unit,integration,pipelines}/` | Git管理。機能と依存の性質に応じて選択 |
| 開発・保守・データ監査ツール | `tools/` | Git管理。退避・移動ツールの実行は十分に確認 |
| 説明、実験設計、引き継ぎ資料 | `docs/`、各README | Git管理 |
| 実験の設定テンプレート | `configs/` | Git管理。**現行CLIはJSONを自動読込しない** |
| 新しい学習済み重み・評価ログ・生成データ | `artifacts/runs/<run_id>/outputs/`、`logs/` | Git管理外。出力先を明示する |
| C++のビルドファイル・共有ライブラリ | `build/` | ローカル生成物。既存の`build/`は消さない |
| 旧Microの3重み | `outputs/micro/` | **例外的にGit追跡中・維持**。新しい実験で上書きしない |
| 過去の大容量データと退避済みモデル | 下記「4. 保全・バックアップ」 | Git管理外。勝手に削除・移動しない |

`src/pipelines/` の実装を `scripts/attack_components.py` 等の旧import経路から呼ぶ互換ラッパーは意図的に残してある。研究コードを整理するときは、Macro状態12D、旧Attack候補25D、Mixed候補31D、手番・報酬・教師種別（MCTS訪問教師、2-step Q、実行済みMC return Q/V）を混同・変更しない。

## 3. Gitで管理するもの / 管理しないもの

- **通常追跡するもの:** `src/`、`scripts/`、`tests/`、`tools/`、`docs/`、`configs/`、`README.md`、`CMakeLists.txt`、`cpp_simulator.cpp`、サブモジュールの参照。
- **通常追跡しないもの:** `*.pt`、`*.pth`、`*.npz`、`*.so`、`*.o`、CMakeの生成キャッシュ、`/build/Makefile`、`/artifacts/`、ローカルの監査TSV・実験ログ（`.gitignore` が適用される場合）。
- **例外:** `outputs/micro/` にある `micro_sac_final.pt`、`micro_sac_step_2500.pt`、`micro_sac_step_5000.pt` の3ファイルは以前からGit追跡中で、2026-11-09時点も維持。ルートの `.gitignore` はすでに追跡済みのファイルには遡及しない。

`build/Makefile` はPhase 4でGit追跡を解除し、`/build/Makefile` としてignoreに登録済み。2026-11-09、DGXでは元のMakefileを退避コピーから復元し、`git check-ignore`・FastFiz回帰11テストの成功を確認。**`build/`そのものや`billiard_env_cpp*.so`は削除しない。** `src/env/fastfiz_simulator.py` が `build/` にあるネイティブ拡張をimportするため。

GitHubにpushしても、無視した実験重み・データ・ `artifacts/` は保存されない。重要な成果はGitコミット、モデルの出所、seed、実行コマンド、出力を関連付け、必要に応じて別媒体へバックアップする。

## 4. 現在の成果物の保全とバックアップ

2026-11-09の監査で、ローカル研究成果物157件を棚卸しした。

| 対象 | 配置/状態 | 取り扱い |
| --- | --- | --- |
| 旧build系モデル107件 | `~/billiards_refactor_quarantine_20261109/` に退避済み | **削除せず保管**。`tools/quarantine_legacy_build_artifacts.py --restore` で戻せる |
| 残る研究成果物50件 | `~/research` 配下に残存 | 主要モデル・比較モデル・dataset・反復実験を保全 |
| 2026-11-09の研究バックアップ | `~/billiards_research_backup_20261109.tar.gz` | 旧107件は退避前に照合済み |
| 旧Micro3モデル | `outputs/micro/` に存在 | tar中の同名ファイルと**SHA-256が3件すべて一致**。現状のまま保管 |
| 元のCMake Makefile | `~/billiards_refactor_safety_20261109/Makefile.original` | DGX上の復元用コピー。元Makefileとの `cmp` 一致を確認 |

**重要:** 上記以外のすべての研究ファイルまでバックアップのバイト一致を保証したわけではない。未確認ファイルを「未使用だから削除可能」と決めない。特にチェックポイント名の文字列検索に現れなくても、CLI引数や動的なファイル名で参照されることがある。

今後 `git rm --cached` などで追跡を変更する場合、**GitHubに削除コミットを作成してpullすると、手元の追跡ファイルが消える可能性がある**。実施前に別パスへコピーし、内容一致・復元方法を検証すること。原ブランチ `phase9-defense-v0` とバックアップは保護する。

## 5. 新しい実験を記録する方法

1. `configs/` のJSONテンプレート（例: `configs/smoke/mixed_one_turn.example.json`）をベースに設定を記録する。
2. 一意のrun ID（例: `20261109_macro_visit_01`）を選ぶ。
3. 以下を実行し、上書き禁止の新規実験ディレクトリを作る。

```bash
cd ~/research
python tools/prepare_run.py \
  --config configs/smoke/mixed_one_turn.example.json \
  --id 20261109_mixed_smoke_01
```

作成される `artifacts/runs/<run_id>/` には、設定のスナップショット `config.json`、準備時点のGitコミット/ブランチ・Python版などを記録した `metadata.json`、`outputs/`、`logs/` が置かれる。**このツール自体は学習・評価を実行しない。**

4. 実際の学習・評価コマンドでは、個々のスクリプトが受け付ける出力オプションを確認し、生成物を `artifacts/runs/<run_id>/outputs/` 配下に保存する。例として `scripts/train_micro.py` は `--output-dir` を受け付ける。指定を省略すると旧 `outputs/micro/` に出力して追跡済みモデルを上書きし得るため、**新しい学習では必ず出力先を明示する**。

```bash
# 書き込み先を指定する場合の例（実行前に学習条件を確認）
python -m scripts.train_micro \
  --output-dir artifacts/runs/<run_id>/outputs/micro \
  --steps <steps>
```

このコマンドの `<run_id>`、`<steps>` は**置換用の説明表記**であり、そのまま実行しない。実行したコマンド・seed・チェックポイント・評価ログは `logs/` に別途保存する。

## 6. ビルドとテストの扱い

- `libs/fastfiz` はサブモジュール。`CMakeLists.txt` にC++/pybind11・GSL等の依存がある。
- **新しいクローンでビルドする際の基本形**（必要なC++依存が導入済みの場合）:

```bash
git submodule update --init --recursive
cmake -S . -B build
cmake --build build --parallel 2
```

- 既存DGXではPhase 4で既存 `build/` を変更せず、別の `/tmp/` ディレクトリでconfigure → C++ビルド → Python 3.11で `billiard_env_cpp` importと `BilliardSimulator()` を確認済み。既存の `build/Makefile` はGit非追跡のまま利用可能。
- `tests/` は **unit 36 / integration 11 / pipelines 30 = 計77ファイル**（Phase 3の76件＋Phase 5で追加した集計テスト1件）。既存76件は各段階の指定テストが通過済み。新規テストはDGXでの確認待ちであり、**全スイート一括テスト済みという意味ではない**。

作業内容に応じて、例えば以下の範囲でテストする。

```bash
python -m pytest tests/unit/macro/ -v
python -m pytest tests/pipelines/macro/ -v
python -m pytest tests/integration/physics/test_fastfiz_simulator.py -q
```

詳細なテストファイルと段階別コマンドは `tests/README.md`。物理テストにはFastFizのPython拡張が必要で、長時間学習を検証のたびに回す必要はない。

## 7. 現時点の整理の到達点と今後の更新

- **Phase 1:** 実験設定と記録の基盤を作成。
- **Phase 2:** `src/pipelines/` に共通処理を分離し、`scripts/` の互換APIを維持。関連テストと5モデルのCPU 1ターンsmokeで確認。
- **Phase 3:** 76テストファイルを機能別に整理し、各段階の指定テストが成功。
- **Phase 4:** 旧研究成果物を保全したまま、CMake生成物 `build/Makefile` をGitの追跡対象から除外。復元後のFastFiz回帰11件がPASS。Micro3モデルは追跡継続かつtarとSHA-256一致。

この構造整理は現行ブランチで一区切り。**mainへのマージ、学習性能改善の実証、全スイート一括テストは未実施。** 今後の変更は研究ロジックの変更とファイル管理の変更を別コミットで扱い、実際のGit構成、`.gitignore`、出力先、バックアップとこの文書を同期する。

研究全体の方針が変わる場合は `docs/RESEARCH_CONTEXT.md` も更新する。変更の技術的な経緯が必要な場合のみ `docs/REFACTORING_PLAN.md` / `docs/PHASE4_ARTIFACT_AUDIT.md` を参照する。

## 8. Phase 5: Mixed Rollout集計（2026-11-10着手）

- **実装:** `src/pipelines/rollout_analysis.py` に保存済みNPZからの集計処理、`scripts/analyze_mixed_rollouts.py` にCLIを追加。物理実行・再学習・既存NPZ更新はしない。
- **指標:** ゲーム完了率（勝率ではない）、停止理由、DIRECT/POSITION/SAFETYの選択数、実際の的球ポケット数、手球スクラッチ、手番交代、Macro局所報酬とゲームスコア差分、MCTS root数/Policy訪問教師に使えるroot数。
- **テスト:** `tests/pipelines/evaluation/test_analyze_mixed_rollouts.py` に6件を追加し、作成時に軽量なローカル検証が成功。**DGX上での新配置テストと既存の実データ集計は未確認**。
- 詳しいCLI例は [tests/README.md](../tests/README.md) を参照。新たな実験結果を性能向上や実際の勝率として表現しない。

## 9. Phase 5: 複数seedのpaired比較

新たに `scripts/compare_mixed_rollout_modes.py` を追加。既存のRollout `rollout_seed_*.npz` を読み取ってseedと初期状態の一致を検証し、`first` / `empirical_q` の初手選択・局所報酬、完了件数、スコア、スクラッチをまとめる。保存先は `artifacts/runs/<run_id>/outputs/` を推奨し、方式ごとに別ディレクトリを用いる。テストは `tests/pipelines/evaluation/test_compare_mixed_rollout_modes.py`。既存モデルや学習処理は変更しない。詳細は [tests/README.md](../tests/README.md)。
