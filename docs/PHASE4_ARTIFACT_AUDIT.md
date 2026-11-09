# Phase 4: 生成ファイル・学習済みモデルの追跡監査

更新: 2026-11-09。作業ブランチ: `refactor-phase9-preserve-20261109`。

**状況:** GitHub上の追跡状態・ソース依存の静的監査は実施済み。DGXローカルの現物確認・追跡解除の可否判断は**未実施**。この段階では**ファイル移動・削除・Git追跡解除を行わない**。

## Phase 3完了の根拠

- 2026-11-09の第8段階で移した9ファイルに対応するユーザー実行結果は **40 passed in 8.91s**。Git作業ツリーはクリーン。
- 第1〜8段階の指定テストはそれぞれ成功。合計76ファイルは `tests/unit/` 36、`tests/integration/` 11、`tests/pipelines/` 29に分類。ルートの `tests/test_*.py` は0件。
- **全pytestスイート一括実行はしていない。** Phase 2での5モデルCPU 1ターンsmokeの一致は機能接続の確認であり性能評価ではない。

## Git追跡状態（GitHub HEAD `2294296` を基準とする監査）

| 対象 | 追跡状態 | Git上のサイズ | 扱い |
| --- | --- | ---: | --- |
| `build/Makefile` | 追跡中 | 12,902 B | CMake自動生成。移動・追跡解除は保留 |
| `build/.gitignore` | 追跡中 | 11 B | 旧build内の `*.pt` / `*.pth` 無視。保留 |
| `outputs/micro/micro_sac_final.pt` | 追跡中 | 1,410,075 B | 既存重み。保持 |
| `outputs/micro/micro_sac_step_2500.pt` | 追跡中 | 1,410,223 B | 既存重み。保持 |
| `outputs/micro/micro_sac_step_5000.pt` | 追跡中 | 1,410,223 B | 既存重み。保持 |

上記3チェックポイントは合計 **4,230,521 B**（約4.03 MiB）。ルート `.gitignore` は `*.pt`、`*.pth`、`*.npz` と一部CMake生成物を無視するが、**すでに追跡しているファイルに遡って適用されない**。チェックポイントを不要と断定する根拠にはならない。

## 物理ビルドの依存関係

- `CMakeLists.txt` は `pybind11`、FastFizのソース、GSLに依存し、`billiard_env_cpp` をビルドする。`libs/fastfiz` はGitサブモジュールとして宣言されている。
- `src/env/fastfiz_simulator.py` はリポジトリ内の **`build/` をPython importパスへ追加して `billiard_env_cpp` をimportする**。したがって `build/` 自体やネイティブ拡張 `*.so` を整理対象として不用意に除去してはいけない。
- 追跡中の `build/Makefile` は `CMAKE generated file: DO NOT EDIT!` と明記され、CMake 4.3によって生成されたもの。CMake実行ファイル、ソース/ビルドディレクトリなどの**環境固有の絶対パス**が埋め込まれている。Gitで共有するファイルとしては再現性・移植性上の改善候補だが、現行DGXのビルドが正常なことから **直ちに削除してよいとは判断しない**。
- GitHubのツリーはローカルの `build/billiard_env_cpp*.so` の存在・版・再ビルド可能性を証明しない。実機確認が必要。

## ローカル資産の保全

以前の監査では、157件のローカル成果物のうち107件をバックアップとのSHA-256照合後に `~/billiards_refactor_quarantine_20261109/` へ退避済み、50件を作業ディレクトリに残していた。バックアップは `~/billiards_research_backup_20261109.tar.gz`。これらは**いずれも削除・移動せず保全**する。Gitブランチだけではローカル専用の重み・データを復元できない。

特に将来 `git rm --cached` による追跡解除を考える場合、**GitHub上で削除コミットを作ってからローカルへ `git pull` すると、もともと追跡されていた現物が作業ツリーから消えることがある**。手元に保存されると軽々しく仮定しない。別パスへの検証済みコピー、バックアップ、使用中の参照、復元手順を先に固めるまで実施しない。

## DGXでの読み取り専用確認手順

```bash
cd ~/research
git status --short --branch
git ls-files -- build/Makefile build/.gitignore outputs/micro/
ls -lh build/billiard_env_cpp*.so
ls -lh outputs/micro/*.pt
test -f ~/billiards_research_backup_20261109.tar.gz && echo "backup present"
test -d ~/billiards_refactor_quarantine_20261109 && echo "quarantine present"
```

存在確認だけでよく、今は `cmake --build` や `git rm` を行わない。詳しい一覧が必要な場合は、既存の**読み取り監査** `tools/audit_local_artifacts.py` で生成先を `local_artifacts_phase4_20261109.tsv` に指定できる。このコマンドはモデルやデータを移動せず、報告TSVのみ作成する。

```bash
python tools/audit_local_artifacts.py \
  --output local_artifacts_phase4_20261109.tsv
```

## 次の判断順

1. 実機にFastFiz拡張と必要モデルが存在し、退避先・バックアップが健在であることを確認。
2. `build/Makefile` のGit追跡方針を決定する前に、CMake再生成・ビルドがDGX上で再現できるかを**別手順で検証**（今回の監査では再ビルドしない）。
3. `outputs/micro/` の3モデルについて現行CLIの使用有無、研究比較・再現の必要性、バックアップのバイト一致を確認。
4. 保全・再現に問題がないことと利用者の判断を経てから、追跡解除やディレクトリ整備を個別コミットで実施。原ブランチ・履歴・研究成果物は引き続き保護する。

**監査時点の結論:** 冗長な追跡対象は存在するが、現時点で無条件に消してよいファイルは確定していない。
