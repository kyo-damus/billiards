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

## DGXでの現物確認（2026-11-09・ユーザー実行）

ユーザーが `git pull --ff-only` 後、読み取り専用コマンドの結果を報告した。

- ブランチ `refactor-phase9-preserve-20261109`、リモートと同期済み、作業ツリークリーン（`git status --short --branch`）。
- Git追跡中: `build/.gitignore`、`build/Makefile`、`outputs/micro/micro_sac_final.pt`、`outputs/micro/micro_sac_step_2500.pt`、`outputs/micro/micro_sac_step_5000.pt`。
- `build/billiard_env_cpp.cpython-311-x86_64-linux-gnu.so`: 299K、2026-11-01 21:51。現用の `bill_sub1`（Python 3.11）に対応する拡張として配置されている。
- `build/billiard_env_cpp.cpython-313-x86_64-linux-gnu.so`: 657K、2026-06-15 15:18。Python 3.13用の別ABI拡張が共存。**容量・更新日が異なることは重複・不要の証拠にならない**。保全継続。
- Git追跡3モデル: 各 `ls -lh` で約1.4M、2026-08-25更新。存在と追跡を確認したが、どの実験で使うか、現行ソースから参照されるか、バックアップにバイト単位で含まれるかは**まだ未確認**。
- `~/billiards_research_backup_20261109.tar.gz` はファイルとして存在、`~/billiards_refactor_quarantine_20261109/` はディレクトリとして存在。`test -f` / `test -d` は内容・整合性・SHA-256を検証するものではない。
- 今回、再ビルド・削除・移動・追跡解除を実施していない。

### 次の読み取り専用調査

現在の `src/`・`scripts/`・`configs/` のコードや設定に、追跡3モデルと `outputs/micro/` への**直接の文字列参照**があるか `git grep` で確認する。結果0件でも動的に参照される可能性は残る。ローカルの既存tarがそれらを含むかファイル名一覧を確認し、バックアップの内容一致は別途SHA-256で検証する。

Makefile追跡解除は**まだ提案段階**。GitHubで追跡解除コミットを作り、そのままユーザーがpullすると `build/Makefile` の作業コピーが消える可能性があるため、**事前の退避と再生成手順の確認なしに変更しない**。`build/` と2種類の `*.so` は現状維持する。

次の判断はユーザーによるDGX現物と参照先の報告を受けてから行う。

## Git上の参照・tar内ファイル名の調査（2026-11-09）

ユーザーが `git grep -n -F` を `src/`・`scripts/`・`configs/` に対して実施し、以下の結果を報告。作業ツリーはクリーン。

- `scripts/eval_micro.py:85`: `--checkpoint` のデフォルトが `outputs/micro/micro_sac_final.pt`。
- `scripts/train_micro.py:228`: `micro_sac_final.pt` を保存。ソースを追加確認すると、`--output-dir` のデフォルトが `outputs/micro`（277行付近）。
- 同スクリプトは `--save-interval` （デフォルト10,000）ごとに `micro_sac_step_{step}.pt` を保存（217〜223行付近）。したがって `micro_sac_step_2500.pt` と `micro_sac_step_5000.pt` のリテラル文字列が検索で出てこないことは、未使用・不要を意味しない。
- `tar -tzf` により `research/outputs/micro/micro_sac_final.pt`、`research/outputs/micro/micro_sac_step_2500.pt`、`research/outputs/micro/micro_sac_step_5000.pt` の**3件全てがtar内に名前として存在**することを確認。
- **tar内と手元の3モデルの内容のSHA-256照合は未実施**。研究モデルの起源・用途や数値比較上の必要性も未確定。
- 今後 `python -m scripts.train_micro` を新規実行すると、デフォルトでは追跡中の `outputs/micro/` に書き込む。現物を誤って上書きしないよう、新規学習では `--output-dir artifacts/runs/<run_id>/outputs/micro` 等の**新しい出力パスを明示する**。現行CLIのデフォルトは変更していない。
- `eval_micro.py` は少なくとも従来の `micro_sac_final.pt` をデフォルトで使用するため、`outputs/micro/` にある3モデルのディレクトリ・ファイルは全て現状維持。追跡解除・削除は検討していない。

### バックアップと現物のSHA-256照合（次の読み取り専用ステップ）

以下は3モデルを**読み込むだけ**で、学習・保存・削除は行わない。tarと現物のSHA-256が一致したか結果を確認してから次の判断に進む。

```bash
cd ~/research
python - <<'PY'
from pathlib import Path
import hashlib
import tarfile

names = ["micro_sac_final.pt", "micro_sac_step_2500.pt", "micro_sac_step_5000.pt"]
archive_path = Path.home() / "billiards_research_backup_20261109.tar.gz"

def sha256(stream):
    h = hashlib.sha256()
    for chunk in iter(lambda: stream.read(1024 * 1024), b""):
        h.update(chunk)
    return h.hexdigest()

with tarfile.open(archive_path, "r:gz") as archive:
    for name in names:
        file_path = Path("outputs/micro") / name
        member = archive.getmember("research/" + file_path.as_posix())
        if not member.isfile():
            raise ValueError(f"not a file in tar: {member.name}")
        with file_path.open("rb") as stream:
            current_hash = sha256(stream)
        with archive.extractfile(member) as stream:
            backup_hash = sha256(stream)
        print(f"{name}: {'MATCH' if current_hash == backup_hash else 'MISMATCH'}")
PY

git status --short --branch
```

`MISMATCH` は削除理由ではなく、バックアップ取得後にファイルが変わった可能性を含む**追加調査の合図**。現物は保持する。

## 3件のMicroチェックポイントのtarとのSHA-256一致（2026-11-09）

ユーザーが既存の `~/billiards_research_backup_20261109.tar.gz` 内の各ファイルと、`outputs/micro/` の現物をstreamごとにSHA-256で比較して以下を報告した。

| 対象 | 現物とバックアップ |
| --- | --- |
| `micro_sac_final.pt` | `MATCH` |
| `micro_sac_step_2500.pt` | `MATCH` |
| `micro_sac_step_5000.pt` | `MATCH` |

`git status --short --branch` もクリーン。**これら3件はバックアップとの内容一致が確認済み**であり、従来の「tar内の名前のみ確認」という不確実性は解消した。ただし、モデルの役割や将来の比較実験への必要性は別問題なので、3件とも削除・移動・追跡解除しない。

### Makefile追跡整理に向けた準備（既存buildは変更しない）

現行 `build/Makefile` はGit追跡中でCMake生成・環境固有パスを含む。Git追跡から外す処理は**まだ実施しない**。以下の順に、まずリポジトリ外へオリジナルを保全し、次に隔離された一時ディレクトリへCMakeを設定できるか試す。既存の `build/`、`*.so`、`outputs/micro/` には書き込まない。

```bash
cd ~/research
mkdir -p ~/billiards_refactor_safety_20261109
backup=~/billiards_refactor_safety_20261109/Makefile.original
if [ ! -e "$backup" ]; then
  cp -p -- build/Makefile "$backup"
fi
cmp -s -- build/Makefile "$backup" && echo "Makefile backup MATCH" || echo "Makefile backup MISMATCH"

cmake --version | head -n 1
python --version
git submodule status -- libs/fastfiz

scratch="$(mktemp -d /tmp/billiards-cmake-phase4.XXXXXX)"
echo "Isolated configure directory: $scratch"
cmake -S . -B "$scratch"
git status --short --branch
```

一時ディレクトリのconfigureでエラーが出ても既存buildは触らず、**エラーログを先に確認してから**対応する。成功しても実際のビルドと `*.so` の再生成はまだ確認したことにならない。安全なバックアップとconfigure検証を経てから `build/Makefile` の追跡方針を改めて判断する。

**注意:** 追跡解除コミットをGitHubで先に作成すると、その後のpullでローカルのMakefileが削除される可能性がある。既存buildと追跡済み重みは現状維持。
