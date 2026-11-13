# 研究引き継ぎコンテキスト / RESEARCH_CONTEXT

**更新日:** 2026-11-02（JST）  
**対象リポジトリ:** https://github.com/kyo-damus/billiards  
**作業ブランチ:** `phase9-defense-v0`  
**確認したベースHEAD:** `ba2a1b4b6cfe4af62e3d38823b579041c4a95695`（この文書を追加する直前）  
**関連実装ログ:** [defense_v0_design.md](defense_v0_design.md)

> **ChatGPTや別の開発支援AIは、実装を進める前に本書と当該ブランチの最新コードを確認すること。** 本書は研究意図と意思決定の引き継ぎ用であり、APIやCLI引数などは最新コードを優先する。記載したsmoke結果を一般的な性能保証と解釈しない。

## 1. 研究の最終目的と研究上の問い

ビリヤードの戦術選択と精密な物理制御を**Macro / Microの階層に分離**し、それらを連携させた階層型強化学習・モデルベース探索システムを構築する。

- **Macro（離散・戦術）:** 現在の盤面、残球、手番、スコアなどに基づき、攻撃・防御の方針、狙う球・ポケット、次ショットに有利な配置といった「何を実現するか」を決定。
- **Micro（連続・ショット制御）:** MacroからのTacticalGoalを条件に、具体的なショットパラメータ（角度・力、将来的には撞点やキュー角度を含む）を決定・実行。
- **探索・学習の統合:** MacroCandidateの候補集合をCandidate Policyで順位付けし、候補依存Q・状態価値VとMCTS、FastFizによる先読みを組み合わせる。
- **研究として検証したい点:** 攻守・配置を含む戦術判断とゴール条件付きショットを分担させる効果、探索による改善、候補評価と実ゲーム成果の関係。これらは**今後の本実験で評価する課題**であり、現時点で改善を実証したとは書かない。

**重要:** MuZero、World Modelの学習、公式ルール全般、全物理パラメータ学習を「すでにある」ものとして扱わない。現行設計は既存のFastFiz物理モデルを利用するモデルベース探索であり、学習済みDynamicsネットワークを前提にしていない。

## 2. 現行アーキテクチャ（目標と実装を区別）

```text
GameState（球位置 / pocket状態 / score / current_player）
  -> Macro候補生成
       Attack : DIRECT_ATTACK, POSITION_ATTACK
       Defense: SAFETY（現状は物理グリッド探索済みショット）
  -> 可変個 MacroCandidate
  -> Candidate Policy prior + 候補依存 Candidate Q
  -> turn-aware MCTS（State Vで葉評価、必要な候補を展開）
  -> 選択した MacroCandidate
       Attack : Goal-conditioned Micro（Base SAC、Position power residual）
       Defense: 保存済みの具体的ショットを実行（学習済みDefense Microではない）
  -> FastFiz（物理状態遷移）
  -> Macro local evaluation + SimplifiedGameRules
  -> next GameState（次の手番で再度Macro判断）
```

### 基本データ契約

- `GameState`: 手球1個＋的球2個、球位置 `(3,2)`、`ball_pocket_indices`、2人分のscore、`current_player`。
- Macro状態表現は**12次元**（球位置6、ポケット済みフラグ3、スコア2、手番1）。
- `MacroCandidate`は`MacroAction(strategy, target_ball, target_pocket)`、`TacticalGoal`、守備のみ`PhysicalShotSpec`を持つ。
- **Candidate特徴量:** Attack旧版25次元と、Attack/Safety混合新版31次元は明確に区別する。現行混合Policy/Qは31次元。既存25次元モデルをDefenseに黙って流用しない。
- `TacticalGoalType`: `DIRECT_ATTACK`、`POSITION_ATTACK`、`SAFETY`。
- 攻撃はMicroが角度・力を選ぶ。Positionには既存Baseに**power-only Residual**を加える経路があり、直近の統合smokeで`residual-scale=0.75`などを使用。
- 守備は、現段階では角度・力・一部の撞点候補を物理的に列挙・実行し、First-contact等を検査して生成する。**スピン全般やキュー仰角に対応した学習済みMicro方策が完成しているわけではない。**

### 環境とルールの限界

- `src/game/simple_rules.py`は**SimplifiedGameRules**。的球を落とすと点が入り手番継続、空振り等は手番交代、スクラッチは簡易ゲーム終了、的球2個を落とせば終了。
- これは9-ballなどの公式競技ルールではない。ファウル、合法な当て球、球数の拡張、手球再配置などは今後の設計課題。
- FastFizを使ってMacro候補の実行後盤面を生成しMCTSで探索することは、モデルベースRL/計画として成立する。ただしシミュレータの誤差や同一モデルに依存した楽観的評価を、現実への汎化性能と混同しない。

## 3. 実装済み機能（2026-11-02時点）

### Micro

- 角度・力のGoal-conditioned SACを基礎として学習・評価する経路。
- DIRECTとPOSITIONのGoalを分離。POSITIONでは成功ショットからの教師あり・Residual補正を段階的に試し、現在の統合経路ではBase＋power-only Position Residualを使用。
- FastFiz物理ショットの実行と、その結果に基づく局所評価。
- **未完成:** 撞点やスピンを含む連続行動全体の学習、キュー仰角の学習、学習済み汎用Defense Micro。仰角探索は近接レールでの物理前提条件の診断に留まる。

### Macro / 探索

- DIRECT / POSITION攻撃の候補生成、Outcome-guidedなPosition目標生成。
- SAFETY候補は物理グリッドで探索し、正しい初回接触・非スクラッチ・非ポケット・手番交代等の**簡易条件**で選別。相手のDIRECT可能ショットをFastFizで先読みして脅威を評価。
- Attack/Safety混合の`MacroTransition`、31次元Candidate encoder、候補依存Policy/Q、turn-aware MCTS、状態価値Vの任意読み込み。
- MCTSは手番が変わる継続価値の符号を反転し、`candidate-q-prior-visits`によって候補Qを探索の事前情報に組み込む。
- **注意:** Safetyの相手脅威・教師候補は同じシミュレータおよび限定的なMicro / DIRECT探索に依存。守備の一般的な有効性、合法性全般、勝率向上は未証明。

### 経験収集と学習

- `scripts/collect_mixed_rollouts.py`: 複数の独立seedで有限ターンのゲームを実行・保存。完了・途中終了いずれの履歴も保存。
- `scripts/build_mixed_rollout_visit_dataset.py`: 未完了ゲームを含むMCTSルート訪問分布から、**Candidate Policy用**教師`policy_target`を生成。
- `scripts/train_macro_visit_policy.py`: 訪問分布へのクロスエントロピーで31次元Policyを学習。train/val分割はゲーム単位。
- `scripts/build_mixed_rollout_return_dataset.py`: **ゲーム終了まで記録された履歴だけ**を使い、実行したMacro行動の割引累積局所報酬を教師値として生成。未選択候補に教師値を割り当てない。途中終了をゼロbootstrapで完了扱いしない。
- `scripts/train_macro_candidate_value.py`: 候補Qを教師あり回帰する。対象データには「全候補の2-stepオラクルQ」と「実行済み選択行動のみのMonte Carlo return」の**異なる種類**がある。後者はゲーム単位でtrain/val分割する。
- `scripts/train_macro_state_value.py`: 完了ゲームから得た実行軌跡のMonte Carlo returnを使う状態価値Vの小規模学習経路。
- `scripts/iterate_mixed_macro_policy.py`: boundedな`収集 -> Policy訪問教師 -> Policy更新 -> 次反復`。オプション`--update-critics`では、**その反復で完了ゲームが2件以上あるときだけ**QとVも再学習し、次反復に引き継ぐ。足りなければ従来のQ/Vを保持。
- 現在の反復Q/V更新は基本的に**新たに初期化したモデルの再学習**であり、従来チェックポイントからの継続fine-tuningを保証しない。
- 学習済みネットワークをMCTSへ読み込めても、**これだけで高性能な強化学習が完成したとは言わない**。

## 4. 報酬・教師信号の意味（重要な設計判断）

### 4.1 2つの「報酬」を混同しない

1. `SimplifiedGameRules`が更新するゲームスコア差分（例: 的球を入れたら+1）。
2. `Macro local evaluation`が返す`reward`。狙いと異なるポケットに入るなどの理由で、**ゲームスコアは増えてもMacro局所報酬は負**の場合がある。

Macroの学習・探索や実行済みリターンでは、現状主に後者の局所報酬を使用する。将来の公式ルール/勝敗目的に合わせた報酬設計とは区別する。

### 4.2 手番を考慮したリターン

```text
G_t = r_t + gamma * sign(player_t, player_(t+1)) * G_(t+1)
sign = +1 (同じプレイヤー) / -1 (相手へ手番交代)
```

`r_t`は**その時点の行動プレイヤー視点の即時Macro局所報酬**であり、常に正とは限らない。相手手番の継続価値を引く。即時報酬まで重複して符号反転しない。完了ゲームの最後は終端として将来価値0。

### 4.3 学習ターゲットの由来を保持する

| ターゲット | 入力 | 何を学習しているか | 限界 |
| --- | --- | --- | --- |
| 2-step Candidate Q oracle | 全root候補とFastFiz先読み | `r + gamma * sign * best(next DIRECT reward)` | 次手がDIRECTのみ、モデル依存・選択バイアス |
| Executed-action MC Q | 完了ゲームで実際に選択した候補 | 行動プレイヤー視点の割引実行リターン | 未選択候補のQは不明、軌跡数が必要 |
| State V | 完了ゲームの訪問状態 | 実際の振る舞いの状態価値の近似 | サンプル数が極小、最適価値ではない |
| Candidate Policy visits | MCTS探索のroot訪問分布 | 現在の探索方策の模倣 | 勝敗改善の保証・最適教師ではない |

**教師タイプが違うデータやチェックポイントを、意味が同一であるかのように置換しない。** `target_kind`、`turn_aware`、`candidate_encoding_version`、episode groupingなどのメタデータを保持する。

## 5. 直近の実装状況とsmoke結果

以下は**動作確認**であり、統計的評価ではない。値がよい/悪いといった考察や、ハイパーパラメータ探索はここから行わない。

- 31-D混合Q・Policyの学習/探索、およびSAFETYを含む混合遷移が動作する段階まで実装。
- 2026-11-02頃: 保存済みの異なる完了2ゲーム・計4実行遷移から、状態V学習が2epoch完走。VをMCTSで読み込み、3ターンの複数プレイヤー実行が完走（そのゲームは`max_turns`で打ち切り）。
- **直近実行（seed 94200～94201）:** `iterate_mixed_macro_policy --iterations 1 --games-per-iteration 2 --max-turns 3 --simulations 10 --policy-epochs 2 --update-critics --critic-epochs 2`
  - 収集2ゲーム、実行6ターン、完了1ゲーム、途中終了1ゲーム。
  - MCTSのroot状態6、候補94。Policy教師生成・2epoch学習・チェックポイント保存に成功。
  - 完了ゲームが`1 < 2`のため、**Q/V再学習をスキップし旧モデルを維持。正常なガード動作。**
  - 出力: `macro_policy_value_iteration_smoke/iteration_000/candidate_policy.pt`（**ユーザーのローカル作業環境内の成果物**であり、Gitに存在するとは限らない）。
- 今回のログは「実装が接続できる」ことのみを示す。勝率、Defense成功率、State V予測性能の主張には使わない。

## 6. 開発上の優先順位と段階的な進め方

### 当面（実装優先、長時間実験より構造整備）

1. **Policy/Q/V/MCTSの接続の安定化:** 最新ブランチには各構成要素と任意Q/V更新経路がある。再実装せず、関連テストと小規模smokeで不整合を除去する。
2. **リファクタリング:** scripts/testsの増大と重複を整理する。データ契約と評価の意味を守りながら、Macro構成要素・ファクトリ・CLI・学習パイプラインを適切に分離。大幅変更はテストとブランチを分けて行う。
3. **Defenseの拡張を段階的に:** まずルール/状態遷移・合法性を明確化し、次いで物理実行可能性の改善、Safety行動表現と学習済みMicroへの拡張を検討する。簡易物理グリッドの性能調整を本実験の代わりにしない。
4. **物理アクションの拡張:** 撞点（スピン）、キュー仰角を、FastFizとの対応、状態/候補エンコーディング、既存重みの互換性を考えて段階的に扱う。
5. **現実的なルール/評価へ発展:** 簡易2球ルールを維持して統合確認しつつ、後続フェーズで正式なゲーム条件と対戦基準を検討。

### 本実験に回すもの

- 数十時間の学習、高密度なパラメータ調整、汎化性能の統計評価、独立seedでの対戦勝率やDefense改善の検証。
- 各学習要素（Policy、Q、V、MCTS、Goal、Residual等）のアブレーション、モデルバイアス検証。
- 小規模smokeの偶然の数値に応じた局所最適化を続けない。

### 研究の進め方・発表資料について

- ユーザーは**実装優先**。短い/中規模の検証は許容するが、今の段階で長時間トレーニングを急がない。ローカルDGX/CUDA環境での実行は原則ユーザーが行い、実行コマンドを明示する。
- テストは**関連ファイルを指定したpytest**を基本とし、必要がなければ`tests/`全体の実行例を要求しない。
- ゼミ資料は前回との差分を重視。フローチャートは簡潔な白黒、変更点は赤字、10～20ページ程度を目安に、コード掲載は最小限。口頭説明は短め・要点中心。論文紹介と実装進捗を区別。
- 既存の実装ログを使い、次回資料で未検証の実行結果を「性能向上」と書かない。

## 7. 重要なファイル・ローカル成果物

### 主なソース

- `src/common/types.py` — `GameState`, `MacroAction`, `ShotAction`
- `src/macro/candidate.py`, `candidate_encoding.py` — MacroCandidateと25/31D表現
- `src/macro/mcts.py` — candidate-mode MCTSと手番符号
- `src/macro/mixed_search.py`, `transition.py`, `defense_transition.py` — 攻守候補・遷移
- `src/macro/candidate_policy.py`, `candidate_value.py`, `value.py` — 予測器・ローダー
- `src/macro/outcome_guided_safety_shots.py`, `opponent_threat.py` — Safety物理グリッドと相手脅威
- `src/controller/mixed_hierarchical_rollout.py`, `mixed_rollout_dataset.py` — ゲーム経験
- `src/game/simple_rules.py` — **暫定**ゲームルール

### 主なスクリプト

- `scripts/collect_mixed_rollouts.py`
- `scripts/build_mixed_rollout_visit_dataset.py`
- `scripts/train_macro_visit_policy.py`
- `scripts/build_mixed_rollout_return_dataset.py`
- `scripts/train_macro_candidate_value.py`
- `scripts/train_macro_state_value.py`
- `scripts/iterate_mixed_macro_policy.py`
- `scripts/smoke_mixed_hierarchical_rollout.py`
- `scripts/build_mixed_candidate_value_dataset.py`

### ユーザー環境のチェックポイント例（Git管理されている保証はない）

- Micro Base: `goal_conditioned_micro_mixed_curriculum_100k.pt`
- Position power-only Residual: `position_power_residual_mlp_2000.pt`
- 混合2-step Candidate Q: `macro_candidate_value_mixed_v2.pt`
- 混合Candidate Policy: `macro_candidate_policy_mixed_v2.pt`
- MCリターン由来のState V（smoke）: `macro_state_value_mixed_v2.pt`
- MCTS訪問分布由来のPolicy（smoke）: `macro_candidate_policy_visits_smoke.pt`
- 反復生成Policy: `macro_policy_iteration_smoke/iteration_000/candidate_policy.pt`

**AIはローカルの`.pt`/`.npz`がGitHubに存在すると仮定しないこと。** 必要ならユーザーの実行ログや手元のファイルで確認する。

## 8. 新しいチャット・作業者への引き継ぎ手順

1. まず本書、[README.md](../README.md)、[defense_v0_design.md](defense_v0_design.md)を読む。
2. 作業ブランチ/直近コミットと`git status`を確認し、**最新の関連ソースとテスト**で本書の記述を再確認する。別ブランチへ移行していればその実装を優先。
3. ユーザーの最新ログから「実装済み」「動作確認済み」「学習精度は未評価」を区別する。本文の過去のsmoke結果は固定の歴史的事実として扱う。
4. 次の作業を小さく定義し、コード変更 -> 関連テスト -> 小規模smoke -> コミット の順で進める。ローカル実行が必要ならコマンドを提供し、結果の貼り付けを受けて判断する。
5. 完了ゲーム数不足によるQ/Vスキップ、未完了ゲームのreturn教師からの除外、Safety候補のモデル依存性は**仕様**として扱う。
6. 既存の設計思想を大きく変えるとき（報酬、特徴量、Action、ルール、探索バックアップ、モデル互換性等）は、変更意図と移行・検証方法を明記してから実装。
7. この文書にないユーザー要求や別のチャットの決定事項については、過去ログまたはユーザーの指示に従い、無理に推測しない。

## 9. この文書の更新ポリシー

- **更新タイミング:** 新しい実装フェーズが一段落したとき、設計の前提が変わったとき、リファクタリングが完了したとき、ゼミ進捗を整理するときなど。毎回のテスト・毎コミットでは更新しない。
- **目安:** 研究開発が継続している間の数回の作業/週単位の節目。自動バックグラウンド更新の設定を意味せず、**次に研究の開発作業を行う際に適宜見直す**運用。
- **各更新で書くこと:** 更新日、作業ブランチ、直近完了事項、未解決課題、次の優先順位、データ契約や学習教師の意味の変化。古い数値を性能実証と書き換えない。
- **主たる整理場所:** 本書を要約・方針の正本にし、`docs/defense_v0_design.md`を詳細な段階別ログとして使う。変更が研究目的に及ぶときはroot `README.md`も更新。

## 10. 2026-11-09 リファクタリングと現行ブランチ

- 現行の開発作業ブランチ: `refactor-phase9-preserve-20261109`。分岐元 `phase9-defense-v0` の2026-11-09時点のHEAD `39cee14c` は無改変。ローカル研究ディレクトリは `~/billiards_research_backup_20261109.tar.gz`（約105MB）に退避済み。
- 重複したAttack/mixedコンポーネント構築を `scripts/attack_components.py`、`scripts/mixed_components.py` に分離し、Macro学習補助関数を `scripts/training_utils.py` に集約。モデル次元・報酬・学習教師・FastFizの仕様は変更していない。
- 旧 `build/` MuZero/SAC Python実装8ファイル（1333行）をGit管理ソースから削除。元ブランチとGit履歴から復元可能。現行のMuZero学習を実装したという意味ではない。
- MCTS・階層Agentの重複テストを統合。監査時点のテストファイル74→72、テスト関数300→294。ユーザー実行の最新関連33テストは**2026-11-09に全件PASS**。全スイートのPASSやモデル性能向上を意味しない。
- ローカル生成物157件/79.62MiBを監査。旧 `build/results/` 等の107件/29.12MiBは**現物・監査記録・tarバックアップのSHA-256全件一致後、2026-11-09に `~/billiards_refactor_quarantine_20261109/` に退避完了（削除なし）**。残る50件/約50.50MiB（主要モデル5、Git追跡3、他モデル22、データセット12、反復実験8）は保全。退避ファイルは `python tools/quarantine_legacy_build_artifacts.py --restore` で復元可。
- 退避後の再監査で **Artifacts: 50**、Git作業ツリークリーンを確認。主要5チェックポイントを読み込むCPU 1ターン実行smoke（seed=94000、MCTS simulations=1）は `POSITION_ATTACK` を選択して正常終了（score=[1,0]、`max_turns`停止、ゲーム非終端）。**機能接続の確認のみで、性能改善の根拠ではない。** 直近の関連33テストも全件成功。全スイート未実行、mainへのマージ未実施。
- 作業の詳細記録: [REFACTORING_PLAN.md](REFACTORING_PLAN.md)。リファクタリング後も現行科学的前提（25/31D、Macro 2-step QとMC return Q/Vの別、MCTS訪問教師、Defense物理候補）は変わらない。

## 11. 2026-11-09 フォルダ・実験管理の整備（Phase 1）

- `docs/PROJECT_STRUCTURE.md` に現状・目標ディレクトリとGit/実験管理ルールを記載。
- `configs/` のJSONは現状**実験設定の記録用**で、既存CLIは自動で読み込まない。
- `tools/prepare_run.py` は `artifacts/runs/<run_id>/` を新規作成し、設定のスナップショットとGitコミット・ブランチ等を記録する。既存runの上書きは禁止。**学習・評価は実行しない。**
- 旧成果物157件のうち退避済み107件・残存50件、Git追跡済みの `outputs/micro/` 3件は、Phase 1では**一切移動・削除しない**。
- Phase 2以降、`scripts/` 内の共通処理の `src/pipelines/` 化、`tests/` の分類、CMake生成物の管理を順に検討。既存 `python -m scripts.*` と旧import互換性を維持する。

## 12. 2026-11-09 フォルダとテスト構成（Phase 2/3）

- Phase 2: 共通学習補助・checkpointローダー・Attack/Mixedファクトリを `src/pipelines/` へ集約。旧 `scripts/` のimportを互換ラッパーで維持し、`python -m scripts.*` のCLIを変更しない。ユーザーDGXで関連22テストがPASSED、5モデル・CPU 1ターン混合階層smokeは移動前と完全一致（POSITION_ATTACK、ball0/pocket5、Macro reward +1.250、score [1,0]）。
- Phase 3: テスト全76ファイルを用途で `tests/unit/`（36）、`tests/integration/`（11）、`tests/pipelines/`（29）へ分類済み。**最終第8段階の40テストもPASSED（8.91秒）**で各段階の指定テスト成功。全スイート一括実行・大規模学習・性能評価をしたという意味ではない。
- 具体的なテスト指定は `tests/README.md`、全体の構成・ファイル保全ルールは `docs/PROJECT_STRUCTURE.md` を参照。
- **次の優先順:** Phase 3完了確認済み。Phase 4のGit追跡と物理ビルド依存について [PHASE4_ARTIFACT_AUDIT.md](PHASE4_ARTIFACT_AUDIT.md) に静的監査記録を追加した。DGXローカルの `build/billiard_env_cpp*.so`、追跡済み3モデル、バックアップ・退避先を**読み取り専用で確認**。既存157件監査のうち退避済み107件、残存50件は保全を続ける。

## 13. 2026-11-09 保守的リファクタリングの到達点（Phase 4完了）

- **Phase 2:** 学習補助・Checkpoint loader・Attack/Mixedファクトリの責務を `src/pipelines/` に分離し、従来の `scripts/` APIをラッパーで保持。関連pytestとCPU 1ターンの5モデルsmokeの一致を確認。
- **Phase 3:** 76のテストファイルを `unit` 36 / `integration` 11 / `pipelines` 29 に分類。各段階の対象テストが全件通過。**全スイート一括実行をしたという意味ではない**。
- **Phase 4:** ローカルCMake 4.3.0 / Python 3.11.15で隔離ビルドに成功し、ビルドしたFastFiz拡張をimportしてコンストラクタを呼び出せた。`build/Makefile` はGit追跡を解除した一方、DGXでは退避コピーから復元して正常動作を維持。`git check-ignore` とFastFiz targeted pytest **11件PASSED**、作業ツリークリーンを確認。
- 旧 `outputs/micro/` のGit追跡3モデルは手元とtarバックアップのSHA-256完全一致を確認し、**そのまま保持**。他のモデル・データ・バックアップ・隔離領域にも変更なし。生成物を追跡解除したことはデータ削除を意味しない。
- このブランチ `refactor-phase9-preserve-20261109` での**仕様を保った保守的リファクタリングは一区切り**。変更範囲・検証結果の詳細は [REFACTORING_PLAN.md](REFACTORING_PLAN.md) と [PHASE4_ARTIFACT_AUDIT.md](PHASE4_ARTIFACT_AUDIT.md) を参照。
- **次に検討すべき作業:** 検証済み作業ブランチのマージ方針を決めることと、Macro Policy/Q/V・探索・Defenseを含む研究実験の優先度を再整理すること。現時点でmainマージやゲーム勝率改善を実施・実証したとは記載しない。

## 14. 2026-11-10 Phase 5: MCTSルート同点の検出と選択方式の比較

- Phase 5で保存済みMixed Rolloutのreadonly集計を実装。DIRECT/POSITION/SAFETYの候補数・rootごとの有無・訪問数、選択された戦術、終了理由、ゲームスコア差分とMacro局所報酬、最大visitの同点数を分離して集計。
- ユーザーDGXで新しい分析関連テスト16件PASSED（3.49秒）し、MCTSの同点処理をデフォルト`first`を保ちながら任意の`empirical_q`へ切り替え可能にした。後者は**訪問数最大で同点かつ訪問済みの候補**を観測遷移からの`reward + gamma * turn_sign * child.value`で比較する。これに関するMacro指定31テストPASSED（4.61秒）。現行ルールと重みの仕様は不変更。
- 既存の反復smoke 2ゲーム（5ターン）ではDIRECT 5回選択。全5 rootで最大visit数1が10候補同点。候補行数はDIRECT48 / POSITION32 / SAFETY20、訪問数合計はDIRECT14 / POSITION32 / SAFETY4。これだけではCandidate PolicyやPOSITIONの強弱は確定できない。
- **単一seed=94100、初手1ターン、探索10回のpaired結果:** firstではDIRECT選択・Macro LOCAL reward -0.5・簡易スコア +1・的球1個・スクラッチ0。empirical_qではPOSITION選択・Macro LOCAL reward +1.25・同スコア +1・的球1個・スクラッチ0。新方式の初手局所報酬差は +1.75。両方式とも非完了（max_turns=1）。rootの候補数・訪問数分布は同一。これは**1盤面の挙動差**であり、ゲーム勝率や総合性能の改善を示す結果ではない。
- ユーザーは今後、計算時間・計算量を以前ほど制約しないと明言した。次は学習なしで**複数seed・複数ターン・探索予算10と50のpaired比較**を行う。seed対応を検証するCLI `scripts/compare_mixed_rollout_modes.py` を追加。実機評価はこれから。

## 15. 2026-11-10 MCTS root同点比較の24 seed×2予算×2方式（DGX実機ログ）

- **研究条件:** seed=95000〜95023 の24初期盤面を対応づけ、MCTS simulations=10/50、最終訪問同点処理`first`/`empirical_q`、最大8ターン、同じMicro base/Residual/Candidate Q/Candidate Policy、CPU実行。合計96個のRolloutが正常に保存され、各予算でpaired比較CLIが完走した。結果ファイルはローカル `artifacts/runs/20261110_mcts_tie_paired24_budget10_50/outputs/budget_{10,50}_paired.json`（Git管理外）。
- **10回探索:** 初手同一seedペアの候補変更23/24、正しい目標ポケットへの初手入球`first`=12/24、`empirical_q`=24/24。初手平均Macro局所報酬 0.6195→1.3621（paired差+0.7425）。ゲーム終端数16/24→22/24。プレイヤー0の平均累積簡易スコア0.8333→1.8750。スクラッチ計3→0。
- **50回探索:** 候補変更2/24だが、初手指定ポケット成功は両方式24/24。初手平均局所報酬は両方式1.3487、ゲーム終端数21/24、プレイヤー0の平均累積スコア1.8750、スクラッチ0。**初手・終了・スコアの指標では差が消えた**。
- **暫定解釈:** 小さい探索予算では、候補数と訪問数の関係で同点が頻発し、`empirical_q` が従来の候補登録順バイアスを回避できた可能性がある。50回では従来方式も十分な探索で良い個別候補を選択でき、改善効果が縮小したと推測される。**探索50回で同点数が実際に減ったかは未測定**。追加した `--root-summary` で検証する。
- **重要な限界:** 24初期seed・同じCheckpoint群・簡易2的球ルール・最大8ターン・訓練時に使用したデータとの独立性未確認。ゲーム終端は勝率ではなく、途中でスクラッチしても終端する。初手24/24は天井効果の可能性がある。**汎化性能の改善や統計的に独立な未見盤面での優位性を示したものではない**。
- 次は4条件別のroot同点率、異なるseed集とhard-state盤面、局所報酬成分と目標ポケット成功の区別を検証。計算量は従来ほど制約しないが、データ上書き・モデル形式/教師定義変更は避ける。

## 16. 2026-11-10 root最大訪問数の同点率（DGX実機ログ）

- 集計用pytest **23 passed in 2.88s**、作業ツリークリーン。
- **10 simulations:** first=59/114 root（51.8%）、empirical_q=46/61 root（75.4%）で最高visit数同点。そのうち最大visit数1の同点がそれぞれ58/114と46/61。選択された非先頭最高visit候補はfirst=0、empirical_q=41 root。
- **50 simulations:** first=4/66 root（6.1%）、empirical_q=4/66 root（6.1%）。最高visit数1の同点rootは両方0、非先頭最高visit候補の選択はfirst=0、empirical_q=2 root。
- **注意:** 10回のfirstとempirical_qでは実行後の盤面が異なり、root数も114と61で異なる。従って、51.8%対75.4%という全root同点率の差は、同じ盤面集団での同点確率差を表さない。
- **次の診断:** `scripts.compare_mixed_rollout_modes` に初手rootのseed対応候補行列・visit配列・最大値同点数を比較する機能を追加。これで初期盤面を揃えた直接的な同点状況と、探索予算を増やしたときの同点解消を分析する。コード追加後のDGXテストは未実施。


## 17. 2026-11-10 Phase 5をゼミ前に区切る

- **ユーザーとの合意:** DIRECT / POSITION / SAFETYの選択割合を均等化する必要はない。最善の戦術がDIRECTのみなら全てDIRECTを選んでもよい。戦術別選択数は診断指標であり、改善目標ではない。
- 同点処理は個別候補の訪問数を優先し、同点・訪問済みの候補間で観測Qを比較する。特定戦術を選ぶ回数を強制する実装はない。
- この契約を確認するため、`tests/unit/macro/test_mixed_candidate_mcts.py` に3戦術それぞれが観測Q最大で選ばれるパラメータ化テストと、DIRECTが候補末尾でも実際のMCTS探索から選ばれるテストを追加。**追加テストについてDGXでの実行は未確認**。
- ゼミ説明用に [SEMINAR_PHASE5_MCTS_20261110.md](SEMINAR_PHASE5_MCTS_20261110.md) を作成。既存96ゲームのseed対応比較、探索10/50回での同点減少、報酬設計のPOSITION加点の限界をまとめた。
- **ここでPhase 5の追加実装を一旦停止し、ゼミ準備に移る。** 次回はPOSITION報酬重みのアブレーション、難盤面での攻守評価、Macro Policy/Q/Vの更新の妥当性を検証する。


## 18. 2026-11-10 開発優先度を大幅変更：全体構成の完成と大規模本実験を最優先

- **ユーザーの明確な要求:** 最終的な大規模実験を可能な限り早く終える。性能が低ければその結果から必要箇所を修正する。小規模なアブレーション、報酬調整、局所的なMCTS実装問題は本発表に値する研究進捗とみなさず、今は追わない。
- **優先順位:** (1) Macro Policy/Q/V→MCTS→Goal-conditioned Micro→FastFiz→ゲーム進行と反復学習を、実際の大きめのrunで完走させる (2) 独立seedで学習前後と最終提案手法対ベースラインを評価 (3) 結果が悪ければ原因に応じて修正。
- 既存`iterate_mixed_macro_policy`はPolicy訪問教師と、必要なら完了ゲームのexecuted-return Q/Vを更新できる。しかし`collect_mixed_rollouts`のCLIにState V指定がなかったので`--state-value-checkpoint`を追加。新規`scripts/run_full_hierarchical_experiment.py`で学習・独立seed評価（学習前/後）を一括実行する。単体の接続テストを追加。コードはGitHub反映後、DGXでのテスト待ち。
- 既存の学習ロジックは完全なend-to-end同時学習ではなく、MacroのPolicy / Q / Vを段階的に学習し、Microは学習済み重みを固定する構成。Safetyは物理候補生成を用いる。**「提案手法全体が最終仕様まで完成した」と断定しない。**
- 現行の評価は手球1・的球2の簡易ルール下。ベースライン対照実験・一般化した盤面/ルール・データ分割・勝敗定義の確定は本実験計画に必要であり、根拠のない勝率性能主張は避ける。


## 19. 2026-11-10 大規模対戦実験最優先・共有GPU制約・競技盤面への拡張開始

- **目前の研究目標:** 「ビリヤードにおける階層型強化学習の実装、実験、考察」。全階層を統合して現行の物理基盤で大規模な学習と評価を行う。最終的には9-ballなどの競技ルールで対戦し、複数ラック先取のマッチを既存手法と比較する。ルールは後で選べる設計とし、小規模盤面はMicro事前学習・Curriculum向けに残す。
- **サーバー:** 共有GPU8枚、1枚あたり32000MiB。**原則1枚のみ使用し、2枚までなら必要性と空き状況を確認した上で同時使用可能。8枚を占有しない。** 共有サーバーなので使用前に `nvidia-smi` で他利用者の使用量を確認し、`CUDA_VISIBLE_DEVICES=<使用許可された空きGPU>` を明示。未許可の自動マルチGPU・長時間独占・8枚占有は行わない。GPUスクリプトを自動実行することはせず、ユーザーがDGXで実行・結果を返信する。
- **現時点の基本実験は完走:** ユーザー提出 `experiment_summary.json` より、Microは固定でMacro反復学習256ゲーム×4、独立seed評価学習前後各128ゲーム、MCTS各50探索を完走。学習前/後の簡易ゲーム完了数111/128→120/128、1ターン平均Macro局所報酬0.656→0.835、平均ターン数3.383→2.703。これは実際の対戦勝率ではない。反復ごとの完成ゲーム数238/230/242/250、Q/Vは4回とも更新。実験ディレクトリ全体と最終学習済みPolicy/Q/Vを保持する。
- **現行コードの主要な構造的制約:** GameStateは3球固定、FastFiz C++ラッパーはcue+2球固定、Micro SACは2-D (power,angle offset)、Macro state 12D・candidate 31D固定、ゲームルールはSimplifiedGameRules。GPU数を増やすだけでは対戦環境完成にならない。
- **新設（このコミット）:** `src/game/rack_state.py` 可変球数のTableGameConfig/RackState（cue+任意の番号付き的球、旧GameStateからの明示変換）、`src/game/match.py` RaceToMatch（ルール外部で決定した各ラック勝者を集計、先攻交代方式を指定可能）、`src/micro/shot_control.py` FastFizの5自由度に対応する物理ショット表現と正規化5-Dデコーダ。既存2的球の動作・チェックポイントを改変せず、新規単体テストを追加。これは**可変球数物理、9-ball rules、5D SAC学習実装の完成ではない**。
- **直近の次優先:** (1) C++/FastFiz多球set/restore/step、Python可変球数adapter、(2) rule engineとrack・matchの連結、(3) Macroの可変球数エンコーダ/候補・Micro 5Dを学習させる経路、(4) 既存手法との同条件対戦と総合実験。改変時は古い12D/31D・2D学習済みCheckpointと新形式の混同を禁じ、明示的にversioningする。


## 20. Phase 7: 可変球数FastFiz別クラス追加（DGX実機テスト待ち）

- FastFiz submodule `ekiefl/fastfiz@6c6df412` のヘッダを確認し、CUE～FIFTEENのBall ID、5自由度`ShotParams`、物理イベント`BALL_COLLISION` / `RAIL_COLLISION` / `POCKETED` 等を確認。
- `cpp_simulator.cpp` で旧3球 `BilliardSimulator` を残し、`BilliardSimulatorMulti` を新設。Cue+任意の的球番号、位置・ポケット状態の復元、ショット実行、first-contactの実球番号、イベント時刻/種類/関与球を提供する構成。
- `src/env/multi_ball_fastfiz.py` は`RackState`と`PhysicalShot5D`を使うPython adapter。ルール処理・ラック勝敗・学習はまだ接続していない。
- `BILLIARDS_FASTFIZ_BUILD_DIR` をオプトイン指定した場合だけ隔離CMakeビルドを読み込む。旧環境は既定`build/`のまま。
- **未確認:** C++実ビルド・9球物理shot・旧FastFiz回帰。関連テスト実行前に多球物理が完成したとは主張しない。実行コマンドは`tests/README.md`。ビルドはCPU2並列程度、GPUは不要。

## 21. 2026-11-10 競技環境実装と自己対戦を最優先（既存手法比較は後回し）

- **優先順位:** (1) 多球FastFizとルールに沿ったラック進行・RaceToMatchの接続、(2) Macro/Microを多球・5自由度で動かす学習経路、(3) 現行モデル同士の自己対戦からゲーム経験を生成して方策を更新、(4) 更新モデル対**固定した過去checkpoint**の先攻交代対戦により学習の変化を評価、(5) 既存研究手法との比較はその後に実施する。
- 自己対戦での同一Policy対同一Policyの勝率は対称条件でほぼ50%になり、実力向上を示す指標にはならない。研究の初期段階では完了ラック率、ファウル、ラック数、先攻/後攻、モデルID、探索予算、勝敗（相手checkpoint付き）を保存し、評価用seed・対戦モデルを固定して学習と分離する。
- 現行MacroのMixed Rolloutは両プレイヤーが同一方策を使う自己対戦に相当するが、**公式ルール・複数ラック・学習を伴う競技対戦パイプラインは未実装**。RaceToMatchはルール側からrack winnerを受け取るだけで自ら試合を進行しない。
- **ユーザーDGXでの新規純Pythonユニットテスト:** 28件中27 PASSED、`tests/unit/game/test_match.py::test_alternate_break_and_race_to_three` だけFAILED。3勝先取に4ラックで到達しているのに、テストが5ラック目の終了処理を呼んだことが原因。対戦管理側の「終了後のラック入力拒否」は正しい。テストを4ラック終了に修正した。**修正後のDGX実行は未確認**。
- GPUは共有の8×32000MiBで、通常1GPU・必要な場合のみ空き確認後2GPUまでを遵守。自己対戦の物理シミュレーションはCPU主体で、学習用GPUは最小限の枚数を指定する。

## 22. 2026-11-10 Phase 7: 9-ballの最小競技ルール・複数ラック進行を実装（DGX未検証）

- 多球FastFizの隔離CMakeビルド成功、`tests/integration/physics/test_multi_ball_fastfiz.py` 3ケース＋既存FastFiz 11ケース、**計14件すべてPASSED（0.20秒）**。大量のFastFiz/CMake警告は非致命で、Gitはクリーンとユーザーログで確認。
- `src/game/nine_ball_rules.py` に通常ショットの最小番号球への最初の接触、入球/接触後レール、スクラッチ、ファウル後手球自由配置、9番球の合法入球によるラック勝利、反則時の9番球再配置、3連続ファウル、無入球ブレイクの4的球レール規定を実装。`src/game/nine_ball_session.py` で菱形の9球ラック生成・物理step→ルール適用→レイアウト同期→複数ラック先取の`RaceToMatch`まで接続。
- **厳守:** 正式WPA準拠は未完成。プッシュアウト後の選択権（明示的にNotImplemented）、ブレイク時の3球ヘッドストリング越えの規定、飛び出した球、試合運営上の反則等は未実装。公式勝率・本格自己対戦として扱わない。3ファウル警告は自動とする暫定仮定。
- 模擬物理を使った最小ルール/ラック/試合の単体検証は別環境で16ケースを通過したが、**DGX上のpytestはまだ実施していない**。本実装には学習済み多球Macro/Microはまだ接続していない。
- 研究の直近目的は9-ball相当のルールで自己対戦→学習→新旧モデル対戦。既存手法との比較は後回し。GPUは通常1枚/必要時に最大2枚を維持。

## 23. 2026-11-10 Phase 7: 実FastFiz→9-ballルール結合テスト（DGX未実行）

- `tests/integration/game/test_nine_ball_native_session.py`を追加。9球菱形ラック、FastFizネイティブ物理ショット、first-contact/rail/pocketイベント、`NineBallRules`適用後の状態同期までを1ショット実行する。
- このテストは物理結果の良さ・合法ブレイク率ではなく、実行経路の結合を検証する。GPU不要で、隔離ビルド`BILLIARDS_FASTFIZ_BUILD_DIR`を使用する。
- ローカルの模擬物理テスト16件はPASS。DGXでの新規ルールと結合テストの実行はまだ確認していない。

## 24. 2026-11-10 自律9-ball自己対戦データ収集（非学習ベースライン）

- DGXでのテスト結果：新規9-ballルールと盤面/ショットのユニットテスト**44件PASSED（0.27秒）**、C++隔離ビルドを使うNative物理・ルール統合テスト**4件PASSED（0.25秒）**。計48件PASS、gitクリーン。
- 新規 `src/game/nine_ball_agents.py` に、幾何から最低番号球を狙う**学習なしの仮エージェント**を追加。打球はFastFizの5自由度で出力するが、上下左右撞点はデフォルト0、仰角10度（5D SACではない）。Breakは最低番号球への直線、通常は障害物を考慮したGhost ballへの簡易DIRECT、不可なら最低番号球を狙うfallback。
- `scripts/collect_nine_ball_self_play.py`で物理FastFiz→NineBallRules→rack/matchをショット制限まで自律反復。ショットの前後状態・選択した行動・物理イベント・反則理由・ラック勝者を`match_seed_*.json`へ記録。`summary.json`には完了/未完了件数を分離し、**未完了を勝利としてカウントしない**。既存出力先には書き込まない。
- 将来の学習済みMacro/Microは`choose_shot()` / `cue_placement_candidates()`インターフェースで差し替える設計。今回はまだMacro/Micro学習統合ではなく、**非学習・完全WPA準拠ではない環境の自己対戦収集基盤**。
- `NineBallRackSession.last_physical`でrule適用後のNative `restore()`によって消えるイベント詳細をログ用に保持。
- 関連ユニット7テスト＋実物理3ショット自律試行の統合1テストを追加。**DGX実機は未実行**。GPU不要。CPUで動く小規模データ収集から確認する。

## 25. 2026-11-10 自己対戦経験ログの保全修正

- 複数ラックの`NineBallMatchSession.step()`はラック勝利後すぐ次の`NineBallRackSession`に置き換えるため、最終ショットの`last_physical`が失われる可能性があった。`MatchSession.last_physical`に毎ショットの物理結果を保存し、JSON収集器はこちらを読むよう修正。ユニット回帰テストを追加。
- `--matches`で連続試合を収集する際にラックseedが重複しないよう、各試合のseedを`start_seed + index * (2*race_to)`に変更。Race-to-Nは最大`2N-1`ラックなので非重複。
- 実Nativeの自律ショット統合テストに`precondition_code`のイベント根拠を追加。ユーザーDGXでの実行は未確認。

## 26. 自己対戦データのルール文脈を記録

- Macroの学習時に現在の盤面だけでは合法手・3ファウル等の判断が不足するため、収集JSONの各ターンに`rule_context`（反則連続回数/プッシュアウト可能フラグ）と`next_rule_context`も出力する。モデル側のObservationとしての使用は**まだ未実装**。

## 27. 2026-11-10 Phase 8: 多球エージェントのv3入力・候補の契約

- 9ball対戦収集（非学習幾何Agent）のユーザーログ: unit/integration計10 PASSED (0.33秒)。seed310000が2-0で39ショット/9ファウルで終端、seed310004は1-0で51ショット/8ファウルの打切り。集計は**1/2マッチ完了、計3ラック、90ショット、17ファウル**。この結果はRL性能ではない。既存JSONは保管。
- 次の主要課題は多球用Macro/Microの**入力次元・球ID**契約。旧12D/31D Macro、9D観測+2D MicroのCheckpointを新10球で読むことはできないため、`src/game/nine_ball_encoding_v3.py` にv3専用 `encode_rack_v3`（実球番号0..15の固定16枠+ルール文脈=72D）、`encode_goal_v3`（BREAK/DIRECT/POSITION/SAFETY、球ID、ポケット、停止目標、明示shot=36D）、Micro結合108Dの仕様を新設。球数が違う練習盤面も同次元のpad表現で受け取れる。
- `generate_nine_ball_goals` は、9ballの最小番号球をfirst contactの前提とし、break/直接ポケット/次球位置/簡易safety候補を列挙する**学習・MCTS未接続**の入口。選択戦術の出現率は強制しない。POSITION目標やSAFETYの現内容は仮の幾何値であり最適性を示さない。
- GPUは共有8×32000MiB、原則1GPU・空き確認後最大2枚。データからの学習を次に接続する。

## 28. Phase 8: 9-ball実物理shotによる5自由度Micro SAC学習経路（DGX未実行）

- `scripts/train_nine_ball_micro_sac.py` 追加。保存済み幾何Agentの`match_seed_*.json`から、ショット前の盤面・foul contextを復元し、direct/positionのTacticalGoalを生成する。FastFizで毎回5D行動を実行、ルールでファウルや正しい目標ポケットを判定して一ショットのGoal条件付き報酬で既存`SACAgent(state_dim=108,action_dim=5)`を訓練する。旧Microの事前学習済み2Dは読み込まない/上書きしない。新checkpoint`micro_sac_5d.pt`にv3/108D/5Dタグを保存。
- これはMacroを更新しない独立した**Micro事前学習の入口**で、目標候補の抽出分布はMacro戦術選択比率の制約と無関係。今後同一v3仕様をMacro Policy/Q/Vに接続し、交互学習・MCTSでMacroとの相互依存を扱う。
- `--device cuda` の場合は`CUDA_VISIBLE_DEVICES`でGPU1枚に限定する仕様。1GPUしか使わない。既存サーバーが共有8×32000MiBであることを再確認。空きGPUのIDは必ずDGXでユーザーが確認して指定する。
- 学習成果の勝率・試合品質は**未検証**。現時点では収集した90ショットの盤面からresetして1-shot SACを学ぶため、未見初期盤面や多球相互作用の一般化は証明できない。補助データ取得だけでは最終的な対戦学習の完了ではない。

## 29. Phase 8: 学習済み5D Micro推論の9-ball自律対戦への接続

- `src/game/nine_ball_learned_agent.py`を追加。明示的にタグ付きのv3/108D状態+5D動作のMicro SAC checkpointのみロードし、候補目標条件で5自由度のショットを生成する。break/safetyは現在明示的な物理ショットのためMicro推論を経由しない。
- 任意のv3/72D状態+36D候補Macro Policy checkpointを渡した場合にのみ、そのPolicyのlogitで候補選択。未指定時は**幾何的な暫定Macro**を使用。まだMacro Policyの9-ball学習/MCTSは未実装であり、学習済みMicroだけでフル階層RLと名乗らない。
- `scripts/collect_nine_ball_self_play.py`に`--agent-kind micro_sac --micro-checkpoint <v3 checkpoint>`、`--macro-policy-checkpoint`（任意）、`--device`を追加。geometry既定値はそのまま。対戦時の`rule_context`をAgentへ渡す。GPUで推論する場合は明示`CUDA_VISIBLE_DEVICES`を1枚に制限する。ユニットテスト・DGXネイティブ実行は未確認。
- **次の重要な実装**は、9-ballのMacro Policy/Q/Vを実際の自己対戦リターンやMCTS教師で更新する流れと、9-ball物理遷移を使うMCTS実装。今のMacro教師は幾何行動のログと一致せず、旧Macro学習をコピーして「完成」としてはいけない。

## 30. 2026-11-10 5D Micro学習開始時のFastFiz撞点浮動小数誤差

- ユーザーDGX実行: `--steps 5000 --device cuda`、checkpointのモデル入力は108D/5D、学習盤面86件まで読み込み成功。学習中の最初の物理ショットで`ValueError: Invalid shot parameters`。
- 原因: `Shot5DDecoder`は半径12.0mmの円境界まで撞点を許し、PythonからC++のfloat32へ変換した際に`a*a+b*b`が144.0fを僅かに超え得る。C++ `BilliardSimulatorMulti::step`は厳密な`>144.0f`を拒否。保存前に失敗したためモデルは未出力。乱数warmup 5000候補の数値検証では81件がこの境界を超える例を確認。
- 修正: `Shot5DDecoder.decode`の有効な最大撞点半径を`min(tip_radius_mm, 11.99)`に制限する。FastFizの厳密な12mm判定は変更しない。CPUでfloat32変換境界の回帰テスト、実native版の回帰テストを追加。隔離CMakeバイナリの再ビルドは不要。
- **DGXでの修正後の学習・テストは未確認。** まず関連テスト→5000stepを再実行。GPUは既存`CUDA_VISIBLE_DEVICES`の1枚指定を継承。

## 31. 2026-11-10 Phase 8: DGXで確認した5D Micro初期学習と自己対戦

- ユーザーがv3/108D→5D Micro SACを5,000 steps学習し、`artifacts/runs/phase8_micro5d_first/outputs/micro_sac_5d.pt`を保存。学習中500-step窓の最終平均報酬 -0.3401、正しい目標ポケットへの入球率 0.040。これは学習時・直近500 shotsの値で、未知盤面の性能や対戦勝率ではない。
- 関連する学習・自己対戦テスト7件がDGXでPASSED（3.02秒）。
- 学習済みMicroと幾何Macroで9-ball自己対戦3試合が全完走（seed 420000,420002,420004）。計3ラック、56ショット、18ファウル。保存先 `artifacts/runs/phase8_micro5d_selfplay_smoke_v1/outputs/`。これは接続・完走確認であり戦術性能の証拠ではない。

## 32. 2026-11-10 Phase 11: 9-ball用Macro Policy v3教師・学習経路（DGX未検証）

- 新作業ブランチ `phase11-nineball-macro-v3-bootstrap` を `phase10-rollout-analysis-20261110` から分岐。既存の旧Macro v2重みと現行v3 Microを上書きしない。
- `NineBallMicroSACAgent.plan_goal` で**任意の指定候補Goal**を固定した5D Microから物理ショットへ変換可能にした。デフォルトの`choose_shot`は維持。
- `scripts/build_nine_ball_macro_oracle.py`: 実観測された非ブレイク前状態・rule contextだけを読み、全候補を同じ盤面からFastFizで一手実行して局所教師を作る。選択結果や保存next_stateは教師に使わない。SAFETYの相手脅威評価は未導入、暫定的に合法なSAFETYを0点とする。
- `scripts/train_nine_ball_macro_policy_v3.py`: 72D状態+36D Goal候補のネットワークを、教師スコアsoftmaxへのlistwise CEで訓練。match seedごとにtrain/validation分離。出力チェックポイントをv3専用タグで保存し、既存の9-ball学習済みエージェントのMacro読み込み口につなぐ。
- **これは一手物理教師からの教師ありPolicy bootstrapであり、RLのMacro Q/VでもMCTSでもない。** 後続はturn-awareなQ/V教師・物理MCTS・反復学習を整えること。
- 実装の仕様・DGXでの関連テストとsmokeコマンド: `docs/PHASE11_NINE_BALL_MACRO_BOOTSTRAP.md`。GitHubへソースはコミット済みだが、DGXの関連テスト・教師データ生成・学習は**未実行／未確認**。これらのログを確認してから実装済み機能として評価する。

## 33. 2026-11-10 Phase 11: DGXでMacro Policy初期学習・対戦まで確認

- 物理教師の生成: 既存学習済みMicro 5Dを固定して6盤面・49候補を同盤面から実行、候補反則率 26.53%、正しいポケット入球率 14.29%。これは**全候補を1ショット評価した平均**でありゲームプレイの成功率ではない。
- 4学習盤面・2検証盤面（検証seed 420004）でMacro Policyを3epoch訓練・保存。held-out teacher regret 1.0、教師と同点最良候補の選択率0.5。盤面2件のため性能主張には不十分。
- 旧Microと新Macroを接続した9-ball自己対戦3試合（seed 430000/430002/430004）は**3件完走、計40ショット、15反則、全勝者がplayer1**。動作統合確認のみ。勝率改善・対戦の強さは未検証。勝者の偏り/反則多発は後続評価課題。

## 34. 2026-11-10 Phase 12: 9-ball Macro Q/Vの選択行動MC学習（DGX未実行）

- 新ブランチ: `phase12-nineball-turn-aware-qv`（Phase11から分岐）。v3選択Macro Goalの36D特徴量をショットログに厳密に保存するため、`ShotPlan.macro_goal`と`chosen_action.goal_v3`を追加。旧Geometry Agentはnull。過去の保存済みphase11自己対戦JSONにはこの項目がないので、**Q教師へ曖昧な再推定はせず新規ログを収集する**。
- `scripts/build_nine_ball_macro_returns_v3.py`: race-to-1の終了試合だけを使い、`winner`視点の終端±1から、手番変化の符号を反転しながら割引リターンを作成。未完了試合は教師に使わない。各実行済み行動だけをラベル付け。複数ラックのrace-to-Nは現行72D表現にマッチのラック勝数が含まれないため除外する。
- `scripts/train_nine_ball_macro_qv_v3.py`: 選択行動のみのQ(s,c)、実行方策の状態価値V(s)を教師ありMSEで学習。match seed単位でtrain/validation分離。v3専用の厳格なチェックポイントメタデータとローダーを追加。
- **重要:** Qは未選択候補のQを正しく評価する保証がない。Vも現行behavior policyの平均的結果を近似するだけ。MCTSは未接続。本実験の有効性、勝率改善は未検証。
- 詳細とDGX実行手順: `docs/PHASE12_NINE_BALL_MACRO_QV_V3.md`。GitHubへのコードコミット済みだが、pytestとDGX実機実行は**未確認**。

## 35. 2026-11-10 Phase 12: DGXでMacro Q/V初期学習成功

- Phase12ブランチで関連テスト24件PASSED（5.26秒）。
- 学習済みMacro+Microでrace-to-1 6試合全完走。計134ショット、55反則。勝者player1が5件、player0が1件。反則率や勝者偏りの原因・再現性は未評価。
- 選択Macro Goalを厳密に保存した134ショットの完成試合MC教師を作成（train 90ショット/4試合、val 44ショット/2試合）。
- 5epochのv3 Candidate Q 検証MSE=0.779848、v3 State V 検証MSE=0.765339。学習集合平均による定数予測baseline=0.793873。少数試合なので性能主張は不可。Checkpointは `artifacts/runs/phase12_nineball_qv_smoke_v1/outputs/` に保存。
- 繰り返すが、このQは**実行済みMacro候補のMCリターンのみ**で訓練したため、未実行候補への一般化は保証されない。

## 36. 2026-11-10 Phase 13: 9-ball用物理PUCT・手番価値バックアップ（DGX未検証）

- 新作業ブランチ `phase13-nineball-mcts-v3`（Phase12から分岐）。
- 9-ball v3 `NineBallGoal`をMacro候補、Candidate Policy v3をPUCT prior、固定5D Micro+FastFiz+NineBallRulesを候補物理遷移、任意のState V v3を葉評価として使用する探索を追加。
- 価値は現在の行動プレイヤー視点。試合終端で+1/-1、非終端は0の即時報酬とし、手番変化で継続価値の符号を反転。未実行候補への信頼性不足から、既存のselected-action Qは探索に**使用しない**。
- 探索中に反則でball-in-handとなった場合、既存のNineBallRackSessionと幾何配置候補で物理的に合法な手球配置を再現する。手球配置の戦術最適化は未対応。
- 自己対戦CLIに `--agent-kind micro_mcts` と探索回数・深さ・State Vを渡す引数を追加。選択Goalとroot訪問分布等をJSON保存して将来のPolicy訪問教師として利用できる形にした。ただし教師生成・Policy反復学習はまだ未実装。
- `docs/PHASE13_NINE_BALL_MCTS_V3.md` に設計とDGXテスト/1試合smokeの実行手順。**コードはGitHubへコミット済みだが、DGX関連pytestと実物理MCTSはまだ未検証。** 中間seedの不足、V未校正、反則勝敗の偏りを改善したと主張しない。

## 37. 2026-11-10 Phase 13: DGXで初の9-ball物理MCTS実行確認

- DGXで指定関連テスト**24件PASSED（4.42秒）**。
- 学習済みMacro Candidate Policy v3と固定Micro 5D、State V v3を用いたPUCT探索（8 simulations、最大深さ2）で、9-ball自己対戦**1試合完走（seed 450000、5ショット、1反則、勝者player0）**。MCTS→物理→ルールまでの接続確認であり、改善の証拠ではない。
- 実行履歴とroot訪問分布は `artifacts/runs/phase13_nineball_mcts_smoke_v1/outputs/` に保存。1試合なので訪問Policyの試合単位train/validation分割には単独で不足。

## 38. 2026-11-10 Phase 14: MCTS訪問分布によるMacro Policy更新（DGX未実行）

- 新作業ブランチ `phase14-nineball-mcts-visits-policy`（Phase13から分岐）。
- `scripts/build_nine_ball_mcts_visit_dataset_v3.py` は保存済み対戦rootの候補v3特徴、訪問数、事前確率、選択候補の厳密一致、simulations数との整合性を検証して、盤面ごとのPolicy教師データを生成。未完了試合もroot訪問教師として利用可能（終端MCリターンとは異なる）。単一候補rootは既定で除外。候補戦術の出現割合は強制しない。
- `scripts/train_nine_ball_macro_visits_v3.py`: v3 Candidate Policyの既存重みを任意の初期値として読み、MCTS訪問分布へのlistwise CEで更新。検証はmatch seed単位。初期checkpointと比較し、検証CEが悪化した場合は更新前の重みを保持。保存形式は従来のv3 Policy互換。
- **MCTS方策の教師あり蒸留**であり、対戦勝率改善、Q/Vの更新、反復学習ループ完成を意味しない。微小な学習・検証データでのCE改善を本実験の結論としない。
- 実装/検証コマンドは `docs/PHASE14_NINE_BALL_VISIT_POLICY.md`。GitHubへのコミットは完了、DGXのpytest・ネイティブ自己対戦・訪問データ生成・学習実行は**未確認**。

## 39. 2026-11-10 Phase 14完走・ゼミ発表準備へ移行（ここで実装を一時停止）

- ユーザーDGX実行: 関連pytest **28件すべてPASS（5.40秒）**。
- 9-ball学習済みMacro Policy + MCTS 16 simulations、深さ2、固定5D Micro、State Vによる自己対戦が**3/3試合完走**。seed 460000/460002/460004、計30ショット、7反則、今回の3試合の勝者は全てplayer0。勝率改善・先後有利の推定には使えない。
- MCTS visitの教師データを生成: 30 root（うち単一候補3を除外）、有効27 root・185候補。train 2試合15 root、validation 1試合12 root。
- Macro Policy v3を5 epoch更新、検証visit CE **1.6741823653 → 1.6720843017**、KL **0.1091053 → 0.1070073**、最大visit候補一致率 **0.8333333のまま**。保存先 `artifacts/runs/phase14_nineball_macro_visit_policy_smoke_v1/outputs/macro_policy_visits_v3.pt`。
- **正確な結論:** 9-ball Macro → MCTS → Goal-conditioned 5D Micro → FastFiz → ルール → 訪問データ → Macro Policy更新の一巡が**個別スクリプトを順番に実行する形で成立**。多回反復の自動ループ、性能向上を検証する対戦比較、大規模データによる学習は未実施。少数データのCE僅差改善はゲーム強さの改善ではない。
- **ユーザー指示により、Phase14を研究実装の区切りとして追加実験・新フェーズ実装を止め、ゼミ資料制作へ移行**。大規模GPU学習もゼミ後に再検討。
- ゼミ資料の構成・数値・限界は `docs/SEMINAR_PHASE8_14_9BALL_HANDOFF_20261110.md` を参照。白黒中心、差分は赤字、コード掲載なし、数式と文章中心。過去環境の数値と新9-ballの数値を混同しない。

## 38. Phase15提案手法整合性監査（2026-11-11）

- Phase14のMacro Policyは物理教師bootstrap→MCTS訪問分布蒸留、Q/Vは完了したrace-to-1の選択行動MC教師。MCTSは学習済みPolicy priors+State V葉評価+固定5D Microを使い、選択行動Qは使用しない。これは探索と統計推定上の整合性から妥当であり、無理にQを未実行候補へ適用しない。
- ただし、ユーザー提案手法として重要なMacro/Micro相互学習は未実装だった。Phase15では`train_nine_ball_micro_sac --init-checkpoint`で旧5D SACを上書きせず継続学習し、`--goal-source logged`で自己対戦でMacroが実際に選んだDIRECT/POSITIONの**一致する**GoalだけをMicro学習に使用可能とした。古いゴールの復元を勝手に推定せず、違う候補生成なら明示エラーとする。
- 重要な報酬設計誤りを修正: POSITION_ATTACKの手球位置ボーナスが目標球の指定ポケット成功なしでも加算されていた。今後は**目標球の指定ポケット成功時のみ**加算し、Goal-conditioned「ポケット成功+配置」の契約に合わせる。
- Microの継続学習時はActor/Critic/Target Critic/エントロピーを読み、optimizer/replayはリセットする設計（メタデータで明示）。完全な同一ラン再開ではなく、新しい収集分布へのfine-tune。新旧Microチェックポイントは厳格なv3/108D/5Dのみ。GPUはユーザー選択済みの可視1枚を利用。
- 未達成: Macro Policy/MCTS+Micro更新を完全な一つの自動反復、MicroのSAFETY/Break条件付き学習、弱いPosition/SAFETY候補改善、独立seed・モデル別対戦、本格的な反復学習実験。Phase15のtest実機確認後に高優先で接続する。**これを階層型RL性能改善とはまだ主張しない。**

## 39. Phase15 Macro/Micro反復実験入口（2026-11-11）

- `scripts/iterate_nine_ball_hierarchical_v3.py` を追加。Phase14のroot visits→Macro Policy蒸留、Phase12の完了試合MC→Q/V更新、Phase15のMacro選択Goal→5D Micro fine-tuneを同一ループに接続。次iterationは更新したMacro Policy, State V, Microのモデルを使用。
- 個別stageのCLIを子プロセスで実行し、各出力を`iteration_###`に分離。各段階のログ・状態を`run_manifest.json`に逐次記録。前roundのモデルや既存結果の上書きを拒否し、失敗時は成果物保全。反復間seed分離。複数ラックrace-to-Nは未対応、race-to-1のみ。
- これは**一式がDGXで実際に完走する前の実装段階**。実機テストと物理自己対戦の再実行が必要。最終自己対戦後の実力比較は未実施で、Phase15で動作保証や性能改善を主張しない。

## 40. 2026-11-11 Phase15 FastFiz Bad Shot Exceptionへの修正

- 2反復・各32試合の`000_self_play`が20試合保存後、Native
  `Bad Shot Exception`で停止。ログの`Cue overlaps Four`は約1.7e-8m、
  `Two overlaps Five`は約0.0294mの重なり。後者は重大な不正配置。
  実際の異常発生が直前のショットからか探索仮想状態にあるかは、失敗局面の
  直接リプレイなしに断定しない。
- `src/game/nine_ball_physics_guard.py`導入。Nativeでショット未実行
  （precondition/placement失敗）、FastFizの`Bad Shot Exception`、
  または大きい球重なりを、**現実の反則や相手勝利と偽装せず**検知する。
  MCTSではそのedgeをinvalidとして訪問せず別候補を試し、予定の実行可能
  探索回数と失敗試行数を区別してroot診断へ保存。全候補が失敗したら明示停止。
- Live環境は不正物理遷移をGameRulesへ流さず例外。Collectorは当該試合を
  `invalid_physics`・未完了・勝者なしで保存し、次seedを続行。以前に完了した
  試合の勝敗や物理状態には触れない。
- 中断した`phase15_hierarchical_iteration_v1`は保全。Collectorの
  `--resume`と反復ランナーの`--resume`で既存試合JSONの本人確認と
  欠落seedだけの実行をサポート。**元runはGitリビジョンを記録していなかったため**
  `--accept-unverified-code-change`が必要。異なるコード改修前後の試合を
  同じ学習データにするのは厳密な比較実験では不適切であり、再開はdebug限定。
  本格的な実験は`phase15_hierarchical_iteration_v2`など新runに収集し直す。
- テスト・DGXの新規実Native実行は未確認。修正成功と学習性能向上は未確定。

## 41. 2026-11-11 Phase15単独リプレイ成功とチャット移行

- ユーザーDGXで`seed=510040`を修正後Phase15の`micro_mcts`+Policy v3+State V v3+5D Microで独立1試合再実行。**finished=True / winner=0 / rack_wins=[1,0] / 13 shots / 4 fouls**、1/1完走。保存先`artifacts/runs/phase15_badshot_replay_seed510040_v1/outputs`。これは例外がその1試合で再発しなかったというスモーク結果であり、根本原因の完全修正や性能改善ではない。
- 元の`phase15_hierarchical_iteration_v1`は初回32試合のうち20件保存して止まったまま。データを混合しないため旧runは保全し、修正後の関連pytest結果を確認した上で**新run`phase15_hierarchical_iteration_v2`**で最初から収集する。旧runでの`--resume --accept-unverified-code-change`はデバッグ目的のみ。
- 現在の反復学習経路は自己対戦→MCTS visits Policy→選択行動MC Q/V→Macro選択Goalで5D Micro継続学習→次反復。完全自動2反復はまだ未完走。
- チャット移行用の最新正本`docs/CHAT_HANDOFF_PHASE15_20261111.md`を新設。新チャットでは必ず最新HEAD・このファイル・現在のpytest/実験ログを照合。研究の優先は**9-ballの提案手法一通り完成、大規模自己対戦、独立評価、考察**であり、微細な最適化と既存研究比較は後回し。
- ユーザーはGitHubのツール承認を「すべて許可」に設定。Private状態で一部書込手段に403が出たことがあるが、別経路`create_file`/`update_file`でのコミットは成功。ユーザーが一旦Publicへ戻し、2026-11-11時点のGitHub連携で`visibility=public`・`push=true`を確認した。アクセス状態は新チャットで再確認すること。

## 42. 2026-11-13 Phase15の2反復完走と独立対戦

- DGXの `d3fd35a` でFastFizネイティブ統合を含む関連pytestが46件通過（4.12秒）。新規 `artifacts/runs/phase15_hierarchical_iteration_v2` は全12ステージを完了し、`run_manifest.json` の `status` は `completed`。2反復目の自己対戦は1反復目のMacro Policy、State V、Micro SACの出力を読み込んだ。
- 1反復目はseed 510000〜510062（偶数32試合）、2反復目は511064〜511126（偶数32試合）。両方32/32試合完了。ショット数は694 / 627、反則数は192 / 138。全ログroot数は694 / 627、単一候補を除いたPolicy教師は662 / 595。探索内の物理棄却件数は各試合JSONのroot診断で確認する。
- Macro Policyは1反復目に初期重みを採用（best epoch 0）。2反復目の検証訪問CEは1.92721 → 1.92506（best epoch 4）、最大訪問候補との一致率は76.1% → 67.5%。State Vの検証MSEは1反復目0.75347（定数基準0.78380）、2反復目0.78695（定数基準0.77789）。検証集合は反復ごとに異なる。
- Micro SACは各反復で有効5,000ステップを収集し、物理棄却は244 / 242試行。各反復末尾500有効ステップの指定球・指定ポケット成功率はともに13.2%。これらは学習中の値。対戦性能は以下の独立seedで評価する。

`scripts/eval_nine_ball_hierarchical_v3.py` は学習manifestの起点モデルと最終反復モデルで、各seedの同一初期ラックをrace-to-1で2回対戦する。ペア内でブレイク担当を交代し、ペア間でプレイヤー番号を交代する。学習時の探索設定を両モデルに適用し、完了試合内勝率・先攻別の内訳・ペア別の結果・未完了理由を `summary.json` に保存する。学習seedとの重複と既存評価出力の再利用は拒否する。

DGXの `bill_sub1` で、評価コードのコミットを取得してから実行する。

```bash
cd ~/research
export BILLIARDS_FASTFIZ_BUILD_DIR=/tmp/billiards-fastfiz-phase7
python -m scripts.eval_nine_ball_hierarchical_v3 \
  --training-manifest artifacts/runs/phase15_hierarchical_iteration_v2/run_manifest.json \
  --output-dir artifacts/runs/phase15_hierarchical_head_to_head_v1 \
  --pairs 32 --start-seed 710000 --device cuda
```

## 43. 2026-11-13 実装優先の確認と9-ball SAFETY Micro

- Phase15の先攻交代対戦は32ペア64試合を処理し、60試合完了、4試合は物理エラーで未完了。評価経路の動作確認として記録した。小規模の学習・対戦は、提案手法の接続と入出力を検証するために使う。優先課題は未完の機構をつなぎ、その後に大規模学習・評価と考察へ進むこと。
- `--learned-safety` を付けた9-ball反復ランナーでは、MCTSが `SAFETY` を選ぶと、攻撃と共通のGoal-conditioned 5D SACがショットを生成する。そのGoalを自己対戦ログに保存し、次のMicro学習で同一のGoalを照合して使う。従来の固定SAFETYと既存v3チェックポイント・ログの読込は既定モードで使える。
- 暫定の守備報酬は合法に手番を渡した後、相手が最小番号球を直接狙えるポケット候補数の減少を評価する。候補判定はMacroと共通の幾何判定を用いる。学習ログには指定ポケットへの入球率と守備目標達成率を別々に記録し、`accepted_steps_by_goal` で各戦術の有効学習件数を確認できる。戦術の選択比率はMCTSの結果に従う。
- 仮の初期重みとネイティブFastFizで2反復の12ステージが完了し、各反復でSAFETYが選択・実行され、そのログに基づくMicro学習へ到達した。DGXの既存チェックポイントで確認する場合は次の短い実行を用いる。各反復の `micro/summary.json` で `accepted_steps_by_goal.SAFETY` を確認する。次の実装課題はPOSITION停止目標の生成、BREAKの学習経路、9-ballルールの残項目。

```bash
conda activate bill_sub1
cd ~/research
git pull --ff-only &&
BILLIARDS_FASTFIZ_BUILD_DIR=/tmp/billiards-fastfiz-phase7 \
python -m scripts.iterate_nine_ball_hierarchical_v3 \
  --base-micro-checkpoint artifacts/runs/phase15_hierarchical_iteration_v2/iteration_001/micro/micro_sac_5d.pt \
  --base-policy-checkpoint artifacts/runs/phase15_hierarchical_iteration_v2/iteration_001/macro_policy_v3.pt \
  --base-state-v-checkpoint artifacts/runs/phase15_hierarchical_iteration_v2/iteration_001/qv/state_v_v3.pt \
  --iterations 2 --matches-per-iteration 8 --start-seed 840000 \
  --mcts-simulations 12 --mcts-max-depth 2 \
  --policy-epochs 1 --qv-epochs 1 --micro-steps 160 \
  --learned-safety --device cuda \
  --output-root artifacts/runs/phase16_learned_safety_smoke_v1
```
