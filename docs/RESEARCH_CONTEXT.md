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
