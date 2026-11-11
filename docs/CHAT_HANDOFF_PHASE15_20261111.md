# チャット移行・研究実装引き継ぎ（2026-11-11、Phase15）

> **このファイルが今回の引き継ぎの最優先資料。** 将来のチャットでは、この記録と GitHub の最新ソースを照合する。最新コード・DGX実行ログが本ファイルより新しい場合、確認して内容を更新する。研究計画を勝手に簡易2球環境へ戻さない。

## 0. 新しいチャットに貼る文章

私は修士研究としてビリヤードの階層型強化学習を実装中です。このチャットは前の長い研究実装チャットからの継続です。最初に GitHub `kyo-damus/billiards` の `phase15-nineball-coupled-learning-audit-20261111` の HEAD、`docs/CHAT_HANDOFF_PHASE15_20261111.md`、`docs/RESEARCH_CONTEXT.md`、`docs/PHASE15_PROPOSAL_ALIGNMENT.md` と関連ソースを読んでください。ユーザーの直近の実機ログでは **seed=510040 が修正後に完走（winner=0、13 shots、4 fouls）** しましたが、以前の2反復×32試合は最初の20試合を保存後に FastFiz の `Bad Shot Exception` で中断したままです。修正後の関連pytest結果はまだ未報告で、Phase15の完全な交互学習も未完走です。まず関連pytest、既存の出力先保全、単体リプレイの記録を確認し、**新しい独立run rootで**2反復学習を実行・検証するところから続きをお願いします。研究の最優先目標は、9-ballの推論・自己対戦・Macro/Micro交互学習・大規模実験・結果考察を速やかに一通り完走させることです。小規模環境の微細な最適化、均等な戦術選択の強制、細かなアブレーション、既存研究との比較は今は後回しです。既存ファイル、学習済み重み、実験ログを削除・上書きせず、ソース変更は関連テスト・ドキュメントとともにGitHubへ反映してください。

## 1. 研究目的（変更禁止の基本前提）

- 戦術選択 **Macro** と連続値ショット制御 **Micro** を組み合わせた、ビリヤードの階層型強化学習を実装・評価・考察する。最終的には一般的な対戦ルールで推論し、自己対戦と（後段で）既存手法との規定ラック先取対戦を評価する。
- 主対象は9-ballの**手球1個＋的球1〜9の計10球**。旧2的球の簡易環境は回帰・事前学習用の歴史的実装。現段階の9-ballは一部ルールのみであり、WPA完全準拠やゲーム強度を主張しない。
- Macro候補は`DIRECT_ATTACK`, `POSITION_ATTACK`, `SAFETY`、ブレイク時は専用処理。可変個数のGoalを生成し、Candidate PolicyをMCTS priorに使い、物理FastFiz＋Micro＋ゲームルール＋State Vを利用した**手番考慮PUCT**で行動を選ぶ。実行済み候補のQは未実行候補の真値を保証しないため、現行MCTSでは使わない。
- Microは **Goal-conditioned SAC**。v3のMacro状態72D＋Goal候補36D、Micro入力108D、ショットアクション5D（方向、速度/強さ、仰角、撞点x/y）。旧2D・12D・31D等の学習済み重みと混在させない。
- 目的は研究全体を動かし結果を得ることであり、勝率や戦術の出現率を事前に「それらしく」強制しない。全てDIRECTが最善ならDIRECTばかりを選んでよい。
- ゼミの発表では前回との差分、研究として重要な実装進捗を優先。軽微なtie-break等は主たる成果として誇張しない。

## 2. Macro/Micro学習機構：Phase14→15での正しい理解

1. 9-ball自己対戦：現在のMacro Policy・State V・MicroでMCTSを回し、物理遷移・ルール・選択Goal・root訪問数を保存する。
2. Macro Policy：MCTS rootの**訪問分布**を教師とするlistwise CEの蒸留。ゲーム途中で打ち切ったmatchの**有効root**も利用可。以前は同じ盤面でMicro物理候補を評価した教師から初期学習している。
3. Macro Q/V：完了した**race-to-1**試合の、実行された選択行動のみについて、勝敗から手番に応じて符号を変えた割引終端MCリターンを教師に回帰。未完了試合や実行していないGoalに正解Qを創作しない。72D状態に対戦スコアがないのでrace-to-N学習へ無条件拡張しない。
4. Micro：学習済み5D SACを `--init-checkpoint` で初期化し、`--goal-source logged` で当該自己対戦でMacroが選んだ**厳密に一致するDIRECT/POSITION Goal**を用いて物理shotをfine-tune。Actor/Critic/Target/Alphaは読み込むがoptimizer/replayはリセット。Break/SAFETYはまだこの学習でカバーしない。
5. 反復ランナー `scripts/iterate_nine_ball_hierarchical_v3.py` が「自己対戦→Macro訪問学習→Q/V回帰→Micro継続SAC→更新済みモデルで次反復」の経路を構築。これは**交互更新**で、MacroとMicroを同時勾配最適化する方式ではない。
6. Phase15で修正した設計不整合：POSITION_ATTACKで指定球を**指定ポケットへ成功**させていないのに手球位置ボーナスを与えるケースを修正。

## 3. 最新の実装・実機結果（2026-11-11現在）

- 作業GitHub: https://github.com/kyo-damus/billiards
- 安全作業ブランチ: `phase15-nineball-coupled-learning-audit-20261111`、今回引き継ぎ前のHEAD `a5cc24b2101c54049040d8770d07af3a28ce13a1`。引き継ぎドキュメント追加時はHEADが前進するので、新チャットで必ず`git rev-parse HEAD`とリモートHEADを確認。
- 原`phase14-nineball-mcts-visits-policy`、旧`refactor-phase9-preserve-20261109`は保護済み。最終的なmainへのマージ状況は自明としない。
- Phase15最初の `artifacts/runs/phase15_hierarchical_iteration_v1`：`--iterations 2 --matches-per-iteration 32 --mcts-simulations 24 --mcts-max-depth 2 --micro-steps 5000`、`000_self_play` で **20試合 (seed 510000〜510038) を保存した後に例外で中断**。Macro訪問学習、Q/V更新、Micro fine-tune、2反復完走は未実施。
- 例外：`RuntimeError: Bad Shot Exception`、ネイティブログに`Cue overlaps Four`約1.67e-8mと`Two overlaps Five`約0.0293973m。後者は約29.4mmの重大な重なり。発生根源が実対戦か仮想盤面かは、詳細な失敗状態をリプレイしない限り断定禁止。
- 修正したソース：`src/game/nine_ball_physics_guard.py`（球重なり、ネイティブ未実行コード）、`src/game/nine_ball_mcts_v3.py`（無効edgeを訪問・報酬に数えず記録）、`src/game/nine_ball_session.py`（ライブ不正物理遷移を反則/勝敗に変えない）、`scripts/collect_nine_ball_self_play.py`（`invalid_physics`未完了記録、保存済みseedの再開）、`scripts/iterate_nine_ball_hierarchical_v3.py`（manifestとseedの保全、再開オプション）。対応テストを追加。**根本的にFastFiz/球重なり生成を修復した証拠ではない**。
- **重要な新規実機ログ**：修正後の1試合単独収集
  `python -m scripts.collect_nine_ball_self_play --matches 1 --race-to 1 --start-seed 510040 --max-shots-per-rack 80 --agent-kind micro_mcts --micro-checkpoint artifacts/runs/phase8_micro5d_first/outputs/micro_sac_5d.pt --macro-policy-checkpoint artifacts/runs/phase14_nineball_macro_visit_policy_smoke_v1/outputs/macro_policy_visits_v3.pt --state-value-checkpoint artifacts/runs/phase12_nineball_qv_smoke_v1/outputs/state_v_v3.pt --mcts-simulations 24 --mcts-max-depth 2 --output-dir artifacts/runs/phase15_badshot_replay_seed510040_v1/outputs --device cuda`
  が **`finished=True winner=0 rack_wins=[1,0] shots=13 fouls=4`** で完走。これは1試合の再実行成功であり、修正の網羅性・大規模実験の成功は未確認。現時点で修正後pytestのユーザーログも未報告。
- GitHub権限：ユーザーはツール承認画面で「すべてのツールを許可」を選択。Private時に一部APIで403が発生したが、`create_file`/`update_file`経由で実際にコミットできた。ユーザーはPublicへ戻し、最後のGitHub確認では`visibility=public`かつ`push=true`。UI許可とGitHub Appのリポジトリ権限は別。必要時は最新権限を確認、勝手に変更しない。

## 4. 現在の重要ファイルとモデル

- 研究の長期記録: `docs/RESEARCH_CONTEXT.md`。提案手法の修正監査: `docs/PHASE15_PROPOSAL_ALIGNMENT.md`。構成: `docs/PROJECT_STRUCTURE.md`。テスト: `tests/README.md`。
- 9-ballのルール/セッション: `src/game/nine_ball_rules.py`, `src/game/nine_ball_session.py`, `src/game/match.py`。
- モデルと学習CLI: `scripts/train_nine_ball_micro_sac.py`, `scripts/train_nine_ball_macro_visits_v3.py`, `scripts/build_nine_ball_macro_returns_v3.py`, `scripts/train_nine_ball_macro_qv_v3.py`。
- 自己対戦/探索/交互学習: `src/game/nine_ball_mcts_v3.py`, `src/game/nine_ball_mcts_agent.py`, `scripts/collect_nine_ball_self_play.py`, `scripts/iterate_nine_ball_hierarchical_v3.py`。
- 起点Micro 5D: `artifacts/runs/phase8_micro5d_first/outputs/micro_sac_5d.pt`。
- 起点Macro Policy v3: `artifacts/runs/phase14_nineball_macro_visit_policy_smoke_v1/outputs/macro_policy_visits_v3.pt`。
- 起点State V v3: `artifacts/runs/phase12_nineball_qv_smoke_v1/outputs/state_v_v3.pt`。
- 起点ファイルの実体はDGXローカルにあり、GitHubにあるとは限らない。勝手に削除/移動/改変/上書きしない。
- 最初の反復失敗 `artifacts/runs/phase15_hierarchical_iteration_v1`、単独成功 `artifacts/runs/phase15_badshot_replay_seed510040_v1/outputs` の両方を保護する。

## 5. 次にやること（優先順、再実行前）

1. `conda activate bill_sub1`、`cd ~/research`、`git status --short --branch`、`git rev-parse --short HEAD` と `git pull --ff-only`。未コミット変更があれば先に保護。強制reset禁止。
2. `export BILLIARDS_FASTFIZ_BUILD_DIR=/tmp/billiards-fastfiz-phase7`。CUDA_VISIBLE_DEVICESは**ユーザーが既に設定**しているためその値を変更しない。共有DGXは8枚×約32GiB、1枚を基本、最大でも2枚程度まで（占有禁止）。
3. 関連pytestをDGXで実行し、実行結果を確認する：
   ```bash
   python -m pytest \
     tests/unit/game/test_nine_ball_physics_guard.py \
     tests/unit/game/test_nine_ball_mcts_v3.py \
     tests/unit/game/test_nine_ball_self_play.py \
     tests/pipelines/macro/test_nine_ball_hierarchical_iteration_v3.py \
     tests/pipelines/micro/test_train_nine_ball_micro_sac.py \
     tests/integration/game/test_nine_ball_native_session.py -v
   ```
4. すでに`seed=510040`単独は完走しているので、**同じ出力先に再実行しない**。次は未確認のedge例外系と異常試合の記録形式をテストで評価。必要なバグ修正をしてから本実験へ進む。
5. 本格実験を再開する際は**新run** `artifacts/runs/phase15_hierarchical_iteration_v2`（未存在なら）で、下記設定の2反復×32試合を最初から取り直す。v1の保存済み20試合は混ぜない（コード変更によるデータ一貫性の問題）。再開オプション `--resume --accept-unverified-code-change` は旧v1のデバッグ限定。必要ならより大規模な学習に段階的に拡大する。
6. 各段階のoutputと`run_manifest.json`、完了/未完了試合数、正常root数・無効物理root数、Policy/Q/V/Microのcheckpoint、test/valを報告し、反復2まで完走後に独立seed・先攻後攻対称な自己対戦評価へ。
7. 重要な未達成はBreak/SAFETYのMicro学習、適切なPOSITION停止Goal、ルール完全性、実力比較。結果が不足なら性能調整。**微細なアブレーションを先にしない**。

### 2反復試験のコマンド（新規出力先、既存を上書きしない）

```bash
cd ~/research
conda activate bill_sub1
export BILLIARDS_FASTFIZ_BUILD_DIR=/tmp/billiards-fastfiz-phase7

python -m scripts.iterate_nine_ball_hierarchical_v3 \
  --base-micro-checkpoint artifacts/runs/phase8_micro5d_first/outputs/micro_sac_5d.pt \
  --base-policy-checkpoint artifacts/runs/phase14_nineball_macro_visit_policy_smoke_v1/outputs/macro_policy_visits_v3.pt \
  --base-state-v-checkpoint artifacts/runs/phase12_nineball_qv_smoke_v1/outputs/state_v_v3.pt \
  --iterations 2 --matches-per-iteration 32 --max-shots-per-rack 80 \
  --mcts-simulations 24 --mcts-max-depth 2 \
  --policy-epochs 10 --qv-epochs 5 --micro-steps 5000 \
  --output-root artifacts/runs/phase15_hierarchical_iteration_v2 \
  --device cuda
```

**実行前に`test ! -e artifacts/runs/phase15_hierarchical_iteration_v2`で出力先が未使用か確認。**

## 6. 研究・資産管理方針

- FastFizは物理シミュレータ。World Model/MuZeroを学習済みと誤認しない。Micro SACは全局面最適化済みと主張しない。
- 2026-11-09付tarバックアップ：`~/billiards_research_backup_20261109.tar.gz`、旧build成果物退避：`~/billiards_refactor_quarantine_20261109/`。これらと旧checkpointは保護。旧2球Macroと9-ball v3を混同しない。
- 大きな機能完成・方針変更時には`docs/RESEARCH_CONTEXT.md`を更新。新チャットへの最新版まとめは本ファイルを保守。
- 重要な診断の原則：物理不正を「反則」「失敗報酬」「負け」と混同せず、結果/候補の欠落は欠落として記録する。性能向上や勝率の主張は独立評価で初めて行う。
