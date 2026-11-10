# ゼミ資料作成用引き継ぎ（2026-11-10）: 9-ball階層型RLの基盤構築と初期学習

**このフェーズで実装はいったん停止。以降は資料作成に移る。**
対象ブランチ: `phase14-nineball-mcts-visits-policy`
研究仕様: `docs/RESEARCH_CONTEXT.md`

## 研究上のメッセージ（冒頭で明言）

以前の簡易2的球環境で構築したMacro候補・Policy/Q/V・MCTS・Microの設計を維持しつつ、
実物理の9-ballに対応した状態・TacticalGoalと5D連続ショット操作を新設した。
固定Microを用いてMacro方策学習、勝敗リターンQ/V学習、手番考慮MCTS、
root訪問分布によるMacro方策更新まで**一通りの実行・学習経路を接続**した。
今回の主な成果は**9-ball移行と接続・学習経路の動作検証**である。
勝率向上や強化学習の収束は証明していない。
一回の訪問分布方策更新ができたのであって、
複数回の自動反復学習・独立条件での対戦比較は未実装／未実施。

## 前回の発表内容と今回の差分

- 旧環境: 簡易2的球、Macro Candidate 25/31D、旧2D Micro＋Power Residual、
  攻守混合の探索・候補Policy/Q/V。
- 新環境: 完全な1..9球 inventory、部分的な9-ballルール、
  新v3 Macro状態72D / Goal36D / Micro入力108D / 5Dショット
  （速度、角度、キュー仰角、撞点x/y）。
- 旧Macro重みは新規v3と互換ではない。モデル構造と学習経路を移行し、
  新データで学習。旧チェックポイントは保全。
- 旧簡易環境のMCTS訪問数同点対処の分析と、
  新9-ball MCTSへの対応は別実験。旧2球の数値を9-ball結果と混ぜない。

## 13枚のスライド案（図は最小限、差分は赤字、数式と短い説明中心）

1. **タイトル・進捗報告の範囲**：9-ball対応とMacro/Micro連携の動作検証
2. **研究目的とこれまでの実装**：戦術選択（Macro）／5D物理操作（Micro）を階層化、
   候補生成→Policy→MCTS→Goal→Micro→FastFiz→局面更新
3. **前回からの変更点**：簡易2球→9-ball、2D→5D、Macro v3表現への変更、
   実試合リターン/MCTS訪問分布学習（赤字で強調）
4. **9-ball環境と物理/ルール**：最低番号球、first contact、rail、scratch、
   ball-in-hand、3ファウル等。WPA完全準拠ではない
5. **状態・Goal・行動表現**：Macro72D、Goal36D、Micro108D、
   Action5D。BREAK/DIRECT/POSITION/SAFETY。
   BREAK/SAFETYはまだヒューリスティック明示shot
6. **Goal-conditioned Micro SAC**：
   状態とGoalから連続ショット、SAC 5000 steps（86訓練盤面）、
   学習終盤500 step窓の平均reward -0.3401、exact pot rate 4.0%。
   未学習盤面での制御性能は未評価
7. **Macro Candidate Policyの初期化**：
   全候補を同じ盤面から固定Micro＋FastFizで1ショット評価し、
   局所教師softmaxへ教師あり学習。
   6盤面49候補、教師上の反則率26.53%、exact pot率14.29%。
   真のQではない
8. **Macro Q(s,c) / V(s)の学習**：
   完了race-to-1試合だけから終端勝敗±1を教師化、
   手番を考慮したリターンの符号反転。
   6試合134ショット、train90/val44。
   Q検証MSE0.77985、V0.76534、定数baseline0.79387。
   Qは実行済み候補だけ、Vはbehavior-policyの値
9. **9-ballに対応したMCTS**：
   候補Policy prior、各Goalの固定Micro＋FastFiz遷移、
   NineBallRules、State V葉評価、
   手番符号 `sign=+1` 同手番／`-1` 相手番、
   `Q\_edge = r + gamma * sign * V\_child`。
   selected-action Qは探索で未使用
10. **MCTS訪問分布によるMacro Policy更新**：
    root訪問分布`pi_i = N_i/sum_j N_j`への
    `L=-sum_i pi_i log P_theta(c_i|s)` を最小化。
    同盤面のGoal群を保ち、match単位train/val分割
11. **Phase14の小規模結果**：
    28関連テストPASS、3試合完走・30ショット・7反則、
    30root中27root（3つの単一候補を除外）／185候補。
    Policy学習2試合15root、検証1試合12root。
    訪問CE 1.674182→1.672084、
    最大visit候補一致率83.3%→83.3%
12. **達成内容と現状の限界**：
    Macro→探索→Micro→物理→ゲーム→学習の接続を確認。
    未実装: 複数反復の自動化、対戦勝率に基づく独立比較、
    高品質SAFETY、Micro十分学習、完全ルール。
    「CEが少し下がった」ことを「戦術性能向上」と書かない
13. **次の研究計画**：
    (1) 自己対戦→探索→Policy更新の複数反復
    (2) Seed・先後を揃えたMacroのみ/MCTS/更新後Policy比較
    (3) Q/VとMicroの学習データ拡大・防御候補改善。
    ゼミ前に大規模学習は行わない

## 実測ログ一覧（用途を混ぜない）

| 段階 | 実測 | 正しい解釈 |
|---|---|---|
| 5D Micro初期学習 | 5000 steps、86訓練状態、終盤窓exact pot 4.0% | 学習ログ、ゲーム性能ではない |
| Phase11物理教師 | 6盤面/49候補、foul 26.53%、exact pot 14.29% | 全候補の単発物理試行の平均 |
| Phase11 Macro学習 | 4 train / 2 val 盤面、教師最良同点一致50% | 小規模教師あり学習 |
| Phase12勝敗データ | 6完了試合、134ショット、55反則 | 対戦経験収集 |
| Phase12 Q/V | train90/val44ショット、Q MSE0.77985、V0.76534 | 完了試合の実行選択のMC教師 |
| Phase13 MCTS統合 | 24 tests PASS、1試合5ショット、1反則 | 統合smoke |
| Phase14 MCTS自己対戦 | 28 tests PASS、3試合30ショット、7反則 | 学習用訪問収集の動作検証 |
| Phase14 Policy訪問学習 | 27root/185候補、train15/val12root。CE 1.67418→1.67208 | MCTS模倣の検証CE僅差改善 |

異なる実験はSeed・手法・探索回数が一致していない。
ショット数や反則率の差を性能比較として利用しない。
Phase11/12/13/14いずれも統計的に十分な汎化評価ではない。

## 発表で言い切ること／言い切らないこと

言い切る：
- 9-ball物理・ルールにMacro/Micro学習とMCTSを接続した。
- 学習済み5D Micro、Macro Policy、Macro Q/Vを保存できる。
- MCTS訪問分布でMacro Policyを一度更新できる。
- 実機DGXで関連テストと小規模試合を完走した。

言い切らない：
- 「MCTSで勝率が向上した」「Qが最適候補を正しく推定した」
- 「5D Microが上手くポケットを達成できる」
- 「SAFETY/Position Goalが最適化されている」
- 「階層型RLの本格的な反復学習を完成した」
- 「正式9-ballルールに完全準拠した」

## 資料制作の運用

資料は10〜20ページ、白黒のシンプルな構成、**前回との差分は赤字**。
フローチャートは多用しない。コードは掲載しない。
手法・学習教師の違いは数式と文章で説明し、結果は小表にまとめる。
前回の実際の発表スライドが別内容の場合は差分の表現をスライドに合わせて調整する。
