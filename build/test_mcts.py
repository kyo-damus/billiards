import torch
import torch.nn.functional as F
import numpy as np

def run_muzero_mcts(obs, rep_net, dyn_net, pred_net, config, num_simulations=30, discount=0.99):
    """
    物理シミュレータを使わず、AIの3つのネットワークだけでMCTS（先読み探索）を行う
    """
    # 1. まず、現在の現実の盤面を「脳内言語（潜在状態）」に変換する
    # ※ここは representation_net の出番です
    obs_tensor = torch.FloatTensor(obs).unsqueeze(0)
    
    with torch.no_grad(): # 探索中は学習しないので勾配計算をオフ
        latent_state = rep_net(obs_tensor)
        
        # 2. 現在の盤面に対するAIの「直感（Prior）」を取得する
        mu, log_std, current_value = pred_net(latent_state)
        std = log_std.exp()
        
        # --- 行動候補のサンプリング (以前の get_sac_candidates と同じ) ---
        action_candidates = []
        best_guess = torch.tanh(mu).cpu().numpy()[0]
        action_candidates.append(best_guess) # 本命
        
        dist = torch.distributions.Normal(mu, std)
        for _ in range(num_simulations - 1): # 揺らぎ
            u = dist.rsample()
            action = torch.tanh(u).cpu().numpy()[0]
            action_candidates.append(action)
            
        # --- 脳内シミュレーション (ここが最大の変更点！) ---
        best_action = None
        best_score = -float('inf')
        
        for action in action_candidates:
            action_tensor = torch.FloatTensor(action).unsqueeze(0)
            
            # 【ズル廃止】env.step(action) の代わりに dyn_net を使う！
            # 脳内で「この行動をとったら、次どうなるか？報酬は？」を予測する
            next_latent, predicted_reward = dyn_net(latent_state, action_tensor)
            
            # 【評価】Criticの代わりに pred_net の value_net を使う！
            # 予測された「未来の脳内状態」がどれくらい有利かを評価
            _, _, future_value = pred_net(next_latent)
            
            # 総合スコア ＝ 脳内予測報酬 ＋ (割引率 × 未来の脳内価値)
            score = predicted_reward.item() + discount * future_value.item()
            
            # 最もスコアが高い行動を記録
            if score > best_score:
                best_score = score
                best_action = action
                
    # --- 学習用の Target(正解) データを返す ---
    # MCTSが脳内でウンウン考えた結果、「この手が一番良かった(best_action)」「価値はこれくらいだった(best_score)」
    # という結果を、ネットワークを鍛えるための「正解ラベル」としてメインループに返す
    target_policy = best_action  # AIの直感(mu)をこの値に近づけるように学習させる
    target_value = best_score    # AIの価値予測(value)をこの値に近づけるように学習させる
    
    return best_action, target_policy, target_value
    