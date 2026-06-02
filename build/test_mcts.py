import numpy as np
from custom_env import MinimalBilliardEnv
import copy
import torch

# 先ほど作ったファイルから、ニューラルネットの設計図をインポート
from train_policy import PolicyNetwork

# 学習済みモデルの読み込み準備
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = PolicyNetwork().to(device)
model.load_state_dict(torch.load("policy_model.pth", weights_only=True))
model.eval() # 推論モードに設定

def get_ai_intuition(env_state, target_ball):
    """
    【本物のAIの直感】
    PyTorchのニューラルネットワークに盤面を見せて、角度を予測させる
    """
    cue_x, cue_y = env_state[0], env_state[1]
    obj_x, obj_y = env_state[2 + target_ball*2], env_state[3 + target_ball*2]
    
    with torch.no_grad():
        inputs = torch.tensor([[cue_x, cue_y, obj_x, obj_y]], dtype=torch.float32).to(device)
        predicted_angle = model(inputs).item()
        
    return predicted_angle

def hierarchical_mcts_search(env, num_simulations=100): 
    # AIの直感がまだ少しズレているため、探索回数を100回に増やします
    original_state = env.get_state()
    best_action = None
    best_reward = -1.0

    print(f"--- 賢い階層型 MCTS開始 (探索回数: {num_simulations}回/ターゲット) ---")
    
    for target_ball in [0, 1]:
        print(f"  >> マクロ戦略: [的球 {target_ball}] を評価します...")
        
        # 1. 【変更】PyTorchモデルから直感をもらう
        suggested_angle = get_ai_intuition(original_state, target_ball)
        print(f"     Policyの直感: {suggested_angle:.2f}度 付近が怪しいです")
        
        # 2. 直感の周辺を重点的に探索する
        for i in range(num_simulations):
            env.set_state(original_state)
            
            test_angle = np.random.normal(loc=suggested_angle, scale=20.0)
            
            test_action = np.array([target_ball, 0.8, test_angle], dtype=np.float32)
            obs, reward, terminated, _, _ = env.step(test_action)
            
            if reward > best_reward:
                best_reward = reward
                best_action = copy.deepcopy(test_action)
                
    print(f"--- MCTS終了: 見つかった最大報酬 = {best_reward} ---")
    env.set_state(original_state)
    return best_action

# 【セクション2】メイン実行ループ
if __name__ == "__main__":
    env = MinimalBilliardEnv()
    
    obs, info = env.reset()
    print(f"初期状態: \n{obs}")
    
    # 探索回数
    action = hierarchical_mcts_search(env, num_simulations=30)
    
    if action is not None:
        print(f"\nMCTSが選択した行動: Target=的球{int(action[0])}, Power={action[1]:.2f}, Angle={action[2]:.2f}度")
        next_obs, reward, terminated, _, _ = env.step(action)
        print(f"結果 -> 報酬: {reward}, 終了: {terminated}")
    else:
        print("良い行動が見つかりませんでした。")
        