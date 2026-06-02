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
    state_for_ai = list(env_state) 
    
    if target_ball == 1:
        state_for_ai[2], state_for_ai[3], state_for_ai[4], state_for_ai[5] = \
            state_for_ai[4], state_for_ai[5], state_for_ai[2], state_for_ai[3]

    with torch.no_grad():
        inputs = torch.tensor([state_for_ai], dtype=torch.float32).to(device)
        # 【変更】AIの出力は [sin, cos] の2つの値になる
        output = model(inputs)[0].cpu().numpy()
        sin_val, cos_val = output[0], output[1]
        
        # arctan2を使って、sinとcosから正しい角度（度数法）を復元
        predicted_angle = np.degrees(np.arctan2(sin_val, cos_val))
        
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
            
            test_angle = np.random.normal(loc=suggested_angle, scale=5.0)
            
            test_action = np.array([target_ball, 0.8, test_angle], dtype=np.float32)
            obs, reward, terminated, _, _ = env.step(test_action)
            
            if reward > best_reward:
                best_reward = reward
                best_action = copy.deepcopy(test_action)
                
    print(f"--- MCTS終了: 見つかった最大報酬 = {best_reward} ---")
    env.set_state(original_state)
    return best_action

if __name__ == "__main__":
    env = MinimalBilliardEnv()
    
    # 【追加】成功体験を保存するためのリスト（Experience Replay Buffer）
    experience_buffer_states = []
    experience_buffer_targets = []
    
    num_episodes = 100  # まずは10回の自己対局でテスト
    print(f"=== Self-Play（自己対局）データ収集開始: {num_episodes}エピソード ===")
    
    for episode in range(num_episodes):
        obs, info = env.reset()
        print(f"\n[Episode {episode+1}/{num_episodes}] 初期状態: {obs}")
        
        # 探索回数は30回のストイック設定
        action = hierarchical_mcts_search(env, num_simulations=30)
        
        if action is not None:
            next_obs, reward, terminated, _, _ = env.step(action)
            print(f" -> 結果: 報酬 {reward}")
            
            # 【手法の要】報酬が1.0（成功）だった場合のみ、その「状態」と「MCTSが見つけた正解角度」を学習データとして保存する
            if reward == 1.0:
                experience_buffer_states.append(list(obs))
                experience_buffer_targets.append([action[2]]) # action[2] は角度
        else:
            print(" -> 失敗: 良い行動が見つかりませんでした。")
            
    # 集めた成功体験をPyTorchのTensor形式に変換してファイルに保存
    if len(experience_buffer_states) > 0:
        states_tensor = torch.tensor(experience_buffer_states, dtype=torch.float32)
        targets_tensor = torch.tensor(experience_buffer_targets, dtype=torch.float32)
        
        # 辞書形式で保存
        torch.save({"states": states_tensor, "targets": targets_tensor}, "mcts_experience.pt")
        print(f"\n=== データ収集完了! {len(experience_buffer_states)}件の成功体験を 'mcts_experience.pt' に保存しました ===")
    else:
        print("\n=== データ収集完了...しかし成功体験は0件でした。===")
        