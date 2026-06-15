import numpy as np
import billiard_env_cpp # C++ FastFiz環境
import copy
import torch
from train_policy import PolicyNetwork

# 学習済みモデルの読み込み
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = PolicyNetwork().to(device)
# 注: 必要に応じてパスを確認してください
model.load_state_dict(torch.load("policy_model.pth", weights_only=True))
model.eval()

def get_ai_intuition(env_state, target_ball):
    # FastFiz環境の state は [cue_x, cue_y, obj0_x, obj0_y, obj1_x, obj1_y]
    state_for_ai = list(env_state)
    
    # ターゲット球が1番の場合、的球の位置を入れ替えてAIに入力するロジックは維持
    if target_ball == 1:
        state_for_ai[2], state_for_ai[3], state_for_ai[4], state_for_ai[5] = \
            state_for_ai[4], state_for_ai[5], state_for_ai[2], state_for_ai[3]

    with torch.no_grad():
        inputs = torch.tensor([state_for_ai], dtype=torch.float32).to(device)
        output = model(inputs)[0].cpu().numpy()
        # [sin, cos] から角度を復元
        predicted_angle = np.degrees(np.arctan2(output[0], output[1]))
    return predicted_angle

def hierarchical_mcts_search(env, num_simulations=30):
    original_state = env.get_state()
    best_action = None
    best_reward = -1.0

    # ターゲット0(的球1)とターゲット1(的球2)の両方を探索
    for target_ball in [0, 1]:
        suggested_angle = get_ai_intuition(original_state, target_ball)
        
        for i in range(num_simulations):
            env.set_state(original_state)
            
            # 物理エンジンのため、角度の揺らぎを適度に設定
            test_angle = np.random.normal(loc=suggested_angle, scale=2.0)
            power = 2.0 # 固定パワー
            
            # FastFizのC++インターフェースに合わせる
            # return: [cue_x, cue_y, obj0_x, obj0_y, obj1_x, obj1_y, reward, done]
            res = env.step(target_ball, power, test_angle)
            reward = res[6] # 報酬
            
            if reward > best_reward:
                best_reward = reward
                best_action = (target_ball, power, test_angle)
                
    return best_action

if __name__ == "__main__":
    env = billiard_env_cpp.BilliardSimulator()
    
    experience_buffer_states = []
    experience_buffer_targets = []
    
    num_episodes = 100
    print(f"=== FastFiz環境での自己対局開始 ===")
    
    for episode in range(num_episodes):
        env.reset()
        obs = env.get_state()
        
        action = hierarchical_mcts_search(env, num_simulations=30)
        
        if action is not None:
            target, power, angle = action
            # 最終結果の反映
            res = env.step(target, power, angle)
            reward = res[6]
            print(f"[Episode {episode+1}] Reward: {reward}")
            
            if reward == 1.0:
                experience_buffer_states.append(list(obs))
                experience_buffer_targets.append([angle])
    
    # データの保存
    if len(experience_buffer_states) > 0:
        torch.save({"states": torch.tensor(experience_buffer_states), 
                    "targets": torch.tensor(experience_buffer_targets)}, "mcts_experience.pt")
        print(f"=== 保存完了: {len(experience_buffer_states)}件 ===")