import torch
import numpy as np
from custom_env import MinimalBilliardEnv # 環境クラス名は実際の環境に合わせてください
from train_policy import MuZeroSACPolicyNet # ネットワークのクラスをインポート

def evaluate_agent(model_path="results/actor.pth", num_episodes=100):
    print("\n--- Starting Evaluation (Deterministic) ---")
    
    # 1. デバイスの設定と環境のロード
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"実行デバイス: {device}")
    env = MinimalBilliardEnv()
    
    # 2. 学習済みモデルのロード (8次元, 256隠れ層)
    actor = MuZeroSACPolicyNet(state_dim=12, action_dim=2, hidden_dim=256).to(device)
    
    try:
        actor.load_state_dict(torch.load(model_path, map_location=device))
        print(f"✅ モデルのロードに成功しました: {model_path}")
    except Exception as e:
        print(f"❌ モデルのロードに失敗しました。ファイル名を確認してください。\nエラー: {e}")
        return

    # 推論モード（これ以上の学習・Dropout等を無効化）
    actor.eval()
    
    success_count = 0
    total_rewards = []

    # 3. 評価ループ（ノイズなし）
    for ep in range(num_episodes):
        state, _ = env.reset()
        done = False
        ep_reward = 0
        
        while not done:
            # 【重要】勾配計算をオフにし、決定論的（ノイズなし）に行動を選択
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
                mu, _, _ = actor(state_tensor)
                # ノイズをサンプリングせず、平均値（mu）をそのまま採用
                action = torch.tanh(mu).cpu().numpy()[0]
                
            next_state, reward, terminated, truncated, _ = env.step(action)
            ep_reward += reward
            done = terminated or truncated
            state = next_state
            
        total_rewards.append(ep_reward)
        
        # ポケットイン判定（+10のボーナスが入っていれば成功とみなす）
        if ep_reward > 5.0: 
            success_count += 1
            
        # 10エピソードごとに進捗を表示
        if (ep + 1) % 10 == 0:
            print(f"Episode {ep+1:3d}/{num_episodes} | Reward: {ep_reward:.2f}")

    # 4. 最終結果の集計
    success_rate = (success_count / num_episodes) * 100
    avg_reward = np.mean(total_rewards)
    
    print("\n=== 🎉 Evaluation Results ===")
    print(f"Total Episodes : {num_episodes}")
    print(f"Success Rate   : {success_rate:.1f}% (ポケットイン成功率)")
    print(f"Average Reward : {avg_reward:.2f}")
    print("===============================\n")

if __name__ == "__main__":
    # 保存されているモデルのファイル名に合わせて適宜書き換えてください
    # 例: "results/sac_actor.pth" など
    evaluate_agent(model_path="results/sac_actor_ep100000.pth", num_episodes=100)