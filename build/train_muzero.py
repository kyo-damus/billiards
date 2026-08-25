import os
import torch
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

# 作成したモジュール群
from muzero_models import MuZeroConfig, RepresentationNetwork, DynamicsNetwork, PredictionNetwork
from muzero_models import compute_muzero_loss
from muzero_buffer import MuZeroBuffer, Episode
from custom_env import MinimalBilliardEnv

def run_muzero_mcts(obs, rep_net, dyn_net, pred_net, config, device, num_simulations=30, discount=0.99):
    obs_tensor = torch.FloatTensor(np.array(obs, dtype=np.float32)).unsqueeze(0).to(device)

    with torch.no_grad():
        latent_state = rep_net(obs_tensor)
        mu, log_std, current_value = pred_net(latent_state)
        # std = log_std.exp()

        exploration_noise = 0.5 
        std = log_std.exp() + exploration_noise

        action_candidates = []
        action_candidates.append(torch.tanh(mu).cpu().numpy()[0])

        dist = torch.distributions.Normal(mu, std)
        for _ in range(num_simulations - 1):
            u = dist.rsample()
            action_candidates.append(torch.tanh(u).cpu().numpy()[0])

        best_action = None
        best_score = -float('inf')

        for action in action_candidates:
            action_tensor = torch.FloatTensor(action).unsqueeze(0).to(device)

            next_latent, predicted_reward = dyn_net(latent_state, action_tensor)
            _, _, future_value = pred_net(next_latent)

            score = predicted_reward.item() + discount * future_value.item()

            if score > best_score:
                best_score = score
                best_action = action

    return best_action, best_action, best_score

def plot_metrics(history, save_dir):
    plt.figure(figsize=(12, 8))
    
    # 1. Total Rewardの推移
    plt.subplot(2, 2, 1)
    plt.plot(history['reward'], label='Reward', alpha=0.8)
    # 移動平均（10区間）を見やすく追加
    if len(history['reward']) >= 10:
        moving_avg = np.convolve(history['reward'], np.ones(10)/10, mode='valid')
        plt.plot(range(9, len(history['reward'])), moving_avg, color='red', label='Moving Avg')
    plt.title('Episode Total Reward')
    plt.xlabel('Iteration')
    plt.ylabel('Reward')
    plt.legend()
    plt.grid(True)
    
    # 2. Total Lossの推移
    plt.subplot(2, 2, 2)
    plt.plot(history['loss'], label='Total Loss', color='purple', alpha=0.8)
    plt.title('Total Loss')
    plt.xlabel('Iteration')
    plt.grid(True)
    
    # 3. 各種Loss（V, P, R）の推移
    plt.subplot(2, 2, 3)
    plt.plot(history['v_loss'], label='Value (V)', color='blue', alpha=0.7)
    plt.plot(history['p_loss'], label='Policy (P)', color='green', alpha=0.7)
    plt.plot(history['r_loss'], label='Reward (R)', color='orange', alpha=0.7)
    plt.title('Loss Components (V, P, R)')
    plt.xlabel('Iteration')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "training_metrics.png"))
    plt.close()

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用デバイス: {device}")

    save_dir = "results_muzero"
    os.makedirs(save_dir, exist_ok=True)

    # 1. 環境と設定の初期化
    env = MinimalBilliardEnv()
    config = MuZeroConfig(obs_dim=12, action_dim=2)
    buffer = MuZeroBuffer(config, max_episodes=2000)
    
    # 2. ネットワークとオプティマイザの準備
    rep_net = RepresentationNetwork(config).to(device)
    dyn_net = DynamicsNetwork(config).to(device)
    pred_net = PredictionNetwork(config).to(device)
    
    params = list(rep_net.parameters()) + list(dyn_net.parameters()) + list(pred_net.parameters())
    optimizer = optim.Adam(params, lr=3e-4, weight_decay=1e-4)
    
    batch_size = 64
    K_steps = 3 
    num_iterations = 100000

    history = {
        'reward': [], 'loss': [], 'v_loss': [], 'p_loss': [], 'r_loss': []
    }

    print(" MuZero学習ループを開始します...")

    for iteration in range(num_iterations):
        # ==========================================
        # フェーズ1: 自己対局 (実環境との対話)
        # ==========================================
        obs, _ = env.reset()
        episode = Episode()
        
        done = False
        step_count = 0
        total_reward = 0.0
        
        while not done and step_count < 20: 
            # ① AIが脳内で考えて最善手を出す（物理エンジンは使わない！）
            action, target_policy, target_value = run_muzero_mcts(
                obs, rep_net, dyn_net, pred_net, config, device, num_simulations=150
            )

            # print(f"Step {step_count}: Action = {action}")
            
            # ② 出した答え(action)を、本物の物理エンジンで打つ
            next_obs, reward, done, _, _ = env.step(action)
            
            # ③ 結果をバッファに保存
            episode.obs.append(obs)
            episode.actions.append(action)
            episode.rewards.append(reward)
            episode.target_values.append(target_value)
            episode.target_policies.append(target_policy)
            
            obs = next_obs
            total_reward += reward
            step_count += 1
            
        buffer.store_episode(episode)
        
        # ==========================================
        # フェーズ2: 学習 (脳のアップデート)
        # ==========================================
        if len(buffer.buffer) < 10:
            continue
            
        updates_per_iteration = 3 
        
        for _ in range(updates_per_iteration):
            batch_data = buffer.sample_batch(batch_size, K_steps)

            batch_data = {k: v.to(device) for k, v in batch_data.items()}

            loss, loss_info = compute_muzero_loss(
                rep_net, dyn_net, pred_net, batch_data, config
            )

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, max_norm=5.0)
            optimizer.step()
            
        # ==========================================
        # フェーズ3: ログ出力
        # ==========================================
        history['reward'].append(total_reward)
        history['loss'].append(loss.item())
        history['v_loss'].append(loss_info['value_loss'])
        history['p_loss'].append(loss_info['policy_loss'])
        history['r_loss'].append(loss_info['reward_loss'])

        if iteration % 10 == 0:
            print(f"Iter: {iteration:04d} | Reward: {total_reward:5.1f} | "
                  f"Loss: {loss.item():.3f} "
                  f"(V:{loss_info['value_loss']:.3f}, P:{loss_info['policy_loss']:.3f}, R:{loss_info['reward_loss']:.3f})")

        # ⭕ 50イテレーションごとにグラフを画像として保存
        if iteration % 50 == 0 and iteration > 0:
            plot_metrics(history, save_dir)

        if iteration % 100 == 0 and iteration > 0:
            torch.save(rep_net.state_dict(), f"{save_dir}/rep_net_latest.pth")
            torch.save(dyn_net.state_dict(), f"{save_dir}/dyn_net_latest.pth")
            torch.save(pred_net.state_dict(), f"{save_dir}/pred_net_latest.pth")
            print(f"Iter {iteration}: モデルとグラフを {save_dir}/ に保存しました。")

if __name__ == "__main__":
    train()