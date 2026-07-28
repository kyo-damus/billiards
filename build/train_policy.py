import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

import os
import matplotlib.pyplot as plt

import copy

from custom_env import MinimalBilliardEnv

# 【1】PyTorchモデルの定義 (SAC + MuZero 統合型アーキテクチャ)
class MuZeroSACPolicyNet(nn.Module):
    def __init__(self, state_dim=12, action_dim=2, hidden_dim=64):
        """
        state_dim: 6 (cue_x, cue_y, obj0_x, obj0_y, obj1_x, obj1_y)
        action_dim: 2 (例: 力の強さ, 角度のズレ など。要件に合わせて変更可)
        """
        super(MuZeroSACPolicyNet, self).__init__()
        
        # 共有特徴抽出層（現在のモデルの次元数を踏襲）
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        
        # --- Policy ヘッド (連続確率分布のパラメータを出力) ---
        # 行動の平均値 μ
        self.policy_mu = nn.Linear(hidden_dim, action_dim)
        # 行動のばらつき log(σ)
        self.policy_log_std = nn.Linear(hidden_dim, action_dim)
        
        # --- Value ヘッド (盤面評価を出力) ---
        self.value_head = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        """
        推論: MCTS等で状態価値や分布のパラメータが欲しい時に使用
        """
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        
        # Policy: μ と log_σ の計算
        mu = self.policy_mu(x)
        log_std = self.policy_log_std(x)
        
        # 分散が極端な値を取り、学習が崩壊するのを防ぐクランプ処理
        log_std = torch.clamp(log_std, min=-20, max=2)
        
        # Value の計算
        value = self.value_head(x)
        
        return mu, log_std, value

    def sample_action(self, state):
        """
        行動決定: シミュレータに渡すアクションをサンプリングする際に使用
        （再パラメータ化トリックの実装）
        """
        mu, log_std, _ = self.forward(state)
        std = log_std.exp()
        
        # 正規分布の作成
        normal = Normal(mu, std)
        
        # 再パラメータ化トリック (u = μ + σ * ε) 
        # rsample()により、サンプリングを挟んでも誤差逆伝播が可能になる
        u = normal.rsample()
        
        # Squashing (押し潰し): tanh関数で [-1, 1] の範囲に正規化
        action = torch.tanh(u)
        
        return action

    def sample_action_and_log_prob(self, state):
        """
        学習ループ時に使用: 行動のサンプリングと同時に、エントロピー計算のための対数確率(log_prob)を返す
        """
        mu, log_std, _ = self.forward(state)
        std = log_std.exp()
        normal = Normal(mu, std)
        
        u = normal.rsample()
        action = torch.tanh(u)
        
        # --- SAC特有の複雑な数学的処理 (tanhの微分による確率密度の補正) ---
        log_prob = normal.log_prob(u)
        # squashing (tanh) による空間の歪みを補正 (ゼロ除算防止のため 1e-6 を足す)
        log_prob -= torch.log(1 - action.pow(2) + 1e-6)
        # 次元方向に和をとる (例: 力と角度の確率をかけ合わせる = 対数では足し算)
        log_prob = log_prob.sum(1, keepdim=True)
        
        return action, log_prob


class TwinQCriticNet(nn.Module):
    def __init__(self, state_dim=12, action_dim=2, hidden_dim=64):
        super(TwinQCriticNet, self).__init__()
        
        # --- Q1 ネットワーク ---
        # 状態(State)と行動(Action)を結合して入力とし、Q値(価値)を出力する
        self.q1_net = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
        # --- Q2 ネットワーク (双子) ---
        # まったく同じ構造を独立してもう一つ用意する
        self.q2_net = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, state, action):
        """
        推論: StateとActionのペアに対する2つの独立したQ値を計算
        """
        # 状態と行動のテンソルを結合 (dim=1 はバッチの次元を維持したまま横に繋ぐ)
        sa = torch.cat([state, action], dim=1)
        
        q1 = self.q1_net(sa)
        q2 = self.q2_net(sa)
        
        return q1, q2

import random
from collections import deque

class ReplayBuffer:
    def __init__(self, capacity=100000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        """シミュレータでの経験を1ステップ分保存する"""
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size, device):
        """学習用にランダムなバッチを取り出し、テンソルに変換する"""
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(np.stack, zip(*batch))
        
        return (
            torch.FloatTensor(state).to(device),
            torch.FloatTensor(action).to(device),
            torch.FloatTensor(reward).unsqueeze(1).to(device),
            torch.FloatTensor(next_state).to(device),
            torch.FloatTensor(done).unsqueeze(1).to(device)
        )

    def __len__(self):
        return len(self.buffer)

import torch.optim as optim
import torch.nn.functional as F

def test_sac_update_step():
    print("--- SAC 学習(Update)ステップのテスト開始 ---")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # ネットワークとオプティマイザの初期化
    actor = MuZeroSACPolicyNet().to(device)
    critic = TwinQCriticNet().to(device)
    actor_optimizer = optim.Adam(actor.parameters(), lr=3e-4)
    critic_optimizer = optim.Adam(critic.parameters(), lr=3e-4)
    
    # エントロピー係数 α (探索と活用のバランス。本来は自動調整するが今回は固定値)
    alpha = 0.01
    gamma = 0.99 # 割引率
    
    # --- ダミーの経験バッチ(Replay Bufferから取り出したと仮定) ---
    batch_size = 4
    state = torch.randn(batch_size, 6).to(device)
    action = torch.clamp(torch.randn(batch_size, 2), min=-1, max=1).to(device)
    reward = torch.randn(batch_size, 1).to(device)
    next_state = torch.randn(batch_size, 6).to(device)
    done = torch.zeros(batch_size, 1).to(device)
    
    # ==========================================
    # 1. Critic (Qネットワーク) の学習
    # ==========================================
    with torch.no_grad():
        # 次の状態での最適な行動と、その時の対数確率を計算
        next_action, next_log_prob = actor.sample_action_and_log_prob(next_state)
        # 次の状態のQ値を計算
        next_q1, next_q2 = critic(next_state, next_action)
        # 悲観的評価 (低い方を採用) し、エントロピー項を引く
        next_q_target = torch.min(next_q1, next_q2) - alpha * next_log_prob
        # ベルマン方程式による目標Q値の計算
        expected_q = reward + (1 - done) * gamma * next_q_target

    # 現在のQ値を計算
    curr_q1, curr_q2 = critic(state, action)
    # MSE損失の計算
    critic_loss = F.mse_loss(curr_q1, expected_q) + F.mse_loss(curr_q2, expected_q)

    critic_optimizer.zero_grad()
    critic_loss.backward()
    critic_optimizer.step()
    
    # ==========================================
    # 2. Actor (Policyネットワーク) の学習
    # ==========================================
    # 現在の状態でActorが新しく行動をサンプリング
    new_action, log_prob = actor.sample_action_and_log_prob(state)
    q1_new, q2_new = critic(state, new_action)
    q_new = torch.min(q1_new, q2_new)
    
    # Actorの損失関数: Q値を最大化(マイナスをつけて最小化)しつつ、エントロピーも最大化する
    actor_loss = (alpha * log_prob - q_new).mean()

    actor_optimizer.zero_grad()
    actor_loss.backward()
    actor_optimizer.step()
    
    print("\n[学習ステップ実行結果]")
    print(f"Critic Loss (Q値の予測誤差): {critic_loss.item():.4f}")
    print(f"Actor Loss (方策の更新誤差): {actor_loss.item():.4f}")
    print("\n--- テスト完了：誤差逆伝播とパラメータ更新に成功 ---")

# if __name__ == "__main__":
#     test_sac_update_step()


def plot_metrics(rewards, actor_losses, critic_losses, save_dir="results"):
    """学習結果を英語表記のグラフとして出力し保存する関数"""
    os.makedirs(save_dir, exist_ok=True)
    fig, axs = plt.subplots(2, 1, figsize=(10, 10))

    # 1. Reward Plot
    axs[0].plot(rewards, alpha=0.3, color='gray', label='Raw Reward')
    # 移動平均 (Moving Average) の計算とプロット
    window = min(100, len(rewards))
    if window > 0:
        rolling_avg = np.convolve(rewards, np.ones(window)/window, mode='valid')
        axs[0].plot(range(window-1, len(rewards)), rolling_avg, color='blue', label=f'Moving Average ({window} eps)')
    
    axs[0].set_title('Episode Reward over Time')
    axs[0].set_xlabel('Episode')
    axs[0].set_ylabel('Reward')
    axs[0].legend()
    axs[0].grid(True)

    # 2. Loss Plot
    axs[1].plot(critic_losses, alpha=0.8, color='red', label='Critic Loss')
    axs[1].plot(actor_losses, alpha=0.8, color='green', label='Actor Loss')
    axs[1].set_title('Training Losses over Time')
    axs[1].set_xlabel('Episode')
    axs[1].set_ylabel('Loss')
    axs[1].legend()
    axs[1].grid(True)

    plt.tight_layout()
    plot_path = os.path.join(save_dir, 'training_metrics.png')
    plt.savefig(plot_path)
    plt.close()
    print(f"Saved training metrics plot to: {plot_path}")

def train_sac_real_env():
    print("\n--- Starting SAC Training Loop on Real Environment (FastFiz) ---")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    env = MinimalBilliardEnv()
    actor = MuZeroSACPolicyNet(state_dim=12, action_dim=2, hidden_dim=256).to(device)
    critic = TwinQCriticNet(state_dim=12, action_dim=2, hidden_dim=256).to(device)
    
    # ターゲットネットワーク
    critic_target = copy.deepcopy(critic).to(device)
    critic_target.eval()
    
    actor_optimizer = optim.Adam(actor.parameters(), lr=3e-4)
    critic_optimizer = optim.Adam(critic.parameters(), lr=3e-4)
    
    # ==========================================
    # α（エントロピー係数）の自動調整用セットアップ
    # ==========================================
    target_entropy = -2.0  # action_dimが2次元なので -2.0
    # αは正の値である必要があるため、対数空間(log_alpha)で学習させる
    log_alpha = torch.zeros(1, requires_grad=True, device=device)
    alpha_optimizer = optim.Adam([log_alpha], lr=3e-4)
    
    # ループ開始前の初期値（exp(0) = 1.0）
    alpha = log_alpha.exp().item() 
    
    replay_buffer = ReplayBuffer(capacity=100000) 
    
    batch_size = 64
    max_episodes = 100000
    gamma = 0.99
    tau = 0.005
    
    history_reward = []
    history_actor_loss = []
    history_critic_loss = []
    
    save_dir = "results"
    os.makedirs(save_dir, exist_ok=True)
    
    for episode in range(max_episodes):
        state, _ = env.reset() 
        episode_reward = 0
        
        ep_actor_loss = 0.0
        ep_critic_loss = 0.0
        update_steps = 0
        
        done = False
        while not done:
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
            with torch.no_grad():
                action_tensor = actor.sample_action(state_tensor)
            
            action = action_tensor.cpu().numpy()[0]
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            replay_buffer.push(state, action, reward, next_state, done)
            state = next_state
            episode_reward += reward
            
            if len(replay_buffer) > batch_size:
                b_state, b_action, b_reward, b_next_state, b_done = replay_buffer.sample(batch_size, device)
                
                # ==========================================
                # 1. Critic Update
                # ==========================================
                with torch.no_grad():
                    next_action, next_log_prob = actor.sample_action_and_log_prob(b_next_state)
                    # 次のQ値（目標値）の計算には、ブレないターゲットネットワークを使う
                    next_q1, next_q2 = critic_target(b_next_state, next_action)
                    next_q_target = torch.min(next_q1, next_q2) - alpha * next_log_prob
                    expected_q = b_reward + (1 - b_done) * gamma * next_q_target

                curr_q1, curr_q2 = critic(b_state, b_action)
                critic_loss = F.mse_loss(curr_q1, expected_q) + F.mse_loss(curr_q2, expected_q)

                critic_optimizer.zero_grad()
                critic_loss.backward()
                critic_optimizer.step()
                
                # ==========================================
                # 2. Actor Update
                # ==========================================
                # 計算効率化：Actor更新中はCriticの勾配計算をフリーズする
                for p in critic.parameters():
                    p.requires_grad = False
                    
                new_action, log_prob = actor.sample_action_and_log_prob(b_state)
                q1_new, q2_new = critic(b_state, new_action)
                q_new = torch.min(q1_new, q2_new)
                actor_loss = (alpha * log_prob - q_new).mean()

                actor_optimizer.zero_grad()
                actor_loss.backward()
                actor_optimizer.step()
                
                # フリーズ解除
                for p in critic.parameters():
                    p.requires_grad = True

                # ==========================================
                # 3. Alpha Update (Auto-tuning)
                # ==========================================
                # 現在の方策の確率密度(log_prob)と目標値(target_entropy)の差分からLossを計算
                alpha_loss = -(log_alpha * (log_prob + target_entropy).detach()).mean()

                alpha_optimizer.zero_grad()
                alpha_loss.backward()
                alpha_optimizer.step()
                
                with torch.no_grad():
                    # log_alpha の下限を ln(0.05) ≈ -2.99 に制限する
                    log_alpha.copy_(torch.clamp(log_alpha, min=np.log(0.01)))

                # 次のステップ用に、更新された tensor から実数値を取り出す
                alpha = log_alpha.exp().item()
                
                # ==========================================
                # 4. Target Network Update
                # ==========================================
                for target_param, param in zip(critic_target.parameters(), critic.parameters()):
                    target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)
                
                ep_critic_loss += critic_loss.item()
                ep_actor_loss += actor_loss.item()
                update_steps += 1
                
        history_reward.append(episode_reward)
        if update_steps > 0:
            history_actor_loss.append(ep_actor_loss / update_steps)
            history_critic_loss.append(ep_critic_loss / update_steps)
        else:
            history_actor_loss.append(0.0)
            history_critic_loss.append(0.0)
                
        if (episode + 1) % 100 == 0:
            avg_rew = np.mean(history_reward[-100:])
            # 【追加】Alphaの推移を確認できるようにprint文を拡張
            print(f"Episode {episode + 1}/{max_episodes} | Avg Reward (last 100): {avg_rew:.4f} | Buffer: {len(replay_buffer)} | Alpha: {alpha:.4f}")
            
        if (episode + 1) % 1000 == 0:
            torch.save(actor.state_dict(), os.path.join(save_dir, f'sac_actor_ep{episode+1}.pth'))
            plot_metrics(history_reward, history_actor_loss, history_critic_loss, save_dir)

    print("--- Training Loop Completed ---")
    torch.save(actor.state_dict(), os.path.join(save_dir, 'sac_actor_final.pth'))
    torch.save(critic.state_dict(), os.path.join(save_dir, 'sac_critic_final.pth'))
    plot_metrics(history_reward, history_actor_loss, history_critic_loss, save_dir)
    print("Models and final plots have been saved in the 'results' directory.")
    
if __name__ == "__main__":
    train_sac_real_env()

# 【確認用テストブロック】
def test_architecture():
    print("--- ネットワークアーキテクチャのテスト開始 ---")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MuZeroSACPolicyNet(state_dim=12, action_dim=2, hidden_dim=256).to(device)
    
    # ダミーの入力データ (バッチサイズ1, state_dim=6)
    # [cue_x, cue_y, obj0_x, obj0_y, obj1_x, obj1_y] を想定
    dummy_state = torch.tensor([[0.5, -0.5, 0.1, 0.2, -0.8, 0.3, 0.9, 0.8]], dtype=torch.float32).to(device)
    
    print(f"入力Stateの形状: {dummy_state.shape}")
    
    # 1. Forward処理のテスト (MuZero用)
    mu, log_std, value = model(dummy_state)
    print("\n[Forward推論結果]")
    print(f"平均 μ (mu): {mu.detach().cpu().numpy()}")
    print(f"対数標準偏差 (log_std): {log_std.detach().cpu().numpy()}")
    print(f"状態価値 (Value): {value.detach().cpu().numpy()}")
    
    # 2. サンプリング処理のテスト (SAC用)
    action = model.sample_action(dummy_state)
    print("\n[サンプリング結果 (Squashed Action)]")
    print(f"決定された行動 (Action ∈ [-1, 1]): {action.detach().cpu().numpy()}")

    # 3. Criticネットワーク (Twin Q-Networks) のテスト
    critic = TwinQCriticNet(state_dim=12, action_dim=2, hidden_dim=256).to(device)
    
    # Policyが決定した行動(action)と状態(dummy_state)をCriticに渡す
    q1, q2 = critic(dummy_state, action)
    
    print("\n[Critic推論結果 (Twin Q-Networks)]")
    print(f"Q1 評価値: {q1.detach().cpu().numpy()}")
    print(f"Q2 評価値: {q2.detach().cpu().numpy()}")
    print(f"採用されるQ値 (min(Q1, Q2)): {torch.min(q1, q2).detach().cpu().numpy()}")
    
    print("\n--- テスト完了：勾配計算可能な確率的出力モデルの構築に成功 ---")
    

# if __name__ == "__main__":
#     test_architecture()