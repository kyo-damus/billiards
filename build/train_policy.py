import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from custom_env import MinimalBilliardEnv

# 【1】PyTorchモデルの定義 (SAC + MuZero 統合型アーキテクチャ)
class MuZeroSACPolicyNet(nn.Module):
    def __init__(self, state_dim=6, action_dim=2, hidden_dim=64):
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
    def __init__(self, state_dim=6, action_dim=2, hidden_dim=64):
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
    alpha = 0.2 
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

if __name__ == "__main__":
    test_sac_update_step()

# ※環境(custom_env)のインポートが必要になります
# from custom_env import CustomEnv 

def train_sac_real_env():
    print("\n--- FastFiz(実環境)でのSAC学習ループ開始 ---")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. 環境とネットワークの初期化
    env = MinimalBilliardEnv() # C++と繋がる実際の環境を初期化
    actor = MuZeroSACPolicyNet(state_dim=6, action_dim=2).to(device)
    critic = TwinQCriticNet(state_dim=6, action_dim=2).to(device)
    
    actor_optimizer = optim.Adam(actor.parameters(), lr=3e-4)
    critic_optimizer = optim.Adam(critic.parameters(), lr=3e-4)
    
    replay_buffer = ReplayBuffer(capacity=50000)
    
    batch_size = 64
    max_episodes = 500
    alpha = 0.2
    gamma = 0.99
    
    for episode in range(max_episodes):
        # 環境の初期化 (Gymnasiumの仕様に合わせて obs と info を受け取る)
        state, _ = env.reset() 
        episode_reward = 0
        
        done = False
        while not done:
            # --- 実行フェーズ ---
            # 状態をテンソル化してActorに行動を決めさせる
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
            with torch.no_grad():
                action_tensor = actor.sample_action(state_tensor)
            
            # GPUテンソルからNumpy配列(1D)に変換して環境に渡す
            action = action_tensor.cpu().numpy()[0]
            
            # 物理エンジンに行動を渡し、結果を受け取る
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            # 経験をバッファに保存
            replay_buffer.push(state, action, reward, next_state, done)
            state = next_state
            episode_reward += reward
            
            # --- 学習フェーズ ---
            if len(replay_buffer) > batch_size:
                b_state, b_action, b_reward, b_next_state, b_done = replay_buffer.sample(batch_size, device)
                
                # Criticの更新
                with torch.no_grad():
                    next_action, next_log_prob = actor.sample_action_and_log_prob(b_next_state)
                    next_q1, next_q2 = critic(b_next_state, next_action)
                    next_q_target = torch.min(next_q1, next_q2) - alpha * next_log_prob
                    expected_q = b_reward + (1 - b_done) * gamma * next_q_target

                curr_q1, curr_q2 = critic(b_state, b_action)
                critic_loss = F.mse_loss(curr_q1, expected_q) + F.mse_loss(curr_q2, expected_q)

                critic_optimizer.zero_grad()
                critic_loss.backward()
                critic_optimizer.step()
                
                # Actorの更新
                new_action, log_prob = actor.sample_action_and_log_prob(b_state)
                q1_new, q2_new = critic(b_state, new_action)
                q_new = torch.min(q1_new, q2_new)
                actor_loss = (alpha * log_prob - q_new).mean()

                actor_optimizer.zero_grad()
                actor_loss.backward()
                actor_optimizer.step()
                
        if (episode + 1) % 10 == 0:
            print(f"Episode {episode + 1} | 獲得報酬: {episode_reward:.4f} | バッファサイズ: {len(replay_buffer)}")

    print("--- 実環境での学習ループ完了 ---")

if __name__ == "__main__":
    train_sac_real_env()

# 【確認用テストブロック】
def test_architecture():
    print("--- ネットワークアーキテクチャのテスト開始 ---")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MuZeroSACPolicyNet(state_dim=6, action_dim=2, hidden_dim=64).to(device)
    
    # ダミーの入力データ (バッチサイズ1, state_dim=6)
    # [cue_x, cue_y, obj0_x, obj0_y, obj1_x, obj1_y] を想定
    dummy_state = torch.tensor([[0.5, -0.5, 0.1, 0.2, -0.8, 0.3]], dtype=torch.float32).to(device)
    
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
    critic = TwinQCriticNet(state_dim=6, action_dim=2, hidden_dim=64).to(device)
    
    # Policyが決定した行動(action)と状態(dummy_state)をCriticに渡す
    q1, q2 = critic(dummy_state, action)
    
    print("\n[Critic推論結果 (Twin Q-Networks)]")
    print(f"Q1 評価値: {q1.detach().cpu().numpy()}")
    print(f"Q2 評価値: {q2.detach().cpu().numpy()}")
    print(f"採用されるQ値 (min(Q1, Q2)): {torch.min(q1, q2).detach().cpu().numpy()}")
    
    print("\n--- テスト完了：勾配計算可能な確率的出力モデルの構築に成功 ---")
    

if __name__ == "__main__":
    test_architecture()