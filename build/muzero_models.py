import torch
import torch.nn as nn
import torch.nn.functional as F

# ============================================================================
# 1. 設定の一元管理 (将来の拡張を容易にする Config クラス)
# ============================================================================
class MuZeroConfig:
    def __init__(self, obs_dim=12, action_dim=2):
        self.obs_dim = obs_dim       # 環境に合わせて動的に変わる入力次元
        self.action_dim = action_dim # 角度・パワーなど
        
        # --- 脳内のネットワーク次元（ここは環境が変わっても固定でOK） ---
        self.hidden_dim = 256        # 中間層の広さ
        self.latent_dim = 128        # 潜在状態（AIの脳内表現）の次元数
        
        # 連続行動空間用のパラメータ
        self.log_std_min = -20
        self.log_std_max = 2

# ============================================================================
# 2. Representation Network (現実 -> 脳内 への翻訳)
# ============================================================================
class RepresentationNetwork(nn.Module):
    def __init__(self, config: MuZeroConfig):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(config.obs_dim, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, config.latent_dim)
        )
        # 潜在状態が計算過程で発散しないように正規化するための層
        self.norm = nn.LayerNorm(config.latent_dim)

    def forward(self, obs):
        latent_state = self.net(obs)
        # MCTSで何回もループさせるため、状態ベクトルを正規化して安定させる
        return self.norm(latent_state)

# ============================================================================
# 3. Dynamics Network (脳内シミュレータ：状態遷移モデル)
# ============================================================================
class DynamicsNetwork(nn.Module):
    def __init__(self, config: MuZeroConfig):
        super().__init__()
        
        # 入力は「現在の潜在状態」と「行動」の結合
        input_dim = config.latent_dim + config.action_dim
        
        # [A] 次の潜在状態を予測するネットワーク
        self.next_state_net = nn.Sequential(
            nn.Linear(input_dim, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, config.latent_dim)
        )
        self.state_norm = nn.LayerNorm(config.latent_dim)
        
        # [B] その行動による即時報酬を予測するネットワーク
        self.reward_net = nn.Sequential(
            nn.Linear(input_dim, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, 1) # 報酬はスカラー値(1次元)
        )

    def forward(self, latent_state, action):
        # 状態と行動を結合
        x = torch.cat([latent_state, action], dim=-1)
        
        # 次の状態と報酬を予測
        next_latent = self.next_state_net(x)
        next_latent = self.state_norm(next_latent)
        predicted_reward = self.reward_net(x)
        
        return next_latent, predicted_reward

# ============================================================================
# 4. Prediction Network (脳内状態の評価：Actor / Criticの役割)
# ============================================================================
class PredictionNetwork(nn.Module):
    def __init__(self, config: MuZeroConfig):
        super().__init__()
        self.config = config
        
        # [A] Actor (方針の出力：連続値対応版)
        self.policy_net = nn.Sequential(
            nn.Linear(config.latent_dim, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim),
            nn.ReLU()
        )
        self.policy_mu = nn.Linear(config.hidden_dim, config.action_dim)
        self.policy_log_std = nn.Linear(config.hidden_dim, config.action_dim)
        
        # [B] Critic (状態価値の出力)
        self.value_net = nn.Sequential(
            nn.Linear(config.latent_dim, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, 1) # 価値はスカラー値(1次元)
        )

    def forward(self, latent_state):
        # 価値の予測
        value = self.value_net(latent_state)
        
        # 行動（方針）の予測
        policy_hidden = self.policy_net(latent_state)
        mu = self.policy_mu(policy_hidden)
        log_std = self.policy_log_std(policy_hidden)
        
        # log_std を安全な範囲にクリップ
        log_std = torch.clamp(log_std, self.config.log_std_min, self.config.log_std_max)
        
        return mu, log_std, value

def compute_muzero_loss(rep_net, dyn_net, pred_net, batch_data, config):
    """
    MuZeroの3つのLoss（価値、方策、報酬）を計算し、合計Lossとデバッグ用辞書を返す。
    
    batch_data は以下を含む辞書を想定（系列長を K とする）:
      - 'obs': [batch_size, obs_dim] (最初の現実の盤面, k=0)
      - 'actions': [batch_size, K, action_dim] (実際に打った行動の系列)
      - 'target_rewards': [batch_size, K] (実際に得た報酬の系列)
      - 'target_values': [batch_size, K+1] (MCTS等で算出した未来の価値の系列)
      - 'target_policies': [batch_size, K+1, action_dim] (MCTSが最適だと考えた行動の系列)
    """
    # データを取得
    obs = batch_data['obs']
    actions = batch_data['actions']
    target_rewards = batch_data['target_rewards']
    target_values = batch_data['target_values']
    target_policies = batch_data['target_policies']
    
    batch_size = obs.size(0)
    K = actions.size(1) # 未来を何ステップ先まで脳内でシミュレーションするか
    
    total_loss = 0.0
    
    # 異常検知・モニタリング用の辞書
    loss_info = {
        "value_loss": 0.0,
        "policy_loss": 0.0,
        "reward_loss": 0.0
    }
    
    # 1. まず現実の盤面を脳内に取り込む (k=0)
    latent_state = rep_net(obs)
    
    # 2. 現在から未来(Kステップ)に向かって脳内シミュレーションを展開
    for k in range(K + 1):
        # [A] 現在の脳内状態に対する価値と方針を予測
        mu, log_std, pred_value = pred_net(latent_state)
        
        # --- Value Loss (価値の予測誤差) ---
        # 予測した価値が、実際の未来の価値(target_values)とどれくらいズレているか
        v_loss = F.smooth_l1_loss(pred_value.squeeze(-1), target_values[:, k])
        loss_info["value_loss"] += v_loss.item()
        total_loss += v_loss
        
        # --- Policy Loss (方策の予測誤差) ---
        # AIの直感(mu)が、MCTSがじっくり考えて出した最善手(target_policies)にどれだけ近いか
        p_loss = F.smooth_l1_loss(mu, target_policies[:, k])
        loss_info["policy_loss"] += p_loss.item()
        total_loss += p_loss
        
        # [B] まだ未来(k < K)が残っているなら、行動を入力して1歩先の未来を予測
        if k < K:
            # 実際にとった行動を脳内シミュレータに入力
            latent_state, pred_reward = dyn_net(latent_state, actions[:, k])
            
            # --- Reward Loss (報酬の予測誤差) ---
            # 予測した報酬が、実際に環境からもらった報酬とどれくらいズレているか
            r_loss = F.smooth_l1_loss(pred_reward.squeeze(-1), target_rewards[:, k])
            loss_info["reward_loss"] += r_loss.item()
            total_loss += r_loss
            
    # 各Lossをステップ数で割って平均化（Kを増やしてもLossが爆発しないようにする）
    total_loss = total_loss / (K + 1)
    loss_info = {k: v / (K + 1) for k, v in loss_info.items()}
    
    return total_loss, loss_info

# ============================================================================
# テスト実行コード (ダミーデータでLoss計算と分解を確認)
# ============================================================================
if __name__ == "__main__":
    # 前回のモデル定義がメモリにある前提で動きます（必要に応じてimportしてください）
    config = MuZeroConfig(obs_dim=12, action_dim=2)
    rep_net = RepresentationNetwork(config)
    dyn_net = DynamicsNetwork(config)
    pred_net = PredictionNetwork(config)
    
    batch_size = 4
    K_steps = 3 # 3手先までシミュレーションして学習する設定
    
    # ダミーの経験バッファデータを作成
    dummy_batch = {
        'obs': torch.randn(batch_size, config.obs_dim),
        'actions': torch.randn(batch_size, K_steps, config.action_dim),
        'target_rewards': torch.randn(batch_size, K_steps),
        'target_values': torch.randn(batch_size, K_steps + 1),
        'target_policies': torch.randn(batch_size, K_steps + 1, config.action_dim)
    }
    
    # Lossの計算
    loss, info = compute_muzero_loss(rep_net, dyn_net, pred_net, dummy_batch, config)
    
    print("=== MuZero Loss 計算結果 ===")
    print(f"Total Loss (逆伝播用): {loss.item():.4f}\n")
    print("--- 内訳 (モニタリング用) ---")
    for key, val in info.items():
        print(f"{key:>12}: {val:.4f}")
        
    # これで一括でネットワークを更新できる
    # optimizer.zero_grad()
    # loss.backward()
    # optimizer.step()