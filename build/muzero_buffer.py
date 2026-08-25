import numpy as np
import torch
import random

class Episode:
    """
    1回ゲームをプレイした最初から最後までの軌跡（ビデオテープ）を保存するクラス
    """
    def __init__(self):
        self.obs = []               # 盤面の状態 (12次元)
        self.actions = []           # 実際に打った行動
        self.rewards = []           # もらった報酬
        self.target_values = []     # MCTSが計算したその盤面の価値
        self.target_policies = []   # MCTSが計算した最適手（方策）

    def __len__(self):
        # このエピソードが何ステップで終わったかを返す
        return len(self.actions)

class MuZeroBuffer:
    def __init__(self, config, max_episodes=1000):
        self.config = config
        self.max_episodes = max_episodes
        self.buffer = [] # Episodeオブジェクトを格納するリスト

    def store_episode(self, episode: Episode):
        """
        ゲームが1回終わるごとに、そのエピソードを丸ごと保存する
        """
        self.buffer.append(episode)
        # 容量を超えたら一番古いエピソードを捨てる (FIFO)
        if len(self.buffer) > self.max_episodes:
            self.buffer.pop(0)

    def sample_batch(self, batch_size, K_steps):
        """
        バッファから、Kステップ先の未来までの「ビデオクリップ」を切り出してバッチを作成する
        """
        batch_obs = []
        batch_actions = []
        batch_rewards = []
        batch_target_values = []
        batch_target_policies = []

        for _ in range(batch_size):
            # 1. バッファからランダムにエピソード（ビデオテープ）を1本選ぶ
            ep = random.choice(self.buffer)
            ep_length = len(ep)
            
            # 2. そのエピソードの中で、ランダムな開始地点(t)を決める
            start_t = random.randint(0, ep_length - 1)
            
            # [A] 最初の現実の盤面（k=0）だけはそのまま取得
            batch_obs.append(ep.obs[start_t])
            
            # [B] Kステップ先までの行動・報酬・価値・方策を切り出す
            actions = []
            rewards = []
            target_values = []
            target_policies = []
            
            for k in range(K_steps + 1):
                t = start_t + k
                
                # --- 価値(Value)と方策(Policy) ---
                # 価値と方策は k=0(現在) から k=K(K手先) までの K+1 個必要
                if t < ep_length:
                    target_values.append(ep.target_values[t])
                    target_policies.append(ep.target_policies[t])
                else:
                    # 【重要: パディング】ゲームが既に終了している未来の場合
                    # 価値は「0」、方策は「ゼロベクトル」として扱う
                    target_values.append(0.0)
                    target_policies.append(np.zeros(self.config.action_dim))
                    
                # --- 行動(Action)と報酬(Reward) ---
                # 脳内シミュレータへの入力と報酬確認のため、k=0 ~ K-1 までの K 個必要
                if k < K_steps:
                    if t < ep_length:
                        actions.append(ep.actions[t])
                        rewards.append(ep.rewards[t])
                    else:
                        # 終了後の行動と報酬も「0」埋め
                        actions.append(np.zeros(self.config.action_dim))
                        rewards.append(0.0)
                        
            batch_actions.append(actions)
            batch_rewards.append(rewards)
            batch_target_values.append(target_values)
            batch_target_policies.append(target_policies)
            
        # 3. 先ほど作ったLoss関数にそのまま渡せるように、PyTorchのTensor辞書に変換
        return {
            'obs': torch.tensor(np.array(batch_obs), dtype=torch.float32),
            'actions': torch.tensor(np.array(batch_actions), dtype=torch.float32),
            'target_rewards': torch.tensor(np.array(batch_rewards), dtype=torch.float32),
            'target_values': torch.tensor(np.array(batch_target_values), dtype=torch.float32),
            'target_policies': torch.tensor(np.array(batch_target_policies), dtype=torch.float32)
        }

# ============================================================================
# テスト実行コード (ダミーエピソードを作ってサンプリングしてみる)
# ============================================================================
if __name__ == "__main__":
    from muzero_models import MuZeroConfig # 先ほどのファイルからConfigだけ借用
    
    config = MuZeroConfig(obs_dim=12, action_dim=2)
    buffer = MuZeroBuffer(config, max_episodes=10)
    
    # ダミーのエピソード（長さ5ステップ）を作って保存してみる
    ep = Episode()
    for _ in range(5):
        ep.obs.append(np.random.randn(12))
        ep.actions.append(np.random.randn(2))
        ep.rewards.append(1.0)
        ep.target_values.append(0.5)
        ep.target_policies.append(np.random.randn(2))
        
    buffer.store_episode(ep)
    
    # バッチサイズ2, K=3ステップ先までサンプリング
    batch = buffer.sample_batch(batch_size=2, K_steps=3)
    
    print("=== MuZero Buffer サンプリング結果 ===")
    for key, tensor in batch.items():
        print(f"{key:>15}: {tensor.shape}")
        