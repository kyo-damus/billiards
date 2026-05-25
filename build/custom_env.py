import gymnasium as gym
from gymnasium import spaces
import numpy as np
import random
import billiard_env_cpp

class MinimalBilliardEnv(gym.Env):
    def __init__(self):
        super().__init__()
        self.sim = billiard_env_cpp.BilliardSimulator()
        
        # 行動: [強さ(0~1), 角度(-180~180)]
        self.action_space = spaces.Box(
            low=np.array([0.0, -180.0], dtype=np.float32),
            high=np.array([1.0, 180.0], dtype=np.float32),
            dtype=np.float32
        )
        # 状態: [手球x, 手球y, 的球x, 的球y]
        self.observation_space = spaces.Box(
            low=-2.0, high=2.0, shape=(4,), dtype=np.float32 
        )

    # 【セクション1】ダミーデータセットの読み込み
    def load_dummy_dataset(self):
        """的球を2個に増やす (合計6次元)"""
        return [
            random.uniform(-1.0, 1.0), # 手球x
            random.uniform(-1.0, 1.0), # 手球y
            random.uniform(0.0, 1.0),  # 的球0x
            random.uniform(0.0, 1.0),  # 的球0y
            random.uniform(-1.0, 0.0), # 的球1x
            random.uniform(-1.0, 0.0)  # 的球1y
        ]

    # 【セクション2】環境の初期化
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # データセットから初期配置を取得してC++にセット
        init_state = self.load_dummy_dataset()
        self.sim.set_state(init_state)
        return np.array(init_state, dtype=np.float32), {}

    # 【セクション3】ステップ実行
    def step(self, action):
        # action[0]=ターゲット(0 or 1), action[1]=パワー, action[2]=角度
        target, power, angle = int(action[0]), action[1], action[2]
        
        result = self.sim.step(target, power, angle)
        
        obs = np.array(result[0:6], dtype=np.float32) # 6次元に拡張
        reward = result[6]
        terminated = bool(result[7])
        
        return obs, reward, terminated, False, {}

    # 【セクション4】MCTS用の特殊関数（重要）
    def get_state(self):
        """現在の状態を保存用に取り出す"""
        return self.sim.get_state()

    def set_state(self, state):
        """保存した状態を復元する"""
        self.sim.set_state(state)

        