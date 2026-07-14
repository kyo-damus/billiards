import gymnasium as gym
from gymnasium import spaces
import numpy as np
import random
import billiard_env_cpp

class MinimalBilliardEnv(gym.Env):
    def __init__(self):
        super().__init__()
        self.sim = billiard_env_cpp.BilliardSimulator()
        
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(2,), dtype=np.float32
        )
        self.observation_space = spaces.Box(
            low=-2.0, high=2.0, shape=(6,), dtype=np.float32 
        )
        
        self.pocket_pos = np.array([1.0, 1.0])
        self.pocket_radius = 0.05

    def load_dummy_dataset(self):
        """【修正1】球が重ならないように（Overlap回避）初期配置を生成"""
        while True:
            cue = np.array([random.uniform(-1.0, 1.0), random.uniform(-1.0, 1.0)])
            obj0 = np.array([random.uniform(0.0, 1.0), random.uniform(0.0, 1.0)])
            obj1 = np.array([random.uniform(-1.0, 0.0), random.uniform(-1.0, 0.0)])
            
            # 球の直径（2 * 約0.028）以上の距離が互いに空いているかチェック
            if np.linalg.norm(cue - obj0) > 0.06 and \
               np.linalg.norm(cue - obj1) > 0.06 and \
               np.linalg.norm(obj0 - obj1) > 0.06:
                return [cue[0], cue[1], obj0[0], obj0[1], obj1[0], obj1[1]]

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        init_state = self.load_dummy_dataset()
        self.sim.set_state(init_state)
        return np.array(init_state, dtype=np.float32), {}

    def step(self, action):
        target = 0
        power = (action[0] + 1.0) / 2.0  # [-1, 1] -> [0.0, 1.0]
        
        # 【修正2】角度の相対化 (Method Refinement)
        prev_obs = self.get_state()
        cue_x, cue_y = prev_obs[0], prev_obs[1]
        obj0_x, obj0_y = prev_obs[2], prev_obs[3]
        
        # 的球0への直線角度（基準角）を計算
        ideal_angle = np.degrees(np.arctan2(obj0_y - cue_y, obj0_x - cue_x))
        
        # ネットワークの出力 [-1, 1] を「基準角 ± 30度」に変換して物理エンジンへ
        angle = ideal_angle + action[1] * 30.0 
        
        # 2. シミュレータの実行
        result = self.sim.step(target, power, angle)
        obs = np.array(result[0:6], dtype=np.float32)
        
        # 3. 独自報酬の計算
        cue_pos = np.array([obs[0], obs[1]])
        obj0_pos = np.array([obs[2], obs[3]])
        prev_obj0_pos = np.array([prev_obs[2], prev_obs[3]])
        
        movement = np.linalg.norm(obj0_pos - prev_obj0_pos)
        dist_to_pocket = np.linalg.norm(self.pocket_pos - obj0_pos)
        
        reward = 0.0
        terminated = False
        
        if movement < 0.001:
            reward = -1.0 # 空振り
            terminated = True 
        else:
            # 接近報酬
            alpha = 2.0 
            reward = float(np.exp(-alpha * dist_to_pocket))
            
            if dist_to_pocket < self.pocket_radius:
                reward += 10.0 # ポケットイン成功
                terminated = True
                
        if np.linalg.norm(self.pocket_pos - cue_pos) < self.pocket_radius:
            reward -= 10.0 # スクラッチペナルティ
            terminated = True
            
        return obs, reward, terminated, False, {}

    def get_state(self):
        return self.sim.get_state()

    def set_state(self, state):
        self.sim.set_state(state)