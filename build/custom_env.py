import gymnasium as gym
from gymnasium import spaces
import numpy as np
import random
import billiard_env_cpp

class MinimalBilliardEnv(gym.Env):
    def __init__(self):
        super().__init__()
        self.sim = billiard_env_cpp.BilliardSimulator()
        
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
        # 角度(3.14)等が入るため、low/highに余裕を持たせ、12次元に変更
        self.observation_space = spaces.Box(low=-10.0, high=10.0, shape=(12,), dtype=np.float32)
        
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
        
        # 【変更】生のinit_stateではなく、相対座標に変換して返す
        raw_obs = np.array(init_state, dtype=np.float32)
        return self._get_relative_obs(raw_obs), {}

    def step(self, action):
        target = 0
        # power = (action[0] + 1.0) / 2.0  # [-1, 1] -> [0.0, 1.0]
        power = 0.4 + 0.6 * ((action[0] + 1.0) / 2.0)
        
        prev_obs = self.get_state()
        cue_x, cue_y = prev_obs[0], prev_obs[1]
        obj0_x, obj0_y = prev_obs[2], prev_obs[3]
        
        ideal_angle = np.degrees(np.arctan2(obj0_y - cue_y, obj0_x - cue_x))
        angle = ideal_angle + action[1] * 45.0 
        
        result = self.sim.step(target, power, angle)
        raw_obs = np.array(result[0:6], dtype=np.float32)
        
        cue_pos = np.array([raw_obs[0], raw_obs[1]])
        obj0_pos = np.array([raw_obs[2], raw_obs[3]])
        prev_obj0_pos = np.array([prev_obs[2], prev_obs[3]])
        
        movement = np.linalg.norm(obj0_pos - prev_obj0_pos)
        
        initial_dist = np.linalg.norm(self.pocket_pos - prev_obj0_pos)
        final_dist = np.linalg.norm(self.pocket_pos - obj0_pos)
        
        terminated = False 
        
        # ==================================================
        # 【修正】報酬ハッキングを防ぐ、厳密な密な報酬設計
        # ==================================================
        # 1. ターン消費の基本ペナルティ（ダラダラするのを防ぐ）
        reward = -1.0 
            
        if movement < 0.001:
            # 空振りはさらにペナルティ
            reward -= 1.0 
        else:
            # 2. 距離の減少量による報酬（近づいた分だけプラス）
            # ※最大でも初期距離分しか稼げないため、無限稼ぎが不可能
            dist_reduction = initial_dist - final_dist
            reward += dist_reduction * 3.0  
            
            # 3. 方向ベースの微小ボーナス（コサイン類似度）
            move_vec = obj0_pos - prev_obj0_pos
            target_vec = self.pocket_pos - prev_obj0_pos
            cos_sim = np.dot(move_vec, target_vec) / (np.linalg.norm(move_vec) * np.linalg.norm(target_vec) + 1e-8)
            
            if cos_sim > 0:
                reward += cos_sim * 0.5  # 稼ぎ防止のため、ボーナスは小さめに設定

            # 4. ポケットイン判定（ゴール）
            if final_dist < self.pocket_radius:
                reward += 10.0
                terminated = True 
                
        # 5. スクラッチペナルティ
        if np.linalg.norm(self.pocket_pos - cue_pos) < self.pocket_radius:
            reward -= 10.0
            terminated = True 

        rel_obs = self._get_relative_obs(raw_obs)
            
        return rel_obs, reward, terminated, False, {}

    # 【修正箇所2】 _get_relative_obs の角度処理を sin / cos の2次元に！
    def _get_relative_obs(self, raw_obs):
        cue_x, cue_y = raw_obs[0], raw_obs[1]
        obj0_x, obj0_y = raw_obs[2], raw_obs[3]
        obj1_x, obj1_y = raw_obs[4], raw_obs[5]
        
        dx0 = obj0_x - cue_x
        dy0 = obj0_y - cue_y
        dx1 = obj1_x - cue_x
        dy1 = obj1_y - cue_y

        pocket_dx = self.pocket_pos[0] - obj0_x
        pocket_dy = self.pocket_pos[1] - obj0_y
        
        # ① 角度の計算
        angle_cue_to_obj = np.arctan2(dy0, dx0)
        angle_obj_to_pocket = np.arctan2(pocket_dy, pocket_dx)
        
        # ② 狙うべき角度の差分
        angle_diff = angle_obj_to_pocket - angle_cue_to_obj
        
        # ==================================================
        # 【大正解の修正】角度を sin と cos の2次元に分解（断層をなくす）
        # ==================================================
        sin_diff = np.sin(angle_diff)
        cos_diff = np.cos(angle_diff)
        
        # ③ パワー調整のための距離
        dist_cue_to_obj = np.hypot(dx0, dy0)
        dist_obj_to_pocket = np.hypot(pocket_dx, pocket_dy)
        
        # 12次元の配列として返す（angle_diff を sin_diff, cos_diff の2つに置き換え）
        return np.array([
            cue_x, cue_y, dx0, dy0, dx1, dy1, pocket_dx, pocket_dy,
            sin_diff, cos_diff, dist_cue_to_obj, dist_obj_to_pocket
        ], dtype=np.float32)

    def get_state(self):
        return self.sim.get_state()

    def set_state(self, state):
        self.sim.set_state(state)