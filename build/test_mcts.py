import numpy as np
from custom_env import MinimalBilliardEnv
import copy

def dummy_policy_network(env_state, target_ball):
    """
    【AIの直感のモックアップ】
    盤面を見て「大体この角度が怪しい」という事前確率（中心となる角度）を返す。
    本来はニューラルネットワークが担当する部分を、今回は幾何学計算で代用します。
    """
    cue_x, cue_y = env_state[0], env_state[1]
    # target_ballが0なら[2][3]、1なら[4][5]の座標を取得
    obj_x, obj_y = env_state[2 + target_ball*2], env_state[3 + target_ball*2]
    pocket_x, pocket_y = 1.0, 1.0
    r = 0.05
    
    # 簡易的なゴーストボールの逆算
    dx, dy = pocket_x - obj_x, pocket_y - obj_y
    dist = np.hypot(dx, dy)
    ghost_x = obj_x - (dx / dist) * (2.0 * r)
    ghost_y = obj_y - (dy / dist) * (2.0 * r)
    
    # 「この角度周辺が怪しい」という直感（mean_angle）を計算
    mean_angle = np.degrees(np.arctan2(ghost_y - cue_y, ghost_x - cue_x))
    return mean_angle

def hierarchical_mcts_search(env, num_simulations=50): 
    # ↑ 探索回数をあえて「50回」という少ない回数に設定します
    original_state = env.get_state()
    best_action = None
    best_reward = -1.0

    print(f"--- 賢い階層型 MCTS開始 (探索回数: {num_simulations}回/ターゲット) ---")
    
    for target_ball in [0, 1]:
        print(f"  >> マクロ戦略: [的球 {target_ball}] を評価します...")
        
        # 1. Policy（直感）に盤面を見せて、怪しい角度を教えてもらう
        suggested_angle = dummy_policy_network(original_state, target_ball)
        print(f"     Policyの直感: {suggested_angle:.2f}度 付近が怪しいです")
        
        # 2. 直感の周辺だけを重點的に探索する（360度ランダムをやめる）
        for i in range(num_simulations):
            env.set_state(original_state)
            
            # 【重要】-180~180の一様分布から、直感を中心とした「正規分布（ガウス分布）」に変更
            # 標準偏差(scale)を2.0度に設定し、直感のすぐ近くを重点的に探る
            test_angle = np.random.normal(loc=suggested_angle, scale=2.0)
            
            test_action = np.array([target_ball, 0.8, test_angle], dtype=np.float32)
            obs, reward, terminated, _, _ = env.step(test_action)
            
            if reward > best_reward:
                best_reward = reward
                best_action = copy.deepcopy(test_action)
                
    print(f"--- MCTS終了: 見つかった最大報酬 = {best_reward} ---")
    env.set_state(original_state)
    return best_action


# 【セクション2】メイン実行ループ
if __name__ == "__main__":
    env = MinimalBilliardEnv()
    
    obs, info = env.reset()
    print(f"初期状態 (手球x, 手球y, 的球0x, 的球0y, 的球1x, 的球1y): \n{obs}")
    
    action = hierarchical_mcts_search(env, num_simulations=1)
    
    if action is not None:
        print(f"\nMCTSが選択した行動: Target=的球{int(action[0])}, Power={action[1]:.2f}, Angle={action[2]:.2f}度")
        next_obs, reward, terminated, _, _ = env.step(action)
        print(f"結果 -> 報酬: {reward}, 終了: {terminated}")
    else:
        print("良い行動が見つかりませんでした。")