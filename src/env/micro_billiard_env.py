import gymnasium as gym
from gymnasium import spaces
import numpy as np

from src.common.types import (
    GameState,
    MacroAction,
    Strategy,
)

from src.env.fastfiz_simulator import FastFizSimulator

from src.env.shot_geometry import (
    ghost_ball_position,
    is_direct_shot_feasible,
)

from src.env.simulator_state import (
    SimulatorSnapshot,
)

from src.common.table import (
    TABLE_WIDTH,
    TABLE_LENGTH,
    POCKET_POSITIONS,
)


class MicroBilliardEnv(gym.Env):
    """
    Macroが指定した目標に対して、
    Micro action（power, angle）を学習する環境。

    現段階では ATTACK のみ対応。
    """

    metadata = {}

    TABLE_WIDTH = TABLE_WIDTH
    TABLE_LENGTH = TABLE_LENGTH

    POCKET_POSITIONS = POCKET_POSITIONS

    MIN_POWER = 0.5
    MAX_POWER = 3.0

    MAX_ANGLE_OFFSET = 10.0

    def __init__(self):
        super().__init__()

        self.sim = FastFizSimulator()

        # [power, angle_offset]
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(2,),
            dtype=np.float32,
        )

        # 旧 custom_env.py の12次元観測を維持
        self.observation_space = spaces.Box(
            low=-10.0,
            high=10.0,
            shape=(12,),
            dtype=np.float32,
        )

        self.pocket_radius = 0.05

        self.macro_action: MacroAction | None = None

        self._episode_active = False

    # ============================================================
    # Macro -> Micro interface
    # ============================================================

    def set_macro_action(self, macro_action: MacroAction) -> None:
        """
        Macro層が決定した行動をMicro環境へ設定する。
        """

        if macro_action.strategy != Strategy.ATTACK:
            raise NotImplementedError(
                "MicroBilliardEnv currently supports ATTACK only."
            )

        if macro_action.target_ball not in (0, 1):
            raise ValueError(
                f"target_ball must be 0 or 1, got "
                f"{macro_action.target_ball}"
            )

        if macro_action.target_pocket not in self.POCKET_POSITIONS:
            raise ValueError(
                f"Unsupported target_pocket: "
                f"{macro_action.target_pocket}"
            )

        self.macro_action = macro_action

    # ============================================================
    # Gymnasium API
    # ============================================================

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        if self.macro_action is None:
            raise RuntimeError(
                "MacroAction must be set before reset(). "
                "Call env.set_macro_action(...) first."
            )

        init_state = self._sample_initial_state()

        self.sim.set_state(init_state)

        raw_obs = self.sim.get_state()

        self._episode_active = True

        return self._build_observation(raw_obs), {}
    
    def reset_to_game_state(
        self,
        game_state,
        macro_action,
    ):
        """
        Macro/MCTSから渡されたGameStateを使って、
        Microの1-shot episodeを開始する。
        """

        self.set_macro_action(macro_action)

        # 対象球が既にポケット済みなら実行不可
        target_state_index = (
            macro_action.target_ball + 1
        )

        if game_state.is_pocketed(
            target_state_index
        ):
            raise ValueError(
                "Cannot target an already pocketed ball."
            )

        positions = np.asarray(
            game_state.ball_positions,
            dtype=np.float32,
        )

        if positions.shape != (3, 2):
            raise ValueError(
                "game_state.ball_positions must "
                f"have shape (3, 2), got {positions.shape}"
            )

        snapshot = SimulatorSnapshot(
            positions=positions,
            pocket_indices=np.asarray(
                game_state.ball_pocket_indices,
                dtype=np.int32,
            ),
        )

        self.sim.restore(snapshot)

        self._episode_active = True

        return self._build_observation(
            self.sim.get_state()
        )

    def step(self, action):

        if not self._episode_active:
            raise RuntimeError(
                "Episode has already finished. Call reset() before the next shot."
            )

        if self.macro_action is None:
            raise RuntimeError(
                "MacroAction has not been set."
            )

        action = np.asarray(action, dtype=np.float32)

        if action.shape != (2,):
            raise ValueError(
                f"action must have shape (2,), got {action.shape}"
            )

        target_ball = self.macro_action.target_ball
        pocket_pos = self._get_target_pocket_position()

        prev_state = self.sim.get_state()

        cue_pos = prev_state[0:2]
        target_pos = self._get_ball_position(
            prev_state,
            target_ball,
        )

        # -----------------------------------------
        # SAC action -> physical shot parameters
        # -----------------------------------------

        power = self.MIN_POWER + (
            self.MAX_POWER - self.MIN_POWER
        ) * ((action[0] + 1.0) / 2.0)

        # 対象球を指定ポケットへ送るための
        # ghost-ball位置を計算
        ghost_pos = ghost_ball_position(
            target_pos,
            pocket_pos,
        )

        # 手球 -> ghost-ball が基準ショット方向
        ideal_angle = np.degrees(
            np.arctan2(
                ghost_pos[1] - cue_pos[1],
                ghost_pos[0] - cue_pos[0],
            )
        )

        # SACは基準方向からの微調整を担当
        angle = (
            ideal_angle
            + action[1] * self.MAX_ANGLE_OFFSET
        )

        result = self.sim.step(
            target_ball=target_ball,
            power=power,
            angle=angle,
        )

        raw_obs = result.state

        target_pocketed_at = self.sim.get_pocket_index(
            target_ball
        )

        cue_pocketed_at = self.sim.get_pocket_index(-1)

        success = (
            target_pocketed_at
            == self.macro_action.target_pocket
        )

        scratched = (
            cue_pocketed_at != -1
        )

        reward = self._calculate_reward(
            prev_state,
            raw_obs,
            target_ball,
            pocket_pos,
            success,
            scratched,
        )

        observation = self._build_observation(raw_obs)

        info = {
            "target_ball": target_ball,
            "target_pocket": self.macro_action.target_pocket,
            "power": float(power),
            "angle": float(angle),
            "ideal_angle": float(ideal_angle),
            "angle_offset": float(
                action[1] * self.MAX_ANGLE_OFFSET
            ),

            "success": success,
            "scratched": scratched,

            "target_pocketed_at": target_pocketed_at,
            "cue_pocketed_at": cue_pocketed_at,

            "target_pocketed_anywhere": (
                target_pocketed_at != -1
            ),

            "physically_possible": (
                result.legacy_reward >= 0.0
            ),
        }

        # Microは1ショットで必ずepisode終了
        self._episode_active = False
        terminated = True

        return observation, reward, terminated, False, info

    # ============================================================
    # Observation
    # ============================================================

    def _build_observation(
        self,
        raw_state: np.ndarray,
    ) -> np.ndarray:

        target_ball = self.macro_action.target_ball
        pocket_pos = self._get_target_pocket_position()

        cue_pos = raw_state[0:2]

        target_pos = self._get_ball_position(
            raw_state,
            target_ball,
        )

        other_ball = 1 - target_ball

        other_pos = self._get_ball_position(
            raw_state,
            other_ball,
        )

        cue_to_target = target_pos - cue_pos
        cue_to_other = other_pos - cue_pos
        target_to_pocket = pocket_pos - target_pos

        angle_cue_to_target = np.arctan2(
            cue_to_target[1],
            cue_to_target[0],
        )

        angle_target_to_pocket = np.arctan2(
            target_to_pocket[1],
            target_to_pocket[0],
        )

        angle_diff = (
            angle_target_to_pocket
            - angle_cue_to_target
        )

        dist_cue_to_target = np.linalg.norm(
            cue_to_target
        )

        dist_target_to_pocket = np.linalg.norm(
            target_to_pocket
        )

        return np.array(
            [
                cue_pos[0],
                cue_pos[1],

                cue_to_target[0],
                cue_to_target[1],

                cue_to_other[0],
                cue_to_other[1],

                target_to_pocket[0],
                target_to_pocket[1],

                np.sin(angle_diff),
                np.cos(angle_diff),

                dist_cue_to_target,
                dist_target_to_pocket,
            ],
            dtype=np.float32,
        )

    # ============================================================
    # Reward
    # ============================================================

    def _calculate_reward(
        self,
        prev_state,
        current_state,
        target_ball,
        pocket_pos,
        success,
        scratched,
    ):
        previous_target_pos = self._get_ball_position(
            prev_state,
            target_ball,
        )

        current_target_pos = self._get_ball_position(
            current_state,
            target_ball,
        )

        cue_pos = current_state[0:2]

        movement = np.linalg.norm(
            current_target_pos - previous_target_pos
        )

        initial_dist = np.linalg.norm(
            pocket_pos - previous_target_pos
        )

        final_dist = np.linalg.norm(
            pocket_pos - current_target_pos
        )

        reward = -1.0

        # 空振り
        if movement < 0.001:
            reward -= 1.0

        else:
            # ポケットへ近づいた量
            dist_reduction = initial_dist - final_dist
            reward += dist_reduction * 3.0

            # 移動方向
            move_vec = (
                current_target_pos
                - previous_target_pos
            )

            target_vec = (
                pocket_pos
                - previous_target_pos
            )

            cos_sim = np.dot(
                move_vec,
                target_vec,
            ) / (
                np.linalg.norm(move_vec)
                * np.linalg.norm(target_vec)
                + 1e-8
            )

            if cos_sim > 0:
                reward += cos_sim * 0.5

        # ポケットイン
        if final_dist < self.pocket_radius:
            reward += 10.0

        # スクラッチ
        if scratched:
            reward -= 10.0

        return float(reward)

    # ============================================================
    # Utility
    # ============================================================

    def _get_ball_position(
        self,
        state,
        ball_id,
    ):
        if ball_id == 0:
            return np.asarray(state[2:4])

        if ball_id == 1:
            return np.asarray(state[4:6])

        raise ValueError(
            f"Unknown ball_id: {ball_id}"
        )

    def _get_target_pocket_position(self):
        return self.POCKET_POSITIONS[
            self.macro_action.target_pocket
        ]

    def _sample_initial_state(self):
        """
        FastFizのテーブル内部に3球をランダム配置する。
        レール・ポケット付近は避ける。
        """

        margin = 0.15

        while True:

            cue = self.np_random.uniform(
                low=[
                    margin,
                    margin,
                ],
                high=[
                    self.TABLE_WIDTH - margin,
                    self.TABLE_LENGTH - margin,
                ],
            )

            obj0 = self.np_random.uniform(
                low=[
                    margin,
                    margin,
                ],
                high=[
                    self.TABLE_WIDTH - margin,
                    self.TABLE_LENGTH - margin,
                ],
            )

            obj1 = self.np_random.uniform(
                low=[
                    margin,
                    margin,
                ],
                high=[
                    self.TABLE_WIDTH - margin,
                    self.TABLE_LENGTH - margin,
                ],
            )

            if (
                np.linalg.norm(cue - obj0) > 0.10
                and np.linalg.norm(cue - obj1) > 0.10
                and np.linalg.norm(obj0 - obj1) > 0.10
            ):
                return np.array(
                    [
                        cue[0],
                        cue[1],
                        obj0[0],
                        obj0[1],
                        obj1[0],
                        obj1[1],
                    ],
                    dtype=np.float32,
                )

    def is_current_macro_feasible(self) -> bool:
        if self.macro_action is None:
            raise RuntimeError(
                "MacroAction has not been set."
            )

        state = self.sim.get_state()

        cue_pos = state[0:2]

        target_ball = self.macro_action.target_ball

        target_pos = self._get_ball_position(
            state,
            target_ball,
        )

        other_pos = self._get_ball_position(
            state,
            1 - target_ball,
        )

        pocket_pos = self._get_target_pocket_position()

        return is_direct_shot_feasible(
            cue_pos=cue_pos,
            target_pos=target_pos,
            other_pos=other_pos,
            pocket_pos=pocket_pos,
            table_width=self.TABLE_WIDTH,
            table_length=self.TABLE_LENGTH,
        )

    def get_state(self):
        return self.sim.get_state()

    def set_state(self, state):
        self.sim.set_state(state)