import numpy as np

from src.common.types import GameState
from src.game.rules import GameTransitionResult


class SimplifiedGameRules:
    """
    2個の的球を使う現在の簡易環境用ルール。

    本物の9-ball等のルールではない。
    階層型RL基盤の状態遷移を検証するための暫定実装。

    Rules:
    - 的球を新しく落とすと +1点 / +1 reward
    - 何も落とさなければ手番交代
    - 的球を落とせば手番継続
    - scratch は -1 reward かつ簡易ゲーム終了
    - 2個の的球が両方落ちればゲーム終了
    """

    def apply(
        self,
        previous_state: GameState,
        physical_state: GameState,
        shot_info: dict,
    ) -> GameTransitionResult:

        previous_pockets = np.asarray(
            previous_state.ball_pocket_indices,
            dtype=np.int32,
        )

        current_pockets = np.asarray(
            physical_state.ball_pocket_indices,
            dtype=np.int32,
        )

        # ------------------------------------------------
        # Scratch
        # ------------------------------------------------

        scratched = bool(
            shot_info.get(
                "scratched",
                current_pockets[0] >= 0,
            )
        )

        if scratched:
            next_state = GameState(
                ball_positions=(
                    physical_state.ball_positions.copy()
                ),
                score=previous_state.score.copy(),
                current_player=(
                    1 - previous_state.current_player
                ),
                ball_pocket_indices=(
                    current_pockets.copy()
                ),
            )

            return GameTransitionResult(
                next_state=next_state,
                reward=-1.0,
                terminated=True,
                turn_changed=True,
            )

        # ------------------------------------------------
        # Newly pocketed object balls
        # GameState:
        # 0 = cue
        # 1 = object ball 0
        # 2 = object ball 1
        # ------------------------------------------------

        newly_pocketed = 0

        for ball_index in (1, 2):
            was_on_table = (
                previous_pockets[ball_index] < 0
            )

            is_now_pocketed = (
                current_pockets[ball_index] >= 0
            )

            if was_on_table and is_now_pocketed:
                newly_pocketed += 1

        # ------------------------------------------------
        # Score
        # ------------------------------------------------

        next_score = np.asarray(
            previous_state.score,
            dtype=np.float32,
        ).copy()

        if newly_pocketed > 0:
            next_score[
                previous_state.current_player
            ] += newly_pocketed

        # ------------------------------------------------
        # Game termination
        # ------------------------------------------------

        all_object_balls_pocketed = bool(
            current_pockets[1] >= 0
            and current_pockets[2] >= 0
        )

        # ------------------------------------------------
        # Turn
        # ------------------------------------------------

        turn_changed = (
            newly_pocketed == 0
            and not all_object_balls_pocketed
        )

        if turn_changed:
            next_player = (
                1 - previous_state.current_player
            )
        else:
            next_player = (
                previous_state.current_player
            )

        next_state = GameState(
            ball_positions=(
                physical_state.ball_positions.copy()
            ),
            score=next_score,
            current_player=next_player,
            ball_pocket_indices=(
                current_pockets.copy()
            ),
        )

        return GameTransitionResult(
            next_state=next_state,
            reward=float(newly_pocketed),
            terminated=all_object_balls_pocketed,
            turn_changed=turn_changed,
        )
        