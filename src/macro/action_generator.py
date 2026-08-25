from src.common.table import (
    TABLE_WIDTH,
    TABLE_LENGTH,
    POCKET_POSITIONS,
)

from src.common.types import (
    GameState,
    MacroAction,
    Strategy,
)

from src.env.shot_geometry import (
    is_direct_shot_feasible,
)


class MacroActionGenerator:
    """
    現在盤面からMacro行動候補を生成する。

    現段階:
        ATTACKのみ
        target_ball = 0, 1
        target_pocket = 0 ... 5

    将来的には:
        DEFENSE
        bank shot
        safety shot
        combination shot
    などを追加する。
    """

    def generate(
        self,
        game_state: GameState,
    ) -> list[MacroAction]:

        positions = game_state.ball_positions

        if positions.shape != (3, 2):
            raise ValueError(
                "game_state.ball_positions must "
                f"have shape (3, 2), got {positions.shape}"
            )

        cue_pos = positions[0]

        candidates = []

        for target_ball in (0, 1):

            # GameState上では
            # 0 = cue
            # 1 = object ball 0
            # 2 = object ball 1
            target_state_index = target_ball + 1

            # 既にポケット済みの球は
            # MacroAction候補にしない。
            if game_state.is_pocketed(
                target_state_index
            ):
                continue

            target_pos = positions[
                target_ball + 1
            ]

            other_ball = 1 - target_ball
            other_state_index = other_ball + 1

            if game_state.is_pocketed(
                other_state_index
            ):
                other_pos = None
            else:
                other_pos = positions[
                    other_state_index
                ]

            for (
                pocket_id,
                pocket_pos,
            ) in POCKET_POSITIONS.items():

                feasible = is_direct_shot_feasible(
                    cue_pos=cue_pos,
                    target_pos=target_pos,
                    other_pos=other_pos,
                    pocket_pos=pocket_pos,
                    table_width=TABLE_WIDTH,
                    table_length=TABLE_LENGTH,
                )

                if not feasible:
                    continue

                candidates.append(
                    MacroAction(
                        strategy=Strategy.ATTACK,
                        target_ball=target_ball,
                        target_pocket=pocket_id,
                    )
                )

        return candidates