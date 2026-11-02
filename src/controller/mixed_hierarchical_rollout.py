"""Bounded multi-turn rollout of the EXISTING hierarchical controller.

A turn is one MacroCandidate decision followed by one physical shot.
This module deliberately does not train, score win rate, or implement
official billiards rules. It only verifies that GameState is carried
forward across independent MCTS decisions.
"""

from dataclasses import dataclass

from src.common.types import GameState
from src.controller.hierarchical_agent import HierarchicalAgent
from src.macro.candidate import MacroCandidate
from src.macro.mcts import NoFeasibleMacroActionError


@dataclass(frozen=True)
class HierarchicalTurnRecord:
    turn_index: int
    player_before: int
    choice: MacroCandidate
    reward: float
    player_after: int
    terminated: bool
    score_delta: tuple[float, float]
    newly_pocketed_balls: tuple[int, ...]


@dataclass
class HierarchicalRolloutResult:
    records: list[HierarchicalTurnRecord]
    final_state: GameState
    stop_reason: str
    terminated: bool


class MixedHierarchicalRollout:
    """Reuse one Agent/MCTS, replan after every executed physical shot."""

    def __init__(self, agent: HierarchicalAgent):
        self.agent = agent

    def run(
        self, initial_state: GameState, max_turns: int = 3
    ) -> HierarchicalRolloutResult:
        if max_turns <= 0:
            raise ValueError("max_turns must be positive.")

        state = initial_state
        records = []

        # SimplifiedGameRules terminates at scratch or after both
        # object balls have been pocketed.
        if state.is_pocketed(0) or (
            state.is_pocketed(1) and state.is_pocketed(2)
        ):
            return HierarchicalRolloutResult(
                records=records,
                final_state=state,
                stop_reason="initial_terminal_state",
                terminated=True,
            )

        for turn_index in range(max_turns):
            try:
                result = self.agent.step(state)
            except NoFeasibleMacroActionError:
                return HierarchicalRolloutResult(
                    records=records,
                    final_state=state,
                    stop_reason="no_feasible_macro_candidates",
                    terminated=False,
                )

            if not isinstance(result.macro_action, MacroCandidate):
                raise TypeError(
                    "Mixed rollout requires a MacroCandidate decision."
                )
            if result.next_state is state:
                raise RuntimeError(
                    "Physical execution must return a new GameState."
                )

            # Score is assigned by SimplifiedGameRules, while the
            # returned reward is the Macro local evaluation used
            # by search. Preserve both without equating them.
            score_delta = tuple(
                float(result.next_state.score[i] - state.score[i])
                for i in range(2)
            )
            newly_pocketed = tuple(
                ball
                for ball in (0, 1)
                if (
                    state.ball_pocket_indices[ball + 1] < 0
                    and result.next_state.ball_pocket_indices[ball + 1] >= 0
                )
            )

            records.append(
                HierarchicalTurnRecord(
                    turn_index=turn_index,
                    player_before=int(state.current_player),
                    choice=result.macro_action,
                    reward=float(result.reward),
                    player_after=int(result.next_state.current_player),
                    terminated=bool(result.terminated),
                    score_delta=score_delta,
                    newly_pocketed_balls=newly_pocketed,
                )
            )
            state = result.next_state

            if result.terminated:
                return HierarchicalRolloutResult(
                    records=records,
                    final_state=state,
                    stop_reason="game_terminated",
                    terminated=True,
                )

        return HierarchicalRolloutResult(
            records=records,
            final_state=state,
            stop_reason="max_turns",
            terminated=False,
        )
