"""Uniform-prior tactical MCTS ablation: no learned Policy or State V.

This keeps tactical candidate generation and is NOT a No-Macro baseline.
"""
from src.game.nine_ball_mcts_agent import NineBallMCTSAgent


class UniformPriorMCTSAgent(NineBallMCTSAgent):
    def __init__(self, micro_checkpoint, **kwargs):
        kwargs.pop("macro_policy_checkpoint", None)
        kwargs.pop("state_value_checkpoint", None)
        super().__init__(
            micro_checkpoint,
            macro_policy_checkpoint=None,
            state_value_checkpoint=None,
            **kwargs,
        )
