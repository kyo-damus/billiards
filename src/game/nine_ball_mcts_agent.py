"""Adapter: v3 fixed-Micro nine-ball PUCT behind the self-play interface."""

from dataclasses import replace

from src.game.nine_ball_learned_agent import NineBallMicroSACAgent
from src.game.nine_ball_mcts_v3 import NineBallMCTS
from src.game.nine_ball_value_v3 import load_state_v_v3


class NineBallMCTSAgent(NineBallMicroSACAgent):
    def __init__(
        self, micro_checkpoint, *,
        macro_policy_checkpoint=None,
        state_value_checkpoint=None,
        simulations=24,
        max_depth=2,
        c_puct=1.5,
        gamma=0.99,
        root_min_visits=1,
        device="cpu",
        learned_safety=False,
        learned_break=False,
        outcome_position=False,
    ):
        super().__init__(
            micro_checkpoint,
            macro_policy_checkpoint=macro_policy_checkpoint,
            deterministic=True,
            learned_safety=learned_safety,
            learned_break=learned_break,
            outcome_position=outcome_position,
            device=device,
        )
        state_v = None
        if state_value_checkpoint is not None:
            state_v, _ = load_state_v_v3(state_value_checkpoint, device=device)
        self.searcher = NineBallMCTS(
            self,
            policy_network=self.macro_policy,
            value_network=state_v,
            simulations=simulations,
            max_depth=max_depth,
            c_puct=c_puct,
            gamma=gamma,
            root_min_visits=root_min_visits,
            learned_safety=learned_safety,
            learned_break=learned_break,
            outcome_position=outcome_position,
        )

    def choose_shot(self, state, context=None):
        goal = self.searcher.search(state, context)
        plan = self.plan_goal(state, goal, context)
        return replace(
            plan,
            goal_label=f"{goal.kind}_MCTS_V3",
            macro_search=self.searcher.last_report,
        )
