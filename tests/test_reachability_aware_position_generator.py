import numpy as np

from src.common.types import (
    GameState,
    MacroAction,
    Strategy,
)
from src.goal.types import (
    TacticalGoal,
    TacticalGoalType,
)
from src.macro.candidate import (
    MacroCandidate,
)
from src.macro.position_feasibility import (
    PositionFeasibilityResult,
)
from src.macro.reachability_aware_position_generator import (
    ReachabilityAwarePositionCandidateGenerator,
)


def make_state():

    return GameState(
        ball_positions=np.array(
            [
                [0.80, 1.10],
                [0.40, 1.10],
                [0.80, 0.80],
            ],
            dtype=np.float32,
        ),
        score=np.zeros(
            2,
            dtype=np.float32,
        ),
        current_player=0,
        ball_pocket_indices=np.array(
            [-1, -1, -1],
            dtype=np.int32,
        ),
    )


def make_candidate(
    pocket,
):

    action = MacroAction(
        strategy=Strategy.ATTACK,
        target_ball=0,
        target_pocket=pocket,
    )

    goal = TacticalGoal(
        goal_type=(
            TacticalGoalType.POSITION_ATTACK
        ),
        target_ball=0,
        target_pocket=pocket,
        cue_target_position=(
            0.50 + 0.01 * pocket,
            1.00,
        ),
        cue_target_radius=0.10,
        next_target_ball=1,
        next_target_pocket=5,
    )

    return MacroCandidate(
        action=action,
        goal=goal,
    )


class FakeBaseGenerator:

    def __init__(
        self,
        candidates,
    ):
        self.candidates = list(
            candidates
        )

    def generate(
        self,
        state,
    ):
        return list(
            self.candidates
        )


class FakeChecker:

    def __init__(
        self,
        results_by_pocket,
    ):
        self.results_by_pocket = (
            results_by_pocket
        )

    def check(
        self,
        state,
        candidate,
    ):
        return (
            self.results_by_pocket[
                candidate.action.target_pocket
            ]
        )


def make_result(
    *,
    reachable,
    pot_found,
    distance,
):

    return PositionFeasibilityResult(
        reachable=reachable,
        pot_found=pot_found,
        best_cue_distance=distance,
        tested_actions=25,
        best_action=(
            np.zeros(
                2,
                dtype=np.float32,
            )
            if pot_found
            else None
        ),
    )


def test_reachable_candidate_is_prioritized():

    candidates = [
        make_candidate(0),
        make_candidate(1),
        make_candidate(2),
    ]

    checker = FakeChecker(
        {
            0: make_result(
                reachable=False,
                pot_found=True,
                distance=0.20,
            ),
            1: make_result(
                reachable=True,
                pot_found=True,
                distance=0.08,
            ),
            2: make_result(
                reachable=False,
                pot_found=False,
                distance=float("inf"),
            ),
        }
    )

    generator = (
        ReachabilityAwarePositionCandidateGenerator(
            base_generator=(
                FakeBaseGenerator(
                    candidates
                )
            ),
            feasibility_checker=checker,
            top_k=2,
        )
    )

    selected = generator.generate(
        make_state()
    )

    assert len(selected) == 2

    assert (
        selected[0].action.target_pocket
        == 1
    )

    assert (
        selected[1].action.target_pocket
        == 0
    )


def test_non_pot_candidate_is_removed_when_pot_exists():

    candidates = [
        make_candidate(0),
        make_candidate(1),
    ]

    checker = FakeChecker(
        {
            0: make_result(
                reachable=False,
                pot_found=False,
                distance=float("inf"),
            ),
            1: make_result(
                reachable=False,
                pot_found=True,
                distance=0.30,
            ),
        }
    )

    generator = (
        ReachabilityAwarePositionCandidateGenerator(
            base_generator=(
                FakeBaseGenerator(
                    candidates
                )
            ),
            feasibility_checker=checker,
            top_k=8,
        )
    )

    selected = generator.generate(
        make_state()
    )

    assert len(selected) == 1

    assert (
        selected[0].action.target_pocket
        == 1
    )


def test_fallback_keeps_candidates_when_no_pot_is_found():

    candidates = [
        make_candidate(0),
        make_candidate(1),
        make_candidate(2),
    ]

    checker = FakeChecker(
        {
            pocket: make_result(
                reachable=False,
                pot_found=False,
                distance=float("inf"),
            )
            for pocket in range(3)
        }
    )

    generator = (
        ReachabilityAwarePositionCandidateGenerator(
            base_generator=(
                FakeBaseGenerator(
                    candidates
                )
            ),
            feasibility_checker=checker,
            top_k=2,
            fallback_to_unfiltered=True,
        )
    )

    selected = generator.generate(
        make_state()
    )

    assert len(selected) == 2


def test_top_k_limits_selected_candidates():

    candidates = [
        make_candidate(0),
        make_candidate(1),
        make_candidate(2),
    ]

    checker = FakeChecker(
        {
            0: make_result(
                reachable=False,
                pot_found=True,
                distance=0.30,
            ),
            1: make_result(
                reachable=False,
                pot_found=True,
                distance=0.20,
            ),
            2: make_result(
                reachable=False,
                pot_found=True,
                distance=0.10,
            ),
        }
    )

    generator = (
        ReachabilityAwarePositionCandidateGenerator(
            base_generator=(
                FakeBaseGenerator(
                    candidates
                )
            ),
            feasibility_checker=checker,
            top_k=2,
        )
    )

    selected = generator.generate(
        make_state()
    )

    assert len(selected) == 2

    assert [
        candidate.action.target_pocket
        for candidate in selected
    ] == [2, 1]
