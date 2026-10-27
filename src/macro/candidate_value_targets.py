from dataclasses import dataclass


@dataclass(frozen=True)
class MacroCandidateValueTarget:
    """
    1つのMacroCandidateに対する
    2-step supervised target。

    target:
        immediate_reward
        + gamma * best_next_reward

    深いMCTSの探索結果には依存せず、
    FastFizで観測した局所rewardだけから作る。
    """

    immediate_reward: float
    next_best_reward: float
    target: float

    terminated: bool
    next_candidate_count: int


def evaluate_two_step_candidate_target(
    state,
    candidate,
    transition_model,
    next_candidate_generator,
    gamma=0.99,
) -> MacroCandidateValueTarget:
    """
    current candidateを1手実行し、
    次状態の候補をさらに1手だけ評価して
    Q(s, c)の教師値を作る。

    next candidateは、現段階では
    cheap DIRECT generatorを利用する想定。
    """

    gamma = float(
        gamma
    )

    if not (
        0.0
        <= gamma
        <= 1.0
    ):
        raise ValueError(
            "gamma must be between 0 and 1."
        )

    first = transition_model.step(
        state,
        candidate,
    )

    immediate_reward = float(
        first.reward
    )

    if first.terminated:
        return MacroCandidateValueTarget(
            immediate_reward=(
                immediate_reward
            ),
            next_best_reward=0.0,
            target=immediate_reward,
            terminated=True,
            next_candidate_count=0,
        )

    next_candidates = list(
        next_candidate_generator.generate(
            first.next_state
        )
    )

    if not next_candidates:
        return MacroCandidateValueTarget(
            immediate_reward=(
                immediate_reward
            ),
            next_best_reward=0.0,
            target=immediate_reward,
            terminated=False,
            next_candidate_count=0,
        )

    next_rewards = []

    for next_candidate in next_candidates:
        second = transition_model.step(
            first.next_state,
            next_candidate,
        )

        next_rewards.append(
            float(
                second.reward
            )
        )

    next_best_reward = max(
        next_rewards
    )

    target = (
        immediate_reward
        + gamma
        * next_best_reward
    )

    return MacroCandidateValueTarget(
        immediate_reward=(
            immediate_reward
        ),
        next_best_reward=float(
            next_best_reward
        ),
        target=float(
            target
        ),
        terminated=False,
        next_candidate_count=len(
            next_candidates
        ),
    )
