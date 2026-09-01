import torch

from scripts.train_goal_conditioned_micro import (
    load_initial_checkpoint,
    save_checkpoint,
)

from src.micro.goal_conditioned_sac import (
    GoalConditionedSACAgent,
)


def test_full_finetune_restores_model_and_alpha(
    tmp_path,
):

    source = GoalConditionedSACAgent(
        action_dim=2,
        device="cpu",
    )

    # alphaを分かりやすい値へ変更
    with torch.no_grad():
        source.agent.log_alpha.fill_(
            -1.5
        )

    checkpoint_path = (
        tmp_path / "test_finetune.pt"
    )

    save_checkpoint(
        source,
        str(checkpoint_path),
        step=100,
    )

    target = GoalConditionedSACAgent(
        action_dim=2,
        device="cpu",
    )

    load_initial_checkpoint(
        agent=target,
        checkpoint_path=str(
            checkpoint_path
        ),
        device="cpu",
        mode="full",
    )

    # Actor
    for source_param, target_param in zip(
        source.agent.actor.parameters(),
        target.agent.actor.parameters(),
    ):
        assert torch.allclose(
            source_param,
            target_param,
        )

    # Critic
    for source_param, target_param in zip(
        source.agent.critic.parameters(),
        target.agent.critic.parameters(),
    ):
        assert torch.allclose(
            source_param,
            target_param,
        )

    # Target Critic
    for source_param, target_param in zip(
        source.agent.critic_target.parameters(),
        target.agent.critic_target.parameters(),
    ):
        assert torch.allclose(
            source_param,
            target_param,
        )

    # alpha
    assert torch.allclose(
        source.agent.log_alpha,
        target.agent.log_alpha,
    )
    