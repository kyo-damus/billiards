import copy

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import Adam

from src.micro.actor import GaussianActor
from src.micro.critic import TwinQCritic


class SACAgent:

    def __init__(
        self,
        state_dim=12,
        action_dim=2,
        hidden_dim=256,
        learning_rate=3e-4,
        gamma=0.99,
        tau=0.005,
        device=None,
    ):
        if device is None:
            device = (
                "cuda"
                if torch.cuda.is_available()
                else "cpu"
            )

        self.device = torch.device(device)

        self.gamma = gamma
        self.tau = tau

        # --------------------------------
        # Networks
        # --------------------------------

        self.actor = GaussianActor(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dim=hidden_dim,
        ).to(self.device)

        self.critic = TwinQCritic(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dim=hidden_dim,
        ).to(self.device)

        self.critic_target = copy.deepcopy(
            self.critic
        ).to(self.device)

        self.critic_target.eval()

        # --------------------------------
        # Optimizers
        # --------------------------------

        self.actor_optimizer = Adam(
            self.actor.parameters(),
            lr=learning_rate,
        )

        self.critic_optimizer = Adam(
            self.critic.parameters(),
            lr=learning_rate,
        )

        # --------------------------------
        # Automatic entropy tuning
        # --------------------------------

        self.target_entropy = -float(action_dim)

        self.log_alpha = torch.zeros(
            1,
            requires_grad=True,
            device=self.device,
        )

        self.alpha_optimizer = Adam(
            [self.log_alpha],
            lr=learning_rate,
        )

    @property
    def alpha(self):
        return self.log_alpha.exp()

    # ============================================================
    # Action selection
    # ============================================================

    def select_action(
        self,
        state,
        deterministic=False,
    ):
        state = torch.as_tensor(
            state,
            dtype=torch.float32,
            device=self.device,
        ).unsqueeze(0)

        with torch.no_grad():

            if deterministic:
                action = self.actor.deterministic(
                    state
                )
            else:
                action, _ = self.actor.sample(
                    state
                )

        return (
            action
            .squeeze(0)
            .cpu()
            .numpy()
        )

    # ============================================================
    # SAC update
    # ============================================================

    def update(self, batch):

        (
            states,
            actions,
            rewards,
            next_states,
            dones,
        ) = batch

        # --------------------------------
        # Critic update
        # --------------------------------

        with torch.no_grad():

            next_actions, next_log_prob = (
                self.actor.sample(next_states)
            )

            target_q1, target_q2 = (
                self.critic_target(
                    next_states,
                    next_actions,
                )
            )

            target_q = torch.min(
                target_q1,
                target_q2,
            )

            target_q -= (
                self.alpha.detach()
                * next_log_prob
            )

            expected_q = (
                rewards
                + (1.0 - dones)
                * self.gamma
                * target_q
            )

        current_q1, current_q2 = self.critic(
            states,
            actions,
        )

        critic_loss = (
            F.mse_loss(
                current_q1,
                expected_q,
            )
            + F.mse_loss(
                current_q2,
                expected_q,
            )
        )

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # --------------------------------
        # Actor update
        # --------------------------------

        for parameter in self.critic.parameters():
            parameter.requires_grad = False

        new_actions, log_prob = self.actor.sample(
            states
        )

        q1, q2 = self.critic(
            states,
            new_actions,
        )

        q = torch.min(q1, q2)

        actor_loss = (
            self.alpha.detach()
            * log_prob
            - q
        ).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        for parameter in self.critic.parameters():
            parameter.requires_grad = True

        # --------------------------------
        # Alpha update
        # --------------------------------

        alpha_loss = -(
            self.log_alpha
            * (
                log_prob
                + self.target_entropy
            ).detach()
        ).mean()

        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()

        # --------------------------------
        # Target critic update
        # --------------------------------

        self._soft_update_target()

        return {
            "actor_loss": actor_loss.item(),
            "critic_loss": critic_loss.item(),
            "alpha_loss": alpha_loss.item(),
            "alpha": self.alpha.item(),
        }

    def _soft_update_target(self):

        with torch.no_grad():

            for target_param, param in zip(
                self.critic_target.parameters(),
                self.critic.parameters(),
            ):

                target_param.data.mul_(
                    1.0 - self.tau
                )

                target_param.data.add_(
                    self.tau * param.data
                )