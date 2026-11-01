"""Small common helpers for 31-D Attack/Safety MCTS experiments.

Keep these separate from the legacy Attack-only candidate classes:
the mixed 31-D representation and the zero state-V baseline are
explicitly temporary experiment choices.
"""

import torch
from torch import nn


class ZeroValue(nn.Module):
    """A neutral state V(s)=0 baseline, not a trained value model."""

    def forward(self, states):
        return torch.zeros(
            states.shape[0],
            device=states.device,
            dtype=states.dtype,
        )


class FixedCandidateGenerator:
    """Snapshot root candidates for fair paired MCTS searches."""

    def __init__(self, candidates):
        self.candidates = tuple(candidates)

    def generate(self, state):
        return list(self.candidates)


class MixedRootGenerator:
    """Generate ATTACK plus outcome-guided executable SAFETY."""

    def __init__(
        self,
        attack_generator,
        safety_shots,
        safety_adapter,
    ):
        self.attack_generator = attack_generator
        self.safety_shots = safety_shots
        self.safety_adapter = safety_adapter
        self.last_attack_count = 0
        self.last_safety_count = 0

    def generate(self, state):
        attacks = list(
            self.attack_generator.generate(state)
        )
        proposals = self.safety_shots.generate(state)
        safeties = self.safety_adapter.from_proposals(
            proposals
        )
        self.last_attack_count = len(attacks)
        self.last_safety_count = len(safeties)
        return attacks + safeties
