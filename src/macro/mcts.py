from dataclasses import dataclass, field
import math

import torch

from src.common.types import (
    GameState,
    MacroAction,
)

from src.macro.encoding import (
    game_state_to_tensor,
    macro_action_to_index,
)

from src.macro.transition import (
    MacroTransitionModel,
)

import numpy as np

from src.macro.candidate import (
    MacroCandidate,
)

from src.macro.candidate_encoding import (
    encode_macro_candidate,
)

MacroChoice = MacroAction | MacroCandidate

@dataclass
class MCTSNode:
    """
    MCTSの1ノード。

    reward:
        親ノードからこのノードへ遷移した際の報酬
    """

    prior: float = 0.0
    reward: float = 0.0

    state: GameState | None = None

    visit_count: int = 0
    value_sum: float = 0.0

    terminated: bool = False

    children: dict[
        MacroChoice,
        "MCTSNode",
    ] = field(default_factory=dict)

    @property
    def value(self) -> float:
        if self.visit_count == 0:
            return 0.0

        return self.value_sum / self.visit_count

    @property
    def expanded(self) -> bool:
        return len(self.children) > 0


class MCTS:
    """
    Macro Policy + Valueを利用するPUCTベースMCTS。

    状態遷移そのものはMacroTransitionModelへ委譲する。
    """

    def __init__(
        self,
        policy_network,
        value_network,
        action_generator,
        transition_model: MacroTransitionModel,
        simulations: int = 50,
        c_puct: float = 1.5,
        gamma: float = 0.99,
        device=None,
        candidate_mode=False,
    ):
        self.policy_network = policy_network
        self.value_network = value_network

        self.action_generator = action_generator
        self.transition_model = transition_model

        self.simulations = simulations
        self.c_puct = c_puct
        self.gamma = gamma

        if device is None:
            device = next(
                policy_network.parameters()
            ).device

        self.device = torch.device(device)

        self.candidate_mode = candidate_mode

    # ============================================================
    # Public API
    # ============================================================

    def search(
        self,
        root_state: GameState,
    ) -> MCTSNode:

        root = MCTSNode(
            state=root_state,
        )

        self._expand(root)

        if not root.children:
            raise RuntimeError(
                "No feasible MacroActions from root state."
            )

        for _ in range(self.simulations):
            self._simulate(root)

        return root

    def select_action(
        self,
        root_state: GameState,
    ) -> MacroAction:

        root = self.search(root_state)

        # 最終行動はvisit count最大
        action, _ = max(
            root.children.items(),
            key=lambda item: item[1].visit_count,
        )

        return action

    # ============================================================
    # One simulation
    # ============================================================

    def _simulate(
        self,
        root: MCTSNode,
    ):
        node = root

        search_path = [node]

        while (
            node.expanded
            and not node.terminated
        ):
            action, child = self._select_child(
                node
            )

            # 初めてこの枝を通る場合、
            # transitionを実行する
            if child.state is None:
                result = self.transition_model.step(
                    node.state,
                    action,
                )

                child.state = result.next_state
                child.reward = float(
                    result.reward
                )
                child.terminated = (
                    result.terminated
                )

                node = child
                search_path.append(node)

                break

            node = child
            search_path.append(node)

        # --------------------------------
        # Leaf evaluation
        # --------------------------------

        if node.terminated:
            leaf_value = 0.0

        else:
            leaf_value = self._evaluate(
                node.state
            )

            self._expand(node)

        # --------------------------------
        # Backup
        # --------------------------------

        self._backup(
            search_path,
            leaf_value,
        )

    # ============================================================
    # Selection
    # ============================================================

    def _select_child(
        self,
        node: MCTSNode,
    ):
        best_score = -float("inf")
        best_action = None
        best_child = None

        for action, child in node.children.items():

            score = self._puct_score(
                parent=node,
                child=child,
            )

            if score > best_score:
                best_score = score
                best_action = action
                best_child = child

        return best_action, best_child

    def _puct_score(
        self,
        parent: MCTSNode,
        child: MCTSNode,
    ) -> float:

        # Edge reward + future value
        q_value = (
            child.reward
            + self.gamma * child.value
        )

        exploration = (
            self.c_puct
            * child.prior
            * math.sqrt(
                parent.visit_count + 1
            )
            / (
                1 + child.visit_count
            )
        )

        return q_value + exploration

    # ============================================================
    # Expansion
    # ============================================================

    def _expand(
        self,
        node: MCTSNode,
    ):
        choices = (
            self.action_generator.generate(
                node.state
            )
        )

        if len(choices) == 0:
            return

        if self.candidate_mode:
            priors = (
                self._candidate_priors(
                    node.state,
                    choices,
                )
            )

        else:
            priors = (
                self._legacy_action_priors(
                    node.state,
                    choices,
                )
            )

        for choice, prior in zip(
            choices,
            priors,
        ):
            node.children[choice] = MCTSNode(
                prior=float(prior)
            )
            
    # ============================================================
    # Evaluation
    # ============================================================

    def _evaluate(
        self,
        state: GameState,
    ) -> float:

        state_tensor = game_state_to_tensor(
            state,
            device=self.device,
        ).unsqueeze(0)

        with torch.no_grad():
            value = self.value_network(
                state_tensor
            )

        return float(
            value.squeeze().item()
        )

    # ============================================================
    # Backup
    # ============================================================

    def _backup(
        self,
        search_path,
        leaf_value: float,
    ):
        value = leaf_value

        for node in reversed(
            search_path
        ):
            node.value_sum += value
            node.visit_count += 1

            value = (
                node.reward
                + self.gamma * value
            )
            
    def _legacy_action_priors(
        self,
        state,
        actions,
    ):
        state_tensor = (
            game_state_to_tensor(
                state,
                device=self.device,
            )
        )

        with torch.no_grad():
            logits = self.policy_network(
                state_tensor
            )

        # Policyの出力が
        # (1, 12), (12,), (12, 1)
        # などでも12要素へ統一する
        logits = logits.reshape(-1)

        valid_logits = []

        for action in actions:
            action_index = (
                macro_action_to_index(
                    action
                )
            )

            valid_logits.append(
                logits[action_index]
            )

        valid_logits = torch.stack(
            valid_logits
        )

        priors = torch.softmax(
            valid_logits,
            dim=0,
        )

        return (
            priors
            .detach()
            .cpu()
            .numpy()
            .reshape(-1)
        )
        
    def _candidate_priors(
        self,
        state,
        candidates,
    ):
        if not all(
            isinstance(
                candidate,
                MacroCandidate,
            )
            for candidate in candidates
        ):
            raise TypeError(
                "candidate_mode=True requires "
                "MacroCandidate objects."
            )

        state_tensor = (
            game_state_to_tensor(
                state,
                device=self.device,
            )
        )

        # CandidatePolicyは
        # state=(12,) を想定
        if (
            state_tensor.dim() == 2
            and state_tensor.shape[0] == 1
        ):
            state_tensor = (
                state_tensor.squeeze(0)
            )

        candidate_array = np.stack(
            [
                encode_macro_candidate(
                    candidate
                )
                for candidate in candidates
            ],
            axis=0,
        )

        candidate_tensor = torch.as_tensor(
            candidate_array,
            dtype=torch.float32,
            device=self.device,
        )

        with torch.no_grad():

            logits = self.policy_network(
                state_tensor,
                candidate_tensor,
            )

            priors = torch.softmax(
                logits,
                dim=0,
            )

        return (
            priors
            .detach()
            .cpu()
            .numpy()
            .reshape(-1)
        )