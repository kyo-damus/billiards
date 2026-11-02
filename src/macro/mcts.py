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
    encode_macro_candidate_v2,
    MACRO_CANDIDATE_DIM,
    MACRO_CANDIDATE_V2_DIM,
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

    # 学習済み候補Value/Qがある場合の
    # 未訪問候補に対する事前予測値。
    predicted_q: float | None = None

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
        root_action_generator=None,
        force_evaluate_unvisited=False,
        root_min_visits_before_puct=0,
        candidate_value_network=None,
        candidate_q_prior_visits=0.0,
        turn_aware=False,
        candidate_encoding_version=1,
        uniform_candidate_priors=False,
    ):
        self.policy_network = policy_network
        self.value_network = value_network

        # Explicit opt-in to physical shot-aware 31D features.
        # Existing 25D Attack checkpoints retain their semantics.
        if candidate_encoding_version not in (1, 2):
            raise ValueError(
                "candidate_encoding_version must be 1 or 2."
            )
        self.candidate_encoding_version = int(
            candidate_encoding_version
        )
        self.uniform_candidate_priors = bool(
            uniform_candidate_priors
        )
        self.candidate_encoder = (
            encode_macro_candidate_v2
            if self.candidate_encoding_version == 2
            else encode_macro_candidate
        )
        if self.candidate_encoding_version == 2:
            if not candidate_mode or not turn_aware:
                raise ValueError(
                    "31D mixed MCTS needs candidate_mode=True "
                    "and turn_aware=True."
                )
            if not self.uniform_candidate_priors:
                # Explicitly require a 31-D candidate policy.
                # A legacy fixed-12-class or 25-D policy must
                # never silently enter the mixed search.
                policy_dim = getattr(
                    policy_network, "candidate_dim", None
                )
                if policy_dim != MACRO_CANDIDATE_V2_DIM:
                    raise ValueError(
                        "31D mixed MCTS requires a 31-D Candidate "
                        "Policy v2 or uniform_candidate_priors=True."
                    )
        self.candidate_value_network = (
            candidate_value_network
        )

        if candidate_q_prior_visits < 0.0:
            raise ValueError(
                "candidate_q_prior_visits "
                "must be >= 0."
            )

        self.candidate_q_prior_visits = float(
            candidate_q_prior_visits
        )

        if self.candidate_value_network is not None:
            actual_dim = getattr(
                self.candidate_value_network,
                "candidate_dim",
                None,
            )
            expected_dim = (
                MACRO_CANDIDATE_V2_DIM
                if self.candidate_encoding_version == 2
                else MACRO_CANDIDATE_DIM
            )
            if (
                actual_dim is not None
                and int(actual_dim) != expected_dim
            ):
                raise ValueError(
                    "Candidate Q feature dimension mismatch: "
                    f"expected {expected_dim}, got {actual_dim}."
                )

        # False: legacy Attack v0 (keeps existing Q semantics).
        # True: player-to-move perspective for multi-player play.
        self.turn_aware = bool(
            turn_aware
        )

        self.action_generator = action_generator
        self.root_action_generator = (
            root_action_generator
            if root_action_generator is not None
            else action_generator
        )
        self.transition_model = transition_model

        self.simulations = simulations
        self.c_puct = c_puct
        self.gamma = gamma

        if device is None:
            if policy_network is not None:
                device = next(
                    policy_network.parameters()
                ).device
            elif self.candidate_value_network is not None:
                device = next(
                    self.candidate_value_network.parameters()
                ).device
            else:
                raise ValueError(
                    "device required without a Policy or Q network."
                )

        self.device = torch.device(device)

        self.candidate_mode = candidate_mode
        self.force_evaluate_unvisited = bool(
            force_evaluate_unvisited
        )

        if root_min_visits_before_puct < 0:
            raise ValueError(
                "root_min_visits_before_puct "
                "must be >= 0."
            )

        self.root_min_visits_before_puct = int(
            root_min_visits_before_puct
        )

        # 直近searchの診断用。
        # 探索ロジックには使用しない。
        self.last_root = None

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

        self._expand(
            root,
            use_root_generator=True,
        )

        if not root.children:
            raise RuntimeError(
                "No feasible MacroActions from root state."
            )

        for _ in range(self.simulations):
            self._simulate(root)

        self.last_root = root

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
                node,
                min_visits=(
                    self.root_min_visits_before_puct
                    if node is root
                    else 0
                ),
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
        min_visits=0,
    ):
        # root候補を同じ深さまで最低限比較したい場合、
        # visit数が閾値未満の候補を優先する。
        if min_visits > 0:
            under_sampled = [
                (action, child)
                for action, child
                in node.children.items()
                if child.visit_count < min_visits
            ]

            if under_sampled:
                min_count = min(
                    child.visit_count
                    for _, child
                    in under_sampled
                )

                least_visited = [
                    (action, child)
                    for action, child
                    in under_sampled
                    if child.visit_count
                    == min_count
                ]

                return max(
                    least_visited,
                    key=lambda item: (
                        item[1].prior
                    ),
                )

        # FastFizの局所評価を比較する場合、
        # 未評価候補を最低1回ずつ遷移させてから
        # PUCTによる再訪へ移る。
        if self.force_evaluate_unvisited:
            unvisited = [
                (action, child)
                for action, child
                in node.children.items()
                if child.state is None
            ]

            if unvisited:
                return max(
                    unvisited,
                    key=lambda item: (
                        item[1].prior
                    ),
                )

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

        q_value = (
            self._candidate_q_value(
                child,
                parent=parent,
            )
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

    def _candidate_q_value(
        self,
        child: MCTSNode,
        parent: MCTSNode | None = None,
    ) -> float:
        """
        候補依存QをMCTSの事前評価として利用する。

        未訪問時はpredicted_qをそのまま使い、
        訪問後は実測Qと予測Qをpseudo visitとして混合する。
        visit数が増えるほど実測Qが支配的になる。
        """

        future_sign = self._turn_sign(
            parent.state if parent is not None else None,
            child.state,
        )

        empirical_q = (
            child.reward
            + self.gamma
            * future_sign
            * child.value
        )

        if child.predicted_q is None:
            return empirical_q

        if child.visit_count == 0:
            return float(
                child.predicted_q
            )

        prior_visits = (
            self.candidate_q_prior_visits
        )

        if prior_visits <= 0.0:
            return empirical_q

        return float(
            (
                child.visit_count
                * empirical_q
                + prior_visits
                * child.predicted_q
            )
            / (
                child.visit_count
                + prior_visits
            )
        )

    def _turn_sign(
        self,
        parent_state: GameState | None,
        child_state: GameState | None,
    ) -> float:
        """
        Returns the sign of the child's continuation value
        from the previous player's perspective.

        Child values are defined for the player to move in
        child_state. A turn change therefore negates them.
        """
        if (
            not self.turn_aware
            or parent_state is None
            or child_state is None
        ):
            return 1.0

        return (
            1.0
            if parent_state.current_player
            == child_state.current_player
            else -1.0
        )

    # ============================================================
    # Expansion
    # ============================================================

    def _expand(
        self,
        node: MCTSNode,
        use_root_generator=False,
    ):
        generator = (
            self.root_action_generator
            if use_root_generator
            else self.action_generator
        )

        choices = (
            generator.generate(
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

            if (
                self.candidate_value_network
                is not None
            ):
                predicted_qs = (
                    self._candidate_values(
                        node.state,
                        choices,
                    )
                )
            else:
                predicted_qs = [
                    None
                    for _ in choices
                ]

        else:
            priors = (
                self._legacy_action_priors(
                    node.state,
                    choices,
                )
            )

            predicted_qs = [
                None
                for _ in choices
            ]

        for (
            choice,
            prior,
            predicted_q,
        ) in zip(
            choices,
            priors,
            predicted_qs,
        ):
            node.children[choice] = MCTSNode(
                prior=float(prior),
                predicted_q=(
                    None
                    if predicted_q is None
                    else float(predicted_q)
                ),
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

        # value always uses the player-to-move perspective
        # of the node that is about to be updated. For a
        # parent, a continuation by the opponent is negative.
        for index in range(
            len(search_path) - 1,
            -1,
            -1,
        ):
            node = search_path[index]
            node.value_sum += value
            node.visit_count += 1

            if index > 0:
                parent = search_path[index - 1]
                sign = self._turn_sign(
                    parent.state,
                    node.state,
                )
                value = (
                    node.reward
                    + self.gamma
                    * sign
                    * value
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
        if self.uniform_candidate_priors:
            if not candidates:
                return np.array([], dtype=np.float32)
            return np.full(
                len(candidates),
                1.0 / len(candidates),
                dtype=np.float32,
            )
        if self.policy_network is None:
            raise ValueError(
                "Candidate Policy is missing; enable "
                "uniform_candidate_priors explicitly."
            )

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
                self.candidate_encoder(
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

    def _candidate_values(
        self,
        state,
        candidates,
    ):
        """
        候補依存Value/Qネットワークから
        各MacroCandidateのQ(s, c)を推定する。
        """

        if (
            self.candidate_value_network
            is None
        ):
            raise RuntimeError(
                "candidate_value_network "
                "is not configured."
            )

        if not all(
            isinstance(
                candidate,
                MacroCandidate,
            )
            for candidate in candidates
        ):
            raise TypeError(
                "Candidate value requires "
                "MacroCandidate objects."
            )

        state_tensor = (
            game_state_to_tensor(
                state,
                device=self.device,
            )
        )

        if (
            state_tensor.dim() == 2
            and state_tensor.shape[0] == 1
        ):
            state_tensor = (
                state_tensor.squeeze(0)
            )

        candidate_array = np.stack(
            [
                self.candidate_encoder(
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
            values = (
                self.candidate_value_network(
                    state_tensor,
                    candidate_tensor,
                )
            )

        return (
            values
            .detach()
            .cpu()
            .numpy()
            .reshape(-1)
        )
