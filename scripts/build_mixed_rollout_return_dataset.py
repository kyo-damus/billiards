"""Convert completed executed mixed rollouts to selected-action Q targets.

This is an *on-policy observed-return* dataset, NOT the existing
all-candidate two-step FastFiz oracle dataset. Only actions that
were actually executed receive labels. Nonterminal truncated
rollouts are rejected instead of silently setting V(s_tail)=0.

For two-player, player-to-move rewards r_t:
    G_t = r_t + gamma * sign(player_t, player_{t+1}) * G_{t+1}
where sign is +1 for the same player and -1 after a turn change.
"""

import argparse
from pathlib import Path

import numpy as np

from src.macro.candidate_encoding import MACRO_CANDIDATE_V2_DIM
from src.macro.encoding import MACRO_STATE_DIM


REQUIRED = (
    "format_version", "candidate_encoding_version", "data_kind",
    "state", "next_state", "candidate", "macro_local_reward",
    "player_before", "player_after", "terminated", "turn_index",
    "rollout_terminated", "stop_reason",
)


def signed_executed_returns(reward, player_before, gamma):
    """Discounted returns from each acting player's perspective."""
    reward = np.asarray(reward, dtype=np.float64)
    players = np.asarray(player_before, dtype=np.int64)
    if reward.ndim != 1 or players.shape != reward.shape:
        raise ValueError("reward/player_before must be matching 1D arrays.")
    if not np.all(np.isfinite(reward)):
        raise ValueError("Observed reward must be finite.")
    if np.any((players < 0) | (players > 1)):
        raise ValueError("Player IDs must be 0 or 1.")
    if not np.isfinite(gamma) or not 0.0 <= gamma <= 1.0:
        raise ValueError("gamma must be in [0, 1].")

    output = np.empty(len(reward), dtype=np.float32)
    future = 0.0
    for i in range(len(reward) - 1, -1, -1):
        if i + 1 == len(reward):
            future = float(reward[i])
        else:
            turn_sign = 1.0 if players[i] == players[i + 1] else -1.0
            future = float(reward[i]) + gamma * turn_sign * future
        output[i] = future
    return output


def load_completed_rollout(path, gamma):
    """Strictly validate the saved experience schema and completion."""
    with np.load(path, allow_pickle=False) as data:
        missing = [key for key in REQUIRED if key not in data]
        if missing:
            raise ValueError(
                f"Rollout {path}: missing fields: {', '.join(missing)}"
            )
        if int(data["format_version"]) != 1 or (
            int(data["candidate_encoding_version"]) != 2
        ):
            raise ValueError(f"Rollout {path}: expected format v1 / 31D v2.")
        if str(data["data_kind"]) != "executed_mixed_rollout":
            raise ValueError(f"Rollout {path}: unexpected data_kind.")

        state = np.asarray(data["state"], dtype=np.float32)
        next_state = np.asarray(data["next_state"], dtype=np.float32)
        candidate = np.asarray(data["candidate"], dtype=np.float32)
        reward = np.asarray(data["macro_local_reward"], dtype=np.float32)
        player_before = np.asarray(data["player_before"], dtype=np.int32)
        player_after = np.asarray(data["player_after"], dtype=np.int32)
        terminated = np.asarray(data["terminated"], dtype=np.bool_)
        turn_index = np.asarray(data["turn_index"], dtype=np.int32)
        n = len(reward)

        if n == 0:
            raise ValueError(f"Rollout {path}: no executed turns.")
        if state.shape != (n, MACRO_STATE_DIM) or (
            next_state.shape != (n, MACRO_STATE_DIM)
            or candidate.shape != (n, MACRO_CANDIDATE_V2_DIM)
        ):
            raise ValueError(f"Rollout {path}: invalid feature shapes.")
        if (
            player_before.shape != (n,)
            or player_after.shape != (n,)
            or terminated.shape != (n,)
            or turn_index.shape != (n,)
        ):
            raise ValueError(f"Rollout {path}: inconsistent turn arrays.")
        if not np.array_equal(turn_index, np.arange(n)):
            raise ValueError(f"Rollout {path}: turn_index is not consecutive.")
        if not (
            np.all(np.isfinite(state))
            and np.all(np.isfinite(next_state))
            and np.all(np.isfinite(candidate))
        ):
            raise ValueError(f"Rollout {path}: non-finite features.")
        if np.any((player_after < 0) | (player_after > 1)):
            raise ValueError(f"Rollout {path}: invalid next-player IDs.")
        if np.any(terminated[:-1]):
            raise ValueError(f"Rollout {path}: terminal flag before last step.")
        if n > 1:
            if not np.array_equal(player_after[:-1], player_before[1:]):
                raise ValueError(f"Rollout {path}: player continuity mismatch.")
            if not np.allclose(next_state[:-1], state[1:], atol=1e-6):
                raise ValueError(f"Rollout {path}: state continuity mismatch.")
        if not (
            bool(data["rollout_terminated"])
            and str(data["stop_reason"]) == "game_terminated"
            and bool(terminated[-1])
        ):
            raise ValueError(
                f"Rollout {path}: incomplete trajectory. "
                "Do not label a truncated episode with zero bootstrap."
            )

        return {
            "state": state.copy(),
            "candidate": candidate.copy(),
            "q_target": signed_executed_returns(
                reward, player_before, gamma
            ),
            "macro_local_reward": reward.copy(),
            "player_before": player_before.copy(),
            "player_after": player_after.copy(),
        }


def build_return_dataset(input_paths, output, gamma=0.99):
    if not input_paths:
        raise ValueError("At least one rollout is required.")
    if not np.isfinite(gamma) or not 0.0 <= gamma <= 1.0:
        raise ValueError("gamma must be in [0, 1].")
    trajectories = [
        load_completed_rollout(path, gamma) for path in input_paths
    ]
    fields = (
        "state", "candidate", "q_target", "macro_local_reward",
        "player_before", "player_after",
    )
    arrays = {
        key: np.concatenate([x[key] for x in trajectories], axis=0)
        for key in fields
    }
    episode_id = np.concatenate([
        np.full(len(x["q_target"]), i, dtype=np.int32)
        for i, x in enumerate(trajectories)
    ])
    # The existing Q trainer splits by state_id. Each observed
    # transition has exactly ONE labeled action, unlike the
    # original full candidate-oracle dataset.
    state_id = np.arange(len(arrays["q_target"]), dtype=np.int32)

    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        **arrays,
        state_id=state_id,
        episode_id=episode_id,
        candidate_dim=np.array(MACRO_CANDIDATE_V2_DIM, dtype=np.int32),
        candidate_encoding_version=np.array(2, dtype=np.int32),
        turn_aware=np.array(True, dtype=np.bool_),
        target_kind=np.array("selected_action_mc_return"),
        reward_kind=np.array("macro_local_reward"),
        gamma=np.array(gamma, dtype=np.float32),
        source_files=np.asarray([str(x) for x in input_paths]),
    )
    return output, len(trajectories), len(state_id)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--rollout", type=Path, nargs="+", required=True,
        help="One or more COMPLETED executed rollout .npz files.",
    )
    parser.add_argument(
        "--output", type=Path,
        default=Path("macro_selected_return_v2.npz"),
    )
    parser.add_argument("--gamma", type=float, default=0.99)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    path, episodes, transitions = build_return_dataset(
        args.rollout, args.output, gamma=args.gamma
    )
    print("=== Completed mixed rollout -> selected-action returns ===")
    print(f"completed trajectories: {episodes}")
    print(f"labeled executed actions: {transitions}")
    print("target: turn-aware discounted Macro LOCAL reward")
    print("This does NOT label unexecuted root candidates.")
    print(f"output: {path}")
