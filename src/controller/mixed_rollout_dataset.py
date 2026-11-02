"""Save bounded mixed-strategy executed trajectories as typed .npz data.

An EXPERIENCE log is NOT a Q-target dataset. Each transition records the
actually executed candidate and GameState; each root row records the MCTS
candidate distribution from the same decision. No automatic target labels
or bootstrapping is performed.
"""

from pathlib import Path

import numpy as np

from src.macro.candidate_encoding import (
    MACRO_CANDIDATE_V2_DIM,
    encode_macro_candidate_v2,
)
from src.macro.encoding import MACRO_STATE_DIM, encode_game_state


FORMAT_VERSION = 1


def save_mixed_rollout_npz(rollout, output, *, seed=None):
    """Write NPZ without Python object arrays; works on truncated rollouts."""
    records = rollout.records
    n = len(records)

    state = np.empty((n, MACRO_STATE_DIM), dtype=np.float32)
    next_state = np.empty((n, MACRO_STATE_DIM), dtype=np.float32)
    selected_candidate = np.empty(
        (n, MACRO_CANDIDATE_V2_DIM), dtype=np.float32
    )

    # Raw GameState fields retain the pocket *IDs*, which the
    # 12-D state encoder intentionally reduces to pocket flags.
    before_positions = np.empty((n, 3, 2), dtype=np.float32)
    after_positions = np.empty((n, 3, 2), dtype=np.float32)
    before_pockets = np.empty((n, 3), dtype=np.int32)
    after_pockets = np.empty((n, 3), dtype=np.int32)
    before_score = np.empty((n, 2), dtype=np.float32)
    after_score = np.empty((n, 2), dtype=np.float32)

    macro_local_reward = np.empty(n, dtype=np.float32)
    score_delta = np.empty((n, 2), dtype=np.float32)
    player_before = np.empty(n, dtype=np.int32)
    player_after = np.empty(n, dtype=np.int32)
    terminated = np.empty(n, dtype=np.bool_)
    turn_index = np.empty(n, dtype=np.int32)

    root_turn_index = []
    root_candidate = []
    root_visit_count = []
    root_prior = []
    root_predicted_q = []
    root_selected = []

    for i, turn in enumerate(records):
        before = turn.state_before
        after = turn.state_after
        state[i] = encode_game_state(before)
        next_state[i] = encode_game_state(after)
        selected_candidate[i] = encode_macro_candidate_v2(turn.choice)

        before_positions[i] = before.ball_positions
        after_positions[i] = after.ball_positions
        before_pockets[i] = before.ball_pocket_indices
        after_pockets[i] = after.ball_pocket_indices
        before_score[i] = before.score
        after_score[i] = after.score

        macro_local_reward[i] = turn.reward
        score_delta[i] = turn.score_delta
        player_before[i] = turn.player_before
        player_after[i] = turn.player_after
        terminated[i] = turn.terminated
        turn_index[i] = turn.turn_index

        for entry in turn.root_candidates:
            root_turn_index.append(i)
            root_candidate.append(
                encode_macro_candidate_v2(entry.choice)
            )
            root_visit_count.append(entry.visits)
            root_prior.append(entry.prior)
            root_predicted_q.append(
                np.nan
                if entry.predicted_q is None
                else entry.predicted_q
            )
            root_selected.append(entry.selected)

    root_matrix = (
        np.stack(root_candidate).astype(np.float32)
        if root_candidate
        else np.empty((0, MACRO_CANDIDATE_V2_DIM), dtype=np.float32)
    )

    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        format_version=np.array(FORMAT_VERSION, dtype=np.int32),
        candidate_encoding_version=np.array(2, dtype=np.int32),
        data_kind=np.array("executed_mixed_rollout"),
        seed=np.array(-1 if seed is None else seed, dtype=np.int64),
        stop_reason=np.array(rollout.stop_reason),
        rollout_terminated=np.array(rollout.terminated, dtype=np.bool_),
        state=state,
        next_state=next_state,
        candidate=selected_candidate,
        state_positions=before_positions,
        next_state_positions=after_positions,
        state_pocket_indices=before_pockets,
        next_state_pocket_indices=after_pockets,
        state_score=before_score,
        next_state_score=after_score,
        macro_local_reward=macro_local_reward,
        score_delta=score_delta,
        player_before=player_before,
        player_after=player_after,
        terminated=terminated,
        turn_index=turn_index,
        root_turn_index=np.asarray(root_turn_index, dtype=np.int32),
        root_candidate=root_matrix,
        root_visit_count=np.asarray(root_visit_count, dtype=np.int32),
        root_prior=np.asarray(root_prior, dtype=np.float32),
        root_predicted_q=np.asarray(root_predicted_q, dtype=np.float32),
        root_selected=np.asarray(root_selected, dtype=np.bool_),
    )
    return output
