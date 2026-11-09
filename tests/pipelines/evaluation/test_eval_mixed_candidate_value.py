import numpy as np
import pytest

from scripts.eval_mixed_candidate_value import (
    inspect_mixed_metadata,
    ranking_summary,
    validate_heldout_seeds,
)


def test_ranking_summary_handles_strategy_comparison():
    # Two boards: Safety is truly best on first, but not on second.
    # Predictor selects Safety on both boards.
    result = ranking_summary(
        prediction=np.array([0.0, 1.0, 1.0, 2.0]),
        target=np.array([1.0, 2.0, 3.0, 1.0]),
        state_ids=np.array([0, 0, 1, 1]),
        strategies=np.array([0, 1, 0, 1]),
    )
    assert result["boards"] == 2
    assert result["boards_with_safety"] == 2
    assert result["selected_safety"] == 2
    assert result["oracle_safety_competitive"] == 1
    assert result["top1_hits"] == 1
    assert np.isclose(result["mean_regret"], 1.0)


def test_heldout_seed_intervals_reject_overlap():
    assert validate_heldout_seeds(93000, 20, 94000, 5)
    with pytest.raises(ValueError, match="overlap"):
        validate_heldout_seeds(93000, 20, 93019, 5)
    assert not validate_heldout_seeds(None, None, 94000, 5)


def test_inspect_mixed_metadata_validates_version_and_strategies(
    tmp_path,
):
    good = tmp_path / "good.npz"
    np.savez_compressed(
        good,
        candidate_encoding_version=np.array(2),
        turn_aware=np.array(True),
        strategy=np.array([0, 1, 0]),
        seed_start=np.array(94000),
        requested_states=np.array(5),
    )
    strategy, seed, states = inspect_mixed_metadata(good)
    np.testing.assert_array_equal(strategy, [0, 1, 0])
    assert seed == 94000
    assert states == 5

    bad = tmp_path / "bad.npz"
    np.savez_compressed(
        bad,
        candidate_encoding_version=np.array(2),
        turn_aware=np.array(False),
        strategy=np.array([0, 1]),
    )
    with pytest.raises(ValueError, match="turn_aware"):
        inspect_mixed_metadata(bad)


def test_ranking_summary_with_no_safety():
    result = ranking_summary(
        prediction=np.array([1.0, 2.0]),
        target=np.array([0.0, 1.0]),
        state_ids=np.array([0, 1]),
        strategies=np.array([0, 0]),
    )
    assert result["boards_with_safety"] == 0
    assert result["selected_safety"] == 0
    assert np.isclose(result["mean_regret"], 0.0)
