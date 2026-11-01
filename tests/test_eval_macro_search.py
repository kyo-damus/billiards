import numpy as np

from scripts.eval_macro_search import (
    summarize,
)


def test_summarize_search_records():

    records = [
        {
            "selected_q": 2.0,
            "best_q": 2.0,
            "regret": 0.0,
            "optimal": 1.0,
            "position_selected": 1.0,
        },
        {
            "selected_q": 1.0,
            "best_q": 2.0,
            "regret": 1.0,
            "optimal": 0.0,
            "position_selected": 0.0,
        },
    ]

    metrics = summarize(
        records
    )

    assert np.isclose(
        metrics["optimal_rate"],
        0.5,
    )
    assert np.isclose(
        metrics["mean_regret"],
        0.5,
    )
    assert np.isclose(
        metrics["mean_selected_q"],
        1.5,
    )
    assert np.isclose(
        metrics[
            "position_selection_rate"
        ],
        0.5,
    )
