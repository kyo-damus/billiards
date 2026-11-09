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



def test_resolve_simulation_budgets_uses_multi_budget_list():
    from scripts.eval_macro_search import (
        resolve_simulation_budgets,
    )

    budgets = resolve_simulation_budgets(
        simulations=50,
        simulation_budgets=[
            5,
            10,
            20,
            50,
            20,
        ],
    )

    assert budgets == [
        5,
        10,
        20,
        50,
    ]


def test_resolve_simulation_budgets_falls_back_to_single_budget():
    from scripts.eval_macro_search import (
        resolve_simulation_budgets,
    )

    assert (
        resolve_simulation_budgets(
            simulations=25,
            simulation_budgets=None,
        )
        == [25]
    )
