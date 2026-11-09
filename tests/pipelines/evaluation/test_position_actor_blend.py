from scripts.eval_position_actor_blend import (
    parse_lambdas,
)


def test_parse_lambdas():

    values = parse_lambdas(
        "0,0.25,1"
    )

    assert values == [
        0.0,
        0.25,
        1.0,
    ]


def test_parse_lambdas_rejects_out_of_range():

    try:
        parse_lambdas(
            "0,1.1"
        )
    except ValueError:
        return

    assert False, (
        "ValueError was not raised"
    )
