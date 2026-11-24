"""Compatibility CLI: grouped runner lives in scripts.comparisons."""
from scripts.comparisons.eval_nine_ball_baselines import (
    BASELINES, _agents, _check_seed_isolation, evaluate, parse_args,
)

if __name__ == "__main__":
    evaluate(parse_args())
