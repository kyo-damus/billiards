"""Iteration wiring tests: no FastFiz or GPU required."""

from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts.iterate_mixed_macro_policy import run_policy_iterations


def args(tmp_path, *, iterations=2, games=2):
    return SimpleNamespace(
        iterations=iterations,
        games_per_iteration=games,
        start_seed=95000,
        max_turns=2,
        simulations=3,
        policy_epochs=2,
        validation_fraction=0.5,
        output_dir=tmp_path / "iteration",
        candidate_policy_checkpoint="original_policy.pt",
        device="cpu",
        gamma=0.99,
        hidden_dim=16,
        learning_rate=1e-3,
        training_seed=123,
    )


def make_fakes():
    seen = {"inputs": [], "seeds": [], "visits": [], "training": []}

    def build_agent(config, device):
        seen["inputs"].append(config.candidate_policy_checkpoint)
        assert device == "cpu"
        return object()

    def collect(agent, factory, **kwargs):
        seen["seeds"].append(
            list(range(
                kwargs["start_seed"],
                kwargs["start_seed"] + kwargs["games"]
            ))
        )
        folder = Path(kwargs["output_dir"])
        folder.mkdir(parents=True)
        for seed in seen["seeds"][-1]:
            assert factory(seed) == seed
            (folder / f"rollout_seed_{seed}.npz").write_bytes(b"log")
        return SimpleNamespace(saved_games=kwargs["games"])

    def visit(paths, output):
        seen["visits"].append([p.name for p in paths])
        assert all(p.exists() for p in paths)
        Path(output).write_bytes(b"visits")
        return output, 4, 40, 0

    def train(config):
        assert Path(config.dataset).exists()
        seen["training"].append(
            (config.seed, config.epochs, config.hidden_dim)
        )
        Path(config.checkpoint).write_bytes(b"policy")

    return seen, build_agent, collect, visit, train


def run_fake(config):
    seen, agent, collect, visit, train = make_fakes()
    records = run_policy_iterations(
        config,
        agent_builder=agent,
        collector=collect,
        visit_builder=visit,
        policy_trainer=train,
        state_factory=lambda seed: seed,
    )
    return records, seen


def test_two_iterations_use_new_policy_and_independent_seed_batches(tmp_path):
    records, seen = run_fake(args(tmp_path))
    root = tmp_path / "iteration"
    first = root / "iteration_000" / "candidate_policy.pt"
    second = root / "iteration_001" / "candidate_policy.pt"

    assert len(records) == 2
    assert seen["inputs"] == ["original_policy.pt", str(first)]
    assert seen["seeds"] == [[95000, 95001], [95002, 95003]]
    assert seen["training"] == [(123, 2, 16), (124, 2, 16)]
    assert records[0].output_policy == first
    assert records[1].output_policy == second
    assert records[0].visited_roots == 4
    assert records[1].candidate_rows == 40
    assert first.exists() and second.exists()


def test_preflight_refuses_existing_future_iteration_outputs(tmp_path):
    config = args(tmp_path)
    future = (
        config.output_dir / "iteration_001" / "rollouts"
        / "rollout_seed_95003.npz"
    )
    future.parent.mkdir(parents=True)
    future.write_bytes(b"prior")
    seen, agent, collect, visit, train = make_fakes()

    with pytest.raises(FileExistsError, match="Existing iteration files"):
        run_policy_iterations(
            config, agent_builder=agent, collector=collect,
            visit_builder=visit, policy_trainer=train,
            state_factory=lambda seed: seed,
        )
    assert seen["inputs"] == []
    assert seen["seeds"] == []
    assert future.read_bytes() == b"prior"


def test_iterations_require_two_games_for_episode_split(tmp_path):
    with pytest.raises(ValueError, match=">=2 games"):
        run_policy_iterations(
            args(tmp_path, games=1),
            state_factory=lambda seed: seed,
        )


def test_trainer_checkpoint_is_required(tmp_path):
    config = args(tmp_path, iterations=1)
    seen, agent, collect, visit, train = make_fakes()

    def does_not_save(config):
        assert Path(config.dataset).exists()

    with pytest.raises(RuntimeError, match="did not save"):
        run_policy_iterations(
            config, agent_builder=agent, collector=collect,
            visit_builder=visit, policy_trainer=does_not_save,
            state_factory=lambda seed: seed,
        )
