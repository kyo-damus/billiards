"""Reserve evaluation rack orders outside listed legacy training seed ranges."""

import argparse
import hashlib
import json
from pathlib import Path

from src.game.nine_ball_rack_split import ALL_ORDERS, legacy_rack_order
from src.pipelines.experiment_io import write_experiment_json


def prepare(args):
    if args.heldout < 1 or args.split_seed < 0:
        raise ValueError("Use a positive --heldout and nonnegative --split-seed")
    ranges = [tuple(pair) for pair in args.exclude_seed_range]
    for start, count in ranges:
        if start < 0 or count < 1:
            raise ValueError("--exclude-seed-range requires nonnegative start and positive count")

    excluded_seeds = set()
    sources = []
    for file in args.exclude_training_manifest:
        raw = file.read_bytes()
        manifest = json.loads(raw)
        if (
            manifest.get("kind") != "nine_ball_macro_micro_alternating_v3"
            or manifest.get("status") != "completed"
            or manifest.get("rack_layout_split")
            or not manifest.get("rounds")
        ):
            raise ValueError(f"Expected a completed legacy training manifest: {file}")
        sources.append({"path": str(file), "sha256": hashlib.sha256(raw).hexdigest()})
        for rnd in manifest["rounds"]:
            start = rnd["start_seed"]
            stride = 2 * rnd.get("race_to", manifest["settings"].get("race_to", 1))
            for match_index in range(rnd["matches"]):
                first = start + stride * match_index
                excluded_seeds.update(range(first, first + stride - 1))
    for start, count in ranges:
        excluded_seeds.update(range(start, start + count))

    seen = {legacy_rack_order(seed) for seed in excluded_seeds}
    candidates = [order for order in ALL_ORDERS if order not in seen]
    if args.heldout > min(len(candidates), len(ALL_ORDERS) - 1):
        raise ValueError(f"Only {len(candidates)} eligible evaluation orders")
    candidates.sort(key=lambda order: hashlib.sha256(
        f"{args.split_seed}:".encode("ascii") + bytes(order),
    ).digest())
    payload = {
        "kind": "nine_ball_rack_order_split_v1",
        "split_seed": args.split_seed,
        "excluded_training_manifests": sources,
        "excluded_legacy_seed_ranges": ranges,
        "excluded_layout_count": len(seen),
        "evaluation_orders": candidates[:args.heldout],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_experiment_json(args.output, payload)
    return {
        "output": str(args.output), "training_layouts": len(ALL_ORDERS) - args.heldout,
        "evaluation_layouts": args.heldout, "excluded_prior_layouts": len(seen),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--heldout", type=int, default=1024)
    parser.add_argument("--split-seed", type=int, default=960000)
    parser.add_argument("--exclude-training-manifest", type=Path, action="append", default=[])
    parser.add_argument("--exclude-seed-range", type=int, nargs=2, action="append", default=[],
                        metavar=("START", "COUNT"))
    print(json.dumps(prepare(parser.parse_args()), indent=2))


if __name__ == "__main__":
    main()
