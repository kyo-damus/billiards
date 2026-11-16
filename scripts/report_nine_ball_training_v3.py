"""Summarize saved nine-ball rollout and learning logs without a GPU."""

import argparse
from collections import Counter
import json
from pathlib import Path

from src.game.nine_ball_encoding_v3 import GOAL_KINDS, RULE_ACTION_CODES


RULE_GOALS_BY_CODE = {code: kind for kind, code in RULE_ACTION_CODES.items()}


def goal_kind(vector):
    if vector is None:
        return "unencoded"
    category = GOAL_KINDS[vector[:4].index(1.0)]
    return RULE_GOALS_BY_CODE.get(vector[28], category) if category == "SAFETY" else category


def report_round(folder):
    stops, breaks, selected, candidates, visits = (Counter() for _ in range(5))
    rollouts = folder / "rollouts"
    shards = sorted(rollouts.glob("match_seed_*.json"))
    racks = shots = bytes_written = 0
    max_rack_shots = 0
    for shard in shards:
        payload = json.loads(shard.read_text(encoding="utf-8"))
        stops[payload["stop_reason"]] += 1
        racks += payload["racks_completed"]
        shots += payload["shots"]
        bytes_written += shard.stat().st_size
        failure = payload.get("physics_failure")
        if failure is not None and failure["state"]["break_shot"]:
            breaks["physics_rejected"] += 1
        rack_shots = Counter()
        for row in payload["trajectory"]:
            if "physical" in row:
                rack_shots[row["rack_index"]] += 1
            action = row["chosen_action"]
            selected[action.get("rule_action") or goal_kind(action.get("goal_v3"))] += 1
            search = action.get("macro_search")
            if search is not None:
                vectors = search["goal_v3"]
                reasons = search.get("root_invalid_reasons", [None] * len(vectors))
                for vector, count, reason in zip(vectors, search["root_visits"], reasons, strict=True):
                    if reason is None:
                        kind = goal_kind(vector)
                        candidates[kind] += 1
                        visits[kind] += count
            if "physical" in row and row["state"]["break_shot"]:
                result = row["rule_result"]
                if result["foul"]:
                    breaks["foul"] += 1
                elif result.get("illegal_break", False):
                    breaks["three_ball_short"] += 1
                else:
                    breaks["legal"] += 1
                    breaks["legal_with_pot"] += bool(result["newly_pocketed"])
        max_rack_shots = max(max_rack_shots, max(rack_shots.values(), default=0))
    executed = breaks["legal"] + breaks["three_ball_short"] + breaks["foul"]
    micro = folder / "micro" / "summary.json"
    micro_steps = None
    if micro.is_file():
        micro_steps = json.loads(micro.read_text(encoding="utf-8")).get("accepted_steps_by_goal")
    return {
        "matches_logged": len(shards),
        "collector_summary_saved": (rollouts / "summary.json").is_file(),
        "stop_reasons": dict(stops),
        "racks_completed": racks,
        "shots_logged": shots,
        "max_rack_shots_logged": max_rack_shots,
        "match_json_bytes": bytes_written,
        "breaks": {
            "executed": executed,
            **{key: breaks[key] for key in (
                "legal", "legal_with_pot", "three_ball_short", "foul", "physics_rejected",
            )},
            "legal_rate_executed": breaks["legal"] / executed if executed else None,
        },
        "macro_candidates_without_invalid_reason_by_goal": dict(candidates),
        "macro_search_visits_by_goal": dict(visits),
        "macro_selected_by_goal": dict(selected),
        "micro_accepted_steps_by_goal": micro_steps,
    }


def report_run(root):
    root = Path(root)
    manifest = json.loads((root / "run_manifest.json").read_text(encoding="utf-8"))
    stages = manifest["stages"]
    return {
        "run_status": manifest["status"],
        "wpa_break_rules": manifest["settings"].get("wpa_break_rules", False),
        "race_to": manifest["settings"].get("race_to", 1),
        "stages_completed": sum(stage["status"] == "completed" for stage in stages),
        "stages_total": len(stages),
        "stage_elapsed_seconds": {
            stage["name"]: stage["elapsed_seconds"] for stage in stages if "elapsed_seconds" in stage
        },
        "rounds": {
            f"iteration_{info['iteration']:03d}": report_round(root / f"iteration_{info['iteration']:03d}")
            for info in manifest["rounds"]
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(report_run(args.run_root), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
