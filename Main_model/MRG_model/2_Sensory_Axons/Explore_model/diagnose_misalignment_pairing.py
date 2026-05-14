from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from MRG_lib import TwoSensoryAxonsPrescott  # noqa: E402


MODES = ("aligned", "misaligned_0.5", "misaligned_0.25", "no_EC")


def collect_pairing_rows(*, fiber_diameter_um: float, edge_dist_um: float, scenario: str, mode: str) -> list[dict]:
    if scenario == "one_branch":
        parent_nodes = 17
        branches = 1
        nodes_dist = 10
    else:
        parent_nodes = 29
        branches = 4
        nodes_dist = 8

    model = TwoSensoryAxonsPrescott(
        fiber_diameter_um=float(fiber_diameter_um),
        edge_dist_um=float(edge_dist_um),
        parent_axon_nodes_A=parent_nodes,
        branch_nodes_A=8,
        branches_num_A=0,
        parent_axon_nodes_B=parent_nodes,
        branch_nodes_B=8,
        branches_num_B=branches,
        nodes_dist_B=nodes_dist,
        dt_ms=0.01,
        h_stop=10.0,
        mode_descriptor=mode,
    )
    rows = model.get_axon_axon_pairing_rows()
    for row in rows:
        row["fiber_diameter_um"] = float(fiber_diameter_um)
        row["edge_dist_um"] = float(edge_dist_um)
        row["scenario"] = str(scenario)
        row["mode"] = str(mode)
        row["offset_B_um"] = float(getattr(model, "_offsetB_um", np.nan))
        row["same_kind"] = int(str(row.get("kind_A", "")) == str(row.get("kind_B", "")))
        row["same_pair_key"] = int(str(row.get("pair_key_A", "")) == str(row.get("pair_key_B", "")))
    return rows


def summarize(rows: list[dict]) -> dict:
    if not rows:
        return {"n_pairs": 0, "n_cross_kind": 0, "n_cross_key": 0, "mean_abs_dx_um": np.nan, "max_abs_dx_um": np.nan}
    dx = np.asarray([abs(float(row["dx_um"])) for row in rows], dtype=float)
    return {
        "n_pairs": int(len(rows)),
        "n_cross_kind": int(sum(1 for row in rows if not int(row["same_kind"]))),
        "n_cross_key": int(sum(1 for row in rows if not int(row["same_pair_key"]))),
        "mean_abs_dx_um": float(np.nanmean(dx)),
        "max_abs_dx_um": float(np.nanmax(dx)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Diagnose one_node_branching aligned/misaligned ephaptic pairing maps.")
    parser.add_argument("--fiber-diameter-um", type=float, default=5.7)
    parser.add_argument("--edge-dist-um", type=float, default=0.1)
    parser.add_argument("--scenario", default="one_branch", choices=("one_branch", "multiple_branches"))
    parser.add_argument("--out-csv", type=Path, default=None)
    args = parser.parse_args()

    all_rows: list[dict] = []
    for mode in MODES:
        rows = collect_pairing_rows(
            fiber_diameter_um=args.fiber_diameter_um,
            edge_dist_um=args.edge_dist_um,
            scenario=args.scenario,
            mode=mode,
        )
        all_rows.extend(rows)
        s = summarize(rows)
        print(
            f"{mode}: pairs={s['n_pairs']} cross_kind={s['n_cross_kind']} "
            f"cross_key={s['n_cross_key']} mean_abs_dx_um={s['mean_abs_dx_um']:.6g} "
            f"max_abs_dx_um={s['max_abs_dx_um']:.6g}"
        )

    if args.out_csv is not None:
        args.out_csv.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = sorted({key for row in all_rows for key in row.keys()})
        with args.out_csv.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_rows)
        print(f"wrote {args.out_csv}")


if __name__ == "__main__":
    main()
