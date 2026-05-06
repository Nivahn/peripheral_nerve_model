"""test_model_misalignment_pairing_map.py

Model-level diagnostics for branch-aware misalignment pairing.

This test does not run stimulation. It only builds the two-axon model,
extracts the axon-axon pairing map, checks a few structural invariants,
and saves visual/text diagnostics for manual inspection.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from MRG_lib import TwoSensoryAxonsPrescott


OUT_DIR = Path(__file__).resolve().parent / "data" / "misalignment_pairing_tests"

MODEL_KWARGS = {
    "fiber_diameter_um": 5.7,
    "edge_dist_um": 0.1,
    "aligned": True,
    "enable_ephaptic": True,
    "parent_axon_nodes_A": 42,
    "branch_nodes_A": 11,
    "branches_num_A": 0,
    "nodes_dist_A": 10,
    "parent_axon_nodes_B": 42,
    "branch_nodes_B": 11,
    "branches_num_B": 1,
    "nodes_dist_B": 10,
    "diam_scale": 0.6,
    "dt_ms": 0.005,
    "h_stop": 120.0,
    "boundary_full_cable": False,
}

CASES = [
    ("aligned", None),
    ("quarter_step", 0.25),
    ("half_step", 0.5),
]


def collect_trunk_section_names(axon) -> set[str]:
    names = set()
    for sec in list(axon.main_axon):
        names.add(sec.name())
    for rec in getattr(axon, "_step_records", []) or []:
        if not rec.get("is_trunk", False):
            continue
        parent = rec.get("parent", None)
        next_name = rec.get("next", None)
        if parent is not None:
            names.add(str(parent))
        if next_name is not None:
            names.add(str(next_name))
        for seg in rec.get("sections", []) or []:
            names.add(str(seg.get("name", "")))
    return names


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    for label, fraction in CASES:
        kwargs = dict(MODEL_KWARGS)
        kwargs["misalignment_fraction"] = fraction
        if fraction is None:
            kwargs["aligned"] = True
        else:
            kwargs["aligned"] = False

        model = TwoSensoryAxonsPrescott(**kwargs)
        rows = model.get_axon_axon_pairing_rows()
        assert rows, f"No pairing rows produced for {label}"

        trunkA = collect_trunk_section_names(model.axonA)
        trunkB = collect_trunk_section_names(model.axonB)

        # Branch-aware invariant: all paired sections must belong to the main path/trunk.
        assert all(row["name_A"] in trunkA for row in rows), f"Non-trunk AxonA section leaked into pairing for {label}"
        assert all(row["name_B"] in trunkB for row in rows), f"Non-trunk AxonB section leaked into pairing for {label}"

        # One-to-one invariant on B side.
        b_names = [row["name_B"] for row in rows]
        assert len(b_names) == len(set(b_names)), f"Duplicate AxonB pair targets found for {label}"

        csv_path = OUT_DIR / f"model_pairing_{label}.csv"
        txt_path = OUT_DIR / f"model_pairing_{label}.txt"
        png_path = OUT_DIR / f"model_pairing_{label}.png"

        pd.DataFrame(rows).to_csv(csv_path, index=False)

        lines = [f"[{label}] misalignment_fraction={fraction}"]
        for row in rows:
            lines.append(
                f"{row['pair_index']:04d} | {row['kind_A']:>4}:{row['name_A']:<18} -> "
                f"{row['kind_B']:>4}:{row['name_B']:<18} | dx={row['dx_um']:+.3f} um"
            )
        txt_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

        model.plot_axon_axon_pairing_map(save_path=str(png_path))

        print(f"Saved {label}: {csv_path}")
        print(f"Saved {label}: {txt_path}")
        print(f"Saved {label}: {png_path}")


if __name__ == "__main__":
    main()
