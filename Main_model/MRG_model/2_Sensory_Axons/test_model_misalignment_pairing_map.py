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
from pairing_quality_metrics import compute_pairing_metrics, plot_dx_profile, plot_metrics_summary


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


def collect_main_pairing_names(axon) -> set[str]:
    return {point.name for point in axon.collect_main_path_pairing_points()}


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    summary_rows = []

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

        trunkA = collect_main_pairing_names(model.axonA)
        trunkB = collect_main_pairing_names(model.axonB)

        # Branch-aware invariant: all paired sections must belong to the main path/trunk.
        assert all(row["name_A"] in trunkA for row in rows), f"Non-trunk AxonA section leaked into pairing for {label}"
        assert all(row["name_B"] in trunkB for row in rows), f"Non-trunk AxonB section leaked into pairing for {label}"

        # One-to-one invariant on B side.
        b_names = [row["name_B"] for row in rows]
        assert len(b_names) == len(set(b_names)), f"Duplicate AxonB pair targets found for {label}"
        assert all(row["kind_A"] == row["kind_B"] for row in rows), f"Cross-kind pairs found for {label}"
        assert all(row["pair_key_A"] == row["pair_key_B"] for row in rows), f"Cross-phase pairs found for {label}"

        lstep = float(model.axonA.mrg_params.get("Lstep", 1.0))
        target_dx_um = 0.0 if fraction is None else float(fraction) * lstep
        branch_points_um = list(getattr(model.axonB, "branch_point_distance_um", []) or [])
        branch_window_um = 1.0 * lstep
        summary, per_class_df, pair_df = compute_pairing_metrics(
            rows,
            target_dx_um=target_dx_um,
            eligible_A=len(model._pairing_points_A),
            eligible_B=len(model._pairing_points_B),
            branch_points_um=branch_points_um,
            branch_window_um=branch_window_um,
        )
        summary["case"] = label
        summary["misalignment_fraction"] = fraction
        summary_rows.append(summary)

        assert float(summary["type_match_rate"]) == 1.0, f"Type match rate dropped for {label}"
        assert float(summary["same_phase_rate"]) == 1.0, f"Phase match rate dropped for {label}"
        assert float(summary["main_main_fraction"]) == 1.0, f"Non-main sections entered pairing for {label}"

        csv_path = OUT_DIR / f"model_pairing_{label}.csv"
        txt_path = OUT_DIR / f"model_pairing_{label}.txt"
        png_path = OUT_DIR / f"model_pairing_{label}.png"
        metrics_csv_path = OUT_DIR / f"model_pairing_{label}_metrics.csv"
        per_class_csv_path = OUT_DIR / f"model_pairing_{label}_per_class.csv"
        profile_png_path = OUT_DIR / f"model_pairing_{label}_dx_profile.png"

        pair_df.to_csv(csv_path, index=False)
        pd.DataFrame([summary]).to_csv(metrics_csv_path, index=False)
        per_class_df.to_csv(per_class_csv_path, index=False)

        lines = [f"[{label}] misalignment_fraction={fraction}"]
        lines.append(f"type_match_rate={summary['type_match_rate']:.6f}")
        lines.append(f"same_phase_rate={summary['same_phase_rate']:.6f}")
        lines.append(f"main_main_fraction={summary['main_main_fraction']:.6f}")
        lines.append(f"mae_dx_um={summary['mae_dx_um']:.6f}")
        lines.append(f"max_abs_dx_error_um={summary['max_abs_dx_error_um']:.6f}")
        lines.append(f"max_dx_jump_um={summary['max_dx_jump_um']:.6f}")
        lines.append(f"branch_zone_mae_dx_um={summary['branch_zone_mae_dx_um']}")
        lines.append(f"outside_branch_mae_dx_um={summary['outside_branch_mae_dx_um']}")
        lines.append("")
        for row in pair_df.to_dict(orient="records"):
            lines.append(
                f"{row['pair_index']:04d} | {row['kind_A']:>4}:{row['name_A']:<18} -> "
                f"{row['kind_B']:>4}:{row['name_B']:<18} | dx={row['dx_um']:+.3f} um"
            )
        txt_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

        model.plot_axon_axon_pairing_map(save_path=str(png_path))
        plot_dx_profile(
            pair_df,
            target_dx_um=target_dx_um,
            branch_points_um=branch_points_um,
            branch_window_um=branch_window_um,
            save_path=profile_png_path,
            title=f"{label} | dx profile | target={target_dx_um:.3f} um",
        )

        print(f"Saved {label}: {csv_path}")
        print(f"Saved {label}: {txt_path}")
        print(f"Saved {label}: {png_path}")
        print(f"Saved {label}: {metrics_csv_path}")
        print(f"Saved {label}: {per_class_csv_path}")
        print(f"Saved {label}: {profile_png_path}")

    summary_df = pd.DataFrame(summary_rows)
    summary_csv_path = OUT_DIR / "pairing_metrics_summary.csv"
    summary_df.to_csv(summary_csv_path, index=False)
    plot_metrics_summary(summary_df, OUT_DIR / "pairing_metrics_summary.png")
    print(f"Saved summary: {summary_csv_path}")


if __name__ == "__main__":
    main()
