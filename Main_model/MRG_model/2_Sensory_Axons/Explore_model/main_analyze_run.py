from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from analyze_lib import (
    build_before_after_metrics,
    build_no_ec_delta_summary,
    collect_h5_index,
    export_all_spikes_to_csv,
)
from plot_analyze_lib import (
    plot_following_and_velocity,
    plot_following_by_mode_grid,
    plot_no_ec_delta_by_mode_grid,
    plot_velocity_by_mode_grid,
)


HERE = Path(__file__).resolve().parent
ROOT_DIR = HERE.parents[0]
ROOT_H5 = ROOT_DIR / "final_result"
OUT_DIR = ROOT_DIR / "data" / "analysis_lib_outputs"
PLOT_DIR = OUT_DIR / "plots"

OUT_FILE_INDEX_CSV = OUT_DIR / "h5_file_index.csv"
OUT_SPIKES_CSV = OUT_DIR / "all_h5_spikes.csv"
OUT_AUDIT_CSV = OUT_DIR / "all_h5_spikes_audit.csv"
OUT_MATCHED_CSV = OUT_DIR / "before_after_matched_spikes.csv"
OUT_SUMMARY_CSV = OUT_DIR / "before_after_summary.csv"
OUT_DELTA_CSV = OUT_DIR / "before_after_summary_vs_no_ec.csv"


START_ANALYSIS_MS = 0.0
PEAK_PROMINENCE_MV = 5.0
PEAK_MIN_DISTANCE_MS = 0.6
SPIKE_HEIGHT_THRESHOLDS_MV_BY_DIAMETER = {2.5: -20.0, 5.7: -20.0}
DEFAULT_SPIKE_HEIGHT_MV = -20.0

TRACE_ROLE_MAP = {
    "AxonA": {
        "before_like": "before",
        "main_like": "after_main",
        "before_branch": "before_branch",
        "branch_point": "branch_point",
        "after_branch_main": "after_main",
        "terminal_main": "terminal_main",
    },
    "AxonB": {
        "before_branch": "before",
        "branch_point": "branch_point",
        "after_branch_main": "after_main",
        "terminal_main": "terminal_main",
    },
}

DIAMETERS = [2.5, 5.7]
TOPOLOGIES = ["connector_branching", "one_node_branching"]
SCENARIOS = ["one_branch", "multiple_branches"]


def export_spikes(root_h5: Path) -> pd.DataFrame:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    file_index = collect_h5_index(root_h5)
    file_index.to_csv(OUT_FILE_INDEX_CSV, index=False)
    print("[SAVED]", OUT_FILE_INDEX_CSV)

    spikes_df, audit_df = export_all_spikes_to_csv(
        root=root_h5,
        out_spikes_csv=OUT_SPIKES_CSV,
        out_audit_csv=OUT_AUDIT_CSV,
        trace_role_map=TRACE_ROLE_MAP,
        start_analysis_ms=START_ANALYSIS_MS,
        peak_min_distance_ms=PEAK_MIN_DISTANCE_MS,
        peak_prominence_mV=PEAK_PROMINENCE_MV,
        spike_thresholds_by_diameter=SPIKE_HEIGHT_THRESHOLDS_MV_BY_DIAMETER,
        default_spike_height_mV=DEFAULT_SPIKE_HEIGHT_MV,
    )
    print("[INFO] audit rows:", len(audit_df))
    return spikes_df


def build_metrics(spikes_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    matched_df, summary_df = build_before_after_metrics(spikes_df)
    delta_df = build_no_ec_delta_summary(summary_df)

    matched_df.to_csv(OUT_MATCHED_CSV, index=False)
    summary_df.to_csv(OUT_SUMMARY_CSV, index=False)
    delta_df.to_csv(OUT_DELTA_CSV, index=False)

    print("[SAVED]", OUT_MATCHED_CSV)
    print("[SAVED]", OUT_SUMMARY_CSV)
    print("[SAVED]", OUT_DELTA_CSV)
    return matched_df, summary_df, delta_df


def build_all_plots(summary_df: pd.DataFrame, delta_df: pd.DataFrame) -> None:
    PLOT_DIR.mkdir(parents=True, exist_ok=True)

    for diameter in DIAMETERS:
        for topology in TOPOLOGIES:
            for scenario in SCENARIOS:
                # Old view: columns are distances, lines are modes.
                plot_following_and_velocity(
                    summary_df,
                    fiber_diameter_um=diameter,
                    topology=topology,
                    scenario=scenario,
                    out_dir=PLOT_DIR / "by_distance",
                )

                # New view: columns are modes, lines are distances.
                plot_following_by_mode_grid(
                    summary_df,
                    fiber_diameter_um=diameter,
                    topology=topology,
                    scenario=scenario,
                    include_no_ec=False,
                    out_dir=PLOT_DIR / "by_mode_3_modes",
                )
                plot_velocity_by_mode_grid(
                    summary_df,
                    fiber_diameter_um=diameter,
                    topology=topology,
                    scenario=scenario,
                    include_no_ec=False,
                    out_dir=PLOT_DIR / "by_mode_3_modes",
                )

                # New view with no_EC and no_EC_isolated controls.
                plot_following_by_mode_grid(
                    summary_df,
                    fiber_diameter_um=diameter,
                    topology=topology,
                    scenario=scenario,
                    include_no_ec=True,
                    out_dir=PLOT_DIR / "by_mode_with_no_ec",
                )
                plot_velocity_by_mode_grid(
                    summary_df,
                    fiber_diameter_um=diameter,
                    topology=topology,
                    scenario=scenario,
                    include_no_ec=True,
                    out_dir=PLOT_DIR / "by_mode_with_no_ec",
                )

                # Delta vs no_EC_isolated: columns are non-isolated modes, lines are distances.
                tag = f"fd{diameter}_{topology}_{scenario}"
                plot_no_ec_delta_by_mode_grid(
                    delta_df,
                    fiber_diameter_um=diameter,
                    topology=topology,
                    scenario=scenario,
                    delta_metric_col="delta_following_fraction_terminal_vs_no_ec_isolated",
                    metric_label="Delta доли следования на terminal_main vs no_EC_isolated",
                    y_lim=(-1.05, 1.05),
                    out_path=PLOT_DIR / "delta_vs_no_ec_isolated" / f"{tag}_delta_following.png",
                )
                plot_no_ec_delta_by_mode_grid(
                    delta_df,
                    fiber_diameter_um=diameter,
                    topology=topology,
                    scenario=scenario,
                    delta_metric_col="delta_median_terminal_velocity_m_s_vs_no_ec_isolated",
                    metric_label="Delta скорости до terminal_main vs no_EC_isolated, м/с",
                    y_lim=None,
                    scale_y_by_axon=True,
                    out_path=PLOT_DIR / "delta_vs_no_ec_isolated" / f"{tag}_delta_velocity.png",
                )


# ============================================================
# ЗАПУСК АНАЛИЗА
# ============================================================

parser = argparse.ArgumentParser(description="Export spikes, compute branch metrics, and build plots.")
parser.add_argument("--skip-export", action="store_true", help="Use existing all_h5_spikes.csv")
parser.add_argument("--no-plots", action="store_true", help="Only write CSV metrics")
args = parser.parse_args()

if args.skip_export:
    spikes_df = pd.read_csv(OUT_SPIKES_CSV)
    print("[LOAD]", OUT_SPIKES_CSV)
else:
    spikes_df = export_spikes(ROOT_H5)

_matched_df, summary_df, delta_df = build_metrics(spikes_df)

if not args.no_plots:
    build_all_plots(summary_df, delta_df)

print("[DONE] analysis complete")
