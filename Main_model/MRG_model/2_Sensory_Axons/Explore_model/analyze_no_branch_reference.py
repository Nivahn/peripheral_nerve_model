from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from analyze_lib import (
    build_before_after_metrics,
    build_no_ec_delta_summary,
    collect_h5_index,
    export_all_spikes_to_csv,
)
from plot_analyze_lib import (
    plot_following_by_mode_grid,
    plot_following_and_velocity,
    plot_metric_grid_2x3,
    plot_metric_by_mode_grid,
    plot_no_ec_delta_by_mode_grid,
    plot_velocity_by_mode_grid,
)


HERE = Path(__file__).resolve().parent
ROOT_DIR = HERE.parents[0]
ROOT_H5 = ROOT_DIR / "final_result"
OUT_DIR = ROOT_DIR / "data" / "analysis_no_branch_reference"
PLOT_DIR = OUT_DIR / "plots"

NO_BRANCH_SCENARIO = "no_branch_reference"
BRANCH_SCENARIO = "one_branch"
COMPARISON_SCENARIO = "one_branch_vs_no_branch_reference"
TOPOLOGIES = ["one_node_branching"]
DIAMETERS = [2.5, 5.7]
METRIC_COLS = [
    "median_terminal_velocity_m_s",
    "median_terminal_latency_ms",
    "following_fraction_terminal",
]

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


def apply_filters(df: pd.DataFrame, *, stim_protocol: str, test_mode: str) -> pd.DataFrame:
    out = df.copy()
    if "stim_protocol" in out.columns:
        out = out[out["stim_protocol"].astype(str) == str(stim_protocol)]
    if test_mode != "include" and "test_mode" in out.columns:
        values = pd.to_numeric(out["test_mode"], errors="coerce").fillna(0).astype(int)
        if test_mode == "exclude":
            out = out[values == 0]
        elif test_mode == "only":
            out = out[values == 1]
    return out.copy()


def pct_change(delta: pd.Series, baseline: pd.Series) -> pd.Series:
    base = pd.to_numeric(baseline, errors="coerce")
    out = pd.to_numeric(delta, errors="coerce") / base * 100.0
    return out.where(np.isfinite(out) & (base != 0.0))


def build_branch_vs_no_branch_summary(summary_df: pd.DataFrame) -> pd.DataFrame:
    key_cols = [
        "topology",
        "fiber_diameter_um",
        "edge_dist_um",
        "mode_norm",
        "stim_protocol",
        "stim_B_delay_ms",
        "freq_hz",
        "axon",
    ]
    no_branch = summary_df[summary_df["scenario"] == NO_BRANCH_SCENARIO].copy()
    branch = summary_df[summary_df["scenario"] == BRANCH_SCENARIO].copy()
    if no_branch.empty or branch.empty:
        return pd.DataFrame()

    base_cols = key_cols + [col for col in METRIC_COLS if col in no_branch.columns]
    base = no_branch[base_cols].copy()
    base = base.rename(columns={col: f"{col}_no_branch" for col in METRIC_COLS if col in base.columns})

    out = branch.merge(base, on=key_cols, how="inner")
    out["scenario"] = COMPARISON_SCENARIO
    for col in METRIC_COLS:
        base_col = f"{col}_no_branch"
        if col in out.columns and base_col in out.columns:
            delta_col = f"delta_{col}_one_branch_vs_no_branch"
            out[delta_col] = out[col] - out[base_col]
            out[f"{delta_col}_pct"] = pct_change(out[delta_col], out[base_col])
    return out


def export_spikes(args: argparse.Namespace) -> pd.DataFrame:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    file_index = collect_h5_index(ROOT_H5)
    file_index = file_index[file_index["scenario"].isin([NO_BRANCH_SCENARIO, BRANCH_SCENARIO])].copy()
    file_index = apply_filters(file_index, stim_protocol=args.stim_protocol, test_mode=args.test_mode)
    file_index.to_csv(OUT_DIR / "h5_file_index.csv", index=False)
    print("[SAVED]", OUT_DIR / "h5_file_index.csv")

    spikes_df, audit_df = export_all_spikes_to_csv(
        root=ROOT_H5,
        out_spikes_csv=OUT_DIR / "spikes_all.csv",
        out_audit_csv=OUT_DIR / "spikes_audit_all.csv",
        trace_role_map=TRACE_ROLE_MAP,
    )
    spikes_df = spikes_df[spikes_df["scenario"].isin([NO_BRANCH_SCENARIO, BRANCH_SCENARIO])].copy()
    spikes_df = apply_filters(spikes_df, stim_protocol=args.stim_protocol, test_mode=args.test_mode)
    spikes_df.to_csv(OUT_DIR / "spikes.csv", index=False)
    audit_df.to_csv(OUT_DIR / "spikes_audit.csv", index=False)
    print("[SAVED]", OUT_DIR / "spikes.csv")
    print("[SAVED]", OUT_DIR / "spikes_audit.csv")
    return spikes_df


def build_outputs(spikes_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    matched_df, summary_df = build_before_after_metrics(spikes_df)
    no_branch_summary = summary_df[summary_df["scenario"] == NO_BRANCH_SCENARIO].copy()
    no_branch_delta = build_no_ec_delta_summary(no_branch_summary, metric_cols=METRIC_COLS)
    branch_vs_no_branch = build_branch_vs_no_branch_summary(summary_df)

    matched_df.to_csv(OUT_DIR / "matched_spikes.csv", index=False)
    summary_df.to_csv(OUT_DIR / "velocity_following_summary_all.csv", index=False)
    no_branch_summary.to_csv(OUT_DIR / "velocity_following_summary_no_branch.csv", index=False)
    no_branch_delta.to_csv(OUT_DIR / "velocity_following_delta_vs_no_ec_isolated_no_branch.csv", index=False)
    branch_vs_no_branch.to_csv(OUT_DIR / "velocity_following_one_branch_vs_no_branch.csv", index=False)
    print("[SAVED]", OUT_DIR / "matched_spikes.csv")
    print("[SAVED]", OUT_DIR / "velocity_following_summary_no_branch.csv")
    print("[SAVED]", OUT_DIR / "velocity_following_delta_vs_no_ec_isolated_no_branch.csv")
    print("[SAVED]", OUT_DIR / "velocity_following_one_branch_vs_no_branch.csv")
    return matched_df, no_branch_summary, no_branch_delta, branch_vs_no_branch


def build_plots(no_branch_summary: pd.DataFrame, no_branch_delta: pd.DataFrame, branch_vs_no_branch: pd.DataFrame) -> None:
    for diameter in DIAMETERS:
        for topology in TOPOLOGIES:
            plot_following_and_velocity(
                no_branch_summary,
                fiber_diameter_um=diameter,
                topology=topology,
                scenario=NO_BRANCH_SCENARIO,
                out_dir=PLOT_DIR / "no_branch_by_distance",
            )
            plot_metric_grid_2x3(
                no_branch_summary,
                fiber_diameter_um=diameter,
                topology=topology,
                scenario=NO_BRANCH_SCENARIO,
                metric_col="median_terminal_latency_ms",
                metric_label="Latency, ms",
                scale_y_by_axon=True,
                y_lower_bound=0.0,
                out_path=PLOT_DIR / "no_branch_by_distance" / f"fd{diameter}_{topology}_{NO_BRANCH_SCENARIO}_latency.png",
            )
            plot_velocity_by_mode_grid(
                no_branch_summary,
                fiber_diameter_um=diameter,
                topology=topology,
                scenario=NO_BRANCH_SCENARIO,
                include_no_ec=True,
                out_dir=PLOT_DIR / "no_branch_by_mode",
            )
            plot_following_by_mode_grid(
                no_branch_summary,
                fiber_diameter_um=diameter,
                topology=topology,
                scenario=NO_BRANCH_SCENARIO,
                include_no_ec=True,
                out_dir=PLOT_DIR / "no_branch_by_mode",
            )
            plot_no_ec_delta_by_mode_grid(
                no_branch_delta,
                fiber_diameter_um=diameter,
                topology=topology,
                scenario=NO_BRANCH_SCENARIO,
                delta_metric_col="delta_median_terminal_velocity_m_s_vs_no_ec_isolated",
                metric_label="Delta velocity, m/s",
                scale_y_by_axon=True,
                out_path=PLOT_DIR / "no_branch_delta_vs_no_ec_isolated" / f"fd{diameter}_{topology}_delta_velocity.png",
            )
            plot_no_ec_delta_by_mode_grid(
                no_branch_delta,
                fiber_diameter_um=diameter,
                topology=topology,
                scenario=NO_BRANCH_SCENARIO,
                delta_metric_col="delta_median_terminal_latency_ms_vs_no_ec_isolated",
                metric_label="Delta latency, ms",
                scale_y_by_axon=True,
                out_path=PLOT_DIR / "no_branch_delta_vs_no_ec_isolated" / f"fd{diameter}_{topology}_delta_latency.png",
            )
            plot_no_ec_delta_by_mode_grid(
                no_branch_delta,
                fiber_diameter_um=diameter,
                topology=topology,
                scenario=NO_BRANCH_SCENARIO,
                delta_metric_col="delta_following_fraction_terminal_vs_no_ec_isolated",
                metric_label="Delta following",
                y_lim=(-1.05, 1.05),
                out_path=PLOT_DIR / "no_branch_delta_vs_no_ec_isolated" / f"fd{diameter}_{topology}_delta_following.png",
            )

            if not branch_vs_no_branch.empty:
                plot_metric_by_mode_grid(
                    branch_vs_no_branch,
                    fiber_diameter_um=diameter,
                    topology=topology,
                    scenario=COMPARISON_SCENARIO,
                    metric_col="delta_median_terminal_velocity_m_s_one_branch_vs_no_branch",
                    metric_label="One-branch minus no-branch velocity, m/s",
                    scale_y_by_axon=True,
                    out_path=PLOT_DIR / "one_branch_vs_no_branch" / f"fd{diameter}_{topology}_delta_velocity.png",
                )
                plot_metric_by_mode_grid(
                    branch_vs_no_branch,
                    fiber_diameter_um=diameter,
                    topology=topology,
                    scenario=COMPARISON_SCENARIO,
                    metric_col="delta_median_terminal_latency_ms_one_branch_vs_no_branch",
                    metric_label="One-branch minus no-branch latency, ms",
                    scale_y_by_axon=True,
                    out_path=PLOT_DIR / "one_branch_vs_no_branch" / f"fd{diameter}_{topology}_delta_latency.png",
                )
                plot_metric_by_mode_grid(
                    branch_vs_no_branch,
                    fiber_diameter_um=diameter,
                    topology=topology,
                    scenario=COMPARISON_SCENARIO,
                    metric_col="delta_following_fraction_terminal_one_branch_vs_no_branch",
                    metric_label="One-branch minus no-branch following",
                    y_lim=(-1.05, 1.05),
                    out_path=PLOT_DIR / "one_branch_vs_no_branch" / f"fd{diameter}_{topology}_delta_following.png",
                )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze no_branch_reference velocity/following and compare one_branch vs no_branch.")
    parser.add_argument("--stim-protocol", default="sync", help="Protocol tag to analyze, e.g. sync or delay_0p5ms.")
    parser.add_argument("--skip-export", action="store_true", help="Reuse data/analysis_no_branch_reference/spikes.csv.")
    parser.add_argument("--no-plots", action="store_true", help="Only write CSV outputs.")
    parser.add_argument("--test-mode", choices=["exclude", "include", "only"], default="exclude", help="How to handle HDF5 files marked test_mode=1.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.skip_export:
        spikes_df = pd.read_csv(OUT_DIR / "spikes.csv")
        print("[LOAD]", OUT_DIR / "spikes.csv")
    else:
        spikes_df = export_spikes(args)

    if spikes_df.empty:
        raise SystemExit("No spikes found for selected filters.")

    _, no_branch_summary, no_branch_delta, branch_vs_no_branch = build_outputs(spikes_df)
    if not args.no_plots:
        build_plots(no_branch_summary, no_branch_delta, branch_vs_no_branch)
    print("[DONE] no-branch analysis complete")


if __name__ == "__main__":
    main()
