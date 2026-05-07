from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _mark_branch_zone(x_values: pd.Series, branch_points_um: list[float], branch_window_um: float) -> pd.Series:
    if not branch_points_um:
        return pd.Series(False, index=x_values.index)
    flags = pd.Series(False, index=x_values.index)
    for bp in branch_points_um:
        flags = flags | ((x_values - float(bp)).abs() <= float(branch_window_um))
    return flags


def compute_pairing_metrics(
    rows: list[dict],
    *,
    target_dx_um: float,
    eligible_A: int,
    eligible_B: int,
    branch_points_um: list[float] | None = None,
    branch_window_um: float | None = None,
):
    df = pd.DataFrame(rows).copy()
    if df.empty:
        raise ValueError("No pairing rows provided")

    df = df.sort_values(["x_A_um", "pair_index"]).reset_index(drop=True)
    df["same_type"] = df["kind_A"] == df["kind_B"]
    df["same_phase"] = df["pair_key_A"] == df["pair_key_B"]
    df["main_main"] = (df["path_type_A"] == "main") & (df["path_type_B"] == "main")
    df["dx_error_um"] = df["dx_um"] - float(target_dx_um)
    df["dx_abs_error_um"] = df["dx_error_um"].abs()
    df["dx_jump_um"] = df["dx_um"].diff().abs()

    if branch_window_um is None:
        branch_window_um = 0.0
    branch_points_um = [] if branch_points_um is None else [float(x) for x in branch_points_um]
    df["in_branch_zone"] = _mark_branch_zone(df["x_A_um"], branch_points_um, float(branch_window_um))

    summary = {
        "pair_count": int(len(df)),
        "eligible_A": int(eligible_A),
        "eligible_B": int(eligible_B),
        "coverage_A": float(len(df)) / float(max(int(eligible_A), 1)),
        "coverage_B": float(len(df)) / float(max(int(eligible_B), 1)),
        "type_match_rate": float(df["same_type"].mean()),
        "same_phase_rate": float(df["same_phase"].mean()),
        "main_main_fraction": float(df["main_main"].mean()),
        "mean_dx_error_um": float(df["dx_error_um"].mean()),
        "mae_dx_um": float(df["dx_abs_error_um"].mean()),
        "max_abs_dx_error_um": float(df["dx_abs_error_um"].max()),
        "max_dx_jump_um": float(df["dx_jump_um"].dropna().max()) if len(df) > 1 else 0.0,
    }

    in_branch = df[df["in_branch_zone"]]
    out_branch = df[~df["in_branch_zone"]]
    summary["branch_zone_pair_count"] = int(len(in_branch))
    summary["outside_branch_pair_count"] = int(len(out_branch))
    summary["branch_zone_mae_dx_um"] = float(in_branch["dx_abs_error_um"].mean()) if not in_branch.empty else float("nan")
    summary["outside_branch_mae_dx_um"] = float(out_branch["dx_abs_error_um"].mean()) if not out_branch.empty else float("nan")
    summary["branch_zone_max_jump_um"] = float(in_branch["dx_jump_um"].dropna().max()) if len(in_branch) > 1 else float("nan")
    summary["outside_branch_max_jump_um"] = float(out_branch["dx_jump_um"].dropna().max()) if len(out_branch) > 1 else float("nan")

    per_class_rows = []
    for kind, sub in df.groupby("kind_A", sort=True):
        per_class_rows.append({
            "kind": str(kind),
            "pair_count": int(len(sub)),
            "type_match_rate": float(sub["same_type"].mean()),
            "same_phase_rate": float(sub["same_phase"].mean()),
            "mean_dx_error_um": float(sub["dx_error_um"].mean()),
            "mae_dx_um": float(sub["dx_abs_error_um"].mean()),
            "max_abs_dx_error_um": float(sub["dx_abs_error_um"].max()),
            "max_dx_jump_um": float(sub["dx_jump_um"].dropna().max()) if len(sub) > 1 else float("nan"),
        })

    per_class_df = pd.DataFrame(per_class_rows)
    return summary, per_class_df, df


def plot_dx_profile(
    pair_df: pd.DataFrame,
    *,
    target_dx_um: float,
    branch_points_um: list[float] | None,
    branch_window_um: float,
    save_path: str | Path,
    title: str,
):
    df = pair_df.sort_values(["x_A_um", "pair_index"]).reset_index(drop=True)
    fig, axes = plt.subplots(2, 1, figsize=(13.0, 7.0), dpi=180, sharex=True)

    axes[0].plot(df["x_A_um"], df["dx_um"], color="#2563eb", lw=2.0, label="dx")
    axes[0].axhline(float(target_dx_um), color="#111827", ls="--", lw=1.3, label="target dx")
    axes[0].set_ylabel("dx (um)")
    axes[0].grid(True, alpha=0.25)
    axes[0].legend(loc="best")

    axes[1].plot(df["x_A_um"], df["dx_abs_error_um"], color="#dc2626", lw=2.0, label="|dx-target|")
    axes[1].set_xlabel("AxonA main-path x (um)")
    axes[1].set_ylabel("abs error (um)")
    axes[1].grid(True, alpha=0.25)
    axes[1].legend(loc="best")

    for bp in ([] if branch_points_um is None else branch_points_um):
        for ax in axes:
            ax.axvspan(float(bp) - float(branch_window_um), float(bp) + float(branch_window_um), color="#f59e0b", alpha=0.12)
            ax.axvline(float(bp), color="#f59e0b", lw=1.0, alpha=0.65)

    fig.suptitle(title)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    plt.savefig(str(save_path), dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_metrics_summary(summary_df: pd.DataFrame, save_path: str | Path, title: str = "Pairing quality summary"):
    fig, axes = plt.subplots(2, 1, figsize=(11.0, 8.0), dpi=180, sharex=True)

    labels = summary_df["case"].astype(str).tolist()
    x = np.arange(len(labels), dtype=float)
    width = 0.22

    axes[0].bar(x - width, summary_df["type_match_rate"], width=width, label="type match")
    axes[0].bar(x, summary_df["same_phase_rate"], width=width, label="same phase")
    axes[0].bar(x + width, summary_df["main_main_fraction"], width=width, label="main-main")
    axes[0].set_ylim(0.0, 1.05)
    axes[0].set_ylabel("fraction")
    axes[0].grid(True, axis="y", alpha=0.25)
    axes[0].legend(loc="best")

    axes[1].plot(x, summary_df["mae_dx_um"], marker="o", lw=2.0, label="MAE dx")
    axes[1].plot(x, summary_df["max_abs_dx_error_um"], marker="s", lw=2.0, label="max abs dx error")
    axes[1].plot(x, summary_df["max_dx_jump_um"], marker="^", lw=2.0, label="max dx jump")
    axes[1].set_ylabel("um")
    axes[1].set_xlabel("case")
    axes[1].grid(True, axis="y", alpha=0.25)
    axes[1].legend(loc="best")

    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels)

    fig.suptitle(title)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    plt.savefig(str(save_path), dpi=300, bbox_inches="tight")
    plt.close(fig)
