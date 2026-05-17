import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analyze_lib import (
    get_detected_spikes_for_trace,
    load_voltage_trace_first_node,
    normalize_mode_name,
    select_h5_file_for_condition,
)


MODE_ORDER = [
    "aligned",
    "misaligned_0.5",
    "misaligned_0.25",
    "no_EC",
    "no_EC_isolated",
]

MODE_LABELS = {
    "aligned": "aligned",
    "misaligned_0.5": "misaligned 0.5",
    "misaligned_0.25": "misaligned 0.25",
    "no_EC": "no EC",
    "no_EC_isolated": "isolated",
}

MODE_COLORS = {
    "aligned": "#2563eb",
    "misaligned_0.5": "#16a34a",
    "misaligned_0.25": "#f59e0b",
    "no_EC": "#111827",
    "no_EC_isolated": "#7c3aed",
}

EDGE_DIST_ORDER = [0.1, 0.5, 1.0]
AXON_ORDER = ["AxonA", "AxonB"]

DEFAULT_PLOT_TRACE_SPECS = [
    {"axon": "AxonA", "trace": "before_like", "title": "AxonA: до ветвления"},
    {"axon": "AxonA", "trace": "main_like", "title": "AxonA: после ветвления / main-like"},
    {"axon": "AxonB", "trace": "before_branch", "title": "AxonB: до ветвления"},
    {"axon": "AxonB", "trace": "after_branch_main", "title": "AxonB: после ветвления / main"},
]

DEFAULT_PLOT_SETTINGS = {
    "figsize": (16, 8),
    "dpi": 150,
    "trace_lw": 1.15,
    "spike_size": 28,
    "grid_alpha": 0.30,
    "legend_fontsize": 8,
    "title_fontsize": 10,
    "suptitle_fontsize": 13,
    "ylim": None,
    "sharex": True,
    "sharey": False,
}


def sanitize_filename(text: str) -> str:
    text = str(text)
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", text)
    return text.strip("_")


def plot_branch_spike_detection_2x2(
    spikes_df: pd.DataFrame,
    *,
    h5_file: str | Path,
    freq_hz: float,
    t_min_ms: float = 0.0,
    t_max_ms: float = 100.0,
    settings: dict | None = None,
    plot_trace_specs: list[dict] | None = None,
):
    """
    Проверочный график 2x2: Vm + detected spikes from CSV.
    """
    settings = dict(DEFAULT_PLOT_SETTINGS if settings is None else settings)
    plot_trace_specs = DEFAULT_PLOT_TRACE_SPECS if plot_trace_specs is None else plot_trace_specs
    h5_file = Path(h5_file)

    fig, axes = plt.subplots(
        2,
        2,
        figsize=settings["figsize"],
        dpi=settings["dpi"],
        sharex=settings["sharex"],
        sharey=settings["sharey"],
    )

    axes = axes.ravel()
    freq_group_name_used = None

    for ax, spec in zip(axes, plot_trace_specs):
        axon_name = spec["axon"]
        trace_name = spec["trace"]

        try:
            t_ms, v_mV, node, freq_group_name = load_voltage_trace_first_node(
                h5_file,
                freq_hz=freq_hz,
                axon=axon_name,
                trace=trace_name,
            )
            freq_group_name_used = freq_group_name
            mask = (t_ms >= float(t_min_ms)) & (t_ms <= float(t_max_ms))

            spikes_sub = get_detected_spikes_for_trace(
                spikes_df,
                h5_file=h5_file,
                freq_hz=freq_hz,
                axon=axon_name,
                trace=trace_name,
                node=node,
                t_min_ms=t_min_ms,
                t_max_ms=t_max_ms,
            )

            ax.plot(t_ms[mask], v_mV[mask], lw=settings["trace_lw"], label="Vm")

            if not spikes_sub.empty:
                ax.scatter(
                    spikes_sub["spike_time_ms"],
                    spikes_sub["spike_amplitude_mV"],
                    s=settings["spike_size"],
                    color="red",
                    zorder=5,
                    label=f"spikes: {len(spikes_sub)}",
                )
            else:
                ax.text(0.03, 0.92, "нет спайков в CSV", transform=ax.transAxes, va="top", fontsize=9)

            ax.set_title(f"{spec['title']}\n{trace_name} | {node}", fontsize=settings["title_fontsize"])
            ax.set_ylabel("Vm, mV")
            ax.grid(True, alpha=settings["grid_alpha"])
            ax.legend(loc="upper right", fontsize=settings["legend_fontsize"])
            if settings["ylim"] is not None:
                ax.set_ylim(*settings["ylim"])

        except Exception as exc:
            ax.set_title(f"{spec['title']}\nERROR")
            ax.text(0.05, 0.5, str(exc), transform=ax.transAxes, va="center", ha="left", fontsize=9, wrap=True)
            ax.grid(True, alpha=settings["grid_alpha"])

    for ax in axes[2:]:
        ax.set_xlabel("Time, ms")

    fig.suptitle(
        f"{h5_file.name}\n"
        f"{freq_group_name_used if freq_group_name_used else f'{freq_hz} Hz'} | "
        f"{t_min_ms:.1f}-{t_max_ms:.1f} ms",
        fontsize=settings["suptitle_fontsize"],
    )
    plt.tight_layout()
    plt.show()


def plot_condition_2x2(
    spikes_df: pd.DataFrame,
    *,
    fiber_diameter_um: float,
    edge_dist_um: float,
    mode: str,
    freq_hz: float,
    t_min_ms: float = 0.0,
    t_max_ms: float = 100.0,
    topology: str | None = None,
    scenario: str | None = None,
    h5_name_contains: str | None = None,
    settings: dict | None = None,
    plot_trace_specs: list[dict] | None = None,
):
    """
    Находит файл по условию и строит 2x2 detection-check plot.
    """
    h5_file = select_h5_file_for_condition(
        spikes_df,
        fiber_diameter_um=fiber_diameter_um,
        edge_dist_um=edge_dist_um,
        mode=mode,
        freq_hz=freq_hz,
        topology=topology,
        scenario=scenario,
        h5_name_contains=h5_name_contains,
    )

    print("[PLOT]", Path(h5_file).name)
    plot_branch_spike_detection_2x2(
        spikes_df,
        h5_file=h5_file,
        freq_hz=freq_hz,
        t_min_ms=t_min_ms,
        t_max_ms=t_max_ms,
        settings=settings,
        plot_trace_specs=plot_trace_specs,
    )


def prepare_summary_for_plotting(summary_df: pd.DataFrame) -> pd.DataFrame:
    """
    Приводит summary table к виду для plotting.
    """
    df = summary_df.copy()
    df["fiber_diameter_um"] = df["fiber_diameter_um"].astype(float)
    df["edge_dist_um"] = df["edge_dist_um"].astype(float)
    df["freq_hz"] = df["freq_hz"].astype(float)
    df["axon"] = df["axon"].astype(str)
    if "mode_norm" not in df.columns:
        df["mode_norm"] = df["mode"].astype(str).apply(normalize_mode_name)
    else:
        df["mode_norm"] = df["mode_norm"].astype(str).apply(normalize_mode_name)
    if "topology" not in df.columns:
        df["topology"] = "unknown"
    if "scenario" not in df.columns:
        df["scenario"] = "unknown"
    return df


def select_plot_data(
    summary_df: pd.DataFrame,
    *,
    fiber_diameter_um: float,
    topology: str | None = None,
    scenario: str | None = None,
) -> pd.DataFrame:
    df = prepare_summary_for_plotting(summary_df)
    df = df[np.isclose(df["fiber_diameter_um"], float(fiber_diameter_um))].copy()
    if topology is not None:
        df = df[df["topology"].astype(str) == str(topology)].copy()
    if scenario is not None:
        df = df[df["scenario"].astype(str) == str(scenario)].copy()
    return df


def compute_metric_y_lims_by_axon(
    df: pd.DataFrame,
    metric_col: str,
    *,
    lower_bound: float | None = None,
    pad_fraction: float = 0.08,
    min_pad: float = 0.5,
) -> dict[str, tuple[float, float]]:
    """
    One fixed y scale per axon row, so AxonA and AxonB do not compress each other.
    """
    y_lims: dict[str, tuple[float, float]] = {}
    for axon in AXON_ORDER:
        values = pd.to_numeric(df.loc[df["axon"] == axon, metric_col], errors="coerce")
        values = values[np.isfinite(values)]
        if values.empty:
            continue
        y_min = float(values.min())
        y_max = float(values.max())
        pad = max((y_max - y_min) * pad_fraction, min_pad)
        if np.isclose(y_min, y_max):
            pad = max(abs(y_max) * pad_fraction, min_pad)
        lo = y_min - pad
        hi = y_max + pad
        if lower_bound is not None:
            lo = max(float(lower_bound), lo)
        if np.isclose(lo, hi):
            hi = lo + min_pad
        y_lims[axon] = (lo, hi)
    return y_lims


def plot_metric_grid_2x3(
    summary_df: pd.DataFrame,
    *,
    fiber_diameter_um: float,
    metric_col: str,
    metric_label: str,
    topology: str | None = None,
    scenario: str | None = None,
    y_lim: tuple[float, float] | None = None,
    y_lims_by_axon: dict[str, tuple[float, float]] | None = None,
    scale_y_by_axon: bool = False,
    y_lower_bound: float | None = None,
    out_path: str | Path | None = None,
):
    """
    2x3 plot: rows AxonA/AxonB, columns edge distances.
    """
    df = select_plot_data(summary_df, fiber_diameter_um=fiber_diameter_um, topology=topology, scenario=scenario)
    if df.empty:
        print("[EMPTY]", f"fiber_diameter_um={fiber_diameter_um}", f"topology={topology}", f"scenario={scenario}")
        return
    if metric_col not in df.columns:
        raise KeyError(f"В summary_df нет колонки {metric_col!r}")
    if scale_y_by_axon and y_lims_by_axon is None:
        y_lims_by_axon = compute_metric_y_lims_by_axon(df, metric_col, lower_bound=y_lower_bound)

    sharey = "row" if y_lims_by_axon is not None else True
    fig, axes = plt.subplots(2, 3, figsize=(18, 8), dpi=150, sharex=True, sharey=sharey)

    for row_i, axon in enumerate(AXON_ORDER):
        for col_i, edge_dist_um in enumerate(EDGE_DIST_ORDER):
            ax = axes[row_i, col_i]
            panel_df = df[(df["axon"] == axon) & np.isclose(df["edge_dist_um"], float(edge_dist_um))].copy()

            for mode in MODE_ORDER:
                mode_df = panel_df[panel_df["mode_norm"] == mode].copy()
                if mode_df.empty:
                    continue
                plot_df = mode_df.groupby("freq_hz", as_index=False)[metric_col].mean().sort_values("freq_hz")
                ax.plot(
                    plot_df["freq_hz"],
                    plot_df[metric_col],
                    marker="o",
                    markersize=4.5,
                    linewidth=2.2,
                    color=MODE_COLORS.get(mode, None),
                    label=MODE_LABELS.get(mode, mode),
                )

            ax.set_title(f"{axon}, distance {edge_dist_um}", fontsize=12, fontweight="bold")
            ax.set_xlabel("Частота стимуляции, Гц", fontsize=11)
            ax.set_ylabel(metric_label, fontsize=11)
            ax.grid(True, alpha=0.30)
            for x in np.arange(50, 1000 + 1, 50):
                ax.axvline(x, color="0.85", linewidth=0.5, zorder=0)
            ax.set_xticks(np.arange(50, 1000 + 1, 100))
            ax.set_xlim(50, 1000)
            ax.tick_params(axis="x", labelbottom=True)
            ax.tick_params(axis="both", labelsize=10)
            if y_lims_by_axon is not None and axon in y_lims_by_axon:
                ax.set_ylim(*y_lims_by_axon[axon])
            elif y_lim is not None:
                ax.set_ylim(*y_lim)
            if row_i == 0 and col_i == 2:
                ax.legend(loc="upper left", bbox_to_anchor=(1.03, 1.0), fontsize=10, frameon=True)

    title_parts = [f"Диаметр {fiber_diameter_um} мкм"]
    if topology is not None:
        title_parts.append(f"тип ветвления: {topology}")
    if scenario is not None:
        title_parts.append(f"сценарий: {scenario}")
    fig.suptitle(metric_label + "\n" + " | ".join(title_parts), fontsize=15, fontweight="bold")
    plt.tight_layout()

    if out_path is not None:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=300, bbox_inches="tight")
        print("[SAVED]", out_path)
        plt.close(fig)
    else:
        plt.show()


def plot_following_fraction_grid(
    summary_df: pd.DataFrame,
    *,
    fiber_diameter_um: float,
    topology: str | None = None,
    scenario: str | None = None,
    out_dir: str | Path | None = None,
):
    out_path = None
    if out_dir is not None:
        out_dir = Path(out_dir)
        tag = sanitize_filename(f"fd{fiber_diameter_um}_{topology or 'all_topology'}_{scenario or 'all_scenario'}")
        out_path = out_dir / f"{tag}_following_fraction_terminal.png"
    plot_metric_grid_2x3(
        summary_df,
        fiber_diameter_um=fiber_diameter_um,
        topology=topology,
        scenario=scenario,
        metric_col="following_fraction_terminal",
        metric_label="Доля следования стимуляции на terminal_main",
        y_lim=(-0.05, 1.05),
        out_path=out_path,
    )


def plot_velocity_grid(
    summary_df: pd.DataFrame,
    *,
    fiber_diameter_um: float,
    topology: str | None = None,
    scenario: str | None = None,
    out_dir: str | Path | None = None,
):
    out_path = None
    if out_dir is not None:
        out_dir = Path(out_dir)
        tag = sanitize_filename(f"fd{fiber_diameter_um}_{topology or 'all_topology'}_{scenario or 'all_scenario'}")
        out_path = out_dir / f"{tag}_median_velocity.png"
    plot_metric_grid_2x3(
        summary_df,
        fiber_diameter_um=fiber_diameter_um,
        topology=topology,
        scenario=scenario,
        metric_col="median_terminal_velocity_m_s",
        metric_label="Скорость проведения до terminal_main, м/с",
        scale_y_by_axon=True,
        y_lower_bound=0.0,
        out_path=out_path,
    )


def plot_following_and_velocity(
    summary_df: pd.DataFrame,
    *,
    fiber_diameter_um: float,
    topology: str | None = None,
    scenario: str | None = None,
    out_dir: str | Path | None = None,
):
    plot_following_fraction_grid(
        summary_df,
        fiber_diameter_um=fiber_diameter_um,
        topology=topology,
        scenario=scenario,
        out_dir=out_dir,
    )
    plot_velocity_grid(
        summary_df,
        fiber_diameter_um=fiber_diameter_um,
        topology=topology,
        scenario=scenario,
        out_dir=out_dir,
    )


def plot_metric_by_mode_grid(
    summary_df: pd.DataFrame,
    *,
    fiber_diameter_um: float,
    metric_col: str,
    metric_label: str,
    topology: str | None = None,
    scenario: str | None = None,
    modes: list[str] | None = None,
    y_lim: tuple[float, float] | None = None,
    y_lims_by_axon: dict[str, tuple[float, float]] | None = None,
    scale_y_by_axon: bool = False,
    y_lower_bound: float | None = None,
    out_path: str | Path | None = None,
):
    """
    Grid: rows AxonA/AxonB, columns modes. Lines = edge distances.
    """
    df = select_plot_data(summary_df, fiber_diameter_um=fiber_diameter_um, topology=topology, scenario=scenario)
    if metric_col not in df.columns:
        raise KeyError(f"В summary_df нет колонки {metric_col!r}")
    modes = modes or MODE_ORDER
    modes = [normalize_mode_name(mode) for mode in modes]
    df = df[df["mode_norm"].isin(modes)].copy()
    if scale_y_by_axon and y_lims_by_axon is None:
        y_lims_by_axon = compute_metric_y_lims_by_axon(df, metric_col, lower_bound=y_lower_bound)

    sharey = "row" if y_lims_by_axon is not None else True
    fig, axes = plt.subplots(2, len(modes), figsize=(4.6 * len(modes), 8), dpi=150, sharex=True, sharey=sharey)
    if len(modes) == 1:
        axes = np.asarray(axes).reshape(2, 1)

    for row_i, axon in enumerate(AXON_ORDER):
        for col_i, mode in enumerate(modes):
            ax = axes[row_i, col_i]
            panel_df = df[(df["axon"] == axon) & (df["mode_norm"] == mode)].copy()

            for edge_dist_um in EDGE_DIST_ORDER:
                line_df = panel_df[np.isclose(panel_df["edge_dist_um"], float(edge_dist_um))].copy()
                if line_df.empty:
                    continue
                plot_df = line_df.groupby("freq_hz", as_index=False)[metric_col].mean().sort_values("freq_hz")
                ax.plot(
                    plot_df["freq_hz"],
                    plot_df[metric_col],
                    marker="o",
                    markersize=4.0,
                    linewidth=2.0,
                    label=f"edge {edge_dist_um:g} um",
                )

            ax.set_title(f"{MODE_LABELS.get(mode, mode)} | {axon}", fontsize=12, fontweight="bold")
            ax.set_xlabel("Частота, Гц")
            ax.set_ylabel(metric_label)
            ax.grid(True, alpha=0.30)
            ax.set_xticks(np.arange(50, 1000 + 1, 100))
            ax.set_xlim(50, 1000)
            if y_lims_by_axon is not None and axon in y_lims_by_axon:
                ax.set_ylim(*y_lims_by_axon[axon])
            elif y_lim is not None:
                ax.set_ylim(*y_lim)
            if row_i == 0 and col_i == len(modes) - 1:
                ax.legend(loc="upper left", bbox_to_anchor=(1.03, 1.0), fontsize=9, frameon=True)

    title_parts = [f"Диаметр {fiber_diameter_um} мкм"]
    if topology is not None:
        title_parts.append(str(topology))
    if scenario is not None:
        title_parts.append(str(scenario))
    fig.suptitle(metric_label + "\n" + " | ".join(title_parts), fontsize=15, fontweight="bold")
    plt.tight_layout()

    if out_path is not None:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=300, bbox_inches="tight")
        print("[SAVED]", out_path)
        plt.close(fig)
    else:
        plt.show()


def plot_following_by_mode_grid(
    summary_df: pd.DataFrame,
    *,
    fiber_diameter_um: float,
    topology: str | None = None,
    scenario: str | None = None,
    include_no_ec: bool = False,
    out_dir: str | Path | None = None,
):
    modes = MODE_ORDER if include_no_ec else [mode for mode in MODE_ORDER if not mode.startswith("no_EC")]
    out_path = None
    if out_dir is not None:
        tag = sanitize_filename(f"fd{fiber_diameter_um}_{topology or 'all_topology'}_{scenario or 'all_scenario'}_{'with_no_ec' if include_no_ec else 'no_no_ec'}")
        out_path = Path(out_dir) / f"{tag}_following_by_mode.png"
    plot_metric_by_mode_grid(
        summary_df,
        fiber_diameter_um=fiber_diameter_um,
        topology=topology,
        scenario=scenario,
        modes=modes,
        metric_col="following_fraction_terminal",
        metric_label="Доля следования стимуляции на terminal_main",
        y_lim=(-0.05, 1.05),
        out_path=out_path,
    )


def plot_velocity_by_mode_grid(
    summary_df: pd.DataFrame,
    *,
    fiber_diameter_um: float,
    topology: str | None = None,
    scenario: str | None = None,
    include_no_ec: bool = False,
    out_dir: str | Path | None = None,
):
    modes = MODE_ORDER if include_no_ec else [mode for mode in MODE_ORDER if not mode.startswith("no_EC")]
    out_path = None
    if out_dir is not None:
        tag = sanitize_filename(f"fd{fiber_diameter_um}_{topology or 'all_topology'}_{scenario or 'all_scenario'}_{'with_no_ec' if include_no_ec else 'no_no_ec'}")
        out_path = Path(out_dir) / f"{tag}_velocity_by_mode.png"
    plot_metric_by_mode_grid(
        summary_df,
        fiber_diameter_um=fiber_diameter_um,
        topology=topology,
        scenario=scenario,
        modes=modes,
        metric_col="median_terminal_velocity_m_s",
        metric_label="Скорость проведения до terminal_main, м/с",
        scale_y_by_axon=True,
        y_lower_bound=0.0,
        out_path=out_path,
    )


def plot_no_ec_delta_by_mode_grid(
    delta_df: pd.DataFrame,
    *,
    fiber_diameter_um: float,
    delta_metric_col: str,
    metric_label: str,
    topology: str | None = None,
    scenario: str | None = None,
    y_lim: tuple[float, float] | None = None,
    scale_y_by_axon: bool = False,
    baseline_mode: str = "no_EC_isolated",
    out_path: str | Path | None = None,
):
    baseline_mode = normalize_mode_name(baseline_mode)
    modes = [mode for mode in MODE_ORDER if mode != baseline_mode]
    plot_metric_by_mode_grid(
        delta_df,
        fiber_diameter_um=fiber_diameter_um,
        topology=topology,
        scenario=scenario,
        modes=modes,
        metric_col=delta_metric_col,
        metric_label=metric_label,
        y_lim=y_lim,
        scale_y_by_axon=scale_y_by_axon,
        out_path=out_path,
    )
