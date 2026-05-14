from __future__ import annotations

import csv
import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import find_peaks


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from MRG_lib import TwoSensoryAxonsPrescott  # noqa: E402


FREQ_HZ = float(os.environ.get("FREQ_HZ", "450"))
T_START_MS = 10.0
T_END_MS = 110.0
H_STOP_MS = 115.0
DT_MS = 0.005

OUT_DIR = Path(__file__).resolve().parent / f"axonB_branching_{int(FREQ_HZ)}hz_100ms"
OUT_PNG = OUT_DIR / f"axonB_before_after_branch_{int(FREQ_HZ)}Hz_100ms.png"
OUT_CSV = OUT_DIR / f"axonB_before_after_branch_{int(FREQ_HZ)}Hz_100ms_summary.csv"

MODEL_BASE = {
    "fiber_diameter_um": 5.7,
    "edge_dist_um": 0.1,
    "parent_axon_nodes_A": 17,
    "branch_nodes_A": 8,
    "branches_num_A": 0,
    "branch_sequence_nodes_A": None,
    "parent_axon_nodes_B": 17,
    "branch_nodes_B": 8,
    "branches_num_B": 1,
    "branch_sequence_nodes_B": None,
    "branch_topology_mode_A": "node",
    "branch_topology_mode_B": "node",
    "diam_scale": 0.6,
    "dt_ms": DT_MS,
    "v_init": -80.0,
    "h_stop": H_STOP_MS,
    "celsius": 37.0,
    "boundary_full_cable": False,
}

STIM = {
    "stim_A": True,
    "stim_B": True,
    "stim_target_mode": "node_index",
    "stim_node_index": 0,
    "stim_x_um": None,
    "mode": "create",
    "biphasic": True,
    "freq_hz": FREQ_HZ,
    "amp": 5.0,
    "t_start": T_START_MS,
    "t_end": T_END_MS,
    "phase_us": 40.0,
    "gap_us": 5.0,
}

MODES = [
    ("aligned", {"aligned": True, "enable_ephaptic": True, "misalignment_fraction": None}),
    ("misaligned", {"aligned": False, "enable_ephaptic": True, "misalignment_fraction": None}),
    ("misaligned_0.25", {"aligned": False, "enable_ephaptic": True, "misalignment_fraction": 0.25}),
    ("no_ec", {"aligned": True, "enable_ephaptic": False, "misalignment_fraction": None}),
]

COLUMNS = [
    ("before_branch", "Before branching"),
    ("after_branch_main", "After branching, main axon"),
]


def label_index(labels: list[str], label: str) -> int:
    try:
        return labels.index(label)
    except ValueError as exc:
        raise RuntimeError(f"Missing recording label {label!r}; available labels: {labels}") from exc


def spike_times_ms(t_ms: np.ndarray, v_mV: np.ndarray) -> np.ndarray:
    mask0 = int(np.searchsorted(t_ms, T_START_MS, side="left"))
    mask1 = int(np.searchsorted(t_ms, T_END_MS, side="right"))
    dt = float(t_ms[1] - t_ms[0])
    min_dist_pts = max(1, int(round(0.6 / max(dt, 1e-9))))
    peaks, _ = find_peaks(v_mV[mask0:mask1], prominence=5.0, height=-20.0, distance=min_dist_pts)
    return t_ms[mask0 + peaks]


def run_mode(mode_name: str, mode_kwargs: dict) -> dict:
    model = TwoSensoryAxonsPrescott(**MODEL_BASE, **mode_kwargs)
    model.set_stimulation_for_axons(**STIM)
    model.run_simulation_two_axons(
        h5_path=None,
        experiment_name=f"{mode_name}_{int(FREQ_HZ)}Hz_100ms",
        record_kinetics=False,
        include_stimulation_point=True,
        record_axonA_before_like=True,
        record_axonA_main_like=True,
        record_terminal_nodes=True,
    )

    t_ms = np.asarray(model.axonB.time_array, dtype=float)
    labels = list(getattr(model.axonB, "recording_labels", []) or [])
    traces = {}
    spikes = {}
    for label, _title in COLUMNS:
        idx = label_index(labels, label)
        v = np.asarray(model.axonB.voltage_matrix[idx], dtype=float)
        traces[label] = v
        spikes[label] = spike_times_ms(t_ms, v)
    return {"mode": mode_name, "model": model, "t_ms": t_ms, "traces": traces, "spikes": spikes}


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    results = [run_mode(mode_name, mode_kwargs) for mode_name, mode_kwargs in MODES]

    fig, axes = plt.subplots(len(MODES), len(COLUMNS), figsize=(13.0, 8.5), dpi=180, sharex=True, sharey=True)
    colors = {"before_branch": "#1d4ed8", "after_branch_main": "#dc2626"}
    plot_mask = None

    rows: list[dict] = []
    for row_i, result in enumerate(results):
        t_ms = result["t_ms"]
        if plot_mask is None:
            plot_mask = (t_ms >= T_START_MS) & (t_ms <= T_END_MS)

        for col_i, (label, title) in enumerate(COLUMNS):
            ax = axes[row_i, col_i]
            v = result["traces"][label]
            spike_ts = result["spikes"][label]
            ax.plot(t_ms[plot_mask], v[plot_mask], color=colors[label], lw=1.0)
            ax.scatter(spike_ts, np.full_like(spike_ts, 35.0), color="black", s=8, zorder=3, label="detected spike")
            ax.set_title(f"{title}\nspikes={len(spike_ts)}", fontsize=10)
            ax.grid(True, alpha=0.25)
            if col_i == 0:
                ax.set_ylabel(f"{result['mode']}\nVm (mV)")
            if row_i == len(MODES) - 1:
                ax.set_xlabel("Time (ms)")

            rows.append(
                {
                    "mode": result["mode"],
                    "axon": "B",
                    "recording_label": label,
                    "freq_hz": FREQ_HZ,
                    "t_start_ms": T_START_MS,
                    "t_end_ms": T_END_MS,
                    "spike_count": int(len(spike_ts)),
                    "spike_times_ms": ";".join(f"{x:.6g}" for x in spike_ts),
                    "offset_B_um": float(getattr(result["model"], "_offsetB_um", np.nan)),
                }
            )

    fig.suptitle(
        f"Axon B before/after branching | {int(FREQ_HZ)} Hz | 100 ms stimulation | edge 0.1 um | fiber 5.7 um",
        fontsize=13,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(OUT_PNG, bbox_inches="tight")
    plt.close(fig)

    with OUT_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"saved_png={OUT_PNG}")
    print(f"saved_csv={OUT_CSV}")
    for row in rows:
        print(f"{row['mode']} {row['recording_label']}: spikes={row['spike_count']} offset_B_um={row['offset_B_um']:.6g}")


if __name__ == "__main__":
    main()
