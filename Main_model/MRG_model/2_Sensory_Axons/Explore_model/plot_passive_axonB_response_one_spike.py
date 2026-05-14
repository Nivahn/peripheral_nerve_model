from __future__ import annotations

import csv
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import numpy as np


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from MRG_lib import TwoSensoryAxonsPrescott  # noqa: E402


OUT_DIR = Path(__file__).resolve().parent / "passive_axonB_response_one_spike"
OUT_PNG = OUT_DIR / "passive_axonB_response_one_spike.png"
OUT_CSV = OUT_DIR / "passive_axonB_response_one_spike_summary.csv"

T_STOP_MS = 20.0
DT_MS = 0.005
SPIKE_TIME_MS = 10.0

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
    "h_stop": T_STOP_MS,
    "celsius": 37.0,
    "boundary_full_cable": False,
}

STIM_A = {
    "mode": "spike_times",
    "biphasic": True,
    "spike_times_ms": [SPIKE_TIME_MS],
    "amp": 2.0,
    "t_end": T_STOP_MS,
    "phase_us": 40.0,
    "gap_us": 5.0,
}

STIM_B_ZERO = {
    "mode": "custom_waveform",
    "biphasic": False,
    "time_points_ms": [0.0, T_STOP_MS],
    "current_points_nA": [0.0, 0.0],
    "t_end": T_STOP_MS,
}

MODES = [
    ("aligned", {"aligned": True, "enable_ephaptic": True, "misalignment_fraction": None}),
    ("misaligned", {"aligned": False, "enable_ephaptic": True, "misalignment_fraction": None}),
    ("misaligned_0.25", {"aligned": False, "enable_ephaptic": True, "misalignment_fraction": 0.25}),
    ("no_ec", {"aligned": True, "enable_ephaptic": False, "misalignment_fraction": None}),
]

AXON_B_LABELS = ["before_branch", "branch_point", "after_branch_main"]


def label_index(labels: list[str], label: str) -> int:
    try:
        return labels.index(label)
    except ValueError as exc:
        raise RuntimeError(f"Missing recording label {label!r}; available labels: {labels}") from exc


def run_mode(mode_name: str, mode_kwargs: dict) -> dict:
    model = TwoSensoryAxonsPrescott(**MODEL_BASE, **mode_kwargs)
    model.set_stimulation_for_axons_independent(
        stim_A=True,
        stim_B=True,
        stim_target_mode_A="node_index",
        stim_node_index_A=0,
        stim_target_mode_B="node_index",
        stim_node_index_B=0,
        stim_kwargs_A=STIM_A,
        stim_kwargs_B=STIM_B_ZERO,
    )
    model.run_simulation_two_axons(
        h5_path=None,
        experiment_name=f"passive_{mode_name}_one_spike",
        record_kinetics=False,
        include_stimulation_point=True,
        record_axonA_before_like=True,
        record_axonA_main_like=True,
        record_terminal_nodes=True,
    )
    return {"mode": mode_name, "model": model}


def response_metrics(t_ms: np.ndarray, v_mV: np.ndarray) -> dict:
    baseline_mask = (t_ms >= 0.0) & (t_ms < SPIKE_TIME_MS - 2.0)
    response_mask = (t_ms >= SPIKE_TIME_MS - 1.0) & (t_ms <= SPIKE_TIME_MS + 20.0)
    baseline = float(np.nanmedian(v_mV[baseline_mask])) if np.any(baseline_mask) else float(v_mV[0])
    response = v_mV[response_mask] - baseline
    t_response = t_ms[response_mask]
    peak_i = int(np.nanargmax(response)) if response.size else 0
    trough_i = int(np.nanargmin(response)) if response.size else 0
    return {
        "baseline_mV": baseline,
        "peak_delta_mV": float(response[peak_i]) if response.size else np.nan,
        "peak_time_ms": float(t_response[peak_i]) if response.size else np.nan,
        "trough_delta_mV": float(response[trough_i]) if response.size else np.nan,
        "trough_time_ms": float(t_response[trough_i]) if response.size else np.nan,
    }


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    results = [run_mode(mode_name, mode_kwargs) for mode_name, mode_kwargs in MODES]
    no_ec = next(result for result in results if result["mode"] == "no_ec")
    no_ec_labels_b = list(getattr(no_ec["model"].axonB, "recording_labels", []) or [])
    no_ec_t = np.asarray(no_ec["model"].axonB.time_array, dtype=float)
    no_ec_traces = {
        label: np.asarray(no_ec["model"].axonB.voltage_matrix[label_index(no_ec_labels_b, label)], dtype=float)
        for label in AXON_B_LABELS
    }
    fig, axes = plt.subplots(len(MODES), len(AXON_B_LABELS) + 1, figsize=(15.5, 9.0), dpi=180, sharex=True)
    rows: list[dict] = []

    for row_i, result in enumerate(results):
        mode = result["mode"]
        model = result["model"]
        t_ms = np.asarray(model.axonB.time_array, dtype=float)
        labels_a = list(getattr(model.axonA, "recording_labels", []) or [])
        labels_b = list(getattr(model.axonB, "recording_labels", []) or [])

        ax = axes[row_i, 0]
        idx_a = label_index(labels_a, "stimulation_point")
        ax.plot(t_ms, model.axonA.voltage_matrix[idx_a], color="#111827", lw=1.0)
        ax.set_title("Axon A stimulated")
        ax.set_ylabel(f"{mode}\nVm (mV)")
        ax.grid(True, alpha=0.25)

        for col_i, label in enumerate(AXON_B_LABELS, start=1):
            ax = axes[row_i, col_i]
            idx_b = label_index(labels_b, label)
            raw_b = np.asarray(model.axonB.voltage_matrix[idx_b], dtype=float)
            if np.array_equal(t_ms, no_ec_t):
                v_b = raw_b - no_ec_traces[label]
            else:
                v_b = raw_b - np.interp(t_ms, no_ec_t, no_ec_traces[label])
            metrics = response_metrics(t_ms, v_b)
            ax.plot(t_ms, v_b - metrics["baseline_mV"], color="#2563eb", lw=1.0)
            ax.axhline(0.0, color="black", lw=0.6, alpha=0.5)
            ax.set_title(f"Axon B {label}\nmode - no_ec peak={metrics['peak_delta_mV']:.4g} mV")
            ax.grid(True, alpha=0.25)
            rows.append(
                {
                    "mode": mode,
                    "axon": "B",
                    "recording_label": label,
                    "offset_B_um": float(getattr(model, "_offsetB_um", np.nan)),
                    "n_axon_axon_pairs": 0 if model.spec_AB is None else int(len(model.spec_AB.sec_names_first)),
                    "trace_transform": "mode_minus_no_ec",
                    **metrics,
                }
            )

        for ax in axes[row_i, :]:
            ax.set_xlim(0.0, 20.0)
        axes[row_i, 1].set_ylabel("Delta Vm (mV)")

    for ax in axes[-1, :]:
        ax.set_xlabel("Time (ms)")
    fig.suptitle("Passive Axon B response to one Axon A spike | Axon B unstimulated | B traces are mode - no_ec", fontsize=13)
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
        print(
            f"{row['mode']} {row['recording_label']}: "
            f"peak_delta_mV={row['peak_delta_mV']:.6g} trough_delta_mV={row['trough_delta_mV']:.6g} "
            f"pairs={row['n_axon_axon_pairs']} offset_B_um={row['offset_B_um']:.6g}"
        )


if __name__ == "__main__":
    main()
