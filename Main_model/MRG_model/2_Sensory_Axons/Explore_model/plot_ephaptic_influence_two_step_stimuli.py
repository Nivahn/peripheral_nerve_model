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


OUT_DIR = Path(__file__).resolve().parent / "ephaptic_influence_two_step_stimuli"
OUT_PNG = OUT_DIR / "ephaptic_influence_two_step_stimuli.png"
OUT_CSV = OUT_DIR / "ephaptic_influence_two_step_stimuli_summary.csv"

DT_MS = 0.005
T_STOP_MS = 60.0
STEP_AMP_NA = 1.0
STEP_WINDOWS_MS = [(20.0, 20.5), (35.0, 35.5)]

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

MODES = [
    "aligned",
    "misaligned_0.5",
    "misaligned_0.25",
    "no_EC",
]

AXON_B_LABELS = ["before_branch", "branch_point", "after_branch_main"]


def step_waveform() -> tuple[list[float], list[float]]:
    t = [0.0]
    i = [0.0]
    for start, stop in STEP_WINDOWS_MS:
        t.extend([start, start, stop, stop])
        i.extend([0.0, STEP_AMP_NA, STEP_AMP_NA, 0.0])
    t.append(T_STOP_MS)
    i.append(0.0)
    return t, i


def label_index(labels: list[str], label: str) -> int:
    try:
        return labels.index(label)
    except ValueError as exc:
        raise RuntimeError(f"Missing recording label {label!r}; available labels: {labels}") from exc


def run_mode(mode_name: str) -> dict:
    t_step, i_step = step_waveform()
    model = TwoSensoryAxonsPrescott(**MODEL_BASE, mode_descriptor=mode_name)
    model.set_stimulation_for_axons_independent(
        stim_A=True,
        stim_B=True,
        stim_target_mode_A="node_index",
        stim_node_index_A=0,
        stim_target_mode_B="node_index",
        stim_node_index_B=0,
        stim_kwargs_A={
            "mode": "custom_waveform",
            "biphasic": False,
            "time_points_ms": t_step,
            "current_points_nA": i_step,
            "t_end": T_STOP_MS,
        },
        stim_kwargs_B={
            "mode": "custom_waveform",
            "biphasic": False,
            "time_points_ms": [0.0, T_STOP_MS],
            "current_points_nA": [0.0, 0.0],
            "t_end": T_STOP_MS,
        },
    )
    model.run_simulation_two_axons(
        h5_path=None,
        experiment_name=f"two_step_{mode_name}",
        record_kinetics=False,
        include_stimulation_point=True,
        record_axonA_before_like=True,
        record_axonA_main_like=True,
        record_terminal_nodes=True,
    )
    return {"mode": mode_name, "model": model}


def response_metrics(t_ms: np.ndarray, y: np.ndarray) -> dict:
    baseline_mask = (t_ms >= 0.0) & (t_ms < STEP_WINDOWS_MS[0][0] - 2.0)
    response_mask = (t_ms >= STEP_WINDOWS_MS[0][0] - 2.0) & (t_ms <= STEP_WINDOWS_MS[-1][1] + 15.0)
    baseline = float(np.nanmedian(y[baseline_mask])) if np.any(baseline_mask) else float(y[0])
    dy = y[response_mask] - baseline
    tt = t_ms[response_mask]
    peak_i = int(np.nanargmax(dy)) if dy.size else 0
    trough_i = int(np.nanargmin(dy)) if dy.size else 0
    return {
        "baseline_mV": baseline,
        "peak_delta_mV": float(dy[peak_i]) if dy.size else np.nan,
        "peak_time_ms": float(tt[peak_i]) if dy.size else np.nan,
        "trough_delta_mV": float(dy[trough_i]) if dy.size else np.nan,
        "trough_time_ms": float(tt[trough_i]) if dy.size else np.nan,
    }


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    results = [run_mode(mode_name) for mode_name in MODES]
    no_ec = next(result for result in results if result["mode"] == "no_EC")
    no_ec_t = np.asarray(no_ec["model"].axonB.time_array, dtype=float)
    no_ec_labels = list(getattr(no_ec["model"].axonB, "recording_labels", []) or [])
    no_ec_b = {
        label: np.asarray(no_ec["model"].axonB.voltage_matrix[label_index(no_ec_labels, label)], dtype=float)
        for label in AXON_B_LABELS
    }

    fig, axes = plt.subplots(len(MODES), len(AXON_B_LABELS) + 2, figsize=(18.0, 9.0), dpi=180, sharex=True)
    rows: list[dict] = []
    t_step, i_step = step_waveform()

    for row_i, result in enumerate(results):
        mode = result["mode"]
        model = result["model"]
        t_ms = np.asarray(model.axonA.time_array, dtype=float)
        labels_a = list(getattr(model.axonA, "recording_labels", []) or [])
        labels_b = list(getattr(model.axonB, "recording_labels", []) or [])

        ax = axes[row_i, 0]
        ax.step(t_step, i_step, where="post", color="#059669", lw=1.2)
        ax.set_title("Axon A IClamp")
        ax.set_ylabel(f"{mode}\nI (nA)")
        ax.grid(True, alpha=0.25)

        ax = axes[row_i, 1]
        idx_a = label_index(labels_a, "stimulation_point")
        ax.plot(t_ms, model.axonA.voltage_matrix[idx_a], color="#111827", lw=1.0)
        ax.set_title("Axon A Vm")
        ax.grid(True, alpha=0.25)

        for col_i, label in enumerate(AXON_B_LABELS, start=2):
            ax = axes[row_i, col_i]
            raw = np.asarray(model.axonB.voltage_matrix[label_index(labels_b, label)], dtype=float)
            if np.array_equal(t_ms, no_ec_t):
                y = raw - no_ec_b[label]
            else:
                y = raw - np.interp(t_ms, no_ec_t, no_ec_b[label])
            metrics = response_metrics(t_ms, y)
            ax.plot(t_ms, y - metrics["baseline_mV"], color="#2563eb", lw=1.0)
            ax.axhline(0.0, color="black", lw=0.6, alpha=0.5)
            ax.set_title(f"B {label}\nmode-no_EC peak={metrics['peak_delta_mV']:.4g} mV")
            ax.grid(True, alpha=0.25)
            rows.append(
                {
                    "mode": mode,
                    "axon": "B",
                    "recording_label": label,
                    "step_amp_nA": STEP_AMP_NA,
                    "step_windows_ms": ";".join(f"{a}-{b}" for a, b in STEP_WINDOWS_MS),
                    "offset_B_um": float(getattr(model, "_offsetB_um", np.nan)),
                    "n_axon_axon_pairs": 0 if model.spec_AB is None else int(len(model.spec_AB.sec_names_first)),
                    "trace_transform": "mode_minus_no_EC",
                    **metrics,
                }
            )

        for ax in axes[row_i, :]:
            ax.set_xlim(15.0, 55.0)
        axes[row_i, 2].set_ylabel("Delta Vm (mV)")

    for ax in axes[-1, :]:
        ax.set_xlabel("Time (ms)")
    fig.suptitle("Ephaptic influence on Axon B from two 5 nA step-current stimuli on Axon A", fontsize=13)
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
