"""run_branch_zone_test_ascent_diams.py

Простой тест branch-zone модели на одном ветвящемся аксоне B.

Конфигурация по запросу:
  - parent branch diameter = 4.7 um
  - main after-branch diameter = 5.7 um
  - daughter branch diameter = 3.0 um
  - frequencies = 50, 400, 450 Hz

Сохраняет графики Vm в точках:
  - before_branch
  - after_branch_main
  - after_branch_daughter

Окно показа: первые 100 мс.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import find_peaks

from MRG_lib import MRGaxon


FREQUENCIES_HZ = [50.0, 400.0, 450.0]

PARENT_FIBER_DIAM_UM = 5.7
MAIN_AFTER_BRANCH_DIAM_UM = 4.7
DAUGHTER_BRANCH_DIAM_UM = 3.0

PARENT_AXON_NODES = 31
BRANCH_NODES = 11
BRANCHES_NUM = 1
NODES_DIST = 10

DT_MS = 0.005
V_INIT_MV = -80.0
T_START_MS = 10.0
T_END_MS = 110.0
T_STOP_MS = 120.0
AMP_NA = 0.8
PHASE_US = 40.0
GAP_US = 5.0

PLOT_WINDOW_MS = 100.0


def detect_spike_count(t_ms: np.ndarray, v_mV: np.ndarray) -> int:
    valid = np.where(t_ms >= 8.0)[0]
    if valid.size == 0:
        return 0
    dt = float(t_ms[1] - t_ms[0])
    min_dist_pts = max(1, int(round(0.6 / dt)))
    peaks, _ = find_peaks(v_mV[valid[0]:], prominence=5.0, height=-20.0, distance=min_dist_pts)
    return int(len(peaks))


def group_index(axon: MRGaxon, group_name: str) -> int | None:
    idxs = axon.find_segment_by_name(None, group_name=group_name)
    if not idxs:
        return None
    return int(sorted(idxs)[0])


def run_one_frequency(freq_hz: float, out_dir: Path) -> dict:
    axon = MRGaxon(
        fiber_diameter=PARENT_FIBER_DIAM_UM,
        parent_axon_nodes=PARENT_AXON_NODES,
        branch_nodes=BRANCH_NODES,
        branches_num=BRANCHES_NUM,
        nodes_dist=NODES_DIST,
        diam_scale=0.6,
        main_after_branch_fiber_diameter=MAIN_AFTER_BRANCH_DIAM_UM,
        daughter_branch_fiber_diameter=DAUGHTER_BRANCH_DIAM_UM,
        main_transition_nodes=3,
        daughter_transition_nodes=3,
        branch_connector_length_um=1.0,
        branch_connector_diam_scale=1.0,
        celsius=37.0,
        dt_ms=DT_MS,
        v_init=V_INIT_MV,
        h_stop=T_STOP_MS,
    )

    axon.set_stim_target(mode="node_index", node_index=0)
    axon.set_stimulation_params(
        mode="create",
        biphasic=True,
        freq_hz=float(freq_hz),
        amp=AMP_NA,
        t_start=T_START_MS,
        t_end=T_END_MS,
        phase_us=PHASE_US,
        gap_us=GAP_US,
    )

    axon.run_simulation(record_kinetics=False, include_stimulation_point=False)

    idx_before = group_index(axon, "before_branch")
    idx_main = group_index(axon, "after_branch_main")
    idx_dau = group_index(axon, "after_branch_daughter")

    t_ms = np.asarray(axon.time_array, dtype=float)
    plot_mask = t_ms <= float(PLOT_WINDOW_MS)

    fig, axes = plt.subplots(3, 1, figsize=(12.0, 9.0), dpi=180, sharex=True)
    fig.suptitle(
        f"Branched axon test | parent {PARENT_FIBER_DIAM_UM} um | main {MAIN_AFTER_BRANCH_DIAM_UM} um | daughter {DAUGHTER_BRANCH_DIAM_UM} um | {int(freq_hz)} Hz",
        fontsize=13,
    )

    if idx_before is not None:
        axes[0].plot(t_ms[plot_mask], axon.voltage_matrix[idx_before][plot_mask], color="#2563eb", lw=2.0)
    axes[0].set_title("До ветвления")
    axes[0].set_ylabel("Vm (мВ)")
    axes[0].grid(True, alpha=0.25)

    if idx_main is not None:
        axes[1].plot(t_ms[plot_mask], axon.voltage_matrix[idx_main][plot_mask], color="#dc2626", lw=2.0)
    axes[1].set_title("После ветвления: главный путь")
    axes[1].set_ylabel("Vm (мВ)")
    axes[1].grid(True, alpha=0.25)

    if idx_dau is not None:
        axes[2].plot(t_ms[plot_mask], axon.voltage_matrix[idx_dau][plot_mask], color="#16a34a", lw=2.0)
    axes[2].set_title("После ветвления: дочерняя ветвь")
    axes[2].set_xlabel("Время (мс)")
    axes[2].set_ylabel("Vm (мВ)")
    axes[2].grid(True, alpha=0.25)

    fig.tight_layout(rect=(0, 0, 1, 0.96))
    plot_path = out_dir / f"branch_zone_test_{int(freq_hz):04d}Hz.png"
    fig.savefig(plot_path, bbox_inches="tight")
    plt.close(fig)

    summary = {
        "freq_hz": float(freq_hz),
        "n_before": detect_spike_count(t_ms, axon.voltage_matrix[idx_before]) if idx_before is not None else 0,
        "n_main": detect_spike_count(t_ms, axon.voltage_matrix[idx_main]) if idx_main is not None else 0,
        "n_daughter": detect_spike_count(t_ms, axon.voltage_matrix[idx_dau]) if idx_dau is not None else 0,
        "plot_path": str(plot_path),
    }
    print(summary)
    return summary


def main():
    out_dir = Path(__file__).resolve().parent / "data" / "branch_zone_test_ascent_diams"
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = [run_one_frequency(freq_hz, out_dir) for freq_hz in FREQUENCIES_HZ]
    df = pd.DataFrame(rows)
    csv_path = out_dir / "summary.csv"
    df.to_csv(csv_path, index=False)
    print(f"Saved summary: {csv_path}")


main()
