from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import json
import os
import sys

import h5py
import numpy as np
import pandas as pd
from scipy.signal import find_peaks

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from MRG_lib import TwoSensoryAxonsPrescott  # noqa: E402


DEFAULT_TOPOLOGY_NAME = "one_node_branching"
DEFAULT_FIBER_DIAMETER_UM = 5.7
DEFAULT_SCENARIO_NAME = "one_branch"
DEFAULT_STIM_PROTOCOL = "sync"
DEFAULT_STIM_B_DELAY_MS = 0.5


AXON_A_SUMMARY_LABELS = ["stimulation_point", "before_like", "main_like", "terminal_main"]
AXON_B_ONE_BRANCH_LABELS = ["stimulation_point", "before_branch", "after_branch_main", "terminal_main"]
AXON_BRANCH_LABELS = [
    "stimulation_point",
    "before_branch",
    "after_branch_main",
    "after_branch_daughter",
    "branch_point",
    "terminal_main",
    "terminal_daughter",
]

MODE_CONFIGS = {
    "aligned": {"filename_tag": "aligned"},
    "misaligned_0.5": {"filename_tag": "misaligned_0.5"},
    "misaligned_0.25": {"filename_tag": "misaligned_0.25"},
    "no_EC": {"filename_tag": "no_EC"},
    "no_EC_isolated": {"filename_tag": "no_EC_isolated"},
}


def stimulus_amp_nA(fiber_diameter_um: float) -> float:
    if np.isclose(float(fiber_diameter_um), 2.5):
        return -1.0
    if np.isclose(float(fiber_diameter_um), 5.7):
        return -5.0
    raise ValueError(f"Unsupported fiber_diameter_um={fiber_diameter_um}; expected 2.5 or 5.7")


def amp_filename_tag(amp_nA: float) -> str:
    amp_abs = abs(float(amp_nA))
    if np.isclose(amp_abs, round(amp_abs)):
        return f"amp{int(round(amp_abs))}"
    return f"amp{amp_abs:g}".replace(".", "p")


def stim_protocol_tag(stim_protocol: str, stim_b_delay_ms: float = 0.0) -> str:
    protocol = str(stim_protocol).strip()
    if protocol == "sync":
        return "sync"
    if protocol == "delay":
        delay = float(stim_b_delay_ms)
        if delay < 0.0:
            raise ValueError(f"stim_b_delay_ms must be >= 0, got {delay}")
        return f"delay_{delay:g}ms".replace(".", "p")
    raise ValueError("stim_protocol must be 'sync' or 'delay'")


def frequency_group_name(freq_hz: int | float) -> str:
    return f"Frequency_{int(freq_hz):03d}Hz"


@dataclass(frozen=True)
class LaunchConfig:
    topology_name: str
    branch_topology_mode: str
    fiber_diameter_um: float
    scenario_name: str
    scenario_dir: str
    prefix: str
    parent_axon_nodes_A: int
    branch_nodes_A: int
    branches_num_A: int
    branch_sequence_nodes_A: list[int] | None
    parent_axon_nodes_B: int
    branch_nodes_B: int
    branches_num_B: int
    branch_sequence_nodes_B: list[int] | None
    edge_distances_um: list[float]
    frequencies_hz: list[int]
    amp_nA: float
    t_start_ms: float
    t_end_ms: float
    h_stop_ms: float
    dt_ms: float
    phase_us: float
    gap_us: float
    biphasic: bool


def build_launch_config(topology_name: str, fiber_diameter_um: float, scenario_name: str, *, test_mode: bool = False) -> LaunchConfig:
    if topology_name not in {"connector_branching", "one_node_branching"}:
        raise ValueError(f"Unknown topology_name={topology_name}")
    if scenario_name not in {"one_branch", "multiple_branches"}:
        raise ValueError(f"Unknown scenario_name={scenario_name}")

    branch_topology_mode = "connector_legacy" if topology_name == "connector_branching" else "node"
    edge_distances = [0.1, 0.5, 1.0]
    frequencies = [50, 500] if test_mode else list(range(50, 1001, 50))
    if test_mode:
        t_start_ms = 0.1
        t_end_ms = 10.0
        h_stop_ms = 10.0
        dt_ms = 0.01
    else:
        t_start_ms = 10.0
        t_end_ms = 1010.0
        h_stop_ms = 1010.0
        dt_ms = 0.005

    if scenario_name == "one_branch":
        # One branch: 8 nodes before branch, then 18 nodes along the main path.
        return LaunchConfig(
            topology_name=topology_name,
            branch_topology_mode=branch_topology_mode,
            fiber_diameter_um=float(fiber_diameter_um),
            scenario_name=scenario_name,
            scenario_dir="one_branch",
            prefix="cb_ob" if topology_name == "connector_branching" else "on_ob",
            parent_axon_nodes_A=27,
            branch_nodes_A=8,
            branches_num_A=0,
            branch_sequence_nodes_A=None,
            parent_axon_nodes_B=27,
            branch_nodes_B=8,
            branches_num_B=1,
            branch_sequence_nodes_B=[8],
            edge_distances_um=edge_distances,
            frequencies_hz=frequencies,
            amp_nA=stimulus_amp_nA(fiber_diameter_um),
            t_start_ms=t_start_ms,
            t_end_ms=t_end_ms,
            h_stop_ms=h_stop_ms,
            dt_ms=dt_ms,
            phase_us=40.0,
            gap_us=5.0,
            biphasic=True,
        )

    # (8) - branch - (4) - branch - (4) - branch - (4) - branch - (8)
    return LaunchConfig(
        topology_name=topology_name,
        branch_topology_mode=branch_topology_mode,
        fiber_diameter_um=float(fiber_diameter_um),
        scenario_name=scenario_name,
        scenario_dir="multiple_branches",
        prefix="cb_mb" if topology_name == "connector_branching" else "on_mb",
        parent_axon_nodes_A=29,
        branch_nodes_A=8,
        branches_num_A=4,
        branch_sequence_nodes_A=[8, 4, 4, 4],
        parent_axon_nodes_B=29,
        branch_nodes_B=8,
        branches_num_B=4,
        branch_sequence_nodes_B=[8, 4, 4, 4],
        edge_distances_um=edge_distances,
        frequencies_hz=frequencies,
        amp_nA=stimulus_amp_nA(fiber_diameter_um),
        t_start_ms=t_start_ms,
        t_end_ms=t_end_ms,
        h_stop_ms=h_stop_ms,
        dt_ms=dt_ms,
        phase_us=40.0,
        gap_us=5.0,
        biphasic=True,
    )


def detect_spike_count(t_ms: np.ndarray, v_mV: np.ndarray) -> int:
    valid = np.where(t_ms >= (0.0 if t_ms[-1] <= 2.0 else 8.0))[0]
    if valid.size == 0:
        return 0
    dt = float(t_ms[1] - t_ms[0])
    min_dist_pts = max(1, int(round(0.6 / max(dt, 1e-9))))
    peaks, _ = find_peaks(v_mV[valid[0]:], prominence=5.0, height=-20.0, distance=min_dist_pts)
    return int(len(peaks))


def write_root_metadata(
    h5_path: Path,
    cfg: LaunchConfig,
    *,
    edge_dist_um: float,
    mode_name: str,
    test_mode: bool,
    stim_protocol: str,
    stim_b_delay_ms: float,
    h_stop_ms: float,
):
    stim_a_start_ms = float(cfg.t_start_ms)
    stim_b_start_ms = float(cfg.t_start_ms + (float(stim_b_delay_ms) if stim_protocol == "delay" else 0.0))
    with h5py.File(h5_path, "w") as f:
        f.attrs["topology"] = cfg.topology_name
        f.attrs["scenario"] = cfg.scenario_name
        f.attrs["fiber_diameter_um"] = float(cfg.fiber_diameter_um)
        f.attrs["edge_dist_um"] = float(edge_dist_um)
        f.attrs["mode"] = str(mode_name)
        f.attrs["amp_nA"] = float(cfg.amp_nA)
        f.attrs["stim_description"] = f"biphasic phase_us={cfg.phase_us} gap_us={cfg.gap_us}"
        f.attrs["dt_ms"] = float(cfg.dt_ms)
        f.attrs["h_stop_ms"] = float(h_stop_ms)
        f.attrs["stim_protocol"] = str(stim_protocol)
        f.attrs["stim_protocol_tag"] = stim_protocol_tag(stim_protocol, stim_b_delay_ms)
        f.attrs["stim_A_start_ms"] = float(stim_a_start_ms)
        f.attrs["stim_B_start_ms"] = float(stim_b_start_ms)
        f.attrs["stim_B_delay_ms"] = float(stim_b_delay_ms if stim_protocol == "delay" else 0.0)
        f.attrs["created_by"] = "final_run_8jobs/final_results_worker.py"
        f.attrs["test_mode"] = int(bool(test_mode))
        f.attrs["frequencies_hz"] = json.dumps(list(cfg.frequencies_hz))

        grpA = f.create_group("AxonA_params")
        grpA.attrs["parent_nodes"] = int(cfg.parent_axon_nodes_A)
        grpA.attrs["branch_nodes"] = int(cfg.branch_nodes_A)
        grpA.attrs["branches_num"] = int(cfg.branches_num_A)
        grpA.attrs["branch_sequence_nodes"] = json.dumps(cfg.branch_sequence_nodes_A)

        grpB = f.create_group("AxonB_params")
        grpB.attrs["parent_nodes"] = int(cfg.parent_axon_nodes_B)
        grpB.attrs["branch_nodes"] = int(cfg.branch_nodes_B)
        grpB.attrs["branches_num"] = int(cfg.branches_num_B)
        grpB.attrs["branch_sequence_nodes"] = json.dumps(cfg.branch_sequence_nodes_B)

        f.create_group("Summary")


def finalize_summary(h5_path: Path, summary_rows: list[dict]):
    df = pd.DataFrame(summary_rows)
    with h5py.File(h5_path, "a") as f:
        grp = f["Summary"]
        for name in list(grp.keys()):
            del grp[name]
        for col in df.columns:
            values = df[col].to_numpy()
            if values.dtype.kind in {"U", "O"}:
                values = values.astype("S")
            grp.create_dataset(col, data=values)


def is_complete_output_file(h5_path: Path, cfg: LaunchConfig, *, stim_protocol: str, stim_b_delay_ms: float, h_stop_ms: float) -> bool:
    if not h5_path.exists():
        return False

    expected_groups = [frequency_group_name(freq_hz) for freq_hz in cfg.frequencies_hz]

    try:
        with h5py.File(h5_path, "r") as f:
            if str(f.attrs.get("stim_protocol", "")) != str(stim_protocol):
                return False
            if not np.isclose(float(f.attrs.get("stim_B_delay_ms", np.nan)), float(stim_b_delay_ms if stim_protocol == "delay" else 0.0)):
                return False
            if not np.isclose(float(f.attrs.get("h_stop_ms", np.nan)), float(h_stop_ms)):
                return False
            if "Summary" not in f or not list(f["Summary"].keys()):
                return False

            for group_name in expected_groups:
                if group_name not in f:
                    return False
                grp = f[group_name]
                for ax_name in ("AxonA", "AxonB"):
                    if ax_name not in grp:
                        return False
                    model_grp = grp[ax_name].get("Model", None)
                    if model_grp is None:
                        return False
                    if "time" not in model_grp:
                        return False
                    traces = model_grp.get("Traces", None)
                    if traces is None or not list(traces.keys()):
                        return False
                    first_group = next(iter(traces.keys()))
                    first_nodes = traces[first_group]
                    if not list(first_nodes.keys()):
                        return False
                    first_node = next(iter(first_nodes.keys()))
                    voltage = first_nodes[first_node].get("voltage", None)
                    if voltage is None:
                        return False
                    arr = voltage[()]
                    if arr.size == 0 or np.isnan(arr).any():
                        return False
    except Exception:
        return False

    return True


def build_output_dir(cfg: LaunchConfig, edge_dist_um: float) -> Path:
    return (
        ROOT_DIR
        / "final_result"
        / cfg.topology_name
        / f"fiber_d_{cfg.fiber_diameter_um}_um"
        / cfg.scenario_dir
        / f"distance_{edge_dist_um}"
    )


def build_h5_name(cfg: LaunchConfig, edge_dist_um: float, mode_name: str, *, stim_protocol: str, stim_b_delay_ms: float) -> str:
    tag = MODE_CONFIGS[mode_name]["filename_tag"]
    protocol_tag = stim_protocol_tag(stim_protocol, stim_b_delay_ms)
    amp_tag = amp_filename_tag(cfg.amp_nA)
    return f"{cfg.prefix}_fd{cfg.fiber_diameter_um}_ed{edge_dist_um}_{tag}_{protocol_tag}_{amp_tag}.h5"


def build_model(cfg: LaunchConfig, edge_dist_um: float, mode_name: str, *, h_stop_ms: float | None = None) -> TwoSensoryAxonsPrescott:
    return TwoSensoryAxonsPrescott(
        fiber_diameter_um=float(cfg.fiber_diameter_um),
        edge_dist_um=float(edge_dist_um),
        mode_descriptor=mode_name,
        ec_strength_scale=1.0,
        branch_topology_mode_A=cfg.branch_topology_mode,
        branch_topology_mode_B=cfg.branch_topology_mode,
        branch_connector_length_um=1.0,
        branch_connector_diam_scale=1.0,
        main_after_branch_diam_scale=0.6,
        daughter_branch_diam_scale=0.6,
        main_after_branch_param_mode="scaled_radial",
        daughter_branch_param_mode="ascent_full",
        dt_ms=float(cfg.dt_ms),
        v_init=-80.0,
        h_stop=float(cfg.h_stop_ms if h_stop_ms is None else h_stop_ms),
        celsius=37.0,
        boundary_full_cable=False,
        parent_axon_nodes_A=int(cfg.parent_axon_nodes_A),
        branch_nodes_A=int(cfg.branch_nodes_A),
        branches_num_A=int(cfg.branches_num_A),
        branch_sequence_nodes_A=cfg.branch_sequence_nodes_A,
        parent_axon_nodes_B=int(cfg.parent_axon_nodes_B),
        branch_nodes_B=int(cfg.branch_nodes_B),
        branches_num_B=int(cfg.branches_num_B),
        branch_sequence_nodes_B=cfg.branch_sequence_nodes_B,
    )


def run_one_file(
    cfg: LaunchConfig,
    *,
    edge_dist_um: float,
    mode_name: str,
    test_mode: bool,
    stim_protocol: str,
    stim_b_delay_ms: float,
) -> Path:
    stim_protocol = str(stim_protocol)
    delay_ms = float(stim_b_delay_ms if stim_protocol == "delay" else 0.0)
    protocol_tag = stim_protocol_tag(stim_protocol, delay_ms)
    h_stop_ms = float(cfg.h_stop_ms + delay_ms)
    out_dir = build_output_dir(cfg, edge_dist_um)
    out_dir.mkdir(parents=True, exist_ok=True)
    h5_path = out_dir / build_h5_name(cfg, edge_dist_um, mode_name, stim_protocol=stim_protocol, stim_b_delay_ms=delay_ms)
    if is_complete_output_file(h5_path, cfg, stim_protocol=stim_protocol, stim_b_delay_ms=delay_ms, h_stop_ms=h_stop_ms):
        print(f"Skipping complete HDF5: {h5_path}")
        return h5_path

    if h5_path.exists():
        h5_path.unlink()

    write_root_metadata(
        h5_path,
        cfg,
        edge_dist_um=edge_dist_um,
        mode_name=mode_name,
        test_mode=test_mode,
        stim_protocol=stim_protocol,
        stim_b_delay_ms=delay_ms,
        h_stop_ms=h_stop_ms,
    )

    summary_rows = []
    model = build_model(cfg, edge_dist_um, mode_name, h_stop_ms=h_stop_ms)
    for freq_hz in cfg.frequencies_hz:
        stim_kwargs_A = {
            "mode": "create",
            "biphasic": bool(cfg.biphasic),
            "freq_hz": float(freq_hz),
            "amp": float(cfg.amp_nA),
            "t_start": float(cfg.t_start_ms),
            "t_end": float(cfg.t_end_ms),
            "phase_us": float(cfg.phase_us),
            "gap_us": float(cfg.gap_us),
        }
        stim_kwargs_B = dict(stim_kwargs_A)
        stim_kwargs_B["t_start"] = float(cfg.t_start_ms + delay_ms)
        stim_kwargs_B["t_end"] = float(cfg.t_end_ms + delay_ms)
        model.set_stimulation_for_axons_independent(
            stim_A=True,
            stim_B=True,
            stim_target_mode_A="node_index",
            stim_node_index_A=0,
            stim_x_um_A=None,
            stim_target_mode_B="node_index",
            stim_node_index_B=0,
            stim_x_um_B=None,
            stim_kwargs_A=stim_kwargs_A,
            stim_kwargs_B=stim_kwargs_B,
        )
        exp_name = frequency_group_name(freq_hz)
        model.run_simulation_two_axons(
            h5_path=str(h5_path),
            experiment_name=exp_name,
            record_kinetics=False,
            include_stimulation_point=True,
            record_axonA_before_like=True,
            record_axonA_main_like=True,
            record_terminal_nodes=True,
        )

        row = {"freq_hz": float(freq_hz), "success": 1, "stim_protocol": protocol_tag, "stim_B_delay_ms": delay_ms}
        labelsA = list(getattr(model.axonA, "recording_labels", []) or [])
        labelsB = list(getattr(model.axonB, "recording_labels", []) or [])
        for label in AXON_A_SUMMARY_LABELS:
            if label in labelsA:
                idx = labelsA.index(label)
                row[f"AxonA_n_spikes_{label}"] = detect_spike_count(np.asarray(model.axonA.time_array), model.axonA.voltage_matrix[idx])
        for label in AXON_B_ONE_BRANCH_LABELS if cfg.scenario_name == "one_branch" else AXON_BRANCH_LABELS:
            if label in labelsB:
                idx = labelsB.index(label)
                row[f"AxonB_n_spikes_{label}"] = detect_spike_count(np.asarray(model.axonB.time_array), model.axonB.voltage_matrix[idx])
        summary_rows.append(row)

    finalize_summary(h5_path, summary_rows)
    print(f"Saved HDF5: {h5_path}")
    return h5_path


def run_launch(
    cfg: LaunchConfig,
    *,
    test_mode: bool = False,
    mode_filter: str | None = None,
    stim_protocol: str = DEFAULT_STIM_PROTOCOL,
    stim_b_delay_ms: float = DEFAULT_STIM_B_DELAY_MS,
) -> list[Path]:
    outputs = []
    mode_names = list(MODE_CONFIGS)
    if mode_filter is not None:
        mode_filter = str(mode_filter)
        if mode_filter not in MODE_CONFIGS:
            raise ValueError(f"Unknown mode_filter={mode_filter}; expected one of {sorted(MODE_CONFIGS)}")
        mode_names = [mode_filter]
    for edge_dist_um in cfg.edge_distances_um:
        for mode_name in mode_names:
            outputs.append(
                run_one_file(
                    cfg,
                    edge_dist_um=float(edge_dist_um),
                    mode_name=mode_name,
                    test_mode=test_mode,
                    stim_protocol=stim_protocol,
                    stim_b_delay_ms=stim_b_delay_ms,
                )
            )
    return outputs


def main():
    topology_name = str(os.getenv("TOPOLOGY_NAME", DEFAULT_TOPOLOGY_NAME))
    fiber_diameter_um = float(os.getenv("FIBER_DIAMETER_UM", str(DEFAULT_FIBER_DIAMETER_UM)))
    scenario_name = str(os.getenv("SCENARIO_NAME", DEFAULT_SCENARIO_NAME))
    mode_filter = os.getenv("MODE_NAME")
    stim_protocol = str(os.getenv("STIM_PROTOCOL", DEFAULT_STIM_PROTOCOL))
    stim_b_delay_ms = float(os.getenv("STIM_B_DELAY_MS", str(DEFAULT_STIM_B_DELAY_MS)))

    if len(sys.argv) >= 4:
        topology_name = str(sys.argv[1])
        fiber_diameter_um = float(sys.argv[2])
        scenario_name = str(sys.argv[3])
    if len(sys.argv) >= 5:
        mode_filter = str(sys.argv[4])
    if len(sys.argv) >= 6:
        stim_protocol = str(sys.argv[5])
    if len(sys.argv) >= 7:
        stim_b_delay_ms = float(sys.argv[6])

    test_mode = bool(int(os.getenv("TEST_MODE", "0")))
    cfg = build_launch_config(topology_name, fiber_diameter_um, scenario_name, test_mode=test_mode)
    outputs = run_launch(
        cfg,
        test_mode=test_mode,
        mode_filter=mode_filter,
        stim_protocol=stim_protocol,
        stim_b_delay_ms=stim_b_delay_ms,
    )
    mode_label = mode_filter if mode_filter is not None else "all_modes"
    protocol_label = stim_protocol_tag(stim_protocol, stim_b_delay_ms if stim_protocol == "delay" else 0.0)
    print(f"Created {len(outputs)} HDF5 files for {cfg.topology_name}, fd={cfg.fiber_diameter_um}, scenario={cfg.scenario_name}, mode={mode_label}, protocol={protocol_label}")


if __name__ == "__main__":
    main()
