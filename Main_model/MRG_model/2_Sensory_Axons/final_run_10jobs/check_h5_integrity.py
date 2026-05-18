from __future__ import annotations

from pathlib import Path
import json
import os

import h5py
import numpy as np


ROOT_DIR = Path(__file__).resolve().parents[1]

EXPECTED_TOPOLOGIES = ["one_node_branching"]
EXPECTED_DIAMETERS = [5.7, 2.5]
EXPECTED_SCENARIOS = ["one_branch", "multiple_branches"]
EXPECTED_DISTANCES = [0.1, 0.5, 1.0]
EXPECTED_MODES = ["aligned", "misaligned_0.5", "misaligned_0.25", "no_EC", "no_EC_isolated"]
EXPECTED_STIM_PROTOCOL = os.getenv("STIM_PROTOCOL", "sync")
EXPECTED_STIM_B_DELAY_MS = float(os.getenv("STIM_B_DELAY_MS", "0.5"))


def expected_amp_nA(diameter: float) -> float:
    if np.isclose(float(diameter), 2.5):
        return -1.0
    if np.isclose(float(diameter), 5.7):
        return -5.0
    raise ValueError(f"Unsupported diameter={diameter}; expected 2.5 or 5.7")


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
        return f"delay_{float(stim_b_delay_ms):g}ms".replace(".", "p")
    raise ValueError("stim_protocol must be 'sync' or 'delay'")


def prefix_for(topology: str, scenario: str) -> str:
    if topology == "one_node_branching":
        return "on_ob" if scenario == "one_branch" else "on_mb"
    return "cb_ob" if scenario == "one_branch" else "cb_mb"


def mode_tag(mode: str) -> str:
    return {
        "aligned": "aligned",
        "misaligned_0.5": "misaligned_0.5",
        "misaligned_0.25": "misaligned_0.25",
        "no_EC": "no_EC",
        "no_EC_isolated": "no_EC_isolated",
    }[mode]


def expected_h5_path(
    base_out: Path,
    topology: str,
    diameter: float,
    scenario: str,
    distance: float,
    mode: str,
    *,
    stim_protocol: str,
    stim_b_delay_ms: float,
) -> Path:
    amp_tag = amp_filename_tag(expected_amp_nA(diameter))
    protocol_tag = stim_protocol_tag(stim_protocol, stim_b_delay_ms)
    return (
        base_out
        / topology
        / f"fiber_d_{diameter}_um"
        / scenario
        / f"distance_{distance}"
        / f"{prefix_for(topology, scenario)}_fd{diameter}_ed{distance}_{mode_tag(mode)}_{protocol_tag}_{amp_tag}.h5"
    )


def validate_h5_file(
    h5_path: Path,
    *,
    expected_amp: float | None = None,
    expected_stim_protocol: str = EXPECTED_STIM_PROTOCOL,
    expected_stim_b_delay_ms: float = EXPECTED_STIM_B_DELAY_MS,
) -> tuple[bool, list[str]]:
    errors: list[str] = []
    if not h5_path.exists():
        return False, [f"missing file: {h5_path}"]

    try:
        with h5py.File(h5_path, "r") as f:
            for attr in [
                "topology",
                "scenario",
                "fiber_diameter_um",
                "edge_dist_um",
                "mode",
                "amp_nA",
                "stim_description",
                "dt_ms",
                "h_stop_ms",
                "stim_protocol",
                "stim_protocol_tag",
                "stim_A_start_ms",
                "stim_B_start_ms",
                "stim_B_delay_ms",
                "created_by",
                "test_mode",
                "frequencies_hz",
            ]:
                if attr not in f.attrs:
                    errors.append(f"missing root attr {attr}")

            for grp_name in ["AxonA_params", "AxonB_params", "Summary"]:
                if grp_name not in f:
                    errors.append(f"missing group {grp_name}")

            if expected_amp is not None and "amp_nA" in f.attrs:
                actual_amp = float(f.attrs["amp_nA"])
                if not np.isclose(actual_amp, float(expected_amp)):
                    errors.append(f"amp_nA={actual_amp:g}, expected {float(expected_amp):g}")

            if "stim_protocol" in f.attrs and str(f.attrs["stim_protocol"]) != str(expected_stim_protocol):
                errors.append(f"stim_protocol={f.attrs['stim_protocol']}, expected {expected_stim_protocol}")
            expected_delay = float(expected_stim_b_delay_ms if expected_stim_protocol == "delay" else 0.0)
            if "stim_B_delay_ms" in f.attrs and not np.isclose(float(f.attrs["stim_B_delay_ms"]), expected_delay):
                errors.append(f"stim_B_delay_ms={float(f.attrs['stim_B_delay_ms']):g}, expected {expected_delay:g}")

            freq_groups = sorted([name for name in f.keys() if name.startswith("Frequency_")])
            if not freq_groups:
                errors.append("no Frequency_* groups")
            else:
                for grp_name in freq_groups:
                    grp = f[grp_name]
                    for ax_name in ["AxonA", "AxonB"]:
                        if ax_name not in grp:
                            errors.append(f"{grp_name} missing {ax_name}")
                            continue
                        model_grp = grp[ax_name].get("Model", None)
                        if model_grp is None:
                            errors.append(f"{grp_name}/{ax_name} missing Model")
                            continue
                        if "time" not in model_grp:
                            errors.append(f"{grp_name}/{ax_name}/Model missing time")
                        traces = model_grp.get("Traces", None)
                        if traces is None or not list(traces.keys()):
                            errors.append(f"{grp_name}/{ax_name}/Model/Traces empty")
                        else:
                            first_group = next(iter(traces.keys()))
                            first_nodes = traces[first_group]
                            if not list(first_nodes.keys()):
                                errors.append(f"{grp_name}/{ax_name}/Model/Traces/{first_group} empty")
                            else:
                                first_node = next(iter(first_nodes.keys()))
                                voltage = first_nodes[first_node].get("voltage", None)
                                if voltage is None:
                                    errors.append(f"{grp_name}/{ax_name}/Model/Traces/{first_group}/{first_node} missing voltage")
                                else:
                                    arr = voltage[()]
                                    if arr.size == 0:
                                        errors.append(f"{grp_name}/{ax_name}/Model/Traces/{first_group}/{first_node} empty voltage")
                                    elif np.isnan(arr).any():
                                        errors.append(f"{grp_name}/{ax_name}/Model/Traces/{first_group}/{first_node} contains NaN")

            if "Summary" in f and isinstance(f["Summary"], h5py.Group):
                if not list(f["Summary"].keys()):
                    errors.append("Summary group empty")
    except Exception as exc:
        errors.append(f"exception while reading file: {exc}")

    return len(errors) == 0, errors


def run_validation(
    base_out: Path,
    *,
    stim_protocol: str = EXPECTED_STIM_PROTOCOL,
    stim_b_delay_ms: float = EXPECTED_STIM_B_DELAY_MS,
) -> tuple[bool, list[str]]:
    failures: list[str] = []
    for topology in EXPECTED_TOPOLOGIES:
        for diameter in EXPECTED_DIAMETERS:
            for scenario in EXPECTED_SCENARIOS:
                for distance in EXPECTED_DISTANCES:
                    for mode in EXPECTED_MODES:
                        h5_path = expected_h5_path(
                            base_out,
                            topology,
                            diameter,
                            scenario,
                            distance,
                            mode,
                            stim_protocol=stim_protocol,
                            stim_b_delay_ms=stim_b_delay_ms,
                        )
                        ok, errors = validate_h5_file(
                            h5_path,
                            expected_amp=expected_amp_nA(diameter),
                            expected_stim_protocol=stim_protocol,
                            expected_stim_b_delay_ms=stim_b_delay_ms,
                        )
                        if not ok:
                            failures.append(str(h5_path))
                            failures.extend(f"  - {msg}" for msg in errors)
    return len(failures) == 0, failures


def main():
    base_out = ROOT_DIR / "final_result"
    ok, failures = run_validation(base_out, stim_protocol=EXPECTED_STIM_PROTOCOL, stim_b_delay_ms=EXPECTED_STIM_B_DELAY_MS)
    if ok:
        print("ACCEPTANCE TEST PASSED")
    else:
        print("ACCEPTANCE TEST FAILED")
        for item in failures:
            print(item)


if __name__ == "__main__":
    main()
