from __future__ import annotations

from pathlib import Path
import json

import h5py
import numpy as np


ROOT_DIR = Path(__file__).resolve().parents[1]

EXPECTED_TOPOLOGIES = ["one_node_branching", "connector_branching"]
EXPECTED_DIAMETERS = [5.7, 2.5]
EXPECTED_SCENARIOS = ["one_branch", "multiple_branches"]
EXPECTED_DISTANCES = [0.1, 0.5, 1.0]
EXPECTED_MODES = ["aligned", "misaligned_0.5", "misaligned_0.25", "no_EC"]


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
    }[mode]


def expected_h5_path(base_out: Path, topology: str, diameter: float, scenario: str, distance: float, mode: str) -> Path:
    return (
        base_out
        / topology
        / f"fiber_d_{diameter}_um"
        / scenario
        / f"distance_{distance}"
        / f"{prefix_for(topology, scenario)}_fd{diameter}_ed{distance}_{mode_tag(mode)}_amp5.h5"
    )


def validate_h5_file(h5_path: Path) -> tuple[bool, list[str]]:
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
                "created_by",
                "test_mode",
                "frequencies_hz",
            ]:
                if attr not in f.attrs:
                    errors.append(f"missing root attr {attr}")

            for grp_name in ["AxonA_params", "AxonB_params", "Summary"]:
                if grp_name not in f:
                    errors.append(f"missing group {grp_name}")

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


def run_validation(base_out: Path) -> tuple[bool, list[str]]:
    failures: list[str] = []
    for topology in EXPECTED_TOPOLOGIES:
        for diameter in EXPECTED_DIAMETERS:
            for scenario in EXPECTED_SCENARIOS:
                for distance in EXPECTED_DISTANCES:
                    for mode in EXPECTED_MODES:
                        h5_path = expected_h5_path(base_out, topology, diameter, scenario, distance, mode)
                        ok, errors = validate_h5_file(h5_path)
                        if not ok:
                            failures.append(str(h5_path))
                            failures.extend(f"  - {msg}" for msg in errors)
    return len(failures) == 0, failures


def main():
    base_out = ROOT_DIR / "final_result"
    ok, failures = run_validation(base_out)
    if ok:
        print("ACCEPTANCE TEST PASSED")
    else:
        print("ACCEPTANCE TEST FAILED")
        for item in failures:
            print(item)


if __name__ == "__main__":
    main()
