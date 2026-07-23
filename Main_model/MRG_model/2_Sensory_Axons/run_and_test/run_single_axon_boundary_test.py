"""run_single_axon_boundary_test.py

Single axon + Prescott-style boundary cable.

How to use:
1. Edit parameters in the CONFIG section below.
2. Run the file from PyCharm or `python run_single_axon_boundary_test.py`.
3. Results are written into `data/single_axon_boundary_test`.

No CLI arguments are required. All settings live in this file.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import find_peaks

from MRG_lib import MRGaxon, BoundaryCable, LinearMechanismCoupler, EphapticSpec, _compute_rg_dimless_from_centers


# =====================================================================================
# CONFIG
# =====================================================================================

OUT_DIR = Path(__file__).resolve().parent / "data" / "single_axon_boundary_test"

FREQUENCIES_HZ = [50.0, 400.0, 1000.0]

MORPH = {
    "fiber_diameter_um": 5.7,
    "parent_axon_nodes": 31,
    "branch_nodes": 11,
    "branches_num": 1,
    "nodes_dist": 10,
    "branch_node_scale": 1.0,
    "main_after_branch_scale": 0.6,
    "daughter_branch_scale": 0.6,
}

BOUNDARY = {
    # Prescott layer-1 settings on the axon side.
    "edge_dist_um": 0.1,
    "XG1": 1e-9,
    # Perineurium resistance used for axon <-> boundary coupling.
    "rho_perineurium_ohm_cm": 1.136e5,
    "perineurium_thickness_cm": 4.7e-4,
    # Full cable = 4000 sections. Sparse = only sections that are actually coupled.
    "boundary_full_cable": False,
    "boundary_n_sections": 4000,
    "boundary_total_length_um": 40000.0,
}

STIM = {
    "target_mode": "node_index",  # "node_index" or "same_x_um"
    "target_node_index": 0,
    "target_x_um": None,
    "amp_nA": 5.0,
    "t_start_ms": 10.0,
    "t_end_ms": 110.0,
    "phase_us": 40.0,
    "gap_us": 5.0,
    "biphasic": True,
}

SIM = {
    "dt_ms": 0.005,
    "v_init_mV": -80.0,
    "t_stop_ms": 120.0,
    "celsius": 37.0,
    "record_kinetics": False,
}

PLOT = {
    "window_ms": 100.0,
    # Extra main-path node indices are useful when branches_num = 0.
    "extra_node_indices": [0, 10, -1],
    # Standard branch-related labels are added when available.
    "include_terminal_main": True,
    "figure_width": 12.0,
    "figure_height_per_panel": 2.8,
}


class SingleAxonBoundaryModel:
    """Minimal single-axon wrapper that reuses existing boundary-cable logic."""

    def __init__(self):
        self.axon = MRGaxon(
            fiber_diameter=MORPH["fiber_diameter_um"],
            parent_axon_nodes=MORPH["parent_axon_nodes"],
            branch_nodes=MORPH["branch_nodes"],
            branches_num=MORPH["branches_num"],
            nodes_dist=MORPH["nodes_dist"],
            branch_node_scale=MORPH["branch_node_scale"],
            main_after_branch_scale=MORPH["main_after_branch_scale"],
            daughter_branch_scale=MORPH["daughter_branch_scale"],
            celsius=SIM["celsius"],
            dt_ms=SIM["dt_ms"],
            v_init=SIM["v_init_mV"],
            h_stop=SIM["t_stop_ms"],
        )

        # Prescott external layer on the axon side.
        self.xr = self.axon.apply_prescott_extracellular_layer1(
            edge_dist_um=BOUNDARY["edge_dist_um"],
            XG1=BOUNDARY["XG1"],
        )

        self.boundary = None
        self.boundary_spec = None
        self.boundary_coupler = None
        self._hoc_keepalive = []
        self._build_boundary_coupler()

    def _spec_boundary_for_axon(self, boundary: BoundaryCable | None) -> EphapticSpec:
        # Same mapping rule as in TwoSensoryAxonsPrescott:
        # node_i <-> nearest boundary.section_j by longitudinal X coordinate.
        trunk = list(self.axon.main_axon)
        extra_nodes = []
        for branch in getattr(self.axon, "branches", []) or []:
            extra_nodes.extend(list(branch))

        seen = set()
        all_nodes = []
        for sec in trunk + extra_nodes:
            nm = sec.name()
            if nm in seen:
                continue
            seen.add(nm)
            all_nodes.append(sec)

        if boundary is not None:
            sec_len = float(boundary.section_length_um)
            nsec = int(boundary.n_sections)
        else:
            sec_len = float(BOUNDARY["boundary_total_length_um"]) / float(BOUNDARY["boundary_n_sections"])
            nsec = int(BOUNDARY["boundary_n_sections"])

        lstep = float(self.axon.mrg_params.get("Lstep", 1.0))
        pairs = []
        for i, sec in enumerate(all_nodes):
            node_name = sec.name()
            x = float(self.axon.node_distance_um.get(node_name, i * lstep))
            bi = int(max(0, min(nsec - 1, round(x / sec_len))))
            pairs.append((x, node_name, f"section_{bi}"))

        pairs.sort(key=lambda item: float(item[0]))

        xs = []
        names_first = []
        names_second = []
        last_x = None
        eps = 1e-3
        for x, node_name, bname in pairs:
            x = float(x)
            if last_x is not None and x <= last_x:
                x = last_x + eps
            last_x = x
            xs.append(x)
            names_first.append(node_name)
            names_second.append(bname)

        s_um = float(self.axon.fiber_diameter)
        rg_dimless = _compute_rg_dimless_from_centers(np.asarray(xs, dtype=float), s_um)
        return EphapticSpec(names_first, names_second, rg_dimless)

    def _build_boundary_coupler(self):
        spec_tmp = self._spec_boundary_for_axon(boundary=None)
        sparse_names = None if BOUNDARY["boundary_full_cable"] else spec_tmp.sec_names_second

        self.boundary = BoundaryCable(
            name_prefix="single_b_",
            n_sections=BOUNDARY["boundary_n_sections"],
            total_length_um=BOUNDARY["boundary_total_length_um"],
            sparse_section_names=sparse_names,
        )
        self.boundary.set_grounded_sink()

        self.boundary_spec = self._spec_boundary_for_axon(boundary=self.boundary)

        rd_b = (
            float(BOUNDARY["rho_perineurium_ohm_cm"]) * 10000.0
            * float(BOUNDARY["perineurium_thickness_cm"]) * 10000.0
        )
        secs_axon = [self.axon.get_sec(nm) for nm in self.boundary_spec.sec_names_first]
        secs_boundary = [self.boundary.secs[nm] for nm in self.boundary_spec.sec_names_second]

        self.boundary_coupler = LinearMechanismCoupler(
            secs_first=secs_axon,
            secs_second=secs_boundary,
            rg_dimless=self.boundary_spec.rg_dimless,
            rd_ohm_um2=rd_b,
            s_um=float(self.axon.fiber_diameter),
            nodeD_um=float(self.axon.mrg_params["nodeD"]),
            layer_index=2,
        ).build()

        # Keep HOC objects alive for entire run.
        self._hoc_keepalive = [self.boundary, self.boundary_coupler]

    def _extra_recording_segments(self):
        extra = []

        # Fixed node-index probes are handy for non-branching runs too.
        for node_index in PLOT["extra_node_indices"]:
            if not self.axon.main_axon:
                continue
            idx = int(node_index)
            if idx < 0:
                idx = len(self.axon.main_axon) + idx
            if 0 <= idx < len(self.axon.main_axon):
                extra.append((f"node_index_{node_index}", self.axon.main_axon[idx](0.5)))

        if PLOT["include_terminal_main"]:
            seg = self.axon.get_terminal_main_segment()
            if seg is not None:
                extra.append(("terminal_main", seg))

        return extra

    def run(self, freq_hz: float):
        self.axon.set_stim_target(
            mode=STIM["target_mode"],
            node_index=STIM["target_node_index"],
            x_um=STIM["target_x_um"],
        )
        self.axon.set_stimulation_params(
            mode="create",
            biphasic=bool(STIM["biphasic"]),
            freq_hz=float(freq_hz),
            amp=float(STIM["amp_nA"]),
            t_start=float(STIM["t_start_ms"]),
            t_end=float(STIM["t_end_ms"]),
            phase_us=float(STIM["phase_us"]),
            gap_us=float(STIM["gap_us"]),
        )
        self.axon.run_simulation(
            record_kinetics=bool(SIM["record_kinetics"]),
            include_stimulation_point=True,
            extra_named_segments=self._extra_recording_segments(),
        )


def detect_spike_count(t_ms: np.ndarray, v_mV: np.ndarray) -> int:
    valid = np.where(t_ms >= 8.0)[0]
    if valid.size == 0:
        return 0
    dt = float(t_ms[1] - t_ms[0])
    min_dist_pts = max(1, int(round(0.6 / dt)))
    peaks, _ = find_peaks(v_mV[valid[0]:], prominence=5.0, height=-20.0, distance=min_dist_pts)
    return int(len(peaks))


def plot_single_run(model: SingleAxonBoundaryModel, freq_hz: float, out_dir: Path) -> dict:
    axon = model.axon
    labels = list(getattr(axon, "recording_labels", []) or [])
    t_ms = np.asarray(axon.time_array, dtype=float)
    plot_mask = t_ms <= float(PLOT["window_ms"])

    selected_labels = [
        "stimulation_point",
        "before_branch",
        "after_branch_main",
        "after_branch_daughter",
        "terminal_main",
    ]
    for idx in PLOT["extra_node_indices"]:
        selected_labels.append(f"node_index_{idx}")

    plot_rows = []
    seen = set()
    for label in selected_labels:
        if label in seen:
            continue
        seen.add(label)
        if label in labels:
            plot_rows.append((label, labels.index(label)))

    if not plot_rows:
        raise RuntimeError("No recording labels selected for plotting")

    fig_h = max(3.0, float(PLOT["figure_height_per_panel"]) * len(plot_rows))
    fig, axes = plt.subplots(len(plot_rows), 1, figsize=(float(PLOT["figure_width"]), fig_h), dpi=180, sharex=True)
    if len(plot_rows) == 1:
        axes = [axes]

    fig.suptitle(
        f"Single axon + boundary | {int(freq_hz)} Hz | edge {BOUNDARY['edge_dist_um']} um | amp {STIM['amp_nA']} nA",
        fontsize=13,
    )

    summary = {"freq_hz": float(freq_hz)}
    for ax, (label, idx) in zip(axes, plot_rows):
        ax.plot(t_ms[plot_mask], axon.voltage_matrix[idx][plot_mask], lw=2.0)
        ax.set_title(label)
        ax.set_ylabel("Vm (mV)")
        ax.grid(True, alpha=0.25)
        summary[f"n_{label}"] = detect_spike_count(t_ms, axon.voltage_matrix[idx])
    axes[-1].set_xlabel("Time (ms)")

    fig.tight_layout(rect=(0, 0, 1, 0.97))
    plot_path = out_dir / f"single_boundary_{int(freq_hz):04d}Hz.png"
    fig.savefig(plot_path, bbox_inches="tight")
    plt.close(fig)
    summary["plot_path"] = str(plot_path)
    return summary


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    for freq_hz in FREQUENCIES_HZ:
        model = SingleAxonBoundaryModel()
        model.run(freq_hz=float(freq_hz))
        row = plot_single_run(model, float(freq_hz), OUT_DIR)
        print(row)
        rows.append(row)

    summary_path = OUT_DIR / "summary.csv"
    pd.DataFrame(rows).to_csv(summary_path, index=False)
    print(f"Saved summary: {summary_path}")


if __name__ == "__main__":
    main()
