from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import h5py
import numpy as np

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from prescott_multifiber import PrescottMultiFiberModel, generate_equal_diameter_geometry  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Multi-axon Prescott model: frequency sweep (1 or many frequencies) over one edge distance.")
    parser.add_argument("--fiber-diameter-um", type=float, default=4.5)
    parser.add_argument("--edge-dist-um", type=float, default=0.1)
    parser.add_argument("--n-axons", type=int, default=7)
    parser.add_argument("--amp-na", type=float, default=-3.0)
    parser.add_argument("--freq-start", type=int, default=50)
    parser.add_argument("--freq-end", type=int, default=1001)
    parser.add_argument("--freq-step", type=int, default=50)
    parser.add_argument("--freq-single", type=float, default=None, help="Run exactly one frequency (for SLURM job array).")
    parser.add_argument("--t-start-ms", type=float, default=10.0)
    parser.add_argument("--t-end-ms", type=float, default=1000.0)
    parser.add_argument("--dt-ms", type=float, default=0.05)
    parser.add_argument("--stimulate-all", action="store_true")
    parser.add_argument("--branch-center-only", action="store_true", help="Only the central axon branches; the rest are unbranched.")
    parser.add_argument("--no-ephaptic", action="store_true", help="Disable ephaptic coupling (isolated control, no LinearMechanism).")
    parser.add_argument("--misalignment-fraction", type=float, default=0.0, help="Longitudinal offset as a fraction of the internode step (e.g. 0.5).")
    parser.add_argument("--skip-existing", action="store_true", help="Skip a frequency if its HDF5 already exists (resume support).")
    parser.add_argument("--out-dir", default=str(ROOT_DIR / "data" / "prescott_7axon_sweep"))
    return parser.parse_args()


def run_single_frequency(
    model: PrescottMultiFiberModel,
    freq_hz: float,
    amp_nA: float,
    t_start_ms: float,
    t_end_ms: float,
    stimulate_all: bool,
    out_dir: Path,
    tag: str,
) -> dict:
    h5_name = f"freq_{int(freq_hz):04d}hz_{tag}.h5"
    h5_path = out_dir / h5_name
    t0 = time.time()
    model.run_smoke_simulation(
        h5_path=h5_path,
        freq_hz=freq_hz,
        amp_nA=amp_nA,
        t_start_ms=t_start_ms,
        t_end_ms=t_end_ms,
        stimulate_all=stimulate_all,
    )
    elapsed = time.time() - t0
    print(f"  freq={freq_hz:.0f} Hz  -> {h5_name}  ({elapsed:.1f} s)")
    return {"freq_hz": freq_hz, "h5": str(h5_path), "elapsed_s": elapsed}


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    edge_tag = f"ed{args.edge_dist_um}".replace(".", "p")
    sweep_tag = f"fd{args.fiber_diameter_um}_{edge_tag}".replace(".", "p")
    sweep_dir = out_dir / sweep_tag
    sweep_dir.mkdir(parents=True, exist_ok=True)

    if args.freq_single is not None:
        frequencies = [float(args.freq_single)]
    else:
        frequencies = list(range(args.freq_start, args.freq_end, args.freq_step))
    print(f"=== Multi-axon Prescott sweep ===")
    print(f"  fiber_diameter = {args.fiber_diameter_um} um")
    print(f"  edge_distance  = {args.edge_dist_um} um")
    print(f"  n_axons        = {args.n_axons}")
    print(f"  amp            = {args.amp_na} nA")
    print(f"  frequencies    = {frequencies[0]}..{frequencies[-1]} Hz ({len(frequencies)} points)")
    print(f"  dt             = {args.dt_ms} ms")
    print(f"  t_end          = {args.t_end_ms} ms")
    print(f"  stimulate_all  = {args.stimulate_all}")
    print(f"  branch_center_only = {args.branch_center_only}")
    print(f"  enable_ephaptic    = {not args.no_ephaptic}")
    print(f"  misalignment_frac  = {args.misalignment_fraction}")
    print(f"  skip_existing  = {args.skip_existing}")
    print(f"  output         = {sweep_dir}")
    print()

    geometry = generate_equal_diameter_geometry(
        n_axons=int(args.n_axons),
        fiber_diameter_um=float(args.fiber_diameter_um),
        edge_dist_um=float(args.edge_dist_um),
    )

    model = PrescottMultiFiberModel(
        geometry,
        parent_axon_nodes=36,
        branch_nodes=8,
        branches_num=1,
        branch_sequence_nodes=[8],
        branch_center_only=bool(args.branch_center_only),
        enable_ephaptic=not bool(args.no_ephaptic),
        misalignment_fraction=float(args.misalignment_fraction),
        main_after_branch_diam_scale=1.0,
        daughter_branch_diam_scale=0.6,
        dt_ms=float(args.dt_ms),
        h_stop_ms=float(args.t_end_ms),
    )

    print("Building axons...")
    t0 = time.time()
    model.build_axons()
    print(f"  {time.time() - t0:.1f} s")

    print("Building pair specs...")
    t0 = time.time()
    model.build_pair_specs()
    print(f"  {time.time() - t0:.1f} s")

    print("Building ephaptic couplers...")
    t0 = time.time()
    model.build_ephaptic_couplers()
    print(f"  {time.time() - t0:.1f} s")

    print("Building boundary couplers...")
    t0 = time.time()
    model.build_boundary_couplers()
    print(f"  {time.time() - t0:.1f} s")

    # В array-режиме (одна частота на задачу) графики и summary.json пишем только
    # один раз, чтобы параллельные задачи не перезаписывали друг друга.
    single_mode = args.freq_single is not None

    if not single_mode or not (sweep_dir / "packing_numbered.png").exists():
        packing = model.plot_packing(sweep_dir / "packing_numbered.png")
        neighbors = model.plot_neighbor_graph(sweep_dir / "neighbor_graph.png")
        boundary = model.plot_boundary_graph(sweep_dir / "boundary_graph.png")
        print(f"Plots: {packing}, {neighbors}, {boundary}")
    else:
        print("Plots: already exist, skip")

    summary = model.summary()
    summary.update({
        "fiber_diameter_um": float(args.fiber_diameter_um),
        "edge_dist_um": float(args.edge_dist_um),
        "n_axons": int(args.n_axons),
        "amp_nA": float(args.amp_na),
        "frequencies_hz": frequencies,
        "dt_ms": float(args.dt_ms),
        "t_start_ms": float(args.t_start_ms),
        "t_end_ms": float(args.t_end_ms),
        "stimulate_all": int(bool(args.stimulate_all)),
        "branch_center_only": int(bool(args.branch_center_only)),
        "enable_ephaptic": int(not bool(args.no_ephaptic)),
        "misalignment_fraction": float(args.misalignment_fraction),
    })

    print()
    print("Running frequency sweep...")
    results = []
    for freq in frequencies:
        h5_name = f"freq_{int(freq):04d}hz_{edge_tag}.h5"
        if args.skip_existing and (sweep_dir / h5_name).exists():
            print(f"  skip (exists): {h5_name}")
            results.append({"freq_hz": freq, "h5": str(sweep_dir / h5_name), "elapsed_s": None, "skipped": True})
            continue
        res = run_single_frequency(
            model=model,
            freq_hz=float(freq),
            amp_nA=float(args.amp_na),
            t_start_ms=float(args.t_start_ms),
            t_end_ms=float(args.t_end_ms),
            stimulate_all=bool(args.stimulate_all),
            out_dir=sweep_dir,
            tag=edge_tag,
        )
        results.append(res)

    summary["results"] = results
    if single_mode:
        summary_path = sweep_dir / f"summary_freq_{int(args.freq_single):04d}hz.json"
    else:
        summary_path = sweep_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nDone. Summary: {summary_path}")


if __name__ == "__main__":
    main()
