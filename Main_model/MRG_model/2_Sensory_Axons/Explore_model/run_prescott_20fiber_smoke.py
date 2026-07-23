from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from prescott_multifiber import PrescottMultiFiberModel, generate_equal_diameter_geometry  # noqa: E402


DEFAULT_OUT_DIR = ROOT_DIR / "data" / "prescott_20fiber_smoke"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run first 20-fiber Prescott-style smoke simulation on Python-generated geometry.")
    parser.add_argument("--fiber-diameter-um", type=float, default=4.5)
    parser.add_argument("--edge-dist-um", type=float, default=1.0)
    parser.add_argument("--freq-hz", type=float, default=50.0)
    parser.add_argument("--amp-na", type=float, default=-1.0)
    parser.add_argument("--n-axons", type=int, default=20)
    parser.add_argument("--t-start-ms", type=float, default=10.0)
    parser.add_argument("--t-end-ms", type=float, default=60.0)
    parser.add_argument("--dt-ms", type=float, default=0.01)
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--stimulate-all", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    geometry = generate_equal_diameter_geometry(
        n_axons=int(args.n_axons),
        fiber_diameter_um=float(args.fiber_diameter_um),
        edge_dist_um=float(args.edge_dist_um),
    )
    model = PrescottMultiFiberModel(
        geometry,
        parent_axon_nodes=27,
        branch_nodes=8,
        branches_num=1,
        branch_sequence_nodes=[8],
        main_after_branch_diam_scale=1.0,
        daughter_branch_diam_scale=0.6,
        dt_ms=float(args.dt_ms),
        h_stop_ms=float(args.t_end_ms),
    )
    model.build_axons()
    model.build_pair_specs()
    model.build_ephaptic_couplers()
    model.build_boundary_couplers()

    packing = model.plot_packing(out_dir / "packing_numbered.png")
    neighbors = model.plot_neighbor_graph(out_dir / "neighbor_graph.png")
    boundary = model.plot_boundary_graph(out_dir / "boundary_graph.png")
    h5_path = model.run_smoke_simulation(
        h5_path=out_dir / "smoke_run.h5",
        freq_hz=float(args.freq_hz),
        amp_nA=float(args.amp_na),
        t_start_ms=float(args.t_start_ms),
        t_end_ms=float(args.t_end_ms),
        stimulate_all=bool(args.stimulate_all),
    )

    summary = model.summary()
    summary.update(
        {
            "mode": "aligned",
            "fiber_diameter_um_requested": float(args.fiber_diameter_um),
            "edge_dist_um_requested": float(args.edge_dist_um),
            "freq_hz": float(args.freq_hz),
            "amp_nA": float(args.amp_na),
            "stimulate_all": int(bool(args.stimulate_all)),
            "t_start_ms": float(args.t_start_ms),
            "t_end_ms": float(args.t_end_ms),
            "dt_ms": float(args.dt_ms),
            "outputs": {
                "packing": str(packing),
                "neighbors": str(neighbors),
                "boundary": str(boundary),
                "h5": str(h5_path),
            },
        }
    )
    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"Wrote {packing}")
    print(f"Wrote {neighbors}")
    print(f"Wrote {boundary}")
    print(f"Wrote {h5_path}")
    print(f"Wrote {summary_path}")


if __name__ == "__main__":
    main()
