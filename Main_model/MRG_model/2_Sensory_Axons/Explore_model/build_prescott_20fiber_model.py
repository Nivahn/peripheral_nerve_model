from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from prescott_multifiber import PrescottMultiFiberModel, load_prescott_geometry  # noqa: E402


DEFAULT_SOURCE_DIR = ROOT_DIR.parents[2] / "Prescott_ephaptic_coupling_MRG_model" / "radius = 4" / "20 fibers_same diameter_aligned" / "edge dist = 0.1"
DEFAULT_OUT_DIR = ROOT_DIR / "data" / "prescott_20fiber_python"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build Prescott-style 20-fiber geometry and Python branch morphology.")
    parser.add_argument("--source-dir", default=str(DEFAULT_SOURCE_DIR), help="Folder with Prescott geometry/coupling files.")
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR), help="Output folder for plots and summary JSON.")
    parser.add_argument("--parent-axon-nodes", type=int, default=27)
    parser.add_argument("--branch-nodes", type=int, default=8)
    parser.add_argument("--branches-num", type=int, default=1)
    parser.add_argument("--branch-sequence-nodes", default="8", help="Comma-separated branch node thresholds.")
    parser.add_argument("--main-scale", type=float, default=1.0)
    parser.add_argument("--daughter-scale", type=float, default=0.6)
    parser.add_argument("--h-stop-ms", type=float, default=10.0)
    parser.add_argument("--dt-ms", type=float, default=0.005)
    parser.add_argument("--skip-build", action="store_true", help="Only load geometry and draw plots, do not instantiate 20 NEURON axons.")
    parser.add_argument("--build-couplers", action="store_true", help="Build LinearMechanism couplers from Prescott pair maps.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source_dir = Path(args.source_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    branch_sequence_nodes = [int(x.strip()) for x in str(args.branch_sequence_nodes).split(",") if x.strip()]
    geometry = load_prescott_geometry(source_dir)
    model = PrescottMultiFiberModel(
        geometry,
        parent_axon_nodes=int(args.parent_axon_nodes),
        branch_nodes=int(args.branch_nodes),
        branches_num=int(args.branches_num),
        branch_sequence_nodes=branch_sequence_nodes,
        main_after_branch_diam_scale=float(args.main_scale),
        daughter_branch_diam_scale=float(args.daughter_scale),
        dt_ms=float(args.dt_ms),
        h_stop_ms=float(args.h_stop_ms),
    )

    if not args.skip_build:
        model.build_axons()
        pair_specs = model.build_pair_specs()
        if args.build_couplers:
            model.build_ephaptic_couplers()
    else:
        pair_specs = []

    pair_rows = [
        {
            "axon_i": spec.axon_i,
            "axon_j": spec.axon_j,
            "pair_key": spec.pair_key,
            "n_sections": len(spec.source_section_names),
            "n_rg": int(len(spec.rg_dimless)),
            "has_areas": int(spec.areas_um2 is not None),
        }
        for spec in pair_specs
    ]
    if pair_rows:
        pair_df = pd.DataFrame(pair_rows)
        pair_df.to_csv(out_dir / "pair_specs_summary.csv", index=False)

    outputs = {
        "packing": str(model.plot_packing(out_dir / "packing_numbered.png", show_numbers=True)),
        "neighbors": str(model.plot_neighbor_graph(out_dir / "neighbor_graph.png")),
        "boundary": str(model.plot_boundary_graph(out_dir / "boundary_graph.png")),
    }
    summary = model.summary()
    summary["source_dir"] = str(source_dir)
    summary["outputs"] = outputs
    summary["pair_specs_summary_csv"] = str(out_dir / "pair_specs_summary.csv") if pair_rows else ""
    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"Wrote {summary_path}")
    for value in outputs.values():
        print(f"Wrote {value}")


if __name__ == "__main__":
    main()
