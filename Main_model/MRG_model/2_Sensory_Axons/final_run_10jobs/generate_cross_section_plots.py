from __future__ import annotations

from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.image as mpimg  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from final_results_worker import MODE_CONFIGS, build_launch_config, build_model  # noqa: E402


DIAMETERS_UM = [2.5, 5.7]
EDGE_DISTANCES_UM = [0.1, 0.5, 1.0]
MODES = list(MODE_CONFIGS)
SCENARIO_NAME = "one_branch"
TOPOLOGY_NAME = "one_node_branching"
BOUNDARY_ROWS = [("boundary on", "aligned"), ("boundary off", "no_EC_isolated")]


def safe_tag(value: float | str) -> str:
    return str(value).replace(".", "p")


def main() -> None:
    out_dir = ROOT_DIR / "data" / "cross_section_boundary_plots"
    out_dir.mkdir(parents=True, exist_ok=True)

    outputs: list[Path] = []
    output_by_key: dict[tuple[float, float, str], Path] = {}
    for diameter_um in DIAMETERS_UM:
        cfg = build_launch_config(TOPOLOGY_NAME, diameter_um, SCENARIO_NAME, test_mode=True)
        for edge_dist_um in EDGE_DISTANCES_UM:
            for mode_name in MODES:
                model = build_model(cfg, edge_dist_um=edge_dist_um, mode_name=mode_name)
                png_path = out_dir / (
                    f"two_axons_fd{diameter_um}_ed{edge_dist_um}_{mode_name}_cross.png"
                )
                model.plot_cross_section_2d(save_path=str(png_path), show=False)
                outputs.append(png_path)
                output_by_key[(diameter_um, edge_dist_um, mode_name)] = png_path
                print(f"Saved {png_path}")

        fig, axes = plt.subplots(len(MODES), len(EDGE_DISTANCES_UM), figsize=(12.0, 16.0), dpi=180)
        for row_i, mode_name in enumerate(MODES):
            for col_i, edge_dist_um in enumerate(EDGE_DISTANCES_UM):
                ax = axes[row_i, col_i]
                img = mpimg.imread(output_by_key[(diameter_um, edge_dist_um, mode_name)])
                ax.imshow(img)
                ax.set_axis_off()
                if row_i == 0:
                    ax.set_title(f"edge={edge_dist_um} um", fontsize=11)
                if col_i == 0:
                    ax.text(-0.04, 0.5, mode_name, transform=ax.transAxes, rotation=90,
                            va="center", ha="right", fontsize=11)
        fig.suptitle(f"Cross-section boundary layout | fiber diameter {diameter_um} um", fontsize=14)
        fig.tight_layout(rect=(0, 0, 1, 0.98))
        overview_path = out_dir / f"overview_fd{diameter_um}_all_modes_all_distances_cross.png"
        fig.savefig(overview_path, bbox_inches="tight")
        plt.close(fig)
        outputs.append(overview_path)
        print(f"Saved {overview_path}")

        fig, axes = plt.subplots(len(BOUNDARY_ROWS), len(EDGE_DISTANCES_UM), figsize=(12.0, 7.0), dpi=180)
        for row_i, (row_label, mode_name) in enumerate(BOUNDARY_ROWS):
            for col_i, edge_dist_um in enumerate(EDGE_DISTANCES_UM):
                ax = axes[row_i, col_i]
                img = mpimg.imread(output_by_key[(diameter_um, edge_dist_um, mode_name)])
                ax.imshow(img)
                ax.set_axis_off()
                if row_i == 0:
                    ax.set_title(f"edge={edge_dist_um} um", fontsize=12)
                if col_i == 0:
                    ax.text(-0.04, 0.5, row_label, transform=ax.transAxes, rotation=90,
                            va="center", ha="right", fontsize=12)
        fig.suptitle(f"Boundary layout | fiber diameter {diameter_um} um", fontsize=14)
        fig.tight_layout(rect=(0, 0, 1, 0.97))
        boundary_overview_path = out_dir / f"overview_fd{diameter_um}_boundary_on_off_by_distance_cross.png"
        fig.savefig(boundary_overview_path, bbox_inches="tight")
        plt.close(fig)
        outputs.append(boundary_overview_path)
        print(f"Saved {boundary_overview_path}")

    print(f"Created {len(outputs)} cross-section plots in {out_dir}")


if __name__ == "__main__":
    main()
