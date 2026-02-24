from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np


@dataclass
class PrescottConfig:
    radii_um: np.ndarray
    centers_xy_um: np.ndarray
    groups: np.ndarray
    total_axon_length_um: float = 31500.0
    neighbor_threshold_um: float = 1000.0
    boundary_neighbor_threshold_um: float = 8.0
    boundary_distance_to_neuron_um: float = 7.0
    neurons_on_edge_up: Tuple[int, ...] = (0, 1)
    paralength2_segments: int = 5
    internode_segments: int = 40


@dataclass
class PrescottOutputs:
    R: np.ndarray
    C: np.ndarray
    G: np.ndarray
    parameters: np.ndarray
    number_of_nodes: np.ndarray
    unique_radius: np.ndarray
    neighboring_axon: np.ndarray
    dist: np.ndarray
    distance_edge: np.ndarray
    boundary_coordinates: np.ndarray
    boundary_neighboring: np.ndarray
    boundary_dist: np.ndarray
    rg_11: np.ndarray
    boundary_rg_1: np.ndarray


class PrescottPythonRewriter:
    """Python rewrite of the MATLAB `Main_Code.m` preprocessing for Prescott model.

    This class reproduces the key text outputs used by downstream ephaptic scripts.
    """

    def __init__(self, config: PrescottConfig):
        self.cfg = config

    @staticmethod
    def _pairwise_dist(points: np.ndarray) -> np.ndarray:
        diff = points[:, None, :] - points[None, :, :]
        return np.sqrt(np.sum(diff * diff, axis=2))

    def _morph_parameters(self, unique_radius: np.ndarray) -> np.ndarray:
        parameters = np.zeros((len(unique_radius), 6), dtype=float)
        for i, radius in enumerate(unique_radius):
            x = radius * 2.0

            axon_d = 0.01876 * x**2 + 0.4787 * x + 0.1204
            node_d = 0.006304 * x**2 + 0.2071 * x + 0.5339
            deltax = -8.215 * x**2 + 272.4 * x - 780.2
            paralength2 = -0.0199 * x**2 + 3.016 * x + 17.44
            nl = np.round(-0.389 * x**2 + 14.88 * x + 9.721)

            paralength1 = 3.0
            nodelength = 1.0
            interlength = deltax - nodelength - (2 * paralength1) - (2 * paralength2)

            parameters[i, 0] = axon_d
            parameters[i, 1] = node_d
            parameters[i, 2] = deltax
            parameters[i, 3] = paralength2 / self.cfg.paralength2_segments
            parameters[i, 4] = nl
            parameters[i, 5] = interlength / self.cfg.internode_segments

        return parameters

    def _build_section_layout(
        self, parameters: np.ndarray, unique_radius: np.ndarray
    ) -> Tuple[Dict[int, np.ndarray], Dict[int, np.ndarray], Dict[int, List[str]], np.ndarray]:
        nodelength = 1.0
        paralength1 = 3.0

        coords = {}
        mids = {}
        names = {}
        number_of_nodes = np.zeros((len(unique_radius), 1), dtype=float)

        for j in range(len(unique_radius)):
            z1 = z2 = z3 = z4 = 0
            boundaries = [0.0]
            midpoints: List[float] = []
            sec_names: List[str] = []

            def add_section(length: float, sec_name: str) -> None:
                nonlocal boundaries, midpoints, sec_names
                start = boundaries[-1]
                end = start + length
                boundaries.append(end)
                midpoints.append((start + end) / 2.0)
                sec_names.append(sec_name)

            # First set
            add_section(nodelength, f"node_{z1}")
            z1 += 1
            add_section(paralength1, f"MYSA_{z2}")
            z2 += 1

            for _ in range(self.cfg.paralength2_segments):
                add_section(parameters[j, 3], f"FLUT_{z3}")
                z3 += 1

            for _ in range(self.cfg.internode_segments):
                add_section(parameters[j, 5], f"STIN_{z4}")
                z4 += 1

            for _ in range(self.cfg.paralength2_segments):
                add_section(parameters[j, 3], f"FLUT_{z3}")
                z3 += 1

            add_section(paralength1, f"MYSA_{z2}")
            z2 += 1

            length_one_set = boundaries[-1]
            number_sets = int(np.round(self.cfg.total_axon_length_um / length_one_set))

            # Remaining sets
            for _ in range(2, number_sets + 1):
                add_section(nodelength, f"node_{z1}")
                z1 += 1
                add_section(paralength1, f"MYSA_{z2}")
                z2 += 1

                for _ in range(self.cfg.paralength2_segments):
                    add_section(parameters[j, 3], f"FLUT_{z3}")
                    z3 += 1

                for _ in range(self.cfg.internode_segments):
                    add_section(parameters[j, 5], f"STIN_{z4}")
                    z4 += 1

                for _ in range(self.cfg.paralength2_segments):
                    add_section(parameters[j, 3], f"FLUT_{z3}")
                    z3 += 1

                add_section(paralength1, f"MYSA_{z2}")
                z2 += 1

            coords[j + 1] = np.array(boundaries, dtype=float)
            mids[j + 1] = np.array(midpoints, dtype=float)
            names[j + 1] = sec_names
            number_of_nodes[j, 0] = z1

        return coords, mids, names, number_of_nodes

    @staticmethod
    def _connect_types_self(number_of_nodes: int) -> List[str]:
        node_names = [f"node_{n}" for n in range(number_of_nodes)]
        return node_names + node_names

    def run(self) -> PrescottOutputs:
        R = self.cfg.radii_um.reshape(-1, 1).astype(float)
        C = self.cfg.centers_xy_um.astype(float)
        G = self.cfg.groups.reshape(1, -1).astype(float)

        unique_radius = np.unique(R)
        parameters = self._morph_parameters(unique_radius)

        distance_centers = self._pairwise_dist(C)
        distance_edge = np.zeros_like(distance_centers)
        for i in range(distance_centers.shape[0]):
            for j in range(distance_centers.shape[1]):
                if i != j:
                    distance_edge[i, j] = distance_centers[i, j] - (R[i, 0] + R[j, 0])

        N = np.zeros_like(distance_centers)
        dist = np.zeros_like(distance_centers)
        m = self.cfg.neighbor_threshold_um
        for k in range(distance_centers.shape[0]):
            for i in range(distance_centers.shape[1]):
                if k != i and distance_centers[k, i] < m:
                    N[k, i] = 1
                    dist[k, i] = distance_centers[k, i]

        for j in range(N.shape[0]):
            for i in range(N.shape[1]):
                if N[j, i] == 1:
                    N[i, j] = 0
                    dist[i, j] = 0

        coords, mids, names, number_of_nodes = self._build_section_layout(parameters, unique_radius)

        compartments_in_set = 3 + self.cfg.internode_segments + (2 * self.cfg.paralength2_segments)

        # Boundary coordinates (same convention as MATLAB script)
        boundary_coords = np.zeros((len(self.cfg.neurons_on_edge_up), 2), dtype=float)
        for i, idx in enumerate(self.cfg.neurons_on_edge_up):
            boundary_coords[i, 0] = C[idx, 0]
            boundary_coords[i, 1] = C[idx, 1] + self.cfg.boundary_distance_to_neuron_um

        c_and_b = np.vstack([C, boundary_coords])
        boundary_distance = self._pairwise_dist(c_and_b)
        boundary_N = np.zeros_like(boundary_distance)
        boundary_dist = np.zeros_like(boundary_distance)

        for k in range(boundary_distance.shape[0]):
            for i in range(boundary_distance.shape[1]):
                if k != i and boundary_distance[k, i] < self.cfg.boundary_neighbor_threshold_um:
                    boundary_N[k, i] = 1
                    boundary_dist[k, i] = boundary_distance[k, i]

        size_c = C.shape[0]
        boundary_neighboring = boundary_N[size_c:, :size_c]
        boundary_dist_out = boundary_dist[size_c:, :size_c]

        # Build boundary sections and mapping Boundary_to_1
        number_of_sections = 4000
        sectionlength = self.cfg.total_axon_length_um / number_of_sections
        boundary_section_edges = np.linspace(0.0, self.cfg.total_axon_length_um, number_of_sections + 1)

        # Connect_types_11 only for this canonical same-radius case
        connect_types_11 = self._connect_types_self(int(number_of_nodes[0, 0]))

        # Node section indices in the serialized layout
        node_midpoints = mids[1][0 : compartments_in_set * int(number_of_nodes[0, 0]) : compartments_in_set]

        boundary_to_1_first = []
        boundary_to_1_second = []
        for node_idx, f1 in enumerate(node_midpoints):
            m_idx = np.searchsorted(boundary_section_edges, f1, side="right") - 1
            m_idx = min(max(m_idx, 0), number_of_sections - 1)
            boundary_to_1_first.append(f"node_{node_idx}")
            boundary_to_1_second.append(f"section_{m_idx}")

        # Rg_11
        def section_mid(name: str) -> float:
            nlist = names[1]
            try:
                idx = nlist.index(name)
            except ValueError as exc:
                raise KeyError(f"Unknown section {name}") from exc
            return mids[1][idx]

        L = len(connect_types_11) // 2
        rg_11 = np.zeros((L, 1), dtype=float)
        for z in range(1, L - 1):
            g1 = section_mid(connect_types_11[z - 1])
            g2 = section_mid(connect_types_11[z + 1])
            rg_11[z, 0] = (g2 - g1) / 2.0

        rg_11[0, 0] = (section_mid(connect_types_11[1]) - section_mid(connect_types_11[0])) / 2.0
        rg_11[L - 1, 0] = (section_mid(connect_types_11[L - 1]) - section_mid(connect_types_11[L - 2])) / 2.0

        # Boundary_Rg_1 based on Boundary_to_1 mapping
        boundary_to_1 = boundary_to_1_first + boundary_to_1_second
        Lb = len(boundary_to_1) // 2
        boundary_rg_1 = np.zeros((Lb, 1), dtype=float)
        for z in range(1, Lb - 1):
            g1 = section_mid(boundary_to_1[z - 1])
            g2 = section_mid(boundary_to_1[z + 1])
            boundary_rg_1[z, 0] = (g2 - g1) / 2.0

        boundary_rg_1[0, 0] = (section_mid(boundary_to_1[1]) - section_mid(boundary_to_1[0])) / 2.0
        boundary_rg_1[Lb - 1, 0] = (section_mid(boundary_to_1[Lb - 1]) - section_mid(boundary_to_1[Lb - 2])) / 2.0

        return PrescottOutputs(
            R=R,
            C=C,
            G=G,
            parameters=parameters,
            number_of_nodes=number_of_nodes,
            unique_radius=unique_radius.reshape(-1, 1),
            neighboring_axon=N,
            dist=dist,
            distance_edge=distance_edge,
            boundary_coordinates=boundary_coords,
            boundary_neighboring=boundary_neighboring,
            boundary_dist=boundary_dist_out,
            rg_11=rg_11,
            boundary_rg_1=boundary_rg_1,
        )


def canonical_edge01_config() -> PrescottConfig:
    return PrescottConfig(
        radii_um=np.array([4.0, 4.0]),
        centers_xy_um=np.array([[0.0, 0.0], [8.1, 0.0]]),
        groups=np.array([4.0, 4.0]),
    )


def save_outputs(outputs: PrescottOutputs, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    np.savetxt(out_dir / "R.txt", outputs.R, fmt="%.15g")
    np.savetxt(out_dir / "C.txt", outputs.C, fmt="%.15g")
    np.savetxt(out_dir / "G.txt", outputs.G, fmt="%.15g")
    np.savetxt(out_dir / "parameters.txt", outputs.parameters, fmt="%.15g")
    np.savetxt(out_dir / "Number_of_nodes.txt", outputs.number_of_nodes, fmt="%.15g")
    np.savetxt(out_dir / "unique_radius.txt", outputs.unique_radius, fmt="%.15g")
    np.savetxt(out_dir / "neighboringAxon.txt", outputs.neighboring_axon, fmt="%.15g")
    np.savetxt(out_dir / "dist.txt", outputs.dist, fmt="%.15g")
    np.savetxt(out_dir / "Distance_edge.txt", outputs.distance_edge, fmt="%.15g")
    np.savetxt(out_dir / "Boundary_coordinates.txt", outputs.boundary_coordinates, fmt="%.15g")
    np.savetxt(out_dir / "Boundary_neighboring.txt", outputs.boundary_neighboring, fmt="%.15g")
    np.savetxt(out_dir / "Boundary_dist.txt", outputs.boundary_dist, fmt="%.15g")
    np.savetxt(out_dir / "Rg_11.txt", outputs.rg_11, fmt="%.15g")
    np.savetxt(out_dir / "Boundary_Rg_1.txt", outputs.boundary_rg_1, fmt="%.15g")


if __name__ == "__main__":
    target = Path(
        "Prescott_ephaptic_coupling_MRG_model/radius = 4/2 fibers_same diameter_aligned/edge dist = 0.1"
    )
    cfg = canonical_edge01_config()
    model = PrescottPythonRewriter(cfg)
    out = model.run()
    save_outputs(out, target / "python_rewrite_output")
    print(f"Saved rewritten Prescott outputs to: {target / 'python_rewrite_output'}")
