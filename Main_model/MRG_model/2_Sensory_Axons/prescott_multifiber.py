from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import re

import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import numpy as np
from scipy.io import loadmat
from scipy.spatial import ConvexHull
import h5py

from MRG_lib import LinearMechanismCoupler, MRGaxon


class PrescottFullMRGaxon(MRGaxon):
    FULL_FLUT_PER_SIDE = 5
    FULL_STIN_COUNT = 40

    def _get_mrg_params(self, fiberD):
        x = float(fiberD)
        axonD = 0.01876 * x * x + 0.4787 * x + 0.1204
        nodeD = 0.006304 * x * x + 0.2071 * x + 0.5339
        deltax = -8.215 * x * x + 272.4 * x - 780.2
        paralength2_total = -0.0199 * x * x + 3.016 * x + 17.44
        nl = int(round(-0.389 * x * x + 14.88 * x + 9.721))
        paralength1 = float(self.paralength1)
        nodelength = float(self.nodelength)
        flut_segment_length = float(paralength2_total) / float(self.FULL_FLUT_PER_SIDE)
        total_internode_len = float(deltax) - nodelength - 2.0 * paralength1 - 2.0 * float(paralength2_total)

        if total_internode_len <= 0 or deltax <= 0:
            raise ValueError(
                f"Prescott MRG formulas invalid for fiberD={x:.2f} um: "
                f"deltax={deltax:.3f}, total_internode_len={total_internode_len:.3f}. "
                f"These formulas are valid for fiberD >= ~5.7 um. "
                f"Use daughter_branch_param_mode='scaled_radial' for smaller diameters."
            )

        stin_segment_length = float(total_internode_len) / float(self.FULL_STIN_COUNT)
        return {
            'fiberD': float(x),
            'axonD': float(axonD),
            'nodeD': float(nodeD),
            'paraD1': float(nodeD),
            'paraD2': float(axonD),
            'paral1': float(paralength1),
            'paral2': float(flut_segment_length),
            'interL': float(stin_segment_length),
            'nl': int(nl),
            'rpn0': self._rin_peri(float(nodeD), self.space_p1),
            'rpn1': self._rin_peri(float(nodeD), self.space_p1),
            'rpn2': self._rin_peri(float(axonD), self.space_p2),
            'rpx': self._rin_peri(float(axonD), self.space_i),
            'Lstep': float(deltax),
        }

    def append_one_step(self, parent_node, params, track_trunk: bool = False):
        mysa0 = self.make_mysa(params['fiberD'], params['paraD1'], params['paral1'], params['nl'], params['rpn1'])
        flut_left = [self.make_flut(params['fiberD'], params['paraD2'], params['paral2'], params['nl'], params['rpn2']) for _ in range(self.FULL_FLUT_PER_SIDE)]
        stin_sections = [self.make_stin(params['fiberD'], params['axonD'], params['interL'], params['nl'], params['rpx']) for _ in range(self.FULL_STIN_COUNT)]
        flut_right = [self.make_flut(params['fiberD'], params['paraD2'], params['paral2'], params['nl'], params['rpn2']) for _ in range(self.FULL_FLUT_PER_SIDE)]
        mysa1 = self.make_mysa(params['fiberD'], params['paraD1'], params['paral1'], params['nl'], params['rpn1'])
        next_node = self.make_node(params['nodeD'], self.nodelength, params['rpn0'])

        mysa0.connect(parent_node, 1.0, 0.0)
        prev = mysa0
        for sec in flut_left:
            sec.connect(prev, 1.0, 0.0)
            prev = sec
        for sec in stin_sections:
            sec.connect(prev, 1.0, 0.0)
            prev = sec
        for sec in flut_right:
            sec.connect(prev, 1.0, 0.0)
            prev = sec
        mysa1.connect(prev, 1.0, 0.0)
        next_node.connect(mysa1, 1.0, 0.0)

        if track_trunk:
            node_center = float(getattr(self, "_trunk_last_node_center_um", 0.0))
            mid_target = node_center + 0.5 * float(params.get('Lstep', 0.0))
            node_half = float(self.nodelength) / 2.0
            mysa_L = float(params['paral1'])
            flut_L = float(params['paral2'])
            stin_L = float(params['interL'])
            start_x = node_center + node_half + mysa_L + self.FULL_FLUT_PER_SIDE * flut_L
            cand = [start_x + (k + 0.5) * stin_L for k in range(len(stin_sections))]
            mid_idx = int(np.argmin(np.abs(np.asarray(cand, dtype=float) - float(mid_target))))
            mid = stin_sections[mid_idx]
            self._trunk_stin_mid_by_next_node[next_node.name()] = mid
            self._trunk_stin_mid_idx_by_next_node[next_node.name()] = int(mid_idx)
            self.trunk_center_um[mid.name()] = node_center + 0.5 * float(params.get('Lstep', 0.0))

        try:
            self._step_records.append({
                "parent": parent_node.name(),
                "next": next_node.name(),
                "is_trunk": bool(track_trunk),
                "sections": [
                    {"type": "mysa", "name": mysa0.name(), "L": float(mysa0.L)},
                    *[{"type": "flut", "name": sec.name(), "L": float(sec.L)} for sec in flut_left],
                    *[{"type": "stin", "name": sec.name(), "L": float(sec.L)} for sec in stin_sections],
                    *[{"type": "flut", "name": sec.name(), "L": float(sec.L)} for sec in flut_right],
                    {"type": "mysa", "name": mysa1.name(), "L": float(mysa1.L)},
                ],
            })
        except Exception:
            pass
        return next_node

    def append_terminal_tail(self, parent_node, params):
        mysa0 = self.make_mysa(params['fiberD'], params['paraD1'], params['paral1'], params['nl'], params['rpn1'])
        flut_left = [self.make_flut(params['fiberD'], params['paraD2'], params['paral2'], params['nl'], params['rpn2']) for _ in range(self.FULL_FLUT_PER_SIDE)]
        stin_sections = [self.make_stin(params['fiberD'], params['axonD'], params['interL'], params['nl'], params['rpx']) for _ in range(self.FULL_STIN_COUNT)]
        flut_right = [self.make_flut(params['fiberD'], params['paraD2'], params['paral2'], params['nl'], params['rpn2']) for _ in range(self.FULL_FLUT_PER_SIDE)]
        mysa1 = self.make_mysa(params['fiberD'], params['paraD1'], params['paral1'], params['nl'], params['rpn1'])

        mysa0.connect(parent_node, 1.0, 0.0)
        prev = mysa0
        for sec in flut_left:
            sec.connect(prev, 1.0, 0.0)
            prev = sec
        for sec in stin_sections:
            sec.connect(prev, 1.0, 0.0)
            prev = sec
        for sec in flut_right:
            sec.connect(prev, 1.0, 0.0)
            prev = sec
        mysa1.connect(prev, 1.0, 0.0)

        try:
            self._step_records.append({
                "parent": parent_node.name(),
                "next": None,
                "is_trunk": False,
                "is_terminal_tail": True,
                "sections": [
                    {"type": "mysa", "name": mysa0.name(), "L": float(mysa0.L)},
                    *[{"type": "flut", "name": sec.name(), "L": float(sec.L)} for sec in flut_left],
                    *[{"type": "stin", "name": sec.name(), "L": float(sec.L)} for sec in stin_sections],
                    *[{"type": "flut", "name": sec.name(), "L": float(sec.L)} for sec in flut_right],
                    {"type": "mysa", "name": mysa1.name(), "L": float(mysa1.L)},
                ],
            })
        except Exception:
            pass


def _load_ascii_array(path: Path) -> np.ndarray:
    data = np.loadtxt(str(path), dtype=float)
    return np.atleast_2d(data)


def _parse_connect_type_mat(path: Path) -> tuple[list[str], list[str]]:
    mat = loadmat(str(path))
    raw = mat.get("SAVE")
    if raw is None:
        raise KeyError(f"SAVE not found in {path}")
    flat = []
    for item in np.asarray(raw).ravel():
        text = str(item).strip()
        if text:
            flat.append(text)
    half = len(flat) // 2
    return flat[:half], flat[half:]


def _path_stem_key(path: Path, prefix: str) -> str:
    name = path.stem
    if not name.startswith(prefix):
        return name
    return name[len(prefix):]


@dataclass
class PrescottCouplingMaps:
    neighboring_axon: np.ndarray
    center_distance_um: np.ndarray
    edge_distance_um: np.ndarray
    boundary_neighboring: np.ndarray
    boundary_distance_um: np.ndarray
    boundary_coordinates_um: np.ndarray
    connect_types: dict[str, tuple[list[str], list[str]]]
    rg_by_pair: dict[str, np.ndarray]
    areas_by_pair: dict[str, np.ndarray]


@dataclass
class PrescottPairSpec:
    axon_i: int
    axon_j: int
    pair_key: str
    source_tokens: list[str]
    target_tokens: list[str]
    source_section_names: list[str]
    target_section_names: list[str]
    rg_dimless: np.ndarray
    areas_um2: np.ndarray | None


class GroundSinkArray:
    def __init__(self, name_prefix: str, n_sections: int):
        self.name_prefix = str(name_prefix)
        self.secs = []
        for idx in range(int(n_sections)):
            sec = MRGaxon.make_branch_connector  # type: ignore[attr-defined]
            sec = None
            s = None
            from MRG_lib import h  # local import to reuse loaded NEURON handle
            s = h.Section(name=f"{self.name_prefix}sink_{idx}")
            s.nseg = 1
            s.L = 0.01
            s.diam = 0.01
            s.Ra = 1e9
            s.cm = 1e-9
            if int(h.ismembrane('extracellular', sec=s)) == 0:
                s.insert('extracellular')
            for seg in s:
                seg.xraxial[0] = 1e9
                seg.xraxial[1] = 1e9
                seg.xg[0] = 1e9
                seg.xg[1] = 1e9
                seg.xc[0] = 0.0
                seg.xc[1] = 0.0
            self.secs.append(s)


def _compute_rg_dimless_from_centers(centers_um: np.ndarray, s_um: float) -> np.ndarray:
    x = np.asarray(centers_um, dtype=float)
    if x.size < 2:
        return np.asarray([1.0], dtype=float)
    rg_um = np.zeros_like(x)
    rg_um[0] = x[1] - x[0]
    rg_um[-1] = x[-1] - x[-2]
    if x.size > 2:
        rg_um[1:-1] = 0.5 * (x[2:] - x[:-2])
    rg_um = np.maximum(rg_um, 1e-9)
    return rg_um / float(s_um)


def generate_equal_diameter_geometry(
    *,
    n_axons: int,
    fiber_diameter_um: float,
    edge_dist_um: float,
    boundary_offset_um: float = 7.0,
) -> PrescottGeometry:
    radius_um = 0.5 * float(fiber_diameter_um)
    spacing = float(fiber_diameter_um) + float(edge_dist_um)

    points = []
    grid_limit = 6
    for q in range(-grid_limit, grid_limit + 1):
        for r in range(-grid_limit, grid_limit + 1):
            x = spacing * (q + 0.5 * r)
            y = spacing * (np.sqrt(3.0) / 2.0) * r
            dist = np.hypot(x, y)
            points.append((dist, x, y, q, r))
    points.sort(key=lambda item: (item[0], np.arctan2(item[2], item[1])))
    centers = np.asarray([[x, y] for _, x, y, _, _ in points[: int(n_axons)]], dtype=float)
    radii = np.full(int(n_axons), radius_um, dtype=float)
    groups = np.full(int(n_axons), 4.0, dtype=float)
    unique_radius = np.asarray([radius_um], dtype=float)
    full_ref = PrescottFullMRGaxon(fiber_diameter=float(fiber_diameter_um), parent_axon_nodes=36, branch_nodes=8, branches_num=1, branch_sequence_nodes=[8], main_after_branch_diam_scale=1.0, daughter_branch_diam_scale=0.6, reset_nrn=True, h_stop=1.0, dt_ms=0.01)
    params = np.asarray([[full_ref.mrg_params['axonD'], full_ref.mrg_params['nodeD'], full_ref.mrg_params['Lstep'], full_ref.mrg_params['paral2'] * PrescottFullMRGaxon.FULL_FLUT_PER_SIDE, full_ref.mrg_params['nl'], full_ref.mrg_params['interL']]], dtype=float)
    number_of_nodes = np.asarray([36.0], dtype=float)

    center_dist = np.linalg.norm(centers[:, None, :] - centers[None, :, :], axis=2)
    edge_dist = center_dist.copy()
    for i in range(int(n_axons)):
        for j in range(int(n_axons)):
            if i == j:
                edge_dist[i, j] = 0.0
            else:
                edge_dist[i, j] = center_dist[i, j] - 2.0 * radius_um
    neighboring = np.triu(np.ones((int(n_axons), int(n_axons)), dtype=float), k=1)

    hull = ConvexHull(centers)
    hull_indices = list(map(int, hull.vertices.tolist()))
    centroid = centers.mean(axis=0)
    boundary_points = []
    for idx in hull_indices:
        vec = centers[idx] - centroid
        norm = np.linalg.norm(vec)
        direction = np.asarray([1.0, 0.0]) if norm <= 1e-12 else vec / norm
        boundary_points.append(centers[idx] + direction * float(boundary_offset_um))
    boundary_coords = np.asarray(boundary_points, dtype=float)
    boundary_neighboring = np.zeros((len(hull_indices), int(n_axons)), dtype=float)
    boundary_dist = np.zeros((len(hull_indices), int(n_axons)), dtype=float)
    for row, ax_idx in enumerate(hull_indices):
        boundary_neighboring[row, ax_idx] = 1.0
        boundary_dist[row, ax_idx] = float(boundary_offset_um)

    return PrescottGeometry(
        source_dir=Path("<generated_python_geometry>"),
        centers_um=centers,
        radii_um=radii,
        groups=groups,
        unique_radius_um=unique_radius,
        parameters=params,
        number_of_nodes=number_of_nodes,
        coupling=PrescottCouplingMaps(
            neighboring_axon=neighboring,
            center_distance_um=center_dist,
            edge_distance_um=edge_dist,
            boundary_neighboring=boundary_neighboring,
            boundary_distance_um=boundary_dist,
            boundary_coordinates_um=boundary_coords,
            connect_types={},
            rg_by_pair={},
            areas_by_pair={},
        ),
    )


@dataclass
class PrescottGeometry:
    source_dir: Path
    centers_um: np.ndarray
    radii_um: np.ndarray
    groups: np.ndarray
    unique_radius_um: np.ndarray
    parameters: np.ndarray
    number_of_nodes: np.ndarray
    coupling: PrescottCouplingMaps

    @property
    def fiber_diameter_um(self) -> float:
        return float(np.median(self.radii_um) * 2.0)

    @property
    def n_axons(self) -> int:
        return int(self.centers_um.shape[0])


def load_prescott_geometry(source_dir: str | Path) -> PrescottGeometry:
    source_dir = Path(source_dir)
    centers = _load_ascii_array(source_dir / "C.txt")
    radii = np.asarray(_load_ascii_array(source_dir / "R.txt")).reshape(-1)
    groups = np.asarray(_load_ascii_array(source_dir / "G.txt")).reshape(-1)
    unique_radius = np.asarray(_load_ascii_array(source_dir / "unique_radius.txt")).reshape(-1)
    parameters = _load_ascii_array(source_dir / "parameters.txt")
    number_of_nodes = np.asarray(_load_ascii_array(source_dir / "Number_of_nodes.txt")).reshape(-1)

    neighboring_axon = _load_ascii_array(source_dir / "neighboringAxon.txt")
    center_distance_um = _load_ascii_array(source_dir / "dist.txt")
    edge_distance_um = _load_ascii_array(source_dir / "Distance_edge.txt")
    boundary_neighboring = _load_ascii_array(source_dir / "Boundary_neighboring.txt")
    boundary_distance_um = _load_ascii_array(source_dir / "Boundary_dist.txt")
    boundary_coordinates_um = _load_ascii_array(source_dir / "Boundary_coordinates.txt")

    connect_types = {}
    for path in sorted(source_dir.glob("Connect_types_*.mat")):
        connect_types[_path_stem_key(path, "Connect_types_")] = _parse_connect_type_mat(path)

    rg_by_pair = {}
    for path in sorted(source_dir.glob("Rg_*.txt")):
        rg_by_pair[_path_stem_key(path, "Rg_")] = np.asarray(_load_ascii_array(path)).reshape(-1)

    areas_by_pair = {}
    for path in sorted(source_dir.glob("Areas_*.txt")):
        areas_by_pair[_path_stem_key(path, "Areas_")] = np.asarray(_load_ascii_array(path)).reshape(-1)

    return PrescottGeometry(
        source_dir=source_dir,
        centers_um=centers,
        radii_um=radii,
        groups=groups,
        unique_radius_um=unique_radius,
        parameters=parameters,
        number_of_nodes=number_of_nodes,
        coupling=PrescottCouplingMaps(
            neighboring_axon=neighboring_axon,
            center_distance_um=center_distance_um,
            edge_distance_um=edge_distance_um,
            boundary_neighboring=boundary_neighboring,
            boundary_distance_um=boundary_distance_um,
            boundary_coordinates_um=boundary_coordinates_um,
            connect_types=connect_types,
            rg_by_pair=rg_by_pair,
            areas_by_pair=areas_by_pair,
        ),
    )


class PrescottMultiFiberModel:
    def __init__(
        self,
        geometry: PrescottGeometry,
        *,
        parent_axon_nodes: Optional[int] = None,
        branch_nodes: int = 8,
        branches_num: int = 1,
        branch_sequence_nodes: Optional[list[int]] = None,
        main_after_branch_diam_scale: float = 1.0,
        daughter_branch_diam_scale: float = 0.6,
        dt_ms: float = 0.005,
        h_stop_ms: float = 6.0,
    ):
        self.geometry = geometry
        if parent_axon_nodes is None:
            positive_nodes = [int(x) for x in np.asarray(self.geometry.number_of_nodes).reshape(-1) if int(round(float(x))) > 0]
            parent_axon_nodes = positive_nodes[0] if positive_nodes else 27
        self.parent_axon_nodes = int(parent_axon_nodes)
        self.branch_nodes = int(branch_nodes)
        self.branches_num = int(branches_num)
        self.branch_sequence_nodes = branch_sequence_nodes if branch_sequence_nodes is not None else [8]
        self.main_after_branch_diam_scale = float(main_after_branch_diam_scale)
        self.daughter_branch_diam_scale = float(daughter_branch_diam_scale)
        self.dt_ms = float(dt_ms)
        self.h_stop_ms = float(h_stop_ms)
        self.axons: list[MRGaxon] = []
        self.pair_specs: list[PrescottPairSpec] = []
        self.couplers: list[LinearMechanismCoupler] = []
        self.boundary_sinks: list[GroundSinkArray] = []
        self.boundary_couplers: list[LinearMechanismCoupler] = []

    def build_axons(self) -> list[MRGaxon]:
        self.axons = []
        for idx in range(self.geometry.n_axons):
            axon = PrescottFullMRGaxon(
                fiber_diameter=self.geometry.fiber_diameter_um,
                parent_axon_nodes=self.parent_axon_nodes,
                branch_nodes=self.branch_nodes,
                branches_num=self.branches_num,
                branch_sequence_nodes=self.branch_sequence_nodes,
                main_after_branch_diam_scale=self.main_after_branch_diam_scale,
                daughter_branch_diam_scale=self.daughter_branch_diam_scale,
                main_after_branch_param_mode="scaled_radial",
                daughter_branch_param_mode="scaled_radial",
                branch_topology_mode="node",
                dt_ms=self.dt_ms,
                h_stop=self.h_stop_ms,
                reset_nrn=idx == 0,
            )
            self.axons.append(axon)
        return self.axons

    def summary(self) -> dict:
        return {
            "n_axons": self.geometry.n_axons,
            "fiber_diameter_um": self.geometry.fiber_diameter_um,
            "main_after_branch_diam_scale": self.main_after_branch_diam_scale,
            "daughter_branch_diam_scale": self.daughter_branch_diam_scale,
            "n_neighbor_pairs": int(np.count_nonzero(self.geometry.coupling.neighboring_axon)),
            "n_boundary_points": int(self.geometry.coupling.boundary_coordinates_um.shape[0]),
            "n_connect_type_maps": len(self.geometry.coupling.connect_types),
            "n_rg_maps": len(self.geometry.coupling.rg_by_pair),
            "n_area_maps": len(self.geometry.coupling.areas_by_pair),
            "n_pair_specs": len(self.pair_specs),
            "n_couplers": len(self.couplers),
            "n_boundary_couplers": len(self.boundary_couplers),
        }

    @staticmethod
    def _pair_key(axon_i: int, axon_j: int) -> str:
        return f"{int(axon_i) + 1}{int(axon_j) + 1}"

    def _connect_type_key_for_pair(self, axon_i: int, axon_j: int) -> str:
        key = self._pair_key(axon_i, axon_j)
        if key in self.geometry.coupling.connect_types:
            return key
        if "11" in self.geometry.coupling.connect_types:
            return "11"
        raise KeyError(f"No Connect_types mapping for pair ({axon_i}, {axon_j})")

    def _rg_key_for_pair(self, axon_i: int, axon_j: int) -> str:
        key = self._pair_key(axon_i, axon_j)
        if key in self.geometry.coupling.rg_by_pair:
            return key
        if "11" in self.geometry.coupling.rg_by_pair:
            return "11"
        raise KeyError(f"No Rg mapping for pair ({axon_i}, {axon_j})")

    def _areas_key_for_pair(self, axon_i: int, axon_j: int) -> str | None:
        key = self._pair_key(axon_i, axon_j)
        if key in self.geometry.coupling.areas_by_pair:
            return key
        return None

    def _prescott_token_maps(self, axon: MRGaxon) -> tuple[dict[str, str], dict[str, float]]:
        token_map: dict[str, str] = {}
        center_map: dict[str, float] = {}
        for unit_idx, node in enumerate(axon.main_axon):
            token = f"node_{unit_idx}"
            token_map[token] = node.name()
            center_map[token] = float(axon.node_distance_um[node.name()])
        step_idx = 0
        for rec in getattr(axon, '_step_records', []) or []:
            if not rec.get('is_trunk', False):
                continue
            sections = list(rec.get('sections', []) or [])
            mysas = [sec for sec in sections if sec.get('type') == 'mysa']
            fluts = [sec for sec in sections if sec.get('type') == 'flut']
            stins = [sec for sec in sections if sec.get('type') == 'stin']
            cur = float(axon.node_distance_um[axon.main_axon[step_idx].name()]) + 0.5 * float(axon.nodelength)
            if len(mysas) >= 1:
                token = f"MYSA_{2 * step_idx}"
                token_map[token] = str(mysas[0]['name'])
                center_map[token] = cur + 0.5 * float(mysas[0]['L'])
                cur += float(mysas[0]['L'])
            if len(mysas) >= 2:
                tail_mysa_token = f"MYSA_{2 * step_idx + 1}"
            for local_idx, sec in enumerate(fluts):
                token = f"FLUT_{10 * step_idx + local_idx}"
                token_map[token] = str(sec['name'])
                center_map[token] = cur + 0.5 * float(sec['L'])
                cur += float(sec['L'])
            for local_idx, sec in enumerate(stins):
                token = f"STIN_{40 * step_idx + local_idx}"
                token_map[token] = str(sec['name'])
                center_map[token] = cur + 0.5 * float(sec['L'])
                cur += float(sec['L'])
            if len(mysas) >= 2:
                token_map[tail_mysa_token] = str(mysas[1]['name'])
                center_map[tail_mysa_token] = cur + 0.5 * float(mysas[1]['L'])
            step_idx += 1
        return token_map, center_map

    def build_pair_specs(self) -> list[PrescottPairSpec]:
        if not self.axons:
            raise RuntimeError("Call build_axons() before build_pair_specs().")

        center_maps = []
        token_maps = []
        for axon in self.axons:
            token_map, center_map = self._prescott_token_maps(axon)
            token_maps.append(token_map)
            center_maps.append(center_map)
        specs: list[PrescottPairSpec] = []
        n_axons = self.geometry.n_axons
        for i in range(n_axons):
            for j in range(i + 1, n_axons):
                if float(self.geometry.coupling.neighboring_axon[i, j]) == 0.0:
                    continue
                if self.geometry.coupling.connect_types:
                    ct_key = self._connect_type_key_for_pair(i, j)
                    rg_key = self._rg_key_for_pair(i, j)
                    areas_key = self._areas_key_for_pair(i, j)
                    source_tokens, target_tokens = self.geometry.coupling.connect_types[ct_key]
                    rg_raw = np.asarray(self.geometry.coupling.rg_by_pair[rg_key], dtype=float).reshape(-1)
                    areas_raw = None if areas_key is None else np.asarray(self.geometry.coupling.areas_by_pair[areas_key], dtype=float).reshape(-1)
                else:
                    shared_tokens = sorted(set(token_maps[i].keys()) & set(token_maps[j].keys()), key=lambda token: center_maps[i][token])
                    source_tokens = list(shared_tokens)
                    target_tokens = list(shared_tokens)
                    centers = np.asarray([center_maps[i][token] for token in shared_tokens], dtype=float)
                    rg_raw = _compute_rg_dimless_from_centers(centers, float(self.geometry.fiber_diameter_um))
                    areas_raw = None

                max_len = min(len(source_tokens), len(target_tokens), len(rg_raw))
                src_secs: list[str] = []
                tgt_secs: list[str] = []
                src_tokens_kept: list[str] = []
                tgt_tokens_kept: list[str] = []
                rg_kept: list[float] = []
                areas_kept: list[float] = []
                for idx in range(max_len):
                    src_token = str(source_tokens[idx]).strip()
                    tgt_token = str(target_tokens[idx]).strip()
                    src_sec = token_maps[i].get(src_token)
                    tgt_sec = token_maps[j].get(tgt_token)
                    if src_sec is None or tgt_sec is None:
                        continue
                    src_tokens_kept.append(src_token)
                    tgt_tokens_kept.append(tgt_token)
                    src_secs.append(src_sec)
                    tgt_secs.append(tgt_sec)
                    rg_kept.append(float(rg_raw[idx]))
                    if areas_raw is not None and idx < len(areas_raw):
                        areas_kept.append(float(areas_raw[idx]))

                if not src_secs:
                    continue
                specs.append(
                    PrescottPairSpec(
                        axon_i=i,
                        axon_j=j,
                        pair_key=self._pair_key(i, j),
                        source_tokens=src_tokens_kept,
                        target_tokens=tgt_tokens_kept,
                        source_section_names=src_secs,
                        target_section_names=tgt_secs,
                        rg_dimless=np.asarray(rg_kept, dtype=float),
                        areas_um2=np.asarray(areas_kept, dtype=float) if areas_kept else None,
                    )
                )

        self.pair_specs = specs
        return specs

    def build_ephaptic_couplers(self, *, conductance_scale: float = 1.0) -> list[LinearMechanismCoupler]:
        if not self.pair_specs:
            self.build_pair_specs()
        couplers: list[LinearMechanismCoupler] = []
        rho_ohm_um = 1211.0 * 10000.0
        for spec in self.pair_specs:
            axon_i = self.axons[spec.axon_i]
            axon_j = self.axons[spec.axon_j]
            secs_i = [axon_i.get_sec(name) for name in spec.source_section_names]
            secs_j = [axon_j.get_sec(name) for name in spec.target_section_names]
            rd = rho_ohm_um * float(self.geometry.coupling.edge_distance_um[spec.axon_i, spec.axon_j])
            coupler = LinearMechanismCoupler(
                secs_first=secs_i,
                secs_second=secs_j,
                rg_dimless=spec.rg_dimless.tolist(),
                rd_ohm_um2=rd,
                s_um=float(self.geometry.fiber_diameter_um),
                nodeD_um=0.5 * (float(axon_i.mrg_params['nodeD']) + float(axon_j.mrg_params['nodeD'])),
                layer_index=2,
                conductance_scale=float(conductance_scale),
            ).build()
            couplers.append(coupler)
        self.couplers = couplers
        return couplers

    def build_boundary_couplers(self) -> list[LinearMechanismCoupler]:
        self.boundary_sinks = []
        couplers: list[LinearMechanismCoupler] = []
        rho_perineurium_ohm_cm = 1.136e5
        perineurium_thickness_cm = 4.7e-4
        rd_b = rho_perineurium_ohm_cm * 10000.0 * perineurium_thickness_cm * 10000.0
        for row_idx in range(self.geometry.coupling.boundary_neighboring.shape[0]):
            connected = np.flatnonzero(self.geometry.coupling.boundary_neighboring[row_idx] > 0)
            if connected.size == 0:
                continue
            ax_idx = int(connected[0])
            axon = self.axons[ax_idx]
            token_map, center_map = self._prescott_token_maps(axon)
            ordered_tokens = sorted(token_map.keys(), key=lambda token: center_map[token])
            secs = [axon.get_sec(token_map[token]) for token in ordered_tokens]
            centers = np.asarray([center_map[token] for token in ordered_tokens], dtype=float)
            rg = _compute_rg_dimless_from_centers(centers, float(self.geometry.fiber_diameter_um))
            sink = GroundSinkArray(name_prefix=f"boundary_{row_idx}_", n_sections=len(secs))
            self.boundary_sinks.append(sink)
            coupler = LinearMechanismCoupler(
                secs_first=secs,
                secs_second=sink.secs,
                rg_dimless=rg.tolist(),
                rd_ohm_um2=rd_b,
                s_um=float(self.geometry.fiber_diameter_um),
                nodeD_um=float(axon.mrg_params['nodeD']),
                layer_index=2,
                conductance_scale=1.0,
            ).build()
            couplers.append(coupler)
        self.boundary_couplers = couplers
        return couplers

    def run_smoke_simulation(
        self,
        *,
        h5_path: str | Path,
        freq_hz: float,
        amp_nA: float,
        t_start_ms: float = 10.0,
        t_end_ms: float = 10.0,
        biphasic: bool = True,
        phase_us: float = 40.0,
        gap_us: float = 5.0,
        stimulate_all: bool = True,
    ) -> Path:
        from MRG_lib import h
        if not self.axons:
            self.build_axons()
        if not self.couplers:
            self.build_pair_specs()
            self.build_ephaptic_couplers()
        if not self.boundary_couplers:
            self.build_boundary_couplers()

        for idx, axon in enumerate(self.axons):
            if stimulate_all or idx == 0:
                axon.set_stimulation_params(
                    mode="create",
                    biphasic=biphasic,
                    freq_hz=float(freq_hz),
                    amp=float(amp_nA),
                    t_start=float(t_start_ms),
                    t_end=float(t_end_ms),
                    phase_us=float(phase_us),
                    gap_us=float(gap_us),
                )
                axon.create_stimulator()

        record_t = h.Vector().record(h._ref_t)
        recordings = []
        for idx, axon in enumerate(self.axons):
            terminal_main = axon.get_terminal_main_segment()
            terminal_daughter = axon.get_terminal_daughter_segment()
            extra = [("terminal_main", terminal_main)]
            if terminal_daughter is not None:
                extra.append(("terminal_daughter", terminal_daughter))
            segs, names = axon.collect_recording_targets(include_stimulation_point=True, extra_named_segments=extra)
            vecs = [h.Vector().record(seg._ref_v) for seg in segs]
            recordings.append((idx, names, segs, vecs))

        h.finitialize(-80.0)
        h.tstop = float(t_end_ms)
        h.run()

        h5_path = Path(h5_path)
        h5_path.parent.mkdir(parents=True, exist_ok=True)
        with h5py.File(h5_path, "w") as f:
            f.attrs["n_axons"] = int(self.geometry.n_axons)
            f.attrs["fiber_diameter_um"] = float(self.geometry.fiber_diameter_um)
            f.attrs["edge_dist_um"] = float(np.min(self.geometry.coupling.edge_distance_um[self.geometry.coupling.edge_distance_um > 0]))
            f.attrs["freq_hz"] = float(freq_hz)
            f.attrs["amp_nA"] = float(amp_nA)
            f.attrs["stimulate_all"] = int(bool(stimulate_all))
            for idx, names, segs, vecs in recordings:
                grp = f.create_group(f"Axon_{idx:02d}/Model")
                grp.create_dataset("time", data=np.asarray(record_t))
                traces = grp.create_group("Traces")
                for name, seg, vec in zip(names, segs, vecs):
                    trace_grp = traces.require_group(str(name))
                    node_name = f"{seg.sec.name().replace('.', '_')}_{seg.x:.2f}"
                    node_grp = trace_grp.create_group(node_name)
                    node_grp.create_dataset("voltage", data=np.asarray(vec))
        return h5_path

    def plot_packing(self, out_path: str | Path, *, show_numbers: bool = True) -> Path:
        out_path = Path(out_path)
        fig, ax = plt.subplots(figsize=(8.5, 8.5), dpi=300)
        for idx, (center, radius) in enumerate(zip(self.geometry.centers_um, self.geometry.radii_um)):
            ax.add_patch(Circle((float(center[0]), float(center[1])), float(radius), fc="#dbeafe", ec="#1d4ed8", lw=1.5, alpha=0.9))
            if show_numbers:
                ax.text(float(center[0]), float(center[1]), str(idx), ha="center", va="center", fontsize=10, color="#111827")
        ax.set_title("Упаковка 20 аксонов", fontsize=18, fontweight="bold")
        ax.set_xlabel("x (мкм)")
        ax.set_ylabel("y (мкм)")
        ax.set_aspect("equal", adjustable="box")
        ax.grid(True, alpha=0.18)
        fig.tight_layout()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, bbox_inches="tight")
        plt.close(fig)
        return out_path

    def plot_neighbor_graph(self, out_path: str | Path) -> Path:
        out_path = Path(out_path)
        fig, ax = plt.subplots(figsize=(8.5, 8.5), dpi=300)
        coords = self.geometry.centers_um
        N = self.geometry.coupling.neighboring_axon
        for i, (center, radius) in enumerate(zip(coords, self.geometry.radii_um)):
            ax.add_patch(Circle((float(center[0]), float(center[1])), float(radius), fc="#dcfce7", ec="#166534", lw=1.2, alpha=0.85))
            ax.text(float(center[0]), float(center[1]), str(i), ha="center", va="center", fontsize=9, color="#111827")
        for i in range(N.shape[0]):
            for j in range(N.shape[1]):
                if i == j or float(N[i, j]) == 0.0:
                    continue
                ax.plot([coords[i, 0], coords[j, 0]], [coords[i, 1], coords[j, 1]], color="#7c3aed", lw=0.8, alpha=0.35)
        ax.set_title("Схема межаксонных связей Prescott", fontsize=18, fontweight="bold")
        ax.set_xlabel("x (мкм)")
        ax.set_ylabel("y (мкм)")
        ax.set_aspect("equal", adjustable="box")
        ax.grid(True, alpha=0.18)
        fig.tight_layout()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, bbox_inches="tight")
        plt.close(fig)
        return out_path

    def plot_boundary_graph(self, out_path: str | Path) -> Path:
        out_path = Path(out_path)
        fig, ax = plt.subplots(figsize=(8.5, 8.5), dpi=300)
        coords = self.geometry.centers_um
        boundary = self.geometry.coupling.boundary_coordinates_um
        B = self.geometry.coupling.boundary_neighboring
        for idx, (center, radius) in enumerate(zip(coords, self.geometry.radii_um)):
            ax.add_patch(Circle((float(center[0]), float(center[1])), float(radius), fc="#dbeafe", ec="#1d4ed8", lw=1.2, alpha=0.85))
            ax.text(float(center[0]), float(center[1]), str(idx), ha="center", va="center", fontsize=9, color="#111827")
        ax.scatter(boundary[:, 0], boundary[:, 1], s=55, color="#f59e0b", edgecolors="#92400e", label="точки границы")
        for b_idx in range(B.shape[0]):
            for a_idx in range(B.shape[1]):
                if float(B[b_idx, a_idx]) == 0.0:
                    continue
                ax.plot([boundary[b_idx, 0], coords[a_idx, 0]], [boundary[b_idx, 1], coords[a_idx, 1]], color="#6b7280", lw=1.0, alpha=0.55)
        ax.set_title("Граничные связи внешних аксонов", fontsize=18, fontweight="bold")
        ax.set_xlabel("x (мкм)")
        ax.set_ylabel("y (мкм)")
        ax.set_aspect("equal", adjustable="box")
        ax.grid(True, alpha=0.18)
        ax.legend(frameon=False, loc="upper right")
        fig.tight_layout()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, bbox_inches="tight")
        plt.close(fig)
        return out_path


def infer_mode_from_path(path: str | Path) -> str:
    text = str(path)
    if "misaligned" in text.lower():
        return "misaligned"
    return "aligned"
