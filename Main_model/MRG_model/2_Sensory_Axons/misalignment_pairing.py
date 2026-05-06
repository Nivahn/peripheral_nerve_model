from __future__ import annotations

from dataclasses import dataclass, replace


@dataclass(frozen=True)
class PairingPoint:
    name: str
    x_um: float
    kind: str
    path_type: str = "main"
    pair_key: str = ""


def select_main_path_points(points: list[PairingPoint]) -> list[PairingPoint]:
    # В branch-aware misalignment используем только main-path секции.
    # Daughter path полностью пропускаем, branch_node при этом остаётся в main path.
    return [p for p in points if str(p.path_type) == "main"]


def apply_offset_um(points: list[PairingPoint], offset_um: float) -> list[PairingPoint]:
    return [replace(p, x_um=float(p.x_um) + float(offset_um)) for p in points]


def _point_pair_key(point: PairingPoint) -> str:
    return str(point.pair_key) if str(point.pair_key) else str(point.kind)


def pair_points_monotonic_nearest(
    points_a: list[PairingPoint],
    points_b: list[PairingPoint],
    *,
    prefer_higher_x_on_tie: bool = True,
) -> list[tuple[PairingPoint, PairingPoint]]:
    if not points_a or not points_b:
        return []

    a_sorted = sorted(points_a, key=lambda p: (float(p.x_um), str(p.name)))
    b_sorted = sorted(points_b, key=lambda p: (float(p.x_um), str(p.name)))

    n_pairs = min(len(a_sorted), len(b_sorted))
    a_used = a_sorted[:n_pairs]

    out: list[tuple[PairingPoint, PairingPoint]] = []
    start_j = 0
    total_b = len(b_sorted)

    for idx_a, point_a in enumerate(a_used):
        remaining_a = n_pairs - idx_a
        max_j = total_b - remaining_a
        if start_j > max_j:
            break

        best_j = start_j
        best_dx = abs(float(b_sorted[start_j].x_um) - float(point_a.x_um))
        for j in range(start_j + 1, max_j + 1):
            dx = abs(float(b_sorted[j].x_um) - float(point_a.x_um))
            if dx < best_dx:
                best_j = j
                best_dx = dx
            elif dx == best_dx and prefer_higher_x_on_tie:
                if float(b_sorted[j].x_um) > float(b_sorted[best_j].x_um):
                    best_j = j
                    best_dx = dx

        out.append((point_a, b_sorted[best_j]))
        start_j = best_j + 1

    return out


def pair_points_monotonic_nearest_by_key(
    points_a: list[PairingPoint],
    points_b: list[PairingPoint],
    *,
    prefer_higher_x_on_tie: bool = True,
) -> list[tuple[PairingPoint, PairingPoint]]:
    if not points_a or not points_b:
        return []

    groups_a: dict[str, list[PairingPoint]] = {}
    groups_b: dict[str, list[PairingPoint]] = {}
    for point in points_a:
        groups_a.setdefault(_point_pair_key(point), []).append(point)
    for point in points_b:
        groups_b.setdefault(_point_pair_key(point), []).append(point)

    out: list[tuple[PairingPoint, PairingPoint]] = []
    shared_keys = sorted(set(groups_a.keys()) & set(groups_b.keys()))
    for key in shared_keys:
        out.extend(
            pair_points_monotonic_nearest(
                groups_a[key],
                groups_b[key],
                prefer_higher_x_on_tie=prefer_higher_x_on_tie,
            )
        )

    out.sort(key=lambda pair: (float(pair[0].x_um), str(pair[0].name)))
    return out
