from __future__ import annotations

from dataclasses import dataclass, replace


@dataclass(frozen=True)
class PairingPoint:
    name: str
    x_um: float
    kind: str
    path_type: str = "main"


def select_main_path_points(points: list[PairingPoint]) -> list[PairingPoint]:
    # В branch-aware misalignment используем только main-path секции.
    # Daughter path полностью пропускаем, branch_node при этом остаётся в main path.
    return [p for p in points if str(p.path_type) == "main"]


def apply_offset_um(points: list[PairingPoint], offset_um: float) -> list[PairingPoint]:
    return [replace(p, x_um=float(p.x_um) + float(offset_um)) for p in points]


def pair_points_monotonic_nearest(
    points_a: list[PairingPoint],
    points_b: list[PairingPoint],
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

        out.append((point_a, b_sorted[best_j]))
        start_j = best_j + 1

    return out
