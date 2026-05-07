"""test_misalignment_pairing_logic.py

Test-first specification for branch-aware misalignment pairing.

What this test checks:
1. Daughter-side sections are skipped completely.
2. `branch_node` stays in the main-path candidate list.
3. Pairing is nearest-by-X, monotonic, and one-to-one.
4. Offsets `0.25 * Lstep` and `0.5 * Lstep` produce the expected pair map.

This file also saves a visual diagnostic figure so the pairing can be inspected.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt

from misalignment_pairing import (
    PairingPoint,
    PairingUnit,
    apply_offset_um,
    flatten_paired_units,
    pair_units_monotonic_nearest_by_node,
    select_main_path_points,
)


OUT_DIR = Path(__file__).resolve().parent / "data" / "misalignment_pairing_tests"
LSTEP_UM = 100.0


@dataclass(frozen=True)
class RawPoint:
    name: str
    x_um: float
    kind: str
    path_type: str
    pair_key: str


RAW_POINTS_A = [
    RawPoint("branch_node", 100.0, "node", "main", "node"),
    RawPoint("daughter_mysa", 108.0, "mysa", "daughter", "mysa_0"),
    RawPoint("daughter_flut", 122.0, "flut", "daughter", "flut_0"),
    RawPoint("daughter_stin_1", 138.0, "stin", "daughter", "stin_0"),
    RawPoint("daughter_stin_2", 163.0, "stin", "daughter", "stin_1"),
    RawPoint("main_mysa", 108.0, "mysa", "main", "mysa_0"),
    RawPoint("main_flut", 122.0, "flut", "main", "flut_0"),
    RawPoint("main_stin_1", 138.0, "stin", "main", "stin_0"),
    RawPoint("main_stin_2", 163.0, "stin", "main", "stin_1"),
    RawPoint("main_node", 200.0, "node", "main", "node"),
]

RAW_POINTS_B = [
    RawPoint("pre_node", 0.0, "node", "main", "node"),
    RawPoint("pre_mysa", 8.0, "mysa", "main", "mysa_0"),
    RawPoint("pre_flut", 22.0, "flut", "main", "flut_0"),
    RawPoint("pre_stin_1", 38.0, "stin", "main", "stin_0"),
    RawPoint("pre_stin_2", 63.0, "stin", "main", "stin_1"),
    RawPoint("branch_node", 100.0, "node", "main", "node"),
    RawPoint("main_mysa", 108.0, "mysa", "main", "mysa_0"),
    RawPoint("main_flut", 122.0, "flut", "main", "flut_0"),
    RawPoint("main_stin_1", 138.0, "stin", "main", "stin_0"),
    RawPoint("main_stin_2", 163.0, "stin", "main", "stin_1"),
    RawPoint("main_node", 200.0, "node", "main", "node"),
]


EXPECTED_MAIN_SEQUENCE_A = [
    "branch_node",
    "main_mysa",
    "main_flut",
    "main_stin_1",
    "main_stin_2",
    "main_node",
]

EXPECTED_PAIRS = {
    "aligned": {
        "branch_node": "branch_node",
        "main_mysa": "main_mysa",
        "main_flut": "main_flut",
        "main_stin_1": "main_stin_1",
        "main_stin_2": "main_stin_2",
        "main_node": "main_node",
    },
    "quarter_step": {
        "branch_node": "branch_node",
        "main_mysa": "main_mysa",
        "main_flut": "main_flut",
        "main_stin_1": "main_stin_1",
        "main_stin_2": "main_stin_2",
        "main_node": "main_node",
    },
    "half_step": {
        "branch_node": "branch_node",
        "main_mysa": "main_mysa",
        "main_flut": "main_flut",
        "main_stin_1": "main_stin_1",
        "main_stin_2": "main_stin_2",
        "main_node": "main_node",
    },
}


def to_pairing_points(raw_points: list[RawPoint]) -> list[PairingPoint]:
    return [PairingPoint(name=p.name, x_um=float(p.x_um), kind=p.kind, path_type=p.path_type, pair_key=p.pair_key) for p in raw_points]


def pair_map(points_a: list[PairingPoint], points_b: list[PairingPoint]) -> dict[str, str]:
    pairs = flatten_paired_units(pair_units_monotonic_nearest_by_node(points_to_units(points_a), points_to_units(points_b), target_dx_um=0.0))
    return {a.name: b.name for a, b in pairs}


def points_to_units(points: list[PairingPoint]) -> list[PairingUnit]:
    by_name = {p.name: p for p in points}
    return [
        PairingUnit(
            unit_index=0,
            node=by_name["branch_node"],
            mysa_left=by_name["main_mysa"],
            flut_left=by_name["main_flut"],
            stins=(by_name["main_stin_1"], by_name["main_stin_2"]),
            flut_right=None,
            mysa_right=None,
        ),
        PairingUnit(
            unit_index=1,
            node=by_name["main_node"],
            mysa_left=None,
            flut_left=None,
            stins=tuple(),
            flut_right=None,
            mysa_right=None,
        ),
    ]


def plot_case(ax, title: str, points_a, points_b, pairs):
    ax.scatter([p.x_um for p in points_a], [1.0] * len(points_a), color="#2563eb", s=40, label="AxonA")
    ax.scatter([p.x_um for p in points_b], [0.0] * len(points_b), color="#dc2626", s=40, label="AxonB")
    for p in points_a:
        ax.text(p.x_um, 1.05, p.name, rotation=45, ha="left", va="bottom", fontsize=7, color="#2563eb")
    for p in points_b:
        ax.text(p.x_um, -0.05, p.name, rotation=45, ha="left", va="top", fontsize=7, color="#dc2626")
    for pa, pb in pairs:
        ax.plot([pa.x_um, pb.x_um], [1.0, 0.0], color="#111827", lw=1.3, alpha=0.8)
    ax.set_title(title)
    ax.set_yticks([0.0, 1.0])
    ax.set_yticklabels(["AxonB", "AxonA"])
    ax.grid(True, axis="x", alpha=0.25)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    raw_a = to_pairing_points(RAW_POINTS_A)
    raw_b = to_pairing_points(RAW_POINTS_B)

    # Test 1. Daughter side must be skipped completely.
    main_a = select_main_path_points(raw_a)
    main_b = select_main_path_points(raw_b)
    got_names_a = [p.name for p in main_a]
    assert got_names_a == EXPECTED_MAIN_SEQUENCE_A, f"Main-path filter failed: {got_names_a}"
    assert all(p.path_type == "main" for p in main_a), "Daughter points leaked into AxonA main-path candidates"
    assert all(p.path_type == "main" for p in main_b), "Daughter points leaked into AxonB main-path candidates"

    cases = [
        ("aligned", 0.0),
        ("quarter_step", 0.25 * LSTEP_UM),
        ("half_step", 0.5 * LSTEP_UM),
    ]

    fig, axes = plt.subplots(len(cases), 1, figsize=(14.0, 4.0 * len(cases)), dpi=180, sharex=False)
    if len(cases) == 1:
        axes = [axes]

    ascii_lines = []
    for ax, (label, offset_um) in zip(axes, cases):
        shifted_b = apply_offset_um(main_b, offset_um)
        units_a = points_to_units(main_a)
        units_b = points_to_units(shifted_b)
        unit_pairs = pair_units_monotonic_nearest_by_node(units_a, units_b, target_dx_um=offset_um)
        pairs = flatten_paired_units(unit_pairs)
        got = {a.name: b.name for a, b in pairs}
        assert got == EXPECTED_PAIRS[label], f"Pair mismatch for {label}: {got}"
        assert all(a.kind == b.kind for a, b in pairs), f"Cross-kind pair found for {label}: {[(a.kind, b.kind) for a, b in pairs]}"

        ascii_lines.append(f"[{label}] offset_um={offset_um:.3f}")
        for a, b in pairs:
            ascii_lines.append(f"  {a.name:>14} -> {b.name:<14} | dx={b.x_um - a.x_um:+.3f} um")

        plot_case(ax, f"{label} | offset={offset_um:.1f} um", main_a, shifted_b, pairs)

    fig.tight_layout()
    fig_path = OUT_DIR / "synthetic_pairing_cases.png"
    fig.savefig(fig_path, bbox_inches="tight")
    plt.close(fig)

    ascii_path = OUT_DIR / "synthetic_pairing_cases.txt"
    ascii_path.write_text("\n".join(ascii_lines) + "\n", encoding="utf-8")
    print(f"Saved pairing figure: {fig_path}")
    print(f"Saved pairing text: {ascii_path}")


if __name__ == "__main__":
    main()
