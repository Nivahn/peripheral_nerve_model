from __future__ import annotations

"""plot_diploma_multifiber.py

Графики для диплома по мультиаксонной модели (7 аксонов).

Вход: summary.csv от analyze_7axon_sweep.py (EC, no-EC, misaligned).
Выход:
  - following_center_vs_neighbors.png   — following fraction: центр vs соседи (EC)
  - following_ec_vs_noec_center.png     — EC vs no-EC, терминаль центра
  - delta_following_ec_minus_noec.png   — дельта following
  - latency_vs_edge_center.png          — латентность центра vs edge distance
"""

import argparse
import csv
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt  # noqa: E402


def fnum(r, name):
    try:
        return float(r[name])
    except (KeyError, TypeError, ValueError):
        return float("nan")


def load(path: Path) -> list:
    return list(csv.DictReader(open(path, encoding="utf-8")))


def mean_series(rows, *, site, metric, freq, edge=None, is_center=None):
    """Среднее по подходящим строкам (все аксоны/edge, если не задано)."""
    vals = []
    for r in rows:
        if r["site"] != site:
            continue
        if abs(fnum(r, "freq_hz") - freq) > 1e-6:
            continue
        if edge is not None and abs(fnum(r, "edge_dist_um") - edge) > 1e-6:
            continue
        if is_center is not None and int(r.get("is_center", 0)) != int(is_center):
            continue
        v = fnum(r, metric)
        if v == v:
            vals.append(v)
    return float(np.mean(vals)) if vals else float("nan")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ec", required=True)
    ap.add_argument("--noec", required=True)
    ap.add_argument("--mis", default=None)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    ec = load(Path(args.ec))
    noec = load(Path(args.noec))
    mis = load(Path(args.mis)) if args.mis else None

    freqs = sorted({fnum(r, "freq_hz") for r in ec if fnum(r, "freq_hz") == fnum(r, "freq_hz")})
    edges = sorted({fnum(r, "edge_dist_um") for r in ec if fnum(r, "edge_dist_um") == fnum(r, "edge_dist_um")})

    # 1) центр vs соседи (EC), по edge
    fig, ax = plt.subplots(figsize=(9, 5.5), dpi=160)
    for ed in edges:
        yc = [mean_series(ec, site="terminal_main", metric="following_fraction", freq=f, edge=ed, is_center=1) for f in freqs]
        yn = [mean_series(ec, site="terminal_main", metric="following_fraction", freq=f, edge=ed, is_center=0) for f in freqs]
        ax.plot(freqs, yc, marker="o", label=f"center (branched), edge={ed:g}")
        ax.plot(freqs, yn, marker="s", ls="--", alpha=0.6, label=f"neighbors, edge={ed:g}")
    ax.set_xlabel("freq, Hz"); ax.set_ylabel("spikes / stimuli"); ax.set_ylim(-0.05, 1.15)
    ax.set_title("Following fraction: center (branched) vs neighbors (EC)")
    ax.grid(alpha=0.25); ax.legend(fontsize=8)
    fig.tight_layout(); p = out / "following_center_vs_neighbors.png"
    fig.savefig(p, bbox_inches="tight"); plt.close(fig); print("Saved", p)

    # 2) EC vs no-EC (центр), по edge
    fig, ax = plt.subplots(figsize=(9, 5.5), dpi=160)
    for ed in edges:
        ye = [mean_series(ec, site="terminal_main", metric="following_fraction", freq=f, edge=ed, is_center=1) for f in freqs]
        yn = [mean_series(noec, site="terminal_main", metric="following_fraction", freq=f, edge=ed, is_center=1) for f in freqs]
        ax.plot(freqs, ye, marker="o", label=f"EC, edge={ed:g}")
        ax.plot(freqs, yn, marker="x", ls="--", label=f"no-EC, edge={ed:g}")
    ax.set_xlabel("freq, Hz"); ax.set_ylabel("spikes / stimuli"); ax.set_ylim(-0.05, 1.15)
    ax.set_title("Following fraction at terminal_main (center axon): EC vs no-EC")
    ax.grid(alpha=0.25); ax.legend(fontsize=8)
    fig.tight_layout(); p = out / "following_ec_vs_noec_center.png"
    fig.savefig(p, bbox_inches="tight"); plt.close(fig); print("Saved", p)

    # 3) дельта following (EC - noEC), центр
    fig, ax = plt.subplots(figsize=(9, 5.5), dpi=160)
    for ed in edges:
        d = []
        for f in freqs:
            a = mean_series(ec, site="terminal_main", metric="following_fraction", freq=f, edge=ed, is_center=1)
            b = mean_series(noec, site="terminal_main", metric="following_fraction", freq=f, edge=ed, is_center=1)
            d.append(a - b if (a == a and b == b) else float("nan"))
        ax.plot(freqs, d, marker="o", label=f"edge={ed:g}")
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xlabel("freq, Hz"); ax.set_ylabel("delta following (EC - no-EC)")
    ax.set_title("Ephaptic contribution to conduction (center axon)")
    ax.grid(alpha=0.25); ax.legend(fontsize=8)
    fig.tight_layout(); p = out / "delta_following_ec_minus_noec.png"
    fig.savefig(p, bbox_inches="tight"); plt.close(fig); print("Saved", p)

    # 4) латентность центра vs edge (EC), несколько частот
    fig, ax = plt.subplots(figsize=(8, 5.5), dpi=160)
    for fr in [50, 200, 300]:
        y = [mean_series(ec, site="terminal_main", metric="median_latency_ms", freq=fr, edge=ed, is_center=1) for ed in edges]
        ax.plot(edges, y, marker="o", label=f"{fr} Hz")
    ax.set_xlabel("edge distance, um"); ax.set_ylabel("median latency, ms")
    ax.set_title("Latency at terminal_main (center) vs edge distance (EC)")
    ax.grid(alpha=0.25); ax.legend()
    fig.tight_layout(); p = out / "latency_vs_edge_center.png"
    fig.savefig(p, bbox_inches="tight"); plt.close(fig); print("Saved", p)

    # 5) aligned vs misaligned (EC), центр
    if mis:
        fig, ax = plt.subplots(figsize=(9, 5.5), dpi=160)
        for ed in edges:
            ya = [mean_series(ec, site="terminal_main", metric="following_fraction", freq=f, edge=ed, is_center=1) for f in freqs]
            ym = [mean_series(mis, site="terminal_main", metric="following_fraction", freq=f, edge=ed, is_center=1) for f in freqs]
            ax.plot(freqs, ya, marker="o", label=f"aligned, edge={ed:g}")
            ax.plot(freqs, ym, marker="^", ls="--", label=f"misaligned, edge={ed:g}")
        ax.set_xlabel("freq, Hz"); ax.set_ylabel("spikes / stimuli"); ax.set_ylim(-0.05, 1.15)
        ax.set_title("Aligned vs misaligned (center axon, EC)")
        ax.grid(alpha=0.25); ax.legend(fontsize=8)
        fig.tight_layout(); p = out / "aligned_vs_misaligned_center.png"
        fig.savefig(p, bbox_inches="tight"); plt.close(fig); print("Saved", p)


if __name__ == "__main__":
    main()
