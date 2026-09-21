from __future__ import annotations

"""compare_ec_noec.py

Сравнение двух прогонов мультиаксонной модели: с эфаптической связью (EC) и без (no-EC).

На вход — два summary.csv, полученных analyze_7axon_sweep.py.
На выход — delta_*.csv и графики (EC - no_EC) по частоте.

Ключ сопоставления: (edge_dist_um, freq_hz, axon, site).
"""

import argparse
import csv
import json
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt  # noqa: E402


def load_summary(path: Path) -> dict:
    rows = list(csv.DictReader(open(path, encoding="utf-8")))
    out = {}
    for r in rows:
        try:
            key = (round(float(r["edge_dist_um"]), 4), round(float(r["freq_hz"]), 4),
                   str(r["axon"]), str(r["site"]))
        except (KeyError, ValueError):
            continue
        out[key] = r
    return out


def fnum(r, name):
    try:
        return float(r[name])
    except (KeyError, TypeError, ValueError):
        return float("nan")


def main() -> None:
    ap = argparse.ArgumentParser(description="Compare EC vs no-EC multifiber summaries.")
    ap.add_argument("--ec", required=True, help="summary.csv for the ephaptic (EC) run")
    ap.add_argument("--noec", required=True, help="summary.csv for the no-EC control run")
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ec = load_summary(Path(args.ec))
    noec = load_summary(Path(args.noec))
    keys = sorted(set(ec) & set(noec))
    print(f"EC rows={len(ec)} noEC rows={len(noec)} matched keys={len(keys)}")

    metrics = ["following_fraction", "conduction_ratio", "median_latency_ms", "median_velocity_m_s"]
    rows = []
    for k in keys:
        ed, fr, axon, site = k
        row = {"edge_dist_um": ed, "freq_hz": fr, "axon": axon, "site": site,
               "is_center": int(str(ec[k].get("is_center", 0)))}
        for m in metrics:
            a, b = fnum(ec[k], m), fnum(noec[k], m)
            row[f"{m}_ec"] = a
            row[f"{m}_noec"] = b
            row[f"delta_{m}"] = (a - b) if (np.isfinite(a) and np.isfinite(b)) else float("nan")
        rows.append(row)

    csv_path = out_dir / "delta_ec_vs_noec.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    print(f"Saved {csv_path} ({len(rows)} rows)")

    edges = sorted({r["edge_dist_um"] for r in rows})
    center_sites = ["before", "branch", "after_main", "after_daughter", "terminal_main", "terminal_daughter"]

    def series(site, ed, metric, is_center=1):
        fr, mv = [], []
        for freq in sorted({r["freq_hz"] for r in rows if r["site"] == site and abs(r["edge_dist_um"] - ed) < 1e-9}):
            vals = [r[f"delta_{metric}"] for r in rows
                    if r["site"] == site and abs(r["edge_dist_um"] - ed) < 1e-9
                    and abs(r["freq_hz"] - freq) < 1e-9 and int(r["is_center"]) == is_center
                    and np.isfinite(r[f"delta_{metric}"])]
            if vals:
                fr.append(freq); mv.append(float(np.mean(vals)))
        return fr, mv

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), dpi=160)
    for ed in edges:
        for site in center_sites:
            fr, mv = series(site, ed, "following_fraction", 1)
            if fr:
                axes[0].plot(fr, mv, marker="o", label=f"ed{ed:g}:{site}")
        fr, mv = series("terminal_main", ed, "following_fraction", 0)
        if fr:
            axes[0].plot(fr, mv, marker="s", ls="--", color="gray", label=f"ed{ed:g}:neighbors")
    axes[0].axhline(0.0, color="k", lw=0.8)
    axes[0].set_title("delta following_fraction (EC - no_EC)")
    axes[0].set_xlabel("freq, Hz"); axes[0].set_ylabel("delta")
    axes[0].grid(alpha=0.25)
    h, l = axes[0].get_legend_handles_labels()
    if h:
        axes[0].legend(fontsize=7, ncol=2)

    for ed in edges:
        for site in center_sites:
            fr, mv = series(site, ed, "median_velocity_m_s", 1)
            if fr:
                axes[1].plot(fr, mv, marker="o", label=f"ed{ed:g}:{site}")
    axes[1].axhline(0.0, color="k", lw=0.8)
    axes[1].set_title("delta median_velocity_m_s (EC - no_EC, center)")
    axes[1].set_xlabel("freq, Hz"); axes[1].set_ylabel("delta, m/s")
    axes[1].grid(alpha=0.25)
    h, l = axes[1].get_legend_handles_labels()
    if h:
        axes[1].legend(fontsize=7, ncol=2)

    fig.tight_layout()
    p = out_dir / "delta_ec_vs_noec.png"
    fig.savefig(p, bbox_inches="tight"); plt.close(fig)
    print(f"Saved {p}")

    # Краткая сводка по терминали центра
    summary = {"n_matched": len(rows),
               "max_abs_delta_following_terminal": float(np.nanmax([
                   abs(r["delta_following_fraction"]) for r in rows
                   if r["site"] == "terminal_main" and int(r["is_center"]) == 1
                   and np.isfinite(r["delta_following_fraction"])] or [0.0])),
               "max_abs_delta_latency_terminal_ms": float(np.nanmax([
                   abs(r["delta_median_latency_ms"]) for r in rows
                   if r["site"] == "terminal_main" and int(r["is_center"]) == 1
                   and np.isfinite(r["delta_median_latency_ms"])] or [0.0]))}
    (out_dir / "delta_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
