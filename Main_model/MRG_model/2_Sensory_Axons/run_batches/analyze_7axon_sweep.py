from __future__ import annotations

"""analyze_7axon_sweep.py

Анализ результатов мультиаксонного (7 аксонов) Prescott-прогона.

Метрики — как в 2-аксонном анализе:
  * детект спайков (height + prominence + min distance);
  * n_spikes; following_fraction = n_spikes / n_stimuli;
  * conduction_ratio = n_matched / n_ref (reference-based, ref = stimulation_point);
  * latency (median/mean) для сматченных спайков;
  * velocity = (distance_target - distance_ref) / latency;
  * ectopic_count.

Вход:  <root>/*/freq_XXXXhz_*.h5   (структура: Axon_XX/Model/time,
       Axon_XX/Model/Traces/<trace>/<node>/voltage, node.attrs["distance_um"]).

Выход: <out>/spikes_all.csv, <out>/summary.csv, <out>/*.png
"""

import argparse
import csv
import json
import re
from pathlib import Path

import h5py
import numpy as np

import matplotlib
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt  # noqa: E402
from scipy.signal import find_peaks  # noqa: E402


TRACE_ORDER = [
    "stimulation_point",
    "before_branch",
    "branch_point",
    "after_branch_main",
    "after_branch_daughter",
    "terminal_main",
    "terminal_daughter",
]

# Какому "сайту" соответствует трейс (для графиков).
TRACE_SITE = {
    "stimulation_point": "stim",
    "before_branch": "before",
    "branch_point": "branch",
    "after_branch_main": "after_main",
    "after_branch_daughter": "after_daughter",
    "terminal_main": "terminal_main",
    "terminal_daughter": "terminal_daughter",
}

NODE_RE = re.compile(r"node_(\d+)")


def node_index(name: str) -> int:
    m = NODE_RE.search(name)
    return int(m.group(1)) if m else 10 ** 9


def _safe_legend(ax, **kw):
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(**kw)


def detect_spikes(t_ms, v_mV, *, threshold_mv, prominence_mv, min_dist_ms, start_ms, dt_ms):
    t_ms = np.asarray(t_ms, dtype=float)
    v_mV = np.asarray(v_mV, dtype=float)
    empty = (np.asarray([], dtype=int), np.asarray([], dtype=float), np.asarray([], dtype=float))
    if t_ms.size < 3 or v_mV.size != t_ms.size:
        return empty
    if not np.isfinite(dt_ms) or dt_ms <= 0:
        return empty
    start_idx = int(np.searchsorted(t_ms, start_ms, side="left"))
    if start_idx >= t_ms.size - 1:
        return empty
    min_dist_pts = max(1, int(round(float(min_dist_ms) / float(dt_ms))))
    peaks, props = find_peaks(
        v_mV[start_idx:],
        height=float(threshold_mv),
        prominence=float(prominence_mv),
        distance=min_dist_pts,
    )
    idx = peaks.astype(int) + start_idx
    return idx, t_ms[idx], v_mV[idx]


def match_causal(ref_t, tgt_t, *, min_latency_ms, max_latency_ms):
    """Жадное причинное сопоставление: для каждого ref-спайка ищем первый tgt в окне."""
    ref_t = np.asarray(ref_t, dtype=float)
    tgt_t = np.asarray(tgt_t, dtype=float)
    matched_t, latencies = [], []
    j = 0
    for rt in ref_t:
        left = rt + float(min_latency_ms)
        right = rt + float(max_latency_ms)
        while j < tgt_t.size and tgt_t[j] < left:
            j += 1
        if j < tgt_t.size and tgt_t[j] <= right:
            matched_t.append(float(tgt_t[j]))
            latencies.append(float(tgt_t[j] - rt))
            j += 1
    return np.asarray(matched_t, dtype=float), np.asarray(latencies, dtype=float)


def node_step_um(fiber_diameter_um: float) -> float:
    d = float(fiber_diameter_um)
    if d < 5.7:
        return -3.22 * d * d + 148.0 * d - 128.0
    return -8.215 * d * d + 272.4 * d - 780.2


def parse_attrs(f: h5py.File) -> dict:
    a = {}
    for k in ("n_axons", "fiber_diameter_um", "edge_dist_um", "freq_hz", "amp_nA", "stimulate_all", "dt_ms",
              "t_start_ms", "t_end_ms"):
        if k in f.attrs:
            a[k] = f.attrs[k]
    return a


def read_trace(f, axon, trace):  # -> (t, v, dist, node_name) or None
    base = f"{axon}/Model"
    if base not in f:
        return None
    grp = f[base]
    if "traces" not in grp and "Traces" not in grp:
        return None
    traces = grp["Traces"] if "Traces" in grp else grp["traces"]
    if trace not in traces:
        return None
    tg = traces[trace]
    node_names = sorted(tg.keys(), key=node_index)
    if not node_names:
        return None
    nd = node_names[0]
    ng = tg[nd]
    if "voltage" not in ng:
        # иногда напряжение лежит прямо в группе трейса
        if "voltage" in tg:
            v = np.asarray(tg["voltage"], dtype=float)
            dist = float(tg.attrs.get("distance_um", np.nan))
        else:
            return None
    else:
        v = np.asarray(ng["voltage"], dtype=float)
        dist = float(ng.attrs.get("distance_um", np.nan))
    if "time" not in grp:
        return None
    t = np.asarray(grp["time"], dtype=float)
    if t.size != v.size:
        n = min(t.size, v.size)
        t, v = t[:n], v[:n]
    return t, v, dist, nd


def main() -> None:
    ap = argparse.ArgumentParser(description="Analyze multifiber Prescott sweep HDF5 outputs.")
    ap.add_argument("--root", default="data/prescott_7axon_sweep")
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--threshold-mv", type=float, default=-43.0)
    ap.add_argument("--prominence-mv", type=float, default=5.0)
    ap.add_argument("--min-dist-ms", type=float, default=0.3)
    ap.add_argument("--min-latency-ms", type=float, default=0.1)
    ap.add_argument("--max-latency-ms", type=float, default=3.0)
    ap.add_argument("--no-plots", action="store_true")
    args = ap.parse_args()

    root = Path(args.root)
    out_dir = Path(args.out_dir) if args.out_dir else (root / "analysis")
    out_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(root.glob("*/freq_*.h5"))
    if not files:
        files = sorted(root.glob("freq_*.h5"))
    print(f"Found {len(files)} HDF5 files under {root}")
    if not files:
        return

    spike_rows = []
    summary_rows = []

    for fp in files:
        with h5py.File(fp, "r") as f:
            attrs = parse_attrs(f)
            axons = sorted([k for k in f.keys() if k.startswith("Axon_")])
            if not axons:
                continue
            t = np.asarray(f[f"{axons[0]}/Model/time"], dtype=float)
            dt_ms = float(attrs.get("dt_ms", np.nan))
            if not np.isfinite(dt_ms) or dt_ms <= 0:
                dt_ms = float(np.median(np.diff(t))) if t.size > 1 else np.nan
            freq_hz = float(attrs.get("freq_hz", np.nan))
            fd = float(attrs.get("fiber_diameter_um", 4.5))
            ed = float(attrs.get("edge_dist_um", np.nan))
            t_start = float(attrs.get("t_start_ms", 10.0))
            t_end = float(attrs.get("t_end_ms", float(t[-1]) if t.size else 0.0))
            duration_s = max(0.0, (t_end - t_start) / 1000.0)
            # Число стимулов = floor((t_end - t_start) * freq / 1000) — как в STIMULATOR
            # (напр. 50 Гц, 990 мс -> 49 пульсов, а не 50).
            n_stimuli = int(freq_hz * duration_s) if (np.isfinite(freq_hz) and duration_s > 0) else 0

            # Максимальное окно матчинга не должно превышать ~80% периода стимуляции.
            period_ms = 1000.0 / freq_hz if (np.isfinite(freq_hz) and freq_hz > 0) else np.inf
            max_lat = float(min(args.max_latency_ms, 0.8 * period_ms)) if np.isfinite(period_ms) else args.max_latency_ms

            for axon in axons:
                # 1) собираем спайки по всем трейсам
                per_trace = {}
                for trace in TRACE_ORDER:
                    got = read_trace(f, axon, trace)
                    if got is None:
                        continue
                    tt, vv, dist, nd = got
                    idx, st, sv = detect_spikes(
                        tt, vv,
                        threshold_mv=args.threshold_mv,
                        prominence_mv=args.prominence_mv,
                        min_dist_ms=args.min_dist_ms,
                        start_ms=t_start,
                        dt_ms=dt_ms,
                    )
                    per_trace[trace] = {"t": st, "v": sv, "dist": dist, "node": nd}
                    for si, (s_t, s_v) in enumerate(zip(st, sv)):
                        spike_rows.append({
                            "h5": fp.name, "edge_dist_um": ed, "freq_hz": freq_hz, "axon": axon,
                            "trace": trace, "site": TRACE_SITE.get(trace, trace), "node": nd,
                            "spike_index": si, "spike_time_ms": float(s_t), "spike_amplitude_mV": float(s_v),
                        })

                # 2) reference = stimulation_point (или before_branch)
                ref_trace = "stimulation_point" if "stimulation_point" in per_trace else (
                    "before_branch" if "before_branch" in per_trace else None)
                ref = per_trace.get(ref_trace) if ref_trace else None
                ref_t = ref["t"] if ref else np.asarray([], dtype=float)
                ref_d = ref["dist"] if ref else np.nan
                n_ref = int(ref_t.size)

                for trace, d in per_trace.items():
                    sd = d["dist"]
                    matched_t, lat = match_causal(
                        ref_t, d["t"], min_latency_ms=args.min_latency_ms, max_latency_ms=max_lat,
                    ) if (n_ref > 0 and trace != ref_trace) else (np.asarray([], dtype=float), np.asarray([], dtype=float))

                    n_matched = int(matched_t.size) if trace != ref_trace else n_ref
                    path_um = (sd - ref_d) if (np.isfinite(sd) and np.isfinite(ref_d)) else np.nan
                    med_lat = float(np.median(lat)) if lat.size else np.nan
                    mean_lat = float(np.mean(lat)) if lat.size else np.nan
                    vel = (float(path_um) / med_lat * 1e-3) if (np.isfinite(path_um) and np.isfinite(med_lat) and med_lat > 0) else np.nan
                    # Отбрасываем физически неправдоподобные значения (ошибочный мэтчинг
                    # спайков на высоких частотах даёт крошечную латентность).
                    if np.isfinite(vel) and (vel <= 0.0 or vel > 150.0):
                        vel = float("nan")

                    summary_rows.append({
                        "h5": fp.name, "edge_dist_um": ed, "fiber_diameter_um": fd, "freq_hz": freq_hz,
                        "axon": axon, "is_center": int(str(axon) == "Axon_00"),
                        "trace": trace, "site": TRACE_SITE.get(trace, trace), "node": d["node"],
                        "n_spikes": int(d["t"].size),
                        "n_stimuli": n_stimuli,
                        "following_fraction": (float(d["t"].size) / n_stimuli) if n_stimuli > 0 else float("nan"),
                        "n_ref": n_ref,
                        "n_matched": n_matched,
                        "conduction_ratio": (float(n_matched) / n_ref) if (n_ref > 0) else float("nan"),
                        "median_latency_ms": med_lat,
                        "mean_latency_ms": mean_lat,
                        "path_length_um": float(path_um) if np.isfinite(path_um) else float("nan"),
                        "median_velocity_m_s": vel,
                    })

    # ---- CSV ----
    spikes_csv = out_dir / "spikes_all.csv"
    summary_csv = out_dir / "summary.csv"
    if spike_rows:
        with open(spikes_csv, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(spike_rows[0].keys()))
            w.writeheader()
            w.writerows(spike_rows)
    if summary_rows:
        with open(summary_csv, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(summary_rows[0].keys()))
            w.writeheader()
            w.writerows(summary_rows)
    print(f"Saved {spikes_csv} ({len(spike_rows)} rows)")
    print(f"Saved {summary_csv} ({len(summary_rows)} rows)")

    if args.no_plots or not summary_rows:
        return

    # ---- Plots ----
    edges = sorted({r["edge_dist_um"] for r in summary_rows if np.isfinite(r["edge_dist_um"])})
    center_sites = ["before", "branch", "after_main", "after_daughter", "terminal_main", "terminal_daughter"]

    def _series(site, ed, metric, *, is_center=1):
        fr, mv = [], []
        fr_list = sorted({r["freq_hz"] for r in summary_rows
                          if r["site"] == site and abs(r["edge_dist_um"] - ed) < 1e-9 and np.isfinite(r["freq_hz"])})
        for freq in fr_list:
            vals = [r[metric] for r in summary_rows
                    if r["site"] == site and int(r["is_center"]) == is_center
                    and abs(r["freq_hz"] - freq) < 1e-9 and abs(r["edge_dist_um"] - ed) < 1e-9
                    and np.isfinite(r[metric])]
            if vals:
                fr.append(freq); mv.append(float(np.median(vals)))
        return fr, mv

    for ed in edges:
        ed_tag = f"{ed:g}".replace(".", "p")
        fig, axes = plt.subplots(1, 3, figsize=(18, 5.5), dpi=160)

        # following fraction: центр + (пунктиром) соседи на terminal_main
        for site in center_sites:
            fr, mv = _series(site, ed, "following_fraction", is_center=1)
            if fr:
                axes[0].plot(fr, mv, marker="o", label=f"center:{site}")
        fr, mv = _series("terminal_main", ed, "following_fraction", is_center=0)
        if fr:
            axes[0].plot(fr, mv, marker="s", ls="--", color="gray", label="neighbors:terminal_main")
        axes[0].set_title(f"following fraction (edge={ed:g} um)")
        axes[0].set_xlabel("freq, Hz"); axes[0].set_ylabel("spikes / stimuli"); axes[0].set_ylim(-0.05, 1.15)
        axes[0].grid(alpha=0.25); _safe_legend(axes[0], fontsize=8)

        # latency (центр)
        for site in center_sites:
            fr, mv = _series(site, ed, "median_latency_ms", is_center=1)
            if fr:
                axes[1].plot(fr, mv, marker="o", label=site)
        axes[1].set_title(f"median latency (center, edge={ed:g} um)")
        axes[1].set_xlabel("freq, Hz"); axes[1].set_ylabel("latency, ms")
        axes[1].grid(alpha=0.25); _safe_legend(axes[1], fontsize=8)

        # velocity (центр, выбросы отфильтрованы)
        for site in center_sites:
            fr, mv = _series(site, ed, "median_velocity_m_s", is_center=1)
            if fr:
                axes[2].plot(fr, mv, marker="o", label=site)
        axes[2].set_title(f"median velocity (center, edge={ed:g} um)")
        axes[2].set_xlabel("freq, Hz"); axes[2].set_ylabel("velocity, m/s")
        axes[2].grid(alpha=0.25); _safe_legend(axes[2], fontsize=8)

        fig.tight_layout()
        p = out_dir / f"following_latency_velocity_ed{ed_tag}.png"
        fig.savefig(p, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved {p}")

    # Сводный график: following fraction на терминали центрального аксона по edge distance
    fig, ax = plt.subplots(figsize=(8, 5.5), dpi=160)
    for ed in edges:
        fr, mv = _series("terminal_main", ed, "following_fraction", is_center=1)
        if fr:
            ax.plot(fr, mv, marker="o", label=f"edge={ed:g} um")
    ax.set_title("following fraction: terminal_main (center axon)")
    ax.set_xlabel("freq, Hz"); ax.set_ylabel("spikes / stimuli"); ax.set_ylim(-0.05, 1.15)
    ax.grid(alpha=0.25); _safe_legend(ax)
    fig.tight_layout()
    p = out_dir / "following_terminal_by_edge.png"
    fig.savefig(p, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {p}")

    meta = {"n_files": len(files), "n_summary_rows": len(summary_rows), "n_spike_rows": len(spike_rows),
            "threshold_mv": args.threshold_mv, "prominence_mv": args.prominence_mv,
            "min_dist_ms": args.min_dist_ms}
    (out_dir / "analysis_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
