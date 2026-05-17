import numpy as np
import pandas as pd
import h5py as h5
import re
from pathlib import Path
from scipy.signal import find_peaks


DEFAULT_START_ANALYSIS_MS = 0.0
DEFAULT_PEAK_PROMINENCE_MV = 5.0
DEFAULT_PEAK_MIN_DISTANCE_MS = 0.6
DEFAULT_SPIKE_HEIGHT_THRESHOLDS_MV_BY_DIAMETER = {2.5: -20.0, 5.7: -20.0}
DEFAULT_SPIKE_HEIGHT_MV = -20.0

DEFAULT_TRACE_ROLE_MAP = {
    "AxonA": {
        "before_like": "before",
        "main_like": "after_main",
        "before_branch": "before_branch",
        "branch_point": "branch_point",
        "after_branch_main": "after_main",
        "terminal_main": "terminal_main",
    },
    "AxonB": {
        "before_branch": "before",
        "branch_point": "branch_point",
        "after_branch_main": "after_main",
        "terminal_main": "terminal_main",
    },
}

NODE_STEP_UM_BY_DIAMETER = {5.7: 500.0, 2.5: 250.0}
STIM_DURATION_MS = 1000.0
MIN_LATENCY_MS = 0.0
MAX_LATENCY_MS = 8.0

TRACE_PAIRS = {
    "AxonA": {"before": "main_like", "after": "terminal_main", "title": "AxonA"},
    "AxonB": {"before": "after_branch_main", "after": "terminal_main", "title": "AxonB"},
}

MULTIPLE_BRANCHES_TRACE_PAIRS = {
    "AxonA": {"before": "after_branch_main", "after": "terminal_main", "title": "AxonA"},
    "AxonB": {"before": "after_branch_main", "after": "terminal_main", "title": "AxonB"},
}


def trace_pairs_for_scenario(scenario: str) -> dict:
    if str(scenario) == "multiple_branches":
        return MULTIPLE_BRANCHES_TRACE_PAIRS
    return TRACE_PAIRS




# ======================================================================================================================
# Проход по .h5 файлам
# ======================================================================================================================

def iter_h5_files(root: str | Path):
    """
    Рекурсивно найти все .h5 файлы в папке root и её подпапках.
    """
    root = Path(root)

    if not root.exists():
        raise FileNotFoundError(f"Папка не найдена: {root}")

    if not root.is_dir():
        raise NotADirectoryError(f"Это не папка: {root}")

    yield from sorted(root.glob("**/*.h5"))


def read_h5_attrs(h5_path: str | Path) -> dict:
    """
    Прочитать основные attrs из одного .h5 файла.
    """
    h5_path = Path(h5_path)

    with h5.File(h5_path, "r") as f:
        attrs = {
            "h5_file": str(h5_path),
            "h5_name": h5_path.name,

            # основные параметры файла
            "topology": str(f.attrs.get("topology", "")),
            "scenario": str(f.attrs.get("scenario", "")),
            "fiber_diameter_um": float(f.attrs.get("fiber_diameter_um", np.nan)),
            "edge_dist_um": float(f.attrs.get("edge_dist_um", np.nan)),
            "mode": str(f.attrs.get("mode", "")),
            "test_mode": int(f.attrs.get("test_mode", 0)),

            # дополнительные параметры, если есть
            "amp": float(f.attrs.get("amp_nA", np.nan)) if "amp_nA" in f.attrs else np.nan,
            "h_stop": float(f.attrs.get("h_stop_ms", np.nan)) if "h_stop_ms" in f.attrs else np.nan,
        }

    return attrs


def collect_h5_index(root: str | Path) -> pd.DataFrame:
    """
    Собрать таблицу всех .h5 файлов и их метаданных из f.attrs.
    """
    rows = []

    for h5_path in iter_h5_files(root):
        try:
            row = read_h5_attrs(h5_path)
            row["read_ok"] = True
            row["read_error"] = ""
        except Exception as exc:
            row = {
                "h5_file": str(h5_path),
                "h5_name": Path(h5_path).name,
                "topology": "",
                "scenario": "",
                "fiber_diameter_um": np.nan,
                "edge_dist_um": np.nan,
                "mode": "",
                "test_mode": np.nan,
                "amp": np.nan,
                "h_stop": np.nan,
                "read_ok": False,
                "read_error": str(exc),
            }

        rows.append(row)

    return pd.DataFrame(rows)


# ======================================================================================================================
# Детекция спайков
# ======================================================================================================================


# ============================================================
# ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
# ============================================================


def node_index(node_name: str) -> int:
    """
    Достаёт номер узла из имени типа node_12_0.50.
    Если номер не найден, отправляет такой узел в конец сортировки.
    """
    match = re.search(r"node_(\d+)", str(node_name))
    if match is None:
        return 10**9
    return int(match.group(1))


def frequency_from_group_name(group_name: str) -> float:
    """
    Достаёт частоту из имени группы Frequency_450Hz.
    """
    match = re.search(r"Frequency_(\d+(?:\.\d+)?)Hz", str(group_name))
    if match is None:
        raise ValueError(f"Не могу достать частоту из группы: {group_name}")
    return float(match.group(1))


def stim_pulse_count_from_group(freq_group, axon_name: str = "AxonA") -> int:
    """
    Counts first-phase pulse starts from Stimulator/current.
    Biphasic second phases are not counted as extra pulses.
    """
    stim_path = f"{axon_name}/Stimulator"
    if stim_path not in freq_group:
        return 0
    stim = freq_group[stim_path]
    if "time" not in stim or "current" not in stim:
        return 0

    current = np.asarray(stim["current"], dtype=float)
    if current.size == 0:
        return 0

    max_abs = float(np.nanmax(np.abs(current)))
    if not np.isfinite(max_abs) or max_abs <= 0.0:
        return 0

    active = np.abs(current) > max(1e-12, 1e-6 * max_abs)
    starts = np.flatnonzero(active & np.r_[True, ~active[:-1]])
    if starts.size == 0:
        return 0

    first_phase_sign = float(np.sign(current[starts[0]]))
    if first_phase_sign == 0.0:
        return int(starts.size)
    return int(np.sum(np.sign(current[starts]) == first_phase_sign))


def spike_height_threshold_mV(
    fiber_diameter_um,
    thresholds_by_diameter=None,
    default_threshold_mV=DEFAULT_SPIKE_HEIGHT_MV,
):
    """
    Выбирает порог детекции спайка по диаметру волокна.
    """
    d = float(fiber_diameter_um)
    thresholds_by_diameter = thresholds_by_diameter or DEFAULT_SPIKE_HEIGHT_THRESHOLDS_MV_BY_DIAMETER

    nearest_d = min(
        thresholds_by_diameter.keys(),
        key=lambda ref_d: abs(float(ref_d) - d),
    )

    return float(thresholds_by_diameter.get(nearest_d, default_threshold_mV))

# ======================================================================================================================
# Детекция спайков (ОСНОВА)
# ======================================================================================================================

def detect_spikes(t_ms,v_mV,
                  threshold_mV,dt_ms,
                  START_ANALYSIS_MS, PEAK_MIN_DISTANCE_MS,PEAK_PROMINENCE_MV
                  ):
    """
    Детекция спайков по voltage trace.

    Возвращает:
    - peak_indices
    - peak_times_ms
    - peak_amplitudes_mV
    """
    t_ms = np.asarray(t_ms, dtype=float)
    v_mV = np.asarray(v_mV, dtype=float)

    if t_ms.size < 2 or v_mV.size != t_ms.size:
        return (
            np.array([], dtype=int),
            np.array([], dtype=float),
            np.array([], dtype=float),
        )

    if not np.isfinite(dt_ms) or dt_ms <= 0:
        return (
            np.array([], dtype=int),
            np.array([], dtype=float),
            np.array([], dtype=float),
        )

    start_idx = int(np.searchsorted(t_ms, START_ANALYSIS_MS, side="left"))
    min_dist_pts = max(1, int(round(PEAK_MIN_DISTANCE_MS / dt_ms)))

    peaks_local, props = find_peaks(
        v_mV[start_idx:],
        height=float(threshold_mV),
        prominence=float(PEAK_PROMINENCE_MV),
        distance=min_dist_pts,
    )

    peak_indices = peaks_local + start_idx
    peak_times_ms = t_ms[peak_indices]
    peak_amplitudes_mV = v_mV[peak_indices]

    return peak_indices, peak_times_ms, peak_amplitudes_mV


def read_basic_h5_attrs(h5_path, f):
    """
    Читает основные метаданные из f.attrs.
    Если какого-то атрибута нет, ставит пустую строку или NaN.
    """
    h5_path = Path(h5_path)

    return {
        "h5_file": str(h5_path),
        "h5_name": h5_path.name,
        "topology": str(f.attrs.get("topology", "")),
        "scenario": str(f.attrs.get("scenario", "")),
        "fiber_diameter_um": float(f.attrs.get("fiber_diameter_um", np.nan)),
        "edge_dist_um": float(f.attrs.get("edge_dist_um", np.nan)),
        "mode": str(f.attrs.get("mode", "")),
        "test_mode": int(f.attrs.get("test_mode", 0)),
    }


def export_spikes_from_h5_file(h5_path,
                               trace_role_map=None,
                               start_analysis_ms=DEFAULT_START_ANALYSIS_MS,
                               peak_min_distance_ms=DEFAULT_PEAK_MIN_DISTANCE_MS,
                               peak_prominence_mV=DEFAULT_PEAK_PROMINENCE_MV,
                               spike_thresholds_by_diameter=None,
                               default_spike_height_mV=DEFAULT_SPIKE_HEIGHT_MV,
                               ):
    """
    Обрабатывает один .h5 файл и возвращает:
    - rows: строки со спайками
    - audit_rows: краткую диагностику по обработке файла
    """
    h5_path = Path(h5_path)

    rows = []
    audit_rows = []
    trace_role_map = trace_role_map or DEFAULT_TRACE_ROLE_MAP

    with h5.File(h5_path, "r") as f:
        # Базовые метаданные
        base_meta = read_basic_h5_attrs(h5_path, f)
        # В зависимости от диаметра берём порог для спайков
        threshold_mV = spike_height_threshold_mV(
            base_meta["fiber_diameter_um"],
            thresholds_by_diameter=spike_thresholds_by_diameter,
            default_threshold_mV=default_spike_height_mV,
        )
        # Сортируем группы частот (поскольку для алгоритмов не всё так просто
        freq_group_names = sorted(
            [name for name in f.keys() if str(name).startswith("Frequency_")],
            key=frequency_from_group_name,
        )
        dt_ms = f.attrs.get("dt_ms")

        # Проходимся по частотам
        for freq_group_name in freq_group_names:
            freq_hz = frequency_from_group_name(freq_group_name)
            stim_pulse_count = stim_pulse_count_from_group(f[freq_group_name], "AxonA")
            # Проходимся по аксонам и выбранным частотам
            for axon_name, trace_map in trace_role_map.items():
                time_path = f"{freq_group_name}/{axon_name}/Model/time"
                traces_path = f"{freq_group_name}/{axon_name}/Model/Traces"

                t_ms = np.asarray(f[time_path][:], dtype=float)
                traces_group = f[traces_path]
                # Проходимся по выбранным заранее точкам регистрации потенциала
                for trace_name, trace_role in trace_map.items():
                    if trace_name not in traces_group:
                        audit_rows.append({
                            **base_meta,
                            "freq_hz": freq_hz,
                            "stim_pulse_count": int(stim_pulse_count),
                            "axon": axon_name,
                            "trace": trace_name,
                            "trace_role": trace_role,
                            "node": "",
                            "status": "missing_trace",
                            "n_spikes": 0,
                            "threshold_mV": float(threshold_mV),
                        })
                        continue
                    # Сортируем ноды
                    nodes_group = traces_group[trace_name]
                    node_names = sorted(list(nodes_group.keys()), key=node_index)
                    if not node_names:
                        audit_rows.append({
                            **base_meta,
                            "freq_hz": freq_hz,
                            "stim_pulse_count": int(stim_pulse_count),
                            "axon": axon_name,
                            "trace": trace_name,
                            "trace_role": trace_role,
                            "node": "",
                            "status": "missing_node",
                            "n_spikes": 0,
                            "threshold_mV": float(threshold_mV),
                        })
                        continue

                    # Простой вариант: берём первый узел в каждой trace-группе.
                    node_name = node_names[0]
                    voltage_path = f"{traces_path}/{trace_name}/{node_name}/voltage"

                    v_mV = np.asarray(f[voltage_path][:], dtype=float)
                    # Собственно детектим спайки
                    peak_indices, peak_times_ms, peak_amplitudes_mV = detect_spikes(
                        t_ms,
                        v_mV,
                        threshold_mV,dt_ms,
                        start_analysis_ms, peak_min_distance_ms, peak_prominence_mV
                    )

                    for spike_i, (peak_idx, peak_time, peak_amp) in enumerate(
                        zip(peak_indices, peak_times_ms, peak_amplitudes_mV)
                    ):
                        rows.append({
                            **base_meta,
                            "freq_hz": freq_hz,
                            "stim_pulse_count": int(stim_pulse_count),
                            "axon": axon_name,
                            "trace": trace_name,
                            "trace_role": trace_role,
                            "node": node_name,
                            "spike_index": int(spike_i),
                            "peak_index": int(peak_idx),
                            "spike_time_ms": float(peak_time),
                            "spike_amplitude_mV": float(peak_amp),
                            "threshold_mV": float(threshold_mV),
                        })

                    audit_rows.append({
                        **base_meta,
                        "freq_hz": freq_hz,
                        "stim_pulse_count": int(stim_pulse_count),
                        "axon": axon_name,
                        "trace": trace_name,
                        "trace_role": trace_role,
                        "node": node_name,
                        "status": "ok",
                        "n_spikes": int(len(peak_times_ms)),
                        "threshold_mV": float(threshold_mV),
                    })

    return rows, audit_rows


def export_all_spikes_to_csv(root,
                             out_spikes_csv,
                             out_audit_csv,
                             trace_role_map=None,
                             start_analysis_ms=DEFAULT_START_ANALYSIS_MS,
                             peak_min_distance_ms=DEFAULT_PEAK_MIN_DISTANCE_MS,
                             peak_prominence_mV=DEFAULT_PEAK_PROMINENCE_MV,
                             spike_thresholds_by_diameter=None,
                             default_spike_height_mV=DEFAULT_SPIKE_HEIGHT_MV):
    """
    Проходит по всем .h5 файлам, детектирует спайки и сохраняет CSV.
    """
    all_rows = []
    all_audit_rows = []

    h5_files = list(iter_h5_files(root))

    print(f"[INFO] Найдено .h5 файлов: {len(h5_files)}")

    for i, h5_path in enumerate(h5_files, start=1):
        print(f"[{i}/{len(h5_files)}] {h5_path.name}")

        try:
            rows, audit_rows = export_spikes_from_h5_file(
                h5_path,
                trace_role_map=trace_role_map,
                start_analysis_ms=start_analysis_ms,
                peak_min_distance_ms=peak_min_distance_ms,
                peak_prominence_mV=peak_prominence_mV,
                spike_thresholds_by_diameter=spike_thresholds_by_diameter,
                default_spike_height_mV=default_spike_height_mV,
            )
            all_rows.extend(rows)
            all_audit_rows.extend(audit_rows)

        except Exception as exc:
            all_audit_rows.append({
                "h5_file": str(h5_path),
                "h5_name": h5_path.name,
                "status": "error",
                "error": str(exc),
            })
            print(f"[ERROR] {h5_path.name}: {exc}")

    spikes_df = pd.DataFrame(all_rows)
    audit_df = pd.DataFrame(all_audit_rows)

    out_spikes_csv = Path(out_spikes_csv)
    out_spikes_csv.parent.mkdir(parents=True, exist_ok=True)
    spikes_df.to_csv(out_spikes_csv, index=False)

    if out_audit_csv is not None:
        out_audit_csv = Path(out_audit_csv)
        out_audit_csv.parent.mkdir(parents=True, exist_ok=True)
        audit_df.to_csv(out_audit_csv, index=False)

    print(f"[DONE] spikes CSV: {out_spikes_csv}")
    if out_audit_csv is not None:
        print(f"[DONE] audit CSV: {out_audit_csv}")

    return spikes_df, audit_df


# ======================================================================================================================
# Общие helpers для графиков и расчёта метрик
# ======================================================================================================================


def normalize_mode_name(mode: str) -> str:
    mode = str(mode)
    replacements = {
        "no_ec": "no_EC",
        "no_EC": "no_EC",
        "noec": "no_EC",
        "no_ephaptic": "no_EC",
        "no_ephaptic_coupling": "no_EC",
        "no_ec_isolated": "no_EC_isolated",
        "no_EC_isolated": "no_EC_isolated",
    }
    return replacements.get(mode, mode)


def baseline_suffix(mode: str) -> str:
    return normalize_mode_name(mode).replace("_EC", "_ec").lower()


def prepare_spikes_df(spikes_df: pd.DataFrame) -> pd.DataFrame:
    df = spikes_df.copy()
    df["h5_file"] = df["h5_file"].astype(str)
    if "h5_name" in df.columns:
        df["h5_name"] = df["h5_name"].astype(str)
    else:
        df["h5_name"] = df["h5_file"].apply(lambda p: Path(str(p)).name)
    df["axon"] = df["axon"].astype(str)
    df["trace"] = df["trace"].astype(str)
    df["node"] = df["node"].astype(str)
    df["mode"] = df["mode"].astype(str)
    df["mode_norm"] = df["mode"].apply(normalize_mode_name)
    df["freq_hz"] = df["freq_hz"].astype(float)
    if "stim_pulse_count" in df.columns:
        df["stim_pulse_count"] = df["stim_pulse_count"].astype(float)
    df["fiber_diameter_um"] = df["fiber_diameter_um"].astype(float)
    df["edge_dist_um"] = df["edge_dist_um"].astype(float)
    df["spike_time_ms"] = df["spike_time_ms"].astype(float)
    df["spike_amplitude_mV"] = df["spike_amplitude_mV"].astype(float)
    if "topology" not in df.columns:
        df["topology"] = "unknown"
    if "scenario" not in df.columns:
        df["scenario"] = "unknown"
    return df


def print_available_conditions(spikes_df: pd.DataFrame):
    df = prepare_spikes_df(spikes_df)
    print("Доступные диаметры:")
    print(sorted(df["fiber_diameter_um"].dropna().unique()))
    print("\nДоступные расстояния edge_dist_um:")
    print(sorted(df["edge_dist_um"].dropna().unique()))
    print("\nДоступные режимы mode:")
    for mode in sorted(df["mode"].dropna().unique()):
        print(" -", mode)
    print("\nДоступные частоты:")
    print(sorted(df["freq_hz"].dropna().unique())[:30], "...")


def find_frequency_group_name(f: h5.File, freq_hz: float) -> str:
    candidates = [name for name in f.keys() if str(name).startswith("Frequency_")]
    if not candidates:
        raise KeyError("В файле нет групп Frequency_...Hz")
    for name in candidates:
        try:
            current_freq = frequency_from_group_name(name)
        except ValueError:
            continue
        if np.isclose(current_freq, float(freq_hz)):
            return str(name)
    available = []
    for name in candidates:
        try:
            available.append(f"{name} -> {frequency_from_group_name(name)} Hz")
        except Exception:
            available.append(str(name))
    raise KeyError(f"Не найдена группа для freq_hz={freq_hz}. Доступные группы: {available}")


def load_voltage_trace_first_node(h5_path: str | Path, *, freq_hz: float, axon: str, trace: str):
    h5_path = Path(h5_path)
    with h5.File(h5_path, "r") as f:
        freq_group_name = find_frequency_group_name(f, freq_hz)
        time_path = f"{freq_group_name}/{axon}/Model/time"
        traces_path = f"{freq_group_name}/{axon}/Model/Traces"
        if traces_path not in f:
            raise KeyError(f"Не найден traces_path: {traces_path}")
        traces_group = f[traces_path]
        if trace not in traces_group:
            raise KeyError(f"Trace {trace!r} не найден. Доступные trace: {list(traces_group.keys())}")
        nodes_group = traces_group[trace]
        node_names = sorted(list(nodes_group.keys()), key=node_index)
        if not node_names:
            raise KeyError(f"В trace {trace!r} нет node-групп")
        node = node_names[0]
        t_ms = np.asarray(f[time_path][:], dtype=float)
        v_mV = np.asarray(nodes_group[node]["voltage"][:], dtype=float)
    return t_ms, v_mV, str(node), str(freq_group_name)


def get_detected_spikes_for_trace(
    spikes_df: pd.DataFrame,
    *,
    h5_file: str | Path,
    freq_hz: float,
    axon: str,
    trace: str,
    node: str,
    t_min_ms: float,
    t_max_ms: float,
) -> pd.DataFrame:
    h5_file = str(h5_file)
    sub = spikes_df[
        (spikes_df["h5_file"].astype(str) == h5_file)
        & (spikes_df["freq_hz"].astype(float) == float(freq_hz))
        & (spikes_df["axon"].astype(str) == str(axon))
        & (spikes_df["trace"].astype(str) == str(trace))
        & (spikes_df["node"].astype(str) == str(node))
    ].copy()
    return sub[
        (sub["spike_time_ms"].astype(float) >= float(t_min_ms))
        & (sub["spike_time_ms"].astype(float) <= float(t_max_ms))
    ].copy()


def select_h5_file_for_condition(
    spikes_df: pd.DataFrame,
    *,
    fiber_diameter_um: float,
    edge_dist_um: float,
    mode: str,
    freq_hz: float,
    topology: str | None = None,
    scenario: str | None = None,
    h5_name_contains: str | None = None,
) -> str:
    df = prepare_spikes_df(spikes_df)
    mode_norm = normalize_mode_name(mode)
    sub = df[
        (np.isclose(df["fiber_diameter_um"], float(fiber_diameter_um)))
        & (np.isclose(df["edge_dist_um"], float(edge_dist_um)))
        & (df["mode_norm"] == mode_norm)
        & (np.isclose(df["freq_hz"], float(freq_hz)))
    ].copy()
    if topology is not None and "topology" in sub.columns:
        sub = sub[sub["topology"].astype(str) == str(topology)]
    if scenario is not None and "scenario" in sub.columns:
        sub = sub[sub["scenario"].astype(str) == str(scenario)]
    if h5_name_contains is not None:
        sub = sub[sub["h5_name"].astype(str).str.contains(str(h5_name_contains), regex=False)]
    if sub.empty:
        raise ValueError(
            "Не найден файл для условий:\n"
            f"fiber_diameter_um={fiber_diameter_um}, edge_dist_um={edge_dist_um}, "
            f"mode={mode}, freq_hz={freq_hz}, topology={topology}, "
            f"scenario={scenario}, h5_name_contains={h5_name_contains}"
        )
    files = sub[["h5_file", "h5_name"]].drop_duplicates().sort_values("h5_name").reset_index(drop=True)
    if len(files) > 1:
        print("[INFO] Найдено несколько файлов, беру первый:")
        print(files.to_string(index=False))
    return str(files.loc[0, "h5_file"])


def nearest_node_step_um(fiber_diameter_um: float) -> float:
    d = float(fiber_diameter_um)
    nearest_d = min(NODE_STEP_UM_BY_DIAMETER.keys(), key=lambda ref_d: abs(float(ref_d) - d))
    return float(NODE_STEP_UM_BY_DIAMETER[nearest_d])


def path_length_from_nodes_um(before_node: str, after_node: str, fiber_diameter_um: float, *, min_node_intervals: int = 1) -> float:
    before_i = node_index(before_node)
    after_i = node_index(after_node)
    if before_i >= 10**9 or after_i >= 10**9:
        node_intervals = int(min_node_intervals)
    else:
        node_intervals = max(abs(int(after_i) - int(before_i)), int(min_node_intervals))
    return float(node_intervals) * nearest_node_step_um(fiber_diameter_um)


def velocity_m_s(path_length_um: float, latency_ms: float) -> float:
    if not np.isfinite(path_length_um):
        return np.nan
    if not np.isfinite(latency_ms) or latency_ms <= 0:
        return np.nan
    return float(path_length_um) / float(latency_ms) * 0.001


def expected_stim_count(freq_hz: float, stim_duration_ms: float = STIM_DURATION_MS) -> int:
    return int(round(float(freq_hz) * float(stim_duration_ms) / 1000.0))


def choose_first_node_spikes(df_trace: pd.DataFrame) -> pd.DataFrame:
    if df_trace.empty:
        return df_trace
    tmp = df_trace.copy()
    tmp["node_order"] = tmp["node"].astype(str).apply(node_index)
    first_node = tmp[["node", "node_order"]].drop_duplicates().sort_values("node_order").iloc[0]["node"]
    return df_trace[df_trace["node"] == first_node].copy()


def match_spikes_causal(
    before_times_ms: np.ndarray,
    after_times_ms: np.ndarray,
    *,
    min_latency_ms: float = MIN_LATENCY_MS,
    max_latency_ms: float = MAX_LATENCY_MS,
) -> pd.DataFrame:
    before_times_ms = np.asarray(before_times_ms, dtype=float)
    after_times_ms = np.asarray(after_times_ms, dtype=float)
    before_times_ms = np.sort(before_times_ms[np.isfinite(before_times_ms)])
    after_times_ms = np.sort(after_times_ms[np.isfinite(after_times_ms)])

    rows = []
    after_j = 0
    for before_i, before_t in enumerate(before_times_ms):
        left = before_t + float(min_latency_ms)
        right = before_t + float(max_latency_ms)
        while after_j < len(after_times_ms) and after_times_ms[after_j] < left:
            after_j += 1
        if after_j < len(after_times_ms) and after_times_ms[after_j] <= right:
            after_t = after_times_ms[after_j]
            rows.append({"before_index": int(before_i), "before_time_ms": float(before_t), "after_time_ms": float(after_t), "latency_ms": float(after_t - before_t), "matched": True})
            after_j += 1
        else:
            rows.append({"before_index": int(before_i), "before_time_ms": float(before_t), "after_time_ms": np.nan, "latency_ms": np.nan, "matched": False})
    return pd.DataFrame(rows)


def build_before_after_metrics(spikes_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = prepare_spikes_df(spikes_df)
    matched_rows = []
    summary_rows = []
    group_cols = ["h5_file", "h5_name", "topology", "scenario", "fiber_diameter_um", "edge_dist_um", "mode", "mode_norm", "freq_hz"]

    for keys, g in df.groupby(group_cols, dropna=False):
        meta = dict(zip(group_cols, keys))
        fiber_diameter_um = float(meta["fiber_diameter_um"])
        freq_hz = float(meta["freq_hz"])
        if "stim_pulse_count" in g.columns and g["stim_pulse_count"].notna().any():
            n_stimuli = int(g["stim_pulse_count"].dropna().iloc[0])
        else:
            n_stimuli = expected_stim_count(freq_hz)

        for axon, spec in trace_pairs_for_scenario(meta["scenario"]).items():
            before_trace = spec["before"]
            after_trace = spec["after"]
            before_df = choose_first_node_spikes(g[(g["axon"] == axon) & (g["trace"] == before_trace)].copy())
            after_df = choose_first_node_spikes(g[(g["axon"] == axon) & (g["trace"] == after_trace)].copy())
            before_times = before_df["spike_time_ms"].to_numpy(dtype=float)
            after_times = after_df["spike_time_ms"].to_numpy(dtype=float)
            before_node = str(before_df["node"].iloc[0]) if not before_df.empty else ""
            after_node = str(after_df["node"].iloc[0]) if not after_df.empty else ""
            path_length_um = path_length_from_nodes_um(before_node, after_node, fiber_diameter_um, min_node_intervals=1)
            matches = match_spikes_causal(before_times, after_times)
            matches["velocity_m_s"] = matches["latency_ms"].apply(lambda x: velocity_m_s(path_length_um, x)) if not matches.empty else pd.Series(dtype=float)

            n_before = int(len(before_times))
            n_after = int(len(after_times))
            n_matched = int(matches["matched"].sum()) if not matches.empty else 0
            matched_only = matches[matches["matched"]].copy() if not matches.empty else pd.DataFrame()

            for _, row in matches.iterrows():
                matched_rows.append({**meta, "axon": axon, "before_trace": before_trace, "after_trace": after_trace, "before_node": before_node, "after_node": after_node, "path_length_um": path_length_um, **row.to_dict()})

            summary_rows.append({
                **meta,
                "axon": axon,
                "before_trace": before_trace,
                "after_trace": after_trace,
                "source_trace": before_trace,
                "terminal_trace": after_trace,
                "before_node": before_node,
                "after_node": after_node,
                "source_node": before_node,
                "terminal_node": after_node,
                "path_length_um": path_length_um,
                "n_stimuli": n_stimuli,
                "n_before": n_before,
                "n_after": n_after,
                "n_source": n_before,
                "n_terminal": n_after,
                "n_matched": n_matched,
                "following_fraction_before": float(n_before) / n_stimuli if n_stimuli > 0 else np.nan,
                "following_fraction_after": float(n_after) / n_stimuli if n_stimuli > 0 else np.nan,
                "following_fraction_terminal": float(n_after) / n_stimuli if n_stimuli > 0 else np.nan,
                "median_latency_ms": float(matched_only["latency_ms"].median()) if not matched_only.empty else np.nan,
                "mean_latency_ms": float(matched_only["latency_ms"].mean()) if not matched_only.empty else np.nan,
                "median_velocity_m_s": float(matched_only["velocity_m_s"].median()) if not matched_only.empty else np.nan,
                "mean_velocity_m_s": float(matched_only["velocity_m_s"].mean()) if not matched_only.empty else np.nan,
                "median_terminal_latency_ms": float(matched_only["latency_ms"].median()) if not matched_only.empty else np.nan,
                "mean_terminal_latency_ms": float(matched_only["latency_ms"].mean()) if not matched_only.empty else np.nan,
                "median_terminal_velocity_m_s": float(matched_only["velocity_m_s"].median()) if not matched_only.empty else np.nan,
                "mean_terminal_velocity_m_s": float(matched_only["velocity_m_s"].mean()) if not matched_only.empty else np.nan,
            })

    return pd.DataFrame(matched_rows), pd.DataFrame(summary_rows)


def build_no_ec_delta_summary(
    summary_df: pd.DataFrame,
    *,
    metric_cols: list[str] | None = None,
    baseline_mode: str = "no_EC_isolated",
) -> pd.DataFrame:
    """
    Adds absolute deltas versus baseline_mode for each metric_col.
    Delta = current mode metric - baseline metric for same config/freq/axon.
    """
    df = summary_df.copy()
    if "mode_norm" not in df.columns:
        df["mode_norm"] = df["mode"].astype(str).apply(normalize_mode_name)
    else:
        df["mode_norm"] = df["mode_norm"].astype(str).apply(normalize_mode_name)

    metric_cols = metric_cols or [
        "following_fraction_after",
        "following_fraction_terminal",
        "median_velocity_m_s",
        "median_terminal_velocity_m_s",
        "median_latency_ms",
        "median_terminal_latency_ms",
    ]

    key_cols = [
        "topology",
        "scenario",
        "fiber_diameter_um",
        "edge_dist_um",
        "freq_hz",
        "axon",
    ]
    key_cols = [col for col in key_cols if col in df.columns]

    suffix = baseline_suffix(baseline_mode)
    baseline_cols = key_cols + [col for col in metric_cols if col in df.columns]
    baseline = df[df["mode_norm"] == normalize_mode_name(baseline_mode)][baseline_cols].copy()
    baseline = baseline.rename(columns={col: f"{col}_{suffix}" for col in metric_cols if col in baseline.columns})

    out = df.merge(baseline, on=key_cols, how="left")
    for col in metric_cols:
        base_col = f"{col}_{suffix}"
        if col in out.columns and base_col in out.columns:
            out[f"delta_{col}_vs_{suffix}"] = out[col] - out[base_col]
    return out



