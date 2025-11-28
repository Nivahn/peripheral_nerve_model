from neuron import h
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

from MRG_lib import MRGaxon  # твоя библиотека


def attach_spike_train_stim(axon,
                            spike_times_ms,
                            amp_nA=1.0,
                            pulse_len_ms=1.0):
    """
    Создаёт стимуляцию в первом узле аксона в виде импульсов тока
    в моменты spike_times_ms.

    spike_times_ms: массив времени спайков в миллисекундах.
    amp_nA: амплитуда импульса (нА).
    pulse_len_ms: длительность каждого импульса (мс).
    """
    if len(spike_times_ms) == 0:
        return None

    # Если раньше уже создавали стимы, отключаем старые
    if not hasattr(axon, "_all_stims"):
        axon._all_stims = []
    else:
        for stim in axon._all_stims:
            stim.dur = 0.0  # выключаем старые IClamp

    stim = h.IClamp(axon.main_axon[0](0.5))
    stim.delay = 0.0
    # dur можно поставить долгую, всё равно форму задаёт play по amp
    stim.dur = float(max(spike_times_ms) + pulse_len_ms + 10.0)
    stim.amp = 0.0  # по умолчанию 0, дальше будет управляться через Vector.play

    # Строим "лестницу" для amp: t, amp, t+dur, 0, ...
    times = []
    amps = []
    for t in spike_times_ms:
        times.extend([t, t + pulse_len_ms])
        amps.extend([amp_nA, 0.0])

    t_vec = h.Vector(times)
    a_vec = h.Vector(amps)
    # шаг 1 (последний аргумент) — интерполяция на каждом шаге
    a_vec.play(stim._ref_amp, t_vec, 1)

    axon._all_stims.append(stim)
    return stim

def run_model_for_spike_train(axon,
                              spike_times_ms,
                              amp_nA=1.0,
                              pulse_len_ms=1.0,
                              threshold=-20.0,
                              min_peak_distance_ms=2.0):
    """
    Прогоняет один спайк-трейн через модель и возвращает времена спайков
    в ключевых точках аксона.

    Возвращает:
        {
          'before_branch': [np.array(spike_times_seg1), ...],
          'branch_point': [np.array(...), ...],
          'after_branch_main': [...],
          'after_branch_daughter': [...]
        }
    """
    # Если нет спайков — просто возвращаем пустые массивы
    if len(spike_times_ms) == 0:
        groups = {}
        groups['before_branch'] = [np.array([]) for _ in getattr(axon, 'before_branch_id', [])]
        groups['branch_point'] = [np.array([]) for _ in getattr(axon, 'branch_point_id', [])]
        groups['after_branch_main'] = [np.array([]) for _ in getattr(axon, 'after_branch_main_id', [])]
        groups['after_branch_daughter'] = [np.array([]) for _ in getattr(axon, 'after_branch_daughter_id', [])]
        return groups

    # Подготовка стимула
    attach_spike_train_stim(axon, spike_times_ms, amp_nA=amp_nA, pulse_len_ms=pulse_len_ms)

    # Словари с сегментами, которые надо записывать
    recording_groups = {
        'before_branch': getattr(axon, 'before_branch_id', []),
        'branch_point': getattr(axon, 'branch_point_id', []),
        'after_branch_main': getattr(axon, 'after_branch_main_id', []),
        'after_branch_daughter': getattr(axon, 'after_branch_daughter_id', []),
    }

    # Времена
    t_vec = h.Vector().record(h._ref_t)

    # Запись потенциалов
    rec = {}
    for group_name, seg_list in recording_groups.items():
        rec[group_name] = []
        for seg in seg_list:
            v_vec = h.Vector().record(seg._ref_v)
            rec[group_name].append(v_vec)

    # Настройка времени симуляции
    h.dt = axon.dt_ms
    h.tstop = float(max(spike_times_ms) + 50.0)
    h.finitialize(axon.v_init)
    h.run()

    time_array = np.array(t_vec)

    # Считаем спайки
    model_spike_times = {}
    for group_name, vec_list in rec.items():
        model_spike_times[group_name] = []
        for v_vec in vec_list:
            trace = np.array(v_vec)
            spike_count, spike_times = axon.count_spikes(
                trace,
                time_array=time_array,
                threshold=threshold,
                min_peak_distance=min_peak_distance_ms
            )
            model_spike_times[group_name].append(spike_times)

    return model_spike_times

def simulate_all_neurons(input_csv_path,
                         output_dir,
                         amp_nA=1.0,
                         pulse_len_ms=1.0):

    df = pd.read_csv(input_csv_path, header=None)
    n_neurons = df.shape[1]

    print(f"Входной файл: {input_csv_path}")
    print(f"Число нейронов: {n_neurons}")

    # ---- Ограничение 5 сек ----
    FIVE_SEC_SEC = 5.0
    FIVE_SEC_MS = FIVE_SEC_SEC * 1000.0

    # ---- Создаем аксон ----
    axon = MRGaxon(
        fiber_diameter=5.7,
        parent_axon_nodes=42,
        branch_nodes=21,
        branches_num=2,
        nodes_dist=10,
        diam_scale=0.6,
        celsius=37.0,
        dt_ms=0.05,
        v_init=-80.0,
        h_stop=5000.0     # Важно: 5 секунд
    )

    recording_groups = {
        'before_branch': getattr(axon, 'before_branch_id', []),
        'branch_point': getattr(axon, 'branch_point_id', []),
        'after_branch_main': getattr(axon, 'after_branch_main_id', []),
        'after_branch_daughter': getattr(axon, 'after_branch_daughter_id', []),
    }

    for name, seg_list in recording_groups.items():
        print(f"{name}: {len(seg_list)} сегментов")

    # ---- структура результатов ----
    results = {}
    for group_name, seg_list in recording_groups.items():
        results[group_name] = [
            [None for _ in range(n_neurons)]
            for _ in range(len(seg_list))
        ]

    # ---- цикл по нейронам ----
    for neuron_idx in range(n_neurons):

        spike_times_sec = df[neuron_idx].dropna().values

        # --- фильтрация первых 5 сек ---
        spike_times_sec = spike_times_sec[spike_times_sec <= FIVE_SEC_SEC]
        spike_times_ms = spike_times_sec * 1000.0

        print(f"\nНейрон {neuron_idx+1}: {len(spike_times_ms)} входных спайков "
              f"(0–5 сек)")

        model_spike_times = run_model_for_spike_train(
            axon,
            spike_times_ms,
            amp_nA=amp_nA,
            pulse_len_ms=pulse_len_ms
        )

        # ---- сохраняем ----
        for group_name, seg_list in recording_groups.items():
            for seg_idx, spikes_arr in enumerate(model_spike_times[group_name]):
                results[group_name][seg_idx][neuron_idx] = spikes_arr

    # ---- сохраняем CSV ----
    os.makedirs(output_dir, exist_ok=True)

    for group_name, seg_list in recording_groups.items():
        n_segments = len(seg_list)
        for seg_idx in range(n_segments):

            neuron_spike_lists = results[group_name][seg_idx]
            max_len = max(len(arr) for arr in neuron_spike_lists)

            if max_len == 0:
                data = np.empty((0, n_neurons))
            else:
                data = np.full((max_len, n_neurons), np.nan)
                for i, arr in enumerate(neuron_spike_lists):
                    if arr is None or len(arr) == 0:
                        continue
                    data[:len(arr), i] = arr

            out_df = pd.DataFrame(data,
                                  columns=[f'Neuron_{i+1}' for i in range(n_neurons)])

            out_path = os.path.join(
                output_dir,
                f"ModelSpikes_{group_name}_seg{seg_idx+1}.csv"
            )

            out_df.to_csv(out_path, header=False, index=False)
            print(f"Сохранено: {out_path}")


input_csv = r"Data\SCS_LTMRs_1kHz_Timestamps.csv"
output_dir = r"Data\MRG_model_spikes"

simulate_all_neurons(
    input_csv_path=input_csv,
    output_dir=output_dir,
    amp_nA=1.0,
    pulse_len_ms=1.0
)