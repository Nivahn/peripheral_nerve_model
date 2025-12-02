from neuron import h
import matplotlib.pyplot as plt
import numpy as np
import math
import time
import os
import pandas as pd
from scipy.signal import find_peaks
from MRG_lib import *


import h5py
import numpy as np
from impulse_generator import STIMULATOR
from MRG_lib import MRGaxon


# --------------------------
# Настройки эксперимента
# --------------------------

frequencies = list(range(50, 1001, 50))   # 50,150,...,950,1000 Гц
stimulation_duration_ms = 5000.0           # 1 секунда
amp_nA = 1.0                               # амплитуда импульса
ton_ms = 1.0                               # длительность импульса
dt_stim = 0.05                             # dt стимула
dt_model = 0.05                            # dt модели
v_init = -80.0
h_stop_ms = 5010.0                         # после стимуляции ещё хвост


# --------------------------
# Создаём аксон
# --------------------------

axon = MRGaxon(
    fiber_diameter=5.7,
    parent_axon_nodes=42,
    branch_nodes=21,
    branches_num=2,
    nodes_dist=10,
    diam_scale=0.6,
    celsius=37.0,
    dt_ms=dt_model,
    v_init=v_init,
    h_stop=h_stop_ms
)

# --------------------------
# Создаём выходной HDF5
# --------------------------
h5_path = "./../../../Data/MRG_MultiFreq_Stim_50_1000Hz.h5"

f = h5py.File(h5_path, "w")

# Глобальные метаданные
f.attrs["experiment_name"] = "MRG_MultiFrequency"
f.attrs["frequencies_hz"] = str(frequencies)
f.attrs["stimulation_duration_ms"] = stimulation_duration_ms
f.attrs["model_dt_ms"] = dt_model
f.attrs["stim_dt_ms"] = dt_stim

# Записываем параметры модели
grp_model = f.create_group("ModelParams")
grp_model.attrs["fiber_diameter_um"] = axon.fiber_diameter
grp_model.attrs["dt_ms"] = axon.dt_ms
grp_model.attrs["celsius"] = axon.celsius
grp_model.attrs["parent_axon_nodes"] = axon.parent_axon_nodes
grp_model.attrs["branch_nodes"] = axon.branch_nodes
grp_model.attrs["branches_num"] = axon.branches_num
grp_model.attrs["nodes_dist"] = axon.nodes_dist
grp_model.attrs["diam_scale"] = axon.diam_scale
grp_model.attrs["h_stop"] = axon.h_stop


# --------------------------
# Основной цикл по частотам
# --------------------------

for freq in frequencies:

    print(f"\n=== Частота {freq} Гц ===")

    axon.set_stimulation_params(mode='create',
                                freq_hz=freq,
                                amp=amp_nA,
                                t_start=10,
                                t_end=h_stop_ms,
                                ton=ton_ms,
                                pulse_len_ms=1,
                                plot_duration=100)

    axon.run_simulation()

    # Имя группы: Frequency_050Hz и т.п.
    group_name = f"Frequency_{freq:03d}Hz"

    # Запускаем симуляцию и сохраняем в HDF5
    axon.run_simulation(
        h5_path=h5_path,
        experiment_name=group_name)
    axon.plot_voltage_traces()

f.close()

print("\nФайл успешно создан:")
print(h5_path)


'''


# Создаем аксон один раз
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
    h_stop=100.0
)

axon.set_stimulation_params(mode = 'create',
                            freq_hz= 50,
                            amp = 1.0,
                            t_start = 10,
                            t_end = 100,
                            ton = 0.1,
                            pulse_len_ms = 1.0,
                            plot_duration = 100)

axon.run_simulation()
axon.plot_voltage_traces()
'''


'''

# Анализ на частоте 50 Гц в течение 10 секунд
detailed_data = axon.analyze_single_frequency(
    freq=700,                       # 50 Гц
    amp=1.0,                       # 1 нА
    stimulation_duration_ms=1000, # 10 секунд
    plot_duration_ms=100          # показывать первые 1000 мс
)

''''''
# Дополнительный анализ эффективности
conduction_analysis = axon.analyze_conduction_efficiency(
    detailed_data['voltage_matrix'],
    detailed_data['time_array']
)

print(f"Анализ проведения при {detailed_data['frequency']} Гц:")
print(f"  Спайков до ветвления: {conduction_analysis['spikes_before']}")
print(f"  Спайков в основной ветви: {conduction_analysis['spikes_main']} (эффективность: {conduction_analysis['main_efficiency']:.1%})")
print(f"  Спайков в дочерней ветви: {conduction_analysis['spikes_daughter']} (эффективность: {conduction_analysis['daughter_efficiency']:.1%})")

'''