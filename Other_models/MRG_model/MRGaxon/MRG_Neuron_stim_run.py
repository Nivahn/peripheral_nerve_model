from impulse_generator import STIMULATOR
from neuron import h
import matplotlib.pyplot as plt
import numpy as np
import math
import time
import os
import pandas as pd
from scipy.signal import find_peaks
from MRG_lib import *


csv_50Hz = "./../../../Data/SCS_LTMRs_50Hz_Timestamps.csv"
h5_path = "./../../../Data/MRG_responses_50Hz_biphasic_40um_gap5um.h5"

df = pd.read_csv(csv_50Hz, header=None)
n_neurons = df.shape[1]

axon = MRGaxon(
    fiber_diameter=5.7,
    parent_axon_nodes=42,
    branch_nodes=21,
    branches_num=2,
    nodes_dist=10,
    diam_scale=0.6,
    celsius=37.0,
    dt_ms=0.005,
    v_init=-80.0,
    h_stop=5000.0
)

with h5py.File(h5_path, "w") as f:
    # ---- глобальные метаданные ----
    f.attrs["experiment_name"] = "SCS_50Hz_MRG_5p7um"
    f.attrs["comment"] = "LTMRs → MRGaxon, first 5000 ms"

    grp_global = f.create_group("GlobalModel")
    grp_global.attrs["fiber_diameter_um"] = axon.fiber_diameter
    grp_global.attrs["dt_ms"] = axon.dt_ms
    grp_global.attrs["celsius"] = axon.celsius
    grp_global.attrs["fiber_diameter"] = axon.fiber_diameter
    grp_global.attrs["parent_axon_nodes"] = axon.parent_axon_nodes
    grp_global.attrs["branch_nodes"] = axon.branch_nodes
    grp_global.attrs["branches_num"] = axon.branches_num
    grp_global.attrs["nodes_dist"] = axon.nodes_dist
    grp_global.attrs["diam_scale"] = axon.diam_scale
    grp_global.attrs["h_stop"] = axon.h_stop

    # ---- цикл по нейронам ----
    for neuron_idx in range(n_neurons):
        neuron_name = f"Neuron_{neuron_idx+1:02d}"
        grp_neuron = f.create_group(neuron_name)

        # 1) входные спайки (в мс)
        spike_times_sec = df[neuron_idx].dropna().values
        mask = spike_times_sec <= 5.0   # первые 5 сек
        spike_times_ms = spike_times_sec[mask] *1000.0
        print(f"spike_times_ms: {spike_times_ms}")
        
        grp_neuron.attrs["input_neuron_index"] = neuron_idx
        grp_neuron.attrs["input_source_file"] = os.path.basename(csv_50Hz)
        grp_neuron.attrs["t_max_ms"] = 5000.0
        grp_neuron.attrs["spike_count_input"] = len(spike_times_ms)

        grp_input = grp_neuron.create_group("Input")
        grp_input.create_dataset("spike_times_ms", data=spike_times_ms)

        # 2) создаём стим
        axon.set_stimulation_params(
            mode="preload_data",
            csv_path=csv_50Hz,
            neuron_index=neuron_idx,
            index_is_one_based=False,
            t_end=5000.0,  # МИЛЛИСЕКУНДЫ, не секунды
            amp=-1.0,
            phase_us=40.0,
            gap_us=5.0,
            freq_hz = f"Neuron idx: {neuron_idx}"
        )

        # 3) гоняем модель и сохраняем внутри /Neuron_X
        axon.run_simulation(
            h5_path=h5_path,
            experiment_name=neuron_name)

csv_50Hz = "./../../../Data/SCS_LTMRs_1KHz_Timestamps.csv"
h5_path = "./../../../Data/MRG_responses_1KHz_biphasic_40um_gap5um.h5"

df = pd.read_csv(csv_50Hz, header=None)
n_neurons = df.shape[1]

axon = MRGaxon(
    fiber_diameter=5.7,
    parent_axon_nodes=42,
    branch_nodes=21,
    branches_num=2,
    nodes_dist=10,
    diam_scale=0.6,
    celsius=37.0,
    dt_ms=0.005,
    v_init=-80.0,
    h_stop=5000.0
)

with h5py.File(h5_path, "w") as f:
    # ---- глобальные метаданные ----
    f.attrs["experiment_name"] = "SCS_1KHz_MRG_5p7um"
    f.attrs["comment"] = "LTMRs → MRGaxon, first 5000 ms"

    grp_global = f.create_group("GlobalModel")
    grp_global.attrs["fiber_diameter_um"] = axon.fiber_diameter
    grp_global.attrs["dt_ms"] = axon.dt_ms
    grp_global.attrs["celsius"] = axon.celsius
    grp_global.attrs["fiber_diameter"] = axon.fiber_diameter
    grp_global.attrs["parent_axon_nodes"] = axon.parent_axon_nodes
    grp_global.attrs["branch_nodes"] = axon.branch_nodes
    grp_global.attrs["branches_num"] = axon.branches_num
    grp_global.attrs["nodes_dist"] = axon.nodes_dist
    grp_global.attrs["diam_scale"] = axon.diam_scale
    grp_global.attrs["h_stop"] = axon.h_stop

    # ---- цикл по нейронам ----
    for neuron_idx in range(n_neurons):
        neuron_name = f"Neuron_{neuron_idx + 1:02d}"
        grp_neuron = f.create_group(neuron_name)

        # 1) входные спайки (в мс)
        spike_times_sec = df[neuron_idx].dropna().values
        mask = spike_times_sec <= 5.0  # первые 5 сек
        spike_times_ms = spike_times_sec[mask] * 1000.0
        print(f"spike_times_ms: {spike_times_ms}")

        grp_neuron.attrs["input_neuron_index"] = neuron_idx
        grp_neuron.attrs["input_source_file"] = os.path.basename(csv_50Hz)
        grp_neuron.attrs["t_max_ms"] = 5000.0
        grp_neuron.attrs["spike_count_input"] = len(spike_times_ms)

        grp_input = grp_neuron.create_group("Input")
        grp_input.create_dataset("spike_times_ms", data=spike_times_ms)

        # 2) создаём стим
        axon.set_stimulation_params(
            mode="preload_data",
            csv_path=csv_50Hz,
            neuron_index=neuron_idx,
            index_is_one_based=False,
            t_end=5000.0,  # МИЛЛИСЕКУНДЫ, не секунды
            amp=-1.0,
            phase_us=40.0,
            gap_us=5.0,
            freq_hz=f"Neuron idx: {neuron_idx}"
        )

        # 3) гоняем модель и сохраняем внутри /Neuron_X
        axon.run_simulation(
            h5_path=h5_path,
            experiment_name=neuron_name)
"""
Для mode="preload_data" ожидаются поля:
    csv_path: путь к CSV с таймпоинтами (как у тебя)
    neuron_index: индекс колонки (0-based или 1-based — см. ниже)
    index_is_one_based: bool, если True, neuron_index считает с 1
    t_max_s: максимальное время в секундах (например, 5.0)
    amp: амплитуда импульса (нА)
    pulse_len_ms: длительность импульса (мс)
    time_unit: "s" или "ms" — в чем лежат времена в csv
"""


'''
axon.set_stimulation_params(mode = 'preload_data',
                            csv_path = csv_50Hz,
                            neuron_index = 1,
                            index_is_one_based = True,
                            t_end = 1000.0,
                            amp = 1.0,
                            pulse_len_ms = 1.0)

axon.run_simulation()
axon.plot_voltage_traces()
'''