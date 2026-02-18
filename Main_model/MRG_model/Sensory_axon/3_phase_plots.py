import h5py
import numpy as np
import matplotlib.pyplot as plt
import scipy
import pandas
from scipy.optimize import bracket
from scipy.signal import find_peaks
import os
import matplotlib.animation as animation



def get_all_states_variables(dataset, path_to_node):

    n = dataset[f"{path_to_node}/n"][:]
    m = dataset[f"{path_to_node}/m"][:]
    h = dataset[f"{path_to_node}/h"][:]
    mp = dataset[f"{path_to_node}/mp"][:]
    s = dataset[f"{path_to_node}/s"][:]
    v = dataset[f"{path_to_node}/voltage"][:]

    return n, m ,h, mp, s, v

filepath = r"C:\Users\User\PycharmProjects\peripheral_nerve_model\Data\sensory_axon\MRG_50_1000Hz_diam_5.7_amp_5.h5"

# Создаем папку для сохранения графиков
output_dir = "./../../../Data/sensory_axon/analysis_plots/"
os.makedirs(output_dir, exist_ok=True)

with h5py.File(filepath,'r') as f:
    print(f.keys())
    print(f["Frequency_1000Hz/Model"].keys())
    print(f["Frequency_1000Hz/Model/Traces"].keys())
    print(f["ModelParams"].attrs.keys())
    print(f["ModelParams"].attrs["dt_ms"])
    print(f["ModelParams"].attrs["h_stop"])
    print(f["ModelParams"].attrs["branch_nodes"])
    stim_params = f[f"Frequency_1000Hz/Stimulator"].attrs
    amp_nA = stim_params.get('amp')
    print(f"[START] processing {amp_nA} nA")

    time_trace = f["Frequency_1000Hz/Model/time"][:]

    frequencies = ['Frequency_050Hz', 'Frequency_100Hz', 'Frequency_150Hz', 'Frequency_200Hz', 'Frequency_250Hz', 'Frequency_300Hz', 'Frequency_350Hz', 'Frequency_400Hz', 'Frequency_450Hz', 'Frequency_500Hz', 'Frequency_550Hz', 'Frequency_600Hz', 'Frequency_650Hz', 'Frequency_700Hz', 'Frequency_750Hz', 'Frequency_800Hz', 'Frequency_850Hz', 'Frequency_900Hz', 'Frequency_950Hz', 'Frequency_1000Hz']

    '''
    i = 0
    while i != 1000:
        i += 50
        frequencies.append(f"Frequency_{i}Hz")

    print(frequencies)
    '''

    for freq in frequencies:
        #print(freq)
        current_frequency = f[f"{freq}/Model/Traces"]
        stim_point = f[f"{freq}/Model/Traces"]
        # Получаем параметры стимуляции
        stim_params = f[f"{freq}/Stimulator"].attrs
        amp_nA = stim_params.get('amp')
        print(current_frequency.keys())

        abd = sorted(list(f[f"{freq}/Model/Traces/after_branch_daughter"].keys()))
        abm = sorted(list(f[f"{freq}/Model/Traces/after_branch_main"].keys()))
        bb = sorted(list(f[f"{freq}/Model/Traces/before_branch"].keys()))
        bp = sorted(list(f[f"{freq}/Model/Traces/branch_point"].keys()))

        if abd:  # проверяем, что список не пустой
            print(f"Первый узел в after_branch_daughter: {abd[0]}")
            print(f[f'{freq}/Model/Traces/after_branch_daughter/{abd[0]}/voltage'])



        #height = np.max(abd_trace) - 10
        #sp_trace, sp_isi, sp_times, sp_peaks = analyze_single_point(current_frequency, "stimulation_point" ,time_trace, 0)
        for trace in range(1):
            print(f"Ветвление: {trace}, частота: {freq}")

            # Получаем все переменные состояния для branch_point
            bp_n, bp_m, bp_h, bp_mp, bp_s, bp_v = get_all_states_variables(f,
                f"{freq}/Model/Traces/branch_point/{bp[trace]}")

            # Получаем все переменные состояния для after_branch_daughter
            abd_n, abd_m, abd_h, abd_mp, abd_s, abd_v = get_all_states_variables(f,
                f"{freq}/Model/Traces/after_branch_daughter/{abd[trace]}")


            # Получаем все переменные состояния для before_branch
            bb_n, bb_m, bb_h, bb_mp, bb_s, bb_v = get_all_states_variables(f,
                f"{freq}/Model/Traces/before_branch/{bb[trace]}")


            # Получаем все переменные состояния для after_branch_main
            abm_n, abm_m, abm_h, abm_mp, abm_s, abm_v = get_all_states_variables(f,
                f"{freq}/Model/Traces/after_branch_main/{abm[trace]}")


            def plot_phase_panels(data_dict, freq, output_dir):

                fig, axes = plt.subplots(1, 4, figsize=(16, 4), sharey=True)

                titles = [
                    "Before branch",
                    "Branch point",
                    "After main",
                    "After daughter"
                ]

                for ax, key, title in zip(axes, data_dict.keys(), titles):
                    n, m, h, mp, s, v = data_dict[key]

                    # линии с alpha
                    ax.plot(v, n, label="n", alpha=0.7)
                    ax.plot(v, m, label="m", alpha=0.7)
                    ax.plot(v, h, label="h", alpha=0.7)
                    ax.plot(v, mp, label="mp", alpha=0.7)
                    ax.plot(v, s, label="s", alpha=0.7)

                    # начальные точки
                    ax.scatter(v[0], n[0], color="black", s=20, zorder=5)
                    ax.scatter(v[0], m[0], color="black", s=20, zorder=5)
                    ax.scatter(v[0], h[0], color="black", s=20, zorder=5)
                    ax.scatter(v[0], mp[0], color="black", s=20, zorder=5)
                    ax.scatter(v[0], s[0], color="black", s=20, zorder=5)

                    ax.set_title(title)
                    ax.set_xlabel("Voltage (mV)")
                    ax.set_xlim(-100, 50)
                    ax.set_ylim(0, 1.05)
                    ax.grid(True)

                axes[0].set_ylabel("Gating variables")
                axes[-1].legend(loc="upper right")

                plt.tight_layout()

                save_path = os.path.join(output_dir, f"{freq}_phase.png")
                plt.savefig(save_path, dpi=300)
                plt.close()


            data_dict = {
                "before_branch": (bb_n, bb_m, bb_h, bb_mp, bb_s, bb_v),
                "branch_point": (bp_n, bp_m, bp_h, bp_mp, bp_s, bp_v),
                "after_main": (abm_n, abm_m, abm_h, abm_mp, abm_s, abm_v),
                "after_daughter": (abd_n, abd_m, abd_h, abd_mp, abd_s, abd_v)
            }

            plot_phase_panels(data_dict, freq, output_dir)



