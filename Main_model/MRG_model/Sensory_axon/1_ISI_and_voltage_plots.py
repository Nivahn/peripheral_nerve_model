import h5py
import numpy as np
import matplotlib.pyplot as plt
import scipy
import pandas
from scipy.optimize import bracket
from scipy.signal import find_peaks
import os



def analyze_single_point(path_to_V ,time_trace, trace):

    bp_trace = path_to_V[:]
    bp_peaks, _ = find_peaks(bp_trace, distance=50, height= -20)#np.max(bp_trace) - 10)
    bp_isi = np.diff(time_trace[bp_peaks])
    bp_times = (time_trace[bp_peaks[:-1]] + time_trace[bp_peaks[1:]]) / 2


    return bp_trace, bp_isi, bp_times, bp_peaks

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
            bp_trace, bp_isi, bp_times, bp_peaks = analyze_single_point(f[f"{freq}/Model/Traces/branch_point/{bp[trace]}/voltage"] ,time_trace, trace)
            abd_trace, abd_isi, abd_times, abd_peaks = analyze_single_point(f[f"{freq}/Model/Traces/after_branch_daughter/{abd[trace]}/voltage"],time_trace, trace)
            bb_trace, bb_isi, bb_times, bb_peaks = analyze_single_point(f[f"{freq}/Model/Traces/before_branch/{bb[trace]}/voltage"] ,time_trace, trace)
            abm_trace, abm_isi, abm_times, abm_peaks = analyze_single_point(f[f"{freq}/Model/Traces/after_branch_main/{abm[trace]}/voltage"] ,time_trace, trace)


            if time_trace[-1] >= 2000:
                start_idx = np.where(time_trace >= 2000)[0][0]
                if time_trace[-1] >= 2500:
                    end_idx = np.where(time_trace <= 2500)[0][-1]
                else:
                    end_idx = len(time_trace) - 1
            else:
                # Если 2000 мс нет в данных, используем последние 500 мс
                start_idx = max(0, len(time_trace) - 500)
                end_idx = len(time_trace) - 1

            # Создаем фигуру с 6 графиками (3 строки, 2 колонки)
            fig, axes = plt.subplots(3, 2, figsize=(16, 12))

            # Заголовок для всех графиков
            main_title = f"Частота: {freq}, Амплитуда: {amp_nA} нА, Ветвление: {trace}"
            fig.suptitle(main_title, fontsize=16, fontweight='bold', y=1.02)

            # Цвета для графиков
            colors = {
                'before_branch': 'blue',
                'after_branch_main': 'red',
                'after_branch_daughter': 'green'
            }

            # График 1,1: До ветвления (2000-2500 мс)
            ax = axes[0, 0]
            time_segment = time_trace[start_idx:end_idx]
            trace_segment = bb_trace[start_idx:end_idx]
            ax.plot(time_segment, trace_segment, linewidth=1.5, color=colors['before_branch'])

            # Фильтруем пики, которые попадают в выбранный диапазон
            bb_peaks_segment = [p for p in bb_trace if start_idx <= p < end_idx]
            if len(bb_peaks_segment) > 0:
                ax.scatter(time_trace[bb_peaks_segment], bb_trace[bb_peaks_segment],
                          s=80, color='black', zorder=5, label='Пики')

            ax.set_xlabel("Время (мс)")
            ax.set_ylabel("Потенциал (мВ)")
            ax.set_title("До ветвления")
            ax.grid(True, alpha=0.3)
            ax.legend()

            # График 2,1: После ветвления - материнский аксон
            ax = axes[1, 0]
            trace_segment = abm_trace[start_idx:end_idx]
            ax.plot(time_segment, trace_segment, linewidth=1.5, color=colors['after_branch_main'])

            # Фильтруем пики, которые попадают в выбранный диапазон
            abm_peaks_segment = [p for p in abm_peaks if start_idx <= p < end_idx]
            if len(abm_peaks_segment) > 0:
                ax.scatter(time_trace[abm_peaks_segment], abm_trace[abm_peaks_segment],
                          s=80, color='black', zorder=5, label='Пики')

            ax.set_xlabel("Время (мс)")
            ax.set_ylabel("Потенциал (мВ)")
            ax.set_title("После ветвления - материнский аксон")
            ax.grid(True, alpha=0.3)
            ax.legend()

            # График 3,1: После ветвления - дочерняя ветвь
            ax = axes[2, 0]
            trace_segment = abd_trace[start_idx:end_idx]
            ax.plot(time_segment, trace_segment, linewidth=1.5, color=colors['after_branch_daughter'])

            # Фильтруем пики, которые попадают в выбранный диапазон
            abd_peaks_segment = [p for p in abd_peaks if start_idx <= p < end_idx]
            if len(abd_peaks_segment) > 0:
                ax.scatter(time_trace[abd_peaks_segment], abd_trace[abd_peaks_segment],
                          s=80, color='black', zorder=5, label='Пики')

            ax.set_xlabel("Время (мс)")
            ax.set_ylabel("Потенциал (мВ)")
            ax.set_title("После ветвления - дочерняя ветвь")
            ax.grid(True, alpha=0.3)
            ax.legend()

            # График 1,2: ISI vs Time
            ax = axes[0, 1]

            if len(bb_isi) > 0:
                ax.scatter(bb_times, bb_isi, s=40, alpha=0.7, color=colors['before_branch'], label='До ветвления')
            if len(abm_isi) > 0:
                ax.scatter(abm_times, abm_isi, s=40, alpha=0.7, color=colors['after_branch_main'], label='После (материнский)')
            if len(abd_isi) > 0:
                ax.scatter(abd_times, abd_isi, s=40, alpha=0.7, color=colors['after_branch_daughter'], label='После (дочерняя)')

            ax.set_xlabel("Время (мс)")
            ax.set_ylabel("ISI (мс)")
            ax.set_title("Интервалы между спайками (ISI) vs Время")
            ax.legend(fontsize=10)
            ax.grid(True, alpha=0.3)

            # График 2,2: ISI_n vs ISI_{n+1}
            ax = axes[1, 1]

            if len(bb_isi) > 1:
                ax.scatter(bb_isi[:-1], bb_isi[1:], s=40, alpha=0.7, color=colors['before_branch'], label='До ветвления')
            if len(abm_isi) > 1:
                ax.scatter(abm_isi[:-1], abm_isi[1:], s=40, alpha=0.7, color=colors['after_branch_main'], label='После (материнский)')
            if len(abd_isi) > 1:
                ax.scatter(abd_isi[:-1], abd_isi[1:], s=40, alpha=0.7, color=colors['after_branch_daughter'], label='После (дочерняя)')

            # Добавляем линию y=x
            max_isi = max(
                np.max(bb_isi) if len(bb_isi) > 0 else 0,
                np.max(abm_isi) if len(abm_isi) > 0 else 0,
                np.max(abd_isi) if len(abd_isi) > 0 else 0
            )
            if max_isi > 0:
                ax.plot([0, max_isi], [0, max_isi], 'k--', alpha=0.5, label='y = x')

            ax.set_xlabel("ISI_n (мс)")
            ax.set_ylabel("ISI_{n+1} (мс)")
            ax.set_title("Карта возврата: ISI_n vs ISI_n+1")
            ax.legend(fontsize=10)
            ax.grid(True, alpha=0.3)
            ax.axis('equal')

            # График 3,2: Гистограмма ISI
            ax = axes[2, 1]

            bins = 30
            all_isi = []
            if len(bb_isi) > 0:
                ax.hist(bb_isi, bins=bins, alpha=0.5, color=colors['before_branch'], label='До ветвления', density=True)
                all_isi.extend(bb_isi)
            if len(abm_isi) > 0:
                ax.hist(abm_isi, bins=bins, alpha=0.5, color=colors['after_branch_main'], label='После (материнский)', density=True)
                all_isi.extend(abm_isi)
            if len(abd_isi) > 0:
                ax.hist(abd_isi, bins=bins, alpha=0.5, color=colors['after_branch_daughter'], label='После (дочерняя)', density=True)
                all_isi.extend(abd_isi)

            ax.set_xlabel("ISI (мс)")
            ax.set_ylabel("Плотность вероятности")
            ax.set_title("Гистограмма интервалов между спайками")
            ax.legend(fontsize=10)
            ax.grid(True, alpha=0.3)

            # Настраиваем layout
            plt.tight_layout()

            # Сохраняем график в файл
            save_filename = f"{freq}_branch_{trace}.png"
            save_path = os.path.join(output_dir, save_filename)
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close(fig)  # Закрываем фигуру для экономии памяти

            print(f"    График сохранен: {save_path}")
