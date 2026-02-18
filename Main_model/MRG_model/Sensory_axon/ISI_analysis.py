import h5py
import numpy as np
import matplotlib.pyplot as plt
import scipy
import pandas
from scipy.optimize import bracket
from scipy.signal import find_peaks
import os

def parse_freq_hz(freq_key: str) -> float:
    # "Frequency_1000Hz" -> 1000
    return float(freq_key.split("Frequency_")[-1].replace("Hz", ""))

def count_peaks_in_window(peaks_idx, start_idx, end_idx):
    # peaks_idx: np.array индексов пиков по всему треку
    if peaks_idx is None or len(peaks_idx) == 0:
        return 0
    peaks_idx = np.asarray(peaks_idx, dtype=int)
    return int(np.sum((peaks_idx >= start_idx) & (peaks_idx < end_idx)))


def analyze_single_point(freq, current_frequency, branch_point, time_trace, trace, order):
    #dir = "./../../../Data/test/"
    bp = current_frequency[branch_point]
    bp_trace = bp[order[trace]][:]
    min_dist_ms = 0.5  # рефрактер, под 1000 Гц ок
    min_dist_pts = max(1, int(min_dist_ms / (time_trace[1] - time_trace[0])))
    dv = np.diff(bp_trace) / (time_trace[1] - time_trace[0])  # mV/ms
    thr = np.percentile(dv, 99.5) * 0.5  # адаптивно
    cand = np.where((dv[1:] >= thr) & (dv[:-1] < thr))[0]  # моменты быстрого фронта
    # потом можно "почистить" по refractory
    bp_peaks = []
    last = -10 ** 9
    for c in cand:
        if c - last >= min_dist_pts:
            # ищем локальный максимум в окне после фронта (например 1 мс)
            w = int(1.0 / (time_trace[1] - time_trace[0]))
            j = c + np.argmax(bp_trace[c:c + w])
            bp_peaks.append(j)
            last = j
    bp_peaks = np.array(bp_peaks, dtype=int)

    #bp_peaks, _ = find_peaks(bp_trace, height= -20)#np.max(bp_trace) - 10)
    bp_isi = np.diff(time_trace[bp_peaks])
    bp_times = (time_trace[bp_peaks[:-1]] + time_trace[bp_peaks[1:]]) / 2
    #plt.figure(figsize=(20, 10))
    #plt.title(f"{freq}, {branch_point}, {trace}")
    #plt.plot(bp_trace)
    #plt.plot(bp_peaks, bp_trace[bp_peaks], "x")
    #plt.show()
    #save_filename = f"{freq}, {branch_point}, {trace}.png"
    #save_path = os.path.join(dir, save_filename)
    #plt.savefig(save_path, dpi=150, bbox_inches='tight')
    #plt.close()

    return bp_trace, bp_isi, bp_times, bp_peaks



filepath = r"C:\Users\User\PycharmProjects\peripheral_nerve_model\Data\MRG_MultiFreq_Stim_50_1000Hz_new_amp_10.h5"

# Создаем папку для сохранения графиков
output_dir = "./../../../Data/analysis_plots/new/10_nA/test"
os.makedirs(output_dir, exist_ok=True)

with h5py.File(filepath,'r') as f:
    results_rows = []  # сюда будем складывать строки для сводной таблицы

    print(f.keys())
    print(f["Frequency_1000Hz/Model"].keys())
    print(f["Frequency_1000Hz/Model/Traces"].keys())
    print(f["ModelParams"].attrs.keys())
    print(f["ModelParams"].attrs["dt_ms"])
    print(f["ModelParams"].attrs["h_stop"])
    print(f["ModelParams"].attrs["branch_nodes"])

    time_trace = f["Frequency_1000Hz/Model/time"][:]
    for freq in f.keys():
        if freq == "ModelParams":
            continue
        current_frequency = f[f"{freq}/Model/Traces"]

        # --- окно анализа (как у тебя), но вынесем, чтобы использовать для подсчётов ---
        if time_trace[-1] >= 0:
            start_idx = np.where(time_trace >= 0)[0][0]
            if time_trace[-1] >= 5000:
                end_idx = np.where(time_trace <= 5000)[0][-1]  # у тебя так было
            else:
                end_idx = len(time_trace) - 1
        else:
            start_idx = max(0, len(time_trace) - 500)
            end_idx = len(time_trace) - 1

        window_ms = float(time_trace[end_idx] - time_trace[start_idx])
        window_s = window_ms / 1000.0 if window_ms > 0 else 1e-9

        freq_hz_val = parse_freq_hz(freq)


        def sort_keys_by_node_index(grp):
            keys = list(grp.keys())

            def node_idx(k):
                s = grp[k].attrs.get("node", k)  # "node_14_0.50" или "trace_node_14_0.50"
                s = s.split("node_")[-1]  # "14_0.50" или "14_0.50..."
                return int(s.split("_")[0])  # 14

            return sorted(keys, key=node_idx)


        order_bb = sort_keys_by_node_index(current_frequency["before_branch"])
        order_abm = sort_keys_by_node_index(current_frequency["after_branch_main"])
        order_abd = sort_keys_by_node_index(current_frequency["after_branch_daughter"])
        order_bp = sort_keys_by_node_index(current_frequency["branch_point"])

        # Получаем параметры стимуляции
        stim_params = f[f"{freq}/Stimulator"].attrs
        amp_nA = stim_params.get('amp', 1.0)
        #height = np.max(abd_trace) - 10
        #sp_trace, sp_isi, sp_times, sp_peaks = analyze_single_point(freq ,current_frequency, "stimulation_point" ,time_trace, 0)
        for trace in range(4):
            print(f"Ветвление: {trace}, частота: {freq}")
            bp_trace, bp_isi, bp_times, bp_peaks = analyze_single_point(freq, current_frequency, "branch_point",
                                                                        time_trace, trace, order_bp)
            abd_trace, abd_isi, abd_times, abd_peaks = analyze_single_point(freq, current_frequency,
                                                                            "after_branch_daughter", time_trace, trace,
                                                                            order_abd)
            bb_trace, bb_isi, bb_times, bb_peaks = analyze_single_point(freq, current_frequency, "before_branch",
                                                                        time_trace, trace, order_bb)
            abm_trace, abm_isi, abm_times, abm_peaks = analyze_single_point(freq, current_frequency,
                                                                            "after_branch_main", time_trace, trace,
                                                                            order_abm)
            sp_trace, sp_isi, sp_times, sp_peaks = analyze_single_point(freq, current_frequency, "stimulation_point",
                                                                        time_trace, 0,
                                                                        sort_keys_by_node_index(
                                                                            current_frequency["stimulation_point"]))

            # --- считаем сколько спайков попало в окно анализа ---
            n_before = count_peaks_in_window(bb_peaks, start_idx, end_idx)
            n_after_main = count_peaks_in_window(abm_peaks, start_idx, end_idx)
            n_after_daughter = count_peaks_in_window(abd_peaks, start_idx, end_idx)
            n_branch_point = count_peaks_in_window(bp_peaks, start_idx, end_idx)
            n_stim = count_peaks_in_window(sp_peaks, start_idx, end_idx)

            # --- пропускная способность (spikes/s) ---
            rate_before = n_before / window_s
            rate_after_main = n_after_main / window_s
            rate_after_daughter = n_after_daughter / window_s
            rate_branch_point = n_branch_point / window_s
            rate_stim = n_stim / window_s

            # --- коэффициент проведения через ветвление ---
            ratio_main = (n_after_main / n_before) if n_before > 0 else np.nan
            ratio_daughter = (n_after_daughter / n_before) if n_before > 0 else np.nan

            results_rows.append({
                "freq_key": freq,
                "freq_hz": freq_hz_val,
                "branch_id": trace,
                "amp_nA": amp_nA,
                "window_ms": window_ms,
                "n_stim": n_stim,
                "n_before": n_before,
                "n_branch_point": n_branch_point,
                "n_after_main": n_after_main,
                "n_after_daughter": n_after_daughter,
                "rate_stim_hz": rate_stim,
                "rate_before_hz": rate_before,
                "rate_branch_point_hz": rate_branch_point,
                "rate_after_main_hz": rate_after_main,
                "rate_after_daughter_hz": rate_after_daughter,
                "ratio_main": ratio_main,
                "ratio_daughter": ratio_daughter,
            })

            if time_trace[-1] >= 2000:
                start_idx = np.where(time_trace >= 100)[0][0]
                if time_trace[-1] >= 2500:
                    end_idx = np.where(time_trace <= 1000)[0][-1]
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
            bb_peaks_segment = [p for p in bb_peaks if start_idx <= p < end_idx]

            if len(bb_peaks_segment) > 0:
                ax.scatter(time_trace[bb_peaks_segment], bb_trace[bb_peaks_segment],
                          s=20, color='black', zorder=5, label='spike')

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


import pandas as pd

df = pd.DataFrame(results_rows)

# на всякий случай сортировка
df = df.sort_values(["freq_hz", "branch_id"]).reset_index(drop=True)

# ===== 1) Спайки до/после vs Гц (среднее по ветвлениям) =====
g = df.groupby("freq_hz", as_index=False).agg({
    "n_before": "mean",
    "n_after_main": "mean",
    "n_after_daughter": "mean",
})

plt.figure(figsize=(10,6))
plt.plot(g["freq_hz"], g["n_before"], marker="o", label="До ветвления (mean)")
plt.plot(g["freq_hz"], g["n_after_main"], marker="o", label="После (основная, mean)")
plt.xlabel("Частота стимуляции (Гц)")
plt.ylabel("Кол-во спайков в окне анализа")
plt.title("Спайки: до vs после (основная) в зависимости от частоты")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "summary_spikes_main_vs_freq.png"), dpi=150)
plt.close()

plt.figure(figsize=(10,6))
plt.plot(g["freq_hz"], g["n_before"], marker="o", label="До ветвления (mean)")
plt.plot(g["freq_hz"], g["n_after_daughter"], marker="o", label="После (дочерняя, mean)")
plt.xlabel("Частота стимуляции (Гц)")
plt.ylabel("Кол-во спайков в окне анализа")
plt.title("Спайки: до vs после (дочерняя) в зависимости от частоты")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "summary_spikes_daughter_vs_freq.png"), dpi=150)
plt.close()

# ===== 2) Спайки vs Гц + по номеру ветвления =====
for branch_id in sorted(df["branch_id"].unique()):
    d = df[df["branch_id"] == branch_id].sort_values("freq_hz")

    plt.figure(figsize=(10,6))
    plt.plot(d["freq_hz"], d["n_before"], marker="o", label="До ветвления")
    plt.plot(d["freq_hz"], d["n_after_main"], marker="o", label="После (основная)")
    plt.xlabel("Частота стимуляции (Гц)")
    plt.ylabel("Кол-во спайков в окне анализа")
    plt.title(f"Спайки: основная ветвь vs частота (ветвление #{branch_id})")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"spikes_main_vs_freq_branch_{branch_id}.png"), dpi=150)
    plt.close()

    plt.figure(figsize=(10,6))
    plt.plot(d["freq_hz"], d["n_before"], marker="o", label="До ветвления")
    plt.plot(d["freq_hz"], d["n_after_daughter"], marker="o", label="После (дочерняя)")
    plt.xlabel("Частота стимуляции (Гц)")
    plt.ylabel("Кол-во спайков в окне анализа")
    plt.title(f"Спайки: дочерняя ветвь vs частота (ветвление #{branch_id})")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"spikes_daughter_vs_freq_branch_{branch_id}.png"), dpi=150)
    plt.close()

# ===== 3) Пропускная способность (spikes/s) до/после vs Гц (mean по ветвлениям) =====
gr = df.groupby("freq_hz", as_index=False).agg({
    "rate_before_hz": "mean",
    "rate_after_main_hz": "mean",
    "rate_after_daughter_hz": "mean",
})

plt.figure(figsize=(10,6))
plt.plot(gr["freq_hz"], gr["rate_before_hz"], marker="o", label="До ветвления (mean)")
plt.plot(gr["freq_hz"], gr["rate_after_main_hz"], marker="o", label="После (основная, mean)")
plt.xlabel("Частота стимуляции (Гц)")
plt.ylabel("Пропускная способность (спайков/с)")
plt.title("Пропускная способность: до vs после (основная) в зависимости от частоты")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "throughput_main_vs_freq.png"), dpi=150)
plt.close()

plt.figure(figsize=(10,6))
plt.plot(gr["freq_hz"], gr["rate_before_hz"], marker="o", label="До ветвления (mean)")
plt.plot(gr["freq_hz"], gr["rate_after_daughter_hz"], marker="o", label="После (дочерняя, mean)")
plt.xlabel("Частота стимуляции (Гц)")
plt.ylabel("Пропускная способность (спайков/с)")
plt.title("Пропускная способность: до vs после (дочерняя) в зависимости от частоты")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "throughput_daughter_vs_freq.png"), dpi=150)
plt.close()

# ===== 4) Пропускная способность + по номеру ветвления =====
for branch_id in sorted(df["branch_id"].unique()):
    d = df[df["branch_id"] == branch_id].sort_values("freq_hz")

    plt.figure(figsize=(10,6))
    plt.plot(d["freq_hz"], d["rate_before_hz"], marker="o", label="До ветвления")
    plt.plot(d["freq_hz"], d["rate_after_main_hz"], marker="o", label="После (основная)")
    plt.xlabel("Частота стимуляции (Гц)")
    plt.ylabel("Пропускная способность (спайков/с)")
    plt.title(f"Пропускная способность: основная ветвь (ветвление #{branch_id})")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"throughput_main_vs_freq_branch_{branch_id}.png"), dpi=150)
    plt.close()

    plt.figure(figsize=(10,6))
    plt.plot(d["freq_hz"], d["rate_before_hz"], marker="o", label="До ветвления")
    plt.plot(d["freq_hz"], d["rate_after_daughter_hz"], marker="o", label="После (дочерняя)")
    plt.xlabel("Частота стимуляции (Гц)")
    plt.ylabel("Пропускная способность (спайков/с)")
    plt.title(f"Пропускная способность: дочерняя ветвь (ветвление #{branch_id})")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"throughput_daughter_vs_freq_branch_{branch_id}.png"), dpi=150)
    plt.close()

# ===== 5) График "проведение через branch point" (ratio) =====
gc = df.groupby("freq_hz", as_index=False).agg({
    "ratio_main": "mean",
    "ratio_daughter": "mean",
})

plt.figure(figsize=(10,6))
plt.plot(gc["freq_hz"], gc["ratio_main"], marker="o", label="Conduction ratio (основная, mean)")
plt.plot(gc["freq_hz"], gc["ratio_daughter"], marker="o", label="Conduction ratio (дочерняя, mean)")
plt.xlabel("Частота стимуляции (Гц)")
plt.ylabel("Доля прошедших спайков (after/before)")
plt.title("Проведение через ветвление (branch point): after/before vs частота")
plt.ylim(-0.05, 1.05)
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "conduction_ratio_vs_freq.png"), dpi=150)
plt.close()

# ===== 6) Conduction ratio + по номеру ветвления =====
for branch_id in sorted(df["branch_id"].unique()):
    d = df[df["branch_id"] == branch_id].sort_values("freq_hz")

    plt.figure(figsize=(10,6))
    plt.plot(d["freq_hz"], d["ratio_main"], marker="o", label="ratio main")
    plt.plot(d["freq_hz"], d["ratio_daughter"], marker="o", label="ratio daughter")
    plt.xlabel("Частота стимуляции (Гц)")
    plt.ylabel("Доля прошедших спайков (after/before)")
    plt.title(f"Проведение через ветвление: after/before (ветвление #{branch_id})")
    plt.ylim(-0.05, 1.05)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"conduction_ratio_vs_freq_branch_{branch_id}.png"), dpi=150)
    plt.close()

# (опционально) сохраним таблицу результатов
df.to_csv(os.path.join(output_dir, "spike_conduction_summary.csv"), index=False)
print(f"[OK] Summary saved to: {os.path.join(output_dir, 'spike_conduction_summary.csv')}")
