from neuron import h
import matplotlib.pyplot as plt
import numpy as np
import math
import time
import os
import pandas as pd
from scipy.signal import find_peaks
from MRG_lib import *


def run_stimulation_sweep(axon, freq_list, amp_list, output_dir="sweep_results", threshold=-20.0):
    """
    Запускает серию симуляций с разными параметрами стимуляции.

    Args:
        axon: экземпляр MRGaxon
        freq_list: список частот стимуляции (Гц)
        amp_list: список амплитуд тока (нА)
        output_dir: директория для сохранения результатов
        threshold: порог для обнаружения спайков

    Returns:
        DataFrame с результатами всех симуляций
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    results = []
    total_start_time = time.time()

    print(
        f"Запуск сканирования параметров: {len(freq_list)} частот × {len(amp_list)} амплитуд = {len(freq_list) * len(amp_list)} симуляций")

    for i, freq in enumerate(freq_list):
        for j, amp in enumerate(amp_list):
            sim_start_time = time.time()

            print(f"Симуляция {i * len(amp_list) + j + 1}/{(len(freq_list) * len(amp_list))}: "
                  f"freq={freq} Гц, amp={amp} нА")

            try:
                # Устанавливаем параметры стимуляции
                axon.set_stimulation_parameters(freq_hz=freq, amp=amp)

                # Запускаем симуляцию
                time_array, voltage_matrix = axon.run_simulation()
                axon.plot_voltage_traces()

                # Анализируем спайки
                spike_stats = axon.analyze_branching_spikes(threshold=threshold)

                # Создаем график
                fig, axes = plt.subplots(2, 1, figsize=(12, 10))

                # Верхний график: потенциалы
                if 'before_branch' in spike_stats:
                    before_idx = [i for i, name in axon.recording_indices.items() if name == 'before_branch'][0]
                    axes[0].plot(time_array, voltage_matrix[before_idx],
                                 label='До ветвления', linewidth=2, color='blue')

                if 'after_branch_main' in spike_stats:
                    after_main_idx = [i for i, name in axon.recording_indices.items() if name == 'after_branch_main'][0]
                    axes[0].plot(time_array, voltage_matrix[after_main_idx],
                                 label='После ветвления (основная)', linewidth=2, color='red')

                if 'after_branch_daughter' in spike_stats:
                    after_daughter_idx = \
                    [i for i, name in axon.recording_indices.items() if name == 'after_branch_daughter'][0]
                    axes[0].plot(time_array, voltage_matrix[after_daughter_idx],
                                 label='После ветвления (дочерняя)', linewidth=2, color='green')

                axes[0].set_xlabel('Время (мс)')
                axes[0].set_ylabel('Потенциал (мВ)')
                axes[0].set_title(f'Потенциалы действия: {freq} Гц, {amp} нА')
                axes[0].legend()
                axes[0].grid(True, alpha=0.3)

                # Нижний график: стимуляция
                if hasattr(axon, 't_points') and hasattr(axon, 'i_points'):
                    axes[1].plot(axon.t_points, axon.i_points, linewidth=2, color='purple')
                    axes[1].set_xlabel('Время (мс)')
                    axes[1].set_ylabel('Ток (нА)')
                    axes[1].set_title('Протокол стимуляции')
                    axes[1].grid(True, alpha=0.3)
                    axes[1].set_ylim(-0.1, amp * 1.2)

                plt.tight_layout()

                # Сохраняем график
                plot_filename = os.path.join(output_dir, f"freq_{freq}_amp_{amp}.png")
                plt.savefig(plot_filename, dpi=150, bbox_inches='tight')
                plt.close()

                sim_time = time.time() - sim_start_time

                # Собираем результаты
                result = {
                    'frequency_hz': freq,
                    'amplitude_na': amp,
                    'simulation_time_sec': sim_time,
                    'total_spikes_before': spike_stats.get('before_branch', {}).get('spike_count', 0),
                    'total_spikes_main': spike_stats.get('after_branch_main', {}).get('spike_count', 0),
                    'total_spikes_daughter': spike_stats.get('after_branch_daughter', {}).get('spike_count', 0),
                    'conduction_ratio_main': spike_stats.get('after_branch_main', {}).get('conduction_ratio', 0),
                    'conduction_ratio_daughter': spike_stats.get('after_branch_daughter', {}).get('conduction_ratio', 0)
                }

                results.append(result)

                print(f"  Спайков: до={result['total_spikes_before']}, "
                      f"основная={result['total_spikes_main']}, "
                      f"дочерняя={result['total_spikes_daughter']}, "
                      f"время={sim_time:.2f}с")

            except Exception as e:
                print(f"  Ошибка при симуляции: {e}")
                # Добавляем запись об ошибке
                result = {
                    'frequency_hz': freq,
                    'amplitude_na': amp,
                    'simulation_time_sec': time.time() - sim_start_time,
                    'total_spikes_before': 0,
                    'total_spikes_main': 0,
                    'total_spikes_daughter': 0,
                    'conduction_ratio_main': 0,
                    'conduction_ratio_daughter': 0,
                    'error': str(e)
                }
                results.append(result)

    total_time = time.time() - total_start_time

    # Создаем DataFrame и сохраняем
    df = pd.DataFrame(results)
    csv_filename = os.path.join(output_dir, "stimulation_sweep_results.csv")
    df.to_csv(csv_filename, index=False)

    # Создаем сводный отчет
    successful_simulations = len([r for r in results if r.get('total_spikes_before', 0) > 0])

    summary = {
        'total_simulations': len(results),
        'total_time_sec': total_time,
        'avg_time_per_simulation_sec': total_time / len(results) if results else 0,
        'successful_simulations': successful_simulations,
        'success_rate': successful_simulations / len(results) if results else 0
    }

    print("\n" + "=" * 50)
    print("СВОДКА СКАНИРОВАНИЯ ПАРАМЕТРОВ")
    print("=" * 50)
    print(f"Всего симуляций: {summary['total_simulations']}")
    print(f"Общее время: {summary['total_time_sec']:.2f} сек")
    print(f"Среднее время на симуляцию: {summary['avg_time_per_simulation_sec']:.2f} сек")
    print(f"Успешных симуляций (со спайками): {summary['successful_simulations']}")
    print(f"Успешность: {summary['success_rate']:.1%}")
    print(f"Результаты сохранены в: {output_dir}")

    # Сохраняем сводку
    with open(os.path.join(output_dir, "summary.txt"), 'w') as f:
        for key, value in summary.items():
            f.write(f"{key}: {value}\n")

    return df, summary


# Пример использования:
if __name__ == "__main__":
    # Создаем аксон один раз
    axon = MRGaxon(
        fiber_diameter=10.0,
        parent_axon_nodes=42,
        branch_nodes=21,
        branches_num=2,
        nodes_dist=10,
        diam_scale=0.6,
        celsius=37.0,
        dt_ms=0.05,
        v_init=-80.0,
        h_stop=10000.0
    )

    # Определяем параметры для сканирования
    freq_list = [10, 20, 50, 100, 200, 400, 1000]  # Гц
    amp_list = [0.1, 0.5, 1.0, 1.5]  # нА

    # Запускаем сканирование параметров
    results_df, summary = run_stimulation_sweep(
        axon=axon,
        freq_list=freq_list,
        amp_list=amp_list,
        output_dir="my_sweep_results",
        threshold=-20.0
    )

    # Выводим результаты
    print("\nРезультаты сканирования:")
    print(results_df)
