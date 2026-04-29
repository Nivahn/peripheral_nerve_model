import pandas as pd
from neuron import h
import matplotlib.pyplot as plt
import numpy as np

class STIMULATOR:
    """
    Генератор тока для NEURON, работающий в двух режимах:
    1) 'generate'  – классический поезд прямоугольных импульсов (del, ton, toff, num, amp)
    2) 'from_spikes' – ток формируется по временам спайков (spike_times_ms)
        """

    def __init__(self, section, position=0.5, mode="None"):
        """
        section: секция для стимуляции
        position: позиция на секции (0-1)
        mode: 'generate' или 'from_spikes' (по факту задаётся методами ниже)
        """

        self.section = section
        self.position = position
        self.stim = None
        self.time_vec = None
        self.current_vec = None

        self.mode = mode

        self.total_time_ms = None
        self.dt = 0.05

        # Параметры по умолчанию как в MOD-файле
        self.t_start = 0  # ms - задержка
        self.ton = 0  # ms - длительность импульса
        self.toff = 0  # ms - межпульсовый интервал
        self.num = 1  # количество импульсов
        self.amp = 0  # nA - амплитуда

        self._create_stimulator()

    def _create_stimulator(self):
        """Создает стимулятор IClamp"""
        self.stim = h.IClamp(self.section(self.position))
        self.stim.dur = 1e9  # очень длинная длительность, управляем через play
        self.stim.amp = 0.0
        self.stim.delay = 0.0

    def _set_play_vectors(self, t_points, i_points):
        """Обновляет time_vec и current_vec и подключает их к IClamp"""
        # Обновляем NEURON-векторы
        self.time_vec = h.Vector(t_points)
        self.current_vec = h.Vector(i_points)

        # Подключаем к стимулятору
        self.current_vec.play(self.stim._ref_amp, self.time_vec, 1)

    def set_custom_waveform(self, t_points_ms, i_points_nA, dt=0.01):
        """Устанавливает произвольную форму тока.

        t_points_ms и i_points_nA должны иметь одинаковую длину.
        """
        self.mode = "custom_waveform"
        t_points = np.asarray(t_points_ms, dtype=float)
        i_points = np.asarray(i_points_nA, dtype=float)
        if t_points.ndim != 1 or i_points.ndim != 1 or t_points.size != i_points.size:
            raise ValueError("set_custom_waveform: t_points_ms и i_points_nA должны быть 1D и одинаковой длины")
        if t_points.size < 2:
            raise ValueError("set_custom_waveform: нужно минимум 2 временные точки")
        self.dt = float(dt)
        self.total_time_ms = float(t_points[-1])
        self._set_play_vectors(t_points, i_points)

    # -----------------------------------------------------------------------------
    # РЕЖИМ 1: Генерация импульсов
    # -----------------------------------------------------------------------------
    def set_parameters(self, t_start, n_pulses, amp, dt, phase_ms, gap_ms, T_ms, biphasic=True):
        """
        Устанавливает параметры стимуляции в режиме 'generate' (поезд импульсов).
        del_val: задержка (ms)
        ton: длительность импульса (ms)
        toff: пауза между импульсами (ms)
        num: количество импульсов
        amp: амплитуда (nA)
        """

        self.mode = "generate_biphasic" if biphasic else "generate_monophasic"
        self.T_ms = T_ms
        self.del_val = t_start
        self.n_pulses = n_pulses
        self.amp = amp
        self.dt = dt
        self.t_start = t_start
        self.phase_ms = phase_ms
        self.gap_ms = gap_ms

        self._generate_waveform()

    def _generate_waveform(self, biphasic=True):
        """
        Если biphasic=True:
            +amp → 0 → -amp
        Если biphasic=False:
            +amp → 0
        """

        self.total_time_ms = self.t_start + self.n_pulses * self.T_ms

        t_points = np.arange(0.0, self.total_time_ms, self.dt)
        i_points = np.zeros_like(t_points)

        for pulse in range(self.n_pulses):
            t0 = self.t_start + pulse * self.T_ms

            # + фаза
            t1_start = t0
            t1_end = t0 + self.phase_ms

            mask1 = (t_points >= t1_start) & (t_points < t1_end)
            i_points[mask1] = self.amp

            if self.mode == "generate_biphasic":
                # gap
                t2_start = t1_end + self.gap_ms
                t2_end = t2_start + self.phase_ms

                mask2 = (t_points >= t2_start) & (t_points < t2_end)
                i_points[mask2] = -self.amp

        self._set_play_vectors(t_points, i_points)

    # -----------------------------------------------------------------------------
    # РЕЖИМ 2: Формирование тока по временам спайков
    # -----------------------------------------------------------------------------
    def set_spike_times(self,
                        spike_times_ms,
                        amp=1.0,
                        t_max_ms=None,
                        dt=0.01,
                        phase_ms=0.0004,
                        gap_ms=0.00005):
        """
        Бифазный стим на основе spike_times_ms:
        +amp  ⋅ phase_us
         0   ⋅ gap_us
        -amp ⋅ phase_us
        """

        self.mode = "from_spikes"

        spike_times_ms = np.asarray(spike_times_ms, dtype=float)
        spike_times_ms = spike_times_ms[~np.isnan(spike_times_ms)]
        spike_times_ms = np.sort(spike_times_ms)

        self.spike_times_ms = spike_times_ms
        self.amp = float(amp)
        self.dt = float(dt)

        if spike_times_ms.size == 0:
            self.total_time_ms = t_max_ms if t_max_ms is not None else 100.0
            t_points = np.arange(0.0, self.total_time_ms, self.dt)
            i_points = np.zeros_like(t_points)
            self._set_play_vectors(t_points, i_points)
            return

        # Общее время симуляции
        if t_max_ms is None:
            t_max_ms = spike_times_ms.max() + 5 * (phase_ms + gap_ms)

        self.total_time_ms = t_max_ms

        # Сетка времени
        t_points = np.arange(0.0, t_max_ms, self.dt)
        i_points = np.zeros_like(t_points)

        # --- Бифазный стим на каждый spike ---
        for t0 in spike_times_ms:

            # фаза +
            t1s = t0
            t1e = t0 + phase_ms

            # пауза
            t2s = t1e
            t2e = t2s + gap_ms

            # фаза -
            t3s = t2e
            t3e = t3s + phase_ms

            # индексы
            i1s = int(t1s / dt)
            i1e = int(t1e / dt)
            i3s = int(t3s / dt)
            i3e = int(t3e / dt)

            # первая фаза
            if i1s < len(i_points):
                i_points[i1s:min(i1e, len(i_points))] = amp

            # вторая фаза
            if i3s < len(i_points):
                i_points[i3s:min(i3e, len(i_points))] = -amp

        self._set_play_vectors(t_points, i_points)

    def load_spike_times_from_csv(self,
                                  csv_path,
                                  neuron_index=0,
                                  index_is_one_based=False,
                                  t_max=None,
                                  amp=1.0,
                                  dt=0.05,
                                  phase_ms=0.0004,
                                  gap_ms=0.00005):
        """
        Загружает времена спайков из CSV и настраивает стимуляцию в режиме 'from_spikes'.

        csv_path: путь к CSV без заголовка (каждая колонка – нейрон)
        neuron_index: индекс колонки (0-based по умолчанию)
        index_is_one_based: если True, neuron_index считается с 1
        t_max: максимальное время (в  миллисекундах)
        amp: амплитуда импульса (nA)
        pulse_len_ms: длительность импульса (ms)
        dt: шаг по времени (ms)
        """
        self.neuron_index = neuron_index

        orig_index = neuron_index  # для логов/метаданных
        df = pd.read_csv(csv_path, header=None)

        if index_is_one_based:
            neuron_index = neuron_index - 1

        if neuron_index < 0 or neuron_index >= df.shape[1]:
            raise IndexError(f"neuron_index {neuron_index} вне диапазона [0, {df.shape[1] - 1}]")

        # В CSV время в СЕКУНДАХ → переводим в миллисекунды
        spike_times_sec = df[neuron_index].dropna().values

        # Ограничиваем по времени (в МИЛЛИСЕКУНДАХ)
        if t_max is not None:
            spike_times_sec = spike_times_sec[spike_times_sec * 1000.0 <= t_max]

        spike_times_ms = spike_times_sec * 1000.0

        print(f"[stimulator] Загрузка спайков из '{csv_path}', "
              f"нейрон {orig_index} ({neuron_index + 1}-я колонка), "
              f"спайков: {len(spike_times_ms)}")

        self.set_spike_times(spike_times_ms,
                             amp=amp,
                             t_max_ms=t_max,
                             dt=dt,
                             phase_ms=phase_ms,
                             gap_ms=gap_ms)

    # -----------------------------------------------------------------------------
    # Общая часть: подключение к NEURON и визуализация
    # -----------------------------------------------------------------------------

    def plot_waveform(self, show_plot=True, plot_end=1000):
        """Визуализирует форму сигнала"""
        if self.time_vec is None or self.current_vec is None:
            print("Сначала задайте стимуляцию (set_parameters или set_spike_times / load_spike_times_from_csv).")
            return

        t = np.array(self.time_vec)
        i = np.array(self.current_vec)

        plt.figure(figsize=(12, 4))
        plt.plot(t, i, 'b-', linewidth=2)
        plt.xlabel('Время (мс)')
        plt.ylabel('Ток (нА)')

        if self.mode == "generate":
            plt.title(f'Stim (generate): {self.num} импульсов, Ton={self.ton}мс, '
                      f'Toff={self.toff}мс, Amp={self.amp}нА')
        elif self.mode == "from_spikes":
            n_spikes = 0 if self.spike_times_ms is None else len(self.spike_times_ms)
            plt.title(f'Stim (from_spikes): {n_spikes} спайков, '
                      f'Amp={self.amp}нА, Neuron_idx = {self.neuron_index}')
        else:
            plt.title('Stimulator')

        plt.grid(True, alpha=0.3)
        plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
        plt.legend([f"mode={self.mode}"], loc="upper right")

        if show_plot:
            plt.show()

        return plt.gcf()

    def get_info(self):
        """Возвращает информацию о текущих параметрах"""
        info = {
            'mode': self.mode,
            'amp': self.amp,
            'dt': self.dt,
            'total_duration_ms': self.total_time_ms
        }

        if self.mode == "generate":
            info.update({
                'del_val': self.del_val,
                'ton': self.ton,
                'toff': self.toff,
                'num': self.num,
            })
        elif self.mode == "from_spikes":
            info.update({
                'pulse_len_ms': self.pulse_len_ms,
                'n_spikes': 0 if self.spike_times_ms is None else len(self.spike_times_ms),
            })

        return info

