from neuron import h
import matplotlib.pyplot as plt
import numpy as np


class Ipulse1Stimulator:
    """
    Генерирует последовательность импульсов тока
    """

    def __init__(self, section, position=0.5):
        """
        section: секция для стимуляции
        position: позиция на секции (0-1)
        """
        self.section = section
        self.position = position
        self.stim = None
        self.time_vec = None
        self.current_vec = None

        # Параметры по умолчанию как в MOD-файле
        self.del_val = 0  # ms - задержка
        self.ton = 0  # ms - длительность импульса
        self.toff = 0  # ms - межпульсовый интервал
        self.num = 1  # количество импульсов
        self.amp = 0  # nA - амплитуда

        self._create_stimulator()

    def _create_stimulator(self):
        """Создает стимулятор IClamp"""
        self.stim = h.IClamp(self.section(self.position))
        self.stim.dur = 1e9  # очень длинная длительность, управляем через play

    def set_parameters(self, del_val, ton, toff, num, amp):
        """Устанавливает параметры стимуляции"""
        self.del_val = del_val
        self.ton = ton
        self.toff = toff
        self.num = num
        self.amp = amp

        self._generate_waveform()

    def _generate_waveform(self):
        """Генерирует форму сигнала импульсов"""
        # Рассчитываем общее время симуляции
        total_time = self.del_val + self.num * (self.ton + self.toff) + 100  # +100 мс запас

        # Создаем временной вектор с высоким разрешением
        dt = 0.01  # мс
        t_points = np.arange(0, total_time, dt)
        i_points = np.zeros(len(t_points))

        # Генерируем импульсы
        for i in range(self.num):
            start_time = self.del_val + i * (self.ton + self.toff)
            end_time = start_time + self.ton

            # Находим индексы для этого импульса
            start_idx = int(start_time / dt)
            end_idx = int(end_time / dt)

            # Устанавливаем амплитуду в течение импульса
            if start_idx < len(i_points) and end_idx < len(i_points):
                i_points[start_idx:end_idx] = self.amp

        # Создаем векторы NEURON
        self.time_vec = h.Vector(t_points)
        self.current_vec = h.Vector(i_points)

        # Подключаем к стимулятору
        self.current_vec.play(self.stim._ref_amp, self.time_vec, 1)

    def plot_waveform(self, show_plot=True):
        """Визуализирует форму сигнала"""
        if self.time_vec is None or self.current_vec is None:
            print("Сначала установите параметры с помощью set_parameters()")
            return

        t = np.array(self.time_vec)
        i = np.array(self.current_vec)

        plt.figure(figsize=(12, 4))
        plt.plot(t, i, 'b-', linewidth=2)
        plt.xlabel('Время (мс)')
        plt.ylabel('Ток (нА)')
        plt.title(f'Ipulse1: {self.num} импульсов, Ton={self.ton}мс, Toff={self.toff}мс, Amp={self.amp}нА')
        plt.grid(True, alpha=0.3)

        # Добавляем аннотации
        plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
        plt.axvline(x=self.del_val, color='r', linestyle='--', alpha=0.7, label=f'Задержка: {self.del_val}мс')
        plt.legend()

        if show_plot:
            plt.show()

        return plt.gcf()

    def get_info(self):
        """Возвращает информацию о текущих параметрах"""
        return {
            'del_val': self.del_val,
            'ton': self.ton,
            'toff': self.toff,
            'num': self.num,
            'amp': self.amp,
            'total_duration': self.del_val + self.num * (self.ton + self.toff)
        }

    def __repr__(self):
        info = self.get_info()
        return (f"Ipulse1Stimulator(del={info['del_val']}ms, ton={info['ton']}ms, "
                f"toff={info['toff']}ms, num={info['num']}, amp={info['amp']}nA)")