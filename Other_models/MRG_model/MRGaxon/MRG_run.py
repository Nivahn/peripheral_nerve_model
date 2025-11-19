from neuron import h
import matplotlib.pyplot as plt
import numpy as np
import random
import math
from mpl_toolkits.mplot3d import Axes3D
import os

h.load_file('stdrun.hoc')

MRG_TABLE = {
    5.7: (0.605, 3.4, 1.9, 1.9, 3.4, 500, 35, 80),
    7.3: (0.630, 4.6, 2.4, 2.4, 4.6, 750, 38, 100),
    8.7: (0.661, 5.8, 2.8, 2.8, 5.8, 1000, 40, 110),
    10.0: (0.690, 6.9, 3.3, 3.3, 6.9, 1150, 46, 120),
    11.5: (0.700, 8.1, 3.7, 3.7, 8.1, 1250, 50, 130),
    12.8: (0.719, 9.2, 4.2, 4.2, 9.2, 1350, 54, 135),
    14.0: (0.739, 10.4, 4.7, 4.7, 10.4, 1400, 56, 140),
    15.0: (0.767, 11.5, 5.0, 5.0, 11.5, 1450, 58, 145),
    16.0: (0.791, 12.7, 5.5, 5.5, 12.7, 1500, 60, 150),
}


class MRGAxon:
    """Класс для построения и управления MRG-моделью аксона с ветвлениями."""

    # Константы класса
    NODAL_LENGTH = 1.0  # длина узла (ум)
    PARALENGTH1 = 3.0  # MYSA длина (ум)
    SPACE_P1 = 0.002  # периаксональные зазоры вокруг узла/MYSA
    SPACE_P2 = 0.004  # периаксональные зазоры вокруг FLUT
    SPACE_I = 0.004  # периаксональные зазоры вокруг STIN

    # Электрические константы MRG
    RHO_A = 0.7e6  # Ohm·um
    MY_CM = 0.1  # uF/cm2 per lamella
    MY_GM = 0.001  # S/cm2 per lamella

    def __init__(self,
                 fiber_diameter=10.0,
                 axon_length_nodes=42,
                 branch_nodes=21,
                 num_branches=2,
                 branch_spacing=10,
                 diam_scale=0.6,
                 celsius=37.0,
                 dt_ms=0.05,
                 v_init=-80.0):
        """
        Инициализация MRG аксона с настраиваемыми параметрами.

        Args:
            fiber_diameter: Диаметр волокна (ум)
            axon_length_nodes: Длина основного аксона в узлах
            branch_nodes: Длина ветвей в узлах
            num_branches: Количество ветвлений
            branch_spacing: Расстояние между ветвлениями в узлах
            diam_scale: Коэффициент масштабирования диаметра для ветвей
            celsius: Температура (°C)
            dt_ms: Шаг по времени (мс)
            v_init: Начальный потенциал (мВ)
        """
        self.reset_model()

        # Параметры морфологии
        self.fiber_diameter = fiber_diameter
        self.axon_length_nodes = axon_length_nodes
        self.branch_nodes = branch_nodes
        self.num_branches = num_branches
        self.branch_spacing = branch_spacing
        self.diam_scale = diam_scale

        # Параметры симуляции
        self.celsius = celsius
        self.dt_ms = dt_ms
        self.v_init = v_init

        # Установка глобальных параметров NEURON
        h.celsius = celsius
        h.dt = dt_ms

        # Реестры секций
        self.regions = {
            "node": [],
            "mysa": [],
            "flut": [],
            "stin": []
        }

        # Счётчики ID
        self._node_id = 0
        self._mysa_id = 0
        self._flut_id = 0
        self._stin_id = 0

        # Структуры аксона
        self.main_axon = []
        self.branches = []  # Список ветвей, каждая ветвь - список узлов
        self.terminals = []

        # Параметры стимуляции по умолчанию
        self.stimulation_params = {
            'freq_hz': 10,
            'amp': 0.1,
            't_start': 200.0,
            't_end': 1000.0,
            'pulse_len_ms': 1.0,
            'phase_ms': 0.0
        }
        h.tstop = 1000.0

        # Механизм узла
        self.node_mech = self._pick_node_mech()

        # Получение параметров MRG
        self.mrg_params = self._get_mrg_params(fiber_diameter)

        # Построение аксона
        self._build_axon()

    def set_stimulation_parameters(self, freq_hz=None, amp=None, t_start=None,
                                   t_end=None, pulse_len_ms=None, phase_ms=None):
        """Устанавливает параметры стимуляции."""
        if freq_hz is not None:
            self.stimulation_params['freq_hz'] = freq_hz
        if amp is not None:
            self.stimulation_params['amp'] = amp
        if t_start is not None:
            self.stimulation_params['t_start'] = t_start
        if t_end is not None:
            self.stimulation_params['t_end'] = t_end
            h.tstop = t_end
        if pulse_len_ms is not None:
            self.stimulation_params['pulse_len_ms'] = pulse_len_ms
        if phase_ms is not None:
            self.stimulation_params['phase_ms'] = phase_ms

    def reset_model(self):
        """Удаляет все секции и сбрасывает состояние модели."""
        h('forall delete_section()')

    def _pick_node_mech(self):
        """Автовыбор доступного нодального механизма."""
        tmp = h.Section()
        try:
            tmp.insert('newaxnode')
            tmp.uninsert('newaxnode')
            return 'newaxnode'
        except:
            pass
        try:
            tmp.insert('axnode')
            tmp.uninsert('axnode')
            return 'axnode'
        except:
            pass
        raise RuntimeError("Не найден ни 'newaxnode', ни 'axnode' — скомпилируйте .mod файлы.")

    def _get_mrg_params(self, fiber_diameter):
        """Получает параметры MRG для заданного диаметра волокна."""
        if fiber_diameter not in MRG_TABLE:
            raise ValueError(f"fiberD {fiber_diameter} не в таблице.")

        g, axon_d, node_d, para_d1, para_d2, deltax, paralength2, nl = MRG_TABLE[fiber_diameter]
        interlength = (deltax - self.NODAL_LENGTH - 2 * self.PARALENGTH1 - 2 * paralength2) / 6.0

        return {
            'fiberD': fiber_diameter,
            'axonD': axon_d,
            'nodeD': node_d,
            'paraD1': para_d1,
            'paraD2': para_d2,
            'paral1': self.PARALENGTH1,
            'paral2': paralength2,
            'interL': interlength,
            'nl': nl,
            'Rpn0': self._rin_peri(node_d, self.SPACE_P1),
            'Rpn1': self._rin_peri(para_d1, self.SPACE_P1),
            'Rpn2': self._rin_peri(para_d2, self.SPACE_P2),
            'Rpx': self._rin_peri(axon_d, self.SPACE_I),
            'Lstep': 2 * self.PARALENGTH1 + 2 * paralength2 + 6 * interlength + self.NODAL_LENGTH
        }

    def _rin_peri(self, inner_d_um, gap_um):
        """Продольное сопротивление периаксонального пространства."""
        return (self.RHO_A * 0.01) / (math.pi * (((inner_d_um / 2 + gap_um) ** 2) - (inner_d_um / 2) ** 2))

    def _insert_mechanism(self, sec, mech_name):
        """Вставляет механизм в секцию, если его ещё нет."""
        if int(h.ismembrane(mech_name, sec=sec)) == 0:
            sec.insert(mech_name)

    def _set_extracellular(self, sec, xraxial, xg, xc):
        """Устанавливает extracellular-параметры."""
        self._insert_mechanism(sec, 'extracellular')
        for seg in sec:
            seg.xraxial[0] = xraxial
            seg.xg[0] = xg
            seg.xc[0] = xc

    def _create_node(self, node_diameter, rho_a, rpn0, gnabar=3.0, gnapbar=0.005, el=-90.0):
        """Создает узловую секцию."""
        s = h.Section(name=f'node_{self._node_id}')
        self._node_id += 1

        s.nseg = 1
        s.L = self.NODAL_LENGTH
        s.diam = node_diameter
        s.Ra = rho_a / 10000.0
        s.cm = 2.0

        self._insert_mechanism(s, self.node_mech)

        if self.node_mech == 'newaxnode':
            s.el_newaxnode = el
            s.gnabar_newaxnode = gnabar
            s.gnapbar_newaxnode = gnapbar

        self._set_extracellular(s, rpn0, 1e10, 0.0)
        self.regions["node"].append(s)
        return s

    def _create_mysa(self, fiber_diameter, para_d1, paral1, nl, rpn1):
        """Создает MYSA секцию."""
        s = h.Section(name=f'MYSA_{self._mysa_id}')
        self._mysa_id += 1

        s.nseg = 1
        s.L = paral1
        s.diam = fiber_diameter
        ratio = para_d1 / fiber_diameter
        s.Ra = self.RHO_A * (1.0 / (ratio * ratio)) / 10000.0
        s.cm = 2.0 * ratio

        self._insert_mechanism(s, 'pas')
        s.g_pas = 0.001 * ratio
        s.e_pas = -80.0

        self._set_extracellular(s, rpn1, self.MY_GM / (nl * 2.0), self.MY_CM / (nl * 2.0))
        self.regions["mysa"].append(s)
        return s

    def _create_flut(self, fiber_diameter, para_d2, paral2, nl, rpn2):
        """Создает FLUT секцию."""
        s = h.Section(name=f'FLUT_{self._flut_id}')
        self._flut_id += 1

        s.nseg = 1
        s.L = paral2
        s.diam = fiber_diameter
        ratio = para_d2 / fiber_diameter
        s.Ra = self.RHO_A * (1.0 / (ratio * ratio)) / 10000.0
        s.cm = 2.0 * ratio

        self._insert_mechanism(s, 'pas')
        s.g_pas = 0.0001 * ratio
        s.e_pas = -80.0

        self._set_extracellular(s, rpn2, self.MY_GM / (nl * 2.0), self.MY_CM / (nl * 2.0))
        self.regions["flut"].append(s)
        return s

    def _create_stin(self, fiber_diameter, axon_diameter, inter_length, nl, rpx):
        """Создает STIN секцию."""
        s = h.Section(name=f'STIN_{self._stin_id}')
        self._stin_id += 1

        s.nseg = 1
        s.L = inter_length
        s.diam = fiber_diameter
        ratio = axon_diameter / fiber_diameter
        s.Ra = self.RHO_A * (1.0 / (ratio * ratio)) / 10000.0
        s.cm = 2.0 * ratio

        self._insert_mechanism(s, 'pas')
        s.g_pas = 0.0001 * ratio
        s.e_pas = -80.0

        self._set_extracellular(s, rpx, self.MY_GM / (nl * 2.0), self.MY_CM / (nl * 2.0))
        self.regions["stin"].append(s)
        return s

    def _append_step(self, parent_node, params):
        """Добавляет один шаг MRG (между узлами): MYSA→FLUT→STIN×6→FLUT→MYSA→node."""
        p = params

        # Создание секций
        mysa0 = self._create_mysa(p['fiberD'], p['paraD1'], p['paral1'], p['nl'], p['Rpn1'])
        flut0 = self._create_flut(p['fiberD'], p['paraD2'], p['paral2'], p['nl'], p['Rpn2'])
        stin_sections = [self._create_stin(p['fiberD'], p['axonD'], p['interL'], p['nl'], p['Rpx']) for _ in range(6)]
        flut1 = self._create_flut(p['fiberD'], p['paraD2'], p['paral2'], p['nl'], p['Rpn2'])
        mysa1 = self._create_mysa(p['fiberD'], p['paraD1'], p['paral1'], p['nl'], p['Rpn1'])
        next_node = self._create_node(p['nodeD'], self.RHO_A, p['Rpn0'])

        # Соединение секций
        mysa0.connect(parent_node, 1.0, 0.0)
        flut0.connect(mysa0, 1.0, 0.0)
        stin_sections[0].connect(flut0, 1.0, 0.0)

        for i in range(1, 6):
            stin_sections[i].connect(stin_sections[i - 1], 1.0, 0.0)

        flut1.connect(stin_sections[5], 1.0, 0.0)
        mysa1.connect(flut1, 1.0, 0.0)
        next_node.connect(mysa1, 1.0, 0.0)

        return next_node

    def _build_chain(self, n_nodes, params):
        """Строит цепочку из n_nodes узлов."""
        if n_nodes < 1:
            return []

        nodes = [self._create_node(params['nodeD'], self.RHO_A, params['Rpn0'])]

        for _ in range(n_nodes - 1):
            next_node = self._append_step(nodes[-1], params)
            nodes.append(next_node)

        return nodes

    def _get_scaled_params(self, params, scale_factor=0.6):
        """Возвращает параметры с масштабированными диаметрами."""
        scaled = params.copy()

        # Масштабирование диаметров
        scaled['fiberD'] *= scale_factor
        scaled['axonD'] *= scale_factor
        scaled['nodeD'] *= scale_factor
        scaled['paraD1'] *= scale_factor
        scaled['paraD2'] *= scale_factor

        # Пересчет сопротивлений
        scaled['Rpn0'] = self._rin_peri(scaled['nodeD'], self.SPACE_P1)
        scaled['Rpn1'] = self._rin_peri(scaled['paraD1'], self.SPACE_P1)
        scaled['Rpn2'] = self._rin_peri(scaled['paraD2'], self.SPACE_P2)
        scaled['Rpx'] = self._rin_peri(scaled['axonD'], self.SPACE_I)

        return scaled

    def _build_axon(self):
        """Строит полную структуру аксона с ветвлениями."""
        # Начальный узел
        self.main_axon = [self._create_node(
            self.mrg_params['nodeD'], self.RHO_A, self.mrg_params['Rpn0']
        )]

        nodes_count = 0
        remaining_branches = self.num_branches
        self.branches = []  # Список для хранения всех ветвей

        # Построение основного аксона с ветвлениями
        for i in range(self.axon_length_nodes - 1):
            # Определяем параметры для следующего шага
            if i < 3 and remaining_branches > 0:  # Первые 3 узла после ветвления - уменьшенный диаметр
                current_params = self._get_scaled_params(self.mrg_params, self.diam_scale)
            else:
                current_params = self.mrg_params

            # Добавляем следующий узел
            next_node = self._append_step(self.main_axon[-1], current_params)
            self.main_axon.append(next_node)
            nodes_count += 1

            # Проверяем, нужно ли создавать ветвление
            if (remaining_branches > 0 and nodes_count >= self.branch_spacing and
                    len(self.main_axon) < self.axon_length_nodes - 1):

                print(f"Создание ветвления на узле {len(self.main_axon)}")

                # Создаем ветвь
                branch_params = self._get_scaled_params(self.mrg_params, self.diam_scale)
                branch_chain = self._build_chain(self.branch_nodes, branch_params)

                # Подключаем ветвь к текущему узлу
                if branch_chain:
                    branch_chain[0].connect(self.main_axon[-1], 0.0, 1.0)
                    self.branches.append(branch_chain)

                nodes_count = 0
                remaining_branches -= 1

        self._create_stimulator()
        print(f"Построен аксон с {len(self.main_axon)} узлами и {len(self.branches)} ветвями")

    def get_key_nodes(self):
        """Возвращает ключевые узлы для записи сигналов."""
        key_nodes = {}

        # Начальный стимулируемый узел
        if len(self.main_axon) > 0:
            key_nodes['stimulated'] = self.main_axon[0]

        # Узел за 3 ноды до ветвления (последний ветвящийся узел)
        if len(self.branches) > 0 and len(self.main_axon) > 3:
            branch_point_index = len(self.main_axon) - 3
            if branch_point_index >= 0:
                key_nodes['pre_branch'] = self.main_axon[branch_point_index]

        # Узлы на 3ей ноде после ветвления для каждой ветви
        key_nodes['branch_nodes'] = []
        for branch in self.branches:
            if len(branch) > 3:
                key_nodes['branch_nodes'].append(branch[2])  # 3-й узел (индекс 2)

        return key_nodes

    def _create_stimulator(self):
        """Создает стимулятор в первом узле."""
        if self.main_axon:
            self.stimulator = h.IClamp(self.main_axon[0](0.5))
        else:
            raise RuntimeError("Аксон не построен, невозможно создать стимулятор")

    def get_key_nodes(self):
        """Возвращает ключевые узлы для записи сигналов."""
        key_nodes = {}

        # Начальный стимулируемый узел
        if len(self.main_axon) > 0:
            key_nodes['stimulated'] = self.main_axon[0]

        # Узел за 3 ноды до конца основного аксона (если есть ветвления)
        if len(self.main_axon) > 3:
            pre_branch_index = len(self.main_axon) - 4  # 4-й с конца узел
            if pre_branch_index >= 0:
                key_nodes['pre_branch'] = self.main_axon[pre_branch_index]

        # Узлы на 3ей ноде после ветвления для каждой ветви
        key_nodes['branch_nodes'] = []
        for branch in self.branches:
            if len(branch) > 2:  # 3-й узел имеет индекс 2
                key_nodes['branch_nodes'].append(branch[2])

        return key_nodes

    def set_stimulation_parameters(self, freq_hz=None, amp=None, t_start=None,
                                   t_end=None, pulse_len_ms=None, phase_ms=None):
        """Устанавливает параметры стимуляции и обновляет h.tstop."""
        if freq_hz is not None:
            self.stimulation_params['freq_hz'] = freq_hz
        if amp is not None:
            self.stimulation_params['amp'] = amp
        if t_start is not None:
            self.stimulation_params['t_start'] = t_start
        if t_end is not None:
            self.stimulation_params['t_end'] = t_end
            h.tstop = t_end  # Важно обновить h.tstop!
        if pulse_len_ms is not None:
            self.stimulation_params['pulse_len_ms'] = pulse_len_ms
        if phase_ms is not None:
            self.stimulation_params['phase_ms'] = phase_ms

    def run_simulation(self, stimulation_params=None):
        """Запускает симуляцию с заданными параметрами стимуляции."""
        # Сбрасываем модель перед каждой симуляцией
        self._reset_simulation_state()

        if stimulation_params:
            self.set_stimulation_parameters(**stimulation_params)

        if not self.stimulator:
            raise RuntimeError("Стимулятор не создан")

        # Получаем ключевые узлы для записи
        key_nodes = self.get_key_nodes()

        # Создание векторов для записи данных
        time_vec = h.Vector()
        voltage_data = {}

        # Запись потенциалов в ключевых узлах
        for node_name, node in key_nodes.items():
            if node_name == 'branch_nodes':
                for i, branch_node in enumerate(node):
                    v_vec = h.Vector()
                    v_vec.record(branch_node(0.5)._ref_v)
                    voltage_data[f'branch_{i}'] = v_vec
            else:
                v_vec = h.Vector()
                v_vec.record(node(0.5)._ref_v)
                voltage_data[node_name] = v_vec

        time_vec.record(h._ref_t)

        # Генерация стимулирующего тока с текущими параметрами
        stim = self.stimulation_params
        dt_ms = 0.01
        t_points = np.arange(0, stim['t_end'] + dt_ms, dt_ms)
        i_points = np.zeros(len(t_points))

        T_ms = 1000.0 / stim['freq_hz']
        n_pulses = int(np.floor((stim['t_end'] - (stim['t_start'] + stim['phase_ms'])) / T_ms)) + 1

        print(f"Параметры стимуляции: {stim['amp']} нА, {stim['freq_hz']} Гц, {n_pulses} импульсов")

        for k in range(n_pulses):
            t0 = stim['t_start'] + stim['phase_ms'] + k * T_ms
            t1 = t0 + stim['pulse_len_ms']
            if t0 > stim['t_end']:
                break
            mask = (t_points >= t0) & (t_points < t1)
            i_points[mask] = stim['amp']

        # Удаляем старый стимулятор и создаем новый
        if hasattr(self, 'current_stim_vec'):
            del self.current_stim_vec
        if hasattr(self, 'time_stim_vec'):
            del self.time_stim_vec

        # Создаем новые векторы для стимуляции
        self.time_stim_vec = h.Vector().from_python(t_points)
        self.current_stim_vec = h.Vector().from_python(i_points)
        self.current_stim_vec.play(self.stimulator._ref_amp, self.time_stim_vec, 1)

        # Запуск симуляции с полным сбросом
        h.finitialize(self.v_init)
        h.run()

        # Конвертация результатов
        time_array = np.array(time_vec)
        voltage_results = {}
        for name, v_vec in voltage_data.items():
            voltage_results[name] = np.array(v_vec)

        return time_array, voltage_results

    def _reset_simulation_state(self):
        """Сбрасывает состояние симуляции между запусками."""
        # Сбрасываем все записи векторов
        for vec in h.Vector:
            vec.resize(0)

        # Сбрасываем стимулятор
        if hasattr(self, 'stimulator'):
            self.stimulator.amp = 0

        # Явно вызываем сброс NEURON
        h.fcurrent()
        h.finitialize(self.v_init)



    def plot_and_save_results(self, time_array, voltage_results, save_path=None, stimulation_params=None):
        """Визуализирует и сохраняет результаты симуляции для ключевых узлов."""
        plt.figure(figsize=(12, 8))

        # Определяем стиль линий для разных узлов
        styles = {
            'stimulated': {'color': 'blue', 'label': 'Стимулируемый узел', 'linewidth': 2},
            'pre_branch': {'color': 'green', 'label': 'За 3 узла до ветвления', 'linewidth': 2},
        }

        # Рисуем сигналы из основного аксона
        for node_name in ['stimulated', 'pre_branch']:
            if node_name in voltage_results:
                plt.plot(time_array, voltage_results[node_name],
                         **styles[node_name])

        # Рисуем сигналы из ветвей
        branch_colors = ['red', 'orange', 'purple', 'brown']
        for i in range(len(self.branches)):
            branch_key = f'branch_{i}'
            if branch_key in voltage_results:
                plt.plot(time_array, voltage_results[branch_key],
                         color=branch_colors[i % len(branch_colors)],
                         label=f'Ветвь {i + 1} (3-й узел)',
                         linewidth=2,
                         linestyle='--')

        plt.xlabel("Время (мс)")
        plt.ylabel("Мембранный потенциал (мВ)")

        # Создаем информативный заголовок
        title = "Потенциалы действия в ключевых узлах"
        if stimulation_params:
            title += f"\nСтимуляция: {stimulation_params['amp']} нА, {stimulation_params['freq_hz']} Гц"
        plt.title(title)

        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        # Сохраняем график
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"График сохранен: {save_path}")

        plt.show()

    def plot_morphology_3d(self, save_path=None):
        """Визуализирует 3D морфологию аксона."""
        h.define_shape()

        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')

        colors = {
            "node": "red",
            "mysa": "orange",
            "flut": "blue",
            "stin": "green",
            "other": "gray"
        }

        plotted_labels = set()

        for sec, sec_type in self._iter_sections_by_type():
            n3d = int(h.n3d(sec=sec))
            if n3d < 2:
                continue

            xs = [h.x3d(i, sec=sec) for i in range(n3d)]
            ys = [h.y3d(i, sec=sec) for i in range(n3d)]
            zs = [h.z3d(i, sec=sec) for i in range(n3d)]

            label = sec_type if sec_type not in plotted_labels else None
            if label:
                plotted_labels.add(sec_type)

            ax.plot(xs, ys, zs,
                    color=colors.get(sec_type, "black"),
                    linewidth=2.0,
                    label=label)

        ax.set_xlabel("X (µm)")
        ax.set_ylabel("Y (µm)")
        ax.set_zlabel("Z (µm)")
        ax.set_title("3D Морфология MRG аксона с ветвлениями")
        ax.legend()

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"3D морфология сохранена: {save_path}")

        plt.show()

    def _iter_sections_by_type(self):
        """Итератор по секциям с их типами."""
        for sec in self.regions.get("node", []):
            yield sec, "node"
        for sec in self.regions.get("mysa", []):
            yield sec, "mysa"
        for sec in self.regions.get("flut", []):
            yield sec, "flut"
        for sec in self.regions.get("stin", []):
            yield sec, "stin"

        # Добавление секций не из реестра
        reg_set = set()
        for sec_list in self.regions.values():
            reg_set.update(sec_list)

        for sec in h.allsec():
            if sec not in reg_set:
                yield sec, "other"


def run_stimulation_sweep(axon, parameter_sweep, output_dir="results"):
    """
    Запускает серию симуляций с разными параметрами стимуляции.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    results = []

    for i, params in enumerate(parameter_sweep):
        print(f"\n=== Запуск симуляции {i + 1}/{len(parameter_sweep)} ===")
        print(f"Параметры: амплитуда={params['amp']} нА, частота={params['freq_hz']} Гц")

        # Полный сброс между симуляциями
        axon.reset_model()

        # Пересоздаем аксон (важно для чистого состояния)
        axon._build_axon()

        # Запуск симуляции
        time_array, voltage_results = axon.run_simulation(stimulation_params=params)

        # Сохранение графиков
        save_path = os.path.join(output_dir, f"simulation_{i + 1:02d}_amp{params['amp']}_freq{params['freq_hz']}.png")

        axon.plot_and_save_results(time_array, voltage_results,
                                   save_path=save_path,
                                   stimulation_params=params)

        results.append({
            'parameters': params,
            'time': time_array,
            'voltages': voltage_results,
            'save_path': save_path
        })

        # Принудительная сборка мусора
        import gc
        gc.collect()

    return results


# Пример использования
if __name__ == "__main__":
    # Создание аксона с настраиваемыми параметрами
    axon = MRGAxon(
        fiber_diameter=10.0,
        axon_length_nodes=30,
        branch_nodes=15,
        num_branches=2,
        branch_spacing=8,
        diam_scale=0.6
    )

    # Визуализация морфологии
    axon.plot_morphology_3d("axon_morphology.png")

    # Параметры для тестирования
    parameter_sweep = [
        {'amp': 0.05, 'freq_hz': 10, 't_start': 100, 't_end': 500},
        {'amp': 0.1, 'freq_hz': 10, 't_start': 100, 't_end': 500},
        {'amp': 0.2, 'freq_hz': 10, 't_start': 100, 't_end': 500},
        {'amp': 0.1, 'freq_hz': 20, 't_start': 100, 't_end': 500},
        {'amp': 0.1, 'freq_hz': 50, 't_start': 100, 't_end': 500},
    ]

    # Запуск серии симуляций
    results = run_stimulation_sweep(axon, parameter_sweep, "stimulation_sweep_results")

    print(f"Завершено {len(results)} симуляций")