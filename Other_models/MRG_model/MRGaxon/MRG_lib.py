from neuron import h
import matplotlib.pyplot as plt
import numpy as np
import random
import math
from mpl_toolkits.mplot3d import Axes3D

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


#   fiberD: (g, axonD, nodeD, paraD1, paraD2, deltax, paralength2, nl)


class MRGAxon:
    """Класс для построения и управления MRG-моделью аксона."""

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

    def __init__(self, fiber_diameter=10.0, celsius=37.0, dt_ms=0.05, v_init=-80.0):
        """
        Инициализация MRG аксона.

        Args:
            fiber_diameter: Диаметр волокна (ум)
            celsius: Температура (°C)
            dt_ms: Шаг по времени (мс)
            v_init: Начальный потенциал (мВ)
        """
        self.reset_model()

        # Параметры модели
        self.fiber_diameter = fiber_diameter
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

        # Параметры ветвления
        self.diam_scale = 0.6
        self.parent_axon_nodes = 42
        self.branch_nodes = 21
        self.branches_num = 2
        self.nodes_dist = 10

        # Параметры стимуляции
        self.freq_hz = 10
        self.amp = 0.1
        self.t_start = 200.0
        self.t_end = 1000.0
        self.pulse_len_ms = 1.0
        self.phase_ms = 0.0
        h.tstop = 1000.0

        # Механизм узла
        self.node_mech = self._pick_node_mech()

        # Получение параметров MRG
        self.mrg_params = self._get_mrg_params(fiber_diameter)

        # Построение аксона
        self.main_axon = []
        self.terminals = []
        self.stimulator = None
        self._build_axon()

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
        node_d_after_branching = False
        count_nodes_after_branching = 0

        for _ in range(self.parent_axon_nodes - 1):
            if node_d_after_branching:
                if count_nodes_after_branching > 0:
                    scaled_params = self._get_scaled_params(self.mrg_params, self.diam_scale)
                    next_node = self._append_step(self.main_axon[-1], scaled_params)
                    self.main_axon.append(next_node)
                    count_nodes_after_branching -= 1

                    if count_nodes_after_branching == 0:
                        node_d_after_branching = False
                else:
                    next_node = self._append_step(self.main_axon[-1], self.mrg_params)
                    self.main_axon.append(next_node)
            else:
                next_node = self._append_step(self.main_axon[-1], self.mrg_params)
                self.main_axon.append(next_node)

            nodes_count += 1

            # Создание ветвления
            if nodes_count >= self.nodes_dist and self.branches_num > 0:
                print(f"Ветвление при ноде: {nodes_count}")

                # Узел ветвления
                branch_node = self._create_node(
                    self.mrg_params['nodeD'], self.RHO_A, self.mrg_params['Rpn0']
                )
                branch_node.connect(self.main_axon[-1], 1.0, 0.0)

                # Создание ветви
                branch_params = self._get_scaled_params(self.mrg_params, self.diam_scale)
                terminal_chain = self._build_chain(self.branch_nodes, branch_params)
                terminal_chain[0].connect(branch_node, 0.0, 1.0)

                # Продолжение основного аксона
                continuation_node = self._create_node(
                    self.mrg_params['nodeD'], self.RHO_A, self.mrg_params['Rpn0']
                )
                continuation_node.connect(branch_node, 1.0, 0.0)

                self.terminals.append(branch_node)
                self.main_axon.append(continuation_node)
                node_d_after_branching = True
                count_nodes_after_branching = 3
                self.branches_num -= 1
                nodes_count = 0

        self._check_branching()
        self._create_stimulator()

    def _create_stimulator(self):
        """Создает стимулятор в первом узле."""
        if self.main_axon:
            self.stimulator = h.IClamp(self.main_axon[0](0.5))
        else:
            raise RuntimeError("Аксон не построен, невозможно создать стимулятор")

    def _check_branching(self):
        """Проверяет топологию ветвлений (отладочная функция)."""
        for sec in h.allsec():
            parent = sec.parentseg()
            if parent:
                parent_sec = parent.sec
                children = []
                for child in h.allsec():
                    if hasattr(child, 'parentseg') and child.parentseg():
                        if child.parentseg().sec == sec:
                            children.append(child.name())
                if len(children) > 1:
                    print(f"ВЕТВЛЕНИЕ: {sec.name()} -> {children}")

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

    def plot_morphology_3d(self):
        """Визуализирует 3D морфологию аксона."""
        h.define_shape()

        fig = plt.figure(figsize=(10, 8))
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
        ax.set_title("3D Морфология MRG аксона")
        ax.legend()

        plt.tight_layout()
        plt.show()

    def run_simulation(self):
        """Запускает симуляцию и возвращает результаты."""
        if not self.stimulator:
            raise RuntimeError("Стимулятор не создан")

        # Создание векторов для записи данных
        time_vec = h.Vector()
        voltage_vectors = []

        # Запись потенциалов в узлах
        for i, node in enumerate(self.main_axon):
            v_vec = h.Vector()
            v_vec.record(node(0.5)._ref_v)
            voltage_vectors.append(v_vec)

        time_vec.record(h._ref_t)

        # Генерация стимулирующего тока
        dt_ms = 0.01
        t_points = np.arange(0, self.t_end + dt_ms, dt_ms)
        i_points = np.zeros(len(t_points))

        T_ms = 1000.0 / self.freq_hz
        n_pulses = int(np.floor((self.t_end - (self.t_start + self.phase_ms)) / T_ms)) + 1

        for k in range(n_pulses):
            t0 = self.t_start + self.phase_ms + k * T_ms
            t1 = t0 + self.pulse_len_ms
            if t0 > self.t_end:
                break
            mask = (t_points >= t0) & (t_points < t1)
            i_points[mask] = self.amp

        # Воспроизведение формы тока
        time_stim_vec = h.Vector().from_python(t_points)
        current_stim_vec = h.Vector().from_python(i_points)
        current_stim_vec.play(self.stimulator._ref_amp, time_stim_vec, 1)

        # Запуск симуляции
        h.finitialize(self.v_init)
        h.run()

        # Конвертация результатов
        time_array = np.array(time_vec)
        voltage_matrix = np.array([np.array(v_vec) for v_vec in voltage_vectors])

        return time_array, voltage_matrix

    def plot_results(self, time_array, voltage_matrix):
        """Визуализирует результаты симуляции."""
        plt.figure(figsize=(12, 8))

        # График всех узлов
        plt.subplot(2, 1, 1)
        for i, voltage in enumerate(voltage_matrix):
            plt.plot(time_array, voltage, alpha=0.7, label=f"Узел {i}")

        plt.xlabel("Время (мс)")
        plt.ylabel("Мембранный потенциал (мВ)")
        plt.title("Ответ на стимуляцию в главном аксоне")
        plt.legend(fontsize=8)
        plt.grid(True)

        # График ключевых узлов
        plt.subplot(2, 1, 2)
        if len(voltage_matrix) > 1:
            plt.plot(time_array, voltage_matrix[1], alpha=0.7, label="Первый узел")
        if len(voltage_matrix) > self.nodes_dist + 4:
            plt.plot(time_array, voltage_matrix[self.nodes_dist + 4], alpha=0.7, label="Узел ветвления")
        if len(voltage_matrix) > self.nodes_dist + self.branch_nodes + 4:
            plt.plot(time_array, voltage_matrix[self.nodes_dist + self.branch_nodes + 4],
                     alpha=0.7, label="Узел в ветви")

        plt.xlabel("Время (мс)")
        plt.ylabel("Мембранный потенциал (мВ)")
        plt.title("Ключевые узлы аксона")
        plt.legend()
        plt.grid(True)

        plt.tight_layout()
        plt.show()


# Пример использования
if __name__ == "__main__":
    # Создание и запуск модели
    axon = MRGAxon(fiber_diameter=10.0)

    # Визуализация морфологии
    axon.plot_morphology_3d()

    # Запуск симуляции
    time, voltages = axon.run_simulation()

    # Визуализация результатов
    axon.plot_results(time, voltages)