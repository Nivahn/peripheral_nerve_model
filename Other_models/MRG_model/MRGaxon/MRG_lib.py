from neuron import h
import matplotlib.pyplot as plt
import numpy as np
import math
import time
import os
import pandas as pd
from scipy.signal import find_peaks
from impulse_generator import *
import h5py

MRG_TABLE = {
    5.7:  (0.605, 3.4, 1.9, 1.9, 3.4,  500, 35,  80),
    7.3:  (0.630, 4.6, 2.4, 2.4, 4.6,  750, 38, 100),
    8.7:  (0.661, 5.8, 2.8, 2.8, 5.8, 1000, 40, 110),
    10.0: (0.690, 6.9, 3.3, 3.3, 6.9, 1150, 46, 120),
    11.5: (0.700, 8.1, 3.7, 3.7, 8.1, 1250, 50, 130),
    12.8: (0.719, 9.2, 4.2, 4.2, 9.2, 1350, 54, 135),
    14.0: (0.739,10.4, 4.7, 4.7,10.4, 1400, 56, 140),
    15.0: (0.767,11.5, 5.0, 5.0,11.5, 1450, 58, 145),
    16.0: (0.791,12.7, 5.5, 5.5,12.7, 1500, 60, 150),
}

class MRGaxon:

    h.load_file('stdrun.hoc')
    # ------------------------------------------------------------------------------------
    # ------------------------------ ГЛОБАЛЬНЫЕ ПАРАМЕТРЫ --------------------------------
    # ------------------------------------------------------------------------------------
    celsius = 37.0
    dt_ms   = 0.05
    v_init  = -80.0

    # Электрические константы, как в MRG
    rho_a = 0.7e6  # Ohm·um
    mycm = 0.1    # uF/cm2 per lamella
    mygm = 0.001  # S/cm2 per lamella

    # Периакисональные зазоры (ум)
    space_p1 = 0.002  # вокруг узла/MYSA
    space_p2 = 0.004  # вокруг FLUT
    space_i  = 0.004  # вокруг STIN

    # Геометрия "минимал" MRG
    paralength1 = 3.0   # MYSA длина (ум)
    nodelength  = 1.0   # узел (ум)

    gna_axnode = 3.0 # Проводимость
    bouton_L = 1 # мкм

    def __init__(self,
             fiber_diameter=10.0,
             parent_axon_nodes=42,
             branch_nodes=21,
             branches_num=2,
             nodes_dist=10,
             branch_every_um=None,
             diam_scale=0.6,
             celsius=37.0,
             dt_ms=0.05,
             v_init=-80.0,
             h_stop = 1000.0,
             gnapbar_scale=0.5):

        self.reset_model()

        # Параметры морфологии
        self.fiber_diameter = fiber_diameter

        self.parent_axon_nodes = parent_axon_nodes
        self.branch_nodes = branch_nodes
        self.branches_num = branches_num
        self.nodes_dist = nodes_dist
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
            'mode' : 'None',
            'freq_hz': 10,
            'amp': 0.8,
            't_start': 200.0,
            't_end': 1000.0, #  "ms"
            'phase_us': 0.0,
            'gap_us': 0.0,
            'plot_duration': 100.0,
            'csv_path' : "None",
            'neuron_index': 0,
            "index_is_one_based": False,
            "pulse_len_ms": 1.0,

        }





        self.h_stop = h_stop
        h.tstop = self.h_stop

        # Механизм узла
        self.node_mech = self._pick_node_mech()

        self.gnapbar_scale = gnapbar_scale

        # Получение параметров MRG
        self.mrg_params = self._get_mrg_params(fiber_diameter)\

        self.branch_every_um = branch_every_um
        if self.branch_every_um is not None:
            # Lstep — длина одного "шага" node->node в MRG (в мкм)
            self.nodes_dist = max(1, int(round(self.branch_every_um / self.mrg_params['Lstep'])))

        # (опционально) полезно знать фактический шаг ветвления в мкм после округления
        self.branch_every_um_effective = self.nodes_dist * self.mrg_params['Lstep']

        self.print_all_parameters()
        # Построение аксона
        self.build_axon()

    # ------------------------------------------------------------------------------------
    # ------------------------------ СБОРКА МОДЕЛИ ------ --------------------------------
    # ------------------------------------------------------------------------------------

    def reset_model(self):
        # удалить все секции из ядра NEURON
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

    def _get_mrg_params(self, fiberD):
        """Получает параметры MRG для заданного диаметра волокна."""
        if fiberD not in MRG_TABLE:
            raise ValueError(f"fiberD {fiberD} не в таблице.")

        g, axon_d, node_d, para_d1, para_d2, deltax, paralength2, nl = MRG_TABLE[fiberD]
        interlength = (deltax - self.nodelength - 2 * self.paralength1 - 2 * paralength2) / 6.0

        return {
            'fiberD': fiberD,
            'axonD': axon_d,
            'nodeD': node_d,
            'paraD1': para_d1,
            'paraD2': para_d2,
            'paral1': self.paralength1,
            'paral2': paralength2,
            'interL': interlength,
            'nl': nl,
            'rpn0': self._rin_peri(node_d, self.space_p1),
            'rpn1': self._rin_peri(para_d1, self.space_p1),
            'rpn2': self._rin_peri(para_d2, self.space_p2),
            'rpx': self._rin_peri(axon_d, self.space_i),
            'Lstep': 2 * self.paralength1 + 2 * paralength2 + 6 * interlength + self.nodelength
        }

    def _rin_peri(self, inner_d_um, gap_um):
        """Продольное сопротивление периаксонального пространства."""
        return (self.rho_a*0.01)/(math.pi*(((inner_d_um/2+gap_um)**2) - (inner_d_um/2)**2))

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

    # ---------- КОНСТРУКТОРЫ СЕКЦИЙ ----------
    def make_node(self, nodeD, nodel, Rpn0, gnabar=3.0, gnapbar=0.005, el=-90.0):

        s = h.Section(name=f'node_{self._node_id}')
        self._node_id +=1
        s.nseg = 1
        s.L    = nodel
        s.diam = nodeD
        s.Ra   = self.rho_a/10000.0
        s.cm   = 2.0

        self._insert_mechanism(s, self.node_mech)

        if self.node_mech == 'newaxnode':
            s.el_newaxnode      = -90.0 if el is None else el
            s.gnabar_newaxnode  = 3.0   if gnabar  is None else gnabar
            s.gnapbar_newaxnode = gnapbar * self.gnapbar_scale if gnapbar is None else gnapbar
        self._set_extracellular(s, Rpn0, 1e10, 0.0)
        self.regions["node"].append(s)
        return s

    def make_mysa(self, fiberD, paraD1, paral1, nl, rpn1):

        s = h.Section(name=f'MYSA_{self._mysa_id}')

        self._mysa_id += 1

        s.nseg = 1
        s.L    = paral1
        s.diam = fiberD
        ratio  = paraD1/fiberD
        s.Ra   = self.rho_a*(1.0/(ratio*ratio))/10000.0
        s.cm   = 2.0*ratio

        self._insert_mechanism(s, 'pas')
        s.g_pas = 0.001*ratio
        s.e_pas = -80.0

        self._set_extracellular(s, rpn1, self.mygm / (nl * 2.0), self.mycm / (nl * 2.0))
        self.regions["mysa"].append(s)
        return s

    def make_flut(self, fiberD, paraD2, paral2, nl, rpn2):

        s = h.Section(name=f'FLUT_{self._flut_id}')
        self._flut_id += 1

        s.nseg = 1
        s.L    = paral2
        s.diam = fiberD
        ratio  = paraD2/fiberD
        s.Ra   = self.rho_a*(1.0/(ratio*ratio))/10000.0
        s.cm   = 2.0*ratio

        self._insert_mechanism(s, 'pas')
        s.g_pas = 0.0001*ratio
        s.e_pas = -80.0
        self._set_extracellular(s, rpn2, self.mygm / (nl * 2.0), self.mycm / (nl * 2.0))
        self.regions["flut"].append(s)

        return s

    def make_stin(self, fiberD, axonD, interL, nl, rpx):

        s = h.Section(name=f'STIN_{self._stin_id}')
        self._stin_id += 1

        s.nseg = 1
        s.L    = interL
        s.diam = fiberD
        ratio  = axonD/fiberD
        s.Ra   = self.rho_a*(1.0/(ratio*ratio))/10000.0
        s.cm   = 2.0*ratio
        self._insert_mechanism(s, 'pas')
        s.g_pas = 0.0001*ratio
        s.e_pas = -80.0
        self._set_extracellular(s, rpx, self.mygm / (nl * 2.0), self.mycm / (nl * 2.0))
        self.regions["stin"].append(s)
        return s

    # ------------------------------------------------------------------------------------
    # ---------- ОДИН ШАГ MRG (между узлами): MYSA→FLUT→STIN×6→FLUT→MYSA→node ------------
    # ------------------------------------------------------------------------------------

    def append_one_step(self, parent_node, params):

        mysa0 = self.make_mysa(params['fiberD'], params['paraD1'], params['paral1'], params['nl'], params['rpn1'])
        flut0 = self.make_flut(params['fiberD'], params['paraD2'], params['paral2'], params['nl'], params['rpn2'])
        stin_sections = [self.make_stin(params['fiberD'], params['axonD'], params['interL'], params['nl'], params['rpx']) for _ in range(6)]
        flut1 = self.make_flut(params['fiberD'], params['paraD2'], params['paral2'], params['nl'], params['rpn2'])
        mysa1 = self.make_mysa(params['fiberD'], params['paraD1'], params['paral1'], params['nl'], params['rpn1'])
        next_node   = self.make_node(params['nodeD'], self.nodelength, params['rpn0'])

        # топология
        mysa0.connect(parent_node, 1.0, 0.0)
        flut0.connect(mysa0, 1.0, 0.0)
        stin_sections[0].connect(flut0,       1.0, 0.0)
        for k in range(1,6):
            stin_sections[k].connect(stin_sections[k-1], 1.0, 0.0)
        flut1.connect(stin_sections[5],       1.0, 0.0)
        mysa1.connect(flut1,       1.0, 0.0)
        next_node.connect(mysa1,         1.0, 0.0)

        return next_node

    def build_chain(self, n_nodes, params, node_mech=None):

        nodes = [self.make_node(params['nodeD'], self.nodelength, params['Rpn0'])]
        for _ in range(n_nodes-1):
            next_node = self.append_one_step(nodes[-1], params)
            nodes.append(next_node)

        return nodes

    def scaled_params(self, params, diam_scale=0.6):
        """Сужение диаметров после ветвления."""
        scaled = params.copy()

        scaled['fiberD'] *= diam_scale
        scaled['axonD']  *= diam_scale
        scaled['nodeD']  *= diam_scale
        scaled['paraD1'] *= diam_scale
        scaled['paraD2'] *= diam_scale

        scaled['Rpn0'] = self._rin_peri(scaled['nodeD'],  self.space_p1)
        scaled['Rpn1'] = self._rin_peri(scaled['paraD1'], self.space_p1)
        scaled['Rpn2'] = self._rin_peri(scaled['paraD2'], self.space_p2)
        scaled['Rpx']  = self._rin_peri(scaled['axonD'],  self.space_i)

        return scaled


    def build_axon(self):

        terminals = []
        params = self.mrg_params

        self.main_axon = [self.make_node(self.mrg_params['nodeD'], self.nodelength, self.mrg_params['rpn0'])]

        self.node_distance_um = {}
        total_length_um = 0.0
        self.node_distance_um[self.main_axon[0].name()] = total_length_um

        # >>> INSERT: список расстояний в точках ветвления (мкм)
        self.branch_point_distance_um = []

        node_D_after_branching = False
        count_nodes_after_branching = 0
        nodes = 0

        self.branch_point_id = []
        self.before_branch_id = []
        self.after_branch_main_id = []
        self.after_branch_daughter_id = []

        for _ in range(self.parent_axon_nodes - 1):

            if node_D_after_branching == True:

                P_main_axon = self.scaled_params(params, self.diam_scale)
                nxt = self.append_one_step(self.main_axon[-1], P_main_axon)
                self.main_axon.append(nxt)

                total_length_um += params['Lstep']
                self.node_distance_um[self.main_axon[-1].name()] = total_length_um

                # вставляем 1 шаг со скейлом на 60 %

                count_nodes_after_branching -= 1
                # print(f"count_nodes_after_branching: {count_nodes_after_branching}")
                if count_nodes_after_branching == 0:
                    node_D_after_branching = False
                    self.after_branch_main_id.extend(self.main_axon[-1])

            if node_D_after_branching == False:
                nxt = self.append_one_step(self.main_axon[-1], params)
                self.main_axon.append(nxt)

                # >>> INSERT: обновляем пройденную длину (мкм) до этой ноды
                total_length_um += params['Lstep']
                self.node_distance_um[self.main_axon[-1].name()] = total_length_um

            nodes += 1
            # print(nodes)
            # Надо высчитать длину типичного шага при append one step, но через ноды удобнее.

            if nodes >= self.nodes_dist and self.branches_num != 0:

                # Точка ветвления — текущая последняя нода главного аксона
                branch_node = self.main_axon[-1]

                branch_distance_um = self.node_distance_um.get(branch_node.name(), None)
                if branch_distance_um is not None:
                    self.branch_point_distance_um.append(branch_distance_um)

                # print(f"Ветвление с шагом в: {nodes} нод")

                self.branch_point_id.extend(branch_node)

                if len(self.main_axon) >= 3:
                    self.before_branch_id.extend(self.main_axon[-3])
                else:
                    self.before_branch_id.extend(branch_node)

                #P_branch = self.scaled_params(params, self.diam_scale)
                #term_chain = self.build_chain(self.branch_nodes, P_branch)

                P_base = params
                P_scaled = self.scaled_params(params, self.diam_scale)

                term_chain = []

                # первая нода дочки
                d0 = self.make_node(P_scaled['nodeD'], self.nodelength, P_scaled['rpn0'])
                d0.connect(branch_node, 0.0, 1.0)
                term_chain.append(d0)

                # дальше шаги
                prev = d0
                for i in range(1, self.branch_nodes):
                    P = P_scaled if i < 3 else P_base  # первые 3 после d0 — scaled, дальше base
                    nxt = self.append_one_step(prev, P)
                    term_chain.append(nxt)
                    prev = nxt

                # --- Продолжение основного аксона
                node_3 = self.make_node(params['nodeD'], self.nodelength, params['rpn0'])

                #term_chain[0].connect(branch_node, 0.0, 1.0)  # дочерняя ветвь
                node_3.connect(branch_node, 1.0, 0.0)  # продолжение main

                # Запоминаем "3 ноды после" в дочерней ветке
                if len(term_chain) >= 4:
                    self.after_branch_daughter_id.extend(term_chain[3])
                else:
                    self.after_branch_daughter_id.extend(term_chain[-1])

                # Запоминаем "3 ноды после" в главном аксоне:
                # В твоем коде это делалось через count_nodes_after_branching и append_one_step,
                # поэтому здесь ничего не добавляем вручную — оно заполнится позже в цикле.

                if branch_distance_um is None:
                    print(f"[build_axon] Bifurcation at: {branch_node.name()}")
                else:
                    print(f"[build_axon] Bifurcation at: {branch_node.name()}  |  dist≈{branch_distance_um:.1f} µm")

                print(f"[build_axon] Daughter: {branch_node.name()} -> {term_chain[0].name()}")
                print(f"[build_axon] Main:     {branch_node.name()} -> {node_3.name()}")

                terminals.append(branch_node)

                # main продолжается с node_3
                self.main_axon.append(node_3)

                node_D_after_branching = True
                self.branches_num -= 1
                nodes = 0
                count_nodes_after_branching = 3

        print("[build_axon] Ноды на которых будет вестись запись:")
        print(f"[build_axon] Ветвление в точке: {self.branch_point_id}")
        print(f"[build_axon] 3 ноды до точки ветвления: {self.before_branch_id}")
        print(f"[build_axon] 3 ноды после точки ветвления в главном аксоне: {self.after_branch_main_id}")
        print(f"[build_axon] 3 ноды после точки ветвления в дочерней ветке: {self.after_branch_daughter_id}")
        print(f"[build_axon] Расстояния ветвления (µm): {self.branch_point_distance_um}")
        if hasattr(self, "branch_every_um_effective"):
            print(f"[build_axon] Фактический шаг ветвления (µm) после округления: {self.branch_every_um_effective:.1f}")


    # ------------------------------------------------------------------------------------
    # ------------------------------ СОЗДАНИЕ СТИМУЛЯТОРА --------------------------------
    # ------------------------------------------------------------------------------------
    def set_stimulation_params(self, mode="create", **kwargs):
        """
        Устанавливает параметры стимуляции.
        mode: "create" или "preload_data"

        Для mode="create" ожидаются поля:
            freq_hz, amp, t_start, t_end, del_val, ton, pulse_len_ms (опц.)

        Для mode="preload_data" ожидаются поля:
            csv_path: путь к CSV с таймпоинтами (как у тебя)
            neuron_index: индекс колонки (0-based или 1-based — см. ниже)
            index_is_one_based: bool, если True, neuron_index считает с 1
            t_max: максимальное время в секундах (например, 5.0)
            amp: амплитуда импульса (нА)
            pulse_len_ms: длительность импульса (мс)
        """
        self.stimulation_params = dict(kwargs)
        self.stimulation_params["mode"] = mode

    def create_stimulator(self):
        """
        Создает стимулятор в зависимости от режима:
        - "preload_data": подгрузка спайк-трейна из CSV и подача как ток в первый узел
        - "create": классический стимулятор по частоте

        Параметры берутся из self.stimulation_params (задать через set_stimulation_params).
        """
        mode = self.stimulation_params["mode"]

        if not hasattr(self, "stimulation_params"):
            raise ValueError("Сначала вызовите set_stimulation_params()")

        if mode is None:
            mode = self.stimulation_params.get("mode", "create")

        dt = self.dt_ms
        if mode == "preload_data":
            params = self.stimulation_params

            csv_path = params["csv_path"]
            neuron_index = params["neuron_index"]
            index_is_one_based = params["index_is_one_based"]
            t_end = params["t_end"]
            amp = params["amp"]
            phase_ms = params["phase_us"] / 1000
            gap_ms = params["gap_us"] / 1000


            ipulse_stimulator = STIMULATOR(self.main_axon[0], position=0.5, mode="preload_data")
            ipulse_stimulator.load_spike_times_from_csv(
                csv_path=csv_path,
                neuron_index=neuron_index,
                index_is_one_based=index_is_one_based,
                t_max=t_end,  # t_max в МИЛЛИСЕКУНДАХ
                amp=amp,
                dt=dt,
                phase_ms=phase_ms,
                gap_ms=gap_ms,
            )

            self.stimulator = ipulse_stimulator
            # tstop с запасом
            total_time = t_end # * 1000
            h.tstop = total_time
            self.h_stop = total_time
            ipulse_stimulator.plot_waveform(plot_end=int(10//dt))

        elif mode == "create":
            params = self.stimulation_params

            freq = params["freq_hz"]
            amp = params["amp"]
            t_start = params["t_start"]
            t_end = params["t_end"]

            phase_ms = params["phase_us"] / 1000
            gap_ms = params["gap_us"] / 1000

            bi_width_ms = 2 * phase_ms + gap_ms
            stimulation_duration_ms = t_end - t_start

            # Расчет параметров стимуляции
            T_ms = 1000.0 / freq
            if bi_width_ms > T_ms:
                raise ValueError(
                    f"Бифазный пульс ({bi_width_ms:.3f} ms) длиннее периода ({T_ms:.3f} ms) при freq={freq} Гц"
                )

            n_pulses = int(stimulation_duration_ms / T_ms)

            print(f"[create_stimulator] freq={freq} Гц:")
            print(f"[create_stimulator] Период: {T_ms:.3f} мс")
            print(f"[create_stimulator] Ширина бифазного пакета: {bi_width_ms:.3f} мс "
                  f"[create_stimulator] (2×{phase_ms:.3f} + {gap_ms:.3f})")
            print(f"[create_stimulator] Количество пульсов: {n_pulses}")
            print(f"[create_stimulator] Длительность стимуляции: {stimulation_duration_ms} мс")

            ipulse_stimulator = STIMULATOR(self.main_axon[0], position=0.5, mode="create")
            ipulse_stimulator.set_parameters(t_start, n_pulses, amp, dt, phase_ms, gap_ms, T_ms)

            self.stimulator = ipulse_stimulator
            # tstop с запасом
            total_time = stimulation_duration_ms
            h.tstop = total_time
            self.h_stop = total_time
            #ipulse_stimulator.plot_waveform()

        else:
            raise ValueError(f"[create_stimulator] Неизвестный режим стимуляции: {mode}")

    # ------------------------------------------------------------------------------------
    # ------------------------------ ЗАПУСК СИМУЛЯЦИИ --- --------------------------------
    # ------------------------------------------------------------------------------------

    def run_simulation(self, h5_path=None, experiment_name=None):
        """
        Запускает симуляцию с заданными параметрами стимуляции.
        mode: "preload_data" или "create" (если None — берётся из stimulation_params)
        hdf5_path: путь к HDF5-файлу (если не None — результат сохраняется)
        hdf5_group: имя группы внутри HDF5 (например, "neuron_1_preload")
        """

        # Сбрасываем состояние симуляции (но не морфологию)
        if hasattr(self, "_reset_simulation_state"):
            self._reset_simulation_state()
        else:
            # мягкий reset, если нет специального метода
            h.finitialize(self.v_init)

        # Создаем стимулятор
        self.create_stimulator()

        # Готовим список сегментов для записи
        key_segments = []
        key_names = []

        if hasattr(self, 'before_branch_id') and self.before_branch_id:
            key_segments.extend(self.before_branch_id)
            key_names.extend(['before_branch'] * len(self.before_branch_id))

        if hasattr(self, 'after_branch_main_id') and self.after_branch_main_id:
            key_segments.extend(self.after_branch_main_id)
            key_names.extend(['after_branch_main'] * len(self.after_branch_main_id))

        if hasattr(self, 'after_branch_daughter_id') and self.after_branch_daughter_id:
            key_segments.extend(self.after_branch_daughter_id)
            key_names.extend(['after_branch_daughter'] * len(self.after_branch_daughter_id))

        if hasattr(self, 'branch_point_id') and self.branch_point_id:
            key_segments.extend(self.branch_point_id)
            key_names.extend(['branch_point'] * len(self.branch_point_id))

        # Добавляем точку стимуляции
        if self.main_axon:
            key_segments.append(self.main_axon[0](0.5))
            key_names.append('stimulation_point')

        record_v = []
        record_t = h.Vector().record(h._ref_t)

        self.recording_indices = {}
        for i, (seg, name) in enumerate(zip(key_segments, key_names)):
            vec = h.Vector().record(seg._ref_v)
            record_v.append(vec)

            node_id = f"{seg.sec.name().replace('.', '_')}_{seg.x:.2f}"

            self.recording_indices[i] = {
                "group": name,
                "node": node_id
            }

        # Запуск
        h.finitialize(self.v_init)
        h.run()

        # Сохраняем в numpy
        self.voltage_matrix = np.vstack([np.array(v) for v in record_v])
        self.time_array = np.array(record_t)
        self.recording_labels = key_names  # по оси 0 voltage_matrix

        #print(f"voltage matrix: {self.voltage_matrix.shape}")
        #print(f"time array: {self.time_array.shape}")
        #print(f"recording_labels: {(self.recording_indices)}")

        if h5_path is not None:
            if experiment_name is None:
                experiment_name = "experiment"

            # если стимулятор не передали явно — попробуем взять из self
            stimulator = self.stimulator

            self.save_to_hdf5(
                h5_path=h5_path,
                experiment_name=experiment_name,
                stimulator=stimulator
            )

        return self.time_array, self.voltage_matrix

    # ------------------------------------------------------------------------------------
    # ------------------------------ ГРАФИКИ И СТАТИСТИКА --------------------------------
    # ------------------------------------------------------------------------------------


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

    def plot_morphology_3d(self, save_path=None):
        """Визуализирует 3D морфологию аксона + точки записи."""
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

        # ---------------------------------------------------------------------
        # ДОБАВКА: точки записи (branch/before/after main/after daughter)
        # ---------------------------------------------------------------------

        def _seg_xyz(seg_or_sec):
            """Возвращает (x,y,z) точки вдоль секции в позиции seg.x (или 0.5 если это Section)."""
            # 1) приводим к (sec, x)
            try:
                # если это Segment
                sec = seg_or_sec.sec
                pos = float(seg_or_sec.x)
            except Exception:
                # если это Section
                sec = seg_or_sec
                pos = 0.5

            n3d = int(h.n3d(sec=sec))
            if n3d < 2:
                # fallback: если нет 3D-точек
                return None

            # 2) кумулятивная длина по 3D-точкам
            arc = [h.arc3d(i, sec=sec) for i in range(n3d)]
            xs = [h.x3d(i, sec=sec) for i in range(n3d)]
            ys = [h.y3d(i, sec=sec) for i in range(n3d)]
            zs = [h.z3d(i, sec=sec) for i in range(n3d)]

            L = float(sec.L)

            s = pos * L  # целевая длина вдоль секции

            # 3) находим отрезок, где лежит s, и линейно интерполируем
            for i in range(n3d - 1):
                if arc[i] <= s <= arc[i + 1]:
                    denom = (arc[i + 1] - arc[i])
                    if denom <= 0:
                        t = 0.0
                    else:
                        t = (s - arc[i]) / denom
                    x = xs[i] + t * (xs[i + 1] - xs[i])
                    y = ys[i] + t * (ys[i + 1] - ys[i])
                    z = zs[i] + t * (zs[i + 1] - zs[i])
                    return (x, y, z)

            # если s вне диапазона (редко, но бывает из-за округлений)
            return (xs[-1], ys[-1], zs[-1])

        def _scatter_points(obj_list, label, marker):
            pts = []
            for obj in obj_list:
                xyz = _seg_xyz(obj)
                if xyz is not None:
                    pts.append(xyz)
            if pts:
                X, Y, Z = zip(*pts)
                ax.scatter(X, Y, Z, s=80, marker=marker, depthshade=True, label=label)

        # ВАЖНО: если у тебя списки хранят Sections — тоже ок.
        # Если хранят Segments — тоже ок.
        _scatter_points(getattr(self, "branch_point_id", []), "record: branch", "x")
        _scatter_points(getattr(self, "before_branch_id", []), "record: before", "^")
        _scatter_points(getattr(self, "after_branch_main_id", []), "record: after main", "o")
        _scatter_points(getattr(self, "after_branch_daughter_id", []), "record: after daughter", "s")

        # ---------------------------------------------------------------------

        ax.set_xlabel("X (µm)")
        ax.set_ylabel("Y (µm)")
        ax.set_zlabel("Z (µm)")
        ax.set_title("3D Морфология MRG аксона с ветвлениями (с точками записи)")
        ax.legend()

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"3D морфология сохранена: {save_path}")

        plt.show()

    def plot_voltage_traces(self, save_path=None, plot_start = 0, plot_end = 1000, plot_branch=0 ):
        """Визуализирует потенциалы действия в ключевых точках аксона."""



        # Проверяем, что у нас есть данные для построения
        if not (hasattr(self, 'voltage_matrix') and hasattr(self, 'time_array') and
                hasattr(self, 'recording_indices')):
            print("Нет данных для построения графиков. Сначала запустите run_simulation().")
            return

        # Находим индексы для каждой группы записей
        before_indices = [
            i for i, meta in self.recording_indices.items()
            if meta["group"] == 'before_branch'
        ]
        after_main_indices = [
            i for i, meta in self.recording_indices.items()
            if meta["group"] == 'after_branch_main'
        ]
        after_daughter_indices = [
            i for i, meta in self.recording_indices.items()
            if meta["group"] == 'after_branch_daughter'
        ]

        print(f"Индексы для построения:")
        print(f"  До ветвления: {before_indices}")
        print(f"  После ветвления (основная): {after_main_indices}")
        print(f"  После ветвления (дочерняя): {after_daughter_indices}")

        # Проверяем, что у нас есть данные для построения
        if not (before_indices and after_main_indices and after_daughter_indices):
            print("Недостаточно данных для построения графиков")
            return

        # Берем первые ноды из каждой группы
        before_idx = before_indices[plot_branch]
        after_main_idx = after_main_indices[plot_branch]
        after_daughter_idx = after_daughter_indices[plot_branch]

        # Создаем фигуру с 5 графиками в одном столбце
        fig, axes = plt.subplots(5, 1, figsize=(12, 15))
        title_text = (f"MRG Аксон: диаметр {self.fiber_diameter} мкм, "
                      f"Частота: {self.stimulation_params['freq_hz']} Гц, "
                      f"Сила тока: {self.stimulation_params['amp']} нА")

        fig.suptitle(title_text, fontsize=14, fontweight='bold', y=0.98)
        # График 1: До ветвления
        axes[0].plot(self.time_array[plot_start:plot_end], self.voltage_matrix[before_idx][plot_start:plot_end],
                     label=f'До ветвления (индекс {before_idx})', linewidth=2, color='blue')
        axes[0].set_xlabel("Время (мс)")
        axes[0].set_ylabel("Мембранный потенциал (мВ)")
        axes[0].set_title("До ветвления")
        axes[0].grid(True, alpha=0.3)
        axes[0].legend()

        # График 2: После ветвления (основная ветвь)
        axes[1].plot(self.time_array[plot_start:plot_end], self.voltage_matrix[after_main_idx][plot_start:plot_end],
                     label=f'После ветвления (основная, индекс {after_main_idx})', linewidth=2, color='red')
        axes[1].set_xlabel("Время (мс)")
        axes[1].set_ylabel("Мембранный потенциал (мВ)")
        axes[1].set_title("После ветвления - основная ветвь")
        axes[1].grid(True, alpha=0.3)
        axes[1].legend()

        # График 3: После ветвления (дочерняя ветвь)
        axes[2].plot(self.time_array[plot_start:plot_end], self.voltage_matrix[after_daughter_idx][plot_start:plot_end],
                     label=f'После ветвления (дочерняя, индекс {after_daughter_idx})', linewidth=2, color='green')
        axes[2].set_xlabel("Время (мс)")
        axes[2].set_ylabel("Мембранный потенциал (мВ)")
        axes[2].set_title("После ветвления - дочерняя ветвь")
        axes[2].grid(True, alpha=0.3)
        axes[2].legend()

        # График 4: Все три вместе
        axes[3].plot(self.time_array[plot_start:plot_end], self.voltage_matrix[before_idx][plot_start:plot_end],
                     label=f'До ветвления (индекс {before_idx})', linewidth=2, color='blue')
        axes[3].plot(self.time_array[plot_start:plot_end], self.voltage_matrix[after_main_idx][plot_start:plot_end],
                     label=f'После ветвления (основная, индекс {after_main_idx})', linewidth=2, color='red')
        axes[3].plot(self.time_array[plot_start:plot_end], self.voltage_matrix[after_daughter_idx][plot_start:plot_end],
                     label=f'После ветвления (дочерняя, индекс {after_daughter_idx})', linewidth=2, color='green')
        axes[3].set_xlabel("Время (мс)")
        axes[3].set_ylabel("Мембранный потенциал (мВ)")
        axes[3].set_title("Сравнение потенциалов до и после ветвления")
        axes[3].grid(True, alpha=0.3)
        axes[3].legend()

        self.t_points = np.array(self.stimulator.time_vec)
        self.i_points = np.array(self.stimulator.current_vec)
        # График 5: Стимулы
        if hasattr(self, 't_points') and hasattr(self, 'i_points'):

            axes[4].plot(self.t_points[plot_start:plot_end], self.i_points[plot_start:plot_end], linewidth=2, color='purple')
            axes[4].set_xlabel("Время (мс)")
            axes[4].set_ylabel("Ток стимуляции (нА)")
            axes[4].set_title("Протокол стимуляции")
            axes[4].grid(True, alpha=0.3)
            #axes[4].set_ylim(-1 * self.stimulation_params['amp'] * 1.1, self.stimulation_params['amp'] * 1.1)

        # Настраиваем layout
        plt.tight_layout()

        # Сохраняем если указан путь
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"График сохранен в: {save_path}")

        plt.show()


    def count_spikes(self, voltage_trace, time_array=None, threshold=-20.0, min_peak_distance=2.0):
        """
        Считает количество спайков в voltage_trace.

        Args:
            voltage_trace: массив напряжений
            time_array: массив времени (если None, используется self.time_array)
            threshold: порог для обнаружения спайков (мВ)
            min_peak_distance: минимальное расстояние между пиками (мс)

        Returns:
            spike_count: количество спайков
            spike_times: времена спайков (мс)
        """
        if time_array is None:
            time_array = self.time_array

        # Преобразуем min_peak_distance в количество точек
        min_distance_points = int(min_peak_distance / self.dt_ms)

        try:
            # Находим пики выше порога
            peaks, properties = find_peaks(voltage_trace, height=threshold, distance=min_distance_points)

            spike_times = time_array[peaks] if len(peaks) > 0 else np.array([])
            spike_count = len(peaks)

            return spike_count, spike_times
        except Exception as e:
            print(f"Ошибка при подсчете спайков: {e}")
            return 0, np.array([])

    def analyze_branching_spikes(self, threshold=-20.0):
        """
        Анализирует количество спайков до и после ветвления.

        Returns:
            dict: статистика по спайкам
        """
        if not hasattr(self, 'voltage_matrix') or not hasattr(self, 'recording_indices'):
            raise ValueError("Сначала запустите симуляцию с помощью run_simulation()")

        results = {}

        # Находим индексы для каждой группы записей
        before_indices = [
            i for i, meta in self.recording_indices.items()
            if meta["group"] == 'before_branch'
        ]
        after_main_indices = [
            i for i, meta in self.recording_indices.items()
            if meta["group"] == 'after_branch_main'
        ]
        after_daughter_indices = [
            i for i, meta in self.recording_indices.items()
            if meta["group"] == 'after_branch_daughter'
        ]

        # Анализируем спайки в каждой группе
        for indices, group_name in [(before_indices, 'before_branch'),
                                    (after_main_indices, 'after_branch_main'),
                                    (after_daughter_indices, 'after_branch_daughter')]:
            if indices:
                # Берем первую запись из группы
                voltage_trace = self.voltage_matrix[indices[0]]
                spike_count, spike_times = self.count_spikes(voltage_trace, self.time_array, threshold)

                results[group_name] = {
                    'spike_count': spike_count,
                    'spike_times': spike_times,
                    'conduction_ratio': None  # будет заполнено позже
                }

        # Вычисляем коэффициент проведения через ветвление
        if 'before_branch' in results and 'after_branch_main' in results:
            before_count = results['before_branch']['spike_count']
            after_main_count = results['after_branch_main']['spike_count']

            if before_count > 0:
                results['after_branch_main']['conduction_ratio'] = after_main_count / before_count

        if 'before_branch' in results and 'after_branch_daughter' in results:
            before_count = results['before_branch']['spike_count']
            after_daughter_count = results['after_branch_daughter']['spike_count']

            if before_count > 0:
                results['after_branch_daughter']['conduction_ratio'] = after_daughter_count / before_count



        return results

    def save_to_hdf5(self, h5_path, experiment_name="experiment", stimulator=None):
        """
        Сохраняет стимул и ответ модели в HDF5.

        Структура:

        /experiment_name/
            Stimulator/
                time      (N,)
                current   (N,)
                attrs[...]  – метаданные стимула (mode, amp, dt, ...)
            Model/
                time      (T,)
                Traces/
                    before_branch/
                        trace0
                        trace1
                    branch_point/
                        trace0
                        trace1
                    after_branch_main/
                        trace0
                        trace1
                    after_branch_daughter/
                        trace0
                        trace1
                    stimulation_point/
                        trace0
        """

        if not hasattr(self, "time_array") or not hasattr(self, "voltage_matrix"):
            raise RuntimeError("Нет данных симуляции: сначала вызови run_simulation().")

        with h5py.File(h5_path, "a") as f:
            if experiment_name in f:
                grp_exp = f[experiment_name]
            else:
                grp_exp = f.create_group(experiment_name)

            # --------- Stimulator ---------
            if stimulator is not None:
                grp_stim = grp_exp.require_group("Stimulator")

                # удалим старые данные, если были
                for name in ["time", "current"]:
                    if name in grp_stim:
                        del grp_stim[name]

                if stimulator.time_vec is not None and stimulator.current_vec is not None:
                    t_stim = np.array(stimulator.time_vec)
                    i_stim = np.array(stimulator.current_vec)

                    grp_stim.create_dataset("time", data=t_stim)
                    grp_stim.create_dataset("current", data=i_stim)

                # метаданные стимула как атрибуты
                try:
                    info = stimulator.get_info()
                except Exception:
                    info = {}

                for k, v in info.items():
                    try:
                        grp_stim.attrs[k] = v
                    except TypeError:
                        grp_stim.attrs[k] = str(v)

            # --------- Model: время ---------
            grp_model = grp_exp.require_group("Model")

            if "time" in grp_model:
                del grp_model["time"]
            grp_model.create_dataset("time", data=self.time_array)

            grp_traces = grp_model.require_group("Traces")

            # Упаковываем по группам имён из self.recording_indices
            # voltage_matrix.shape = (n_traces, T)
            for idx, meta in self.recording_indices.items():

                grp_name = meta["group"]  # before_branch / branch_point / ...
                node_name = meta["node"]  # node_36_0.50

                grp_grp = grp_traces.require_group(grp_name)

                dset_name = f"trace_{node_name}"

                # удаляем, если уже существует
                if dset_name in grp_grp:
                    del grp_grp[dset_name]

                dset = grp_grp.create_dataset(dset_name, data=self.voltage_matrix[idx, :])

                # добавим метаданные в атрибуты
                dset.attrs["node"] = node_name
                dset.attrs["group"] = grp_name
                dset.attrs["index_in_matrix"] = idx

            # Немного метаданных по аксону
            grp_model.attrs["fiber_diameter_um"] = self.fiber_diameter
            grp_model.attrs["dt_ms"] = self.dt_ms
            grp_model.attrs["celsius"] = self.celsius
            grp_model.attrs["h_stop_ms"] = getattr(self, "h_stop", np.nan)

            print(f"[save_to_hdf5] Сохранено в {h5_path} под группой '{experiment_name}'")

    def print_all_parameters(self):
        """Выводит все параметры модели для отладки и записи"""
        print("\n" + "=" * 80)
        print("ПАРАМЕТРЫ МОДЕЛИ")
        print("=" * 80)

        print("\n--- ОСНОВНЫЕ ПАРАМЕТРЫ ---")
        print(f"Диаметр волокна: {self.fiber_diameter} мкм")
        print(f"Температура: {self.celsius} °C")
        print(f"Шаг времени: {self.dt_ms} мс")
        print(f"Начальное напряжение: {self.v_init} мВ")
        print(f"Время симуляции: {self.h_stop} мс")

        print("\n--- ПАРАМЕТРЫ МОРФОЛОГИИ ---")
        print(f"Узлов в основном аксоне: {self.parent_axon_nodes}")
        print(f"Узлов в ветви: {self.branch_nodes}")
        print(f"Количество ветвей: {self.branches_num}")
        print(f"Расстояние между ветвлениями: {self.nodes_dist} сегментов")
        print(f"Масштаб диаметра: {self.diam_scale}")
        print(f"Шаг ветвления (запрошенный): {self.branch_every_um} мкм")
        print(f"Шаг ветвления (фактический): {self.branch_every_um_effective:.1f} мкм")

        print("\n--- ПАРАМЕТРЫ MRG ДЛЯ ДАННОГО ДИАМЕТРА ---")
        print(f"Диаметр аксона: {self.mrg_params['axonD']} мкм")
        print(f"Диаметр узла: {self.mrg_params['nodeD']} мкм")
        print(f"Диаметр MYSA: {self.mrg_params['paraD1']} мкм")
        print(f"Диаметр FLUT: {self.mrg_params['paraD2']} мкм")
        print(f"Длина MYSA: {self.mrg_params['paral1']} мкм")
        print(f"Длина FLUT: {self.mrg_params['paral2']} мкм")
        print(f"Длина интернода (STIN): {self.mrg_params['interL']} мкм")
        print(f"Шаг между узлами (Lstep): {self.mrg_params['Lstep']} мкм")
        print(f"Плотность натриевых каналов (g): {self.mrg_params.get('g', 'N/A')}")

        print("\n--- ЭЛЕКТРИЧЕСКИЕ ПАРАМЕТРЫ ---")
        print(f"Удельное сопротивление аксоплазмы (rho_a): {self.rho_a} Ом·мкм")
        print(f"Ёмкость миелина (mycm): {self.mycm} мкФ/см²")
        print(f"Проводимость миелина (mygm): {self.mygm} См/см²")
        print(f"Проводимость натриевых каналов в узле: {self.gna_axnode} См/см²")
        print(f"Масштаб проводимости nap: {self.gnapbar_scale}")
        print("=" * 80 + "\n")
