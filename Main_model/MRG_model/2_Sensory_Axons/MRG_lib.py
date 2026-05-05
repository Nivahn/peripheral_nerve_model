
import matplotlib.pyplot as plt
import numpy as np
import math
import time
import os
import pandas as pd
from scipy.signal import find_peaks
from impulse_generator import *
import h5py
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from neuron_loader import load_neuron_h

h = load_neuron_h()

# ==================================================================================================
# ДОБАВЛЕНО (2026-03): Явная загрузка механизмов NEURON (nrnmech.dll) из текущей папки модели.
# Зачем: при работе с двумя аксонами и ephaptic-coupling важно, чтобы использовался именно
#        ваш скомпилированный механизм (newaxnode/axnode) из 2_Sensory_Axons.
# ==================================================================================================

_NRNMECH_DLL_LOADED: bool = False
_NRNMECH_DLL_PATH: Optional[Path] = None


def _candidate_mech_paths(base_dir: Path) -> list[Path]:
    """Returns possible compiled NEURON mechanism library paths for Windows/Linux."""
    return [
        base_dir / "nrnmech.dll",
        base_dir / "x86_64" / "libnrnmech.so",
        base_dir / "x86_64" / ".libs" / "libnrnmech.so",
        base_dir / "aarch64" / "libnrnmech.so",
        base_dir / "aarch64" / ".libs" / "libnrnmech.so",
    ]


def load_nrnmech_dll_once() -> Path:
    """Загружает скомпилированные механизмы NEURON один раз.

    Поддерживает как Windows `nrnmech.dll`, так и Linux `x86_64/libnrnmech.so`.

    Примечание: повторный вызов h.nrn_load_dll() для одной и той же библиотеки
    может приводить к ошибке "The user defined name already exists".
    """
    global _NRNMECH_DLL_LOADED, _NRNMECH_DLL_PATH

    if _NRNMECH_DLL_LOADED and _NRNMECH_DLL_PATH is not None:
        return _NRNMECH_DLL_PATH

    base_dir = Path(__file__).resolve().parent
    dll_path = None
    for cand in _candidate_mech_paths(base_dir):
        if cand.exists():
            dll_path = cand
            break
    if dll_path is None:
        expected = "\n".join(str(p) for p in _candidate_mech_paths(base_dir))
        raise FileNotFoundError(f"Не найдены скомпилированные механизмы NEURON. Ожидались пути:\n{expected}")

    # Иногда NEURON может автоматически загрузить nrnmech.dll из текущей папки.
    # Если механизмы уже доступны, НЕ вызываем h.nrn_load_dll(), чтобы не получать
    # шумное сообщение "The user defined name already exists".
    tmp = h.Section()
    already_loaded = False
    try:
        try:
            tmp.insert('newaxnode')
            tmp.uninsert('newaxnode')
            already_loaded = True
        except Exception:
            try:
                tmp.insert('axnode')
                tmp.uninsert('axnode')
                already_loaded = True
            except Exception:
                already_loaded = False
    finally:
        try:
            h.delete_section(sec=tmp)
        except Exception:
            pass

    if not already_loaded:
        try:
            h.nrn_load_dll(str(dll_path))
        except RuntimeError:
            # NEURON часто печатает причину в консоль, а в исключении даёт только hocobj_call.
            # Проверим, доступны ли механизмы. Если да — продолжаем.
            tmp = h.Section()
            ok = False
            try:
                tmp.insert('newaxnode')
                ok = True
            except Exception:
                try:
                    tmp.insert('axnode')
                    ok = True
                except Exception:
                    ok = False
            try:
                h.delete_section(sec=tmp)
            except Exception:
                pass
            if not ok:
                raise

    _NRNMECH_DLL_LOADED = True
    _NRNMECH_DLL_PATH = dll_path
    return dll_path

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


def _small_mrg_params_from_ascent(fiberD: float):
    """ASCENT SMALL_MRG_INTERPOLATION geometry for small myelinated fibers.

    ASCENT documents this mode for 1.011-16 um and uses adjusted nodal
    conductances for small diameters to avoid multiple spikes per pulse.
    We use it here only as an extension below the original discrete MRG table.
    """
    d = float(fiberD)
    if d < 1.011 or d >= 5.7:
        raise ValueError("ASCENT small MRG interpolation is intended for 1.011 <= diameter < 5.7 um")

    g_ratio = 0.020 * (d - 2.39) + 0.55
    axon_d = g_ratio * d
    node_to_axon_ratio = -0.011 * (axon_d - 7.15) + 0.40
    node_d = node_to_axon_ratio * axon_d

    deltax = -3.22 * d * d + 148.0 * d - 128.0
    paralength2 = -0.171 * d * d + 6.48 * d - 0.935
    nl = int(round(math.exp(0.5 * (axon_d - 1.75) + 3.2)))

    return {
        'fiberD': d,
        'axonD': axon_d,
        'nodeD': node_d,
        'paraD1': node_d,
        'paraD2': axon_d,
        'paral1': MRGaxon.paralength1,
        'paral2': paralength2,
        'interL': (deltax - MRGaxon.nodelength - 2 * MRGaxon.paralength1 - 2 * paralength2) / 6.0,
        'nl': nl,
        'delta_z': deltax,
        'node_channel_overrides': {
            'gnabar': 2.333,
            'gkbar': 0.116,
        },
    }

class MRGaxon:

    h.load_file('stdrun.hoc')
    # ------------------------------------------------------------------------------------
    # ------------------------------ ГЛОБАЛЬНЫЕ ПАРАМЕТРЫ --------------------------------
    # ------------------------------------------------------------------------------------
    celsius = 37.0 # Сelsius
    dt_ms   = 0.0005 # ms
    v_init  = -80.0 # mV

    # Электрические константы, как в MRG
    rho_a = 0.7e6  # Ohm·um
    mycm = 0.1    # uF/cm2 per lamella
    mygm = 0.001  # S/cm2 per lamella

    # Периакисональные зазоры (ум)
    space_p1 = 0.002  # around node/MYSA
    space_p2 = 0.004  # around FLUT
    space_i  = 0.004  # around STIN

    # Геометрия "минимал" MRG
    paralength1 = 3.0   # MYSA длина (µm)
    nodelength  = 1.0   # узел (µm)

    gna_axnode = 3.0 # (mho/cm2) or S/cm2
    bouton_L = 1 # µm

    def __init__(self,
             fiber_diameter=5.7,
             parent_axon_nodes=42,
             branch_nodes=21,
             branches_num=2,
             nodes_dist=10,
             branch_every_um=None,
             diam_scale=0.6,
             main_after_branch_scale=None,
             daughter_branch_scale=None,
             main_after_branch_fiber_diameter=None,
             daughter_branch_fiber_diameter=None,
             main_transition_nodes=3,
             daughter_transition_nodes=3,
             branch_connector_length_um=1.0,
             branch_connector_diam_scale=1.0,
             celsius=37.0,
             dt_ms=0.0005,
             v_init=-80.0,
             h_stop = 1000.0,
             gnapbar_scale=0.5,
             reset_nrn: bool = True):

        # ---------------------------------------------------------------------------------
        # ДОБАВЛЕНО (2026-03):
        # 1) загружаем механизмы из nrnmech.dll именно этой папки
        # 2) включаем 2 слоя extracellular (vext[0] и vext[1]) для ephaptic coupling
        # 3) НЕ удаляем секции, чтобы можно было построить 2 аксона в одной симуляции
        # ---------------------------------------------------------------------------------
        load_nrnmech_dll_once()
        h.nlayer_extracellular(2)

        self.reset_nrn = bool(reset_nrn)
        if self.reset_nrn:
            self.reset_model()

        # Параметры морфологии
        self.fiber_diameter = fiber_diameter
        self.parent_axon_nodes = parent_axon_nodes
        self.branch_nodes = branch_nodes
        self.branches_num = branches_num
        self.nodes_dist = nodes_dist
        self.diam_scale = diam_scale

        # Отдельные масштабы main и daughter лучше отражают, что после branch point
        # главный путь и дочерняя ветвь не обязаны иметь одинаковую геометрию.
        self.main_after_branch_scale = float(diam_scale if main_after_branch_scale is None else main_after_branch_scale)
        self.daughter_branch_scale = float(diam_scale if daughter_branch_scale is None else daughter_branch_scale)
        self.main_after_branch_fiber_diameter = (
            None if main_after_branch_fiber_diameter is None else float(main_after_branch_fiber_diameter)
        )
        self.daughter_branch_fiber_diameter = (
            None if daughter_branch_fiber_diameter is None else float(daughter_branch_fiber_diameter)
        )
        self.main_transition_nodes = max(0, int(main_transition_nodes))
        self.daughter_transition_nodes = max(0, int(daughter_transition_nodes))
        self.branch_connector_length_um = float(branch_connector_length_um)
        self.branch_connector_diam_scale = float(branch_connector_diam_scale)

        # Параметры симуляции
        self.celsius = celsius
        self.dt_ms = dt_ms
        self.v_init = v_init

        # Установка глобальных параметров NEURON
        h.celsius = celsius
        h.dt = dt_ms
        self.h_stop = h_stop
        h.tstop = self.h_stop

        # Реестры секций
        self.regions = {
            "node": [],
            "mysa": [],
            "flut": [],
            "stin": [],
            "connector": [],
        }

        # Счётчики ID
        self._node_id = 0
        self._mysa_id = 0
        self._flut_id = 0
        self._stin_id = 0
        self._connector_id = 0

        # Структуры аксона
        self.main_axon = []
        self.branches = []  # Список ветвей, каждая ветвь - список узлов
        self.terminals = []

        # ---------------------------------------------------------------------------------
        # ДОБАВЛЕНО (2026-03): индекс секций по имени.
        # Зачем: для построения ephaptic coupling (LinearMechanism) нужно быстро получать
        #        Section по строковому имени ("node_15", "STIN_123", ...)
        # ---------------------------------------------------------------------------------
        self.secs_by_name = {}

        # ---------------------------------------------------------------------------------
        # ДОБАВЛЕНО (2026-03): продольные координаты (по стволу) для некоторых секций.
        # Используются для построения карт coupling "как у Prescott" (по центрам секций).
        # ---------------------------------------------------------------------------------
        self.trunk_center_um = {}
        self._trunk_last_node_center_um = 0.0
        self._trunk_last_node_name = None
        self._trunk_stin_mid_by_next_node = {}
        self._trunk_stin_mid_idx_by_next_node = {}

        # Отдельные длины пути нужны, чтобы AxonA-like точки выбирались не по общему x,
        # а по длине сопоставимого пути по main stem или daughter branch.
        self.main_path_distance_um = {}
        self.daughter_path_distance_um = {}

        # ДОБАВЛЕНО (2026-03): продольный сдвиг аксона (для Prescott misaligned).
        self.longitudinal_offset_um = 0.0

        # ДОБАВЛЕНО (2026-03): опциональная точка стимуляции.
        self._stim_target_sec = None
        self._stim_target_desc = None

        # ДОБАВЛЕНО (2026-03): записи шагов топологии (для аккуратной 2D-отрисовки анатомии).
        # Каждый шаг append_one_step сохраняет список секций (MYSA/FLUT/STIN) между parent_node и next_node.
        self._step_records = []

        # Параметры стимуляции по умолчанию
        self.stimulation_params = {
            'mode' : 'None', # Biphasic / Monophasic
            'freq_hz': 10, # Hz
            'amp': 5, # nA
            't_start': 200.0, #  ms
            't_end': 1000.0, #  ms
            'phase_us': 0.0, # us
            'gap_us': 0.0, # "us"
            'plot_duration': 100.0, # ms
            'csv_path' : "None",
            'neuron_index': 0,
            "index_is_one_based": False,
            "pulse_len_ms": 1.0, # ms

        }


        # Механизм узла
        self.node_mech = self._pick_node_mech()

        self.gnapbar_scale = gnapbar_scale

        # Получение параметров MRG
        self.mrg_params = self._get_mrg_params(fiber_diameter)
        self.node_channel_overrides = dict(self.mrg_params.get('node_channel_overrides', {}))

        self.branch_every_um = branch_every_um
        if self.branch_every_um is not None:
            # Lstep — длина одного "шага" node->node в MRG (в мкм)
            self.nodes_dist = max(1, int(round(self.branch_every_um / self.mrg_params['Lstep'])))

        # (опционально) полезно знать фактический шаг ветвления в мкм после округления
        self.branch_every_um_effective = self.nodes_dist * self.mrg_params['Lstep']

        # Готовим целевые пост-branch параметры один раз, чтобы дальше build_axon был простым.
        self.main_after_branch_params = self._build_branch_target_params(
            target_fiber_diameter=self.main_after_branch_fiber_diameter,
            fallback_scale=self.main_after_branch_scale,
        )
        self.daughter_branch_params = self._build_branch_target_params(
            target_fiber_diameter=self.daughter_branch_fiber_diameter,
            fallback_scale=self.daughter_branch_scale,
        )

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
        if 1.011 <= float(fiberD) < 5.7:
            base = _small_mrg_params_from_ascent(float(fiberD))
            base['rpn0'] = self._rin_peri(base['nodeD'], self.space_p1)
            base['rpn1'] = self._rin_peri(base['paraD1'], self.space_p1)
            base['rpn2'] = self._rin_peri(base['paraD2'], self.space_p2)
            base['rpx'] = self._rin_peri(base['axonD'], self.space_i)
            base['Lstep'] = float(base['delta_z'])
            return base

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

    def _build_branch_target_params(self, *, target_fiber_diameter, fallback_scale: float):
        """Параметры участка после ветвления.

        Если задан реальный диаметр пост-branch волокна, берём полный набор параметров
        через MRG/ASCENT interpolation. Это научно лучше, чем простое масштабирование
        только диаметров. Если реальный диаметр не задан, используем старый подход со scale.
        """
        if target_fiber_diameter is not None:
            return self._get_mrg_params(float(target_fiber_diameter))
        return self.scaled_params(self.mrg_params, float(fallback_scale))

    def _params_for_branch_step(self, *, target_params: dict, transition_nodes: int, step_index_from_branch: int):
        """Простая branch transition zone.

        Здесь без идеализации: в течение первых N узлов после ветвления используем локальные
        branch-параметры, после чего возвращаемся к базовой геометрии. Это проще и прозрачнее,
        чем скрытое правило с жёстко зашитыми тремя узлами.
        """
        if int(step_index_from_branch) <= int(transition_nodes):
            return target_params
        return self.mrg_params

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

    # ------------------------------------------------------------------------------------
    # ДОБАВЛЕНО (2026-03): Prescott-style настройки 2-го слоя extracellular (vext[1]).
    # В статье/ноутбуке Prescott/Abdollahi внешний слой (vext[1]) моделирует эндоневрий.
    # Там задают:
    #   - xraxial[1] = xr (Mohm/cm)
    #   - xg[1]      = XG1 (обычно 1e-9, т.е. "отсоединено" от земли)
    #   - xc[1]      = 0
    # xr рассчитывается так же, как в ноутбуке Prescott:
    #   rho2 = 1211*1e-6 (Mohm*cm)
    #   AVE  = edge_dist_um/2
    #   xr   = rho2 / (pi*((radi+AVE)^2 - radi^2)*1e-8)
    # где radi = радиус волокна (ум).
    # ------------------------------------------------------------------------------------
    def apply_prescott_extracellular_layer1(self, edge_dist_um: float, XG1: float = 1e-9):
        """Применяет параметры vext[1] для всех секций аксона."""
        rho2 = 1211.0 * 1e-6  # Mohm*cm
        radi = float(self.fiber_diameter) / 2.0
        ave = float(edge_dist_um) / 2.0
        xr = rho2 / (math.pi * (((radi + ave) ** 2) - (radi**2)) * 1e-8)  # Mohm/cm

        # Важно: во втором слое нам нужно выставить xraxial[1]/xg[1]/xc[1] для всех секций.
        for sec_list in self.regions.values():
            for sec in sec_list:
                self._insert_mechanism(sec, 'extracellular')
                for seg in sec:
                    seg.xraxial[1] = xr
                    seg.xg[1] = float(XG1)
                    seg.xc[1] = 0.0

        # Для отладки полезно вернуть xr
        return xr

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

        # Для малых диаметров ASCENT рекомендует ослабить fast Na и усилить K,
        # чтобы избежать множественных спайков на один стимул.
        overrides = getattr(self, 'node_channel_overrides', {}) or {}
        for seg in s:
            if self.node_mech == 'newaxnode':
                seg.gnabar_newaxnode = float(gnabar)
                if 'gnabar' in overrides:
                    seg.gnabar_newaxnode = float(overrides['gnabar'])
                if 'gkbar' in overrides:
                    seg.gkbar_newaxnode = float(overrides['gkbar'])
                seg.gnapbar_newaxnode = float(gnapbar) * float(self.gnapbar_scale)
                if 'gl' in overrides:
                    seg.gl_newaxnode = float(overrides['gl'])
            elif self.node_mech == 'axnode':
                if hasattr(seg, 'gnabar_axnode') and 'gnabar' in overrides:
                    seg.gnabar_axnode = float(overrides['gnabar'])
                if hasattr(seg, 'gkbar_axnode') and 'gkbar' in overrides:
                    seg.gkbar_axnode = float(overrides['gkbar'])


        self._set_extracellular(s, Rpn0, 1e10, 0.0)
        self.regions["node"].append(s)
        return s

    def make_branch_connector(self, diam_um: float, length_um: float):
        """Короткая пассивная секция в зоне bifurcation.

        Она не претендует на точную ultrastructure branch point, но делает локальную геометрию
        менее грубой, чем прямое node->node ветвление.
        """
        s = h.Section(name=f'BCONN_{self._connector_id}')
        self._connector_id += 1
        s.nseg = 1
        s.L = float(length_um)
        s.diam = float(max(diam_um, 0.2))
        s.Ra = self.rho_a / 10000.0
        s.cm = 2.0
        self._insert_mechanism(s, 'pas')
        s.g_pas = 0.001
        s.e_pas = -80.0
        self._set_extracellular(s, 1e10, 1e10, 0.0)
        self.regions["connector"].append(s)
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

    def append_one_step(self, parent_node, params, track_trunk: bool = False):

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

        # ---------------------------------------------------------------------------------
        # Сохранение "опорной" STIN-секции между узлами на стволе.
        # Зачем: для misaligned ephaptic coupling (как у Prescott) нам нужно уметь
        #        связать узел одного аксона с интернодальным сегментом другого аксона.
        # Реализация: берём STIN, ближайший к середине интерноды, и сохраняем его для
        #             next_node (то есть для интерноды parent_node -> next_node).
        # ---------------------------------------------------------------------------------
        if track_trunk:
            # Выбираем STIN, максимально близкий к середине между node->node (midpoint).
            # Это нужно для честного Prescott misaligned: половинный сдвиг вдоль оси X.
            n_st = len(stin_sections)
            node_center = float(getattr(self, "_trunk_last_node_center_um", 0.0))
            mid_target = node_center + 0.5 * float(params.get('Lstep', 0.0))

            node_half = float(self.nodelength) / 2.0
            mysa_L = float(params['paral1'])
            flut_L = float(params['paral2'])
            stin_L = float(params['interL'])

            cand = [node_center + node_half + mysa_L + flut_L + (k + 0.5) * stin_L for k in range(n_st)]
            mid_idx = int(np.argmin(np.abs(np.asarray(cand, dtype=float) - float(mid_target))))
            mid = stin_sections[mid_idx]

            self._trunk_stin_mid_by_next_node[next_node.name()] = mid
            self._trunk_stin_mid_idx_by_next_node[next_node.name()] = int(mid_idx)

            # Prescott: для misaligned нам важен midpoint между узлами.
            # Используем реальный STIN-сегмент как "прокси", но координату считаем как
            # середину шага node->node, чтобы совпадало с постановкой Prescott.
            mid_center = node_center + 0.5 * float(params.get('Lstep', 0.0))
            self.trunk_center_um[mid.name()] = mid_center

        # ---------------------------------------------------------------------------------
        # Сохраняем геометрию шага для продольной схемы.
        # Это НЕ влияет на NEURON, только на построение графиков.
        # ---------------------------------------------------------------------------------
        try:
            self._step_records.append({
                "parent": parent_node.name(),
                "next": next_node.name(),
                "is_trunk": bool(track_trunk),
                "sections": [
                    {"type": "mysa", "name": mysa0.name(), "L": float(mysa0.L)},
                    {"type": "flut", "name": flut0.name(), "L": float(flut0.L)},
                    *[{"type": "stin", "name": st.name(), "L": float(st.L)} for st in stin_sections],
                    {"type": "flut", "name": flut1.name(), "L": float(flut1.L)},
                    {"type": "mysa", "name": mysa1.name(), "L": float(mysa1.L)},
                ],
            })
        except Exception:
            # Если по каким-то причинам логирование шага не получилось, модель всё равно должна работать.
            pass

        return next_node

    def build_chain(self, n_nodes, params, node_mech=None):

        nodes = [self.make_node(params['nodeD'], self.nodelength, params['rpn0'])]
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

        scaled['rpn0'] = self._rin_peri(scaled['nodeD'],  self.space_p1)
        scaled['rpn1'] = self._rin_peri(scaled['paraD1'], self.space_p1)
        scaled['rpn2'] = self._rin_peri(scaled['paraD2'], self.space_p2)
        scaled['rpx']  = self._rin_peri(scaled['axonD'],  self.space_i)

        return scaled


    def build_axon(self):

        terminals = []
        params = self.mrg_params

        # Сброс служебных структур (на случай повторной сборки)
        self._step_records = []

        # Список ветвей (каждая ветвь = список нод). Нужен для корректного boundary coupling.
        self.branches = []

        self.main_axon = [self.make_node(self.mrg_params['nodeD'], self.nodelength, self.mrg_params['rpn0'])]

        self.node_distance_um = {}
        total_length_um = 0.0
        self.node_distance_um[self.main_axon[0].name()] = total_length_um
        self.main_path_distance_um[self.main_axon[0].name()] = total_length_um

        # ДОБАВЛЕНО (2026-03): продольные координаты центров секций на стволе
        self.trunk_center_um[self.main_axon[0].name()] = total_length_um
        self._trunk_last_node_center_um = total_length_um
        self._trunk_last_node_name = self.main_axon[0].name()

        # >>> INSERT: список расстояний в точках ветвления (мкм)
        self.branch_point_distance_um = []

        node_D_after_branching = False
        count_nodes_after_branching = 0
        nodes = 0

        self.branch_point_id = []
        self.before_branch_id = []
        self.after_branch_main_id = []
        self.after_branch_daughter_id = []

        # ---------------------------------------------------------------------------------
        # Вспомогательная функция для записи сегмента в середине ноды.
        # В вашем коде местами использовалось .extend(Section), что неявно даёт Segment.
        # Явный вариант делает код читаемее.
        # ---------------------------------------------------------------------------------
        def _seg05(sec):
            return sec(0.5)

        for _ in range(self.parent_axon_nodes - 1):

            if node_D_after_branching == True:
                step_idx_from_branch = int(self.main_transition_nodes - count_nodes_after_branching + 1)
                P_main_axon = self._params_for_branch_step(
                    target_params=self.main_after_branch_params,
                    transition_nodes=self.main_transition_nodes,
                    step_index_from_branch=step_idx_from_branch,
                )
                # ДОБАВЛЕНО: запоминаем центр последней ноды перед шагом (для расчёта центров STIN)
                self._trunk_last_node_center_um = total_length_um
                self._trunk_last_node_name = self.main_axon[-1].name()
                nxt = self.append_one_step(self.main_axon[-1], P_main_axon, track_trunk=True)
                self.main_axon.append(nxt)

                total_length_um += P_main_axon['Lstep']
                self.node_distance_um[self.main_axon[-1].name()] = total_length_um
                self.main_path_distance_um[self.main_axon[-1].name()] = total_length_um

                # ДОБАВЛЕНО: центр новой ноды
                self.trunk_center_um[self.main_axon[-1].name()] = total_length_um

                # вставляем 1 шаг со скейлом на 60 %

                count_nodes_after_branching -= 1
                # print(f"count_nodes_after_branching: {count_nodes_after_branching}")
                if count_nodes_after_branching == 0:
                    node_D_after_branching = False
                    # ---------------------------------------------------------------------------------
                    # ИЗМЕНЕНО (2026-03): запись только одной контрольной ноды на main-ветви
                    # (третья по порядку после точки ветвления), как в вашем исходном коде.
                    # ---------------------------------------------------------------------------------
                    self.after_branch_main_id.append(self.main_axon[-1](0.5))

            if node_D_after_branching == False:
                # ДОБАВЛЕНО: запоминаем центр последней ноды перед шагом (для расчёта центров STIN)
                self._trunk_last_node_center_um = total_length_um
                self._trunk_last_node_name = self.main_axon[-1].name()
                nxt = self.append_one_step(self.main_axon[-1], params, track_trunk=True)
                self.main_axon.append(nxt)

                # >>> INSERT: обновляем пройденную длину (мкм) до этой ноды
                total_length_um += params['Lstep']
                self.node_distance_um[self.main_axon[-1].name()] = total_length_um
                self.main_path_distance_um[self.main_axon[-1].name()] = total_length_um

                # ДОБАВЛЕНО: центр новой ноды
                self.trunk_center_um[self.main_axon[-1].name()] = total_length_um

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

                # Запоминаем саму ноду ветвления
                self.branch_point_id.append(_seg05(branch_node))

                # ---------------------------------------------------------------------------------
                # Запись "до ветвления" как в исходной логике.
                # Вы хотите хранить в HDF5 НЕ 3 ноды, а одну фиксированную контрольную точку.
                # В вашем коде это была нода self.main_axon[-3].
                # ---------------------------------------------------------------------------------
                if len(self.main_axon) >= 3:
                    self.before_branch_id.append(_seg05(self.main_axon[-3]))
                else:
                    self.before_branch_id.append(_seg05(branch_node))

                #P_branch = self.scaled_params(params, self.diam_scale)
                #term_chain = self.build_chain(self.branch_nodes, P_branch)

                P_base = params
                P_main_target = self.main_after_branch_params
                P_daughter_target = self.daughter_branch_params

                term_chain = []

                # Простые branch connectors делают локальную bifurcation zone менее грубой,
                # чем прямое node->node ветвление. Это не полная ultrastructure, а reduced branch zone.
                connector_diam_um = float(branch_node.diam) * float(self.branch_connector_diam_scale)
                daughter_conn = self.make_branch_connector(connector_diam_um, self.branch_connector_length_um)
                main_conn = self.make_branch_connector(connector_diam_um, self.branch_connector_length_um)
                daughter_conn.connect(branch_node, 1.0, 0.0)
                main_conn.connect(branch_node, 1.0, 0.0)

                # первая нода дочки
                d0 = self.make_node(P_daughter_target['nodeD'], self.nodelength, P_daughter_target['rpn0'])

                # ВАЖНО (фикс 2026-03): правильная ориентация соединения ветви.
                # Мы хотим, чтобы ветвь выходила из дистального конца branch_node (x=1.0)
                # и входила в проксимальный конец дочерней ноды d0 (x=0.0).
                #
                # Неправильный вариант (как было раньше): d0.connect(branch_node, 0.0, 1.0)
                # подключает d0(1) к branch_node(0) и приводит к неверной топологии:
                # "дочка" может выглядеть как активирующаяся раньше, чем точки ДО ветвления.
                d0.connect(daughter_conn, 1.0, 0.0)
                term_chain.append(d0)

                # Примерная продольная координата нод дочерней ветви (для boundary cable).
                # В Prescott boundary-cable задаёт условия для vext[1] по продольной координате.
                # Если ветвь не подключить к boundary-cable, vext[1] на ветви может "плавать"
                # и давать не-физичные ранние/самопроизвольные деполяризации.
                if branch_distance_um is None:
                    branch_distance_um = float(total_length_um)
                Lstep_d0 = float(P_daughter_target.get('Lstep', 1.0))
                self.node_distance_um[d0.name()] = float(branch_distance_um + self.branch_connector_length_um + 1.0 * Lstep_d0)
                self.daughter_path_distance_um[d0.name()] = float(branch_distance_um + self.branch_connector_length_um + 1.0 * Lstep_d0)

                # дальше шаги
                prev = d0
                for i in range(1, self.branch_nodes):
                    P = self._params_for_branch_step(
                        target_params=P_daughter_target,
                        transition_nodes=self.daughter_transition_nodes,
                        step_index_from_branch=i + 1,
                    )
                    nxt = self.append_one_step(prev, P, track_trunk=False)
                    term_chain.append(nxt)
                    prev = nxt

                    prev_path = float(self.daughter_path_distance_um[term_chain[-2].name()])
                    self.node_distance_um[nxt.name()] = float(prev_path + P['Lstep'])
                    self.daughter_path_distance_um[nxt.name()] = float(prev_path + P['Lstep'])

                # --- Продолжение основного аксона
                # Фикс (2026-03): первый узел ПОСЛЕ ветвления должен использовать те же
                # scaled-параметры, что и первые узлы дочерней ветви, иначе получается
                # асимметрия (main проходит, daughter нет) даже при одинаковом diam_scale.
                node_3 = self.make_node(P_main_target['nodeD'], self.nodelength, P_main_target['rpn0'])

                #term_chain[0].connect(branch_node, 0.0, 1.0)  # дочерняя ветвь
                node_3.connect(main_conn, 1.0, 0.0)  # продолжение main

                # ---------------------------------------------------------------------------------
                # ИЗМЕНЕНО (2026-03): после ветвления на основном стволе в HDF5 пишем
                # ТОЛЬКО "третью" контрольную ноду (как у вас было раньше).
                # Поэтому здесь (на node_3) мы ничего не записываем, а записываем позже,
                # когда count_nodes_after_branching дойдёт до 0.
                # ---------------------------------------------------------------------------------

                # ---------------------------------------------------------------------------------
                # ИЗМЕНЕНО (2026-03): запись "после ветвления" в дочерней ветви.
                # Требование: писать только ТРЕТЬЮ ноду по порядку после ветвления.
                # В нашей нумерации term_chain:
                #   term_chain[0] = 1-я нода после ветвления (d0)
                #   term_chain[1] = 2-я
                #   term_chain[2] = 3-я  <-- записываем её
                # ---------------------------------------------------------------------------------
                if len(term_chain) >= 3:
                    self.after_branch_daughter_id.append(_seg05(term_chain[2]))
                else:
                    self.after_branch_daughter_id.append(_seg05(term_chain[-1]))

                # Запоминаем "3 ноды после" в главном аксоне:
                # В твоем коде это делалось через count_nodes_after_branching и append_one_step,
                # поэтому здесь ничего не добавляем вручную — оно заполнится позже в цикле.

                if branch_distance_um is None:
                    print(f"[build_axon] Bifurcation at: {branch_node.name()}")
                else:
                    # ASCII-only, чтобы не падать на Windows-консоли
                    print(f"[build_axon] Bifurcation at: {branch_node.name()}  |  dist~{branch_distance_um:.1f} um")

                print(f"[build_axon] Daughter: {branch_node.name()} -> {term_chain[0].name()}")
                print(f"[build_axon] Main:     {branch_node.name()} -> {node_3.name()}")

                terminals.append(branch_node)

                # Сохраняем ветвь целиком (ноды) для последующего boundary coupling.
                self.branches.append(term_chain)

                # main продолжается с node_3
                self.main_axon.append(node_3)
                self.node_distance_um[node_3.name()] = float(branch_distance_um + self.branch_connector_length_um + P_main_target['Lstep'])
                self.main_path_distance_um[node_3.name()] = float(branch_distance_um + self.branch_connector_length_um + P_main_target['Lstep'])
                self.trunk_center_um[node_3.name()] = self.node_distance_um[node_3.name()]
                total_length_um = self.node_distance_um[node_3.name()]

                node_D_after_branching = True
                self.branches_num -= 1
                nodes = 0
                count_nodes_after_branching = max(0, self.main_transition_nodes - 1)

        print("[build_axon] Ноды на которых будет вестись запись:")
        print(f"[build_axon] Ветвление в точке: {self.branch_point_id}")
        print(f"[build_axon] 3 ноды до точки ветвления: {self.before_branch_id}")
        print(f"[build_axon] 3 ноды после точки ветвления в главном аксоне: {self.after_branch_main_id}")
        print(f"[build_axon] 3 ноды после точки ветвления в дочерней ветке: {self.after_branch_daughter_id}")
        print(f"[build_axon] Расстояния ветвления (um): {self.branch_point_distance_um}")
        if hasattr(self, "branch_every_um_effective"):
            print(f"[build_axon] Фактический шаг ветвления (um) после округления: {self.branch_every_um_effective:.1f}")

        # ---------------------------------------------------------------------------------
        # ДОБАВЛЕНО (2026-03): создаём индекс секций по имени.
        # Это используется позже для построения ephaptic coupling и boundary-cable coupling.
        # ---------------------------------------------------------------------------------
        self._refresh_secs_by_name()


    # ------------------------------------------------------------------------------------
    # ------------------------------ ДОБАВЛЕНО: СЛУЖЕБНЫЕ МЕТОДЫ -------------------------
    # ------------------------------------------------------------------------------------
    def _refresh_secs_by_name(self):
        """Обновляет словарь {"имя_секции": Section} для быстрого доступа."""
        d = {}
        for sec_list in self.regions.values():
            for sec in sec_list:
                d[sec.name()] = sec
        self.secs_by_name = d

    def get_sec(self, name: str):
        """Возвращает Section по имени (node_*, MYSA_*, FLUT_*, STIN_*)."""
        if not hasattr(self, "secs_by_name") or not self.secs_by_name:
            self._refresh_secs_by_name()
        if name not in self.secs_by_name:
            raise KeyError(f"Секция '{name}' не найдена. Пример доступных: {list(self.secs_by_name.keys())[:10]}")
        return self.secs_by_name[name]

    # ------------------------------------------------------------------------------------
    # ДОБАВЛЕНО (2026-03): пересчёт продольных координат ствола для ephaptic coupling.
    #
    # Проблема:
    #   При ветвлении в вашем билдере создаются дополнительные ноды (для дочерней ветви),
    #   из-за чего имена нод на стволе после ветвления могут "прыгать" (node_10 -> node_32).
    #   Для построения карты связи (и для схемы) нам важна НЕ строка имени, а координата по X.
    #
    # Решение:
    #   После построения аксона мы берём фактический список ствола self.main_axon и
    #   назначаем координаты: x = i * Lstep (как у Prescott для ровного аксона).
    #   Также пересчитываем центры mid-STIN секций между соседними узлами ствола.
    # ------------------------------------------------------------------------------------
    def recompute_trunk_geometry_for_coupling(self):
        if not hasattr(self, 'main_axon') or not self.main_axon:
            return
        if not hasattr(self, 'mrg_params'):
            return

        Lstep = float(self.mrg_params.get('Lstep', 1.0))
        node_half = float(self.nodelength) / 2.0
        mysa_L = float(self.mrg_params.get('paral1', self.paralength1))
        flut_L = float(self.mrg_params.get('paral2', 0.0))
        stin_L = float(self.mrg_params.get('interL', 0.0))

        # Узлы ствола
        for i, sec in enumerate(list(self.main_axon)):
            nm = sec.name()
            x = float(i) * Lstep
            self.node_distance_um[nm] = x
            self.trunk_center_um[nm] = x

        # mid-STIN по каждому ребру (node[i] -> node[i+1])
        for i in range(1, len(self.main_axon)):
            prev = self.main_axon[i - 1]
            nxt = self.main_axon[i]
            prev_x = float(self.node_distance_um.get(prev.name(), (i - 1) * Lstep))

            mid = self._trunk_stin_mid_by_next_node.get(nxt.name(), None)
            if mid is None:
                continue

            mid_idx = self._trunk_stin_mid_idx_by_next_node.get(nxt.name(), None)
            if mid_idx is None:
                # fallback: выбираем ближайший STIN к midpoint
                mid_target = prev_x + 0.5 * Lstep
                cand = [prev_x + node_half + mysa_L + flut_L + (k + 0.5) * stin_L for k in range(6)]
                mid_idx = int(np.argmin(np.abs(np.asarray(cand, dtype=float) - float(mid_target))))

            # Prescott midpoint между node centers
            mid_center = prev_x + 0.5 * Lstep
            self.trunk_center_um[mid.name()] = float(mid_center)


    # ------------------------------------------------------------------------------------
    # ДОБАВЛЕНО (2026-03): применить продольный сдвиг (Prescott misaligned).
    # ------------------------------------------------------------------------------------
    def apply_longitudinal_offset(self, offset_um: float):
        """Сдвигает все сохраненные продольные координаты на offset_um."""
        offset_um = float(offset_um)
        delta = offset_um - float(getattr(self, 'longitudinal_offset_um', 0.0))
        if abs(delta) < 1e-12:
            self.longitudinal_offset_um = offset_um
            return

        # Сдвигаем координаты узлов (ствол + ветви)
        for k in list(getattr(self, 'node_distance_um', {}).keys()):
            try:
                self.node_distance_um[k] = float(self.node_distance_um[k]) + delta
            except Exception:
                pass

        # Сдвигаем центры (узлы + trunk mid-STIN)
        for k in list(getattr(self, 'trunk_center_um', {}).keys()):
            try:
                self.trunk_center_um[k] = float(self.trunk_center_um[k]) + delta
            except Exception:
                pass

        self.longitudinal_offset_um = offset_um

    # ------------------------------------------------------------------------------------
    # ДОБАВЛЕНО (2026-03): выбор точки стимуляции.
    # ------------------------------------------------------------------------------------
    def set_stim_target(
        self,
        *,
        mode: str = 'node_index',
        node_index: int = 0,
        x_um: Optional[float] = None,
        section_name: Optional[str] = None,
    ):
        """Выбирает секцию-узел для IClamp."""
        mode = str(mode)
        self._stim_target_desc = mode

        if not self.main_axon:
            self._stim_target_sec = None
            return

        if mode == 'node_index':
            idx = int(node_index)
            idx = max(0, min(idx, len(self.main_axon) - 1))
            self._stim_target_sec = self.main_axon[idx]
            return

        if mode == 'same_x_um':
            if x_um is None:
                raise ValueError("set_stim_target(mode='same_x_um') требует x_um")
            x_um = float(x_um)
            best = None
            best_d = None
            for sec in list(self.main_axon):
                nm = sec.name()
                if nm not in self.node_distance_um:
                    continue
                x = float(self.node_distance_um[nm])
                d = abs(x - x_um)
                if best_d is None or d < best_d:
                    best = sec
                    best_d = d
            self._stim_target_sec = best if best is not None else self.main_axon[0]
            return

        if mode == 'section_name':
            if section_name is None:
                raise ValueError("set_stim_target(mode='section_name') требует section_name")
            sec = self.get_sec(str(section_name))
            if not str(sec.name()).startswith('node_'):
                raise ValueError(f"Стимуляция поддерживается только на node_*, получено: {sec.name()}")
            self._stim_target_sec = sec
            return

        raise ValueError(f"Неизвестный mode для set_stim_target(): {mode}")

    # ------------------------------------------------------------------------------------
    # ДОБАВЛЕНО (2026-03): утилиты для выбора точек записи.
    # ------------------------------------------------------------------------------------
    def get_node_segment_nearest_x(self, x_um: float):
        """Возвращает узловой сегмент ствола, ближайший к продольной координате x_um."""
        if not self.main_axon:
            return None
        best_sec = None
        best_d = None
        for sec in list(self.main_axon):
            nm = sec.name()
            x = float(self.node_distance_um.get(nm, float("nan")))
            if not np.isfinite(x):
                continue
            d = abs(float(x_um) - x)
            if best_d is None or d < best_d:
                best_sec = sec
                best_d = d
        return best_sec(0.5) if best_sec is not None else self.main_axon[0](0.5)

    def get_node_segment_nearest_main_path_distance(self, distance_um: float):
        """Возвращает узловой сегмент главного пути, ближайший к длине пути distance_um."""
        if not self.main_axon:
            return None
        best_sec = None
        best_d = None
        for sec in list(self.main_axon):
            nm = sec.name()
            x = float(self.main_path_distance_um.get(nm, float("nan")))
            if not np.isfinite(x):
                continue
            d = abs(float(distance_um) - x)
            if best_d is None or d < best_d:
                best_sec = sec
                best_d = d
        return best_sec(0.5) if best_sec is not None else self.main_axon[0](0.5)

    def get_terminal_main_segment(self):
        """Последняя нода главного ствола."""
        if not self.main_axon:
            return None
        return self.main_axon[-1](0.5)

    def get_terminal_daughter_segment(self):
        """Последняя нода первой дочерней ветви (если ветвление есть)."""
        if not getattr(self, 'branches', None):
            return None
        br0 = self.branches[0]
        if not br0:
            return None
        return br0[-1](0.5)

    def collect_recording_targets(self, *, include_stimulation_point: bool = True, extra_named_segments: Optional[list] = None):
        """Собирает стандартные точки записи + опциональные дополнительные."""
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

        if extra_named_segments:
            for name, seg in extra_named_segments:
                if seg is None:
                    continue
                key_segments.append(seg)
                key_names.append(str(name))

        if include_stimulation_point and self.main_axon:
            key_segments.append(self.main_axon[0](0.5))
            key_names.append('stimulation_point')

        return key_segments, key_names


    # ------------------------------------------------------------------------------------
    # ------------------------------ СОЗДАНИЕ СТИМУЛЯТОРА --------------------------------
    # ------------------------------------------------------------------------------------
    def set_stimulation_params(self, mode="create", biphasic = True, **kwargs):
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

        Для mode="spike_times" ожидаются поля:
            spike_times_ms: список времён импульсов (мс)
            t_end: максимальное время симуляции (мс)
            amp: амплитуда импульса (нА)

        Для mode="custom_waveform" ожидаются поля:
            time_points_ms: массив времени (мс)
            current_points_nA: массив тока (нА)
            t_end: максимальное время симуляции (мс)
        """
        self.stimulation_params = dict(kwargs)
        self.stimulation_params["mode"] = mode
        self.stimulation_params["biphasic"] = biphasic

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


            stim_sec = self._stim_target_sec if self._stim_target_sec is not None else self.main_axon[0]
            ipulse_stimulator = STIMULATOR(stim_sec, position=0.5, mode="preload_data")
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


        elif mode == "spike_times":

            params = self.stimulation_params
            spike_times_ms = params["spike_times_ms"]
            t_end = params["t_end"]
            amp = params["amp"]
            phase_ms = params["phase_us"] / 1000
            gap_ms = params["gap_us"] / 1000

            stim_sec = self._stim_target_sec if self._stim_target_sec is not None else self.main_axon[0]
            ipulse_stimulator = STIMULATOR(stim_sec, position=0.5, mode="from_spikes")
            ipulse_stimulator.set_spike_times(
                spike_times_ms=spike_times_ms,
                amp=amp,
                t_max_ms=t_end,
                dt=dt,
                phase_ms=phase_ms,
                gap_ms=gap_ms,
            )

            self.stimulator = ipulse_stimulator
            total_time = t_end
            h.tstop = total_time
            self.h_stop = total_time

        elif mode == "custom_waveform":

            params = self.stimulation_params
            t_points_ms = params["time_points_ms"]
            i_points_nA = params["current_points_nA"]
            t_end = params["t_end"]

            stim_sec = self._stim_target_sec if self._stim_target_sec is not None else self.main_axon[0]
            ipulse_stimulator = STIMULATOR(stim_sec, position=0.5, mode="custom_waveform")
            ipulse_stimulator.set_custom_waveform(t_points_ms, i_points_nA, dt=dt)

            self.stimulator = ipulse_stimulator
            total_time = t_end
            h.tstop = total_time
            self.h_stop = total_time

        elif mode == "create":

            params = self.stimulation_params

            freq = params["freq_hz"]

            amp = params["amp"]

            t_start = params["t_start"]

            t_end = params["t_end"]

            phase_ms = params["phase_us"] / 1000

            gap_ms = params["gap_us"] / 1000

            biphasic = params.get("biphasic", True)  # ← Получаем параметр biphasic!

            bi_width_ms = 2 * phase_ms + gap_ms if biphasic else phase_ms  # ← Учитываем biphasic!

            stimulation_duration_ms = t_end - t_start

            # Расчет параметров стимуляции

            T_ms = 1000.0 / freq

            if biphasic and bi_width_ms > T_ms:  # ← Проверка только для бифазного

                raise ValueError(

                    f"Бифазный пульс ({bi_width_ms:.3f} ms) длиннее периода ({T_ms:.3f} ms) при freq={freq} Гц"

                )

            n_pulses = int(stimulation_duration_ms / T_ms)

            print(f"[create_stimulator] freq={freq} Гц:")

            print(f"[create_stimulator] Период: {T_ms:.3f} мс")

            if biphasic:

                # ASCII-only: используем "2x" вместо символа умножения "×"
                print(
                    f"[create_stimulator] Ширина бифазного пакета: {bi_width_ms:.3f} мс "
                    f"(2x{phase_ms:.3f} + {gap_ms:.3f})"
                )

            else:

                print(f"[create_stimulator] Ширина монофазного пульса: {phase_ms:.3f} мс")

            print(f"[create_stimulator] Количество пульсов: {n_pulses}")

            print(f"[create_stimulator] Длительность стимуляции: {stimulation_duration_ms} мс")

            print(f"[create_stimulator] biphasic: {biphasic}")

            stim_sec = self._stim_target_sec if self._stim_target_sec is not None else self.main_axon[0]
            ipulse_stimulator = STIMULATOR(stim_sec, position=0.5, mode="create")

            ipulse_stimulator.set_parameters(

                t_start, n_pulses, amp, dt, phase_ms, gap_ms, T_ms,

                biphasic=biphasic  # ← Передаем параметр!

            )

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

    def run_simulation(
        self,
        h5_path=None,
        experiment_name=None,
        record_kinetics=True,
        include_stimulation_point: bool = True,
        extra_named_segments: Optional[list] = None,
    ):
        """
        Запускает симуляцию с возможностью записи кинетических переменных.
        """
        # Сбрасываем состояние симуляции
        if hasattr(self, "_reset_simulation_state"):
            self._reset_simulation_state()
        else:
            h.finitialize(self.v_init)

        # Создаем стимулятор
        self.create_stimulator()

        # Готовим список сегментов для записи
        key_segments, key_names = self.collect_recording_targets(
            include_stimulation_point=include_stimulation_point,
            extra_named_segments=extra_named_segments,
        )

        # Векторы записи
        record_v = []
        record_t = h.Vector().record(h._ref_t)

        # Если нужно записать кинетику
        if record_kinetics:
            print("\n[DEBUG] Проверка доступных переменных для записи...")

            # Проверяем первую секцию
            test_seg = key_segments[0] if key_segments else self.main_axon[0](0.5)

            test_vars = ['ina', 'inap', 'ikf', 'iks', 'il', 'm', 'h', 's', 'mp', 'n']
            for var in test_vars:
                ref_name = f"_ref_{var}_newaxnode"
                try:
                    ref = getattr(test_seg, ref_name)
                    print(f"  ✓ {var}_newaxnode доступен")
                except AttributeError:
                    print(f"  ✗ {var}_newaxnode НЕ доступен")

            # Инициализируем списки для записи
            self.record_ina = []
            self.record_inap = []
            self.record_ikf = []
            self.record_iks = []
            self.record_il = []
            self.record_m = []
            self.record_h = []
            self.record_s = []
            self.record_mp = []
            self.record_n = []
            #self.record_ko = []

        self.recording_indices = {}

        for i, (seg, name) in enumerate(zip(key_segments, key_names)):
            # Запись напряжения
            vec = h.Vector().record(seg._ref_v)
            record_v.append(vec)

            # Запись кинетических переменных
            if record_kinetics:
                self.record_ina.append(h.Vector().record(seg._ref_ina_newaxnode))
                self.record_inap.append(h.Vector().record(seg._ref_inap_newaxnode))
                self.record_ikf.append(h.Vector().record(seg._ref_ikf_newaxnode))
                self.record_iks.append(h.Vector().record(seg._ref_iks_newaxnode))
                self.record_il.append(h.Vector().record(seg._ref_il_newaxnode))
                self.record_m.append(h.Vector().record(seg._ref_m_newaxnode))
                self.record_h.append(h.Vector().record(seg._ref_h_newaxnode))
                self.record_s.append(h.Vector().record(seg._ref_s_newaxnode))
                self.record_mp.append(h.Vector().record(seg._ref_mp_newaxnode))
                self.record_n.append(h.Vector().record(seg._ref_n_newaxnode))
                #self.record_ko.append(h.Vector().record(seg._ref_ko_newaxnode))

            node_id = f"{seg.sec.name().replace('.', '_')}_{seg.x:.2f}"
            self.recording_indices[i] = {
                "group": name,
                "node": node_id
            }

        # Запуск симуляции
        h.finitialize(self.v_init)
        h.run()

        # Сохранение данных в numpy массивы
        self.voltage_matrix = np.vstack([np.array(v) for v in record_v])
        self.time_array = np.array(record_t)
        self.recording_labels = key_names

        # Сохранение кинетических данных
        if record_kinetics:
            self.ina_matrix = np.vstack([np.array(v) for v in self.record_ina])
            self.inap_matrix = np.vstack([np.array(v) for v in self.record_inap])
            self.ikf_matrix = np.vstack([np.array(v) for v in self.record_ikf])
            self.iks_matrix = np.vstack([np.array(v) for v in self.record_iks])
            self.il_matrix = np.vstack([np.array(v) for v in self.record_il])
            self.m_matrix = np.vstack([np.array(v) for v in self.record_m])
            self.h_matrix = np.vstack([np.array(v) for v in self.record_h])
            self.s_matrix = np.vstack([np.array(v) for v in self.record_s])
            self.mp_matrix = np.vstack([np.array(v) for v in self.record_mp])
            self.n_matrix = np.vstack([np.array(v) for v in self.record_n])
            #self.ko_matrix = np.vstack([np.array(v) for v in self.record_ko])

        # Сохранение в HDF5
        if h5_path is not None:
            if experiment_name is None:
                experiment_name = "experiment"

            stimulator = self.stimulator if hasattr(self, 'stimulator') else None

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

        # ASCII-only подписи осей
        ax.set_xlabel("X (um)")
        ax.set_ylabel("Y (um)")
        ax.set_zlabel("Z (um)")
        ax.set_title("3D Морфология MRG аксона с ветвлениями (с точками записи)")
        ax.legend()

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"3D морфология сохранена: {save_path}")

        plt.show()

    def plot_kinetics(self, segment_index=0, plot_start=0, plot_end=1000, save_path=None):
        """
        Визуализирует кинетику токов и воротных переменных на выбранном узле.

        segment_index: индекс сегмента в recording_indices
        plot_start, plot_end: начальный и конечный индексы временного интервала
        """
        # Проверяем, что данные записаны
        if not (hasattr(self, 'voltage_matrix') and hasattr(self, 'time_array')):
            print("Нет данных для построения графиков. Сначала запустите run_simulation().")
            return

        # Проверяем, что кинетические данные записаны
        if not hasattr(self, 'ina_matrix'):
            print("Кинетические данные не записаны. Запустите run_simulation с record_kinetics=True.")
            return

        # Проверяем индекс сегмента
        if segment_index >= len(self.recording_indices):
            print(
                f"Ошибка: segment_index={segment_index} превышает количество записанных сегментов ({len(self.recording_indices)})")
            print("Доступные сегменты:")
            for i, meta in self.recording_indices.items():
                print(f"  {i}: {meta['node']} ({meta['group']})")
            return

        # Получаем информацию о сегменте
        seg_info = self.recording_indices.get(segment_index, {})
        seg_name = seg_info.get("node", f"сегмент_{segment_index}")
        seg_group = seg_info.get("group", "unknown")

        # Корректируем plot_end, если он выходит за пределы
        total_points = len(self.time_array)
        if plot_end > total_points:
            plot_end = total_points
            print(f"plot_end скорректирован до {plot_end} (всего точек: {total_points})")

        # Создаем фигуру с 5 графиками в одном столбце
        fig, axes = plt.subplots(5, 1, figsize=(14, 16))

        title_text = (f"Кинетика в узле: {seg_name} ({seg_group})\n"
                      f"Диаметр: {self.fiber_diameter} мкм, "
                      f"Частота: {self.stimulation_params['freq_hz']} Гц, "
                      f"Сила тока: {self.stimulation_params['amp']} нА")
        fig.suptitle(title_text, fontsize=14, fontweight='bold', y=0.98)

        # График 1: Мембранный потенциал
        axes[0].plot(self.time_array[plot_start:plot_end],
                     self.voltage_matrix[segment_index][plot_start:plot_end],
                     label='Мембранный потенциал', linewidth=2, color='black')
        axes[0].set_ylabel("Потенциал (мВ)")
        axes[0].set_title("Мембранный потенциал")
        axes[0].grid(True, alpha=0.3)
        axes[0].legend(loc='upper right')

        # График 2: Токи (синхронизирован с первым графиком)
        time_slice = slice(plot_start, plot_end)

        # Находим максимальный ток для масштабирования оси Y
        all_currents = np.concatenate([
            self.ina_matrix[segment_index][time_slice],
            self.inap_matrix[segment_index][time_slice],
            self.ikf_matrix[segment_index][time_slice],
            self.iks_matrix[segment_index][time_slice],
            self.il_matrix[segment_index][time_slice]
        ])

        current_abs_max = np.max(np.abs(all_currents))
        ylim_current = current_abs_max * 1.1 if current_abs_max > 0 else 1.0

        axes[1].plot(self.time_array[time_slice], self.ina_matrix[segment_index][time_slice],
                     label='Быстрый Na⁺ (ina)', linewidth=1.5, color='red')
        axes[1].plot(self.time_array[time_slice], self.inap_matrix[segment_index][time_slice],
                     label='Персистирующий Na⁺ (inap)', linewidth=1.5, color='orange')
        axes[1].plot(self.time_array[time_slice], self.ikf_matrix[segment_index][time_slice],
                     label='Быстрый K⁺ (ikf)', linewidth=1.5, color='blue')
        axes[1].plot(self.time_array[time_slice], self.iks_matrix[segment_index][time_slice],
                     label='Медленный K⁺ (iks)', linewidth=1.5, color='cyan')
        axes[1].plot(self.time_array[time_slice], self.il_matrix[segment_index][time_slice],
                     label='Утечка (il)', linewidth=1.5, color='green')
        # ASCII: избегаем символа '²', чтобы не падать в Windows-консолях (cp1251).
        axes[1].set_ylabel("Ток (мА/см^2)")
        axes[1].set_title("Ионные токи")
        axes[1].legend(loc='upper right', fontsize=8)
        axes[1].grid(True, alpha=0.3)
        axes[1].set_ylim(-ylim_current, ylim_current)

        # График 3: Вероятность открытия каналов Na
        axes[2].plot(self.time_array[time_slice], self.m_matrix[segment_index][time_slice],
                     label='Активация быстрого Na⁺ (m)', linewidth=1.5, color='red')
        axes[2].plot(self.time_array[time_slice], self.h_matrix[segment_index][time_slice],
                     label='Инактивация быстрого Na⁺ (h)', linewidth=1.5, color='blue')
        axes[2].plot(self.time_array[time_slice], self.mp_matrix[segment_index][time_slice],
                     label='Активация персистирующего Na⁺ (mp)', linewidth=1.5, color='orange')
        axes[2].set_ylabel("Вероятность открытия")
        axes[2].set_title("Воротные переменные натриевых каналов")
        axes[2].legend(loc='upper right')
        axes[2].grid(True, alpha=0.3)
        axes[2].set_ylim(-0.1, 1.1)

        # График 4: Вероятность открытия каналов K
        axes[3].plot(self.time_array[time_slice], self.n_matrix[segment_index][time_slice],
                     label='Активация быстрого K⁺ (n)', linewidth=1.5, color='blue')
        axes[3].plot(self.time_array[time_slice], self.s_matrix[segment_index][time_slice],
                     label='Активация медленного K⁺ (s)', linewidth=1.5, color='cyan')
        axes[3].set_xlabel("Время (мс)")
        axes[3].set_ylabel("Вероятность открытия")
        axes[3].set_title("Воротные переменные калиевых каналов")
        axes[3].legend(loc='upper right')
        axes[3].grid(True, alpha=0.3)
        axes[3].set_ylim(-0.1, 1.1)

        # График 5: Паттерн стимуляции
        if hasattr(self, 'stimulator') and hasattr(self.stimulator, 'time_vec') and hasattr(self.stimulator,
                                                                                            'current_vec'):
            # Преобразуем в numpy массивы
            t_stim = np.array(self.stimulator.time_vec)
            i_stim = np.array(self.stimulator.current_vec)

            # Находим индексы для синхронизации со временем
            # Используем ближайшие временные точки
            time_window = self.time_array[time_slice]
            stim_start_idx = np.argmin(np.abs(t_stim - time_window[0]))
            stim_end_idx = np.argmin(np.abs(t_stim - time_window[-1])) + 1

            # Ограничиваем индексы
            stim_start_idx = max(0, stim_start_idx)
            stim_end_idx = min(len(t_stim), stim_end_idx)

            # Строим график стимуляции
            axes[4].plot(t_stim[stim_start_idx:stim_end_idx],
                         i_stim[stim_start_idx:stim_end_idx],
                         linewidth=2, color='purple')
            axes[4].set_xlabel("Время (мс)")
            axes[4].set_ylabel("Ток стимуляции (нА)")
            axes[4].set_title("Протокол стимуляции")
            axes[4].grid(True, alpha=0.3)

            # Добавляем информацию о стимуляции
            if len(i_stim[stim_start_idx:stim_end_idx]) > 0:
                stim_max = np.max(i_stim[stim_start_idx:stim_end_idx])
                stim_min = np.min(i_stim[stim_start_idx:stim_end_idx])
                axes[4].set_ylim(stim_min * 1.2, stim_max * 1.2)
        else:
            axes[4].text(0.5, 0.5, 'Данные стимуляции не доступны',
                         horizontalalignment='center', verticalalignment='center',
                         transform=axes[4].transAxes, fontsize=12)
            axes[4].set_xlabel("Время (мс)")
            axes[4].set_ylabel("Ток стимуляции (нА)")
            axes[4].set_title("Протокол стимуляции")
            axes[4].grid(True, alpha=0.3)

        # Настраиваем layout
        plt.tight_layout()

        # Сохраняем если указан путь
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"График кинетики сохранен в: {save_path}")

        plt.show()

        # Выводим статистику по сегменту
        print(f"\nСтатистика для сегмента {segment_index} ({seg_name}):")
        print(
            f"  Временной интервал: {self.time_array[plot_start]:.1f} - {self.time_array[min(plot_end - 1, len(self.time_array) - 1)]:.1f} мс")
        print(
            f"  Потенциал min/max: {self.voltage_matrix[segment_index][time_slice].min():.1f}/{self.voltage_matrix[segment_index][time_slice].max():.1f} мВ")
        print(
            f"  Ток ina min/max: {self.ina_matrix[segment_index][time_slice].min():.3f}/{self.ina_matrix[segment_index][time_slice].max():.3f} мА/см^2")
        print(
            f"  Ток ikf min/max: {self.ikf_matrix[segment_index][time_slice].min():.3f}/{self.ikf_matrix[segment_index][time_slice].max():.3f} мА/см^2")
        print(
            f"  Ток iks min/max: {self.iks_matrix[segment_index][time_slice].min():.3f}/{self.iks_matrix[segment_index][time_slice].max():.3f} мА/см^2")
        print(
            f"  Переменная m min/max: {self.m_matrix[segment_index][time_slice].min():.3f}/{self.m_matrix[segment_index][time_slice].max():.3f}")
        print(
            f"  Переменная n min/max: {self.n_matrix[segment_index][time_slice].min():.3f}/{self.n_matrix[segment_index][time_slice].max():.3f}")

    def find_segment_by_name(self, node_name_pattern, group_name=None):
        """
        Находит индекс сегмента по имени узла или группе.

        node_name_pattern: часть имени узла для поиска (например, "node_0")
        group_name: опциональное имя группы для фильтрации

        Возвращает список индексов подходящих сегментов
        """
        matching_indices = []

        for idx, meta in self.recording_indices.items():
            match = True

            # Проверяем имя узла
            if node_name_pattern and node_name_pattern not in meta['node']:
                match = False

            # Проверяем группу
            if group_name and meta['group'] != group_name:
                match = False

            if match:
                matching_indices.append(idx)

        return matching_indices


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
        Теперь сохраняет токи для каждой ноды, если они записаны.
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

            # Проверяем, есть ли кинетические данные
            has_kinetics = (hasattr(self, 'ina_matrix') and
                            hasattr(self, 'ina_matrix') and
                            len(self.ina_matrix) > 0)

            # Упаковываем по группам имён из self.recording_indices
            for idx, meta in self.recording_indices.items():

                grp_name = meta["group"]  # before_branch / branch_point / ...
                node_name = meta["node"]  # node_36_0.50

                grp_grp = grp_traces.require_group(grp_name)

                # Создаем подгруппу для узла
                node_grp = grp_grp.require_group(node_name)

                # Удаляем старые данные, если есть
                for name in ["voltage", "ina", "inap", "ikf", "iks", "il",
                             "m", "h", "s", "mp", "n", "ko"]:
                    if name in node_grp:
                        del node_grp[name]

                # Сохраняем напряжение
                node_grp.create_dataset("voltage", data=self.voltage_matrix[idx, :])

                # Сохраняем токи, если они есть
                if has_kinetics:
                    # Токи
                    if hasattr(self, 'ina_matrix'):
                        node_grp.create_dataset("ina", data=self.ina_matrix[idx, :])
                    if hasattr(self, 'inap_matrix'):
                        node_grp.create_dataset("inap", data=self.inap_matrix[idx, :])
                    if hasattr(self, 'ikf_matrix'):
                        node_grp.create_dataset("ikf", data=self.ikf_matrix[idx, :])
                    if hasattr(self, 'iks_matrix'):
                        node_grp.create_dataset("iks", data=self.iks_matrix[idx, :])
                    if hasattr(self, 'il_matrix'):
                        node_grp.create_dataset("il", data=self.il_matrix[idx, :])

                    # Воротные переменные
                    if hasattr(self, 'm_matrix'):
                        node_grp.create_dataset("m", data=self.m_matrix[idx, :])
                    if hasattr(self, 'h_matrix'):
                        node_grp.create_dataset("h", data=self.h_matrix[idx, :])
                    if hasattr(self, 's_matrix'):
                        node_grp.create_dataset("s", data=self.s_matrix[idx, :])
                    if hasattr(self, 'mp_matrix'):
                        node_grp.create_dataset("mp", data=self.mp_matrix[idx, :])
                    if hasattr(self, 'n_matrix'):
                        node_grp.create_dataset("n", data=self.n_matrix[idx, :])
                    if hasattr(self, 'ko_matrix'):
                        node_grp.create_dataset("ko", data=self.ko_matrix[idx, :])

                # Метаданные узла
                node_grp.attrs["node"] = node_name
                node_grp.attrs["group"] = grp_name
                node_grp.attrs["index_in_matrix"] = idx
                node_grp.attrs["has_kinetics"] = has_kinetics

            # Немного метаданных по аксону
            grp_model.attrs["fiber_diameter_um"] = self.fiber_diameter
            grp_model.attrs["dt_ms"] = self.dt_ms
            grp_model.attrs["celsius"] = self.celsius
            grp_model.attrs["h_stop_ms"] = getattr(self, "h_stop", np.nan)
            grp_model.attrs["has_kinetics"] = has_kinetics

            print(f"[save_to_hdf5] Сохранено в {h5_path} под группой '{experiment_name}'")
            print(f"               Всего сегментов: {len(self.recording_indices)}")
            print(f"               Кинетические данные: {'ДА' if has_kinetics else 'НЕТ'}")

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
        print(f"Main-after-branch scale: {self.main_after_branch_scale}")
        print(f"Daughter-branch scale: {self.daughter_branch_scale}")
        print(f"Main transition nodes: {self.main_transition_nodes}")
        print(f"Daughter transition nodes: {self.daughter_transition_nodes}")
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

        #print("\n--- ЭЛЕКТРИЧЕСКИЕ ПАРАМЕТРЫ ---")
        #print(f"Удельное сопротивление аксоплазмы (rho_a): {self.rho_a} Ом·мкм")
        # Примечание: используем ASCII-обозначение "см^2", чтобы не ловить UnicodeEncodeError в Windows-консоли.
        #print(f"Ёмкость миелина (mycm): {self.mycm} мкФ/см^2")
        #print(f"Проводимость миелина (mygm): {self.mygm} См/см^2")
        #print(f"Проводимость натриевых каналов в узле: {self.gna_axnode} См/см^2")
        #print(f"Масштаб проводимости nap: {self.gnapbar_scale}")
        print("=" * 80 + "\n")

    def check_hdf5_contents(self, h5_path, experiment_name="experiment"):
        """
        Проверяет содержимое HDF5 файла.
        """
        import h5py

        with h5py.File(h5_path, "r") as f:
            if experiment_name not in f:
                print(f"Эксперимент '{experiment_name}' не найден")
                return

            grp = f[experiment_name]

            print(f"Содержимое {experiment_name}:")

            def print_structure(name, obj, indent=0):
                prefix = "  " * indent
                if isinstance(obj, h5py.Group):
                    print(f"{prefix}[DIR] {name}/")
                    for key in obj.keys():
                        print_structure(key, obj[key], indent + 1)
                elif isinstance(obj, h5py.Dataset):
                    print(f"{prefix}[DATA] {name}: {obj.shape} {obj.dtype}")

            print_structure(experiment_name, grp)


# ==================================================================================================
# ДОБАВЛЕНО (2026-03): ДВА СЕНСОРНЫХ АКСОНА + Ephaptic coupling + Boundary cable (Prescott/Abdollahi)
#
# ВАЖНО:
# - Реализовано по той же идее, что в статье Prescott/Abdollahi:
#   1) используем 2 слоя extracellular (vext[0] и vext[1])
#   2) строим "boundary cable" из 4000 секций (как Boundarycable.hoc)
#   3) связываем axon<->axon и axon<->boundary через h.LinearMechanism по vext[1]
#   4) проводимости считаем по формулам из ноутбука Prescott
#
# Этот код написан так, чтобы НЕ ломать ваш текущий класс MRGaxon.
# ==================================================================================================


@dataclass(frozen=True)
class EphapticSpec:
    """Описание связей для LinearMechanism (по аналогии с Connect_types_*.mat + Rg_*.txt)."""

    sec_names_first: list
    sec_names_second: list
    rg_dimless: list  # rg = (rg_um / s_um), как в Prescott (потом умножаем на s_um)


class BoundaryCable:
    """Boundary cable (как Boundarycable.hoc у Prescott): 4000 секций + 'soma'."""

    def __init__(
        self,
        *,
        name_prefix: str,
        n_sections: int = 4000,
        total_length_um: float = 40000.0,
        sparse_section_names: Optional[list] = None,
    ):
        self.name_prefix = str(name_prefix)
        self.n_sections = int(n_sections)
        self.total_length_um = float(total_length_um)
        self.section_length_um = self.total_length_um / float(self.n_sections)

        # ---------------------------------------------------------------------------------
        # ДОБАВЛЕНО (2026-03): оптимизация boundary cable.
        #
        # В оригинальном Boundarycable.hoc:
        #   - Ra = 1e9
        #   - cm = 1e-9
        # То есть кабель практически изолирован по продольной оси и играет роль
        # набора "точек-стоков". Для длительных симуляций (секунды, dt=0.005) создание
        # 4000+4000 секций резко замедляет расчёт.
        #
        # Поэтому мы по умолчанию создаём ТОЛЬКО те boundary-секции, которые реально
        # используются в связи axon<->boundary (LinearMechanism). Это эквивалентно
        # исходному кабелю при Ra=1e9, но намного быстрее.
        #
        # Если sparse_section_names=None — создаётся полный кабель 4000 секций.
        # ---------------------------------------------------------------------------------
        self.sparse_section_names = None if sparse_section_names is None else list(sparse_section_names)

        self.secs = {}
        self._create()

    def _create(self):
        h.nlayer_extracellular(2)

        sec_len = float(self.section_length_um)

        def _mk_section(idx: int):
            s = h.Section(name=f"{self.name_prefix}section_{idx}")
            s.nseg = 1
            s.diam = 0.01
            s.L = sec_len
            s.Ra = 1e9
            s.cm = 1e-9
            if int(h.ismembrane('extracellular', sec=s)) == 0:
                s.insert('extracellular')
            self.secs[f"section_{idx}"] = s
            return s

        if self.sparse_section_names is None:
            chain = []
            for i in range(self.n_sections):
                chain.append(_mk_section(i))

            for i in range(self.n_sections - 1):
                chain[i + 1].connect(chain[i], 1.0, 0.0)

            last = chain[-1]
        else:
            # Создаём только нужные индексы.
            idxs = []
            for nm in self.sparse_section_names:
                if isinstance(nm, str) and nm.startswith('section_'):
                    idxs.append(int(nm.split('_')[1]))
                elif isinstance(nm, int):
                    idxs.append(int(nm))
            idxs = sorted(set([i for i in idxs if 0 <= i < self.n_sections]))
            if not idxs:
                idxs = [0]
            for i in idxs:
                _mk_section(int(i))
            # В sparse-режиме цепочку НЕ соединяем.
            last = self.secs[f"section_{idxs[-1]}"]

        soma = h.Section(name=f"{self.name_prefix}soma") # Что здесь делает soma?
        soma.nseg = 1
        soma.diam = 0.01
        soma.L = 0.01
        soma.Ra = 1e9
        soma.cm = 1e-9
        if int(h.ismembrane('extracellular', sec=soma)) == 0:
            soma.insert('extracellular')
        # В full-режиме подключаем к последней секции кабеля.
        # В sparse-режиме подключаем к какой-то существующей секции (для совместимости).
        soma.connect(last, 1.0, 0.0)
        self.secs["soma"] = soma

    def set_grounded_sink(self):
        """Делаем boundary почти идеальным стоком (как в Prescott)."""
        for sec in self.secs.values():
            for seg in sec:
                seg.xraxial[0] = 1e9
                seg.xraxial[1] = 1e9
                seg.xg[0] = 1e9
                seg.xg[1] = 1e9
                seg.xc[0] = 0.0
                seg.xc[1] = 0.0


def _compute_rg_dimless_from_centers(centers_um: np.ndarray, s_um: float) -> list:
    """Считаем rg (в безразмерном виде), как в Prescott.

    В их коде дальше делается rg_um = rg_dimless * s_um.
    Поэтому здесь возвращаем rg_um/s_um.
    """
    x = np.asarray(centers_um, dtype=float)
    if x.size < 2:
        return [1.0]

    rg_um = np.zeros_like(x)
    rg_um[0] = x[1] - x[0]
    rg_um[-1] = x[-1] - x[-2]
    if x.size > 2:
        rg_um[1:-1] = 0.5 * (x[2:] - x[:-2])

    rg_um = np.maximum(rg_um, 1e-9)
    return (rg_um / float(s_um)).tolist()


class LinearMechanismCoupler:
    """Сборка LinearMechanism для связи vext[1] между двумя наборами секций."""

    def __init__(
        self,
        *,
        secs_first: list,
        secs_second: list,
        rg_dimless: list,
        rd_ohm_um2: float,
        s_um: float,
        nodeD_um: float,
        layer_index: int = 2,
        conductance_scale: float = 1.0,
    ):
        self.secs_first = list(secs_first)
        self.secs_second = list(secs_second)
        self.rg_dimless = list(rg_dimless)
        self.rd_ohm_um2 = float(rd_ohm_um2)
        self.s_um = float(s_um)
        self.nodeD_um = float(nodeD_um)
        self.layer_index = int(layer_index)
        self.conductance_scale = float(conductance_scale)

        self._lm = None
        self._keep = {}

    @property # Зачем здесь стоит Property?
    def lm(self):
        return self._lm

    def build(self):
        if len(self.secs_first) != len(self.secs_second):
            raise ValueError("Списки секций должны быть одинаковой длины")

        sl = h.SectionList()
        nsegs = 0
        for s in self.secs_first:
            sl.append(s)
            nsegs += int(s.nseg)

        nsegs2 = 0
        for s in self.secs_second:
            sl.append(s)
            nsegs2 += int(s.nseg)

        if nsegs != nsegs2:
            raise ValueError("Несовпадение числа сегментов для LinearMechanism")

        rg = np.asarray(self.rg_dimless, dtype=float)
        if rg.size not in (nsegs, nsegs + 1):
            raise ValueError(f"rg_dimless должен быть длины nsegs или nsegs+1. Получено {rg.size}, ожидалось {nsegs}")
        if rg.size == nsegs + 1:
            rg = rg[:nsegs]

        gmat = h.Matrix(2 * nsegs, 2 * nsegs)
        cmat = h.Matrix(2 * nsegs, 2 * nsegs)
        bvec = h.Vector(2 * nsegs)
        xl = h.Vector(2 * nsegs)
        layer = h.Vector(2 * nsegs)
        e = h.Vector(2 * nsegs)

        layer.fill(self.layer_index)
        for i in range(2 * nsegs):
            xl[i] = 0.5

        # Prescott: rg_um = rg_dimless * s_um
        rg_um = rg * float(self.s_um)
        resistance_ohm = float(self.rd_ohm_um2) / rg_um

        # Prescott: area_um2 = pi * nodeD
        area_um2 = math.pi * float(self.nodeD_um)
        area_cm2 = area_um2 * 1e-8
        ge = (1.0 / (resistance_ohm * area_cm2)) * float(self.conductance_scale)

        for i in range(nsegs):
            g = float(ge[i])
            gmat.setval(i, i, g)
            gmat.setval(i, nsegs + i, -g)
            gmat.setval(nsegs + i, i, -g)
            gmat.setval(nsegs + i, nsegs + i, g)

        self._keep = {
            "sl": sl,
            "gmat": gmat,
            "cmat": cmat,
            "bvec": bvec,
            "xl": xl,
            "layer": layer,
            "e": e,
        }
        self._lm = h.LinearMechanism(cmat, gmat, e, bvec, sl, xl, layer)
        return self


class TwoSensoryAxonsPrescott:
    """Два аксона + boundary-cable + ephaptic coupling (основной режим, как у Prescott)."""

    def __init__(
        self,
        *,
        fiber_diameter_um: float = 5.7,
        edge_dist_um: float = 0.1,
        aligned: bool = True,
        enable_ephaptic: bool = True,
        # Параметры аксона A
        parent_axon_nodes_A: int = 42,
        branch_nodes_A: int = 21,
        branches_num_A: int = 0,
        nodes_dist_A: int = 10,
        branch_every_um_A=None,
        # Параметры аксона B
        parent_axon_nodes_B: int = 42,
        branch_nodes_B: int = 21,
        branches_num_B: int = 2,
        nodes_dist_B: int = 10,
        branch_every_um_B=None,
        # Общие
        diam_scale: float = 0.6,
        celsius: float = 37.0,
        dt_ms: float = 0.005,
        v_init: float = -80.0,
        h_stop: float = 1000.0,
        gnapbar_scale: float = 0.5,
        XG1: float = 1e-9,
        rho_endoneurium_ohm_cm: float = 1211.0,
        rho_perineurium_ohm_cm: float = 1.136e5,
        perineurium_thickness_cm: float = 4.7e-4,
        boundary_full_cable: bool = False,
        misalignment_um: Optional[float] = None,
        ec_strength_scale: float = 1.0,
    ):
        self.fiber_diameter_um = float(fiber_diameter_um)
        self.edge_dist_um = float(edge_dist_um)
        self.aligned = bool(aligned)
        self.enable_ephaptic = bool(enable_ephaptic)

        self.XG1 = float(XG1)
        self.rho_endoneurium_ohm_cm = float(rho_endoneurium_ohm_cm)
        self.rho_perineurium_ohm_cm = float(rho_perineurium_ohm_cm)
        self.perineurium_thickness_cm = float(perineurium_thickness_cm)
        self.boundary_full_cable = bool(boundary_full_cable)
        self.misalignment_um = misalignment_um
        self.ec_strength_scale = float(ec_strength_scale)

        # 1) Строим аксон A (он очищает NEURON)
        self.axonA = MRGaxon(
            fiber_diameter=self.fiber_diameter_um,
            parent_axon_nodes=parent_axon_nodes_A,
            branch_nodes=branch_nodes_A,
            branches_num=branches_num_A,
            nodes_dist=nodes_dist_A,
            branch_every_um=branch_every_um_A,
            diam_scale=diam_scale,
            celsius=celsius,
            dt_ms=dt_ms,
            v_init=v_init,
            h_stop=h_stop,
            gnapbar_scale=gnapbar_scale,
            reset_nrn=True,
        )

        # 2) Строим аксон B (НЕ очищает NEURON)
        self.axonB = MRGaxon(
            fiber_diameter=self.fiber_diameter_um,
            parent_axon_nodes=parent_axon_nodes_B,
            branch_nodes=branch_nodes_B,
            branches_num=branches_num_B,
            nodes_dist=nodes_dist_B,
            branch_every_um=branch_every_um_B,
            diam_scale=diam_scale,
            celsius=celsius,
            dt_ms=dt_ms,
            v_init=v_init,
            h_stop=h_stop,
            gnapbar_scale=gnapbar_scale,
            reset_nrn=False,
        )

        # 3) Применяем Prescott-параметры для vext[1]
        self.xrA = self.axonA.apply_prescott_extracellular_layer1(edge_dist_um=self.edge_dist_um, XG1=self.XG1)
        self.xrB = self.axonB.apply_prescott_extracellular_layer1(edge_dist_um=self.edge_dist_um, XG1=self.XG1)

        # ---------------------------------------------------------------------------------
        # Prescott: координаты по X важны для aligned/misaligned.
        # 1) пересчитываем геометрию ствола в собственной системе координат
        # 2) применяем продольный сдвиг (misaligned) к аксону B
        # ---------------------------------------------------------------------------------
        self.axonA.recompute_trunk_geometry_for_coupling()
        self.axonB.recompute_trunk_geometry_for_coupling()

        Lstep = float(self.axonA.mrg_params.get('Lstep', 1.0))
        if self.aligned:
            offB = 0.0
        else:
            # Prescott misaligned: полушаг по продольной оси.
            offB = float(misalignment_um) if misalignment_um is not None else 0.5 * Lstep
        self.axonA.apply_longitudinal_offset(0.0)
        self.axonB.apply_longitudinal_offset(offB)
        self._offsetA_um = 0.0
        self._offsetB_um = offB

        # 4) Boundary cables (по одному на аксон, как в 2-fiber сценариях Prescott)
        # ВАЖНО: по умолчанию boundary строится в "sparse" виде (см. комментарии в BoundaryCable)
        #         для скорости. Полный кабель можно включить boundary_full_cable=True.
        self.boundaryA = None
        self.boundaryB = None

        # 5) Couplers
        self.coupler_AB = None
        self.coupler_A_boundary = None
        self.coupler_B_boundary = None
        self.spec_AB = None
        self.specA_boundary = None
        self.specB_boundary = None
        self._hoc_keepalive = []

        self._build_all_couplers()

    # --------------------------- построение карт связи ---------------------------
    def _spec_aligned_nodes(self) -> EphapticSpec:
        # ---------------------------------------------------------------------------------
        # ИЗМЕНЕНО (2026-03): раньше тут было namesA=["node_0","node_1",...]
        # Это неверно для ветвящегося аксона: после ветвления имена нод на стволе
        # становятся НЕ последовательными (например, node_0..node_10, затем node_32,...).
        # Поэтому строим карту по фактическому списку ствола axon.main_axon.
        # ---------------------------------------------------------------------------------
        trunkA = list(self.axonA.main_axon)
        trunkB = list(self.axonB.main_axon)
        n = min(len(trunkA), len(trunkB))
        namesA = [trunkA[i].name() for i in range(n)]
        namesB = [trunkB[i].name() for i in range(n)]

        # Центры берём по продольной координате узлов на стволе
        centers = np.asarray([
            float(self.axonA.node_distance_um.get(namesA[i], i * float(self.axonA.mrg_params.get('Lstep', 1.0))))
            for i in range(n)
        ], dtype=float)
        s_um = float(self.fiber_diameter_um)
        rg_dimless = _compute_rg_dimless_from_centers(centers, s_um)
        return EphapticSpec(namesA, namesB, rg_dimless)

    def _spec_misaligned_node_stin(self) -> EphapticSpec:
        """Misaligned как у Prescott: полушаговый продольный сдвиг + чередование node/STIN_mid.

        Важно:
          - геометрический сдвиг задаётся через apply_longitudinal_offset() для аксона B
          - здесь мы строим пары так, чтобы они совпадали по глобальной координате X:
              A: STIN_mid(i)  <-> B: node_i
              A: node_{i+1}   <-> B: STIN_mid(i+1)
        """
        trunkA = list(self.axonA.main_axon)
        trunkB = list(self.axonB.main_axon)
        n = min(len(trunkA), len(trunkB))
        if n < 2:
            return self._spec_aligned_nodes()

        first = []
        second = []
        centers = []
        s_um = float(self.fiber_diameter_um)

        for i in range(n - 1):
            # i задаёт интервал
            nA1 = trunkA[i + 1].name()   # next node on A
            nB0 = trunkB[i].name()       # current node on B
            nB1 = trunkB[i + 1].name()   # next node on B

            stA = self.axonA._trunk_stin_mid_by_next_node.get(nA1, None)
            stB = self.axonB._trunk_stin_mid_by_next_node.get(nB1, None)

            if stA is None or stB is None:
                # Если нет STIN-mid (в вашей топологии такое бывает около ветвления,
                # потому что там есть прямое соединение node->node), то этот интервал
                # пропускаем. node-node тут даст искусственный dx=offset.
                continue

            # Пара 1: A STIN_mid(i) <-> B node_i
            first.append(stA.name())
            second.append(nB0)
            centers.append(float(self.axonA.trunk_center_um.get(stA.name(), self.axonA.node_distance_um.get(nA1, i))))

            # Пара 2: A node_{i+1} <-> B STIN_mid(i+1)
            first.append(nA1)
            second.append(stB.name())
            centers.append(float(self.axonA.node_distance_um.get(nA1, i + 1)))

        rg_dimless = _compute_rg_dimless_from_centers(np.asarray(centers, dtype=float), s_um)
        return EphapticSpec(first, second, rg_dimless)

    def _spec_boundary_for_axon(self, axon: MRGaxon, boundary: Optional[BoundaryCable]) -> EphapticSpec:
        """Карта node_i <-> boundary.section_j по продольной координате (как у Prescott)."""
        trunk = list(axon.main_axon)

        # ДОБАВЛЕНО (2026-03): включаем дочерние ветви в boundary coupling.
        # Иначе vext[1] на ветви остаётся "плавающим" (изолированным) и может давать
        # ранние/самопроизвольные деполяризации и плохую проводимость.
        extra_nodes = []
        for br in getattr(axon, 'branches', []) or []:
            extra_nodes.extend(list(br))

        # Уникализируем по имени
        seen = set()
        all_nodes = []
        for sec in trunk + extra_nodes:
            nm = sec.name()
            if nm in seen:
                continue
            seen.add(nm)
            all_nodes.append(sec)

        pairs = []  # (x, node_name, boundary_name)

        # В sparse-режиме boundary может быть None при первом проходе (когда мы только
        # вычисляем, какие секции boundary понадобятся).
        if boundary is not None:
            sec_len = float(boundary.section_length_um)
            nsec = int(boundary.n_sections)
        else:
            # по умолчанию как в Prescott: 40000/4000 = 10 um
            sec_len = 10.0
            nsec = 4000
        Lstep = float(axon.mrg_params.get('Lstep', 1.0)) if hasattr(axon, 'mrg_params') else 1.0

        for i, sec in enumerate(all_nodes):
            node_name = sec.name()
            x = float(axon.node_distance_um.get(node_name, i * Lstep))
            bi = int(max(0, min(nsec - 1, round(x / sec_len))))
            bname = f"section_{bi}"
            pairs.append((x, node_name, bname))

        # Сортируем по координате (для корректного rg)
        pairs.sort(key=lambda t: float(t[0]))

        # Разрешаем дубликаты координаты: добавляем микросдвиг, чтобы rg не занулялся.
        xs = []
        names_first = []
        names_second = []
        last_x = None
        eps = 1e-3  # um
        for k, (x, node_name, bname) in enumerate(pairs):
            x = float(x)
            if last_x is not None and x <= last_x:
                x = last_x + eps
            last_x = x
            xs.append(x)
            names_first.append(node_name)
            names_second.append(bname)

        s_um = float(axon.fiber_diameter)
        rg_dimless = _compute_rg_dimless_from_centers(np.asarray(xs, dtype=float), s_um)
        return EphapticSpec(names_first, names_second, rg_dimless)

    # --------------------------- сборка LinearMechanism ---------------------------
    def _build_all_couplers(self):
        # 5.1 axon-axon coupling (эндoневрий) -- можно отключить для сравнения
        if self.enable_ephaptic:
            if self.aligned:
                spec = self._spec_aligned_nodes()
            else:
                spec = self._spec_misaligned_node_stin()

            # сохраним для графиков анатомии/связи
            self.spec_AB = spec

            secsA = [self.axonA.get_sec(nm) for nm in spec.sec_names_first]
            secsB = [self.axonB.get_sec(nm) for nm in spec.sec_names_second]

            rho_ohm_um = self.rho_endoneurium_ohm_cm * 10000.0
            rd = rho_ohm_um * float(self.edge_dist_um)  # ohm*um^2
            s_um = 0.5 * (float(self.axonA.fiber_diameter) + float(self.axonB.fiber_diameter))
            nodeD_um = 0.5 * (float(self.axonA.mrg_params['nodeD']) + float(self.axonB.mrg_params['nodeD']))

            self.coupler_AB = LinearMechanismCoupler(
                secs_first=secsA,
                secs_second=secsB,
                rg_dimless=spec.rg_dimless,
                rd_ohm_um2=rd,
                s_um=s_um,
                nodeD_um=nodeD_um,
                layer_index=2,
                conductance_scale=self.ec_strength_scale,
            ).build()
        else:
            self.coupler_AB = None

        # 5.2 axon-boundary coupling (perineurium)
        rd_b = self.rho_perineurium_ohm_cm * 10000.0 * self.perineurium_thickness_cm * 10000.0

        # Сначала создаём boundary, под нужные секции.
        specA_tmp = self._spec_boundary_for_axon(self.axonA, boundary=None)
        specB_tmp = self._spec_boundary_for_axon(self.axonB, boundary=None)

        sparseA = None if self.boundary_full_cable else specA_tmp.sec_names_second
        sparseB = None if self.boundary_full_cable else specB_tmp.sec_names_second

        self.boundaryA = BoundaryCable(name_prefix="bA_", sparse_section_names=sparseA)
        self.boundaryB = BoundaryCable(name_prefix="bB_", sparse_section_names=sparseB)
        self.boundaryA.set_grounded_sink()
        self.boundaryB.set_grounded_sink()

        specA = self._spec_boundary_for_axon(self.axonA, boundary=self.boundaryA)
        specB = self._spec_boundary_for_axon(self.axonB, boundary=self.boundaryB)

        self.specA_boundary = specA
        self.specB_boundary = specB

        secsA1 = [self.axonA.get_sec(nm) for nm in specA.sec_names_first]
        bAsecs = [self.boundaryA.secs[nm] for nm in specA.sec_names_second]
        self.coupler_A_boundary = LinearMechanismCoupler(
            secs_first=secsA1,
            secs_second=bAsecs,
            rg_dimless=specA.rg_dimless,
            rd_ohm_um2=rd_b,
            s_um=float(self.axonA.fiber_diameter),
            nodeD_um=float(self.axonA.mrg_params['nodeD']),
            layer_index=2,
        ).build()

        secsB1 = [self.axonB.get_sec(nm) for nm in specB.sec_names_first]
        bBsecs = [self.boundaryB.secs[nm] for nm in specB.sec_names_second]
        self.coupler_B_boundary = LinearMechanismCoupler(
            secs_first=secsB1,
            secs_second=bBsecs,
            rg_dimless=specB.rg_dimless,
            rd_ohm_um2=rd_b,
            s_um=float(self.axonB.fiber_diameter),
            nodeD_um=float(self.axonB.mrg_params['nodeD']),
            layer_index=2,
        ).build()

        # Сохраняем ссылки на hoc-объекты, чтобы NEURON не удалил их сборщиком мусора.
        self._hoc_keepalive = [
            self.coupler_AB,
            self.coupler_A_boundary,
            self.coupler_B_boundary,
            self.boundaryA,
            self.boundaryB,
        ]

    # ------------------------------------------------------------------------------------
    # ДОБАВЛЕНО (2026-03): График "анатомии" в 2D (продольная схема)
    #
    # Требование из ТЗ:
    #   - показать два аксона рядом
    #   - показать связи (ephaptic coupling) между ними
    #   - показать ветвление под 90 градусов (в сторону от ствола)
    #
    # Это СХЕМА, а не реальная 3D геометрия NEURON.
    # ------------------------------------------------------------------------------------
    def plot_anatomy_2d(self, save_path: Optional[str] = None, max_links: int = 120):
        import matplotlib.pyplot as plt

        # расстояние между центрами (как в Prescott C.txt/R.txt: dx = 2R + edge_dist)
        y_sep = float(self.fiber_diameter_um) + float(self.edge_dist_um)

        # ---------------------------------------------------------------------------------
        # ИЗМЕНЕНО (2026-03): строим ствол по фактическому списку axon.main_axon.
        # После ветвления имена нод на стволе могут "перепрыгивать" (node_10 -> node_32),
        # поэтому вариант node_0..node_N рисовал ствол обрезанным.
        # ---------------------------------------------------------------------------------
        def _trunk_xy(axon: MRGaxon, y0: float):
            xs = []
            ys = []
            Lstep = float(axon.mrg_params.get('Lstep', 1.0)) if hasattr(axon, 'mrg_params') else 1.0
            for i, sec in enumerate(list(axon.main_axon)):
                nm = sec.name()
                x = float(axon.node_distance_um.get(nm, i * Lstep))
                xs.append(x)
                ys.append(float(y0))
            # На всякий случай сортируем по x
            order = np.argsort(np.asarray(xs, dtype=float))
            xs = np.asarray(xs, dtype=float)[order]
            ys = np.asarray(ys, dtype=float)[order]
            return xs, ys

        xa, ya = _trunk_xy(self.axonA, 0.0)
        xb, yb = _trunk_xy(self.axonB, y_sep)

        fig, ax = plt.subplots(1, 1, figsize=(13, 4.2), dpi=160)
        ax.plot(xa, ya, color='green', lw=2.5, alpha=0.8, label='Axon A')
        ax.plot(xb, yb, color='green', lw=2.5, alpha=0.8, ls='--', label='Axon B')

        # Ветвления (рисуем как вертикальный отвод + короткий горизонтальный хвост)
        def _draw_branches(axon: MRGaxon, y0: float, up: bool):
            if not hasattr(axon, 'branch_point_distance_um'):
                return
            for x0 in getattr(axon, 'branch_point_distance_um', []):
                x0 = float(x0)
                dy = (0.6 * y_sep) if up else (-0.6 * y_sep)
                ax.plot([x0, x0], [y0, y0 + dy], color='black', lw=2.0, alpha=0.8)
                ax.plot([x0, x0 + 0.15 * y_sep], [y0 + dy, y0 + dy], color='black', lw=2.0, alpha=0.8)

        _draw_branches(self.axonA, 0.0, up=False)
        _draw_branches(self.axonB, y_sep, up=True)

        # Линии связи между аксонами (чтобы не было "месива", рисуем не больше max_links)
        if self.enable_ephaptic and self.spec_AB is not None:
            namesA = self.spec_AB.sec_names_first
            namesB = self.spec_AB.sec_names_second
            n = min(len(namesA), len(namesB))
            if n > 0:
                step = max(1, int(round(n / max_links)))
                for i in range(0, n, step):
                    a = namesA[i]
                    b = namesB[i]
                    xa0 = float(self.axonA.trunk_center_um.get(a, self.axonA.node_distance_um.get(a, np.nan)))
                    xb0 = float(self.axonB.trunk_center_um.get(b, self.axonB.node_distance_um.get(b, np.nan)))
                    if not np.isfinite(xa0) or not np.isfinite(xb0):
                        continue
                    ax.plot([xa0, xb0], [0.0, y_sep], color='#1f2937', lw=1.0, alpha=0.18)

        ax.set_title(
            f"Two sensory axons (Prescott-style ephaptic + boundary) | edge_dist={self.edge_dist_um} um | "
            f"{'aligned' if self.aligned else 'misaligned'}"
        )
        ax.set_xlabel('X (um)')
        ax.set_ylabel('Lateral position (um)')
        ax.grid(True, alpha=0.25)
        ax.legend(frameon=False)
        ax.set_ylim(-1.3 * y_sep, 2.3 * y_sep)

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close(fig)

    # ------------------------------------------------------------------------------------
    # ДОБАВЛЕНО (2026-03): Анатомия "как в Matlab", но схема.
    #
    # Требование:
    #   1) Поперечный разрез: 2 аксона + внешний контур + boundary
    #   2) Продольный разрез: все сегменты (node/MYSA/FLUT/STIN) подкрашены + связи
    #      при aligned/misaligned + ephaptic/no ephaptic
    #   3) В продольной схеме boundary рисуем как одну аккуратную линию/полосу,
    #      а не тысячи boundary-секций.
    # ------------------------------------------------------------------------------------
    def plot_cross_section_2d(self, save_path: Optional[str] = None, show: bool = True):
        """Поперечный разрез: 2 волокна + внешний контур + boundary (схема)."""
        import matplotlib.pyplot as plt
        from matplotlib.patches import Circle

        R = float(self.fiber_diameter_um) / 2.0
        sep = 2.0 * R + float(self.edge_dist_um)  # расстояние между центрами

        # Позиции центров двух волокон
        cA = (-0.5 * sep, 0.0)
        cB = (0.5 * sep, 0.0)

        # Толщина периневрия (для внешнего контура). По умолчанию ~4.7 um.
        per_th_um = float(self.perineurium_thickness_cm) * 10000.0

        # "Boundary" берём как контур, который чуть с запасом охватывает оба волокна.
        bound_r = max(abs(cA[0]), abs(cB[0])) + R + 0.5 * float(self.edge_dist_um)
        outer_r = bound_r + per_th_um

        fig, ax = plt.subplots(1, 1, figsize=(5.4, 5.4), dpi=160)

        # Внешний контур
        ax.add_patch(Circle((0.0, 0.0), outer_r, fill=False, lw=2.0, ec="#111827", alpha=0.85))
        ax.add_patch(Circle((0.0, 0.0), bound_r, fill=False, lw=2.0, ec="#6b7280", alpha=0.85, ls="--"))

        # Волокна
        ax.add_patch(Circle(cA, R, fc="#93c5fd", ec="#1e3a8a", lw=2.0, alpha=0.85))
        ax.add_patch(Circle(cB, R, fc="#86efac", ec="#065f46", lw=2.0, alpha=0.85))

        ax.text(cA[0], cA[1], "A", ha="center", va="center", fontsize=13, fontweight="bold", color="#0f172a")
        ax.text(cB[0], cB[1], "B", ha="center", va="center", fontsize=13, fontweight="bold", color="#0f172a")

        title = (
            f"Поперечная схема | D={self.fiber_diameter_um} мкм | edge={self.edge_dist_um} мкм | "
            f"{'aligned' if self.aligned else 'misaligned'} | "
            f"{'EC' if self.enable_ephaptic else 'No EC'}"
        )
        ax.set_title(title)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("x (мкм)")
        ax.set_ylabel("y (мкм)")
        ax.grid(True, alpha=0.18)

        # Пределы
        lim = outer_r * 1.15
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)

        # Легенда (минимальная)
        ax.plot([], [], color="#111827", lw=2.0, label="Внешний контур")
        ax.plot([], [], color="#6b7280", lw=2.0, ls="--", label="Граница (boundary)")
        ax.legend(frameon=False, loc="upper right")

        fig.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
        if show:
            plt.show()
        plt.close(fig)

    def plot_longitudinal_2d(self, save_path: Optional[str] = None, show: bool = True, max_links: int = 160):
        """Продольный разрез: все сегменты подкрашены + межаксонные связи."""
        import matplotlib.pyplot as plt
        from matplotlib.lines import Line2D

        # расстояние между центрами (как в Prescott: dx = 2R + edge_dist)
        y_sep = float(self.fiber_diameter_um) + float(self.edge_dist_um)
        yA = 0.0
        yB = y_sep

        # Цвета сегментов
        colors = {
            "node": "#1d4ed8",  # blue
            "mysa": "#f59e0b",  # amber
            "flut": "#f97316",  # orange
            "stin": "#9ca3af",  # gray
            "boundary": "#6b7280",
            "link": "#111827",
        }

        # Толщина дорожек
        lw_node = 6.0
        lw_other = 4.0

        def _node_x(axon: MRGaxon, name: str) -> float:
            # STIN mid на стволе лежит в trunk_center_um
            if hasattr(axon, "trunk_center_um") and name in axon.trunk_center_um:
                return float(axon.trunk_center_um[name])
            if hasattr(axon, "node_distance_um") and name in axon.node_distance_um:
                return float(axon.node_distance_um[name])
            return float("nan")

        def _trunk_x_limits(axon: MRGaxon):
            xs = [float(axon.node_distance_um.get(sec.name(), float("nan"))) for sec in list(axon.main_axon)]
            xs = [x for x in xs if np.isfinite(x)]
            if not xs:
                return 0.0, 1.0
            x0 = min(xs) - 0.5 * float(axon.nodelength)
            x1 = max(xs) + 0.5 * float(axon.nodelength)
            return x0, x1

        def _draw_boundary_band(ax, x0: float, x1: float, y0: float):
            # "одна полоса" boundary над аксоном
            band_y = y0 + 0.28 * y_sep
            band_h = 0.10 * y_sep
            ax.fill_between([x0, x1], [band_y - 0.5 * band_h, band_y - 0.5 * band_h],
                            [band_y + 0.5 * band_h, band_y + 0.5 * band_h],
                            color=colors["boundary"], alpha=0.10, linewidth=0)
            ax.plot([x0, x1], [band_y, band_y], color=colors["boundary"], lw=2.0, alpha=0.55)

        def _draw_trunk(ax, axon: MRGaxon, y0: float, label: str):
            # 1) boundary band
            x0, x1 = _trunk_x_limits(axon)
            _draw_boundary_band(ax, x0, x1, y0)

            # 2) node segments
            node_half = 0.5 * float(axon.nodelength)
            for sec in list(axon.main_axon):
                nm = sec.name()
                xc = float(axon.node_distance_um.get(nm, float("nan")))
                if not np.isfinite(xc):
                    continue
                ax.plot([xc - node_half, xc + node_half], [y0, y0], color=colors["node"], lw=lw_node,
                        solid_capstyle="round")

            # 3) internode segments from step records (only trunk)
            for rec in getattr(axon, "_step_records", []) or []:
                if not rec.get("is_trunk", False):
                    continue
                parent = rec.get("parent", None)
                if not parent:
                    continue
                x_parent = float(axon.node_distance_um.get(parent, float("nan")))
                if not np.isfinite(x_parent):
                    continue
                cur = x_parent + node_half
                for seg in rec.get("sections", []) or []:
                    typ = str(seg.get("type", ""))
                    L = float(seg.get("L", 0.0))
                    if L <= 0:
                        continue
                    col = colors.get(typ, "#000000")
                    ax.plot([cur, cur + L], [y0, y0], color=col, lw=lw_other, solid_capstyle="butt")
                    cur += L

            # 4) branch stubs (схематично, под 90 градусов)
            for xbr in getattr(axon, "branch_point_distance_um", []) or []:
                if xbr is None:
                    continue
                xbr = float(xbr)
                dy = 0.55 * y_sep
                # небольшой цветной "стек" сегментов (node->mysa->flut->stin)
                # в продольной схеме это только обозначение типа сегментов на ветви.
                y1 = y0 + (dy if y0 > 0 else -dy)
                ax.plot([xbr, xbr], [y0, y1], color="#111827", lw=2.0, alpha=0.8)
                # цветные маркеры по пути
                for frac, typ in [(0.12, "node"), (0.30, "mysa"), (0.48, "flut"), (0.70, "stin")]:
                    yy = y0 + (y1 - y0) * frac
                    ax.plot([xbr, xbr], [yy - 0.02 * y_sep, yy + 0.02 * y_sep], color=colors[typ], lw=6.0,
                            solid_capstyle="round")

            # подпись аксона
            ax.text(x0, y0 + 0.10 * y_sep, label, ha="left", va="bottom", fontsize=11, color="#111827")

        fig, ax = plt.subplots(1, 1, figsize=(13.8, 4.8), dpi=160)
        _draw_trunk(ax, self.axonA, yA, "Аксон A")
        _draw_trunk(ax, self.axonB, yB, "Аксон B")

        # Межаксонные связи
        if self.enable_ephaptic and self.spec_AB is not None:
            namesA = list(self.spec_AB.sec_names_first)
            namesB = list(self.spec_AB.sec_names_second)
            n = min(len(namesA), len(namesB))
            if n > 0:
                step = max(1, int(round(n / max_links)))
                for i in range(0, n, step):
                    a = namesA[i]
                    b = namesB[i]
                    xa0 = _node_x(self.axonA, a)
                    xb0 = _node_x(self.axonB, b)
                    if not np.isfinite(xa0) or not np.isfinite(xb0):
                        continue
                    ax.plot([xa0, xb0], [yA, yB], color=colors["link"], lw=1.0, alpha=0.16)

        title = (
            f"Продольная схема | edge={self.edge_dist_um} мкм | "
            f"{'aligned' if self.aligned else 'misaligned'} | "
            f"{'EC' if self.enable_ephaptic else 'No EC'}"
        )
        ax.set_title(title)
        ax.set_xlabel("X (мкм)")
        ax.set_ylabel("Поперечная позиция (мкм)")
        ax.grid(True, alpha=0.20)

        # Легенда
        handles = [
            Line2D([0], [0], color=colors["node"], lw=lw_node, label="node"),
            Line2D([0], [0], color=colors["mysa"], lw=lw_other, label="MYSA"),
            Line2D([0], [0], color=colors["flut"], lw=lw_other, label="FLUT"),
            Line2D([0], [0], color=colors["stin"], lw=lw_other, label="STIN"),
            Line2D([0], [0], color=colors["boundary"], lw=2.0, alpha=0.6, label="boundary (полоса)"),
        ]
        if self.enable_ephaptic:
            handles.append(Line2D([0], [0], color=colors["link"], lw=1.0, alpha=0.35, label="EC-связи"))
        ax.legend(handles=handles, frameon=False, loc="upper right", ncol=2)

        # Пределы по Y
        ax.set_ylim(-0.95 * y_sep, 2.25 * y_sep)

        fig.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
        if show:
            plt.show()
        plt.close(fig)

    # --------------------------- стим/запуск/сохранение ---------------------------
    def set_stimulation_for_axons(
        self,
        *,
        stim_A: bool,
        stim_B: bool,
        stim_target_mode: str = 'node_index',
        stim_node_index: int = 0,
        stim_x_um: Optional[float] = None,
        **stim_kwargs,
    ):
        """Удобный метод: одинаковый стим на A и/или B.

        stim_target_mode:
          - 'node_index' (по умолчанию): стимулируем main_axon[stim_node_index]
          - 'same_x_um': стимулируем узел ствола, ближайший к stim_x_um
        """
        if stim_A:
            self.axonA.set_stim_target(mode=stim_target_mode, node_index=stim_node_index, x_um=stim_x_um)
            self.axonA.set_stimulation_params(**stim_kwargs)
        if stim_B:
            self.axonB.set_stim_target(mode=stim_target_mode, node_index=stim_node_index, x_um=stim_x_um)
            self.axonB.set_stimulation_params(**stim_kwargs)

    def set_stimulation_for_axons_independent(
        self,
        *,
        stim_A: bool,
        stim_B: bool,
        stim_target_mode_A: str = 'node_index',
        stim_node_index_A: int = 0,
        stim_x_um_A: Optional[float] = None,
        stim_target_mode_B: str = 'node_index',
        stim_node_index_B: int = 0,
        stim_x_um_B: Optional[float] = None,
        stim_kwargs_A: Optional[dict] = None,
        stim_kwargs_B: Optional[dict] = None,
    ):
        """Независимая настройка стимула для AxonA и AxonB.

        Удобно для сценариев со сдвигом по времени, например:
          AxonA: t_start = 10 мс
          AxonB: t_start = 11 мс
        """
        stim_kwargs_A = {} if stim_kwargs_A is None else dict(stim_kwargs_A)
        stim_kwargs_B = {} if stim_kwargs_B is None else dict(stim_kwargs_B)

        if stim_A:
            self.axonA.set_stim_target(mode=stim_target_mode_A, node_index=stim_node_index_A, x_um=stim_x_um_A)
            self.axonA.set_stimulation_params(**stim_kwargs_A)
        if stim_B:
            self.axonB.set_stim_target(mode=stim_target_mode_B, node_index=stim_node_index_B, x_um=stim_x_um_B)
            self.axonB.set_stimulation_params(**stim_kwargs_B)

    def _collect_two_axon_recordings(
        self,
        *,
        include_stimulation_point: bool = True,
        record_axonA_before_like: bool = False,
        record_axonA_main_like: bool = False,
        record_terminal_nodes: bool = False,
    ):
        """Собирает точки записи для двухаксоновой модели.

        Доп. режим по запросу пользователя:
          - без stimulation_point
          - AxonA: точки before_like / main_like + terminal_main
          - AxonB: branch-related точки + terminal_main + terminal_daughter
        """

        extraA = []
        extraB = []

        if record_axonA_before_like and getattr(self.axonB, 'before_branch_id', None):
            seg_ref = self.axonB.before_branch_id[0]
            x_ref = float(self.axonB.main_path_distance_um.get(seg_ref.sec.name(), float('nan')))
            if np.isfinite(x_ref):
                segA = self.axonA.get_node_segment_nearest_main_path_distance(x_ref)
                if segA is not None:
                    extraA.append(('before_like', segA))

        if record_axonA_main_like and getattr(self.axonB, 'after_branch_main_id', None):
            seg_ref = self.axonB.after_branch_main_id[0]
            x_ref = float(self.axonB.main_path_distance_um.get(seg_ref.sec.name(), float('nan')))
            if np.isfinite(x_ref):
                segA = self.axonA.get_node_segment_nearest_main_path_distance(x_ref)
                if segA is not None:
                    extraA.append(('main_like', segA))

        if record_terminal_nodes:
            segA_term = self.axonA.get_terminal_main_segment()
            segB_term_main = self.axonB.get_terminal_main_segment()
            segB_term_dau = self.axonB.get_terminal_daughter_segment()
            if segA_term is not None:
                extraA.append(('terminal_main', segA_term))
            if segB_term_main is not None:
                extraB.append(('terminal_main', segB_term_main))
            if segB_term_dau is not None:
                extraB.append(('terminal_daughter', segB_term_dau))

        segA, namesA = self.axonA.collect_recording_targets(
            include_stimulation_point=include_stimulation_point,
            extra_named_segments=extraA,
        )
        segB, namesB = self.axonB.collect_recording_targets(
            include_stimulation_point=include_stimulation_point,
            extra_named_segments=extraB,
        )
        return segA, namesA, segB, namesB

    def run_simulation_two_axons(
        self,
        *,
        h5_path: Optional[str] = None,
        experiment_name: str = "Run",
        record_kinetics: bool = True,
        include_stimulation_point: bool = True,
        record_axonA_before_like: bool = False,
        record_axonA_main_like: bool = False,
        record_terminal_nodes: bool = False,
    ):
        """Запуск одной симуляции для двух аксонов.

        Параметры:
          h5_path: путь к HDF5. Если None, ничего не сохраняем (удобно для тестов).
          experiment_name: имя группы в HDF5 (если сохраняем).
          record_kinetics: записывать ли токи/гейты (newaxnode).
          include_stimulation_point: включать ли запись stimulation_point.
          record_axonA_before_like: добавить для AxonA точку, ближайшую к before_branch AxonB.
          record_axonA_main_like: добавить для AxonA точку, ближайшую к after_branch_main AxonB.
          record_terminal_nodes: добавить terminal_main / terminal_daughter точки записи.
        """

        # 1) Создаём стимуляторы (как в вашем коде)
        self.axonA.create_stimulator()
        self.axonB.create_stimulator()

        # 2) Собираем сегменты для записи
        segA, namesA, segB, namesB = self._collect_two_axon_recordings(
            include_stimulation_point=include_stimulation_point,
            record_axonA_before_like=record_axonA_before_like,
            record_axonA_main_like=record_axonA_main_like,
            record_terminal_nodes=record_terminal_nodes,
        )

        # 3) Векторы записи
        record_t = h.Vector().record(h._ref_t)
        recA_v = [h.Vector().record(seg._ref_v) for seg in segA]
        recB_v = [h.Vector().record(seg._ref_v) for seg in segB]

        # 4) Кинетика (по вашему же механизму newaxnode)
        recA_cur = {}
        recB_cur = {}
        if record_kinetics and self.axonA.node_mech == 'newaxnode':
            recA_cur['ina'] = [h.Vector().record(seg._ref_ina_newaxnode) for seg in segA]
            recA_cur['inap'] = [h.Vector().record(seg._ref_inap_newaxnode) for seg in segA]
            recA_cur['ikf'] = [h.Vector().record(seg._ref_ikf_newaxnode) for seg in segA]
            recA_cur['iks'] = [h.Vector().record(seg._ref_iks_newaxnode) for seg in segA]
            recA_cur['il'] = [h.Vector().record(seg._ref_il_newaxnode) for seg in segA]
            recA_cur['m'] = [h.Vector().record(seg._ref_m_newaxnode) for seg in segA]
            recA_cur['h'] = [h.Vector().record(seg._ref_h_newaxnode) for seg in segA]
            recA_cur['s'] = [h.Vector().record(seg._ref_s_newaxnode) for seg in segA]
            recA_cur['mp'] = [h.Vector().record(seg._ref_mp_newaxnode) for seg in segA]
            recA_cur['n'] = [h.Vector().record(seg._ref_n_newaxnode) for seg in segA]
        if record_kinetics and self.axonB.node_mech == 'newaxnode':
            recB_cur['ina'] = [h.Vector().record(seg._ref_ina_newaxnode) for seg in segB]
            recB_cur['inap'] = [h.Vector().record(seg._ref_inap_newaxnode) for seg in segB]
            recB_cur['ikf'] = [h.Vector().record(seg._ref_ikf_newaxnode) for seg in segB]
            recB_cur['iks'] = [h.Vector().record(seg._ref_iks_newaxnode) for seg in segB]
            recB_cur['il'] = [h.Vector().record(seg._ref_il_newaxnode) for seg in segB]
            recB_cur['m'] = [h.Vector().record(seg._ref_m_newaxnode) for seg in segB]
            recB_cur['h'] = [h.Vector().record(seg._ref_h_newaxnode) for seg in segB]
            recB_cur['s'] = [h.Vector().record(seg._ref_s_newaxnode) for seg in segB]
            recB_cur['mp'] = [h.Vector().record(seg._ref_mp_newaxnode) for seg in segB]
            recB_cur['n'] = [h.Vector().record(seg._ref_n_newaxnode) for seg in segB]

        # 5) Запуск NEURON
        h.finitialize(self.axonA.v_init)
        h.run()

        # 6) Упаковываем в numpy
        t = np.array(record_t)
        vA = np.vstack([np.array(v) for v in recA_v])
        vB = np.vstack([np.array(v) for v in recB_v])

        # 7) Сохраняем в объекты аксонов, чтобы работали их plot_* методы
        self.axonA.time_array = t
        self.axonA.voltage_matrix = vA
        self.axonA.recording_labels = namesA
        self.axonA.recording_indices = {i: {"group": nm, "node": f"{segA[i].sec.name().replace('.', '_')}_{segA[i].x:.2f}"} for i, nm in enumerate(namesA)}

        # Записываем матрицы кинетики в привычные имена, чтобы save_to_hdf5() сохранил их как у вас
        if record_kinetics and self.axonA.node_mech == 'newaxnode':
            self.axonA.ina_matrix = np.vstack([np.array(v) for v in recA_cur['ina']])
            self.axonA.inap_matrix = np.vstack([np.array(v) for v in recA_cur['inap']])
            self.axonA.ikf_matrix = np.vstack([np.array(v) for v in recA_cur['ikf']])
            self.axonA.iks_matrix = np.vstack([np.array(v) for v in recA_cur['iks']])
            self.axonA.il_matrix = np.vstack([np.array(v) for v in recA_cur['il']])
            self.axonA.m_matrix = np.vstack([np.array(v) for v in recA_cur['m']])
            self.axonA.h_matrix = np.vstack([np.array(v) for v in recA_cur['h']])
            self.axonA.s_matrix = np.vstack([np.array(v) for v in recA_cur['s']])
            self.axonA.mp_matrix = np.vstack([np.array(v) for v in recA_cur['mp']])
            self.axonA.n_matrix = np.vstack([np.array(v) for v in recA_cur['n']])

        self.axonB.time_array = t
        self.axonB.voltage_matrix = vB
        self.axonB.recording_labels = namesB
        self.axonB.recording_indices = {i: {"group": nm, "node": f"{segB[i].sec.name().replace('.', '_')}_{segB[i].x:.2f}"} for i, nm in enumerate(namesB)}

        if record_kinetics and self.axonB.node_mech == 'newaxnode':
            self.axonB.ina_matrix = np.vstack([np.array(v) for v in recB_cur['ina']])
            self.axonB.inap_matrix = np.vstack([np.array(v) for v in recB_cur['inap']])
            self.axonB.ikf_matrix = np.vstack([np.array(v) for v in recB_cur['ikf']])
            self.axonB.iks_matrix = np.vstack([np.array(v) for v in recB_cur['iks']])
            self.axonB.il_matrix = np.vstack([np.array(v) for v in recB_cur['il']])
            self.axonB.m_matrix = np.vstack([np.array(v) for v in recB_cur['m']])
            self.axonB.h_matrix = np.vstack([np.array(v) for v in recB_cur['h']])
            self.axonB.s_matrix = np.vstack([np.array(v) for v in recB_cur['s']])
            self.axonB.mp_matrix = np.vstack([np.array(v) for v in recB_cur['mp']])
            self.axonB.n_matrix = np.vstack([np.array(v) for v in recB_cur['n']])

        # 8) (опционально) Сохранение в HDF5: используем ваш save_to_hdf5() без изменения формата.
        #    Просто кладём данные в подгруппы /AxonA и /AxonB.
        if h5_path:
            self.axonA.save_to_hdf5(h5_path, experiment_name=f"{experiment_name}/AxonA", stimulator=self.axonA.stimulator)
            self.axonB.save_to_hdf5(h5_path, experiment_name=f"{experiment_name}/AxonB", stimulator=self.axonB.stimulator)

        return t, vA, vB

    # ------------------------------------------------------------------------------------
    # ДОБАВЛЕНО (2026-03): графики "как в одном аксоне", но сразу для двух.
    #
    # Требование:
    #   - вызывать как model.plot_voltage_traces(...) и model.plot_kinetics(...)
    #   - один аксон рисуется пунктиром, второй обычной линией
    #
    # Принятое соглашение:
    #   - AxonB: сплошная линия
    #   - AxonA: пунктир
    # ------------------------------------------------------------------------------------
    def plot_voltage_traces(self, save_path=None, plot_start=0, plot_end=1000, plot_branch=0):
        """Графики мембранного потенциала.

        По вашему решению (2026-03): в двух-аксоновой модели на графиках показываем
        только AxonB, потому что сравнение с AxonA чаще вводит в заблуждение
        (разные точки записи/ветвление/геометрия).

        Визуальный стиль берём из вашего оригинального MRGaxon.plot_voltage_traces().
        """

        # Делегируем в ваш аккуратный "одноаксоновый" метод.
        self.axonB.plot_voltage_traces(save_path=save_path, plot_start=plot_start, plot_end=plot_end, plot_branch=plot_branch)

    def plot_kinetics(self, segment_index_B=0, segment_index_A=None, plot_start=0, plot_end=1000, save_path=None, plot_branch=0):
        """Графики кинетики.

        По вашему решению (2026-03): показываем только AxonB.
        Визуальный стиль берём из вашего оригинального MRGaxon.plot_kinetics().

        Параметры segment_index_A/plot_branch оставлены для совместимости сигнатуры.
        """

        self.axonB.plot_kinetics(segment_index=int(segment_index_B), plot_start=plot_start, plot_end=plot_end, save_path=save_path)
