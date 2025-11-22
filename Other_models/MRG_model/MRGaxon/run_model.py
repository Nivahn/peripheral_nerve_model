from neuron import h
import matplotlib.pyplot as plt
import numpy as np
import random
import math
from lib import *

h.load_file('stdrun.hoc')

# ---------- РЕЕСТР, чтобы секции не удалялись GC ----------
REG = {"node": [], "mysa": [], "flut": [], "stin": []}
_node_id = _mysa_id = _flut_id = _stin_id = 0
reset_model()
# ---------- ГЛОБАЛЬНЫЕ ПАРАМЕТРЫ ----------
CELSIUS = 37.0
DT_MS   = 0.05
V_INIT  = -80.0

h.celsius = CELSIUS
h.dt = DT_MS

# Электрические константы, как в MRG
rhoa = 0.7e6  # Ohm·um
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
num_branching = 1

# ---------- СТРОЕНИЕ АКСОНА ----------

diam_scale = 0.6
parent_axon_nodes = 42
branch_nodes = 21
branches_num = 2
nodes_dist = 10
nodes = 0


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
#   fiberD: (g, axonD, nodeD, paraD1, paraD2, deltax, paralength2, nl)


terminals = []


P0 = mrg_params(fiberD=10)
P_curr = dict(P0)
# print("P0", P0, "P_curr", P_curr)

main_axon = [make_node(P_curr['nodeD'], nodelength, rhoa, P_curr['Rpn0'], NODE_MECH)]

node_D_after_branching = True
count_nodes_after_branching = 0


for _ in range(parent_axon_nodes - 1):

    if node_D_after_branching == True:
        count_nodes_after_branching = 3
        P_main_axon = scaled_params(P_curr, diam_scale=diam_scale)
        nxt = append_one_step(main_axon[-1], P_main_axon, node_mech=NODE_MECH)
        main_axon.append(nxt)
        # вставляем 1 шаг со скейлом на 60 %

        count_nodes_after_branching -= 1
        if count_nodes_after_branching == 0:
            node_D_after_branching = False

    elif node_D_after_branching == False:

        nxt = append_one_step(main_axon[-1], P_curr, node_mech=NODE_MECH)
        main_axon.append(nxt)

    nodes += 1
    # TODO Надо высчитать длину типичного шага при append one step, но через ноды удобнее.
    # TODO через вычисления длины нод сделать ветвление через какие-то промежутки
    if nodes >= nodes_dist and branches_num != 0:
        print(f"Ветвление при ноде: {nodes}")

        node_2 = make_node(P_curr['nodeD'], nodelength, rhoa, P_curr['Rpn0'], node_mech=NODE_MECH)
        node_2.connect(main_axon[-1], 1.0, 0.0)

        # Создаем ветвь
        P_branch = scaled_params(P_curr, diam_scale=diam_scale)
        term_chain = build_chain(branch_nodes, P_branch, node_mech=NODE_MECH)
        term_chain[0].connect(node_2, 0.0, 1.0)

        # Продолжение основного аксона
        node_3 = make_node(P_curr['nodeD'], nodelength, rhoa, P_curr['Rpn0'], node_mech=NODE_MECH)
        node_3.connect(node_2, 1.0, 0.0)

        # ИСПРАВЛЕННАЯ ОТЛАДОЧНАЯ ПЕЧАТЬ:
        print(f"Ветвление: {main_axon[-1].name()} -> {node_2.name()}")
        print(f"  Ветвь: {node_2.name()} -> {term_chain[0].name()}")
        print(f"  Продолжение: {node_2.name()} -> {node_3.name()}")

        terminals.append(node_2)
        main_axon.append(node_3)
        node_D_after_branching = True
        branches_num -= 1
        nodes = 0



check_branching()

# Пример: стим на узле до ветвления (предположим, это main_axon[3])
inj_sec = main_axon[0]

# Параметры стимуляции
freq_hz = 5              # Гц
T_ms = 1000.0 / freq_hz  # период, мс (200 мс)
amp = 0.1                # амплитуда тока (нА)
dt_ms = 0.01             # шаг, мс
t_start = 10           # мс
t_end = 1000.0           # мс
pulse_len_ms = 1.0       # длительность импульса, мс
phase_ms = 0.0           # сдвиг старта пачки, мс

# Создание стимулятора
stim = h.IClamp(inj_sec(0.5))

# Создание временного вектора и вектора тока
time_vec = h.Vector()
current_vec = h.Vector()

# Генерация формы сигнала
t_points = np.arange(0, t_end + dt_ms, dt_ms)
i_points = np.zeros(len(t_points))

# Расчет количества импульсов
n_pulses = int(np.floor((t_end - (t_start + phase_ms)) / T_ms)) + 1

# Заполнение импульсов
for k in range(n_pulses):
    t0 = t_start + phase_ms + k * T_ms
    t1 = t0 + pulse_len_ms
    if t0 > t_end:
        break
    # Находим индексы для включения импульса
    mask = (t_points >= t0) & (t_points < t1)
    i_points[mask] = amp

# Конвертируем numpy arrays в NEURON Vectors
time_vec.from_python(t_points)
current_vec.from_python(i_points)

# Подключаем форму тока к стимулятору
current_vec.play(stim._ref_amp, time_vec, 1)

# Настройки симуляции
h.dt = dt_ms
h.tstop = t_end

h.run()


record_v = []
record_t = []
i = 0
for _ in REG["node"]:
    vec = h.Vector().record(main_axon[i](0.5)._ref_v)
    t = h.Vector().record(h._ref_t)
    record_t.append(t)
    record_v.append(vec)
    i+1


h.tstop = 10000.0
#h.finitialize(V_INIT)
h.run()
#time_array = np.array(t)
voltage_matrix = np.vstack([np.array(v) for v in record_v])  # shape: (Nnodes, Nt)
time_matrix = np.vstack([np.array(t) for t in record_t])

# --- plot ---
first_node_voltage = voltage_matrix[0]
branch_voltage = voltage_matrix[int(nodes_dist + 4)]
branch_axon_voltage = voltage_matrix[int(nodes_dist + branch_nodes + 4)]

first_node_t = time_matrix[1]
branch_t = time_matrix[int(nodes_dist + 4)]
branch_axon_t = time_matrix[int(nodes_dist + branch_nodes + 4)]

plt.plot(first_node_t,first_node_voltage, alpha=0.7, label= f"first_node_voltage")
plt.plot(branch_t,branch_voltage, alpha=0.7, label= f"branch_voltage")
plt.plot(branch_axon_t,branch_axon_voltage, alpha=0.7, label= f"branch_axon_voltage")


plt.legend()
plt.xlabel("Время (мс)")
plt.ylabel("Мембранный потенциал (мВ)")
plt.title("Ответ на стимуляцию в главном аксоне")
plt.grid(True)
plt.show()