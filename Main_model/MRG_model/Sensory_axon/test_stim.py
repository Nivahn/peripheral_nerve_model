from neuron import h
import matplotlib.pyplot as plt
import numpy as np
from MRG_lib import *
dt = 0.005


def plot_start_end(start, end, dt):
    # Запуск с записью кинетики

    before_indices = axon.find_segment_by_name(node_name_pattern=None, group_name='before_branch')
    print(before_indices[0])
    plot_start = int(start // dt)
    plot_end = int(end // dt)

    axon.plot_voltage_traces(plot_start=plot_start, plot_end=plot_end)

    # Пример 2: Построить кинетику для сегмента "до ветвления"
    before_indices = axon.find_segment_by_name(node_name_pattern=None, group_name='before_branch')
    print(before_indices[0])
    if before_indices:
        axon.plot_kinetics(segment_index=before_indices[0],
                           plot_start=plot_start, plot_end=plot_end)
# Создание и запуск модели с записью кинетики
axon = MRGaxon(
    fiber_diameter=5.7,
    parent_axon_nodes=42,
    branch_nodes=21,
    branches_num=2,
    nodes_dist=10,
    diam_scale=0.6,
    dt_ms=dt,
    celsius=37.0
)

# Настройка стимуляции
axon.set_stimulation_params(
    mode="create",
    biphasic=False,
    freq_hz=50,
    amp=20,
    t_start=50.0,
    t_end=500.0,
    phase_us=40.0,
    gap_us=5.0,
)

axon.plot_morphology_3d()
axon.run_simulation(record_kinetics=True)


plot_start_end(0, 100, dt=dt)

plot_start_end(0, 500, dt=dt)

#plot_start_end(2000, 300, dt=dt)

#lot_start_end(200, 250, dt=dt)

#plot_start_end(200, 500, dt=dt)

print("Would you like to inspect more? Yes/No")
x = input()
if x == "Yes":
    while x == "Yes":
        print("What start of plot?")
        start = int(input())
        print("What end of plot?")
        end = int(input())
        plot_start_end(start, end, dt=dt)
        print("continue ?")
        x = input()
elif x == "No":
    exit()
elif x != "Yes" or "No":
    print("Invalid insert")
