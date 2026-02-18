from impulse_generator import *
from MRG_lib import *


frequencies = list(range(50, 1001, 50))   # 50,150,...,950,1000 Гц
stimulation_duration_ms = 5000.0           # 1 секунда
amp_nA = 1                       # амплитуда импульса
dt = 0.005                            # dt модели
v_init = -80.0
h_stop_ms = 1010.0                         # после стимуляции ещё хвост


axon = MRGaxon(
    fiber_diameter=5.7,
    parent_axon_nodes=42,
    branch_nodes=21,
    branches_num=2,
    nodes_dist=10,
    diam_scale=0.6,
    celsius=37.0,
    dt_ms=dt,
    v_init=v_init,
    h_stop=h_stop_ms
)
frequencies = [10]


for freq in frequencies:

    print(f"\n=== Частота {freq} Гц ===")
    axon.set_stimulation_params(mode='create',
                                freq_hz=freq,
                                amp=amp_nA,
                                t_start=10,
                                t_end=h_stop_ms,
                                phase_us=40.0,
                                gap_us=5,
                                plot_duration=1)


    axon.run_simulation()

    plot_start = int(10 // dt)
    plot_end = int(1000 // dt)

    axon.plot_voltage_traces(plot_start =plot_start, plot_end=plot_end)

    plot_start = int(10 // dt)
    plot_end = int(100 // dt)

    axon.plot_voltage_traces(plot_start =plot_start, plot_end=plot_end)

    plot_start = int(10 // dt)
    plot_end = int(30 // dt)

    axon.plot_voltage_traces(plot_start =plot_start, plot_end=plot_end)

    plot_start = int(50 // dt)
    plot_end = int(100 // dt)

    axon.plot_voltage_traces(plot_start =plot_start, plot_end=plot_end)

    plot_start = int(10 // dt)
    plot_end = int(5000 // dt)

    axon.plot_voltage_traces(plot_start =plot_start, plot_end=plot_end)

"""    plot_start = int(4010 // dt)
    plot_end = int(5000 // dt)

    axon.plot_voltage_traces(plot_start =plot_start, plot_end=plot_end)

    plot_start = int(4500 // dt)
    plot_end = int(5000 // dt)

    axon.plot_voltage_traces(plot_start =plot_start, plot_end=plot_end)

    plot_start = int(1000 // dt)
    plot_end = int(2000 // dt)

    axon.plot_voltage_traces(plot_start =plot_start, plot_end=plot_end)"""
'''
csv_50Hz = "./../../../Data/SCS_LTMRs_50Hz_Timestamps.csv"

# 2) создаём стим
axon.set_stimulation_params(
    mode="preload_data",
    csv_path=csv_50Hz,
    neuron_index=0,
    index_is_one_based=False,
    t_end=h_stop_ms,  # МИЛЛИСЕКУНДЫ, не секунды
    amp=amp_nA,
    phase_us=40.0,
    gap_us=5.0,
    freq_hz = "Neuron"
)

plot_start = int(10 // dt)
plot_end = int(100 // dt)

axon.run_simulation()

axon.plot_voltage_traces(plot_start =plot_start, plot_end=plot_end)
'''