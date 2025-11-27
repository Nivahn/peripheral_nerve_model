from neuron import h
import matplotlib.pyplot as plt
import numpy as np
import math
import time
import os
import pandas as pd
from scipy.signal import find_peaks
from MRG_lib import *

# Создаем аксон один раз
axon = MRGaxon(
    fiber_diameter=5.7,
    parent_axon_nodes=42,
    branch_nodes=21,
    branches_num=2,
    nodes_dist=10,
    diam_scale=0.6,
    celsius=37.0,
    dt_ms=0.05,
    v_init=-80.0,
    h_stop=1000.0
)


# Анализ на частоте 50 Гц в течение 10 секунд
detailed_data = axon.analyze_single_frequency(
    freq=700,                       # 50 Гц
    amp=1.0,                       # 1 нА
    stimulation_duration_ms=1000, # 10 секунд
    plot_duration_ms=100          # показывать первые 1000 мс
)

''''''
# Дополнительный анализ эффективности
conduction_analysis = axon.analyze_conduction_efficiency(
    detailed_data['voltage_matrix'],
    detailed_data['time_array']
)

print(f"Анализ проведения при {detailed_data['frequency']} Гц:")
print(f"  Спайков до ветвления: {conduction_analysis['spikes_before']}")
print(f"  Спайков в основной ветви: {conduction_analysis['spikes_main']} (эффективность: {conduction_analysis['main_efficiency']:.1%})")
print(f"  Спайков в дочерней ветви: {conduction_analysis['spikes_daughter']} (эффективность: {conduction_analysis['daughter_efficiency']:.1%})")

