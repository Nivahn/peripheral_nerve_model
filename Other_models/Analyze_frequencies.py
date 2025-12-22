
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import h5py
import scipy
from scipy.signal import find_peaks

file_50_1Khz = r"C:\Users\User\PycharmProjects\peripheral_nerve_model\Data\MRG_MultiFreq_Stim_50_1000Hz_phase40_gap5_us.h5"


after_branch_daughter_1 = []
after_branch_daughter_2 = []
after_branch_main_1 = []
after_branch_main_2 = []
before_branch_main_1 = []
before_branch_main_2 = []
branch_main_1 = []
branch_main_2 = []
stimulation_point = []
#stimulator = []

with h5py.File(file_50_1Khz, 'r') as f:
    # Optional: see top-level keys
    for key in f.keys():
        print(key)


    for frequency in f.keys():
        if "Frequency" not in frequency:
            continue

        print("\n==============================")
        print(f"Частота: {frequency}")
        print("==============================")

        fig, ax = plt.subplots(nrows = 4, ncols = 2, figsize = (40, 40))
        #print(f"Neuron {neuron}")

        grp_traces = f[f"{frequency}/Model/Traces"]
        t_signal = f[f"{frequency}/Model/time"][:]  # общий time для модели
        t_stimulator = f[f"{frequency}/Stimulator/time"][:]
        stimulator = f[f"{frequency}/Stimulator/current"][:]

        for group_name, group in grp_traces.items():
            print(f"\nTrace group: {group_name}")
            keys = list(group.keys())
            if len(keys) == 2:
                node_1 = group[keys[0]]
                node_2 = group[keys[1]]

                print("Node 1:", node_1)
                print("Node 2:", node_2)

            for dset_name, dset in group.items():
                '''
                print(20 * "=")
                print("Общая информация по графику")
                print(f"Частота: {frequency}")
                print(f"Группа: {group_name}")
                print(f"Датасет: {dset_name}")
                print(20 * "=")
                '''



                signal = dset[:]
                t_signal = f[f"{frequency}/Model/time"][:]

                #plt.figure(figsize=(10, 4))
                #plt.plot(t_signal, signal, label="мембранный потенциал")



                #plt.plot(t_stimulator, stimulator)
                #plt.show()

                #print(f"Node {node}")
                #signal = trace_group[node][:]

                peaks, _ = find_peaks(signal, distance=10)
                print(f"Peaks {len(peaks)}")

                #plt.plot(signal[:10000])
                #plt.show()
                #plt.plot(stimulator[:10000])
                #plt.show()

                if group_name == "after_branch_daughter":
                    after_branch_daughter_1.append(peaks)
                    #ax[]
                elif group_name == "after_branch_main":
                    after_branch_main_1.append(peaks)
                elif group_name == "before_branch_main":
                    before_branch_main_1.append(peaks)
                elif group_name == "stimulation_point":
                    stimulation_point.append(peaks)
                    
