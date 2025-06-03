# Отладка
# import -----------------------
import numpy as np
import h5py
from scipy.stats import gamma
from numba import njit, prange
from Simulate_lib import *
# for optimization -------------
import cProfile
import pstats
import time
from tqdm import tqdm
import sys
# ------------------------------



class SPM_Model:
    def __init__(self, N=100, X=300, dx=0.1, max_time=300, mean_ISI=15,
                 stim_pattern_type='Regular', seed=45):
        self.N = N  # Количество аксонов
        self.X = X  # Длина аксона (мм)
        self.dx = dx  # Пространственный шаг (мм)
        self.max_time = max_time  # Максимальное время (мс)
        self.mean_ISI = mean_ISI  # Средний ISI (мс)
        self.stim_pattern_type = stim_pattern_type  # Тип паттерна
        self.seed = seed  # Сид для воспроизводимости

        np.random.seed(self.seed)
        self._initialize_simulation()

    def _initialize_simulation(self):
        """Инициализация параметров модели."""
        self.x = np.arange(0, self.X + self.dx, self.dx)
        self.v = np.zeros((self.N, len(self.x)))  # Напряжение мембраны
        self.I = np.zeros((self.N, len(self.x)))  # Ток
        self.dt = 0.00005  # Шаг по времени (мс)
        self.t = 0  # Начальное время



        self.active_axons = np.arange(self.N)
        self._initialize_patterns()
        self._initialize_geometry()
        self._initialize_hh_model()

    def _initialize_patterns(self):

        """Инициализация паттернов стимуляции аксонов."""
        reg_pct, irreg_pct, semi_pct = 64, 18, 18
        num_reg = int(round(reg_pct / 100 * self.N))
        num_irreg = int(round(irreg_pct / 100 * self.N))
        num_semi = int(round(semi_pct / 100 * self.N))

        shuffled_axons = np.random.permutation(self.active_axons)
        self.regular_axons = shuffled_axons[:num_reg]
        self.irregular_axons = shuffled_axons[num_reg:num_reg + num_irreg]
        self.semiregular_axons = shuffled_axons[num_reg + num_irreg:]

        self.stim_times = np.empty(self.N, dtype=object)
        self.spike_patterns = np.empty(self.N, dtype=object)

        for axon in self.active_axons:
            pattern = ('Regular' if axon in self.regular_axons else
                       'Irregular' if axon in self.irregular_axons else
                       'Semiregular')

            self.spike_patterns[axon] = pattern

            if pattern == "Regular":
                k_reg = 500
                theta_reg = self.mean_ISI / k_reg
                num_spikes = round(self.max_time / self.mean_ISI)
                ISI = gamma.rvs(k_reg, scale=theta_reg, size=num_spikes)

                ton = np.random.uniform(0, 0, 1).ravel()  # Начальная задержка
                self.stim_times[axon] = np.concatenate((ton, ton + np.cumsum(ISI[:-1])))

            elif pattern == "Irregular":
                k_reg = 1
                theta_reg = self.mean_ISI / k_reg
                num_spikes = round(self.max_time / self.mean_ISI)
                ISI = gamma.rvs(k_reg, scale=theta_reg, size=num_spikes)

                ton = np.random.uniform(0, 0, 1).ravel()  # Начальная задержка
                self.stim_times[axon] = np.concatenate((ton, ton + np.cumsum(ISI[:-1])))

            elif pattern == "Semiregular":
                k_reg = 500
                theta_reg = self.mean_ISI / k_reg
                num_spikes = round(self.max_time / self.mean_ISI)
                ISI_full = gamma.rvs(k_reg, scale=theta_reg, size=num_spikes)

                ton = np.random.uniform(0, 0, 1).ravel()  # Начальная задержка
                self.stim_times[axon] = np.concatenate((ton, ton + np.cumsum(ISI[:-1])))

                current_spike_times = ton + np.cumsum(ISI_full)
                num_to_remove = int(round(0.3 * len(current_spike_times)))
                indices_to_remove = np.random.choice(
                    len(current_spike_times),
                    size=num_to_remove,
                    replace=False
                )
                current_spike_times = np.delete(current_spike_times, indices_to_remove)
                self.stim_times[axon] = current_spike_times


    def _initialize_geometry(self):
        """Настройка геометрии аксонов."""
        self.r = np.ones(self.N)  # Гомогенные диаметры
        self.g_ratio = 0.6
        self.tau = 0.47
        self.taun = 0.03
        self.rm = 130

        self.lambc = 1.93 * self.r * np.sqrt(-np.log(self.g_ratio))
        self.lambn = 0.055 * np.sqrt(self.r)
        self.rho = 0.6
        self.sigrat = 1 / 3

        self.dn = np.round(0.2 * self.r / self.dx).astype(int)
        self.sn = 5
        self.nn = np.ceil((self.X - 2 * self.sn) / self.dx / self.dn).astype(int)

        self.mask = np.zeros((self.N, len(self.x)))
        self.idn = np.empty(self.N, dtype=object)
        self.track = np.empty(self.N, dtype=object)

        for o in range(self.N):
            indices = (self.sn / self.dx + np.arange(self.nn[o]) * self.dn[o]).astype(int)
            self.mask[o, indices] = 1
            self.mask[o, -int(self.sn / self.dx)] = 1
            self.idn[o] = indices
            self.track[o] = np.zeros(len(indices))

        self.trackn = np.zeros(self.N)
        self.Vtrack = 40

        deltaL = 0.002 / self.dx
        self.lamb2 = ((1 - deltaL) / self.lambc ** 2 + deltaL / self.lambn ** 2) ** (-0.5)
        self.tau1 = self.tau
        self.tau2 = self.lamb2 ** 2 * ((1 - deltaL) * self.tau / self.lambc ** 2 + deltaL * self.taun / self.lambn ** 2)

        self.V = np.zeros((self.N, len(self.x)))
        self.I = np.zeros((self.N, len(self.x)))
        self.tau = self.tau1 * np.ones((self.N, len(self.x)))
        self.lamb = np.outer(self.lambc, np.ones(len(self.x)))

        for o in range(self.N):
            self.tau[o, self.mask[o, :] == 1] = self.tau2[o]
            self.lamb[o, self.mask[o, :] == 1] = self.lamb2[o]

        r2 = self.r ** 2  # Для использования в дальнейшем
        sum_r2 = np.sum(r2)  # Общая сумма r^2
        self.correction = (r2[:, np.newaxis] / sum_r2) / (
                1 + self.sigrat * ((1 - self.rho) / (self.g_ratio ** 2 * self.rho)))  # Коррекция

    def _initialize_hh_model(self):
        """Инициализация параметров модели Ходжкина-Хаксли."""
        max_nodes = int(np.max(np.sum(self.mask, axis=1)) + 1)
        self.Vn = np.zeros((self.N, max_nodes))
        self.m = 0.0529 * np.ones_like(self.Vn)
        self.h = 0.5961 * np.ones_like(self.Vn)
        self.n = 0.3177 * np.ones_like(self.Vn)
        self.Inode = np.zeros_like(self.Vn)

        self.gNa = 4800
        self.eNa = 115
        self.gK = 720
        self.eK = -12
        self.gL = 30


    def save_spike_data(self, filename, spike_times_start, spike_times_middle, spike_times_end, spike_patterns):
        """
        Сохраняет данные симуляции в HDF5-файл.
        """
        vlen_float = h5py.special_dtype(vlen=np.float64)
        vlen_str = h5py.string_dtype(encoding='utf-8')

        total_steps = int(self.max_time / self.dt)
        total_time = total_steps * self.dt
        t_array = np.arange(0, total_time, self.dt)

        with h5py.File(filename, 'w') as file:
            # --- 1. Глобальные атрибуты и параметры симуляции ---
            file.attrs['file_creation_date'] = np.string_(time.strftime('%Y-%m-%d %H:%M:%S'))

            # --- 2. Группа для основных параметров симуляции ---
            sim_params = file.create_group("simulation_parameters")
            sim_params.attrs['N_axons'] = self.N
            sim_params.attrs['axon_length_X'] = self.X
            sim_params.attrs['spatial_step_dx'] = self.dx
            sim_params.attrs['max_simulation_time'] = self.max_time
            sim_params.attrs['time_step_dt'] = self.dt
            sim_params.attrs['mean_ISI_setting'] = self.mean_ISI
            sim_params.attrs['overall_stim_pattern_type_setting'] = self.stim_pattern_type
            sim_params.attrs['random_seed_used'] = self.seed
            sim_params.create_dataset('t_array', data=t_array)

            # --- 3. Группа для параметров геометрии и модели Ходжкина-Хаксли ---
            model_props_grp = file.create_group("model_properties")
            model_props_grp.attrs['g_ratio_setting'] = self.g_ratio

            # --- 4. Группа для данных о спайках ---
            grp = file.create_group("spikes")
            grp.create_dataset('patterns', data=np.array(spike_patterns, dtype=vlen_str))

            dset_start = grp.create_dataset('start', (len(spike_times_start),), dtype=vlen_float)
            dset_middle = grp.create_dataset('middle', (len(spike_times_middle),), dtype=vlen_float)
            dset_end = grp.create_dataset('end', (len(spike_times_end),), dtype=vlen_float)

            for i in range(len(spike_times_start)):
                dset_start[i] = spike_times_start[i][~np.isnan(spike_times_start[i])]
                dset_middle[i] = spike_times_middle[i][~np.isnan(spike_times_middle[i])]
                dset_end[i] = spike_times_end[i][~np.isnan(spike_times_end[i])]

        print(f'Данные сохранены в {filename}')


    def run_simulation(self):
        """Запуск моделирования (заглушка, требует реализации)."""
        print("Запуск симуляции...")

        #spike_times_start = np.empty(self.N, dtype=object)  # Для начала аксона
        #spike_times_middle = np.empty(self.N, dtype=object)  # Для середины аксона
        #spike_times_end = np.empty(self.N, dtype=object)  # Для конца аксона
        max_spikes = 10000
        spike_times_start = np.zeros((self.N, max_spikes))  # Массив времён спайков для начала аксона
        spike_times_middle = np.zeros((self.N, max_spikes))  # Массив времён спайков для среднего узла
        spike_times_end = np.zeros((self.N, max_spikes))  # Массив времён спайков для конца аксона

        # Флаги состояния для каждого узла аксона
        is_spiking_start = np.zeros(self.N, dtype=bool)  # Для начала аксона
        is_spiking_middle = np.zeros(self.N, dtype=bool)  # Для середины аксона
        is_spiking_end = np.zeros(self.N, dtype=bool)  # Для конца аксона

        self.next_stim_idx = np.ones(self.N, dtype=int)
        self.spikes_reached_end = np.zeros(len(self.active_axons), dtype=bool)

        self.stimEndTimes = np.zeros(self.N)

        # Массивы для хранения времени спайков
        spike_times_start = np.full((self.N, max_spikes), np.nan)  # NaN для пустых мест
        spike_times_middle = np.full((self.N, max_spikes), np.nan)
        spike_times_end = np.full((self.N, max_spikes), np.nan)

        # Флаги спайков
        is_spiking_start = np.zeros(self.N, dtype=np.bool_)
        is_spiking_middle = np.zeros(self.N, dtype=np.bool_)
        is_spiking_end = np.zeros(self.N, dtype=np.bool_)

        # Счётчик спайков
        spike_count_start = np.zeros(self.N, dtype=np.int32)
        spike_count_middle = np.zeros(self.N, dtype=np.int32)
        spike_count_end = np.zeros(self.N, dtype=np.int32)

        timesteps = int(self.max_time / self.dt)

        last_print_time_model = 0.0
        real_start_time = time.time()

        #for t in range(timesteps):

        for t in tqdm(range(timesteps), desc="Симуляция", unit=" шаг", total=timesteps, smoothing=0.1):
            current_time = t * self.dt
            '''
            current_time = t * self.dt
            sys.stdout.write(f"\rt текущее: {current_time} мс из {self.max_time} мс")
            sys.stdout.flush()

            if current_time - last_print_time_model >= 1:
                real_elapsed = time.time() - real_start_time
                print(f"Прошло {current_time:.1f} мс моделируемого времени — {real_elapsed:.1f} сек реально")
                last_print_time_model = current_time
            '''

            #print("t текущее", t * self.dt, "из", self.max_time)

            t = t + 1

            Istim = np.zeros(self.N)

            for idx in self.active_axons:
                # print(idx)
                axon = self.active_axons[idx]
                # print(axon)
                if t * self.dt < self.stimEndTimes[axon]:
                    Istim[axon] = self.r[axon] * 5e3  # Стимуляция продолжается
                elif self.next_stim_idx[axon] <= len(self.stim_times[axon]) and self.stim_times[axon][
                    self.next_stim_idx[axon] - 1] <= t * self.dt:
                    # print("works?")
                    Istim[axon] = self.r[axon] * 5e3  # Новый стимул
                    self.stimEndTimes[axon] = t * self.dt + 2.5  # Окончание стимула
                    self.next_stim_idx[axon] = self.next_stim_idx[axon] + 1

            for o in range(self.N):
                self.Vn[o, :self.nn[o] + 1] = self.V[o, self.mask[o, :] == 1]

            self.m, self.h, self.n = update_gate(self.Vn, self.dt, self.m, self.h, self.n)


            # --------------------------------------------------------------------------------------------------------------
            # --------------------------------------------------------------------------------------------------------------
            # --------------------------------------------------------------------------------------------------------------


            self.Inode = compute_Inode(self.r, self.gNa, self.m, self.h, self.eNa, self.Vn, self.gK, self.n, self.eK, self.gL, Istim)

            self.I = update_I(self.mask, self.Inode, self.nn, self.I, Istim)

            self.Vxx = compute_Vxx(self.V, self.dx)

            sum_factor_Vxx_row = np.sum(self.correction * self.Vxx, axis=0) # Результат (X_len,)

            self.dV = compute_dV(self.N, self.V, self.I, sum_factor_Vxx_row, self.lamb, self.Vxx, self.dt, self.tau)


            # --------------------------------------------------------------------------------------------------------------
            # --------------------------------------------------------------------------------------------------------------
            # --------------------------------------------------------------------------------------------------------------
            self.V += self.dV

            for o in prange(self.N):
                nodeStart = self.idn[o][1]  # Второй узел
                nodeMiddle = self.idn[o][self.nn[o] // 2]  # Средний узел
                nodeEnd = self.idn[o][-1]  # Последний узел

                is_spiking_start, spike_times_start, spike_count_start = spike_times_get(
                    self.V, o, nodeStart, self.Vtrack, is_spiking_start, spike_times_start, t, self.dt, max_spikes, spike_count_start
                )
                is_spiking_middle, spike_times_middle, spike_count_middle = spike_times_get(
                    self.V, o, nodeMiddle, self.Vtrack, is_spiking_middle, spike_times_middle, t, self.dt, max_spikes,
                    spike_count_middle
                )
                is_spiking_end, spike_times_end, spike_count_end = spike_times_get(
                    self.V, o, nodeEnd, self.Vtrack, is_spiking_end, spike_times_end, t, self.dt, max_spikes, spike_count_end
                )

            # Обновление последнего возбужденного узла
            for p in range(self.N):
                if self.trackn[p] < self.nn[p]:
                    spiking_nodes = np.where(self.V[p, self.idn[p][int(self.trackn[p]) + 1:]] > self.Vtrack)[0]
                    # print(type(trackn[p]), trackn[p])

                    if len(spiking_nodes) > 0:
                        self.trackn[p] += spiking_nodes[-1] + 1  # Обновляем индекс

            # Проверка, дошел ли спайк до конца аксона
            for idx, axon in enumerate(self.active_axons):
                if self.trackn[axon] >= self.nn[axon]:
                    self.spikes_reached_end[idx] = True

        self.save_spike_data(
            filename='SpikeData_300.h5',
            spike_times_start=spike_times_start,
            spike_times_middle=spike_times_middle,
            spike_times_end=spike_times_end,
            spike_patterns=self.spike_patterns
        )

        return spike_times_start, spike_times_middle, spike_times_end, self.spike_patterns



        pass



# Фиксируем момент старта
start_time = time.strftime('%d-%m-%Y %H:%M:%S')
print(f'Начало расчета: {start_time}')

# Засекаем время выполнения
start_tic = time.time()

model = SPM_Model()
profiler = cProfile.Profile()
profiler.enable()
model.run_simulation()
profiler.disable()
stats = pstats.Stats(profiler).sort_stats('cumulative')
stats.print_stats()

# Завершаем измерение времени
elapsedTime = time.time() - start_tic  # время с начала расчета
endTime = time.strftime('%d-%m-%Y %H:%M:%S')

print(f'Конец расчета: {endTime}')
print(f'Длительность расчета: {elapsedTime:.2f} секунд')