import numpy as np
from numba import njit, prange, cuda

import time



@cuda.jit
def compute_Vxx_gpu(V, dx, Vxx):
    x, y = cuda.grid(2)  # Получаем индексы в сетке

    if 0 < y < V.shape[1] - 1:  # Избегаем краевых эффектов
        Vxx[x, y] = (V[x, y - 1] + V[x, y + 1] - 2 * V[x, y]) / (dx ** 2)

@njit(parallel=True)
def compute_Inode(r, gNa, m, h, eNa, Vn, gK, n, eK, gL, Istim):
    # Создаем выходной массив Inode
    Inode = np.zeros((r.shape[0], m.shape[1]))  # Инициализация с правильными размерами

    # Параллельное вычисление для каждой строки
    for i in prange(r.shape[0]):  # Параллельная обработка по строкам (по оси 0)
        Inode[i, :] = r[i] * (
                gNa * m[i, :] ** 3 * h[i, :] * (eNa - Vn[i, :]) + gK * n[i, :] ** 4 * (eK - Vn[i, :])
        ) / gL

    # Применение стимула для первого узла
    Inode[:, 0] = Istim  # Стимул прикладывается к первому узлу Ранвье

    return Inode


@njit(parallel=True)
def update_I(mask, Inode, nn, I, Istim):
    # Преобразуем маску в логический массив (True/False) без использования astype
    mask_1d = mask != 0  # Сравниваем с нулем для получения булевского массива

    # Используем prange для параллельной обработки
    for i in prange(mask.shape[0]):
        # Применяем значение для всех позиций на основе маски
        I[i, mask_1d[i, :]] = Inode[i, :nn.max() + 1]

    # Стимул прикладывается к первому узлу
    I[:, 0] = Istim  # стимул для первого узла Ранвье

    return I


@njit(parallel=True)
def compute_Vxx(V, dx):
    # Создаём массив для Ve с добавлением дополнительных столбцов
    Ve = np.empty((V.shape[0], V.shape[1] + 2), dtype=np.float64)  # Включаем 2 столбца для дополнения

    # Заполняем массив Ve: добавляем столбцы слева и справа
    Ve[:, 1:-1] = V  # Центральная часть
    for i in prange(V.shape[0]):
        Ve[i, 0] = V[i, 0]  # Первый столбец
        Ve[i, -1] = V[i, -1]  # Последний столбец

    # Вычисляем второй производный (Лапласиан)
    Vxx = np.empty((V.shape[0], V.shape[1]), dtype=np.float64)  # Размерность Vxx = (V.shape[0], V.shape[1])

    # Вычисляем Лапласиан с учетом дополненных столбцов
    for i in prange(V.shape[0]):
        Vxx[i, :] = (Ve[i, 2:] + Ve[i, :-2] - 2 * Ve[i, 1:-1]) / dx ** 2

    return Vxx

@njit(parallel=True)

def compute_summed_correction(correction, Vxx, N):
    summed_correction = np.zeros((N, Vxx.shape[1]))
    sum_correction = np.sum(correction * Vxx, axis=0)[np.newaxis, :]  # Убираем keepdims и добавляем размерность вручную

    # Параллельный цикл для повторения суммы по оси N
    for i in prange(N):
        summed_correction[i] = sum_correction

    return summed_correction

@njit(parallel=True)
def compute_dV(N, V, I, correction, lamb, Vxx, dt, tau):

    #summed_correction = compute_summed_correction(correction, Vxx, N)
    dV = np.empty_like(V)

    for i in prange(V.shape[0]):

        dV[i, :] = dt * (
            lamb[i, :] ** 2 * Vxx[i, :]
            - lamb[i, :] ** 2 * correction[i]
            - V[i, :] + I[i, :]
        ) / tau[i, :]

    return dV







@njit
def create_spike_times_data(N):

    spike_times_start = np.empty(N, dtype=object)  # Для начала аксона
    spike_times_middle = np.empty(N, dtype=object)  # Для середины аксона
    spike_times_end = np.empty(N, dtype=object)  # Для конца аксона

    # Флаги состояния для каждого узла аксона
    is_spiking_start = np.zeros(N, dtype=bool)  # Для начала аксона
    is_spiking_middle = np.zeros(N, dtype=bool)  # Для середины аксона
    is_spiking_end = np.zeros(N, dtype=bool)  # Для конца аксона

    return spike_times_start, spike_times_middle, spike_times_end, is_spiking_start, is_spiking_middle,is_spiking_end


@njit
def spike_times_get(V, o, node, Vtrack, is_spiking, spike_times, t, dt, max_spikes, spike_count):
    """Фиксирует время спайка, если напряжение превышает порог."""
    if V[o, node] > Vtrack:
        if not is_spiking[o]:  # Спайк еще не зафиксирован
            if spike_count[o] < max_spikes:  # Проверка на переполнение
                spike_times[o, spike_count[o]] = t * dt
                spike_count[o] += 1  # Увеличиваем счётчик спайков
            is_spiking[o] = True
    else:
        is_spiking[o] = False

    return is_spiking, spike_times, spike_count

@njit(parallel=True)
def spike_times_fix(N, V, idn, nn, Vtrack, dt, spike_times_start, spike_times_middle, spike_times_end,
                    is_spiking_start, is_spiking_middle, is_spiking_end, spike_count_start, spike_count_middle, spike_count_end, t, max_spikes):
    """Фиксирует спайки во всех аксонах."""
    for o in prange(N):
        nodeStart = idn[o][1]  # Второй узел
        nodeMiddle = idn[o][nn[o] // 2]  # Средний узел
        nodeEnd = idn[o][-1]  # Последний узел

        is_spiking_start, spike_times_start, spike_count_start = spike_times_get(
            V, o, nodeStart, Vtrack, is_spiking_start, spike_times_start, t, dt, max_spikes, spike_count_start
        )
        is_spiking_middle, spike_times_middle, spike_count_middle = spike_times_get(
            V, o, nodeMiddle, Vtrack, is_spiking_middle, spike_times_middle, t, dt, max_spikes, spike_count_middle
        )
        is_spiking_end, spike_times_end, spike_count_end = spike_times_get(
            V, o, nodeEnd, Vtrack, is_spiking_end, spike_times_end, t, dt, max_spikes, spike_count_end
        )

    return (spike_times_start, spike_times_middle, spike_times_end,
            is_spiking_start, is_spiking_middle, is_spiking_end,
            spike_count_start, spike_count_middle, spike_count_end)

'''
@njit
def spike_times_get(V, o, node, Vtrack, is_spiking, spike_times, t, dt):
    if V[o, node] > Vtrack:
        if not is_spiking[o]:  # Спайк еще не зафиксирован
            spike_times[o].append(t * dt)
            is_spiking[o] = True
    else:
        is_spiking[o] = False

    return is_spiking[o], spike_times[o]

@njit(parallel = True)
def spike_times_fix(self, t):
    for o in range(self.N):
        # Индексы узлов Ранвье
        # print('idn[o]:', idn[o])
        nodeStart = self.idn[o][1]  # Второй узел
        nodeMiddle = self.idn[o][int(self.nn[o] / 2)]  # Средний узел
        nodeEnd = self.idn[o][-1]  # Последний узел

        # Фиксация спайков
        self.is_spiking_start[o], self.spike_times_start[o] = spike_times_get(self.V, o, nodeStart, self.Vtrack, self.is_spiking_start[o], self.spike_times_start[o], t, self.dt)
        self.is_spiking_middle[o], self.spike_times_middle[o] = spike_times_get(self.V, o, nodeMiddle, self.Vtrack, self.is_spiking_middle[o], self.spike_times_middle[o], t, self.dt)
        self.is_spiking_end[o], self.spike_times_end[o] = spike_times_get(self.V, o, nodeEnd, self.Vtrack, self.is_spiking_end[o], self.spike_times_end[o], t, self.dt)
'''

@njit(parallel=True)
def update_gate(Vn, dt, m, h, n):
    m += dt * (((2.5 - 0.1 * Vn) / (np.exp(2.5 - 0.1 * Vn) - 1)) * (1 - m) - (4 * np.exp(-Vn / 18)) * m)
    h += dt * ((0.07 * np.exp(-Vn / 20)) * (1 - h) - (1 / (np.exp(3.0 - 0.1 * Vn) + 1)) * h)
    n += dt * (((0.1 - 0.01 * Vn) / (np.exp(1 - 0.1 * Vn) - 1)) * (1 - n) - (0.125 * np.exp(-Vn / 80)) * n)
    return m, h, n

'''
@njit(parallel=True)
def simulate(maxTime, dt, N, active_axons, stimEndTimes, r , next_stim_idx,
            stim_times, Vn, mask, nn, V, gNa, eNa, gK, eK, gL, lamb, sigrat, rho, dx, I, tau, idn,g,
             Vtrack, trackn, m,h,n, spikes_reached_end, spike_patterns, Inode,
             spike_times_start, spike_times_middle, spike_times_end,
             is_spiking_start, is_spiking_middle, is_spiking_end):
    
   '''

#@njit(parallel=True)
def simulate(maxTime, dt, N, active_axons, stimEndTimes, r , next_stim_idx,
            stim_times, Vn, mask, nn, V, gNa, eNa, gK, eK, gL, lamb, sigrat, rho, dx, I, tau, idn,g,
             Vtrack, trackn, m,h,n, spikes_reached_end, spike_patterns, Inode):


    #spike_times_start = np.empty(N, dtype=np.float64)  # Для начала аксона
    #spike_times_middle = np.empty(N, dtype=np.float64)  # Для середины аксона
    #spike_times_end = np.empty(N, dtype=np.float64)  # Для конца аксона

    spike_times_start = np.empty(N, dtype=object)  # Для начала аксона
    spike_times_middle = np.empty(N, dtype=object)  # Для середины аксона
    spike_times_end = np.empty(N, dtype=object)  # Для конца аксона

    # Флаги состояния для каждого узла аксона
    is_spiking_start = np.zeros(N, dtype=bool)  # Для начала аксона
    is_spiking_middle = np.zeros(N, dtype=bool)  # Для середины аксона
    is_spiking_end = np.zeros(N, dtype=bool)  # Для конца аксона

    timesteps = int(maxTime / dt)

    print("Работает?")

    for t in range(timesteps):
        print("t текущее", t * dt, "из", maxTime)

        # Засекаем время начала итерации по времени
        time_step_start = time.time()

        # Засекаем время начала цикла по активным аксонам
        active_axons_start = time.time()

        t = t + 1
        print("t текущее", t * dt, "из", maxTime)

        Istim = np.zeros(N)

        for idx in active_axons:
            # print(idx)
            axon = active_axons[idx]
            # print(axon)
            if t * dt < stimEndTimes[axon]:
                Istim[axon] = r[axon] * 5e3  # Стимуляция продолжается
            elif next_stim_idx[axon] <= len(stim_times[axon]) and stim_times[axon][next_stim_idx[axon] - 1] <= t * dt:
                # print("works?")
                Istim[axon] = r[axon] * 5e3  # Новый стимул
                stimEndTimes[axon] = t * dt + 2.5  # Окончание стимула
                next_stim_idx[axon] = next_stim_idx[axon] + 1

        # Засекаем время окончания цикла по активным аксонам
        active_axons_end = time.time()
        print(f"Время на цикл по активным аксонам: {active_axons_end - active_axons_start:.4f} секунд")

        time_Vn_start = time.time()
        for o in range(N):
            Vn[o, :nn[o] + 1] = V[o, mask[o, :] == 1]
        time_Vn_end = time.time()

        print(f"Vn time: {time_Vn_end - time_Vn_start:.4f} секунд")

        time_for_HH_stuff_start = time.time()
        # HH stuff

        m, h, n = update_gate(Vn, dt, m, h, n)

        time_for_HH_stuff_end = time.time()
        print(f"Время на HH: {time_for_HH_stuff_end - time_for_HH_stuff_start:.4f} секунд")
        # print('m:', m, 'h:', h, 'n:', n)

        time_for_some_stuff_start = time.time()
        # --------------------------------------------------------------------------------------------------------------
        # --------------------------------------------------------------------------------------------------------------
        # --------------------------------------------------------------------------------------------------------------


        '''
        Inode = compute_Inode(r, gNa, m, h, eNa, Vn, gK, n, eK, gL)

        # Обновление I с использованием маски
        I = update_I(mask, Inode, nn, I, Istim)

        # Вычисление Лапласиана
        Vxx = compute_Vxx(V, dx)

        r2 = r ** 2  # Для использования в дальнейшем
        sum_r2 = np.sum(r2)  # Общая сумма r^2
        correction = (r2[:, np.newaxis] / sum_r2) / (1 + sigrat * ((1 - rho) / (g ** 2 * rho)))  # Коррекция
        # Вычисление dV
        dV = compute_dV(lamb, Vxx, correction, V, I, tau, dt)
        '''


        start_time = time.time()
        # Предварительные вычисления для улучшения производительности
        r2 = r ** 2  # Для использования в дальнейшем
        sum_r2 = np.sum(r2)  # Общая сумма r^2

        print(f"Время для r ** 2: {time.time() - start_time:.4f} сек.")

        start_time = time.time()
        correction = (r2[:, np.newaxis] / sum_r2) / (1 + sigrat * ((1 - rho) / (g ** 2 * rho)))  # Коррекция
        print(f"Время для вычисления коррекции: {time.time() - start_time:.4f} сек.")

        start_time = time.time()

        # Прямое вычисление Inode без использования np.tile
        #print(f"Размер m: {m.shape}")
        #print(f"Размер h: {h.shape}")
        #print(f"Размер Vn: {Vn.shape}")
        #print(f"Размер n: {n.shape}")
        Inode = compute_Inode(r, gNa, m, h, eNa, Vn, gK, n, eK, gL, Istim)


        #Inode = r[:, np.newaxis] * (gNa * m ** 3 * h * (eNa - Vn) + gK * n ** 4 * (eK - Vn)) / gL
        #Inode[:, 0] = Istim  # Стимул прикладывается к первому узлу Ранвье

        print(f"Время для вычисления Inode: {time.time() - start_time:.4f} сек.")


        # Используем булевую маску для быстрого обновления I
        #mask_1d = mask.astype(bool)  # Преобразуем маску в 1D булевую
        #I[mask_1d] = Inode[:, :nn.max() + 1].flatten()  # Применяем значение для всех позиций
        start_time = time.time()
        I = update_I(mask, Inode, nn, I, Istim)
        print(f"Время для вычисления I: {time.time() - start_time:.4f} сек.")

        start_time = time.time()
        # Векторизация для вычисления Лапласиана
        #Ve = np.hstack([V[:, [0]], V, V[:, [-1]]])
        #Vxx = (Ve[:, 2:] + Ve[:, :-2] - 2 * Ve[:, 1:-1]) / dx ** 2  # Лапласиан
        Vxx = compute_Vxx(V, dx)
        print(f"Время для вычисления Лапласиана Vxx: {time.time() - start_time:.4f} сек.")

        start_time = time.time()

        '''
        # Векторизация для вычисления dV
        dV = dt * (
                lamb ** 2 * Vxx
                - lamb ** 2 * np.sum(correction * Vxx, axis=0, keepdims=True)  # Суммируем коррекцию по осям
                - V + I
        ) / tau
        '''

        correction = np.tile(np.sum(correction * Vxx, axis=0, keepdims=True), (N, 1))

        dV = compute_dV(N, V, I, correction, lamb, Vxx, dt, tau)

        print(f"Время для вычисления dV: {time.time() - start_time:.4f} сек.")

        '''
        Inode = np.tile(r[:, np.newaxis], (1, Inode.shape[1])) * (
                gNa * m ** 3 * h * (eNa - Vn) + gK * n ** 4 * (eK - Vn)
        ) / gL

        Inode[:, 0] = Istim  # стимул прикладывается к первому узлу Ранвье

        for o in range(N):
            I[o, mask[o, :] == 1] = Inode[o, :nn[o] + 1]

        Ve = np.hstack([V[:, [0]], V, V[:, [-1]]])

        # Вторая производная (Лапласиан)
        Vxx = (Ve[:, 2:] + Ve[:, :-2] - 2 * Ve[:, 1:-1]) / dx ** 2

        # Коэффициент коррекции
        correction = (r[:, np.newaxis] ** 2 / np.sum(r ** 2)) / (1 + sigrat * ((1 - rho) / (g ** 2 * rho)))

        # Вычисление dV
        dV = dt * (
                lamb ** 2 * Vxx
                - lamb ** 2 * np.tile(np.sum(correction * Vxx, axis=0, keepdims=True), (N, 1))
                - V + I
        ) / tau

        # Обновление потенциала
        V += dV
        time_for_some_stuff_end = time.time()
        print(f"Время на всячину?: {time_for_some_stuff_end - time_for_some_stuff_start:.4f} секунд")
        '''
        # --------------------------------------------------------------------------------------------------------------
        # --------------------------------------------------------------------------------------------------------------
        # --------------------------------------------------------------------------------------------------------------
        V += dV
        time_for_some_stuff_end = time.time()
        print(f"Время на всячину?: {time_for_some_stuff_end - time_for_some_stuff_start:.4f} секунд")

        # Засекаем время начала обработки спайков для каждого аксона
        spike_processing_start = time.time()


        for o in range(N):
            # Индексы узлов Ранвье
            # print('idn[o]:', idn[o])
            nodeStart = idn[o][1]  # Второй узел
            nodeMiddle = idn[o][int(nn[o] / 2)]  # Средний узел
            nodeEnd = idn[o][-1]  # Последний узел

            # Фиксация спайков
            if V[o, nodeStart] > Vtrack:
                if not is_spiking_start[o]:  # Спайк еще не зафиксирован
                    spike_times_start[o].append(t * dt)
                    is_spiking_start[o] = True
            else:
                is_spiking_start[o] = False

            if V[o, nodeMiddle] > Vtrack:
                if not is_spiking_middle[o]:
                    spike_times_middle[o].append(t * dt)
                    is_spiking_middle[o] = True
            else:
                is_spiking_middle[o] = False

            if V[o, nodeEnd] > Vtrack:
                if not is_spiking_end[o]:
                    spike_times_end[o].append(t * dt)
                    is_spiking_end[o] = True
            else:
                is_spiking_end[o] = False

        # Обновление последнего возбужденного узла
        for p in range(N):
            if trackn[p] < nn[p]:
                spiking_nodes = np.where(V[p, idn[p][int(trackn[p]) + 1:]] > Vtrack)[0]
                # print(type(trackn[p]), trackn[p])

                if len(spiking_nodes) > 0:
                    trackn[p] += spiking_nodes[-1] + 1  # Обновляем индекс

        # Проверка, дошел ли спайк до конца аксона
        for idx, axon in enumerate(active_axons):
            if trackn[axon] >= nn[axon]:
                spikes_reached_end[idx] = True

        # Засекаем время окончания обработки спайков
        spike_processing_end = time.time()
        #print(f"Время на обработку спайков: {spike_processing_end - spike_processing_start:.4f} секунд")

        # Засекаем время конца шага
        time_step_end = time.time()
        #print(f"Время на итерацию по времени (t = {t * dt}): {time_step_end - time_step_start:.4f} секунд")

    return  spike_times_start, spike_times_middle, spike_times_end, spike_patterns

