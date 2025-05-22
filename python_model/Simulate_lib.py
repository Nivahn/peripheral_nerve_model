import numpy as np
from numba import njit, prange, cuda
import time


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
def compute_dV(N, V, I, sum_terms_1D_arg, lamb, Vxx, dt, tau): # sum_terms_1D_arg - это 1D массив формы (X_len,)
    dV = np.empty_like(V)
    for i in prange(N): # Цикл по аксонам
        dV[i, :] = dt * (
            lamb[i, :] ** 2 * Vxx[i, :]
            - lamb[i, :] ** 2 * sum_terms_1D_arg  # <--- КЛЮЧЕВОЕ ИЗМЕНЕНИЕ: используем 1D массив целиком
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

@njit(parallel=True)
def update_gate(Vn, dt, m, h, n):
    m += dt * (((2.5 - 0.1 * Vn) / (np.exp(2.5 - 0.1 * Vn) - 1)) * (1 - m) - (4 * np.exp(-Vn / 18)) * m)
    h += dt * ((0.07 * np.exp(-Vn / 20)) * (1 - h) - (1 / (np.exp(3.0 - 0.1 * Vn) + 1)) * h)
    n += dt * (((0.1 - 0.01 * Vn) / (np.exp(1 - 0.1 * Vn) - 1)) * (1 - n) - (0.125 * np.exp(-Vn / 80)) * n)
    return m, h, n

