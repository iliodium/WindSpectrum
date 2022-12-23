import numpy as np
import scipy.interpolate
import matplotlib.cm as cm
import matplotlib.tri as mtri
import matplotlib.pyplot as plt
from scipy.fft import fft, rfftfreq
from scipy.signal import argrelextrema, welch
from matplotlib.colors import Normalize


class Plot:
    """
    Класс отвечает за отрисовку графиков
       Виды графиков:
           Изополя:
               -Максимальные значения
               -Средние значения
               -Минимальные значения
               -Среднее квадратичное отклонение
           Огибающие:
               -Максимальные значения
               -Минимальные значения
           Коэффициенты обеспеченности:
               -Максимальные значения
               -Минимальные значения
           Спектры
           Нестационарные сигналы
       Нумерация:
       Нумерация графиков нужная для очищения памяти
       num = alpha_model_name_angle_mode
       """

    @staticmethod
    def interpolator(coords, val):
        return scipy.interpolate.RBFInterpolator(coords, val, kernel='cubic')

    @staticmethod
    def discrete_isofield(model_name, alpha, angle, mode, pressure_coefficients, coordinates):
        """Отрисовка дискретных изополей"""
        # Виды изополей
        mods = {
            'max': np.max(pressure_coefficients, axis=0),
            'mean': np.mean(pressure_coefficients, axis=0),
            'min': np.min(pressure_coefficients, axis=0),
            'std': np.std(pressure_coefficients, axis=0),
        }

        pressure_coefficients = mods[mode]
        breadth, depth, height = int(model_name[0]) / 10, int(model_name[1]) / 10, int(model_name[2]) / 10
        count_sensors_on_model = len(pressure_coefficients)
        count_sensors_on_middle = int(model_name[0]) * 5
        count_sensors_on_side = int(model_name[1]) * 5
        count_row = count_sensors_on_model // (2 * (count_sensors_on_middle + count_sensors_on_side))
        pressure_coefficients = np.reshape(pressure_coefficients, (count_row, -1))
        pressure_coefficients = np.split(pressure_coefficients,
                                         [count_sensors_on_middle,
                                          count_sensors_on_middle + count_sensors_on_side,
                                          2 * count_sensors_on_middle + count_sensors_on_side,
                                          2 * (count_sensors_on_middle + count_sensors_on_side)
                                          ], axis=1)
        del pressure_coefficients[4]

        x, z = np.array(coordinates)
        z = np.array(z[::2 * (count_sensors_on_middle + count_sensors_on_side)])[::-1]

        num_fig = f'{alpha}_{model_name}_{angle}_{mode}'
        num_fig_b = ''
        b = num_fig.encode()
        num_fig_b = int(num_fig_b.join([str(b[i]) for i in range(len(b))]))
        fig, ax = plt.subplots(1, 4, dpi=200, num=num_fig_b, clear=True)

        cmap = cm.get_cmap(name="jet")
        min_v = np.min(pressure_coefficients)
        max_v = np.max(pressure_coefficients)
        normalizer = Normalize(min_v, max_v)
        im = cm.ScalarMappable(norm=normalizer, cmap=cmap)
        for i in range(4):
            data_old = pressure_coefficients[i]
            ax[i].pcolormesh(np.flip(data_old, axis=0), cmap=cmap)
            ax[i].set_yticks(np.arange(0, count_row + 1, 2.5))
            ax[i].set_yticklabels(labels=np.arange(0, height + 0.01, 0.05))
            if i in [0, 2]:
                ax[i].set_xticks([i for i in range(0, count_sensors_on_middle + 1, 5)])
                ax[i].set_xticklabels(labels=list(map(str, np.arange(0, breadth + 0.1, 0.1))))
            else:
                ax[i].set_xticks([i for i in range(0, count_sensors_on_side + 1, 5)])
                ax[i].set_xticklabels(labels=list(map(str, np.arange(0, depth + 0.1, 0.1))))
            x, y = np.meshgrid(np.arange(0.5, count_sensors_on_middle + 0.5, 1), z * count_row / height)
            ax[i].plot(x, y, '.k')

        if breadth == depth == height:
            [ax[i].set_aspect('equal') for i in range(4)]
        ticks = np.arange(np.round(min_v, 1), np.round(max_v, 1) + 0.1, 0.1)
        fig.colorbar(im, ax=ax, location='bottom', cmap=cmap, ticks=ticks).ax.tick_params(
            labelsize=7)

        return fig

    @staticmethod
    def integral_isofield(model_name, alpha, angle, mode, pressure_coefficients, coordinates):
        """Отрисовка интегральных изополей"""
        # Виды изополей
        mods = {
            'max': np.max(pressure_coefficients, axis=0),
            'mean': np.mean(pressure_coefficients, axis=0),
            'min': np.min(pressure_coefficients, axis=0),
            'std': np.std(pressure_coefficients, axis=0),
        }
        # Номера графиков
        pressure_coefficients = mods[mode]
        breadth, depth, height = int(model_name[0]) / 10, int(model_name[1]) / 10, int(model_name[2]) / 10
        count_sensors_on_model = len(pressure_coefficients)
        count_sensors_on_middle = int(model_name[0]) * 5
        count_sensors_on_side = int(model_name[1]) * 5
        count_row = count_sensors_on_model // (2 * (count_sensors_on_middle + count_sensors_on_side))

        x, z = np.array(coordinates)

        # Шаги для изополей
        steps = {
            'max': 0.2,
            'mean': 0.2 if alpha == 6 else 0.1,
            'min': 0.2,
            'std': 0.05,
        }
        #  Шаг для изополей и контурных линий

        min_v = np.min(pressure_coefficients)
        max_v = np.max(pressure_coefficients)
        levels = np.arange(np.round(min_v, 1), np.round(max_v, 1) + 0.1, steps[mode]).round(2)

        pressure_coefficients = np.reshape(pressure_coefficients, (count_row, -1))
        pressure_coefficients = np.split(pressure_coefficients, [count_sensors_on_middle,
                                                                 count_sensors_on_middle + count_sensors_on_side,
                                                                 2 * count_sensors_on_middle + count_sensors_on_side,
                                                                 2 * (count_sensors_on_middle + count_sensors_on_side)
                                                                 ], axis=1)
        x = np.reshape(x, (count_row, -1))
        x = np.split(x, [count_sensors_on_middle,
                         count_sensors_on_middle + count_sensors_on_side,
                         2 * count_sensors_on_middle + count_sensors_on_side,
                         2 * (count_sensors_on_middle + count_sensors_on_side)
                         ], axis=1)

        z = np.reshape(z, (count_row, -1))
        z = np.split(z, [count_sensors_on_middle,
                         count_sensors_on_middle + count_sensors_on_side,
                         2 * count_sensors_on_middle + count_sensors_on_side,
                         2 * (count_sensors_on_middle + count_sensors_on_side)
                         ], axis=1)

        del pressure_coefficients[4]
        del x[4]
        del z[4]

        # x это тензор со всеми координатами граней по ширине,x1...x4 координаты отдельных граней
        x1, x2, x3, x4 = list(x[0]), list(x[1]), list(x[2]), list(x[3])
        # z это тензор со всеми координатами граней по высоте,z1...z4 координаты отдельных граней
        z1, z2, z3, z4 = list(z[0]), list(z[1]), list(z[2]), list(z[3])

        # Расширение матрицы координат по бокам
        for i in range(len(x1)):
            x1[i] = np.append(np.insert(x1[i], 0, 0), breadth)
            x2[i] = np.append(np.insert(x2[i], 0, breadth), breadth + depth)
            x3[i] = np.append(np.insert(x3[i], 0, breadth + depth), 2 * breadth + depth)
            x4[i] = np.append(np.insert(x4[i], 0, 2 * breadth + depth), 2 * (breadth + depth))

        x1.append(x1[0])
        x2.append(x2[0])
        x3.append(x3[0])
        x4.append(x4[0])

        x1.insert(0, x1[0])
        x2.insert(0, x2[0])
        x3.insert(0, x3[0])
        x4.insert(0, x4[0])

        # Расширение матрицы координат по бокам
        for i in range(len(z1)):
            z1[i] = np.append(np.insert(z1[i], 0, z1[i][0]), z1[i][0])
            z2[i] = np.append(np.insert(z2[i], 0, z2[i][0]), z2[i][0])
            z3[i] = np.append(np.insert(z3[i], 0, z3[i][0]), z3[i][0])
            z4[i] = np.append(np.insert(z4[i], 0, z4[i][0]), z4[i][0])

        z1.append(np.array([0 for _ in range(len(z1[0]))]))
        z2.append(np.array([0 for _ in range(len(z2[0]))]))
        z3.append(np.array([0 for _ in range(len(z3[0]))]))
        z4.append(np.array([0 for _ in range(len(z4[0]))]))

        z1.insert(0, np.array([height for _ in range(len(z1[0]))]))
        z2.insert(0, np.array([height for _ in range(len(z2[0]))]))
        z3.insert(0, np.array([height for _ in range(len(z3[0]))]))
        z4.insert(0, np.array([height for _ in range(len(z4[0]))]))

        # Расширенные координаты для изополей
        z_extended = np.array([np.array(z1), np.array(z2), np.array(z3), np.array(z4)])
        x_extended = np.array([np.array(x1), np.array(x2), np.array(x3), np.array(x4)])

        num_fig = f'{alpha}_{model_name}_{angle}_{mode}'
        num_fig_b = ''
        b = num_fig.encode()
        num_fig_b = int(num_fig_b.join([str(b[i]) for i in range(len(b))]))
        fig, graph = plt.subplots(1, 4, dpi=200, num=num_fig_b, clear=True, figsize=(9, 5))

        cmap = cm.get_cmap(name="jet")
        data_colorbar = None

        for i in range(4):
            # x это координаты по ширине
            x_new = x_extended[i].reshape(1, -1)[0]
            x_old = x[i].reshape(1, -1)[0]
            # z это координаты по высоте
            z_new = z_extended[i].reshape(1, -1)[0]
            z_old = z[i].reshape(1, -1)[0]
            # Вычитаем чтобы все координаты по x находились в интервале [0, 1]
            if i == 1:
                x_old -= breadth
                x_new -= breadth
            elif i == 2:
                x_old -= (breadth + depth)
                x_new -= (breadth + depth)
            elif i == 3:
                x_old -= (2 * breadth + depth)
                x_new -= (2 * breadth + depth)

            data_old = pressure_coefficients[i].reshape(1, -1)[0]
            # data_old_integer.append(data_old)
            coords = [[i1, j1] for i1, j1 in zip(x_old, z_old)]  # Старые координаты
            # Интерполятор полученный на основе имеющихся данных
            interpolator = Plot.interpolator(coords, data_old)

            # Получаем данные для несуществующих датчиков
            data_new = [float(interpolator([[X, Y]])) for X, Y in zip(x_new, z_new)]

            triang = mtri.Triangulation(x_new, z_new)
            refiner = mtri.UniformTriRefiner(triang)
            grid, value = refiner.refine_field(data_new, subdiv=4)
            data_colorbar = graph[i].tricontourf(grid, value, cmap=cmap, levels=levels, extend='both')
            aq = graph[i].tricontour(grid, value, linewidths=1, linestyles='solid', colors='black', levels=levels)
            x_dots, y_dots = np.meshgrid(x_old, z_old)
            graph[i].plot(x_dots, y_dots, '.k', **dict(markersize=3.7))
            graph[i].clabel(aq, fontsize=10)
            graph[i].set_ylim([0, height])
            if breadth == depth == height:
                graph[i].set_aspect('equal')
            if i in [0, 2]:
                graph[i].set_xlim([0, breadth])
                graph[i].set_xticks(ticks=np.arange(0, breadth + 0.1, 0.1))
            else:
                graph[i].set_xlim([0, depth])
                graph[i].set_xticks(ticks=np.arange(0, depth + 0.1, 0.1))
        fig.colorbar(data_colorbar, ax=graph, location='bottom', cmap=cmap, ticks=levels).ax.tick_params(labelsize=7)

        return fig

    @staticmethod
    def welch_graphs(model_name, alpha, angle, speed, scale, mode, data):
        # size = float(model_name[0]) / 10

        num_fig = f'{alpha}_{model_name}_{angle}_{mode}_{scale}'
        num_fig_b = ''
        b = num_fig.encode()
        num_fig_b = int(num_fig_b.join([str(b[i]) for i in range(len(b))]))

        fig, ax = plt.subplots(dpi=200, num=num_fig_b, clear=True)
        if scale == 'linear':
            ax.set_xlim([0, 15])

        elif scale == 'log':
            ax.set_xlim([10 ** -2, 10 ** 3])
            ax.set_xscale('log')
            ax.set_yscale('log')

        ax.grid()
        # ax.set_xlabel('Sh')
        ax.set_xlabel('Frequency')
        ax.set_ylabel('PSD, V**2/Hz')

        # frequency = [i for i in range(1, 16)]

        for name in data.keys():
            if data[name] is not None:
                temp, psd = welch(data[name], fs=1000, nperseg=int(32768 / 5))
                ax.plot(temp, psd, label=name)
            # peak = np.max(psd)
            # x = temp[np.where(psd == peak)]
            # y = peak
            # annotates.append(ax.annotate(np.array(x * size / speed).round(4)[0], xy=(x, y)))

        # ax.set_xticks(frequency, labels=[np.array(i * size / speed).round(3) for i in frequency])

        ax.legend(loc='upper right', fontsize=9)
        return fig

    @staticmethod
    def summary_coefficients(model_name, alpha, angle, mode, data):
        num_fig = f'{alpha}_{model_name}_{angle}_{mode}'
        num_fig_b = ''
        b = num_fig.encode()
        num_fig_b = int(num_fig_b.join([str(b[i]) for i in range(len(b))]))
        fig, ax = plt.subplots(dpi=200, num=num_fig_b, clear=True)
        ax.grid()
        ax.set_xlim(0, 32.768)
        ax.set_ylabel('Суммарные аэродинамические коэффициенты')
        ax.set_xlabel('Время, с')
        ox = np.linspace(0, 32.768, 32768)
        for name in data.keys():
            if data[name] is not None:
                ax.plot(ox, data[name], label=name)
        ax.legend(loc='upper right', fontsize=9)

        return fig
