import time
from typing import Tuple

from utils.utils import interpolator as intp
from utils.utils import ks10, alpha_standards, wind_regions

import toml
import numpy as np
from scipy.signal import welch
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.tri as mtri
from matplotlib.axis import rcParams
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import Normalize
from matplotlib.ticker import MultipleLocator, ScalarFormatter

plt.switch_backend('Agg')


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

    config = toml.load('config.toml')

    dpi = config['plots']['dpi']  # качество графиков

    @staticmethod
    def pseudocolor_coefficients(model_name: str, mode: str, angle: str, alpha: str, pressure_coefficients,
                                 coordinates):
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

        num_fig = f'Дискретные изополя {model_name} {mode} {alpha} {angle}'
        fig, ax = plt.subplots(1, 4, num=num_fig, dpi=Plot.dpi, clear=True)
        cmap = cm.get_cmap(name="jet")

        min_v = np.min([np.min(pressure_coefficients[i]) for i in range(4)])
        max_v = np.max([np.max(pressure_coefficients[i]) for i in range(4)])

        normalizer = Normalize(min_v, max_v)
        im = cm.ScalarMappable(norm=normalizer, cmap=cmap)
        for i in range(4):
            data_old = pressure_coefficients[i]
            ax[i].pcolormesh(np.flip(data_old, axis=0), cmap=cmap, norm=normalizer)
            labels = np.arange(0, height + 0.01, 0.05).round(2)
            ax[i].set_yticks(np.linspace(0, count_row, len(labels)))
            ax[i].set_yticklabels(labels=labels, fontsize=5)
            if i in [0, 2]:
                ax[i].set_xticks([i for i in range(0, count_sensors_on_middle + 1, 5)])
                ax[i].set_xticklabels(labels=list(map(str, np.arange(0, breadth + 0.1, 0.1).round(2))), fontsize=5)
            else:
                ax[i].set_xticks([i for i in range(0, count_sensors_on_side + 1, 5)])
                ax[i].set_xticklabels(labels=list(map(str, np.arange(0, depth + 0.1, 0.1).round(2))), fontsize=5)
            # x, y = np.meshgrid(np.arange(0.5, count_sensors_on_middle + 0.5, 1), z * count_row / height)
            # ax[i].plot(x, y, '.k')

        if breadth == depth == height:
            [ax[i].set_aspect('equal') for i in range(4)]
        ticks = np.arange(np.round(min_v, 1), np.round(max_v, 1) + 0.1, 0.1)
        fig.colorbar(im, ax=ax, location='bottom', cmap=cmap, ticks=ticks).ax.tick_params(labelsize=4)

        return fig

    @staticmethod
    def isofields_coefficients(model_name: str, model_size, scale_factors, alpha: str, mode: str, angle: str,
                               pressure_coefficients, coordinates, pressure_plot_parameters = None):
        """Отрисовка интегральных изополей"""
        coefficient_for_pressure = None
        levels = None

        # Виды изополей
        mods = {
            'max': np.max(pressure_coefficients, axis=0),
            'mean': np.mean(pressure_coefficients, axis=0),
            'min': np.min(pressure_coefficients, axis=0),
            'std': np.std(pressure_coefficients, axis=0),
        }

        size_x, size_y, size_z = map(float, model_size)
        x_scale_factor, y_scale_factor, z_scale_factor = scale_factors

        pressure_coefficients = mods[mode]
        breadth, depth, height = int(model_name[0]) / 10, int(model_name[1]) / 10, int(model_name[2]) / 10
        count_sensors_on_model = len(pressure_coefficients)
        count_sensors_on_middle = int(model_name[0]) * 5
        count_sensors_on_side = int(model_name[1]) * 5
        count_row = count_sensors_on_model // (2 * (count_sensors_on_middle + count_sensors_on_side))

        x, z = np.array(coordinates)
        # Шаги для изополей коэффициентов
        steps = {
            'max': 0.2,
            'mean': 0.2 if alpha == '6' else 0.1,
            'min': 0.2,
            'std': 0.05,
        }
        # Уровни для изополей коэффициентов
        if not pressure_plot_parameters:
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
        z_extended = [np.array(z1), np.array(z2), np.array(z3), np.array(z4)]
        x_extended = [np.array(x1), np.array(x2), np.array(x3), np.array(x4)]

        num_fig = f'Коэффициенты изополя {model_name} {model_size} {mode} {alpha} {angle}'
        fig, ax = plt.subplots(1, 4, num=num_fig, dpi=Plot.dpi, clear=True)
        cmap = cm.get_cmap(name="jet")
        data_colorbar = None

        h_scaled = height * z_scale_factor
        b_scaled = breadth * x_scale_factor
        d_scaled = depth * y_scale_factor

        x_new_list = []
        z_new_list = []

        x_old_list = []
        z_old_list = []

        data_new_list = []
        data_old_list = []

        if pressure_plot_parameters:
            type_area = pressure_plot_parameters['type_area']
            wind_region = pressure_plot_parameters['wind_region']
            alpha_standard = alpha_standards[type_area]
            k10 = ks10[type_area]
            wind_pressure = wind_regions[wind_region]

            PO = 1.225
            yf = 1.4
            ks = 1
            w0 = wind_pressure * 1000
            u10 = np.sqrt(2 * yf * k10 * w0 / PO)
            wind_profile = u10 * (ks * size_z / 10) ** alpha_standard
            coefficient_for_pressure = wind_profile ** 2 * PO / 2

        for i in range(4):
            x_new = x_extended[i].reshape(1, -1)[0]
            x_old = x[i].reshape(1, -1)[0]

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

            # Масштабирование координат
            z_new = z_new * z_scale_factor
            z_old = z_old * z_scale_factor

            if i in [0, 2]:
                x_new = x_new * x_scale_factor
                x_old = x_old * x_scale_factor

            else:
                x_new = x_new * y_scale_factor
                x_old = x_old * y_scale_factor

            data_old = pressure_coefficients[i].reshape(1, -1)[0]

            if pressure_plot_parameters:
                data_old = [coefficient_for_pressure * coefficient for coefficient in data_old]

            # data_old_integer.append(data_old)
            coords = [[i1, j1] for i1, j1 in zip(x_old, z_old)]  # Старые координаты
            # Интерполятор полученный на основе имеющихся данных
            interpolator = intp(coords, data_old)

            # Получаем данные для несуществующих датчиков
            data_new = [float(interpolator([[X, Y]])) for X, Y in zip(x_new, z_new)]

            x_new_list.append(x_new)
            z_new_list.append(z_new)

            x_old_list.append(x_old)
            z_old_list.append(z_old)

            data_new_list.append(data_new)
            #data_old_list.append(data_old)

        # Уровни для изополей давления
        if pressure_plot_parameters:
            min_0_2 = np.min(np.array([data_new_list[0], data_new_list[2]]))
            min_1_3 = np.min(np.array([data_new_list[1], data_new_list[3]]))
            min_v = np.min(np.array(min_0_2, min_1_3))

            max_0_2 = np.max(np.array([data_new_list[0], data_new_list[2]]))
            max_1_3 = np.max(np.array([data_new_list[1], data_new_list[3]]))
            max_v = np.max(np.array(max_0_2, max_1_3))

            levels = np.arange(min_v, max_v, (np.abs(min_v) + np.abs(max_v)) * 0.1).round(2)

        # for i in range(4):
        #     min_value = np.min(data_new_list[i])
        #     max_value = np.max(data_new_list[i])

        all_levels = np.linspace(np.min([np.min(data_new_list[i]) for i in range(4)]), np.max([np.max(data_new_list[i]) for i in range(4)]), 11)

        for i in range(4):
            triang = mtri.Triangulation(x_new_list[i], z_new_list[i])
            refiner = mtri.UniformTriRefiner(triang)
            grid, value = refiner.refine_field(data_new_list[i], subdiv=4)
            #min_value = np.min(data_new_list[i])
            #max_value = np.max(data_new_list[i])
            #temp_levels = np.linspace(min_value, max_value, 11)
            #print(all_levels)
            #data_colorbar = ax[i].tricontourf(grid, value, cmap=cmap, extend='both', levels=11)
            #print(data_colorbar.levels)
            data_colorbar = ax[i].tricontourf(grid, value, cmap=cmap, extend='both', levels=all_levels)
            #data_colorbar = ax[i].tricontourf(grid, value, cmap=cmap, extend='both', levels=temp_levels)
            #data_colorbar = ax[i].tricontourf(grid, value, cmap=cmap, extend='both', levels=levels)

            #aq = ax[i].tricontour(grid, value, linewidths=1, linestyles='solid', colors='black', levels=11)
            #print(aq.levels)
            aq = ax[i].tricontour(grid, value, linewidths=1, linestyles='solid', colors='black', levels=all_levels)
            #aq = ax[i].tricontour(grid, value, linewidths=1, linestyles='solid', colors='black', levels=temp_levels)
            #aq = ax[i].tricontour(grid, value, linewidths=1, linestyles='solid', colors='black', levels=levels)
            ax[i].clabel(aq, fontsize=10)

            ax[i].set_ylim([0, h_scaled])
            ax[i].set_yticks(np.arange(0, h_scaled + h_scaled * 0.01, h_scaled * 0.2))
            ax[i].set_yticklabels(labels=np.arange(0, size_z + size_z * 0.01, size_z * 0.2).round(2), fontsize=5)

            if i in [0, 2]:
                xlim = b_scaled
                label_b = size_x
            else:
                xlim = d_scaled
                label_b = size_y

            ax[i].set_xlim([0, xlim])
            ax[i].set_xticks(ticks=np.arange(0, xlim + xlim * 0.01, xlim * 0.2))
            ax[i].set_xticklabels(labels=np.arange(0, label_b + label_b * 0.01, label_b * 0.2).round(2), fontsize=5)

            if not pressure_plot_parameters:
                x_dots, y_dots = np.meshgrid(x_old_list[i], z_old_list[i])
                ax[i].plot(x_dots, y_dots, '.k', **dict(markersize=3))
        #print(all_levels)
        #fig.colorbar(data_colorbar, ax=ax, location='bottom', cmap=cmap, ticks=levels).ax.tick_params(labelsize=4)
        fig.colorbar(data_colorbar, ax=ax, location='bottom', cmap=cmap, ticks=all_levels).ax.tick_params(labelsize=4)

        return fig

    @staticmethod
    def summary_coefficients(data, model_scale: str, alpha: str, angle: str):
        """Графики суммарных аэродинамических коэффициентов в декартовой системе координат
        data = {name:array,
                ...
                }
        """
        num_fig = f'Суммарные коэффициенты декартовая система координат {model_scale} {alpha} {angle}'
        fig, ax = plt.subplots(dpi=Plot.dpi, num=num_fig, clear=True)
        ax.grid()
        ax.set_xlim(0, 32.768)
        ax.set_ylabel('Суммарные аэродинамические коэффициенты')
        ax.set_xlabel('Время, с', labelpad=.3)
        ox = np.linspace(0, 32.768, 32768)
        for name in data.keys():
            if data[name] is not None:
                ax.plot(ox, data[name], label=name)
        ax.legend(loc='upper right', fontsize=9)

        return fig

    @staticmethod
    def scaling_data(x, y = None, angle_border = 50):
        """Масштабирование данных до 360 градусов"""
        if angle_border == 50:
            if y is not None:
                a = np.array(y)
                b = np.append(a, np.flip(x)[1:])
                c = np.append(b, np.flip(b)[1:])
                x_scale = np.append(c, np.flip(c)[1:])

                a = np.array(x)
                b = np.append(a, np.flip(y)[1:])
                c = np.append(b, np.flip(b)[1:])
                y_scale = np.append(c, np.flip(c)[1:])

                return x_scale, y_scale

            else:
                a = np.array(x)
                b = np.append(a, np.flip(x)[1:])
                c = np.append(b, np.flip(b)[1:])
                x_scale = np.append(c, np.flip(c)[1:])
                return x_scale

        elif angle_border == 95:
            if y is not None:
                a = np.array(y)
                b = np.append(a, np.flip(x)[1:])
                x_scale = np.append(b, np.flip(b)[1:])

                a = np.array(x)
                b = np.append(a, np.flip(y)[1:])
                y_scale = np.append(b, np.flip(b)[1:])

                return x_scale, y_scale

            else:
                a = np.array(x)
                b = np.append(a, np.flip(x)[1:])
                x_scale = np.append(b, np.flip(b)[1:])
                return x_scale

    @staticmethod
    def polar_plot(data, title: str, model_size, alpha: str):
        """Графики суммарных аэродинамических коэффициентов в полярной системе координат.
        data = {name:array,
                ...
                }
        """
        angles = np.array([angle for angle in range(0, 365, 5)]) * np.pi / 180.0
        num_fig = f'Суммарные коэффициенты декартовая система координат {title} {" ".join(model_size)} {alpha}'

        fig, ax = plt.subplots(dpi=Plot.dpi, num=num_fig, clear=True, subplot_kw={'projection': 'polar'})

        for name in data.keys():
            ax.plot(angles, data[name], label=name)

        ax.set_theta_direction(-1)
        ax.set_theta_zero_location('N')
        ax.set_thetagrids([i for i in range(0, 360, 15)])
        ax.legend(loc='upper right', fontsize=9)
        ax.set_title(title)

        return fig

    @staticmethod
    def model_pic(model_size, model_scale, coordinates):
        """Отрисовка развертки модели"""
        breadth_real, depth_real, height_real = float(model_size[0]), float(model_size[1]), float(model_size[2])

        size_x = 2 * (breadth_real + depth_real)
        x, z = coordinates

        num_fig = f'Развертка модели {" ".join(model_size)}'
        fig, ax = plt.subplots(figsize=(16, 9), num=num_fig, dpi=Plot.dpi, clear=True)
        ax.set_title('Развертка датчиков по модели', fontweight='semibold', fontsize=8)
        ax.set_xlabel('Горизонтальная развертка /м', fontweight='semibold', fontsize=8)
        ax.set_ylabel('Высота модели /м', fontweight='semibold', fontsize=8)
        ax.set_ylim(0, height_real)
        ax.set_xlim(0, size_x)

        xticks = np.array([0, breadth_real, breadth_real + depth_real, 2 * breadth_real + depth_real, size_x])
        yticks = np.arange(0, height_real + height_real * 0.01, height_real * 0.2)

        ax.set_xticks(ticks=xticks)
        ax.set_yticks(ticks=yticks)

        ax.set_xticklabels(labels=(xticks / 10).round(3))
        ax.set_yticklabels(labels=(yticks / 10).round(3))

        ax.xaxis.set_minor_locator(MultipleLocator(size_x * 0.03125))
        ax.yaxis.set_minor_locator(MultipleLocator(height_real * 0.05))

        ax.xaxis.set_minor_formatter(ScalarFormatter())
        ax.yaxis.set_minor_formatter(ScalarFormatter())

        minor_xticks = (np.array(ax.get_xticks(minor=True)) / 10).round(3)
        minor_yticks = (np.array(ax.get_yticks(minor=True)) / 10).round(3)

        ax.set_xticklabels(labels=minor_xticks, minor=True)
        ax.set_yticklabels(labels=minor_yticks, minor=True)

        ax.tick_params(axis='x', which='minor', pad=5, labelsize=7)
        ax.tick_params(axis='x', which='major', pad=10, labelsize=10)

        ax.tick_params(axis='y', which='minor', pad=5, labelsize=7)
        ax.tick_params(axis='y', which='major', pad=10, labelsize=10)

        count_sensors_on_model = len(x)
        count_sensors_on_middle = int(model_scale[0]) * 5
        count_sensors_on_side = int(model_scale[1]) * 5
        count_row = count_sensors_on_model // (2 * (count_sensors_on_middle + count_sensors_on_side))

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
        del x[4]
        del z[4]

        for i in (breadth_real, breadth_real + depth_real, breadth_real + depth_real, 2 * breadth_real + depth_real):
            ax.plot([i, i], [0, height_real], linestyle='--', color='black')

        for i in range(4):
            ax.plot(x[i], z[i], 'b+')

        labels = [str(i) for i in range(1, count_sensors_on_model + 1)]
        labels = np.reshape(labels, (count_row, -1))
        labels = np.split(labels, [count_sensors_on_middle,
                                   count_sensors_on_middle + count_sensors_on_side,
                                   2 * count_sensors_on_middle + count_sensors_on_side,
                                   2 * (count_sensors_on_middle + count_sensors_on_side)
                                   ], axis=1)

        d = height_real * 0.02

        for f in range(4):
            for i in range(len(x[f])):
                for j in range(len(x[f][i])):
                    ax.text(x[f][i][j], z[f][i][j] - d, labels[f][i][j], fontsize=8)

        return fig

    @staticmethod
    def model_polar(model_size):
        """Отрисовка модели в полярной системе координат"""
        x, y, _ = float(model_size[0]), float(model_size[1]), float(model_size[2])
        min_size = min(x, y)
        b_scale, d_scale = x / min_size, y / min_size

        num_fig = f'Модель в полярной системе координат {" ".join(model_size)}'
        fig = plt.figure(num=num_fig, dpi=Plot.dpi, clear=True)
        pos = [0.1, 0.1, 0.8, 0.8]
        polar = fig.add_axes(pos, projection='polar')
        polar.patch.set_alpha(0)
        polar.set_theta_zero_location('N')
        polar.set_theta_direction(-1)
        polar.set_thetagrids([i for i in range(0, 360, 15)], fontsize=7)
        polar.set_yticks([0, 1, 2, 3])
        polar.set_yticklabels([0, 1, 2, 3], visible=False)
        polar.set_autoscale_on(False)
        angles = np.array([angle for angle in range(0, 365, 5)]) * np.pi / 180.0
        polar.annotate("", xy=(angles[0], 2), xytext=(0, 0),
                       arrowprops=dict(arrowstyle="->",
                                       linewidth=2.5))
        polar.annotate("", xy=(angles[18], 2), xytext=(0, 0),
                       arrowprops=dict(arrowstyle="->",
                                       linewidth=2.5))
        polar.annotate("y", xy=(angles[0], 2))
        polar.annotate("x", xy=(angles[18], 2))

        ax = fig.add_subplot(111, position=pos)
        ax.set_visible(True)
        ax.set_autoscale_on(False)
        ax.set_aspect(1)
        ax.axis('off')
        dx = dy = (0.7 - 0.3) / 3
        b = dx * b_scale
        d = dy * d_scale
        mid = 0.5
        x0 = mid - b / 2
        x1 = mid + b / 2
        y0 = mid - d / 2
        y1 = mid + d / 2
        ax.fill_between([x0, x1], [y0, y0], [y1, y1], color='grey', alpha=0.5)

        return fig

    @staticmethod
    def model_3d(model_size):
        """Отрисовка модели в трехмерном виде"""
        x, y, z = float(model_size[0]), float(model_size[1]), float(model_size[2])
        min_size = min(x, y, z)
        b_scale, d_scale, h_scale = x / min_size, y / min_size, z / min_size

        num_fig = f'Модель в трехмерном представлении {" ".join(model_size)}'
        fig = plt.figure(num=num_fig, dpi=Plot.dpi, clear=True)
        ax = fig.add_subplot(1, 1, 1, projection='3d')

        count_nodes = 2
        x = np.linspace(1, b_scale + 1, count_nodes)
        y = np.linspace(1, d_scale + 1, count_nodes)
        z = np.linspace(0, h_scale, count_nodes)

        ones1d = np.ones(count_nodes)
        zeros2d = np.zeros((count_nodes, count_nodes))

        X_top, Y_top = np.meshgrid(x, y)
        Z_top = zeros2d + h_scale
        ax.plot_surface(X_top, Y_top, Z_top, color='grey')

        X_face, Y_face = np.meshgrid(x, ones1d)
        Z_face = (zeros2d + z).T
        ax.plot_surface(X_face, Y_face, Z_face, color='grey')

        Y_side, X_side = np.meshgrid(y, ones1d + b_scale)
        ax.plot_surface(X_side, Y_side, Z_face, color='grey')

        ax.set_xlabel('Depth')
        ax.set_ylabel('Breadth')
        ax.set_zlabel('Height')

        # Get rid of the panes
        ax.w_xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        ax.w_yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        ax.w_zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))

        # Get rid of the spines
        ax.w_xaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
        ax.w_yaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
        ax.w_zaxis.line.set_color((1.0, 1.0, 1.0, 0.0))

        # Get rid of the ticks
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])

        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False

        # Now set color to white (or whatever is "invisible")
        ax.xaxis.pane.set_edgecolor('w')
        ax.yaxis.pane.set_edgecolor('w')
        ax.zaxis.pane.set_edgecolor('w')

        ax.grid(False)
        max_range = np.array([b_scale, d_scale, h_scale]).max() / 2 + 1

        mid_x = (b_scale + 1) * 0.5
        mid_y = (d_scale + 1) * 0.5
        mid_z = (h_scale + 1) * 0.5
        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)
        ax.set_zlim(mid_z - max_range, mid_z + max_range)

        return fig

    @staticmethod
    def envelopes(pressure_coefficients, alpha: str, model_scale: str, angle: str, mods):
        """Отрисовка огибающих"""

        colors = ('b', 'g', 'r', 'c', 'y')[:len(mods)]
        names = [i for i in ('mean', 'rms', 'std', 'max', 'min') if i in mods]
        lambdas = {
            'mean': lambda coefficients: np.mean(coefficients, axis=0).round(4),
            'rms': lambda coefficients: np.array([np.sqrt(i.dot(i) / i.size) for i in coefficients.T]).round(4),
            'std': lambda coefficients: np.std(coefficients, axis=0).round(4),
            'max': lambda coefficients: np.max(coefficients, axis=0).round(4),
            'min': lambda coefficients: np.min(coefficients, axis=0).round(4),
        }

        figs = []  # массив для графиков так как на 1 графике максимум 100 датчиков
        step_x = 20
        step_x_minor = 5
        step_y = 0.4

        step_sens = 100
        count_sensors_plot = len(pressure_coefficients[0])

        sensors = list(range(0, count_sensors_plot, step_sens))

        for q in sensors:
            num_fig = f'Огибающие {model_scale} {alpha} {angle} {q}_{"_".join(names)}'
            fig, ax = plt.subplots(dpi=Plot.dpi, num=num_fig, clear=True)
            ax.grid(visible=True, which='minor', color='black', linestyle='--')
            ax.grid(visible=True, which='major', color='black', linewidth=1.5)

            coefficients = pressure_coefficients.T[q:q + step_sens].T
            data_for_plot = [lambdas[name](coefficients) for name in names]

            if q == sensors[-1]:
                len_data = len(data_for_plot[0])
            else:
                len_data = step_sens

            ox = [i for i in range(q + 1, q + len_data + 1)]

            for i, j, c in zip(data_for_plot, names, colors):
                ax.plot(ox, i, '-', label=j, linewidth=3, color=c)

            ax.set_xlim([q + 1, q + len_data])

            all_data = []
            [all_data.extend(i) for i in data_for_plot]

            yticks = np.arange(np.min(all_data) - step_y, np.max(all_data) + step_y, step_y).round(2)
            ax.set_ylim(np.min(yticks) + 0.2, np.max(yticks) + 0.2)
            ax.set_yticks(yticks)

            ax.set_xticks([q + 1] + [i for i in range(q + step_x, q + len_data + 1, 20)])

            ax.xaxis.set_minor_locator(MultipleLocator(step_x_minor))
            ax.xaxis.set_minor_formatter(ScalarFormatter())
            ax.xaxis.set_tick_params(which='major', labelsize=10)
            ax.xaxis.set_tick_params(which='minor', labelsize=7)

            ax.legend(loc='upper right', fontsize=9)
            ax.set_title('Огибающие')

            figs.append(fig)

        return figs

    @staticmethod
    def welch_graphs(data, model_size, alpha: str, angle: str):
        """Отрисовка графиков спектральной плотности мощности"""
        num_fig = f'Спектральная плотность мощности {model_size} {alpha} {angle}'
        fig, ax = plt.subplots(dpi=Plot.dpi, num=num_fig, clear=True)

        ax.set_xlim([10 ** -2, 10 ** 3])
        ax.set_xscale('log')
        ax.set_yscale('log')

        ax.grid()
        ax.set_xlabel('Frequency')
        ax.set_ylabel('PSD, V**2/Hz')

        for name in data.keys():
            if data[name] is not None:
                temp, psd = welch(data[name], fs=1000, nperseg=int(32768 / 5))
                ax.plot(temp, psd, label=name)

        ax.legend(loc='upper right', fontsize=9)

        return fig
