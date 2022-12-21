import numpy as np
import scipy.interpolate
import scipy.io as sio
import random
import matplotlib.cm as cm
import matplotlib.tri as mtri
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize


class Plot:
    """Класс отвечает за отрисовку графиков
       Виды графиков:
           -Изополя:
               -Максимальные значения
               -Средние значения
               -Минимальные значения
               -Среднее квадратичное отклонение
           -Огибающие:
               -Максимальные значения
               -Минимальные значения
           -Коэффициенты обеспеченности:
               -Максимальные значения
               -Минимальные значения
           -Спектры
           -Нестационарные сигналы
       Нумерация графиков нужная для очищения памяти:
        1   discrete isofields_min
        2   discrete isofields_mean
        3   discrete isofields_max
        4   discrete isofields_std
        5   integral isofields_min
        6   integral isofields_mean
        7   integral isofields_max
        8   integral isofields_std
       """

    @staticmethod
    def discrete_isofield(model_name, mode, pressure_coefficients, coordinates):
        """Отрисовка дискретных изополей"""
        # Виды изополей
        mods = {
            'max': np.max(pressure_coefficients, axis=0),
            'mean': np.mean(pressure_coefficients, axis=0),
            'min': np.min(pressure_coefficients, axis=0),
            'std': np.std(pressure_coefficients, axis=0),
        }
        num = {
            'min': 1,
            'mean': 2,
            'max': 3,
            'std': 4,
        }
        pressure_coefficients = mods[mode]
        min_v = np.min(pressure_coefficients)
        max_v = np.max(pressure_coefficients)
        x, z = np.array(coordinates)
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

        z = np.array(z[::2 * (count_sensors_on_middle + count_sensors_on_side)])[::-1]
        fig, ax = plt.subplots(1, 4, dpi=200, num=num[mode], clear=True)
        cmap = cm.get_cmap(name="jet")
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
                x, y = np.meshgrid(np.arange(0.5, count_sensors_on_middle + 0.5, 1), z * count_row / height)
                ax[i].plot(x, y, '.k')
            else:
                ax[i].set_xticks([i for i in range(0, count_sensors_on_side + 1, 5)])
                ax[i].set_xticklabels(labels=list(map(str, np.arange(0, depth + 0.1, 0.1))))
                x, y = np.meshgrid(np.arange(0.5, count_sensors_on_middle + 0.5, 1), z * count_row / height)
                ax[i].plot(x, y, '.k')

        if breadth == depth == height:
            [ax[i].set_aspect('equal') for i in range(4)]
        ticks = np.arange(np.round(min_v, 1), np.round(max_v, 1) + 0.1, 0.1)
        # fig.colorbar(im, ax=ax.ravel().tolist(), location='bottom', cmap=cmap, ticks=ticks).ax.tick_params(
        #     labelsize=7)
        fig.colorbar(im, ax=ax, location='bottom', cmap=cmap, ticks=ticks).ax.tick_params(
            labelsize=7)
        return fig

    @staticmethod
    def interpolator(coords, val):
        return scipy.interpolate.RBFInterpolator(coords, val, kernel='cubic')

    @staticmethod
    def integral_isofield(model_name, mode, pressure_coefficients, coordinates, alpha):
        """Отрисовка интегральных изополей"""
        integral_func = []
        mods = {
            'max': np.max(pressure_coefficients, axis=0),
            'mean': np.mean(pressure_coefficients, axis=0),
            'min': np.min(pressure_coefficients, axis=0),
            'std': np.std(pressure_coefficients, axis=0),
        }  # Виды изополей
        num = {
            'min': 5,
            'mean': 6,
            'max': 7,
            'std': 8,
        }
        pressure_coefficients = mods[mode]
        min_v = np.min(pressure_coefficients)
        max_v = np.max(pressure_coefficients)
        steps = {
            'max': 0.2,
            'mean': 0.2 if alpha == 6 else 0.1,
            'min': 0.2,
            'std': 0.05,
        }  # Шаги для изополей
        #  Шаг для изополей и контурных линий
        levels = np.arange(np.round(min_v, 1), np.round(max_v, 1) + 0.1, steps[mode]).round(2)
        x, z = np.array(coordinates)
        # if integral[1] == 1:
        #     # Масштабирование
        #     print(model_name, 1.25)
        #     k = 1.25  # коэффициент масштабирования по высоте
        #     z *= k
        # else:
        #     k = 1
        #     print(model_name, 1)
        breadth, depth, height = int(model_name[0]) / 10, int(model_name[1]) / 10, int(model_name[2]) / 10
        count_sensors_on_model = len(pressure_coefficients)
        count_sensors_on_middle = int(model_name[0]) * 5
        count_sensors_on_side = int(model_name[1]) * 5
        count_row = count_sensors_on_model // (2 * (count_sensors_on_middle + count_sensors_on_side))
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

        for i in range(4):
            x[i] = x[i].tolist()
            z[i] = z[i].tolist()

        x1, x2, x3, x4 = x  # x это тензор со всеми координатами граней по ширине,x1...x4 координаты отдельных граней
        z1, z2, z3, z4 = z  # z это тензор со всеми координатами граней по высоте,z1...z4 координаты отдельных граней

        x = [np.array(x1), np.array(x2), np.array(x3), np.array(x4)]  # Входные координаты для изополей
        z = [np.array(z1), np.array(z2), np.array(z3), np.array(z4)]  # Входные координаты для изополей
        ret_int = []  # массив с функциями изополей
        # Расширение матрицы координат по бокам
        for i in range(len(x1)):
            x1[i] = [0] + x1[i] + [breadth]
            x2[i] = [breadth] + x2[i] + [breadth + depth]
            x3[i] = [breadth + depth] + x3[i] + [2 * breadth + depth]
            x4[i] = [2 * breadth + depth] + x4[i] + [2 * (breadth + depth)]

        x1.append(x1[0])
        x2.append(x2[0])
        x3.append(x3[0])
        x4.append(x4[0])

        x1.insert(0, x1[0])
        x2.insert(0, x2[0])
        x3.insert(0, x3[0])
        x4.insert(0, x4[0])

        x_extended = [np.array(x1), np.array(x2), np.array(x3), np.array(x4)]  # Расширенные координаты для изополей

        # Расширение матрицы координат по бокам
        for i in range(len(z1)):
            z1[i] = [z1[i][0]] + z1[i] + [z1[i][0]]
            z2[i] = [z2[i][0]] + z2[i] + [z2[i][0]]
            z3[i] = [z3[i][0]] + z3[i] + [z3[i][0]]
            z4[i] = [z4[i][0]] + z4[i] + [z4[i][0]]

        z1.append([0 for _ in range(len(z1[0]))])
        z2.append([0 for _ in range(len(z2[0]))])
        z3.append([0 for _ in range(len(z3[0]))])
        z4.append([0 for _ in range(len(z4[0]))])

        z1.insert(0, [height for _ in range(len(z1[0]))])
        z2.insert(0, [height for _ in range(len(z2[0]))])
        z3.insert(0, [height for _ in range(len(z3[0]))])
        z4.insert(0, [height for _ in range(len(z4[0]))])

        z_extended = [np.array(z1), np.array(z2), np.array(z3), np.array(z4)]  # Расширенные координаты для изополей
        fig, graph = plt.subplots(1, 4, dpi=200, num=num[mode], clear=True, figsize=(9, 5))
        cmap = cm.get_cmap(name="jet")
        data_colorbar = None
        # data_old_integer = []  # данные для дискретного интегрирования по осям
        # data_for_3d_model = []  # данные для 3D модели
        for i in range(4):
            # x это координаты по ширине
            x_new = x_extended[i].reshape(1, -1)[0]
            x_old = x[i].reshape(1, -1)[0]
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
            # z это координаты по высоте
            z_new = z_extended[i].reshape(1, -1)[0]
            z_old = z[i].reshape(1, -1)[0]

            data_old = pressure_coefficients[i].reshape(1, -1)[0]
            # data_old_integer.append(data_old)
            coords = [[i1, j1] for i1, j1 in zip(x_old, z_old)]  # Старые координаты
            # Интерполятор полученный на основе имеющихся данных
            interpolator = Plot.interpolator(coords, data_old)
            integral_func.append(interpolator)
            # Получаем данные для несуществующих датчиков
            data_new = [float(interpolator([[X, Y]])) for X, Y in zip(x_new, z_new)]

            triang = mtri.Triangulation(x_new, z_new)
            refiner = mtri.UniformTriRefiner(triang)
            grid, value = refiner.refine_field(data_new, subdiv=4)
            # data_for_3d_model.append((grid, value))
            data_colorbar = graph[i].tricontourf(grid, value, cmap=cmap, levels=levels, extend='both')
            aq = graph[i].tricontour(grid, value, linewidths=1, linestyles='solid', colors='black', levels=levels)
            X, Y = np.meshgrid(x_old, z_old)
            graph[i].plot(X, Y, '.k', **dict(markersize=3.7))
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
            ret_int.append(interpolator)
            # graph[i].axis('off')
        fig.colorbar(data_colorbar, ax=graph, location='bottom', cmap=cmap, ticks=levels).ax.tick_params(labelsize=7)

        return fig

# @staticmethod
# def interpolator(coords, val):
#     return scipy.interpolate.RBFInterpolator(coords, val, kernel='cubic')
#
# @staticmethod
# def film(coeffs, coordinates, model_name):
#     print('START')
#     min_v = np.min(coeffs)
#     max_v = np.max(coeffs)
#     #  Шаг для изополей и контурных линий
#     levels = np.arange(np.round(min_v, 1), np.round(max_v, 1) + 0.1, 0.1)
#     x, z = np.array(coordinates)
#
#     breadth, depth, height = int(model_name[0]) / 10, int(model_name[1]) / 10, int(model_name[2]) / 10
#     count_sensors_on_model = len(coeffs[0])
#     count_sensors_on_middle = int(model_name[0]) * 5
#     count_sensors_on_side = int(model_name[1]) * 5
#     count_row = count_sensors_on_model // (2 * (count_sensors_on_middle + count_sensors_on_side))
#     x = np.reshape(x, (count_row, -1))
#     x = np.split(x, [count_sensors_on_middle,
#                      count_sensors_on_middle + count_sensors_on_side,
#                      2 * count_sensors_on_middle + count_sensors_on_side,
#                      2 * (count_sensors_on_middle + count_sensors_on_side)
#                      ], axis=1)
#     z = np.reshape(z, (count_row, -1))
#     z = np.split(z, [count_sensors_on_middle,
#                      count_sensors_on_middle + count_sensors_on_side,
#                      2 * count_sensors_on_middle + count_sensors_on_side,
#                      2 * (count_sensors_on_middle + count_sensors_on_side)
#                      ], axis=1)
#
#     del x[4]
#     del z[4]
#
#     for i in range(4):
#         x[i] = x[i].tolist()
#         z[i] = z[i].tolist()
#
#     x1, x2, x3, x4 = x  # x это тензор со всеми координатами граней по ширине,x1...x4 координаты отдельных граней
#     z1, z2, z3, z4 = z  # z это тензор со всеми координатами граней по высоте,z1...z4 координаты отдельных граней
#
#     x = [np.array(x1), np.array(x2), np.array(x3), np.array(x4)]  # Входные координаты для изополей
#     z = [np.array(z1), np.array(z2), np.array(z3), np.array(z4)]  # Входные координаты для изополей
#     ret_int = []  # массив с функциями изополей
#
#     # Расширение матрицы координат по бокам
#     for i in range(len(x1)):
#         x1[i] = [0] + x1[i] + [breadth]
#         x2[i] = [breadth] + x2[i] + [breadth + depth]
#         x3[i] = [breadth + depth] + x3[i] + [2 * breadth + depth]
#         x4[i] = [2 * breadth + depth] + x4[i] + [2 * (breadth + depth)]
#
#     x1.append(x1[0])
#     x2.append(x2[0])
#     x3.append(x3[0])
#     x4.append(x4[0])
#
#     x1.insert(0, x1[0])
#     x2.insert(0, x2[0])
#     x3.insert(0, x3[0])
#     x4.insert(0, x4[0])
#
#     x_extended = [np.array(x1), np.array(x2), np.array(x3), np.array(x4)]  # Расширенные координаты для изополей
#
#     # Расширение матрицы координат по бокам
#     for i in range(len(z1)):
#         z1[i] = [z1[i][0]] + z1[i] + [z1[i][0]]
#         z2[i] = [z2[i][0]] + z2[i] + [z2[i][0]]
#         z3[i] = [z3[i][0]] + z3[i] + [z3[i][0]]
#         z4[i] = [z4[i][0]] + z4[i] + [z4[i][0]]
#
#     z1.append([0 for _ in range(len(z1[0]))])
#     z2.append([0 for _ in range(len(z2[0]))])
#     z3.append([0 for _ in range(len(z3[0]))])
#     z4.append([0 for _ in range(len(z4[0]))])
#
#     z1.insert(0, [height for _ in range(len(z1[0]))])
#     z2.insert(0, [height for _ in range(len(z2[0]))])
#     z3.insert(0, [height for _ in range(len(z3[0]))])
#     z4.insert(0, [height for _ in range(len(z4[0]))])
#
#     z_extended = [np.array(z1), np.array(z2), np.array(z3), np.array(z4)]  # Расширенные координаты для изополей
#
#     fig, graph = plt.subplots(1, 4, figsize=(16, 9))
#     camera = Camera(fig)
#
#     cmap = cm.get_cmap(name="jet")
#
#     x_new = []
#     x_old = []
#     z_new = []
#     z_old = []
#     coords = []
#     data_new = []
#     for i in range(4):
#         # x это координаты по ширине
#         x_new.append(x_extended[i].reshape(1, -1)[0])
#         x_old.append(x[i].reshape(1, -1)[0])
#         # Вычитаем чтобы все координаты по x находились в интервале [0, 1]
#         if i == 1:
#             x_old[i] = x_old[i] - breadth
#             x_new[i] = x_new[i] - breadth
#         elif i == 2:
#             x_old[i] = x_old[i] - (breadth + depth)
#             x_new[i] = x_new[i] - (breadth + depth)
#         elif i == 3:
#             x_old[i] = x_old[i] - (2 * breadth + depth)
#             x_new[i] = x_new[i] - (2 * breadth + depth)
#         # z это координаты по высоте
#         z_new.append(z_extended[i].reshape(1, -1)[0])
#         z_old.append(z[i].reshape(1, -1)[0])
#         coords.append([[i1, j1] for i1, j1 in zip(x_old[i], z_old[i])])
#
#         graph[i].set_ylim([0, height])
#         if breadth == depth == height:
#             graph[i].set_aspect('equal')
#         if i in [0, 2]:
#             graph[i].set_xlim([0, breadth])
#             graph[i].set_xticks(ticks=np.arange(0, breadth + 0.1, 0.1))
#         else:
#             graph[i].set_xlim([0, depth])
#             graph[i].set_xticks(ticks=np.arange(0, depth + 0.1, 0.1))
#
#     qqq = 0
#     for pressure_coefficients in coeffs[:300]:
#         qqq += 1
#         print(qqq)
#
#         pressure_coefficients = np.reshape(pressure_coefficients, (count_row, -1))
#         pressure_coefficients = np.split(pressure_coefficients,
#                                          [count_sensors_on_middle,
#                                           count_sensors_on_middle + count_sensors_on_side,
#                                           2 * count_sensors_on_middle + count_sensors_on_side,
#                                           2 * (
#                                                   count_sensors_on_middle + count_sensors_on_side)
#                                           ], axis=1)
#         del pressure_coefficients[4]
#
#         for i in range(4):
#             data_old = pressure_coefficients[i].reshape(1, -1)[0]
#             # Интерполятор полученный на основе имеющихся данных
#             interpolator = Artist.interpolator(coords[i], data_old)
#             # Получаем данные для несуществующих датчиков
#             data_new = [float(interpolator([[X, Y]])) for X, Y in zip(x_new[i], z_new[i])]
#             triang = mtri.Triangulation(x_new[i], z_new[i])
#             refiner = mtri.UniformTriRefiner(triang)
#             grid, value = refiner.refine_field(data_new, subdiv=4)
#             data_colorbar = graph[i].tricontourf(grid, value, cmap=cmap, levels=levels, extend='both')
#             aq = graph[i].tricontour(grid, value, linewidths=1, linestyles='solid', colors='black', levels=levels)
#             graph[i].clabel(aq, fontsize=13)
#         camera.snap()
#
#     cbar = fig.colorbar(data_colorbar, ax=graph, location='bottom', cmap=cmap, ticks=levels)
#     cbar.ax.tick_params(labelsize=5)
#     animation = camera.animate(interval=125)
#     animation.save('wind.gif')
#
#     plt.clf()
#     plt.close()
#
# @staticmethod
# def isofield(mode, pressure_coefficients, coordinates, alpha, model_name, angle = 0):
#     """Отрисовка изополей"""
#     integral_func = []
#     mods = {
#         'max': np.max(pressure_coefficients, axis=0),
#         'mean': np.mean(pressure_coefficients, axis=0),
#         'min': np.min(pressure_coefficients, axis=0),
#         'std': np.std(pressure_coefficients, axis=0),
#     }  # Виды изополей
#     pressure_coefficients = mods[mode]
#     min_v = np.min(pressure_coefficients)
#     max_v = np.max(pressure_coefficients)
#     steps = {
#         'max': 0.2,
#         'mean': 0.2 if alpha == 6 else 0.1,
#         'min': 0.2,
#         'std': 0.05,
#     }  # Шаги для изополей
#     #  Шаг для изополей и контурных линий
#     levels = np.arange(np.round(min_v, 1), np.round(max_v, 1) + 0.1, steps[mode]).round(2)
#     x, z = np.array(coordinates)
#     # if integral[1] == 1:
#     #     # Масштабирование
#     #     print(model_name, 1.25)
#     #     k = 1.25  # коэффициент масштабирования по высоте
#     #     z *= k
#     # else:
#     #     k = 1
#     #     print(model_name, 1)
#     breadth, depth, height = int(model_name[0]) / 10, int(model_name[1]) / 10, int(model_name[2]) / 10
#     count_sensors_on_model = len(pressure_coefficients)
#     count_sensors_on_middle = int(model_name[0]) * 5
#     count_sensors_on_side = int(model_name[1]) * 5
#     count_row = count_sensors_on_model // (2 * (count_sensors_on_middle + count_sensors_on_side))
#     pressure_coefficients = np.reshape(pressure_coefficients, (count_row, -1))
#     pressure_coefficients = np.split(pressure_coefficients, [count_sensors_on_middle,
#                                                              count_sensors_on_middle + count_sensors_on_side,
#                                                              2 * count_sensors_on_middle + count_sensors_on_side,
#                                                              2 * (count_sensors_on_middle + count_sensors_on_side)
#                                                              ], axis=1)
#     x = np.reshape(x, (count_row, -1))
#     x = np.split(x, [count_sensors_on_middle,
#                      count_sensors_on_middle + count_sensors_on_side,
#                      2 * count_sensors_on_middle + count_sensors_on_side,
#                      2 * (count_sensors_on_middle + count_sensors_on_side)
#                      ], axis=1)
#
#     z = np.reshape(z, (count_row, -1))
#     z = np.split(z, [count_sensors_on_middle,
#                      count_sensors_on_middle + count_sensors_on_side,
#                      2 * count_sensors_on_middle + count_sensors_on_side,
#                      2 * (count_sensors_on_middle + count_sensors_on_side)
#                      ], axis=1)
#
#     del pressure_coefficients[4]
#     del x[4]
#     del z[4]
#
#     for i in range(4):
#         x[i] = x[i].tolist()
#         z[i] = z[i].tolist()
#
#     x1, x2, x3, x4 = x  # x это тензор со всеми координатами граней по ширине,x1...x4 координаты отдельных граней
#     z1, z2, z3, z4 = z  # z это тензор со всеми координатами граней по высоте,z1...z4 координаты отдельных граней
#
#     x = [np.array(x1), np.array(x2), np.array(x3), np.array(x4)]  # Входные координаты для изополей
#     z = [np.array(z1), np.array(z2), np.array(z3), np.array(z4)]  # Входные координаты для изополей
#     ret_int = []  # массив с функциями изополей
#     # Расширение матрицы координат по бокам
#     for i in range(len(x1)):
#         x1[i] = [0] + x1[i] + [breadth]
#         x2[i] = [breadth] + x2[i] + [breadth + depth]
#         x3[i] = [breadth + depth] + x3[i] + [2 * breadth + depth]
#         x4[i] = [2 * breadth + depth] + x4[i] + [2 * (breadth + depth)]
#
#     x1.append(x1[0])
#     x2.append(x2[0])
#     x3.append(x3[0])
#     x4.append(x4[0])
#
#     x1.insert(0, x1[0])
#     x2.insert(0, x2[0])
#     x3.insert(0, x3[0])
#     x4.insert(0, x4[0])
#
#     x_extended = [np.array(x1), np.array(x2), np.array(x3), np.array(x4)]  # Расширенные координаты для изополей
#
#     # Расширение матрицы координат по бокам
#     for i in range(len(z1)):
#         z1[i] = [z1[i][0]] + z1[i] + [z1[i][0]]
#         z2[i] = [z2[i][0]] + z2[i] + [z2[i][0]]
#         z3[i] = [z3[i][0]] + z3[i] + [z3[i][0]]
#         z4[i] = [z4[i][0]] + z4[i] + [z4[i][0]]
#
#     z1.append([0 for _ in range(len(z1[0]))])
#     z2.append([0 for _ in range(len(z2[0]))])
#     z3.append([0 for _ in range(len(z3[0]))])
#     z4.append([0 for _ in range(len(z4[0]))])
#
#     z1.insert(0, [height for _ in range(len(z1[0]))])
#     z2.insert(0, [height for _ in range(len(z2[0]))])
#     z3.insert(0, [height for _ in range(len(z3[0]))])
#     z4.insert(0, [height for _ in range(len(z4[0]))])
#
#     z_extended = [np.array(z1), np.array(z2), np.array(z3), np.array(z4)]  # Расширенные координаты для изополей
#     fig, graph = plt.subplots(1, 4, dpi=200, num=1, clear=True, figsize=(9, 5))
#     cmap = cm.get_cmap(name="jet")
#     data_colorbar = None
#     # data_old_integer = []  # данные для дискретного интегрирования по осям
#     # data_for_3d_model = []  # данные для 3D модели
#     for i in range(4):
#         # x это координаты по ширине
#         x_new = x_extended[i].reshape(1, -1)[0]
#         x_old = x[i].reshape(1, -1)[0]
#         # Вычитаем чтобы все координаты по x находились в интервале [0, 1]
#         if i == 1:
#             x_old -= breadth
#             x_new -= breadth
#         elif i == 2:
#             x_old -= (breadth + depth)
#             x_new -= (breadth + depth)
#         elif i == 3:
#             x_old -= (2 * breadth + depth)
#             x_new -= (2 * breadth + depth)
#         # z это координаты по высоте
#         z_new = z_extended[i].reshape(1, -1)[0]
#         z_old = z[i].reshape(1, -1)[0]
#
#         data_old = pressure_coefficients[i].reshape(1, -1)[0]
#         # data_old_integer.append(data_old)
#         coords = [[i1, j1] for i1, j1 in zip(x_old, z_old)]  # Старые координаты
#         # Интерполятор полученный на основе имеющихся данных
#         interpolator = Artist.interpolator(coords, data_old)
#         integral_func.append(interpolator)
#         # Получаем данные для несуществующих датчиков
#         data_new = [float(interpolator([[X, Y]])) for X, Y in zip(x_new, z_new)]
#
#         triang = mtri.Triangulation(x_new, z_new)
#         refiner = mtri.UniformTriRefiner(triang)
#         grid, value = refiner.refine_field(data_new, subdiv=4)
#         # data_for_3d_model.append((grid, value))
#         data_colorbar = graph[i].tricontourf(grid, value, cmap=cmap, levels=levels, extend='both')
#         aq = graph[i].tricontour(grid, value, linewidths=1, linestyles='solid', colors='black', levels=levels)
#         X, Y = np.meshgrid(x_old, z_old)
#         graph[i].plot(X, Y, '.k', **dict(markersize=3.7))
#         graph[i].clabel(aq, fontsize=10)
#
#         graph[i].set_ylim([0, height])
#         if breadth == depth == height:
#             graph[i].set_aspect('equal')
#         if i in [0, 2]:
#             graph[i].set_xlim([0, breadth])
#             graph[i].set_xticks(ticks=np.arange(0, breadth + 0.1, 0.1), fontsize=10)
#         else:
#             graph[i].set_xlim([0, depth])
#             graph[i].set_xticks(ticks=np.arange(0, depth + 0.1, 0.1), fontsize=10)
#         ret_int.append(interpolator)
#         # graph[i].axis('off')
#     fig.colorbar(data_colorbar, ax=graph, location='bottom', cmap=cmap, ticks=levels).ax.tick_params(labelsize=7)
#
#     # plt.savefig(f'{folder_out}\\Изополя {model_name}_{alpha} {mode}')
#     plt.savefig(
#         f'{os.getcwd()}\\Отчет {model_name}_{alpha}\\Изополя ветровых нагрузок и воздействий\\Непрерывные\\Изополя {model_name}_{alpha}_{angle:02} {mode}',
#         bbox_inches='tight')
#     # plt.show()
#
#     # isofield_model().show(data_for_3d_model, levels)
#     # return ret_int
#     # # Численное интегрирование
#     #
#     # if integral[0] == -1:
#     #     return None
#     # elif integral[0] == 0:
#     #     count_zone = count_row
#     # else:
#     #     count_zone = integral[0]
#     # ran = random.uniform
#     # face1, face2, face3, face4 = [], [], [], []
#     # model = face1, face2, face3, face4
#     # count_point = integral[1]
#     # step_floor = height / count_row
#     # for i in range(4):
#     #     top = step_floor
#     #     down = 0
#     #     for _ in range(count_zone):
#     #         floor = []
#     #         if i in [0, 2]:
#     #             for _ in range(count_point):
#     #                 floor.append(integral_func[i]([[ran(0, breadth), ran(down, top)]]))
#     #         else:
#     #             for _ in range(count_point):
#     #                 floor.append(integral_func[i]([[ran(0, depth), ran(down, top)]]))
#     #         down = top
#     #         top += step_floor
#     #         # Обычный метод Монте-Синякина
#     #         if i in [0, 2]:
#     #             model[i].append(sum(floor) * breadth / count_point)
#     #         else:
#     #             model[i].append(sum(floor) * depth / count_point)
#     # print(face1)
#     # print(face2)
#     # print(face3)
#     # print(face4)
#
# @staticmethod
# def func(x):
#     return 15 * x ** 3 + 21 * x ** 2 + 41 * x + 3 * np.sin(x) * np.cos(x)
#
# @staticmethod
# def check():
#     arr = []
#     a = 0
#     b = 5
#     n = 2000000
#     for i in range(n):
#         arr.append(Artist.func(random.uniform(a, b)))
#
#     print(sum(arr) * (b - a) / n)
#
# @staticmethod
# def signal(pressure_coefficients, pressure_coefficients1, alpha, model_name, angle):
#     time = [i / 1000 for i in range(5001)]
#     plt.plot(time, [i[0] for i in pressure_coefficients[:5001]], label='113')
#     plt.plot(time, [i[0] for i in pressure_coefficients1[:5001]], label='115')
#     plt.legend()
#     plt.show()
#
# @staticmethod
# def signal1(pressure_coefficients, pressure_coefficients_2):
#     time = [i / 1000 for i in range(len(pressure_coefficients))]
#     plt.plot(time, [i[206] for i in pressure_coefficients], label='old')
#     plt.plot(time, pressure_coefficients_2, label='new')
#     plt.legend()
#     plt.show()
#
# @staticmethod
# def spectrum(pressure_coefficients, pressure_coefficients1, alpha, model_name, angle):
#     N = len(pressure_coefficients)
#     yf = (1 / N) * (np.abs(fft([i[0] for i in pressure_coefficients])))[1:N // 2]
#     yf1 = (1 / N) * (np.abs(fft([i[0] for i in pressure_coefficients1])))[1:N // 2]
#     FD = 1000
#     xf = rfftfreq(N, 1 / FD)[1:N // 2]
#     xf = xf[1:]
#     yf = yf[1:]
#     yf1 = yf1[1:]
#     plt.plot(xf[:200], yf[:200], antialiased=True, label='113')
#     plt.plot(xf[:200], yf1[:200], antialiased=True, label='115')
#     plt.legend()
#     plt.show()
