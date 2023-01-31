<<<<<<< HEAD
import asyncio
import numpy as np
import matplotlib.cm as cm
import matplotlib.tri as mtri
import matplotlib.pyplot as plt
from scipy.fft import fft, rfftfreq
from scipy.signal import argrelextrema, welch
from matplotlib.colors import Normalize
from matplotlib.axis import rcParams
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import MultipleLocator, ScalarFormatter
import threading

from utils.utils import interpolator as intp
=======
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import numpy as np
import scipy.interpolate
from matplotlib.colors import Normalize
from scipy.signal import welch
>>>>>>> b28311c22755b3806c0ef8b7874d2ae258d6b3ea


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

    dpi = 200
    clear = False

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

        fig, ax = plt.subplots(1, 4, num=1, dpi=Plot.dpi, clear=True)
        cmap = cm.get_cmap(name="jet")
        min_v = np.min(pressure_coefficients)
        max_v = np.max(pressure_coefficients)
        normalizer = Normalize(min_v, max_v)
        im = cm.ScalarMappable(norm=normalizer, cmap=cmap)
        for i in range(4):
            data_old = pressure_coefficients[i]
            ax[i].pcolormesh(np.flip(data_old, axis=0), cmap=cmap, norm=normalizer)
            ax[i].set_yticks(np.arange(0, count_row + 1, 2.5))
            ax[i].set_yticklabels(labels=np.arange(0, height + 0.01, 0.05).round(2), fontsize=5)
            if i in [0, 2]:
                ax[i].set_xticks([i for i in range(0, count_sensors_on_middle + 1, 5)])
                ax[i].set_xticklabels(labels=list(map(str, np.arange(0, breadth + 0.1, 0.1))), fontsize=5)
            else:
                ax[i].set_xticks([i for i in range(0, count_sensors_on_side + 1, 5)])
                ax[i].set_xticklabels(labels=list(map(str, np.arange(0, depth + 0.1, 0.1))), fontsize=5)
            x, y = np.meshgrid(np.arange(0.5, count_sensors_on_middle + 0.5, 1), z * count_row / height)
            ax[i].plot(x, y, '.k')

        if breadth == depth == height:
            [ax[i].set_aspect('equal') for i in range(4)]
        ticks = np.arange(np.round(min_v, 1), np.round(max_v, 1) + 0.1, 0.1)
        fig.colorbar(im, ax=ax, location='bottom', cmap=cmap, ticks=ticks).ax.tick_params(labelsize=4)

        return fig

    @staticmethod
    def integral_isofield(model_name, model_size, scale_factors, alpha, mode, pressure_coefficients, coordinates):
        """Отрисовка интегральных изополей"""
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
        z_extended = [np.array(z1), np.array(z2), np.array(z3), np.array(z4)]
        x_extended = [np.array(x1), np.array(x2), np.array(x3), np.array(x4)]

        fig, ax = plt.subplots(1, 4, num=1, dpi=Plot.dpi, clear=True)

        cmap = cm.get_cmap(name="jet")
        data_colorbar = None

        h_scaled = height * z_scale_factor
        b_scaled = breadth * x_scale_factor
        d_scaled = depth * y_scale_factor

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
            # data_old_integer.append(data_old)
            coords = [[i1, j1] for i1, j1 in zip(x_old, z_old)]  # Старые координаты
            # Интерполятор полученный на основе имеющихся данных
            interpolator = intp(coords, data_old)

            # Получаем данные для несуществующих датчиков
            data_new = [float(interpolator([[X, Y]])) for X, Y in zip(x_new, z_new)]

            triang = mtri.Triangulation(x_new, z_new)
            refiner = mtri.UniformTriRefiner(triang)
            grid, value = refiner.refine_field(data_new, subdiv=4)
            data_colorbar = ax[i].tricontourf(grid, value, cmap=cmap, levels=levels, extend='both')
            aq = ax[i].tricontour(grid, value, linewidths=1, linestyles='solid', colors='black', levels=levels)
            x_dots, y_dots = np.meshgrid(x_old, z_old)
            ax[i].plot(x_dots, y_dots, '.k', **dict(markersize=3.7))

            ax[i].clabel(aq, fontsize=10)
            if breadth == depth == height:
                ax[i].set_aspect('equal')

            ax[i].set_ylim([0, h_scaled])
            ax[i].set_yticks(np.arange(0, h_scaled + h_scaled * 0.01, h_scaled * 0.2))
            ax[i].set_yticklabels(labels=np.arange(0, size_z + size_z * 0.01, size_z * 0.2).round(2), fontsize=5)

            if i in [0, 2]:
                ax[i].set_xlim([0, b_scaled])
                ax[i].set_xticks(ticks=np.arange(0, b_scaled + b_scaled * 0.01, b_scaled * 0.2))
                ax[i].set_xticklabels(labels=np.arange(0, size_x + size_x * 0.01, size_x * 0.2).round(2), fontsize=5)

            else:
                ax[i].set_xlim([0, d_scaled])
                ax[i].set_xticks(ticks=np.arange(0, d_scaled + d_scaled * 0.01, d_scaled * 0.2))
                ax[i].set_xticklabels(labels=np.arange(0, size_y + size_y * 0.01, size_y * 0.2).round(2), fontsize=5)

        fig.colorbar(data_colorbar, ax=ax, location='bottom', cmap=cmap, ticks=levels).ax.tick_params(labelsize=4)

        return fig

    @staticmethod
    def welch_graphs(data):
        fig, ax = plt.subplots(dpi=Plot.dpi, num=1, clear=True)

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

    @staticmethod
    def summary_coefficients(data):
        """Графики суммарных аэродинамических коэффициентов в декартовой системе координат
        data = {name:array,
                ...
                }
        """
        fig, ax = plt.subplots(dpi=Plot.dpi, num=1, clear=True)
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
    def scaling_data(x, y = None):
        """Масштабирование данных до 360 градусов"""

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

    @staticmethod
    def polar_plot(data, title):
        """Графики суммарных аэродинамических коэффициентов в полярной системе координат.
        data = {name:array,
                ...
                }
        """
        angles = np.array([angle for angle in range(0, 365, 5)]) * np.pi / 180.0
        fig, ax = plt.subplots(dpi=Plot.dpi, num=1, clear=True, subplot_kw={'projection': 'polar'})

        for name in data.keys():
            ax.plot(angles, data[name], label=name)

        ax.set_theta_direction(-1)
        ax.set_theta_zero_location('N')
        ax.set_thetagrids([i for i in range(0, 360, 15)])
        ax.legend(loc='upper right', fontsize=9)
        ax.set_title(title)

        return fig

    @staticmethod
    def model_pic(model_scale, coordinates):
        breadth, depth, height = int(model_scale[0]) / 10, int(model_scale[1]) / 10, int(model_scale[2]) / 10
        size_x = 2 * (breadth + depth)
        x, z = coordinates
        count_sensors = len(x)
        fig, ax = plt.subplots(figsize=(12, 6), num=1, dpi=Plot.dpi, clear=True)
        ax.set_title('Развертка датчиков по модели', fontweight='semibold', fontsize=8)
        ax.set_xlabel('Горизонтальная развертка /м', fontweight='semibold', fontsize=8)
        ax.set_ylabel('Высота модели /м', fontweight='semibold', fontsize=8)
        ax.set_ylim(0, height)
        ax.set_xlim(0, size_x)
        xtick_s = 0.05
        ytick_s = 0.02 if height in [0.1, 0.2] else 0.05
        xticks = np.arange(0, size_x + xtick_s, xtick_s)
        yticks = np.arange(0, height + ytick_s, ytick_s)
        xlabels = ['0'] + [str(i)[:4].rstrip('0') for i in xticks[1:]]
        ylabels = ['0'] + [str(i)[:4].rstrip('0') for i in yticks[1:]]
        ax.set_xticks(ticks=xticks)
        ax.set_xticklabels(labels=xlabels)
        ax.set_yticks(yticks)
        ax.set_yticklabels(labels=ylabels)
        xticks_minor = np.arange(0, size_x, 0.02)
        ax.set_xticks(xticks_minor, minor=True)
        ax.set_xticklabels(labels=xticks_minor, minor=True, fontsize=7)
        ax.tick_params(axis='x', which='minor', pad=5)
        ax.tick_params(axis='x', which='major', pad=10)

        for i in range(1, int(size_x * 10)):
            ax.plot([i / 10, i / 10], [0, height], linestyle='--', color='black')
        ax.plot(x, z, '+')
        for i, j, text in zip(x, z, [str(i) for i in range(1, count_sensors + 1)]):
            ax.text(i, j - 0.01, text, fontsize=8)
        ax.set_aspect('equal') if height == 0.1 else None

        return fig

    @staticmethod
    def model_polar(model_scale):
        plt.close()
        b_scale, d_scale, h_scale = int(model_scale[0]), int(model_scale[1]), int(model_scale[2])
        fig = plt.figure(figsize=(8, 8), num=1, dpi=Plot.dpi, clear=True)

        polar = fig.add_axes([0.1, 0.1, 0.8, 0.8], projection='polar')
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

        ax = fig.add_subplot(111, position=[0.1, 0.1, 0.8, 0.8])
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
    def model_cube(model_scale):
        b_scale, d_scale, h_scale = int(model_scale[0]), int(model_scale[1]), int(model_scale[2])

        fig = plt.figure(figsize=(8, 8), num=1, dpi=Plot.dpi, clear=True)
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
    def envelopes(pressure_coefficients):
        # print(threading.get_ident())
        figs = []  # массив для графиков так как на 1 графике максимум 100 датчиков
        step_x = 20
        step_x_minor = 5
        step_y = 0.4

        step_sens = 100
        count_sensors_plot = len(pressure_coefficients[0])

        for q in range(0, count_sensors_plot, step_sens):
            fig, ax = plt.subplots(dpi=Plot.dpi, num=f'Огибающие датчики от {q + 1} до {q + step_sens + 1}', clear=True)
            ax.grid(visible=True, which='minor', color='black', linestyle='--')
            ax.grid(visible=True, which='major', color='black', linewidth=1.5)

            coefficients = pressure_coefficients.T[q:q + step_sens].T
            mean_pr = np.mean(coefficients, axis=0).round(4)
            rms_pr = np.array([np.sqrt(i.dot(i) / i.size) for i in coefficients.T]).round(4)
            std_pr = np.std(coefficients, axis=0).round(4)
            max_pr = np.max(coefficients, axis=0).round(4)
            min_pr = np.min(coefficients, axis=0).round(4)

            ox = [i for i in range(q + 1, q + step_sens + 1)]
            for i, j, c in zip((mean_pr, rms_pr, std_pr, max_pr, min_pr), ('MEAN', 'RMS', 'STD', 'MAX', 'MIN'),
                               ('b', 'g', 'r', 'c', 'y')):
                ax.plot(ox, i, '-', label=j, linewidth=3, color=c)

            ax.set_xlim([q + 1, q + step_sens])
            yticks = np.arange(np.min(min_pr) - step_y, np.max(max_pr) + step_y, step_y).round(2)
            ax.set_ylim(np.min(yticks) + 0.2, np.max(yticks) + 0.2)
            ax.set_yticks(yticks)

            ax.set_xticks([q + 1] + [i for i in range(q + step_x, q + step_sens + 1, 20)])
            # ax.xaxis.set_major_locator(MultipleLocator(20))
            ax.xaxis.set_minor_locator(MultipleLocator(5))
            ax.xaxis.set_minor_formatter(ScalarFormatter())
            ax.xaxis.set_tick_params(which='major', labelsize=10)
            ax.xaxis.set_tick_params(which='minor', labelsize=7)

            ax.legend(loc='upper right', fontsize=9)
            ax.set_title('Огибающие')

            figs.append(fig)

        return figs

    @staticmethod
    def old_welch_graphs(sum_cx, sum_cy, sum_cmz):
        fig, ax = plt.subplots(dpi=Plot.dpi, clear=True)
        ax.set_xlim([0, 15])
        ax.grid()
        ax.set_xlabel('Sh')
        ax.set_ylabel('PSD, V**2/Hz')
        ax.set_xlim([10 ** -2, 10 ** 3])
        ax.set_xscale('log')
        ax.set_yscale('log')

        for data, name in zip((sum_cx, sum_cy, sum_cmz), ('Cx', 'Cy', 'CMz')):
            temp, psd = welch(data, fs=1000, nperseg=int(32768 / 5))
            ax.plot(temp, psd, label=name)

        ax.legend(loc='upper right', fontsize=9)

        return fig
