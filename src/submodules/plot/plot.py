from typing import Any

import matplotlib
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import BoundaryNorm
from matplotlib.ticker import MultipleLocator, ScalarFormatter, MaxNLocator
from pydantic import validate_call
from scipy.signal import welch

from src.common.DbType import DbType
from src.common.annotation import CoordinatesType, ChartModeType, ModelNameIsolatedType
from src.submodules.plot.utils import get_labels, round_list_model_pic
from src.submodules.utils import utils
from src.submodules.utils.data_features import lambdas
from src.submodules.utils.scaling import get_model_and_scale_factors
from src.ui.common.ChartMode import ChartMode


class Plot:
    DPI = 50

    @staticmethod
    @validate_call
    def envelopes(
            pressure_coefficients,
            parameters: ChartModeType,
            step_major_x: int = 20,
            step_minor_x: int = 5,
            step_major_y: float = 0.4,
            step_sensors_for_plot: int = 100
    ) -> plt.Figure:
        """
        Отрисовка огибающих.

        Parameters:
        - pressure_coefficients: np.ndarray
            Матрица коэффициентов давления.
        - parameters: tuple
            Параметры графика.
        - step_major_x: int
            Шаг для основных отметок по оси X.
        - step_minor_x: int
            Шаг для второстепенных отметок по оси X.
        - step_major_y: float
            Шаг для основных отметок по оси Y.
        - step_sensors_for_plot: int
            Количество датчиков на одном графике.

        Returns:
        - figs: list
            Список объектов matplotlib.figure.Figure с графиками.
        """

        colors = ('b', 'g', 'r', 'c', 'y')
        # an array for plots because there are 100 sensors on 1 plot
        figs = []
        # total number of sensors
        total_count_of_sensors = len(pressure_coefficients[0])

        pressure_coefficients = pressure_coefficients.T
        sensors = np.arange(0, total_count_of_sensors, step_sensors_for_plot)

        for sensor_index_start in sensors:
            fig, ax = plt.subplots(dpi=Plot.DPI)
            ax.grid(visible=True, which='minor', color='black', linestyle='--')
            ax.grid(visible=True, which='major', color='black', linewidth=1.5)

            coefficients = pressure_coefficients[sensor_index_start:sensor_index_start + step_sensors_for_plot].T
            data_for_plot = [lambdas[mode](coefficients) for mode in parameters]

            count_sensors_ox = len(data_for_plot[0]) if sensor_index_start == sensors[-1] else step_sensors_for_plot

            ox = np.arange(sensor_index_start + 1, sensor_index_start + count_sensors_ox + 1)

            for i, j, c in zip(data_for_plot, parameters, colors):
                ax.plot(ox, i, '-', label=j, linewidth=3, color=c)

            ax.set_xlim([sensor_index_start + 1, sensor_index_start + count_sensors_ox])

            # increasing the borders by 10 percent
            start_yticks = np.min(data_for_plot) * 1.1
            stop_yticks = np.max(data_for_plot) * 1.1
            yticks = np.arange(start_yticks, stop_yticks, step_major_y).round(2)
            ax.set_yticks(yticks)

            start_xticks = sensor_index_start + step_major_x
            stop_xticks = sensor_index_start + count_sensors_ox + 1
            xticks = np.arange(start_xticks, stop_xticks, step_major_x)
            xticks = np.insert(xticks, 0, sensor_index_start + 1)
            ax.set_xticks(xticks)

            ax.xaxis.set_minor_locator(MultipleLocator(step_minor_x))
            ax.xaxis.set_minor_formatter(ScalarFormatter())
            ax.xaxis.set_tick_params(which='major', labelsize=10)
            ax.xaxis.set_tick_params(which='minor', labelsize=7)

            ax.legend(loc='upper right', fontsize=9)
            ax.set_title('Огибающие')
            ax.set_xlabel('Номер датчика')
            ax.set_ylabel('Аэродинамический коэффициент')

            figs.append(fig)

        return figs

    @staticmethod
    @validate_call
    def pseudocolor_coefficients(
            pressure_coefficients,
            coordinates: CoordinatesType,
            model_name: ModelNameIsolatedType,
            parameter: ChartMode,
    ) -> plt.Figure:
        """
        Отрисовка дискретных изополей.

        Parameters:
        - pressure_coefficients: np.ndarray
            Матрица коэффициентов давления.
        - coordinates: CoordinatesType
            Координаты точек модели.
        - model_name: ModelNameIsolatedType
            Название модели.
        - parameter: ChartMode
            Параметр для визуализации.

        Returns:
        - plt.Figure: Сгенерированный график.
        """
        size, count_sensors = utils.get_size_and_count_sensors(pressure_coefficients.shape[1],
                                                               model_name,
                                                               )

        pressure_coefficients = lambdas[parameter](pressure_coefficients)

        breadth, depth, height = size
        count_sensors_on_model, count_sensors_on_middle_row, count_sensors_on_side_row = count_sensors

        count_row = count_sensors_on_model // (2 * (count_sensors_on_middle_row + count_sensors_on_side_row))

        # can return array different size
        pressure_coefficients = aot_integration.split_1d_array(
            count_row,
            count_sensors_on_middle_row,
            count_sensors_on_side_row,
            pressure_coefficients
        )

        _, z = coordinates

        z = np.array(z[::2 * (count_sensors_on_middle_row + count_sensors_on_side_row)])[::-1]

        fig, ax = plt.subplots(1, 4, dpi=Plot.DPI)

        cmap = matplotlib.colormaps.get_cmap("jet")

        # use a generator, there can be arrays of different sizes
        # the 111 model has all arrays of the same size
        # but for example 215 212 315 and so on is not
        min_v = np.min([np.min(pressure_coefficients[i]) for i in range(4)])
        max_v = np.max([np.max(pressure_coefficients[i]) for i in range(4)])

        ticks = np.arange(np.round(min_v, 1), np.round(max_v, 1) + 0.2, 0.1)
        levels = MaxNLocator(len(ticks)).tick_values(min_v, max_v)
        norm = BoundaryNorm(levels, ncolors=cmap.N, clip=True)

        xticks_middle = np.arange(0, count_sensors_on_middle_row + 1, 5)
        xticklabels_middle = get_labels(breadth)

        xticks_side = np.arange(0, count_sensors_on_side_row + 1, 5)
        xticklabels_side = get_labels(depth)

        ytickslabels = np.arange(0, height + 0.01, 0.05).round(2)
        yticks = np.linspace(0, count_row, ytickslabels.size)

        xm, ym = np.meshgrid(np.arange(0.5, count_sensors_on_middle_row + 0.5, 1), z * count_row / height)
        xs, ys = np.meshgrid(np.arange(0.5, count_sensors_on_side_row + 0.5, 1), z * count_row / height)

        for i in range(4):
            pr_coefficients = np.flip(pressure_coefficients[i], axis=0)

            im = ax[i].pcolormesh(pr_coefficients, cmap=cmap, norm=norm)
            ax[i].set_yticks(yticks, labels=ytickslabels)

            if i in [0, 2]:
                ax[i].set_xticks(xticks_middle, labels=xticklabels_middle)
                ax[i].plot(xm, ym, '.k')

            else:
                ax[i].set_xticks(xticks_side, labels=xticklabels_side)
                ax[i].plot(xs, ys, '.k')

        if breadth == depth == height:
            for i in range(4):
                ax[i].set_aspect('equal')

        fig.colorbar(im, ax=ax, location='bottom', ticks=ticks)

        return fig

    @staticmethod
    @validate_call
    def summary_coefficients(
            pressure_coefficients: dict[str, Any],
            db: DbType,
    ) -> plt.Figure:
        """
        Построение графиков суммарных аэродинамических коэффициентов в декартовой системе координат.

        Args:
            pressure_coefficients (Dict[str, np.array]): Словарь с коэффициентами, где ключ — тип графика,
                                                          а значение — массив данных.
            db (DbType): Тип базы данных.

        Returns:
            plt.Figure: График в формате Matplotlib.
        """
        fig, ax = plt.subplots(dpi=Plot.DPI)

        match db:
            case DbType.ISOLATED:
                # breadth, depth, height = map(float, model_size)
                # breadth_tpu, depth_tpu, height_tpu = map(int, list(model_scale))

                # match alpha:
                #     case 4:
                #         speed_sp = speed_sp_b(height)
                #         speed_tpu = interp_025_tpu(height)
                #     case 6:
                #         speed_sp = speed_sp_a(height)
                #         speed_tpu = interp_016_tpu(height)
                #
                # l_m = aot_integration.calculate_projection_on_the_axis(breadth, depth, angle)
                # l_tpu = aot_integration.calculate_projection_on_the_axis(breadth_tpu, depth_tpu, angle)

                # kv = speed_sp / speed_tpu
                # km = l_m / l_tpu

                # kt = km / kv

                # ax.set_xlim(0, 32.768 * kt)
                # ox = np.linspace(0, 32.768 * kt, 32768)

                ax.set_xlim(0, 32.768)
                ox = np.linspace(0, 32.768, 32768)

            case DbType.INTERFERENCE:
                ax.set_xlim(0, 7.5)
                ox = np.linspace(0, 7.5, 5858)

        ax.grid()
        ax.set_ylabel('Суммарные аэродинамические коэффициенты')
        ax.set_xlabel('Время, с', labelpad=.3)

        for name in pressure_coefficients.keys():
            if pressure_coefficients[name] is not None:
                ax.plot(ox, pressure_coefficients[name], label=name)
        ax.legend(loc='upper right', fontsize=9)

        return fig

    @staticmethod
    @validate_call
    def polar_plot(
            data: dict[str, Any],
            title: ChartMode,
    ) -> plt.Figure:
        """
        Построение графиков суммарных аэродинамических коэффициентов в полярной системе координат.

        Args:
            data (Dict[str, np.array]): Словарь с коэффициентами, где ключ — тип графика,
                                        а значение — данные для графика.
            title (str): Заголовок графика.

        Returns:
            plt.Figure: Объект графика.
        """

        angles = np.arange(0, 365, 5) * np.pi / 180.0

        fig, ax = plt.subplots(dpi=Plot.DPI, subplot_kw={'projection': 'polar'})

        for name in data.keys():
            ax.plot(angles, data[name], label=name)
        # Обратное направление
        ax.set_theta_direction(-1)
        # "Север" — нулевая позиция
        ax.set_theta_zero_location('N')
        # Шаг 15° для сетки
        ax.set_thetagrids(np.arange(0, 360, 15))
        ax.legend(loc='upper right', fontsize=9)
        ax.set_title(title)

        ylim = ax.get_ylim()
        ax.set_ylim(ylim)

        mean = (ylim[1] + ylim[0]) / 2

        ax.annotate("",
                    xy=(angles[0], mean),
                    xytext=(ylim[0], ylim[0]),
                    arrowprops=dict(arrowstyle="->", linewidth=2))
        ax.annotate("Y",
                    xy=(angles[0], mean))

        ax.annotate("",
                    xy=(angles[18], mean),
                    xytext=(ylim[0], ylim[0]),
                    arrowprops=dict(arrowstyle="->", linewidth=2))
        ax.annotate("X",
                    xy=(angles[18], mean))

        return fig

    # TODO не помню как должно быть правильно, подправить
    @staticmethod
    @validate_call
    def welch_graphs(db='isolated', **kwargs):
        """Отрисовка графиков спектральной плотности мощности"""
        data = kwargs['data']

        model_size = kwargs['model_size']
        breadth, depth, height = map(float, model_size)
        angle = int(kwargs['angle'])
        model_scale = kwargs['model_scale']
        print(model_scale)
        print(model_size)

        b_s, d_s, h_s = [int(i) / 10 for i in model_scale]
        h_s = height / min(breadth, depth, height) / 10
        print(h_s)
        print(height / h_s)
        ks = height / h_s
        if db == 'isolated':
            alpha = kwargs['alpha']
            fs = 1000
            counts = 32768

            if alpha == '4':
                # speed_sp_s = speed_sp_b(h_s, scale_ks=1/400)
                # speed_tpu_s = interp_025_tpu(h_s)
                #
                # speed_sp = speed_sp_b_m(height)
                # # speed_tpu = interp_025_tpu_400(height)
                # speed_tpu = scipy.interpolate.interp1d(y_016 * height/h_s, x_016)(height)
                speed_tpu_m_ist_height = interp_025_tpu_400(height)
                speed_sp = speed_sp_b_m(height)

            elif alpha == '6':
                speed_sp_s = speed_sp_a(h_s)
                speed_tpu_s = interp_016_tpu(h_s)

                speed_sp = speed_sp_a_m(height)
                speed_tpu = interp_016_tpu_400(height)

            l_m = breadth * np.cos(np.deg2rad(angle)) + depth * np.sin(np.deg2rad(angle))

            print(speed_sp, speed_tpu_m_ist_height)
            kv1 = speed_sp / speed_tpu_m_ist_height

            # print(speed_sp, speed_tpu)
            #
            # kv1 = speed_sp_s / speed_tpu_s
            # kv2 = speed_sp / speed_tpu

            # print(kv1, kv2)

            sh = lambda f: f * l_m / speed_tpu
            # sh = lambda f: f * l_m / speed_sp

        elif db == 'interference':
            case = kwargs['case']
            fs = 781
            counts = 5858

        fig, ax = plt.subplots(dpi=Plot.DPI)

        # ax.set_xlim([10 ** -1, 10 ** 3])
        # ax.set_xlim([10 ** -1, 10 ** 2])
        # ax.set_xlim([10 ** -1, 0.5])
        ax.set_xlim([10 ** -1, 2])
        # ax.set_xscale('log')
        # ax.set_yscale('log')

        ax.grid()
        # ax.set_xlabel('Frequency')
        ax.set_xlabel('Sh')
        ax.set_ylabel('PSD, V**2/Hz')

        for name in data.keys():
            if data[name] is not None:
                freq, psd = welch(data[name], fs=fs, nperseg=int(counts / 5))
                ax.plot([sh(f) for f in freq], psd, label=name)
                # print(freq)
                # print(psd)
                # print(psd.max())
                # print(np.where(psd == psd.max()))
                # print(temp[np.where(psd == psd.max())])

        ax.legend(loc='upper right', fontsize=9)

        return fig

    @staticmethod
    # @validate_call
    def isofields_coefficients(
            db,
            model_name,
            model_size,
            scale_factors,
            parameter,
            pressure_coefficients,
            coordinates,
            pressure_plot_parameters,
            coefficient_for_pressure,
            model_scale
    ):
        """Отрисовка интегральных изополей"""
        count_sensors_on_model = pressure_coefficients.shape[1]
        # size_x, size_y, size_z = map(float, model_size)
        # x_scale_factor, y_scale_factor, z_scale_factor = scale_factors

        match db:
            case DbType.ISOLATED:
                size, count_sensors = utils.get_size_and_count_sensors(pressure_coefficients.shape[1],
                                                                       model_name,
                                                                       )

                breadth, depth, height = size
                count_sensors_on_model, count_sensors_on_middle_row, count_sensors_on_side_row = count_sensors

            case DbType.INTERFERENCE:
                height = model_scale / 1000
                breadth, depth = 0.07, 0.07
                count_sensors_on_middle_row = 7
                count_sensors_on_side_row = 7

        # pressure_coefficients = lambdas[parameter](pressure_coefficients)

        x, z = coordinates

        # pressure_coefficients = aot_integration.split_1d_array(
        #     count_sensors_on_model,
        #     count_sensors_on_middle_row,
        #     count_sensors_on_side_row,
        #     pressure_coefficients
        # )

        x = aot_integration.split_1d_array(
            count_sensors_on_model,
            count_sensors_on_middle_row,
            count_sensors_on_side_row,
            np.array(x)
        )
        z = aot_integration.split_1d_array(
            count_sensors_on_model,
            count_sensors_on_middle_row,
            count_sensors_on_side_row,
            np.array(z)
        )
        result = np.insert(x, 0, 0, axis=2)
        result = np.append(result, np.zeros((result.shape[0], result.shape[1], 1), 123), axis=2)

        first_elements = result[:, 0:1]  # Берём первый элемент каждой строки

        array_with_first = np.concatenate((first_elements, result), axis=1)
        array_with_first = np.concatenate((array_with_first, first_elements), axis=1)
        print(array_with_first)

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
        print(x_extended)
        # # fig, ax = plt.subplots(1, 4, num=num_fig, dpi=Plot.dpi, clear=True)
        # sum_size = size_x + size_y + size_z
        #
        # breadth_ratios = size_x / sum_size
        # depth_ratios = size_y / sum_size
        # height_ratios = size_z / sum_size
        #
        # fig = plt.figure(dpi=Plot.DPI)
        # grid = plt.GridSpec(1, 4,
        #                     width_ratios=[breadth_ratios, depth_ratios, breadth_ratios, depth_ratios],
        #                     height_ratios=[height_ratios])
        #
        # ax = [fig.add_subplot(grid[0, i]) for i in range(4)]
        #
        # cmap = matplotlib.colormaps.get_cmap("jet")
        # data_colorbar = None
        #
        # h_scaled = height * z_scale_factor
        # b_scaled = breadth * x_scale_factor
        # d_scaled = depth * y_scale_factor
        #
        # x_new_list = []
        # z_new_list = []
        #
        # x_old_list = []
        # z_old_list = []
        #
        # data_new_list = []
        #
        # if pressure_plot_parameters:
        #     type_area = pressure_plot_parameters['type_area']
        #     wind_region = pressure_plot_parameters['wind_region']
        #     alpha_standard = alpha_standards[type_area]
        #     k10 = ks10[type_area]
        #     wind_pressure = wind_regions[wind_region]
        #
        #     PO = 1.225
        #     yf = 1.4
        #     ks = 1
        #     w0 = wind_pressure * 1000
        #     u10 = np.sqrt(2 * yf * k10 * w0 / PO)
        #     wind_profile = u10 * (ks * size_z / 10) ** alpha_standard
        #     coefficient_for_pressure = wind_profile ** 2 * PO / 2
        #
        # for i in range(4):
        #     x_new = x_extended[i].reshape(1, -1)[0]
        #     x_old = x[i].reshape(1, -1)[0]
        #
        #     z_new = z_extended[i].reshape(1, -1)[0]
        #     z_old = z[i].reshape(1, -1)[0]
        #     # Вычитаем чтобы все координаты по x находились в интервале [0, 1]
        #     if i == 1:
        #         x_old -= breadth
        #         x_new -= breadth
        #     elif i == 2:
        #         x_old -= (breadth + depth)
        #         x_new -= (breadth + depth)
        #     elif i == 3:
        #         x_old -= (2 * breadth + depth)
        #         x_new -= (2 * breadth + depth)
        #
        #     # Масштабирование координат
        #     z_new = z_new * z_scale_factor
        #     z_old = z_old * z_scale_factor
        #
        #     if i in [0, 2]:
        #         x_new = x_new * x_scale_factor
        #         x_old = x_old * x_scale_factor
        #
        #     else:
        #         x_new = x_new * y_scale_factor
        #         x_old = x_old * y_scale_factor
        #
        #     data_old = pressure_coefficients[i].reshape(1, -1)[0]
        #
        #     if pressure_plot_parameters:
        #         data_old = [coefficient_for_pressure * coefficient for coefficient in data_old]
        #
        #     # data_old_integer.append(data_old)
        #     coords = [[i1, j1] for i1, j1 in zip(x_old, z_old)]  # Старые координаты
        #     # Интерполятор полученный на основе имеющихся данных
        #     interpolator = intp(coords, data_old)
        #
        #     # Получаем данные для несуществующих датчиков
        #     data_new = [float(interpolator([[X, Y]])) for X, Y in zip(x_new, z_new)]
        #
        #     x_new_list.append(x_new)
        #     z_new_list.append(z_new)
        #
        #     x_old_list.append(x_old)
        #     z_old_list.append(z_old)
        #
        #     data_new_list.append(data_new)
        #     # data_old_list.append(data_old)
        #
        # # Уровни для изополей давления
        # if pressure_plot_parameters:
        #     min_0_2 = np.min(np.array([data_new_list[0], data_new_list[2]]))
        #     min_1_3 = np.min(np.array([data_new_list[1], data_new_list[3]]))
        #     min_v = np.min(np.array(min_0_2, min_1_3))
        #
        #     max_0_2 = np.max(np.array([data_new_list[0], data_new_list[2]]))
        #     max_1_3 = np.max(np.array([data_new_list[1], data_new_list[3]]))
        #     max_v = np.max(np.array(max_0_2, max_1_3))
        #
        #     all_levels = np.arange(min_v, max_v, (np.abs(min_v) + np.abs(max_v)) * 0.1)
        #     all_levels = all_levels.astype(np.int32, copy=False)
        #
        # if db == 'isolated' and not pressure_plot_parameters:
        #     all_levels = np.linspace(np.min([np.min(data_new_list[i]) for i in range(4)]),
        #                              np.max([np.max(data_new_list[i]) for i in range(4)]), 11)
        #     step = 0.1
        #     LEVELS = np.arange(-1.2, 1.2 + step, step)
        #     all_levels = LEVELS
        # elif db == 'interference' and not pressure_plot_parameters:
        #     if mode in ('max', 'min'):
        #         all_levels = np.arange(-6, 6.2, .2)
        #     elif mode == 'mean':
        #         all_levels = np.arange(-2, 2.1, .1)
        #     elif mode in ('rms', 'std'):
        #         all_levels = np.arange(-1, 1.1, .1)
        #
        # for i in range(4):
        #     triang = mtri.Triangulation(x_new_list[i], z_new_list[i])
        #     refiner = mtri.UniformTriRefiner(triang)
        #     grid, value = refiner.refine_field(data_new_list[i], subdiv=4)
        #
        #     # min_value = np.min(data_new_list[i])
        #     # max_value = np.max(data_new_list[i])
        #     # temp_levels = np.linspace(min_value, max_value, 11)
        #     # print(all_levels)
        #     # data_colorbar = ax[i].tricontourf(grid, value, cmap=cmap, extend='both', levels=11)
        #     # print(data_colorbar.levels)
        #     data_colorbar = ax[i].tricontourf(grid, value, cmap=cmap, extend='both', levels=all_levels)
        #     # data_colorbar = ax[i].tricontourf(grid, value, cmap=cmap, extend='both', levels=temp_levels)
        #     # data_colorbar = ax[i].tricontourf(grid, value, cmap=cmap, extend='both', levels=levels)
        #
        #     # aq = ax[i].tricontour(grid, value, linewidths=1, linestyles='solid', colors='black', levels=11)
        #     # print(aq.levels)
        #     aq = ax[i].tricontour(grid, value, linewidths=1, linestyles='solid', colors='black', levels=all_levels)
        #     # aq = ax[i].tricontour(grid, value, linewidths=1, linestyles='solid', colors='black', levels=temp_levels)
        #     # aq = ax[i].tricontour(grid, value, linewidths=1, linestyles='solid', colors='black', levels=levels)
        #
        #     if pressure_plot_parameters:
        #         ax[i].clabel(aq, fontsize=10, fmt='%i')
        #     else:
        #         ax[i].clabel(aq, fontsize=10, fmt='%.2f')
        #
        #     ax[i].set_ylim([0, h_scaled])
        #     ax[i].set_yticks(np.arange(0, h_scaled + h_scaled * 0.01, h_scaled * 0.2))
        #     ax[i].set_yticklabels(labels=np.arange(0, size_z + size_z * 0.01, size_z * 0.2).round(2), fontsize=5)
        #
        #     if i in [0, 2]:
        #         xlim = b_scaled
        #         label_b = size_x
        #     else:
        #         xlim = d_scaled
        #         label_b = size_y
        #
        #     ax[i].set_xlim([0, xlim])
        #     ax[i].set_xticks(ticks=np.arange(0, xlim + xlim * 0.01, xlim * 0.2))
        #     ax[i].set_xticklabels(labels=np.arange(0, label_b + label_b * 0.01, label_b * 0.2).round(2), fontsize=5)
        #
        #     if not pressure_plot_parameters:
        #         x_dots, y_dots = np.meshgrid(x_old_list[i], z_old_list[i])
        #         ax[i].plot(x_dots, y_dots, '.k', **dict(markersize=3))
        #
        # # print(all_levels)
        # # fig.colorbar(data_colorbar, ax=ax, location='bottom', cmap=cmap, ticks=levels).ax.tick_params(labelsize=4)
        # fig.colorbar(data_colorbar, ax=ax, location='bottom', cmap=cmap, ticks=all_levels).ax.tick_params(labelsize=7)
        #
        # return fig


if __name__ == "__main__":
    import asyncio

    import matplotlib.pyplot as plt
    from sqlalchemy import create_engine
    from src.submodules.databasetoolkit.isolated import (load_positions,
                                                         load_pressure_coefficients, find_experiment_by_model_name, )
    from compiled_aot.integration import aot_integration

    # engine = create_engine("postgresql://postgres:password@localhost:15432/postgres")
    engine = create_engine("postgresql://postgres:dSJJNjkn42384*$(#@92.246.143.110:5432/windspectrum_db")
    # engine = create_engine("postgresql://postgres:1234@localhost/postgres")

    angle = 0
    alpha = 4  # or 6
    model_name = 111
    # 42
    model_id = asyncio.run(find_experiment_by_model_name(model_name, alpha, engine)).model_id

    coordinates = asyncio.run(load_positions(model_id, alpha, engine))
    pressure_coefficients = asyncio.run(load_pressure_coefficients(model_id, alpha, engine, angle=angle))[angle]

    # size, count_sensors = utils.get_size_and_count_sensors(pressure_coefficients.shape[1],
    #                                                        model_name,
    #                                                        )

    # cx, cy = aot_integration.calculate_cx_cy(
    #     *count_sensors,
    #     *size,
    #     np.array(coordinates[0]),
    #     np.array(coordinates[1]),
    #     pressure_coefficients
    # )
    # cmz = aot_integration.calculate_cmz(
    #     *count_sensors,
    #     angle,
    #     *size,
    #     np.array(coordinates[0]),
    #     np.array(coordinates[1]),
    #     pressure_coefficients
    # )

    # Plot.envelopes(pressure_coefficients, (ChartMode.MEAN, ChartMode.RMS))

    # Plot.pseudocolor_coefficients(pressure_coefficients, coordinates, model_name, ChartMode.MEAN)

    # Plot.pseudocolor_coefficients(pressure_coefficients, coordinates, model_name, ChartMode.MEAN)

    # Plot.summary_coefficients({
    #     'Cx': cx,
    #     'Cy': cy,
    #     'CMz': cmz,
    # }, DbType.ISOLATED)

    # model_scale, scale_factors = get_model_and_scale_factors(*map(int, list(str(model_name))), 4)
    # model_scale_str = str(model_scale)
    #
    # if model_scale_str[0] == model_scale_str[1]:
    #     angle_border = 50
    # else:
    #     angle_border = 95
    #
    # mods = {
    #     ChartMode.MEAN: np.mean,
    #     ChartMode.RMS: rms,
    #     ChartMode.STD: np.std,
    #     ChartMode.MAX: np.max,
    #     ChartMode.MIN: np.min,
    #     ChartMode.SETTLEMENT: settlement,
    #     ChartMode.WARRANTY_PLUS: warranty_plus,
    #     ChartMode.WARRANTY_MINUS: warranty_minus,
    # }
    #
    # x = np.array(coordinates[0])
    # y = np.array(coordinates[1])
    #
    # cx_all = []
    # cy_all = []
    # cmz_all = []
    # mode = ChartMode.MEAN
    # for angle in range(0, angle_border, 5):
    #     pressure_coefficients = asyncio.run(load_pressure_coefficients(model_id, alpha, engine, angle=angle))[angle]
    #
    #     cx, cy = aot_integration.calculate_cx_cy(
    #         *count_sensors,
    #         *size,
    #         x,
    #         y,
    #         pressure_coefficients
    #     )
    #     cmz = aot_integration.calculate_cmz(
    #         *count_sensors,
    #         angle,
    #         *size,
    #         x,
    #         y,
    #         pressure_coefficients
    #     )
    #     cx_all.append(mods[mode](cx))
    #     cy_all.append(mods[mode](cy))
    #     cmz_all.append(mods[mode](cmz))
    #
    # cx_scale, cy_scale = scaling_data(cx_all, cy_all, angle_border=angle_border)
    # cmz_scale = scaling_data(cmz_all, angle_border=angle_border)
    #
    # data = {
    #     'Cx': cx_scale,
    #     'Cy': cy_scale,
    #     'CMz': cmz_scale,
    # }
    # Plot.polar_plot(data, mode)

    Plot.isofields_coefficients(DbType.ISOLATED,
                                model_name,
                                None,
                                None,
                                None,
                                pressure_coefficients,
                                coordinates,
                                None,
                                None,
                                None)

    plt.show()

    # import time
    #
    # s = time.time()
    # for i in range(19):
    # print(time.time() - s)
