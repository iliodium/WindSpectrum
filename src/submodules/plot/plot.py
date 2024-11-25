from typing import Any

import matplotlib
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import BoundaryNorm, Normalize
from matplotlib.ticker import MultipleLocator, ScalarFormatter, MaxNLocator
from pydantic import validate_call
from scipy.signal import welch
import matplotlib.tri as mtri
from src.common.DbType import DbType
from src.common.annotation import CoordinatesType, ChartModeType, ModelNameIsolatedType, ModelSizeType
from src.submodules.plot.utils import get_labels, round_list_model_pic
from src.submodules.utils import utils
from src.submodules.utils.data_features import lambdas
from src.submodules.utils.scaling import get_model_and_scale_factors
from src.ui.common.ChartMode import ChartMode
from src.submodules.plot.utils import interpolator as intp


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
            model_size: ModelSizeType,
            model_name: ModelNameIsolatedType,
            parameter: ChartMode,
            pressure_coefficients,
            coordinates: CoordinatesType
    ):
        """
        Отрисовка интегральных изополей.

        Args:
            model_size (ModelSizeType): Размеры модели
            model_name (ModelNameIsolatedType):
                Название модели.
            parameter (ChartMode):
                Параметр визуализации, определяющий способ обработки аэродинамических коэффициентов
            pressure_coefficients (np.ndarray):
                Массив аэродинамических коэффициентов
            coordinates (CoordinatesType):
                Координаты сенсоров модели, где:
                - Первый элемент: массив координат x,
                - Второй элемент: массив координат z.

        Returns:
            plt.Figure:
                Объект графика
        """
        size, count_sensors = utils.get_size_and_count_sensors(pressure_coefficients.shape[1],
                                                               model_name,
                                                               )

        breadth, depth, height = size
        count_sensors_on_model, count_sensors_on_middle_row, count_sensors_on_side_row = count_sensors

        pressure_coefficients = lambdas[parameter](pressure_coefficients)

        pressure_coefficients = aot_integration.split_1d_array(
            count_sensors_on_model,
            count_sensors_on_middle_row,
            count_sensors_on_side_row,
            pressure_coefficients
        )

        x = aot_integration.split_1d_array(
            count_sensors_on_model,
            count_sensors_on_middle_row,
            count_sensors_on_side_row,
            np.array(coordinates[0])
        )
        z = aot_integration.split_1d_array(
            count_sensors_on_model,
            count_sensors_on_middle_row,
            count_sensors_on_side_row,
            np.array(coordinates[1])
        )

        left_values = np.array([0, breadth, breadth + depth, 2 * breadth + depth])
        right_values = np.array([breadth, breadth + depth, 2 * breadth + depth, 2 * (breadth + depth)])

        x_extended = []
        for i in range(4):
            len_x = len(x[i])
            left_array = np.tile(left_values[i], (len_x, 1))
            right_array = np.tile(right_values[i], (len_x, 1))
            # добавляем 2 колонки, тк расширяем слева и справа
            result = np.column_stack((
                left_array,
                x[i],
                right_array
            ))
            # добавляем 2 строки
            result = np.vstack((
                result[0],
                result,
                result[0]
            ))
            x_extended.append(result)

        z_extended = []
        for i in range(4):
            first_row = z[i][:, 0].reshape(-1)
            # добавляем 2 колонки, тк расширяем слева и справа
            result = np.column_stack((
                first_row,
                first_row,
                z[i]
            ))
            len_z = len(result[0])
            # добавляем строку с нулями и строку с высотой модели
            result = np.vstack((
                np.tile(height, len_z),
                result,
                np.tile(0, len_z)
            ))
            z_extended.append(result)

        fig, ax = plt.subplots(1, 4, dpi=Plot.DPI)

        cmap = matplotlib.colormaps.get_cmap("jet")
        data_colorbar = None

        step = 0.1
        levels = np.arange(-1.2, 1.2 + step, step)

        np.linspace(1, 10, 9)

        count_ticks = 5

        for i in range(4):
            x_z = np.column_stack((x[i].reshape(-1), z[i].reshape(-1)))
            x_z_extended = np.column_stack((x_extended[i].reshape(-1), z_extended[i].reshape(-1)))
            # Интерполятор полученный на основе имеющихся данных
            interpolator = intp(x_z, pressure_coefficients[i].reshape(-1))

            # Получаем данные для несуществующих датчиков
            pressure_coefficients_extended = interpolator(x_z_extended)

            triang = mtri.Triangulation(x_extended[i].reshape(-1), z_extended[i].reshape(-1))
            refiner = mtri.UniformTriRefiner(triang)
            grid, value = refiner.refine_field(pressure_coefficients_extended, subdiv=4)

            data_colorbar = ax[i].tricontourf(grid, value, cmap=cmap, extend='both', levels=levels)
            # Рисуем линии
            labels = ax[i].tricontour(grid, value, linewidths=1, linestyles='solid', colors='black', levels=levels)
            # Подписываем линии
            ax[i].clabel(labels)

            x_start = x_extended[i][0][0]
            x_stop = x_extended[i][0][-1]

            ax[i].set_xticks(np.linspace(x_start, x_stop, count_ticks))
            ax[i].set_xticklabels(np.linspace(0, model_size[i % 2], count_ticks).round(2))

            z_start = 0
            z_stop = z_extended[i][0][0]

            ax[i].set_yticks(np.linspace(z_start, z_stop, count_ticks))
            ax[i].set_yticklabels(np.linspace(0, model_size[2], count_ticks).round(2))

        fig.colorbar(data_colorbar, ax=ax, location='bottom', cmap=cmap, ticks=levels)

        return fig

    @staticmethod
    @validate_call
    def pseudocolor_coefficients(
            model_size: ModelSizeType,
            model_name: ModelNameIsolatedType,
            parameter: ChartMode,
            pressure_coefficients
    ):
        """
        Отрисовка дискретных изополей.

        Args:
            model_size (ModelSizeType):
                Размеры модели
            model_name (ModelNameIsolatedType):
                Название модели.
            parameter (ChartMode):
                Параметр визуализации, определяющий способ обработки аэродинамических коэффициентов
            pressure_coefficients (np.ndarray):
                Массив аэродинамических коэффициентов

        Returns:
            plt.Figure:
                Объект графика
        """
        _, count_sensors = utils.get_size_and_count_sensors(pressure_coefficients.shape[1],
                                                            model_name,
                                                            )

        count_sensors_on_model, count_sensors_on_middle_row, count_sensors_on_side_row = count_sensors
        count_row = count_sensors_on_model // (2 * (count_sensors_on_middle_row + count_sensors_on_side_row))

        pressure_coefficients = lambdas[parameter](pressure_coefficients)
        pressure_coefficients = aot_integration.split_1d_array(
            count_sensors_on_model,
            count_sensors_on_middle_row,
            count_sensors_on_side_row,
            pressure_coefficients
        )

        fig, ax = plt.subplots(1, 4, dpi=Plot.DPI)

        cmap = matplotlib.colormaps.get_cmap("jet")

        step = 0.1
        levels = np.arange(-1.2, 1.2 + step, step)

        norm = BoundaryNorm(levels, ncolors=cmap.N, clip=True)

        z = np.arange(0, count_row, 1) + 0.5

        count_ticks = 5

        yticks = np.linspace(0, count_row, count_ticks)
        yticklabels = np.linspace(0, model_size[2], count_ticks).round(2)

        xticks_13 = np.linspace(0, count_sensors_on_middle_row, count_ticks)
        xticklabels_13 = np.linspace(0, model_size[0], count_ticks).round(2)

        xticks_24 = np.linspace(0, count_sensors_on_side_row, count_ticks)
        xticklabels_24 = np.linspace(0, model_size[1], count_ticks).round(2)

        meshgrid_13 = np.meshgrid(np.arange(0.5, count_sensors_on_middle_row + 0.5, 1), z)
        meshgrid_24 = np.meshgrid(np.arange(0.5, count_sensors_on_side_row + 0.5, 1), z)

        for i in range(4):
            data_colorbar = ax[i].pcolormesh(np.flip(pressure_coefficients[i], axis=0), cmap=cmap, norm=norm)

            match i % 2:
                case 0:
                    ax[i].set_xticks(xticks_13)
                    ax[i].set_xticklabels(xticklabels_13)
                    ax[i].plot(*meshgrid_13, '.k')

                case 1:
                    ax[i].set_xticks(xticks_24)
                    ax[i].set_xticklabels(xticklabels_24)
                    ax[i].plot(*meshgrid_24, '.k')

            ax[i].set_yticks(yticks)
            ax[i].set_yticklabels(yticklabels)

        fig.colorbar(data_colorbar, ax=ax, location='bottom', cmap=cmap, ticks=levels)

        return fig


if __name__ == "__main__":
    import asyncio

    import matplotlib.pyplot as plt
    from sqlalchemy import create_engine
    from src.submodules.databasetoolkit.isolated import (load_positions,
                                                         load_pressure_coefficients, find_experiment_by_model_name, )
    from compiled_aot.integration import aot_integration

    # engine = create_engine("postgresql://postgres:password@localhost:15432/postgres")
    # engine = create_engine("postgresql://postgres:dSJJNjkn42384*$(#@92.246.143.110:5432/windspectrum_db")
    engine = create_engine("postgresql://postgres:1234@localhost/postgres")

    angle = 0
    alpha = 4  # or 6
    model_name = 315
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

    # Plot.isofields_coefficients((10, 10, 10),
    #                             model_name,
    #                             ChartMode.MEAN,
    #                             pressure_coefficients,
    #                             coordinates)

    Plot.pseudocolor_coefficients((30, 10, 50),
                                  model_name,
                                  ChartMode.MEAN,
                                  pressure_coefficients,
                                  )

    plt.show()

    # import time
    #
    # s = time.time()
    # for i in range(19):
    # print(time.time() - s)
