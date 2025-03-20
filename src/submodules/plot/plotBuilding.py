from typing import Any

import matplotlib
import matplotlib.tri as mtri
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import BoundaryNorm
from matplotlib.ticker import (MultipleLocator,
                               ScalarFormatter,)
from pydantic import validate_call
from scipy.signal import welch

from compiled_functions import aot_calculations
from src.common.annotation import (AlphaStandardsOrKs10orNoneType,
                                   ChartModeType,
                                   CoordinatesType,
                                   ModelNameIsolatedType,
                                   ModelSizeType,
                                   WindRegionsOrNoneType,)
from src.submodules.plot.plot import Plot
from src.submodules.plot.utils import calculate_levels
from src.submodules.plot.utils import interpolator as intp
from src.submodules.plot.utils import set_colorbar
from src.submodules.utils.data_features import lambdas
from src.submodules.utils.speed_sp import speed_sp_region
from src.submodules.utils.utils import (get_size_tpu_and_count_sensors,
                                        tpu_size_to_real,)
from src.ui.common.ChartMode import ChartMode


class PlotBuilding(Plot):
    @staticmethod
    @validate_call
    def envelopes(
            pressure_coefficients,
            parameters: ChartModeType,
            step_major_x: int = 20,
            step_minor_x: int = 5,
            step_major_y: float = 0.2,
            step_sensors_for_plot: int = 100
    ) -> list[plt.Figure]:
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
            fig, ax = plt.subplots(dpi=PlotBuilding.DPI)
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
            ax.tick_params(axis='y', labelsize=Plot.YTICKS_FONTSIZE)
            start_xticks = sensor_index_start + step_major_x
            stop_xticks = sensor_index_start + count_sensors_ox + 1
            xticks = np.arange(start_xticks, stop_xticks, step_major_x)
            xticks = np.insert(xticks, 0, sensor_index_start + 1)
            ax.set_xticks(xticks)

            ax.xaxis.set_minor_locator(MultipleLocator(step_minor_x))
            ax.xaxis.set_minor_formatter(ScalarFormatter())
            ax.xaxis.set_tick_params(which='major', labelsize=Plot.XTICKS_FONTSIZE)
            ax.xaxis.set_tick_params(which='minor', labelsize=Plot.XTICKS_FONTSIZE - 5)

            ax.legend(loc='upper right', fontsize=Plot.LEGEND_FONTSIZE)
            ax.set_title('Огибающие', fontsize=Plot.TITLE_FONTSIZE)
            ax.set_xlabel('Номер датчика', fontsize=Plot.XLABEL_FONTSIZE)
            ax.set_ylabel('Аэродинамический коэффициент', fontsize=Plot.YLABEL_FONTSIZE)

            figs.append(fig)

        return figs

    @staticmethod
    @validate_call
    def summary_coefficients(
            pressure_coefficients: dict[str, Any],
            kt: float = 1,
            sample_period: float = 32.768,
            number_of_time_counts: int = 32768

    ) -> plt.Figure:
        """
        Построение графиков суммарных аэродинамических коэффициентов в декартовой системе координат.

        Args:
            pressure_coefficients (dict[str, np.array]): Словарь с коэффициентами, где ключ — тип графика,
                                                          а значение — массив данных.
            kt (Union[float, int]): коэф.

        Returns:
            plt.Figure: График в формате Matplotlib.
        """
        fig, ax = plt.subplots(dpi=PlotBuilding.DPI)

        ax.set_xlim(0, sample_period * kt)
        ox = np.linspace(0, sample_period * kt, number_of_time_counts)

        ax.grid()
        ax.set_ylabel('Суммарные аэродинамические коэффициенты', fontsize=Plot.YLABEL_FONTSIZE)
        ax.set_xlabel('Время, с', labelpad=.3, fontsize=Plot.XLABEL_FONTSIZE)

        for name in pressure_coefficients.keys():
            if pressure_coefficients[name] is not None:
                ax.plot(ox, pressure_coefficients[name], label=name)

        ax.legend(loc='upper right', fontsize=Plot.LEGEND_FONTSIZE)
        ax.tick_params(axis='x', labelsize=Plot.XTICKS_FONTSIZE)
        ax.tick_params(axis='y', labelsize=Plot.YTICKS_FONTSIZE)

        return fig

    @staticmethod
    @validate_call
    def sensor_signal(
            signal: dict[str, Any],
            kt: float = 1,
            sample_period: float = 32.768,
            number_of_time_counts: int = 32768

    ) -> plt.Figure:

        fig, ax = plt.subplots(dpi=PlotBuilding.DPI)

        ax.set_xlim(0, sample_period * kt)
        ox = np.linspace(0, sample_period * kt, number_of_time_counts)

        ax.grid()
        ax.set_ylabel('Аэродинамический коэффициент', fontsize=Plot.YLABEL_FONTSIZE)
        ax.set_xlabel('Время, с', labelpad=.3, fontsize=Plot.XLABEL_FONTSIZE)

        for name in signal.keys():
            if signal[name] is not None:
                ax.plot(ox, signal[name], label=name)

        ax.legend(loc='upper right', fontsize=Plot.LEGEND_FONTSIZE)
        ax.tick_params(axis='x', labelsize=Plot.XTICKS_FONTSIZE)
        ax.tick_params(axis='y', labelsize=Plot.YTICKS_FONTSIZE)

        return fig

    @staticmethod
    @validate_call
    def polar_plot(
            data: dict[str, dict[str, Any]],
            title: str = '',
    ) -> plt.Figure:
        """
        Построение графиков суммарных аэродинамических коэффициентов в полярной системе координат.

        Args:
            data (dict[str, dict[str, np.array]]): Словарь с коэффициентами, где первый ключ — параметр графика
                                        а второй ключ тип графика и значение — данные для графика.
            title (str): Заголовок графика.

        Returns:
            plt.Figure: Объект графика.
        """

        angles = np.arange(0, 365, 5) * np.pi / 180.0

        fig, ax = plt.subplots(dpi=PlotBuilding.DPI, subplot_kw={'projection': 'polar'})

        for name, val in data.items():
            for k, d in val.items():
                ax.plot(angles, d, label=f'{name} {k}')

        # Обратное направление
        ax.set_theta_direction(-1)
        # "Север" — нулевая позиция
        ax.set_theta_zero_location('N')
        # Шаг 15° для сетки
        ax.set_thetagrids(np.arange(0, 360, 15))
        ax.tick_params(axis='x', labelsize=Plot.XTICKS_FONTSIZE)
        ax.tick_params(axis='y', labelsize=Plot.YTICKS_FONTSIZE)

        ax.legend(loc='upper right', fontsize=Plot.LEGEND_FONTSIZE)
        if title:
            ax.set_title(title, fontsize=Plot.TITLE_FONTSIZE)

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

    @staticmethod
    @validate_call
    def welch_graph(
            data,
            height,
            speed,
            sample_frequency: int,
            number_of_time_counts: int

    ):
        """Отрисовка графиков спектральной плотности мощности"""
        fig, ax = plt.subplots(dpi=PlotBuilding.DPI)

        ax.set_xscale('log')
        ax.set_yscale('log')

        ax.grid()
        ax.set_title('Спектральная плотность мощности', fontsize=Plot.TITLE_FONTSIZE)
        ax.set_xlabel(r'$\frac{f \cdot H_{ref}}{U_{ref}}$', fontsize=Plot.XLABEL_FONTSIZE + 20)
        ax.set_ylabel(r'$\frac{S(f)\cdot f}{\sigma^2}$', fontsize=Plot.YLABEL_FONTSIZE + 20)

        ax.tick_params(axis='x', labelsize=Plot.XTICKS_FONTSIZE)
        ax.tick_params(axis='y', labelsize=Plot.YTICKS_FONTSIZE)

        for name in data.keys():
            if data[name] is not None:
                sigma = np.std(data[name]) ** 2
                freq, psd = welch(data[name], fs=sample_frequency, nperseg=int(number_of_time_counts / 5))
                ax.plot((freq * height) / speed, (freq * psd) / sigma, label=name)

        ax.legend(loc='upper right', fontsize=Plot.LEGEND_FONTSIZE)

        return fig

    @staticmethod
    @validate_call
    def isofields_coefficients(
            model_size: ModelSizeType,
            size_tpu,
            count_sensors,
            parameter: ChartMode,
            pressure_coefficients,
            coordinates: CoordinatesType,
            area_type: AlphaStandardsOrKs10orNoneType = None,
            wind_region: WindRegionsOrNoneType = None,
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
        # флаг чтобы понимать что мы рисуем, коэффициенты или давление
        flag_pressure = area_type is not None and wind_region is not None

        breadth, depth, height = size_tpu
        count_sensors_on_model, count_sensors_on_middle_row, count_sensors_on_side_row = count_sensors

        pressure_coefficients = lambdas[parameter](pressure_coefficients)

        pressure_coefficients = list(aot_calculations.split_1d_array(
            count_sensors_on_model,
            count_sensors_on_middle_row,
            count_sensors_on_side_row,
            pressure_coefficients
        ))

        x = aot_calculations.split_1d_array(
            count_sensors_on_model,
            count_sensors_on_middle_row,
            count_sensors_on_side_row,
            np.array(coordinates[0])
        )
        z = aot_calculations.split_1d_array(
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

        fig, ax = plt.subplots(1, 4, dpi=PlotBuilding.DPI)

        cmap = matplotlib.colormaps.get_cmap("jet")
        data_colorbar = None

        count_ticks = 5

        if flag_pressure:
            colorbar_label = 'Давление, Па'
            # tpu_height_to_real_func = tpu_height_to_real(z, model_size[2], height)
            # масштабируем высоту датчика, как если бы он был на реальном здание
            vectorized_function_z = np.vectorize(tpu_size_to_real, otypes=[object])
            # otypes=[object] чтобы np.vectorize не конвертировал str в np.str а то валидация падает
            vectorized_function_coefficient = np.vectorize(speed_sp_region, otypes=[object])
            # np.vectorize чтобы применить функцию к каждому элементу массива
            for i in range(4):
                z_sensors = vectorized_function_z(z[i].reshape(-1),
                                                  building_size=model_size[2],
                                                  tpu_size=height)
                coefficient_for_region = vectorized_function_coefficient(z_sensors,
                                                                         area_type=area_type,
                                                                         wind_region=wind_region)
                pressure_coefficients[i] = pressure_coefficients[i].reshape(-1) * coefficient_for_region

            levels = calculate_levels(parameter, pressure_coefficients, flag_pressure)

        else:
            colorbar_label = 'Коэффициенты'

            levels = calculate_levels(parameter, pressure_coefficients)

            for i in range(4):
                pressure_coefficients[i] = pressure_coefficients[i].reshape(-1)

        for i in range(4):
            x_z = np.column_stack((x[i].reshape(-1), z[i].reshape(-1)))
            x_z_extended = np.column_stack((x_extended[i].reshape(-1), z_extended[i].reshape(-1)))
            # Интерполятор полученный на основе имеющихся данных
            interpolator = intp(x_z, pressure_coefficients[i])

            # Получаем данные для несуществующих датчиков
            pressure_coefficients_extended = interpolator(x_z_extended)
            triang = mtri.Triangulation(x_extended[i].reshape(-1), z_extended[i].reshape(-1))
            refiner = mtri.UniformTriRefiner(triang)
            grid, value = refiner.refine_field(pressure_coefficients_extended, subdiv=4)
            data_colorbar = ax[i].tricontourf(grid, value, cmap=cmap, extend='both', levels=levels)
            # Рисуем линии
            labels = ax[i].tricontour(grid, value, linewidths=1, linestyles='solid', colors='black', levels=levels)
            # Подписываем линии
            ax[i].clabel(labels, fontsize=Plot.PLOT_TEXT_FONTSIZE)

            x_start = x_extended[i][0][0]
            x_stop = x_extended[i][0][-1]

            ax[i].set_xticks(np.linspace(x_start, x_stop, count_ticks))
            ax[i].set_xticklabels(np.linspace(0, model_size[i % 2], count_ticks).round(2),
                                  fontsize=Plot.XTICKS_FONTSIZE)

            z_start = 0
            z_stop = z_extended[i][0][0]

            ax[i].set_yticks(np.linspace(z_start, z_stop, count_ticks))
            ax[i].set_yticklabels(np.linspace(0, model_size[2], count_ticks).round(2),
                                  fontsize=Plot.YTICKS_FONTSIZE)

        set_colorbar(fig, levels, data_colorbar, ax, cmap, label=colorbar_label)

        return fig

    @staticmethod
    @validate_call
    def pseudocolor_coefficients(
            model_size: ModelSizeType,
            count_sensors,
            parameter: ChartMode,
            pressure_coefficients
    ) -> plt.Figure:
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

        count_sensors_on_model, count_sensors_on_middle_row, count_sensors_on_side_row = count_sensors
        count_row = count_sensors_on_model // (2 * (count_sensors_on_middle_row + count_sensors_on_side_row))

        pressure_coefficients = lambdas[parameter](pressure_coefficients)
        pressure_coefficients = aot_calculations.split_1d_array(
            count_sensors_on_model,
            count_sensors_on_middle_row,
            count_sensors_on_side_row,
            pressure_coefficients
        )

        fig, ax = plt.subplots(1, 4, dpi=PlotBuilding.DPI)

        cmap = matplotlib.colormaps.get_cmap("jet")

        levels = calculate_levels(parameter, pressure_coefficients)

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
                    ax[i].set_xticklabels(xticklabels_13, fontsize=Plot.XTICKS_FONTSIZE)
                    ax[i].plot(*meshgrid_13, '.k')

                case 1:
                    ax[i].set_xticks(xticks_24)
                    ax[i].set_xticklabels(xticklabels_24, fontsize=Plot.XTICKS_FONTSIZE)
                    ax[i].plot(*meshgrid_24, '.k')

            ax[i].set_yticks(yticks)
            ax[i].set_yticklabels(yticklabels, fontsize=Plot.YTICKS_FONTSIZE)

        set_colorbar(fig, levels, data_colorbar, ax, cmap, label='Коэффициенты')

        return fig


if __name__ == "__main__":
    import asyncio

    import matplotlib.pyplot as plt
    from sqlalchemy import create_engine

    from compiled_functions import aot_calculations
    from src.submodules.databasetoolkit.isolated import (find_experiment_by_model_name,
                                                         load_positions,
                                                         load_pressure_coefficients,)

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

    size, count_sensors = get_size_tpu_and_count_sensors(pressure_coefficients.shape[1],
                                                         model_name,
                                                         )
