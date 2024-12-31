import numpy as np
import scipy

from src.submodules.plot.plot import Plot
from src.ui.common.ChartMode import ChartMode


def get_labels(point):
    """
    эта функция нужна тк
    np.arange(0, 0.3, 0.1) -> array([0. , 0.1, 0.2])

    np.arange(0, 0.2 + 0.1, 0.1) -> array([0. , 0.1, 0.2, 0.3])
    """
    point = int(point * 10)
    arange = np.arange(0, point + 1)

    return [f"{i / 10:.1f}" for i in arange]


def scaling_data(x, y=None, angle_border=50):
    """Масштабирование данных до 360 градусов для графиков в полярной системе координат для изолированных зданий"""
    match angle_border:
        case 50:
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

        case 95:
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


def interpolator(coords, val):
    """Функция интерполяции"""
    return scipy.interpolate.RBFInterpolator(coords, val, kernel='cubic')


def round_list_model_pic(arr):
    new_arr = []
    for val in arr:
        if isinstance(val, int):
            new_arr.append(str(val))
            continue

        elif isinstance(val, str):
            val = float(val)

        new_arr.append(f'{round(val, 2):.1f}')

    return new_arr


def calculate_function_from_pressure_coefficients(data, func, decimals):
    return np.round(func([func(data[i]) for i in range(4)]), decimals)


def calculate_levels(parameter, pressure_coefficients):
    min_value = calculate_function_from_pressure_coefficients(pressure_coefficients, np.min, 1)
    max_value = calculate_function_from_pressure_coefficients(pressure_coefficients, np.max, 1)

    match parameter:
        case ChartMode.MAX | ChartMode.MIN:
            step = 0.2
        case ChartMode.RMS | ChartMode.STD:
            step = 0.05
        case _:
            step = 0.1

    match parameter:
        case ChartMode.RMS | ChartMode.STD:
            decimals = 2
        case _:
            decimals = 1

    levels = np.round(np.arange(min_value - step, max_value + step, step), decimals)

    return levels


def set_colorbar(fig, levels, data_colorbar, ax, cmap):
    lbot = levels[0:: 2]
    ltop = levels[1:: 2]
    cbar = fig.colorbar(data_colorbar, ax=ax, location='bottom', cmap=cmap, ticks=lbot)
    cbar.ax.tick_params(labelsize=Plot.COLORBAR_FONTSIZE)
    vmin = cbar.norm.vmin
    vmax = cbar.norm.vmax

    # --------------Print top tick labels--------------
    for ii in ltop:
        cbar.ax.text((ii - vmin) / (vmax - vmin), 1.1, ii, transform=cbar.ax.transAxes, va='bottom',
                     ha='center', fontsize=Plot.COLORBAR_FONTSIZE)


if __name__ == "__main__":
    print(get_labels(0.5))
