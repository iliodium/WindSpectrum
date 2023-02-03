import os
import scipy.interpolate
import numpy as np
import multiprocessing
import matplotlib.pyplot as plt
from typing import Tuple


def calculate_cmz(model_name: str, model_size: Tuple[str, str, str], pr_coeff, coordinates):
    cmz = []
    breadth_db, depth_db, _ = int(model_name[0]) / 10, int(model_name[1]) / 10, int(model_name[2]) / 10
    breadth_real, depth_real, _ = float(model_size[0]), float(model_size[1]), float(model_size[2])

    center_1 = breadth_real / 2
    center_2 = breadth_real + depth_real / 2
    center_3 = breadth_real + depth_real + depth_real / 2
    center_4 = breadth_real + 2 * depth_real + breadth_real / 2

    count_sensors_on_model = len(pr_coeff[0])
    count_sensors_on_middle = int(model_name[0]) * 5
    count_sensors_on_side = int(model_name[1]) * 5
    count_row = count_sensors_on_model // (2 * (count_sensors_on_middle + count_sensors_on_side))

    x, _ = coordinates
    x = np.reshape(x, (count_row, -1))
    x = np.split(x, [count_sensors_on_middle,
                     count_sensors_on_middle + count_sensors_on_side,
                     2 * count_sensors_on_middle + count_sensors_on_side,
                     2 * (count_sensors_on_middle + count_sensors_on_side)
                     ], axis=1)

    del x[4]

    x[0] *= breadth_real / breadth_db

    x[1] -= breadth_db
    x[1] *= depth_real / depth_db
    x[1] += breadth_real

    x[2] -= breadth_db + depth_db
    x[2] *= breadth_real / breadth_db
    x[2] += breadth_real + depth_real

    x[3] -= 2 * breadth_db + depth_db
    x[3] *= depth_real / depth_db
    x[3] += 2 * breadth_real + depth_real

    mx1 = np.array([abs(x[0] - center_1)])
    mx2 = np.array([abs(x[1] - center_2)])
    mx3 = np.array([abs(x[2] - center_3)])
    mx4 = np.array([abs(x[3] - center_4)])

    print(x[0])
    print('---------------------')
    print(x[1])
    print('---------------------')
    print(x[2])
    print('---------------------')
    print(x[3])
    print('---------------------')
    print(mx1)
    print('---------------------')
    print(mx2)
    print('---------------------')
    print(mx3)
    print('---------------------')
    print(mx4)
    print('---------------------')

    coeffs_norm_13 = [1 if i <= count_sensors_on_middle // 2 else -1 for i in range(count_sensors_on_middle)]
    coeffs_norm_24 = [1 if i <= count_sensors_on_side // 2 else -1 for i in range(count_sensors_on_side)]

    for coeff in pr_coeff:
        t_cmz = 0
        coeff = np.reshape(coeff, (count_row, -1))
        coeff = np.split(coeff, [count_sensors_on_middle,
                                 count_sensors_on_middle + count_sensors_on_side,
                                 2 * count_sensors_on_middle + count_sensors_on_side,
                                 2 * (count_sensors_on_middle + count_sensors_on_side)
                                 ], axis=1)

        for i in range(4):
            if i in [0, 2]:
                coeff[i] *= coeffs_norm_13
            else:
                coeff[i] *= coeffs_norm_24

        t_cmz += np.sum(mx1 * coeff[0])
        t_cmz += np.sum(mx2 * coeff[1])
        t_cmz += np.sum(mx3 * coeff[2])
        t_cmz += np.sum(mx4 * coeff[3])

        cmz = np.append(cmz, t_cmz)

    return np.array(cmz)


def calculate_cx_cy(model_name: str, pr_coeff):
    cx = []
    cy = []
    count_sensors_on_model = len(pr_coeff[0])
    count_sensors_on_middle = int(model_name[0]) * 5
    count_sensors_on_side = int(model_name[1]) * 5

    count_row = count_sensors_on_model // (2 * (count_sensors_on_middle + count_sensors_on_side))
    count_sensors_on_1_3_face = (count_sensors_on_middle * count_row)
    count_sensors_on_2_4_face = (count_sensors_on_side * count_row)

    for coeff in pr_coeff:
        coeff = np.reshape(coeff, (count_row, -1))
        coeff = np.split(coeff, [count_sensors_on_middle,
                                 count_sensors_on_middle + count_sensors_on_side,
                                 2 * count_sensors_on_middle + count_sensors_on_side,
                                 2 * (count_sensors_on_middle + count_sensors_on_side)
                                 ], axis=1)

        del coeff[4]
        faces_x = []
        faces_y = []
        for face in range(len(coeff)):
            if face in [0, 2]:
                faces_x.append(np.sum(coeff[face]) / count_sensors_on_1_3_face)
            else:
                faces_y.append(np.sum(coeff[face]) / count_sensors_on_2_4_face)
        cx.append((faces_x[0] - faces_x[1]) / 2)
        cy.append((faces_y[0] - faces_y[1]) / 2)
    return np.array(cx), np.array(cy)


def rms(data):
    """Среднеквадратичное отклонение"""
    return np.sqrt(np.array(data).dot(np.array(data)) / np.array(data).size).round(2)


def rach(data):
    """Расчетное"""
    return np.max([np.abs(np.min(data)), np.abs(np.max(data))]).round(2)


def obes_p(data):
    """Обеспеченность +"""
    return (np.abs(np.max(data) - np.mean(data)) / np.std(data)).round(2)


def obes_m(data):
    """Обеспеченность -"""
    return (np.abs(np.min(data) - np.mean(data)) / np.std(data)).round(2)


def get_model_and_scale_factors(x: str, y: str, z: str, alpha: str) -> (str, tuple):
    x, y, z = float(x), float(y), float(z)
    min_size = min(x, y, z)

    # Относительный масштаб фигуры
    x_scale = x / min_size
    y_scale = y / min_size
    z_scale = z / min_size

    # Масштабы из базы данных
    if alpha == '4':
        x_from_db = np.array([1, 2, 3])
    else:
        x_from_db = np.array([1, 3])

    # Расчет коэффициента для X
    difference_x = np.absolute(x_from_db - x_scale)
    index_x = difference_x.argmin()
    x_nearest = x_from_db[index_x]
    x_scale_factor = x_scale / x_nearest

    # Проверка тк в БД отсутствует z == 1 для x == 2 | x == 3
    if x_nearest in [2, 3]:
        z_from_db = np.array([2, 3, 4, 5])
    else:
        z_from_db = np.array([1, 2, 3, 4, 5])

    # Расчет коэффициента для Z
    difference_z = np.absolute(z_from_db - z_scale)
    index_z = difference_z.argmin()
    z_nearest = z_from_db[index_z]
    z_scale_factor = z_scale / z_nearest

    model_from_db = ''.join(map(str, (x_nearest, 1, z_nearest)))  # Модель из БД
    scale_factors = (x_scale_factor, y_scale, z_scale_factor)  # Коэффициенты масштабирования

    return model_from_db, scale_factors


def generate_directory_for_report(path_report: str):
    folders = (
        'Модель',
        'Огибающие',
        'Изополя ветровых нагрузок и воздействий',
        'Изополя ветровых нагрузок и воздействий\\Непрерывные',
        'Изополя ветровых нагрузок и воздействий\\Непрерывные\\MAX',
        'Изополя ветровых нагрузок и воздействий\\Непрерывные\\MEAN',
        'Изополя ветровых нагрузок и воздействий\\Непрерывные\\MIN',
        'Изополя ветровых нагрузок и воздействий\\Непрерывные\\STD',
        'Суммарные аэродинамические коэффициенты',
        'Суммарные аэродинамические коэффициенты\\Декартовая система координат',
        'Суммарные аэродинамические коэффициенты\\Полярная система координат',
        'Спектральная плотность мощности',
        'Спектральная плотность мощности\\Линейная шкала',
        'Спектральная плотность мощности\\Логарифмическая шкала',
    )

    if not os.path.isdir(f'{path_report}'):
        os.mkdir(f'{path_report}')

    for folder in folders:
        if not os.path.isdir(f'{path_report}\\{folder}'):
            os.mkdir(f'{path_report}\\{folder}')


def display_fig(fig):
    plt.close(fig)

    # create a dummy figure and use its
    # manager to display "fig"
    dummy = plt.figure()
    new_manager = dummy.canvas.manager
    new_manager.canvas.figure = fig
    fig.set_canvas(new_manager.canvas)

    plt.show()


def run_proc(fig):
    multiprocessing.Process(target=display_fig, args=(fig,)).start()


def open_fig(*figures):
    for fig in figures:
        run_proc(fig)


def interpolator(coords, val):
    return scipy.interpolate.RBFInterpolator(coords, val, kernel='cubic')


if __name__ == '__main__':
    print(get_model_and_scale_factors('3', '5', '10', '4'))
