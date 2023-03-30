import os
import multiprocessing
from typing import Tuple, List

import numpy as np
import scipy.interpolate
import matplotlib.pyplot as plt


def get_base_angle(angle: int, permutation_view: str, type_base: str = 'square'):
    if permutation_view == "reverse":
        return 90 * (angle // 90 + 1) - angle
    else:
        if type_base == 'square':
            return angle % 90
        elif type_base == 'rectangle' and angle == 270:
            return 90
        else:
            return angle % 90


def changer_sequence_coefficients(coefficients,
                                  permutation_view: str,
                                  model_name: str,
                                  sequence_permutation: Tuple[int, int, int, int]):
    """Меняет порядок следования датчиков тем самым генерируя не существующие углы"""
    f1, f2, f3, f4 = sequence_permutation

    count_sensors_on_middle = int(model_name[0]) * 5
    count_sensors_on_side = int(model_name[1]) * 5

    count_sensors_on_model = len(coefficients[0])
    count_row = count_sensors_on_model // (2 * (count_sensors_on_middle + count_sensors_on_side))

    if permutation_view == 'forward':
        for i in range(len(coefficients)):
            arr = np.array(coefficients[i]).reshape((count_row, -1))
            arr = np.split(arr, [count_sensors_on_middle,
                                 count_sensors_on_middle + count_sensors_on_side,
                                 2 * count_sensors_on_middle + count_sensors_on_side,
                                 2 * (count_sensors_on_middle + count_sensors_on_side)
                                 ], axis=1)

            coefficients[i] = np.concatenate((arr[f1],
                                              arr[f2],
                                              arr[f3],
                                              arr[f4]), axis=1).reshape(count_sensors_on_model)
    else:
        for i in range(len(coefficients)):
            arr = np.array(coefficients[i]).reshape((count_row, -1))
            arr = np.split(arr, [count_sensors_on_middle,
                                 count_sensors_on_middle + count_sensors_on_side,
                                 2 * count_sensors_on_middle + count_sensors_on_side,
                                 2 * (count_sensors_on_middle + count_sensors_on_side)
                                 ], axis=1)

            coefficients[i] = np.concatenate((np.flip(arr[f1], axis=1),
                                              np.flip(arr[f2], axis=1),
                                              np.flip(arr[f3], axis=1),
                                              np.flip(arr[f4], axis=1)), axis=1).reshape(count_sensors_on_model)

    return coefficients


def changer_sequence_numbers(numbers: List[int],
                             model_name: str,
                             sequence_permutation: Tuple[int, int, int, int]):
    print(numbers, model_name)

    f1, f2, f3, f4 = sequence_permutation
    count_sensors_on_middle = int(model_name[1]) * 5
    count_sensors_on_side = int(model_name[0]) * 5
    count_sensors_on_model = len(numbers)
    count_row = count_sensors_on_model // (2 * (count_sensors_on_middle + count_sensors_on_side))

    arr = np.array(numbers).reshape((count_row, -1))
    arr = np.split(arr, [count_sensors_on_middle,
                         count_sensors_on_middle + count_sensors_on_side,
                         2 * count_sensors_on_middle + count_sensors_on_side,
                         2 * (count_sensors_on_middle + count_sensors_on_side)
                         ], axis=1)

    numbers = np.concatenate((arr[f1],
                              arr[f2],
                              arr[f3],
                              arr[f4]), axis=1).reshape(count_sensors_on_model)

    print(numbers)
    return numbers


def get_sequence_permutation_data(type_base: str, permutation_view: str, angle: int):
    """Определяет как менять расстановку датчиков"""

    # определение порядка данных
    sequence = {
        'square': {  # квадрат в основании
            'reverse': {
                45 < angle < 90: (1, 0, 3, 2),
                135 < angle < 180: (2, 1, 0, 3),
                225 < angle < 270: (3, 2, 1, 0),
                315 < angle < 360: (0, 3, 1, 2)
            },
            'forward': {
                0 <= angle <= 45: (0, 1, 2, 3),
                90 <= angle <= 135: (3, 0, 1, 2),
                180 <= angle <= 225: (2, 3, 0, 1),
                270 <= angle <= 315: (1, 2, 3, 0)
            }},
        'rectangle': {  # прямоугольник в основании
            'reverse': {
                90 < angle < 180: (2, 1, 0, 3),
                270 < angle < 360: (0, 3, 2, 1),
            },
            'forward': {
                0 <= angle <= 90: (0, 1, 2, 3),
                180 <= angle <= 270: (2, 3, 0, 1)
            }}
    }

    return sequence[type_base][permutation_view][True]


def get_view_permutation_data(type_base: str, angle: int):
    """Определяет порядок данных для перестановки.
    reverse = нужно перевернуть [1, 2, 3] -> [3, 2, 1]
    forward = не нужно перевернуть [1, 2, 3] -> [1, 2, 3]"""

    if type_base == 'square':  # в основании квадрат
        if any([45 < angle < 90, 135 < angle < 180, 225 < angle < 270, 315 < angle < 360]):
            return 'reverse'
        else:
            return 'forward'

    else:  # в основании прямоугольник
        if any([90 < angle < 180, 270 < angle < 360]):
            return 'reverse'
        else:
            return 'forward'


def calculate_cmz(model_name: str, pr_coeff, coordinates):
    """Вычисление моментов сил CMz"""
    print(model_name)
    cmz = []
    breadth, depth, height = int(model_name[0]) / 10, int(model_name[1]) / 10, int(model_name[2]) / 10

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

    mid13_x = breadth / 2
    mid24_x = depth / 2

    v2 = breadth
    v3 = breadth + depth
    v4 = 2 * breadth + depth
    x[1] -= v2
    x[2] -= v3
    x[3] -= v4

    mx13 = np.array([
        x[0] - mid13_x,
        x[2] - mid13_x,
    ])


    mx24 = np.array([
        x[1] - mid24_x,
        x[3] - mid24_x,
    ])

    for coeff in pr_coeff:
        t_cmz = 0
        coeff = np.reshape(coeff, (count_row, -1))
        coeff = np.split(coeff, [count_sensors_on_middle,
                                 count_sensors_on_middle + count_sensors_on_side,
                                 2 * count_sensors_on_middle + count_sensors_on_side,
                                 2 * (count_sensors_on_middle + count_sensors_on_side)
                                 ], axis=1)

        t_cmz += np.sum(mx13[0] * coeff[0])
        t_cmz += np.sum(mx24[0] * coeff[1])
        t_cmz += np.sum(mx13[1] * coeff[2])
        t_cmz += np.sum(mx24[1] * coeff[3])

        cmz = np.append(cmz, t_cmz)
    print(np.mean(np.array(cmz)))
    return np.array(cmz)


def calculate_cx_cy(model_name: str, pr_coeff):
    """Вычисление CX и CY"""

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
        cx.append(faces_x[0] - faces_x[1])
        cy.append(faces_y[0] - faces_y[1])
    print(np.mean(np.array(cx)))
    print(np.mean(np.array(cy)))
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
    """Вычисление ближайшей модели из БД и коэффициентов масштабирования модели"""
    x, y, z = float(x), float(y), float(z)
    min_size = min(x, y, z)

    # Относительный масштаб фигуры
    x_scale = x / min_size
    y_scale = y / min_size
    z_scale = z / min_size

    # Масштабы из базы данных
    if alpha == '4':
        x_from_db = np.array([1, 2, 3])
        y_from_db = np.array([1, 2, 3])
    else:
        x_from_db = np.array([1, 3])
        y_from_db = np.array([1, 3])

    # Расчет коэффициента для X
    difference_x = np.absolute(x_from_db - x_scale)
    index_x = difference_x.argmin()
    x_nearest = x_from_db[index_x]

    # Расчет коэффициента для Y
    if x_nearest == 1:
        difference_y = np.absolute(y_from_db - y_scale)
        index_y = difference_y.argmin()
        y_nearest = y_from_db[index_y]
    else:
        y_nearest = 1

    # Проверка тк в БД отсутствует z == 1 для x == 2 и x == 3
    if x_nearest in [2, 3] or y_nearest in [2, 3]:
        z_from_db = np.array([2, 3, 4, 5])
    else:
        z_from_db = np.array([1, 2, 3, 4, 5])

    # Расчет коэффициента для Z
    difference_z = np.absolute(z_from_db - z_scale)
    index_z = difference_z.argmin()
    z_nearest = z_from_db[index_z]

    # Коэффициенты масштабирования
    x_scale_factor = x / x_nearest
    y_scale_factor = y / y_nearest
    z_scale_factor = z / z_nearest

    model_from_db = ''.join(map(str, (x_nearest, y_nearest, z_nearest)))  # Модель из БД
    scale_factors = (x_scale_factor, y_scale_factor, z_scale_factor)

    return model_from_db, scale_factors


def generate_directory_for_report(path_report: str):
    """Создание директории под отчет"""
    folders = (
        'Модель',
        'Огибающие',
        'Изополя ветровых нагрузок и воздействий',
        'Изополя ветровых нагрузок и воздействий\\Непрерывные',
        'Изополя ветровых нагрузок и воздействий\\Непрерывные\\MAX',
        'Изополя ветровых нагрузок и воздействий\\Непрерывные\\MEAN',
        'Изополя ветровых нагрузок и воздействий\\Непрерывные\\MIN',
        'Изополя ветровых нагрузок и воздействий\\Непрерывные\\STD',
        'Изополя ветровых нагрузок и воздействий\\Дискретные',
        'Изополя ветровых нагрузок и воздействий\\Дискретные\\MAX',
        'Изополя ветровых нагрузок и воздействий\\Дискретные\\MEAN',
        'Изополя ветровых нагрузок и воздействий\\Дискретные\\MIN',
        'Изополя ветровых нагрузок и воздействий\\Дискретные\\STD',
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
    """Функция для открытия графика"""
    plt.switch_backend('TkAgg')
    plt.close(fig)

    # create a dummy figure and use its
    # manager to display "fig"
    dummy = plt.figure()
    new_manager = dummy.canvas.manager
    new_manager.canvas.figure = fig
    fig.set_canvas(new_manager.canvas)

    plt.show()


def run_proc(fig):
    """Запуск нового процесса для отображения графика"""
    multiprocessing.Process(target=display_fig, args=(fig,)).start()


def open_fig(figures):
    """Отображение графиков"""
    if isinstance(figures, list) or isinstance(figures, tuple):
        for fig in figures:
            run_proc(fig)
    else:
        run_proc(figures)


def interpolator(coords, val):
    """Функция интерполяции"""
    return scipy.interpolate.RBFInterpolator(coords, val, kernel='cubic')


def converter_coordinates(x_old, breadth: float, depth: float, face_number, count_sensors: int):
    """Возвращает из (x_old) -> (x,y)"""
    x = []
    y = []
    for i in range(count_sensors):
        if face_number[i] == 1:
            x.append(float('%.5f' % (-depth / 2)))
            y.append(float('%.5f' % (breadth / 2 - x_old[i])))
        elif face_number[i] == 2:
            x.append(float('%.5f' % (- depth / 2 + x_old[i] - breadth)))
            y.append(float('%.5f' % (-breadth / 2)))
        elif face_number[i] == 3:
            x.append(float('%.5f' % (depth / 2)))
            y.append(float('%.5f' % (-3 * breadth / 2 + x_old[i] - depth)))
        else:
            x.append(float('%.5f' % (3 * depth / 2 - x_old[i] + 2 * breadth)))
            y.append(float('%.5f' % (breadth / 2)))

    return x, y


def converter_coordinates_to_real(x, z, model_size, model_scale):
    breadth_real, depth_real, height_real = float(model_size[0]), float(model_size[1]), float(model_size[2])
    breadth_db, depth_db, height_db = int(model_scale[0]) / 10, int(model_scale[1]) / 10, int(model_scale[2]) / 10

    x_scale_factor = breadth_real / breadth_db
    y_scale_factor = depth_real / depth_db
    z_scale_factor = height_real / height_db

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

    z_real = np.array(z) * z_scale_factor

    del x[4]

    x[0] *= x_scale_factor

    x[1] -= breadth_db
    x[1] *= y_scale_factor
    x[1] += breadth_real

    x[2] -= breadth_db + depth_db
    x[2] *= x_scale_factor
    x[2] += breadth_real + depth_real

    x[3] -= 2 * breadth_db + depth_db
    x[3] *= y_scale_factor
    x[3] += 2 * breadth_real + depth_real

    x_real = np.array([])
    for i in range(count_row):
        x_real = np.append(x_real, x[0][i])
        x_real = np.append(x_real, x[1][i])
        x_real = np.append(x_real, x[2][i])
        x_real = np.append(x_real, x[3][i])

    return x_real, z_real


if __name__ == '__main__':
    print(get_model_and_scale_factors('1', '10', '1', '4'))
