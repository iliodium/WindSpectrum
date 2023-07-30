import os
import logging

from multiprocessing import Process, managers
from typing import Tuple, List

import numpy as np
import scipy.interpolate
import matplotlib.pyplot as plt

alpha_standards = {'A': 0.15, 'B': 0.2, 'C': 0.25}
ks10 = {'A': 1, 'B': 0.65, 'C': 0.4}
wind_regions = {'Iа': 0.17, 'I': 0.23, 'II': 0.30, 'III': 0.38, 'IV': 0.48, 'V': 0.60, 'VI': 0.73, 'VII': 0.85}
id_to_name = {
    'isofieldsPressure': 'Изополя давления',
    'isofieldsCoefficients': 'Изополя коэффициентов',
    'pseudocolorCoefficients': 'Мозаика коэффициентов',
    'envelopes': 'Огибающие',
    'polarSummaryCoefficients': 'Суммарные коэффициенты в полярной системе координат',
    'summaryCoefficients': 'Графики суммарных коэффициентов',
    'summarySpectres': 'Спектры суммарных коэффициентов',
    '3dModel': 'Трехмерная модель',
    'modelPolar': 'Модель в полярной системе координат',
    'pressureTapLocations': 'Система датчиков мониторинга',
    'max': 'Максимальные',
    'mean': 'Средние',
    'min': 'Минимальные',
    'std': 'Среднеквадратическое отклонение',
    'rms': 'Среднеквадратичное значение',
    'rach': 'Расчетное',
    'obesP': 'Обеспеченность -',
    'obesM': 'Обеспеченность +',
    'cx': 'CX',
    'cy': 'CY',
    'cmz': 'CmZ',
    'cx_cy': 'CX CY',
    'cx_cmz': 'CX CmZ',
    'cy_cmz': 'CY CmZ',
    'cx_cy_cmz': 'CX CY CmZ',
    'statisticsSensors': 'Статистика по датчикам',
    'x': 'X(mm)',
    'y': 'Y(mm)',
    'z': 'Z(mm)',
    'statisticsSummaryCoefficients': 'Статистика по суммарным аэродинамическим коэффициентам',

}


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


def calculate_cmz(model_name: str, angle: str, pr_coeff, coordinates):
    """Вычисление моментов сил CMz"""
    cmz = []
    breadth, depth, height = int(model_name[0]) / 10, int(model_name[1]) / 10, int(model_name[2]) / 10

    count_sensors_on_model = len(pr_coeff[0])
    count_sensors_on_middle_row = int(model_name[0]) * 5  # В строке
    count_sensors_on_side_row = int(model_name[1]) * 5  # В строке
    count_row = count_sensors_on_model // (2 * (count_sensors_on_middle_row + count_sensors_on_side_row))

    count_sensors_on_1_3_face = count_sensors_on_middle_row * count_row
    count_sensors_on_2_4_face = count_sensors_on_side_row * count_row

    angle = int(angle)
    shirina = np.cos(angle) * breadth + np.sin(angle) * depth

    x, _ = coordinates
    x = np.reshape(x, (count_row, -1))
    x = np.split(x, [count_sensors_on_middle_row,
                     count_sensors_on_middle_row + count_sensors_on_side_row,
                     2 * count_sensors_on_middle_row + count_sensors_on_side_row,
                     2 * (count_sensors_on_middle_row + count_sensors_on_side_row)
                     ], axis=1)

    del x[4]

    # print(breadth,depth)
    # центры граней
    mid13_x = breadth / 2
    mid24_x = depth / 2

    v2 = breadth
    v3 = breadth + depth
    v4 = 2 * breadth + depth
    x[1] -= v2
    x[2] -= v3
    x[3] -= v4

    # mx плечи для каждого сенсора
    mx13 = np.array([
        x[0] - mid13_x,
        x[2] - mid13_x,
    ])

    mx24 = np.array([
        x[1] - mid24_x,
        x[3] - mid24_x,
    ])

    # print(mx13)
    # print(mx24)
    # Площадь для каждого датчика
    s13 = breadth * height
    s24 = depth * height
    s13i = s13 / count_sensors_on_1_3_face
    s24i = s24 / count_sensors_on_2_4_face
    # print(s13)
    # print(s24)
    # print(len(pr_coeff))
    # print(len(pr_coeff[0]))
    # print(mx13)
    # print(mx24)

    for coeff in pr_coeff:
        t_cmz = 0
        coeff = np.reshape(coeff, (count_row, -1))
        coeff = np.split(coeff, [count_sensors_on_middle_row,
                                 count_sensors_on_middle_row + count_sensors_on_side_row,
                                 2 * count_sensors_on_middle_row + count_sensors_on_side_row,
                                 2 * (count_sensors_on_middle_row + count_sensors_on_side_row)
                                 ], axis=1)
        # print(coeff)
        # print(mx13[0] * coeff[0])
        # print(mx24[0] * coeff[1])
        # t_cmz += np.sum(mx13[0] * coeff[0] * s13i) / (count_sensors_on_1_3_face * s13 * shirina)
        # t_cmz += np.sum(mx24[0] * coeff[1] * s24i) / (count_sensors_on_2_4_face * s24 * shirina)
        # t_cmz += np.sum(mx13[1] * coeff[2] * s13i) / (count_sensors_on_1_3_face * s13 * shirina)
        # t_cmz += np.sum(mx24[1] * coeff[3] * s24i) / (count_sensors_on_2_4_face * s24 * shirina)
        t1 = np.sum(mx13[0] * coeff[0] * s13i) / (count_sensors_on_1_3_face * s13 * shirina)
        t2 = np.sum(mx24[0] * coeff[1] * s24i) / (count_sensors_on_2_4_face * s24 * shirina)
        t3 = np.sum(mx13[1] * coeff[2] * s13i) / (count_sensors_on_1_3_face * s13 * shirina)
        t4 = np.sum(mx24[1] * coeff[3] * s24i) / (count_sensors_on_2_4_face * s24 * shirina)

        # cmz = np.append(cmz, (t_cmz))
        cmz = np.append(cmz, sum([t1, t2, t3, t4]))

    return np.array(cmz)


def calculate_cx_cy(model_name: str, pr_coeff):
    """Вычисление CX и CY"""
    breadth, depth, height = int(model_name[0]) / 10, int(model_name[1]) / 10, int(model_name[2]) / 10

    cx = []
    cy = []

    count_sensors_on_model = len(pr_coeff[0])
    count_sensors_on_middle_row = int(model_name[0]) * 5
    count_sensors_on_side_row = int(model_name[1]) * 5

    count_row = count_sensors_on_model // (2 * (count_sensors_on_middle_row + count_sensors_on_side_row))
    count_sensors_on_1_3_face = count_sensors_on_middle_row * count_row
    count_sensors_on_2_4_face = count_sensors_on_side_row * count_row

    # Площадь для каждого датчика
    s13 = breadth * height
    s24 = depth * height
    s13i = s13 / count_sensors_on_1_3_face
    s24i = s24 / count_sensors_on_2_4_face

    #print(s13, s24, count_sensors_on_1_3_face, count_sensors_on_2_4_face)
    # Домнажать на площадь каждого датчика, а потом делить на площадь данной грани
    for coeff in pr_coeff:
        coeff = np.reshape(coeff, (count_row, -1))
        coeff = np.split(coeff, [count_sensors_on_middle_row,
                                 count_sensors_on_middle_row + count_sensors_on_side_row,
                                 2 * count_sensors_on_middle_row + count_sensors_on_side_row,
                                 2 * (count_sensors_on_middle_row + count_sensors_on_side_row)
                                 ], axis=1)

        del coeff[4]

        gran1 = np.sum(coeff[0] * s13i) / (count_sensors_on_1_3_face * s13)
        gran3 = np.sum(coeff[2] * s13i) / (count_sensors_on_1_3_face * s13)

        gran2 = np.sum(coeff[1] * s24i) / (count_sensors_on_2_4_face * s24)
        gran4 = np.sum(coeff[3] * s24i) / (count_sensors_on_2_4_face * s24)

        cx.append(gran1 - gran3)
        cy.append(gran2 - gran4)

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


def generate_directory_for_report(current_path: str, name_report: str):
    """Создание директории под отчет"""
    folders = (
        'Модель',
        'Изополя ветровых нагрузок и воздействий',
        'Изополя ветровых нагрузок и воздействий\\Коэффициенты',
        'Изополя ветровых нагрузок и воздействий\\Коэффициенты\\max',
        'Изополя ветровых нагрузок и воздействий\\Коэффициенты\\mean',
        'Изополя ветровых нагрузок и воздействий\\Коэффициенты\\min',
        'Изополя ветровых нагрузок и воздействий\\Коэффициенты\\std',
        'Изополя ветровых нагрузок и воздействий\\Давление',
        'Изополя ветровых нагрузок и воздействий\\Давление\\max',
        'Изополя ветровых нагрузок и воздействий\\Давление\\mean',
        'Изополя ветровых нагрузок и воздействий\\Давление\\min',
        'Изополя ветровых нагрузок и воздействий\\Давление\\std',
        'Мозаика коэффициентов',
        'Мозаика коэффициентов\\max',
        'Мозаика коэффициентов\\mean',
        'Мозаика коэффициентов\\min',
        'Мозаика коэффициентов\\std',
        'Огибающие',
        'Суммарные аэродинамические коэффициенты',
        'Суммарные аэродинамические коэффициенты\\Декартовая система координат',
        'Суммарные аэродинамические коэффициенты\\Полярная система координат',
        'Спектральная плотность мощности',
        'Спектральная плотность мощности\\Линейная шкала',
        'Спектральная плотность мощности\\Логарифмическая шкала',
    )
    path = f'{current_path}\\Отчеты'
    if not os.path.isdir(path):
        os.mkdir(path)

    path = f'{path}\\{name_report}'
    if not os.path.isdir(path):
        os.mkdir(path)

    for folder in folders:
        if not os.path.isdir(f'{path}\\{folder}'):
            os.mkdir(f'{path}\\{folder}')


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
    Process(target=display_fig, args=(fig,)).start()


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


def to_dict(item):
    if isinstance(item, managers.DictProxy) or isinstance(item, dict):
        return {k: to_dict(v) for k, v in item.items()}
    return item


def to_multiprocessing_dict(item, manager):
    if isinstance(item, managers.DictProxy) or isinstance(item, dict):
        return manager.dict({k: to_multiprocessing_dict(v, manager) for k, v in item.items()})
    return item


def get_logger(name):
    logger = logging.getLogger(f'{name}'.ljust(15, ' '))
    logger.setLevel(logging.INFO)

    # настройка обработчика и форматировщика
    py_handler = logging.FileHandler("log.log", mode='a')
    py_formatter = logging.Formatter("%(name)s %(asctime)s %(levelname)s %(message)s")

    # добавление форматировщика к обработчику
    py_handler.setFormatter(py_formatter)
    # добавление обработчика к логгеру
    logger.addHandler(py_handler)
    return logger


if __name__ == '__main__':
    # generate_directory_for_report('D:\Projects\WindSpectrum')
    print(get_model_and_scale_factors('4', '6', '12', 6))
