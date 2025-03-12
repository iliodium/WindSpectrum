import numpy as np
from pydantic import validate_call
from src.common.annotation import (AlphaType,
                                   BuildingSizeType,)


@validate_call
def get_model_and_scale_factors(
        x: BuildingSizeType,
        y: BuildingSizeType,
        z: BuildingSizeType,
        alpha: AlphaType
) -> tuple[int, tuple]:
    """
    Функция для расчета модели и коэффициентов масштабирования на основе входных параметров.

    Параметры:
    x : Значение реального здания x.
    y : Значение реального здания y.
    z : Значение реального здания z.
    alpha : Значение альфа.

    Возвращает:
    tuple: Кортеж, содержащий модель из базы данных и коэффициенты масштабирования для x, y и z.
    """
    min_size = min(x, y, z)

    # Относительный масштаб фигуры
    x_scale = x / min_size
    y_scale = y / min_size
    z_scale = z / min_size

    # Масштабы из базы данных
    # TODO selects from db if we will extend it
    if alpha == 4:
        x_from_db = np.array([1, 2, 3])
        y_from_db = np.array([1, 2, 3])
    elif alpha == 6:
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
    # TODO selects from db if we will extend it
    if x_nearest in [2, 3] or y_nearest in [2, 3]:
        z_from_db = np.array([2, 3, 4, 5])
    else:
        z_from_db = np.array([1, 2, 3, 4, 5])

    # Расчет коэффициента для Z
    difference_z = np.absolute(z_from_db - z_scale)
    index_z = difference_z.argmin()
    z_nearest = z_from_db[index_z]

    # Коэффициенты масштабирования
    x_scale_factor = x / x_nearest * 10
    y_scale_factor = y / y_nearest * 10
    z_scale_factor = z / z_nearest * 10

    model_from_db = int(''.join(map(str, (x_nearest, y_nearest, z_nearest))))  # Модель из БД
    scale_factors = (x_scale_factor, y_scale_factor, z_scale_factor)

    return model_from_db, scale_factors


@validate_call
def get_model_and_scale_factors_interference(
        x: BuildingSizeType,
        y: BuildingSizeType,
        z: BuildingSizeType
) -> tuple[int, tuple]:
    """Вычисление ближайшей модели из БД и коэффициентов масштабирования модели"""
    z_and_model_from_db = {2: 140,
                           2.8: 196,
                           4: 280,
                           6: 420,
                           8: 560,
                           }
    min_size = min(x, y, z)

    # Относительный масштаб фигуры
    z_scale = z / min_size

    x_nearest = 1

    y_nearest = 1

    z_from_db = np.array([*z_and_model_from_db])

    # Расчет коэффициента для Z
    difference_z = np.absolute(z_from_db - z_scale)
    index_z = difference_z.argmin()
    z_nearest = z_from_db[index_z]

    # Коэффициенты масштабирования
    x_scale_factor = x / x_nearest
    y_scale_factor = y / y_nearest
    z_scale_factor = z / z_nearest

    model_from_db = z_and_model_from_db[z_nearest]  # Модель из БД
    scale_factors = (x_scale_factor, y_scale_factor, z_scale_factor)

    return model_from_db, scale_factors


def converter_coordinates_to_real(
        x,
        z,
        model_size,
        model_name
):
    breadth_real, depth_real, height_real = model_size
    breadth_db, depth_db, height_db = [int(i) / 10 for i in str(model_name)]

    x_scale_factor = breadth_real / breadth_db
    y_scale_factor = depth_real / depth_db
    z_scale_factor = height_real / height_db

    count_sensors_on_model = len(x)
    count_sensors_on_middle = int(model_name[0]) * 5
    count_sensors_on_side = int(model_name[1]) * 5
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


if __name__ == "__main__":
    print(
        get_model_and_scale_factors(
            '1',
            '1.5',
            '3.4',
            4
        )
    )
