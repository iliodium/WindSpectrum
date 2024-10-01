import numpy as np


def get_model_and_scale_factors(x: str | float, y: str | float, z: str | float, alpha: int) -> tuple[str, tuple]:
    """
    Функция для расчета модели и коэффициентов масштабирования на основе входных параметров.

    Параметры:
    x (str): Координата x.
    y (str): Координата y.
    z (str): Координата z.
    alpha (str): Значение альфа, которое должно быть '4' или '6'.

    Возвращает:
    tuple: Кортеж, содержащий модель из базы данных и коэффициенты масштабирования для x, y и z.
    """
    assert isinstance(x, (int, float)), "x must be int or float"
    assert isinstance(y, (int, float)), "y must be int or float"
    assert isinstance(z, (int, float)), "z must be int or float"
    assert isinstance(alpha, (int)), "alpha must be int"
    assert alpha in (4, 6), "alpha must be 4 or 6"
    x, y, z = float(x), float(y), float(z)
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
    else:
        raise ValueError("alpha must be '4' or '6'")

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

    model_from_db = ''.join(map(str, (x_nearest, y_nearest, z_nearest)))  # Модель из БД
    scale_factors = (x_scale_factor, y_scale_factor, z_scale_factor)

    return model_from_db, scale_factors


def get_model_and_scale_factors_interference(x: str | float, y: str | float, z: str | float) -> tuple[str, tuple]:
    """Вычисление ближайшей модели из БД и коэффициентов масштабирования модели"""
    assert isinstance(x, (int, float)), "x must be int or float"
    assert isinstance(y, (int, float)), "y must be int or float"
    assert isinstance(z, (int, float)), "z must be int or float"
    x, y, z = float(x), float(y), float(z)
    min_size = min(x, y, z)

    # Относительный масштаб фигуры
    z_scale = z / min_size

    x_nearest = 1

    y_nearest = 1

    z_from_db = np.array([2, 2.8, 4, 6, 8])

    # Расчет коэффициента для Z
    difference_z = np.absolute(z_from_db - z_scale)
    index_z = difference_z.argmin()
    z_nearest = z_from_db[index_z]

    # Коэффициенты масштабирования
    x_scale_factor = x / x_nearest
    y_scale_factor = y / y_nearest
    z_scale_factor = z / z_nearest

    model_from_db = {float('2'): 140,
                     float('2.8'): 196,
                     float('4'): 280,
                     float('6'): 420,
                     float('8'): 560,
                     }[z_nearest]  # Модель из БД
    scale_factors = (x_scale_factor, y_scale_factor, z_scale_factor)

    return model_from_db, scale_factors


if __name__ == "__main__":
    print(
        get_model_and_scale_factors(
            '1',
            '1.5',
            '3.4',
            4
        )
    )
