from functools import (cache,
                       wraps, )
from typing import (Tuple,
                    Union, )

import numpy as np
from pydantic import validate_call

from src.common.annotation import (AngleType,
                                   ModelNameIsolatedType, )
from src.common.DbType import DbType


# https://stackoverflow.com/questions/52331944/cache-decorator-for-numpy-arrays
def np_cache(function):
    @cache
    def cached_wrapper(*args, **kwargs):
        args = [np.array(a) if isinstance(a, tuple) else a for a in args]
        kwargs = {
            k: np.array(v) if isinstance(v, tuple) else v for k, v in kwargs.items()
        }

        return function(*args, **kwargs)

    @wraps(function)
    def wrapper(*args, **kwargs):
        args = [tuple(a) if isinstance(a, np.ndarray) else a for a in args]
        kwargs = {
            k: tuple(v) if isinstance(v, np.ndarray) else v for k, v in kwargs.items()
        }
        return cached_wrapper(*args, **kwargs)

    wrapper.cache_info = cached_wrapper.cache_info
    wrapper.cache_clear = cached_wrapper.cache_clear

    return wrapper


# @lru_cache(maxsize=32, typed=True)
# @np_cache
@validate_call
def calculate_cmz(_pressure_coefficients,
                  _angle: AngleType,
                  _coordinates,
                  *,
                  _model_name: Union[ModelNameIsolatedType | None] = None,
                  _height: float | None = None,
                  _db: DbType = DbType.ISOLATED
                  ) -> np.ndarray:
    """Вычисление моментов сил CMz"""
    if _db == DbType.ISOLATED:
        breadth, depth, _height = int(_model_name[0]) / 10, int(_model_name[1]) / 10, int(_model_name[2]) / 10

        count_sensors_on_model = _pressure_coefficients.shape[1]
        count_sensors_on_middle_row = int(_model_name[0]) * 5
        count_sensors_on_side_row = int(_model_name[1]) * 5

    elif _db == DbType.INTERFERENCE:
        assert _height is not None and isinstance(_height,
                                                  float), 'height must be not None float when db == DbType.INTERFERENCE'
        breadth, depth = 0.07, 0.07

        count_sensors_on_model = _pressure_coefficients.shape[1]
        count_sensors_on_middle_row = 7
        count_sensors_on_side_row = 7

    count_row = count_sensors_on_model // (2 * (count_sensors_on_middle_row + count_sensors_on_side_row))

    count_sensors_on_row = 2 * (count_sensors_on_middle_row + count_sensors_on_side_row)

    shirina = np.cos(np.deg2rad(_angle)) * breadth + np.sin(np.deg2rad(_angle)) * depth

    x, _ = _coordinates
    x = np.reshape(x, (count_row, -1))
    x = np.split(x, [count_sensors_on_middle_row,
                     count_sensors_on_middle_row + count_sensors_on_side_row,
                     2 * count_sensors_on_middle_row + count_sensors_on_side_row,
                     2 * (count_sensors_on_middle_row + count_sensors_on_side_row)
                     ], axis=1)

    del x[4]

    # центры граней
    mid13_x = breadth / 2
    mid24_x = depth / 2

    v2 = breadth
    v3 = breadth + depth
    v4 = 2 * breadth + depth
    x[1] -= v2
    x[2] -= v3
    x[3] -= v4

    mxs = np.array(
        [
            x[0] - mid13_x,
            x[1] - mid24_x,
            x[2] - mid13_x,
            x[3] - mid24_x,
        ]
    )

    # Площадь
    s13 = breadth * _height
    s24 = depth * _height

    x = _coordinates[0]
    x = np.reshape(x, (-1, count_sensors_on_row))
    x = np.append(x, np.array([[2 * (breadth + depth)] for _ in range(len(x))]), axis=1)
    x = np.insert(x, 0, 0, axis=1)

    y = _coordinates[1]
    y = [_height for _ in range(count_sensors_on_row)] + y
    y = np.reshape(y, (-1, count_sensors_on_row))
    y = np.append(y, np.array([[0] for _ in range(count_sensors_on_row)]))
    y = np.reshape(y, (-1, count_sensors_on_row))

    squares_faces = np.zeros((count_row, count_sensors_on_row))
    for y_i in range(count_row):
        for x_i in range(count_sensors_on_row):
            y_t = y[y_i][x_i]
            y_m = y[y_i + 1][x_i]
            y_b = y[y_i + 2][x_i]
            if y_i == 0:
                dy = y_t - y_m + (y_m - y_b) / 2
            elif y_i == count_row - 1:
                dy = (y_t - y_m) / 2 + y_m - y_b
            else:
                dy = (y_t - y_m) / 2 + (y_m - y_b) / 2

            x_l = x[y_i][x_i]
            x_m = x[y_i][x_i + 1]
            x_r = x[y_i][x_i + 2]

            if x_i == 0:
                dx = x_m - x_l + (x_r - x_m) / 2
            elif x_i == count_sensors_on_row - 1:
                dx = (x_m - x_l) / 2 + x_r - x_m
            else:
                dx = (x_m - x_l) / 2 + (x_r - x_m) / 2

            squares_faces[y_i, x_i] += dy * dx

    squares_faces = np.split(squares_faces, [count_sensors_on_middle_row,
                                             count_sensors_on_middle_row + count_sensors_on_side_row,
                                             2 * count_sensors_on_middle_row + count_sensors_on_side_row,
                                             2 * (count_sensors_on_middle_row + count_sensors_on_side_row)
                                             ], axis=1)

    pc_copy = np.reshape(_pressure_coefficients, (_pressure_coefficients.shape[0], count_row, -1))
    pc_copy = np.split(pc_copy, [count_sensors_on_middle_row,
                                 count_sensors_on_middle_row + count_sensors_on_side_row,
                                 2 * count_sensors_on_middle_row + count_sensors_on_side_row,
                                 2 * (count_sensors_on_middle_row + count_sensors_on_side_row)
                                 ], axis=2)

    divisors = np.array(
        [
            s13,
            s24,
            s13,
            s24
        ]
    ) * shirina

    result = np.zeros(_pressure_coefficients.shape[0])

    for i in range(4):
        result += (np.sum(pc_copy[i] * squares_faces[i] * mxs[i], axis=(1, 2)) / divisors[i])

    return result


# @lru_cache(maxsize=32, typed=True)
@validate_call
def calculate_cx_cy(_pressure_coefficients,
                    _angle: AngleType,
                    _coordinates,
                    *,
                    _model_name: Union[ModelNameIsolatedType | None] = None,
                    _height: float | None = None,
                    _db: DbType = DbType.ISOLATED
                    ) -> Tuple[np.array, np.array]:
    """Вычисление CX и CY"""

    if _db == DbType.ISOLATED:
        breadth, depth, _height = int(_model_name[0]) / 10, int(_model_name[1]) / 10, int(_model_name[2]) / 10

        count_sensors_on_model = _pressure_coefficients.shape[1]
        count_sensors_on_middle_row = int(_model_name[0]) * 5
        count_sensors_on_side_row = int(_model_name[1]) * 5

    elif _db == DbType.INTERFERENCE:
        assert _height is not None and isinstance(_height,
                                                  float), 'height must be not None float when db == DbType.INTERFERENCE'
        breadth, depth = 0.07, 0.07

        count_sensors_on_model = _pressure_coefficients.shape[1]
        count_sensors_on_middle_row = 7
        count_sensors_on_side_row = 7

    count_row = count_sensors_on_model // (2 * (count_sensors_on_middle_row + count_sensors_on_side_row))
    count_sensors_on_row = 2 * (count_sensors_on_middle_row + count_sensors_on_side_row)

    # Площадь
    s13 = breadth * _height
    s24 = depth * _height

    x = _coordinates[0]
    x = np.reshape(x, (-1, count_sensors_on_row))
    x = np.append(x, np.array([[2 * (breadth + depth)] for _ in range(len(x))]), axis=1)
    x = np.insert(x, 0, 0, axis=1)

    y = _coordinates[1]
    y = [_height for _ in range(count_sensors_on_row)] + y
    y = np.reshape(y, (-1, count_sensors_on_row))
    y = np.append(y, np.array([[0] for _ in range(count_sensors_on_row)]))
    y = np.reshape(y, (-1, count_sensors_on_row))

    squares_faces = np.zeros((count_row, count_sensors_on_row))
    for y_i in range(count_row):
        for x_i in range(count_sensors_on_row):
            y_t = y[y_i][x_i]
            y_m = y[y_i + 1][x_i]
            y_b = y[y_i + 2][x_i]
            if y_i == 0:
                dy = y_t - y_m + (y_m - y_b) / 2
            elif y_i == count_row - 1:
                dy = (y_t - y_m) / 2 + y_m - y_b
            else:
                dy = (y_t - y_m) / 2 + (y_m - y_b) / 2

            x_l = x[y_i][x_i]
            x_m = x[y_i][x_i + 1]
            x_r = x[y_i][x_i + 2]

            if x_i == 0:
                dx = x_m - x_l + (x_r - x_m) / 2
            elif x_i == count_sensors_on_row - 1:
                dx = (x_m - x_l) / 2 + x_r - x_m
            else:
                dx = (x_m - x_l) / 2 + (x_r - x_m) / 2

            squares_faces[y_i, x_i] += dy * dx

    squares_faces = np.split(squares_faces, [count_sensors_on_middle_row,
                                             count_sensors_on_middle_row + count_sensors_on_side_row,
                                             2 * count_sensors_on_middle_row + count_sensors_on_side_row,
                                             2 * (count_sensors_on_middle_row + count_sensors_on_side_row)
                                             ], axis=1)

    cx = np.zeros(_pressure_coefficients.shape[0])
    cy = np.zeros(_pressure_coefficients.shape[0])

    pc_copy = np.reshape(_pressure_coefficients, (_pressure_coefficients.shape[0], count_row, -1))
    pc_copy = np.split(pc_copy, [count_sensors_on_middle_row,
                                 count_sensors_on_middle_row + count_sensors_on_side_row,
                                 2 * count_sensors_on_middle_row + count_sensors_on_side_row,
                                 2 * (count_sensors_on_middle_row + count_sensors_on_side_row)
                                 ], axis=2)

    for i in range(4):
        if i == 0:
            cx += (np.sum(pc_copy[i] * squares_faces[i], axis=(1, 2)) / s13)
        elif i == 2:
            cx -= (np.sum(pc_copy[i] * squares_faces[i], axis=(1, 2)) / s13)
        elif i == 1:
            cy += (np.sum(pc_copy[i] * squares_faces[i], axis=(1, 2)) / s24)
        else:
            cy -= (np.sum(pc_copy[i] * squares_faces[i], axis=(1, 2)) / s24)

    return cx, cy


if __name__ == "__main__":
    import asyncio

    import matplotlib.pyplot as plt
    from sqlalchemy import create_engine

    from src.submodules.databasetoolkit.isolated import (load_positions,
                                                         load_pressure_coefficients, )

    # engine = create_engine("postgresql://postgres:password@localhost:15432/postgres")
    # engine = create_engine("postgresql://postgres:dSJJNjkn42384*$(#@92.246.143.110:5432/windspectrum_db")
    engine = create_engine("postgresql://postgres:1234@localhost/postgres")

    angle = 0
    experiment_id = 3
    alpha = 4  # or 6
    model_name = '113'  # связано с experiment_id
    t = asyncio.run(load_pressure_coefficients(experiment_id, alpha, engine, angle=angle))[angle]

    v = calculate_cmz(
        t,
        angle,
        asyncio.run(load_positions(experiment_id, alpha, engine)),
        _model_name=model_name
    )

    cx, cy = calculate_cx_cy(
        asyncio.run(load_pressure_coefficients(experiment_id, alpha, engine, angle=angle))[angle],
        angle,
        asyncio.run(load_positions(experiment_id, alpha, engine)),
        _model_name=model_name
    )

    plt.plot(np.arange(cx.shape[0]) / 1000, cx)
    plt.plot(np.arange(cy.shape[0]) / 1000, cy)
    plt.plot(np.arange(v.shape[0]) / 1000, v)
    plt.show()
