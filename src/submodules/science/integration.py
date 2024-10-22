import os
from typing import (Tuple,
                    Union, )

import numpy as np
from numba.pycc import CC
from pydantic import validate_call
from src.common.annotation import (AngleType,
                                   CoordinatesType,
                                   ModelNameIsolatedType, )
from src.common.DbType import DbType

cc = CC('aot_integration')
cc1 = CC('aot_integration1')


@validate_call
def _get_size_and_count_sensors(
        pressure_coefficients_shape: int,
        model_name: Union[ModelNameIsolatedType | None] = None,
        height: float | None = None,
        db: DbType = DbType.ISOLATED
):
    match db:
        case DbType.ISOLATED:
            breadth, depth, height = int(model_name[0]) / 10, int(model_name[1]) / 10, int(model_name[2]) / 10

            count_sensors_on_model = pressure_coefficients_shape
            count_sensors_on_middle_row = int(model_name[0]) * 5
            count_sensors_on_side_row = int(model_name[1]) * 5

        case DbType.INTERFERENCE:
            assert height is not None and isinstance(height, float), \
                'height must be not None float when db == DbType.INTERFERENCE'
            breadth, depth = 0.07, 0.07

            count_sensors_on_model = pressure_coefficients_shape
            count_sensors_on_middle_row = 7
            count_sensors_on_side_row = 7

    return ((breadth,
             depth,
             height),
            (count_sensors_on_model,
             count_sensors_on_middle_row,
             count_sensors_on_side_row))


from aot_integration import (aot_calculate_cmz_old,

    # add_borders_to_coordinates_x,
    #                          add_borders_to_coordinates_y,
    #                          calculate_sensors_area_effect,
    #                          split_pressure_coefficients,
    #                          calculate_mxs,
    #                          calculate_projection_on_the_axis,
    #                          calculate_square_of_face,
    #                          aot_calculate_cmz,

                             )
from aot_integration import calculate_cmz_artem
# from aot_integration import calculate_cmz_artem
# from aot_integration1 import calculate_cmz_artem
from integration_to_aot import (add_borders_to_coordinates_x,
                                add_borders_to_coordinates_y,
                                calculate_sensors_area_effect,
                                split_pressure_coefficients,
                                calculate_mxs,
                                calculate_projection_on_the_axis,
                                calculate_square_of_face,
                                aot_calculate_cmz,

                                )


@validate_call
def calculate_cmz(
        pressure_coefficients,
        angle: AngleType,
        coordinates: CoordinatesType,
        model_name: Union[ModelNameIsolatedType | None] = None,
        height: float | None = None,
        db: DbType = DbType.ISOLATED
):
    """Вычисление моментов сил CMz"""
    size, count_sensors = _get_size_and_count_sensors(pressure_coefficients.shape[1],
                                                      model_name,
                                                      height,
                                                      db
                                                      )
    breadth, depth, height = size
    count_sensors_on_model, count_sensors_on_middle_row, count_sensors_on_side_row = count_sensors

    _x = np.array(coordinates[0])
    _y = np.array(coordinates[1])
    return calculate_cmz_artem(
        count_sensors_on_model,
        count_sensors_on_middle_row,
        count_sensors_on_side_row,
        angle,
        breadth,
        depth,
        _x,
        _y,
        height,
        pressure_coefficients

    )
    # count_row = count_sensors_on_model // (2 * (count_sensors_on_middle_row + count_sensors_on_side_row))
    # count_sensors_on_row = 2 * (count_sensors_on_middle_row + count_sensors_on_side_row)

    # x = add_borders_to_coordinates_x(
    #     _x,
    #     count_sensors_on_row,
    #     count_row,
    #     breadth,
    #     depth
    # )
    #
    # y = add_borders_to_coordinates_y(
    #     _y,
    #     count_sensors_on_row,
    #     height
    # )
    #
    # sensors_area_effect = calculate_sensors_area_effect(
    # # sensors_area_effect13, sensors_area_effect24 = calculate_sensors_area_effect(
    #     x,
    #     y,
    #     count_row,
    #     count_sensors_on_row,
    #     count_sensors_on_middle_row,
    #     count_sensors_on_side_row
    # )
    #
    # pc = split_pressure_coefficients(
    # # pc13, pc24 = split_pressure_coefficients(
    #     count_sensors_on_model,
    #     count_sensors_on_middle_row,
    #     count_sensors_on_side_row,
    #     pressure_coefficients
    # )
    #
    #
    # mxs = calculate_mxs(
    # # mxs13, mxs24 = calculate_mxs(
    #     count_sensors_on_model,
    #     count_sensors_on_middle_row,
    #     count_sensors_on_side_row,
    #     _x,
    #     breadth,
    #     depth,
    # )
    #
    # projection_on_the_axis = calculate_projection_on_the_axis(
    #     breadth,
    #     depth,
    #     angle
    # )
    #
    # square13, square24 = calculate_square_of_face(
    #     breadth,
    #     depth,
    #     height
    # )
    #
    # result = aot_calculate_cmz(
    #     projection_on_the_axis,
    #     square13,
    #     square24,
    #     *sensors_area_effect,
    #     *mxs,
    #     *pc
    # )
    #
    # # result = aot_calculate_cmz(
    # #     projection_on_the_axis,
    # #     square13,
    # #     sensors_area_effect13,
    # #     mxs13,
    # #     pc13,
    # #     square24,
    # #     sensors_area_effect24,
    # #     mxs24,
    # #     pc24
    # # )
    #
    # return result


@validate_call
def calculate_cx_cy(
        pressure_coefficients,
        coordinates,
        *,
        model_name: Union[ModelNameIsolatedType | None] = None,
        height: float | None = None,
        db: DbType = DbType.ISOLATED
) -> Tuple[np.array, np.array]:
    """Вычисление CX и CY"""
    size, count_sensors = _get_size_and_count_sensors(pressure_coefficients.shape[1],
                                                      model_name,
                                                      height,
                                                      db
                                                      )
    breadth, depth, height = size
    count_sensors_on_model, count_sensors_on_middle_row, count_sensors_on_side_row = count_sensors

    _x = np.array(coordinates[0])
    _y = np.array(coordinates[1])
    count_row = count_sensors_on_model // (2 * (count_sensors_on_middle_row + count_sensors_on_side_row))
    count_sensors_on_row = 2 * (count_sensors_on_middle_row + count_sensors_on_side_row)

    x = aot_integration.add_borders_to_coordinates_x(
        _x,
        count_sensors_on_row,
        count_row,
        breadth,
        depth
    )
    y = aot_integration.add_borders_to_coordinates_y(
        _y,
        count_sensors_on_row,
        height
    )

    result = aot_integration.aot_calculate_cx_cy(
        count_sensors_on_model,
        count_sensors_on_middle_row,
        count_sensors_on_side_row,
        breadth,
        depth,
        x,
        y,
        height,
        pressure_coefficients
    )

    return result


def calculate_cmz_old(
        _pressure_coefficients,
        _angle: AngleType,
        _coordinates: CoordinatesType,
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
    _coordinates = tuple(list(c) for c in _coordinates)
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


import aot_integration


def pre_calculate_cmz(
        pressure_coefficients,
        angle: AngleType,
        coordinates: CoordinatesType,
        model_name: Union[ModelNameIsolatedType | None] = None,
        height: float | None = None,
        db: DbType = DbType.ISOLATED
):
    """Вычисление моментов сил CMz"""
    match db:
        case DbType.ISOLATED:
            breadth, depth, height = int(model_name[0]) / 10, int(model_name[1]) / 10, int(model_name[2]) / 10

            count_sensors_on_model = pressure_coefficients.shape[1]
            count_sensors_on_middle_row = int(model_name[0]) * 5
            count_sensors_on_side_row = int(model_name[1]) * 5

        case DbType.INTERFERENCE:
            assert height is not None and isinstance(height, float), \
                'height must be not None float when db == DbType.INTERFERENCE'
            breadth, depth = 0.07, 0.07

            count_sensors_on_model = pressure_coefficients.shape[1]
            count_sensors_on_middle_row = 7
            count_sensors_on_side_row = 7

    _x = np.array(coordinates[0])
    _y = np.array(coordinates[1])

    result = aot_integration.aot_calculate_cmz_old(
        count_sensors_on_model,
        count_sensors_on_middle_row,
        count_sensors_on_side_row,
        angle,
        breadth,
        depth,
        _x,
        _x,
        _y,
        height,
        pressure_coefficients
    )
    return result


if __name__ == "__main__":
    import asyncio

    import matplotlib.pyplot as plt
    from sqlalchemy import create_engine
    from src.submodules.databasetoolkit.isolated import (load_positions,
                                                         load_pressure_coefficients, load_experiment_by_id, )

    # engine = create_engine("postgresql://postgres:password@localhost:15432/postgres")
    # engine = create_engine("postgresql://postgres:dSJJNjkn42384*$(#@92.246.143.110:5432/windspectrum_db")
    engine = create_engine("postgresql://postgres:1234@localhost/postgres")
    # angle = 0
    # experiment_id = 1
    # alpha = 4  # or 6
    # model_name = '115'  # связано с experiment_id
    # asyncio.run(load_positions(experiment_id, alpha, engine))
    # t1 = asyncio.run(load_pressure_coefficients(experiment_id, alpha, engine, angle=angle))
    # t2 = asyncio.run(load_positions(experiment_id, alpha, engine))

    # alpha = 4
    # for experiment_id in range(5, 14):
    #     model = asyncio.run(load_experiment_by_id(experiment_id, alpha, engine))
    #     coord = asyncio.run(load_positions(experiment_id, alpha, engine))
    #     if coord:
    #         # for angle in range(0, 360, 5):
    #         for angle in range(0, 10, 5):
    #             print(alpha, model.model_name, angle)
    #             pressure = asyncio.run(load_pressure_coefficients(experiment_id, alpha, engine, angle=angle))
    #             if pressure:
    #                 calculate_cmz(
    #                     pressure[angle],
    #                     angle,
    #                     coord,
    #                     model_name=model.model_name,
    #                     height=None,
    #                     db=DbType.ISOLATED
    #                 )

    experiment_id = 1
    alpha = 4
    angle = 0

    import time

    model = asyncio.run(load_experiment_by_id(experiment_id, alpha, engine))
    coord = asyncio.run(load_positions(experiment_id, alpha, engine))
    pressure = asyncio.run(load_pressure_coefficients(experiment_id, alpha, engine, angle=angle))
    coordinates_list = tuple(list(c) for c in coord)
    COUNT = 500
    t0 = time.time()
    calculate_cmz(
        pressure[angle],
        angle,
        coord,
        model_name=model.model_name,
        height=None,
        db=DbType.ISOLATED
    )
    print(time.time() - t0)
    # разбитая функция, скомпилированная
    t1 = time.time()
    for _ in range(COUNT):
        calculate_cmz(
            pressure[angle],
            angle,
            coord,
            model_name=model.model_name,
            height=None,
            db=DbType.ISOLATED
        )
    print(time.time() - t1)
    #
    # # скомпилированная функция
    # t3 = time.time()
    # for _ in range(COUNT):
    #     pre_calculate_cmz(
    #         pressure[angle],
    #         angle,
    #         coord,
    #         model_name=model.model_name,
    #         height=None,
    #         db=DbType.ISOLATED
    #     )
    # print(time.time() - t3)
    #
    # # стандартная функция
    # t2 = time.time()
    # for _ in range(COUNT):
    #     calculate_cmz_old(
    #         pressure[angle],
    #         angle,
    #         coordinates_list,
    #         _model_name=model.model_name,
    #         _height=None,
    #         _db=DbType.ISOLATED
    #     )
    # print(time.time() - t2)
    # plt.plot(np.arange(cx.shape[0]) / 1000, cx)
    # plt.plot(np.arange(cy.shape[0]) / 1000, cy)
    # plt.plot(np.arange(v.shape[0]) / 1000, v)
    # plt.show()
