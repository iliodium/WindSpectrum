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
cc.output_dir = os.path.dirname(os.path.realpath(__file__))
cc.output_file = 'aot_integration.pyd'


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

    result = aot_integration.aot_calculate_cmz(
        count_sensors_on_model,
        count_sensors_on_middle_row,
        count_sensors_on_side_row,
        angle,
        breadth,
        depth,
        _x,
        x,
        y,
        height,
        pressure_coefficients
    )

    return result


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


if __name__ == "__main__":
    import asyncio

    import matplotlib.pyplot as plt
    from sqlalchemy import create_engine
    from src.submodules.databasetoolkit.isolated import (load_positions,
                                                         load_pressure_coefficients, load_experiment_by_id, )

    # engine = create_engine("postgresql://postgres:password@localhost:15432/postgres")
    # engine = create_engine("postgresql://postgres:dSJJNjkn42384*$(#@92.246.143.110:5432/windspectrum_db")
    engine = create_engine("postgresql://postgres:1234@localhost/postgres")
    angle = 0
    experiment_id = 1
    alpha = 4  # or 6
    model_name = '115'  # связано с experiment_id
    asyncio.run(load_positions(experiment_id, alpha, engine))
    t1 = asyncio.run(load_pressure_coefficients(experiment_id, alpha, engine, angle=angle))
    t2 = asyncio.run(load_positions(experiment_id, alpha, engine))

    import aot_integration

    alpha = 4
    for experiment_id in range(1, 14):
        model = asyncio.run(load_experiment_by_id(experiment_id, alpha, engine))
        t2 = asyncio.run(load_positions(experiment_id, alpha, engine))
        if t2:
            # for angle in range(0, 360, 5):
            for angle in range(0, 15, 5):
                print(alpha, model.model_name, angle)
                t1 = asyncio.run(load_pressure_coefficients(experiment_id, alpha, engine, angle=angle))
                if t1:
                    cmz = calculate_cmz(
                        t1[angle],
                        angle,
                        t2,
                        model_name=model.model_name,
                        height=None,
                        db=DbType.ISOLATED
                    )

                    cx, cy = calculate_cx_cy(
                        t1[angle],
                        t2,
                        model_name=model.model_name
                    )
    print('все хорошо!!!')

    # plt.plot(np.arange(cx.shape[0]) / 1000, cx)
    # plt.plot(np.arange(cy.shape[0]) / 1000, cy)
    # plt.plot(np.arange(v.shape[0]) / 1000, v)
    # plt.show()
