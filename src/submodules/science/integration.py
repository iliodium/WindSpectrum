from typing import (Tuple,
                    )

import numpy as np
from pydantic import validate_call

from compiled_aot.integration import aot_integration
from src.common.DbType import DbType
from src.common.annotation import (CoordinatesType,
                                   ModelNameIsolatedOrNoneType, AngleType, )
from src.submodules.science import utils


@validate_call
def calculate_cmz(
        pressure_coefficients,
        angle: AngleType,
        coordinates: CoordinatesType,
        model_name: ModelNameIsolatedOrNoneType = None,
        height: float | None = None,
        db: DbType = DbType.ISOLATED
):
    """Вычисление моментов сил CMz"""
    size, count_sensors = utils.get_size_and_count_sensors(pressure_coefficients.shape[1],
                                                           model_name,
                                                           height,
                                                           db
                                                           )
    breadth, depth, height = size
    count_sensors_on_model, count_sensors_on_middle_row, count_sensors_on_side_row = count_sensors

    _x = np.array(coordinates[0])
    _y = np.array(coordinates[1])

    return aot_integration.calculate_cmz_aot(
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


@validate_call
def calculate_cx_cy(
        pressure_coefficients,
        coordinates: CoordinatesType,
        *,
        model_name: ModelNameIsolatedOrNoneType = None,
        height: float | None = None,
        db: DbType = DbType.ISOLATED
) -> Tuple[np.array, np.array]:
    """Вычисление CX и CY"""
    size, count_sensors = utils.get_size_and_count_sensors(pressure_coefficients.shape[1],
                                                           model_name,
                                                           height,
                                                           db
                                                           )
    breadth, depth, height = size
    count_sensors_on_model, count_sensors_on_middle_row, count_sensors_on_side_row = count_sensors

    x = np.array(coordinates[0])
    y = np.array(coordinates[1])

    return aot_integration.calculate_cx_cy_aot(
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


if __name__ == "__main__":
    import asyncio
    import time

    import matplotlib.pyplot as plt
    from sqlalchemy import create_engine
    from src.submodules.databasetoolkit.isolated import (load_positions,
                                                         load_pressure_coefficients, )

    start = time.time()
    # engine = create_engine("postgresql://postgres:password@localhost:15432/postgres")
    # engine = create_engine("postgresql://postgres:dSJJNjkn42384*$(#@92.246.143.110:5432/windspectrum_db")
    engine = create_engine("postgresql://postgres:1234@localhost/postgres")

    angle = 0
    experiment_id = 3
    alpha = 4  # or 6
    model_name = 113  # связано с experiment_id

    coordinates = asyncio.run(load_positions(experiment_id, alpha, engine))
    pressure_coefficients = asyncio.run(load_pressure_coefficients(experiment_id, alpha, engine, angle=angle))[angle]

    v = calculate_cmz(
        pressure_coefficients,
        angle,
        coordinates,
        model_name=model_name
    )
    cx, cy = calculate_cx_cy(
        pressure_coefficients,
        coordinates,
        model_name=model_name
    )

    plt.plot(np.arange(cx.shape[0]) / 1000, cx)
    plt.plot(np.arange(cy.shape[0]) / 1000, cy)
    plt.plot(np.arange(v.shape[0]) / 1000, v)
    plt.show()
