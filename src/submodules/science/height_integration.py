import asyncio
import io

import numpy as np
from numba.pycc import CC

from pydantic import validate_call
from src.common import DbType
from src.common.annotation import (AlphaType,
                                   AngleType,
                                   ModelNameIsolatedOrNoneType,
                                   ModelSizeOrNoneType, )
from src.submodules.databasetoolkit.isolated import (find_experiment_by_model_name,
                                                     load_positions,
                                                     load_pressure_coefficients, )
from src.submodules.inner.interpreted_data import (interp_016_tpu,
                                                   interp_025_tpu, )
from src.submodules.utils.scaling import (get_model_and_scale_factors,
                                          get_model_and_scale_factors_interference, )

cc = CC('aot_integration')


@validate_call
def height_integration_cx_cy_cmz_floors_to_txt(
        _db: DbType.DbType,
        _angle: AngleType,
        _engine,
        *,
        _model_size: ModelSizeOrNoneType = None,
        _model_name: ModelNameIsolatedOrNoneType = None,
        _alpha: AlphaType = 4
):
    # добавить кол во этажей после 6
    # в высоту добавить -1 в начало, для времени
    assert not (_model_size is not None and _model_name is not None), \
        "Either _model_size or _model_name must be set, not both"
    assert _model_size is not None or _model_name is not None, "Either _model_size or _model_name must be set"

    x, y, z = _model_size if _model_size is not None else [int(i) / 10 for i in _model_name]

    if _db == DbType.DbType.ISOLATED:
        model_name, _ = get_model_and_scale_factors(
            x,
            y,
            z,
            _alpha
        )
    elif _db == DbType.DbType.INTERFERENCE:
        model_name, _ = get_model_and_scale_factors_interference(
            x,
            y,
            z
        )

    experiment = asyncio.run(find_experiment_by_model_name(model_name, _alpha, _engine))

    pressure_coefficients = asyncio.run(load_pressure_coefficients(experiment.model_id, _alpha, _engine, angle=_angle))[
        _angle]
    coordinates = asyncio.run(load_positions(experiment.model_id, _alpha, _engine))
    uh_speed = np.round(float(experiment.uh_averagewindspeed), 3)

    if _db == DbType.DbType.ISOLATED:
        breadth, depth, height = int(model_name[0]) / 10, int(model_name[1]) / 10, int(model_name[2]) / 10
        count_sensors_on_middle_row = int(model_name[0]) * 5
        count_sensors_on_side_row = int(model_name[1]) * 5
    elif _db == DbType.DbType.INTERFERENCE:
        height = model_name / 1000
        breadth, depth = 0.07, 0.07

    count_sensors_on_model = len(pressure_coefficients[0])

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




if __name__ == "__main__":
    from sqlalchemy import (create_engine,
                            select, )
    from sqlalchemy.orm import Session
    from src.submodules.databasetoolkit.orm.models import (ExperimentsAlpha4,
                                                           ExperimentsAlpha6, )

    # engine = create_engine("postgresql://postgres:password@localhost:15432/postgres")
    # engine = create_engine("postgresql://postgres:dSJJNjkn42384*$(#@92.246.143.110:5432/windspectrum_db")
    engine = create_engine("postgresql://postgres:1234@localhost/postgres")

    alpha = 4
    import aot_integration

    def ttest_model_name(model_name):
        if str(model_name).startswith("11"):
            return False
        return True


    with Session(engine) as _session:
        model_names = _session.execute(
            select(ExperimentsAlpha4.model_name).order_by(ExperimentsAlpha4.model_name)).scalars()
    for model_name in model_names:
        if not ttest_model_name(model_name):
            continue
        if int(model_name) < 314:
            continue
        for angle in range(50 if int(model_name) != 314 else 75, 95, 5):
            with open(f"model_name_{model_name}_angle_{angle}_alpha_{alpha}.txt", mode='w', encoding="utf8") as res:
                res.write(
                    height_integration_cx_cy_cmz_floors_to_txt(
                        DbType.DbType.ISOLATED,
                        angle,
                        engine,
                        _model_name=str(model_name),
                        _alpha=alpha
                    )
                )

    alpha = 6

    with Session(engine) as _session:
        model_names = _session.execute(
            select(ExperimentsAlpha6.model_name).order_by(ExperimentsAlpha6.model_name)).scalars()
    for model_name in model_names:
        if not ttest_model_name(model_name):
            continue
        for angle in range(50, 95, 5):
            with open(f"model_name_{model_name}_angle_{angle}_alpha_{alpha}.txt", mode='w', encoding="utf8") as res:
                res.write(
                    height_integration_cx_cy_cmz_floors_to_txt(
                        DbType.DbType.ISOLATED,
                        angle,
                        engine,
                        _model_name=str(model_name),
                        _alpha=alpha
                    )
                )
