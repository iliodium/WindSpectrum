import asyncio
import io

import numpy as np

from compiled_aot.integration import aot_integration
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
from src.submodules.science import utils
from src.submodules.utils.scaling import (get_model_and_scale_factors,
                                          get_model_and_scale_factors_interference, )


def height_integration_cx_cy_cmz_floors_to_txt_aot(
        _db: DbType.DbType,
        _angle: AngleType,
        _engine,
        *,
        _model_size: ModelSizeOrNoneType = None,
        _model_name: ModelNameIsolatedOrNoneType = None,
        _alpha: AlphaType = 4
):
    x, y, z = _model_size if _model_size is not None else [int(i) / 10 for i in str(_model_name)]

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

    size, count_sensors = utils.get_size_and_count_sensors(pressure_coefficients.shape[1],
                                                           model_name,
                                                           db=_db
                                                           )
    breadth, depth, height = size
    count_sensors_on_model, count_sensors_on_middle_row, count_sensors_on_side_row = count_sensors

    cx, cy, cmz = aot_integration.aot_height_integration_cx_cy_cmz_floors_to_txt(
        count_sensors_on_model,
        count_sensors_on_middle_row,
        count_sensors_on_side_row,
        _angle,
        breadth,
        depth,
        np.array(coordinates[0]),
        np.array(coordinates[1]),
        height,
        pressure_coefficients
    )
    count_row = cx.shape[0]

    z_levels = sorted(set(coordinates[1]), reverse=True)

    time = np.linspace(0, 32.768, 32768).round(5)

    match _alpha:
        case 4:
            speed_2_3 = np.round(interp_025_tpu([height])[0], 3)
        case 6:
            speed_2_3 = np.round(interp_016_tpu([height])[0], 3)

    f = io.StringIO()
    alpha_temp = '0.25' if _alpha == 4 else '0.16'
    f.write(
        f'{model_name} Вариант модели\n{breadth}, {depth}, {height} м размеры модели b d h\n'
        f'{alpha_temp} альфа\n{angle} угол\n{uh_speed} Uh скорость на высоте H\n'
        f'{speed_2_3} скорость на высоте 2/3 H\n'
        f'{count_row} количество этажей\n'
    )

    f.write('time, ')

    for ind in range(1, count_row + 1):
        f.write(f'cx{ind}, cy{ind}, cmz{ind}, ')

    f.write('cxsum, cysum, cmzsum\n')

    f.write(', '.join(map(str, range(count_row * 3 + 1 + 3))) + '\n')

    f.write('-1, ')
    for z in reversed(z_levels):
        z /= height
        f.write(f'{z}, {z}, {z}, ')
    f.write('1, 1, 1\n')

    data_to_txt = np.array([time])

    for ind in range(count_row):
        data_to_txt = np.append(data_to_txt, [cx[ind]], axis=0)
        data_to_txt = np.append(data_to_txt, [cy[ind]], axis=0)
        data_to_txt = np.append(data_to_txt, [cmz[ind]], axis=0)

    data_to_txt = np.append(data_to_txt, [np.sum(cx, axis=0).round(5)], axis=0)
    data_to_txt = np.append(data_to_txt, [np.sum(cy, axis=0).round(5)], axis=0)
    data_to_txt = np.append(data_to_txt, [np.sum(cmz, axis=0).round(5)], axis=0)

    np.savetxt(f, data_to_txt.T, newline="\n", delimiter=',', fmt='%.5f')

    res = str(f.getvalue())

    f.close()

    return res


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
                result = height_integration_cx_cy_cmz_floors_to_txt_aot(
                    DbType.DbType.ISOLATED,
                    angle,
                    engine,
                    _model_name=model_name,
                    _alpha=alpha
                )
                res.write(result)

    alpha = 6

    with Session(engine) as _session:
        model_names = _session.execute(
            select(ExperimentsAlpha6.model_name).order_by(ExperimentsAlpha6.model_name)).scalars()
    for model_name in model_names:
        if not ttest_model_name(model_name):
            continue
        for angle in range(50, 95, 5):
            print(model_name, angle)
            with open(f"model_name_{model_name}_angle_{angle}_alpha_{alpha}.txt", mode='w', encoding="utf8") as res:
                result = height_integration_cx_cy_cmz_floors_to_txt_aot(
                    DbType.DbType.ISOLATED,
                    angle,
                    engine,
                    _model_name=model_name,
                    _alpha=alpha
                )
                res.write(result)
