import asyncio
import io

import numpy as np
from sqlalchemy import Engine

from src.common import DbType
from src.submodules.databasetoolkit.isolated import find_experiment_by_model_name, load_positions, \
    load_pressure_coefficients
from src.submodules.inner.interpreted_data import (interp_016_tpu, interp_025_tpu)
from src.submodules.utils.scaling import (get_model_and_scale_factors,
                                          get_model_and_scale_factors_interference)


def height_integration_cx_cy_cmz_floors_to_txt(
        _db: DbType.DbType, _angle: int, _engine: Engine, *,
        _model_size: tuple[int, int, int] | None = None,
        _model_name: str | None = None, _alpha: int = 4
):
    # добавить кол во этажей после 6
    # в высоту добавить -1 в начало, для времени

    assert isinstance(_db, DbType.DbType), 'db must be DbType'
    assert not (_model_size is not None and _model_name is not None), \
        "Either _model_size or _model_name must be set, not both"
    assert _model_size is not None or _model_name is not None, "Either _model_size or _model_name must be set"
    assert _model_size is None or (
            isinstance(_model_size, tuple) and len(_model_size) == 3
    ), "_model_size should be either None or tuple of len = 3"
    assert _model_name is None or (
            isinstance(_model_name, str) and len(_model_name) == 3 and _model_name.isnumeric()
    ), f"_model_name should be either None or str of len = 3 only of digits. got {_model_name}"
    assert isinstance(_angle, int), "_angle must be an int"
    assert isinstance(_alpha, int), "_alpha must be an int"
    assert _alpha == 4 or _alpha == 6, "_alpha must be either 4 or 6"

    x, y, z = _model_size if _model_size is not None else tuple([int(i) / 10 for i in _model_name])

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

    shirina = np.cos(np.deg2rad(_angle)) * breadth + np.sin(np.deg2rad(_angle)) * depth

    # центры граней
    mid13_x = breadth / 2
    mid24_x = depth / 2

    x1 = coordinates[0]
    x1 = np.reshape(x1, (count_row, -1))
    x1 = np.split(x1, [count_sensors_on_middle_row,
                       count_sensors_on_middle_row + count_sensors_on_side_row,
                       2 * count_sensors_on_middle_row + count_sensors_on_side_row,
                       2 * (count_sensors_on_middle_row + count_sensors_on_side_row)
                       ], axis=1)

    v2 = breadth
    v3 = breadth + depth
    v4 = 2 * breadth + depth
    x1[1] -= v2
    x1[2] -= v3
    x1[3] -= v4

    # mx плечи для каждого сенсора
    mx13 = np.array([
        x1[0] - mid13_x,
        x1[2] - mid13_x,
    ])

    mx24 = np.array([
        x1[1] - mid24_x,
        x1[3] - mid24_x,
    ])

    # Площадь
    s13 = breadth * height
    s24 = depth * height

    x = coordinates[0]
    x = np.reshape(x, (-1, count_sensors_on_row))
    x = np.append(x, np.full((len(x), 1), 2 * (breadth + depth)), axis=1)
    x = np.insert(x, 0, 0, axis=1)

    y = coordinates[1]
    z_levels = sorted(set(y), reverse=True)
    y = np.append(np.full(count_sensors_on_row, height), y)
    y = np.reshape(y, (-1, count_sensors_on_row))
    y = np.append(y, np.zeros((count_sensors_on_row, 1)))
    y = np.reshape(y, (-1, count_sensors_on_row))

    squares = []
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

            squares.append(dy * dx)
    squares_faces = np.reshape(squares, (count_row, -1))
    squares_faces = np.split(squares_faces, [count_sensors_on_middle_row,
                                             count_sensors_on_middle_row + count_sensors_on_side_row,
                                             2 * count_sensors_on_middle_row + count_sensors_on_side_row,
                                             2 * (count_sensors_on_middle_row + count_sensors_on_side_row)
                                             ], axis=1)

    cx = [
        [] for _ in range(count_row)
    ]
    cy = [
        [] for _ in range(count_row)
    ]
    cmz = [
        [] for _ in range(count_row)
    ]
    for pr in pressure_coefficients:
        pr = np.reshape(pr, (count_row, -1))
        pr = np.split(pr, [count_sensors_on_middle_row,
                           count_sensors_on_middle_row + count_sensors_on_side_row,
                           2 * count_sensors_on_middle_row + count_sensors_on_side_row,
                           2 * (count_sensors_on_middle_row + count_sensors_on_side_row)
                           ], axis=1)

        for row_i in range(count_row):
            faces_x = []
            faces_y = []

            for face in range(4):
                if face in [0, 2]:
                    faces_x.append(np.sum(pr[face][row_i] * squares_faces[face][row_i]) / (s13 / count_row))
                else:
                    faces_y.append(np.sum(pr[face][row_i] * squares_faces[face][row_i]) / (s24 / count_row))

            cx[row_i].append(faces_x[0] - faces_x[1])
            cy[row_i].append(faces_y[0] - faces_y[1])

            t1 = np.sum(mx13[0][row_i] * pr[0][row_i] * squares_faces[0][row_i]) / ((s13 / count_row) * shirina)
            t3 = np.sum(mx13[1][row_i] * pr[2][row_i] * squares_faces[2][row_i]) / ((s13 / count_row) * shirina)

            t2 = np.sum(mx24[0][row_i] * pr[1][row_i] * squares_faces[1][row_i]) / ((s24 / count_row) * shirina)
            t4 = np.sum(mx24[1][row_i] * pr[3][row_i] * squares_faces[3][row_i]) / ((s24 / count_row) * shirina)

            cmz[row_i] = np.append(cmz[row_i], sum([t1, t2, t3, t4]))

    cx = (np.array(list(reversed(cx))) / count_row).round(5)
    cy = (np.array(list(reversed(cy))) / count_row).round(5)
    cmz = (np.array(list(reversed(cmz))) / count_row).round(5)

    # fig, ax = plt.subplots(dpi=Plot.dpi, num='snfgkjdsnfkjsdnf', clear=True)
    # ax.plot(list(range(32768)), np.sum(cx, axis=0))

    angle = _angle

    time = np.linspace(0, 32.768, 32768).round(5)

    if _alpha == 4:
        speed_2_3 = np.round(interp_025_tpu([height])[0], 3)
    elif _alpha == 6:
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

    # for ind in range(1, count_row + 1):
    #     temp_name_str += f'cx{count_row - ind}, cy{count_row - ind}, cmz{count_row - ind}, '

    for ind in range(1, count_row + 1):
        f.write(f'cx{ind}, cy{ind}, cmz{ind}, ')

    f.write('cxsum, cysum, cmzsum\n')

    # enumerate_str = ', '.join(map(str, reversed(range(count_row * 3 + 1 + 3)))) + '\n'
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
    from sqlalchemy import create_engine, select
    from sqlalchemy.orm import Session
    from src.submodules.databasetoolkit.orm.models import ExperimentsAlpha4, ExperimentsAlpha6

    engine = create_engine("postgresql://postgres:password@localhost:25432/postgres")

    alpha = 4

    def test_model_name(model_name):
        if str(model_name).startswith("11"):
            return False
        return True

    with Session(engine) as _session:
        model_names = _session.execute(
            select(ExperimentsAlpha4.model_name).order_by(ExperimentsAlpha4.model_name)).scalars()
    for model_name in model_names:
        if not test_model_name(model_name):
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
        if not test_model_name(model_name):
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
