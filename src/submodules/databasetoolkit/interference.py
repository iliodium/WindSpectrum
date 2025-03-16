import json
from typing import (Any,
                    Sequence, Union, )

import numpy
import numpy as np
from pydantic import (BaseModel,
                      ConfigDict,
                      validate_call, )
from pydantic.dataclasses import dataclass
from sqlalchemy import select, insert, create_engine
from sqlalchemy.orm import Session

from src.common.annotation import (AlphaType,
                                   CoordinatesType,
                                   ExperimentIdType,
                                   FaceOrNoneType,
                                   ModelNameIsolatedType,
                                   PositionXOrNoneType,
                                   PositionYOrNoneType,
                                   check_type_engine, AngleType, PositionType, )
from src.submodules.databasetoolkit.orm.models import Buildings, Interference

__SENSOR_VALUES_DISCARD = 1000

# with open(r'config.json', 'r') as file:
#     config_db = json.load(file)
#
__DB_URL_SERVER = r'postgresql://postgres:1234@localhost/postgres'


async def __load_building_by_height(
        height: int,
        _engine
) -> Buildings | None:
    stmt = select(Buildings).where(Buildings.height == int(height))

    with Session(_engine) as session:
        building = session.scalars(stmt).first()
    print(building, 13)
    return building


@validate_call
async def find_id_building_by_height(
        height: int,
        _engine
) -> Buildings.id_building | None:
    check_type_engine(_engine)

    building = await __load_building_by_height(height, _engine)

    if building is None:
        server_engine = create_engine(__DB_URL_SERVER)
        building = await __load_building_by_height(height, server_engine)
        server_engine.dispose()

        if building is None:
            return None
        else:
            await __write_building(building, _engine)

    return building.id_building


async def __write_building(
        building: Buildings,
        _engine,
):
    with Session(_engine) as session:
        stmt = insert(Buildings).values(id_building=building.id_building,
                                        breadth=building.breadth,
                                        depth=building.depth,
                                        height=building.height)
        session.execute(stmt)
        session.commit()


@validate_call
async def __load_pressure_coefficients_by_id_building_instance_angle(
        position: PositionType,
        angle: AngleType,
        id_building: int,
        _engine,
        local_db: bool = False
):
    stmt = select(Interference).where(Interference.position == position,
                                      Interference.angle == angle,
                                      Interference.id_interfering_building == id_building)

    with Session(_engine) as session:
        result = session.scalars(stmt).first()

    if result is None:
        return None

    fc_result = dict()

    print(result)

    if local_db:
        fc_result[result.angle] = np.frombuffer(result.pressure_coefficients, dtype=int).reshape(5858,
                                                                                           -1) / __SENSOR_VALUES_DISCARD
    else:
        fc_result[result.angle] = numpy.array(result.pressure_coefficients, dtype=float) / __SENSOR_VALUES_DISCARD

    return fc_result


async def __write_interference_experiment(
        position,
        angle,
        id_interfering_building,
        pressure_coefficients,
        _engine,
):
    pressure_coefficients = pressure_coefficients * 1000
    pressure_coefficients = pressure_coefficients.astype(int)
    with Session(_engine) as session:
        stmt = insert(Interference).values(position=position,
                                           angle=angle,
                                           id_interfering_building=id_interfering_building,
                                           pressure_coefficients=pressure_coefficients.tobytes())
        session.execute(stmt)
        session.commit()


@validate_call
async def load_pressure_coefficients(
        position: PositionType,
        angle: AngleType,
        id_building: int,
        _engine
):
    check_type_engine(_engine)

    result = await __load_pressure_coefficients_by_id_building_instance_angle(position,
                                                                              angle, id_building,
                                                                              _engine,
                                                                              local_db=True)
    if result is None:
        server_engine = create_engine(__DB_URL_SERVER)
        result = await __load_pressure_coefficients_by_id_building_instance_angle(position,
                                                                                  angle,
                                                                                  id_building,
                                                                                  server_engine,
                                                                                  local_db=False)
        server_engine.dispose()

        if result is None:
            return None
        else:
            await __write_interference_experiment(position, angle, id_building, result[angle], _engine)

    return result


async def __load_experiments_alpha_by_model_name(
        model_name: ModelNameIsolatedType,
        alpha,
        _engine,
        local_db: bool = False

):
    match alpha:
        case 4:
            experiments_alpha = ExperimentsAlpha4
        case 6:
            experiments_alpha = ExperimentsAlpha6
    stmt = select(experiments_alpha).where(experiments_alpha.model_name == model_name)

    with Session(_engine) as session:
        experiment = session.scalars(stmt).first()

    if experiment is None:
        server_engine = create_engine(__DB_URL_SERVER)
        with Session(server_engine) as session:
            experiment = session.scalars(stmt).first()
        server_engine.dispose()

        if experiment is not None:
            __write_experiments_alpha(experiments_alpha, experiment, _engine)

    if experiment is None:
        raise ValueError(f"Experiment with model_name {model_name} not found")

    if local_db:
        experiment.x_coordinates = np.frombuffer(experiment.x_coordinates, dtype=float)
        experiment.z_coordinates = np.frombuffer(experiment.z_coordinates, dtype=float)
        experiment.face_number = np.frombuffer(experiment.face_number, dtype=int)

    return experiment


@validate_call
async def __load_pressure_coefficients_for_type_and_alpha(
        experiment_id: ExperimentIdType,
        models_type,
        alpha: AlphaType,
        _engine,
        *,
        angle: AngleType = None,
        face_number: FaceOrNoneType = None,
        position_x: PositionXOrNoneType = None,
        position_y: PositionYOrNoneType = None,
        local_db: bool = False

) -> dict[int, numpy.ndarray] | None:
    stmt = select(models_type).where(models_type.c.model_id == experiment_id, models_type.c.angle == angle)

    with Session(_engine) as session:
        result = session.execute(stmt).fetchall()

    if result is None or len(result) == 0:
        return None

    _mapper = __identity

    if face_number is not None or position_x is not None or position_y is not None:
        _mapper = __FilterPressureCoefficients(experiment_id, _engine, alpha, face_number, position_x, position_y)

    fc_result = dict()

    for row in result:
        if local_db:
            fc_result[row.angle] = np.frombuffer(row.pressure_coefficients, dtype=int).reshape(32768,
                                                                                               -1) / __SENSOR_VALUES_DISCARD
        else:
            fc_result[row.angle] = _mapper(
                numpy.array(row.pressure_coefficients, dtype=float)) / __SENSOR_VALUES_DISCARD

    return fc_result


@validate_call
def __write_models_alpha(
        models_alpha,
        experiment_id: ExperimentIdType,
        angle: AngleType,
        pressure_coefficients,
        _engine,
):
    pressure_coefficients = pressure_coefficients * 1000
    pressure_coefficients = pressure_coefficients.astype(int)
    with Session(_engine) as session:
        stmt = insert(models_alpha).values(model_id=experiment_id,
                                           angle=angle,
                                           pressure_coefficients=pressure_coefficients.tobytes())
        session.execute(stmt)
        session.commit()


@validate_call
def __write_experiments_alpha(
        experiments_alpha, experiment_description, _engine,
):
    with Session(_engine) as session:
        stmt = insert(experiments_alpha).values(model_id=experiment_description.model_id,
                                                model_name=experiment_description.model_name,
                                                x_coordinates=np.array(experiment_description.x_coordinates).tobytes(),
                                                z_coordinates=np.array(experiment_description.z_coordinates).tobytes(),
                                                face_number=np.array(experiment_description.face_number).tobytes())
        session.execute(stmt)
        session.commit()


if __name__ == "__main__":
    import asyncio

    from sqlalchemy import create_engine

    # local_engine = create_engine('sqlite:///windspectrum.db')
    local_engine = create_engine('postgresql://postgres:1234@localhost/postgres')
    # server_engine = create_engine(__DB_URL_SERVER)

    res = asyncio.run(find_id_building_by_height(196, local_engine))
    print(res)
