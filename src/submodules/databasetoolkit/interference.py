import numpy
import numpy as np
from pydantic import (validate_call, )
from sqlalchemy import select, insert, create_engine
from sqlalchemy.orm import Session

from src.common.annotation import (check_type_engine, AngleType, PositionType, )
from src.submodules.databasetoolkit.orm.models import Buildings, Interference

__SENSOR_VALUES_DISCARD = 1000


async def __load_building_by_height(
        height: int,
        _engine
) -> Buildings | None:
    stmt = select(Buildings).where(Buildings.height == int(height))

    with Session(_engine) as session:
        building = session.scalars(stmt).first()

    return building


@validate_call
async def find_id_building_by_height(
        height: int,
        _engine,
        _db_url_server
) -> Buildings.id_building | None:
    check_type_engine(_engine)

    building = await __load_building_by_height(height, _engine)
    if building is None:
        server_engine = create_engine(_db_url_server)
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

    if local_db:
        fc_result[result.angle] = np.frombuffer(result.pressure_coefficients, dtype=int).reshape(5858,
                                                                                                 -1) / __SENSOR_VALUES_DISCARD
    else:
        fc_result[result.angle] = numpy.array(result.pressure_coefficients, dtype=float) / __SENSOR_VALUES_DISCARD

    return fc_result, result.id_interference


async def __write_interference_experiment(
        position,
        angle,
        id_interfering_building,
        pressure_coefficients,
        id_interference,
        _engine,
):
    pressure_coefficients = pressure_coefficients * 1000
    pressure_coefficients = pressure_coefficients.astype(int)
    with Session(_engine) as session:
        stmt = insert(Interference).values(id_interference=id_interference,
                                           position=position,
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
        _engine,
        _db_url_server
):
    check_type_engine(_engine)

    result = await __load_pressure_coefficients_by_id_building_instance_angle(position,
                                                                              angle, id_building,
                                                                              _engine,
                                                                              local_db=True)
    if result is None:
        server_engine = create_engine(_db_url_server)
        result = await __load_pressure_coefficients_by_id_building_instance_angle(position,
                                                                                  angle,
                                                                                  id_building,
                                                                                  server_engine,
                                                                                  local_db=False)
        server_engine.dispose()

        if result is None:
            return None
        else:
            await __write_interference_experiment(position, angle, id_building, result[0][angle], result[1], _engine)

    return result[0]


if __name__ == "__main__":
    from sqlalchemy import create_engine

    # local_engine = create_engine('sqlite:///windspectrum.db')
    local_engine = create_engine('postgresql://postgres:1234@localhost/postgres')
    # server_engine = create_engine(__DB_URL_SERVER)

    # res = asyncio.run(find_id_building_by_height(196, local_engine))
    # print(res)
