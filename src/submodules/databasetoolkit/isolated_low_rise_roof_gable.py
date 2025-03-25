import numpy
import numpy as np
from pydantic import validate_call
from sqlalchemy import (insert,
                        select,)
from sqlalchemy.orm import Session
from src.common.annotation import check_type_engine
from src.submodules.databasetoolkit.orm.models import (BuildingWithEaves,
                                                       WindAzimuths,)

__SENSOR_VALUES_DISCARD = 1000


async def __write_id_wind_azimuth(
        id_wind_azimuth,
        angle,
        _engine
):
    with Session(_engine) as session:
        stmt = insert(WindAzimuths).values(id_wind_azimuth=id_wind_azimuth,
                                           wind_azimuth=angle)
        session.execute(stmt)
        session.commit()


async def __load_id_wind_azimuth(
        angle,
        _engine
):
    stmt = select(WindAzimuths).where(WindAzimuths.wind_azimuth == angle)

    with Session(_engine) as session:
        result = session.scalars(stmt).first()

    return result


@validate_call
async def load_id_wind_azimuth(
        angle,
        _engine,
        _db_url_server

):
    check_type_engine(_engine)

    result = await __load_id_wind_azimuth(angle, _engine)

    if result is None:
        server_engine = create_engine(_db_url_server)
        result = await __load_id_wind_azimuth(angle, server_engine)
        server_engine.dispose()

        if result is None:
            return None
        else:
            await __write_id_wind_azimuth(result.id_wind_azimuth, result.wind_azimuth, _engine)

    return result.wind_azimuth


async def __write_experiment(
        experiment,
        _engine,
):
    attributes = experiment.__dict__
    del attributes['_sa_instance_state']
    attributes['pressure_coefficients'] = np.array(attributes['pressure_coefficients']).tobytes()

    with Session(_engine) as session:
        stmt = insert(BuildingWithEaves).values(**attributes)
        session.execute(stmt)
        session.commit()


@validate_call
async def __load_experiment(
        id_eave,
        id_wind_azimuth,
        breadth,
        depth,
        height,
        _engine,
        local_db: bool = False
):
    stmt = select(BuildingWithEaves).where(BuildingWithEaves.id_wind_azimuth == id_wind_azimuth,
                                           BuildingWithEaves.id_eave == id_eave,
                                           BuildingWithEaves.breadth == breadth,
                                           BuildingWithEaves.depth == depth,
                                           BuildingWithEaves.height == height)

    with Session(_engine) as session:
        experiment = session.scalars(stmt).first()

    if experiment is None:
        return None

    fc_result = dict()

    if local_db:
        fc_result[experiment.id_wind_azimuth] = np.frombuffer(experiment.pressure_coefficients, dtype=int).reshape(14063,
                                                                                                           -1) / __SENSOR_VALUES_DISCARD
    else:
        fc_result[experiment.id_wind_azimuth] = numpy.array(experiment.pressure_coefficients,
                                                        dtype=float) / __SENSOR_VALUES_DISCARD

    return fc_result[experiment.id_wind_azimuth], experiment


@validate_call
async def load_pressure_coefficients(
        id_eave,
        id_wind_azimuth,
        breadth,
        depth,
        height,
        _engine,
        _db_url_server
):
    check_type_engine(_engine)

    result = await __load_experiment(id_eave,
                                     id_wind_azimuth,
                                     breadth,
                                     depth,
                                     height,
                                     _engine,
                                     local_db=True)
    if result is None:
        server_engine = create_engine(_db_url_server)
        result = await __load_experiment(id_eave,
                                         id_wind_azimuth,
                                         breadth,
                                         depth,
                                         height,
                                         server_engine,
                                         local_db=False)
        server_engine.dispose()

        if result is None:
            return None
        else:
            await __write_experiment(result[1], _engine)

    return result[0]


if __name__ == "__main__":
    import asyncio

    from sqlalchemy import create_engine

    local_engine = create_engine(r'sqlite:///D:\WindSpectrum\WindSpectrum\windspectrum.db')
    db_url_server = "postgresql://postgres:1234@localhost/postgres"

    res = asyncio.run(load_pressure_coefficients(1, 1, 16, 24, 12, local_engine, db_url_server))
    print(res)
