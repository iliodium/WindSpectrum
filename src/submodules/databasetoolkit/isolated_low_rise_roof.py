import numpy
import numpy as np
from pydantic import validate_call
from sqlalchemy import (create_engine,
                        distinct,
                        insert,
                        select,)
from sqlalchemy.orm import Session
from src.common.annotation import check_type_engine
from src.submodules.databasetoolkit.orm.models import (BuildingWithoutEaves,
                                                       RoofPitches,
                                                       SurfacesRoof,
                                                       WindAzimuths,
                                                       XCoordinatesRoof,
                                                       YCoordinatesRoof,)

__SENSOR_VALUES_DISCARD = 1000


async def __write_id_data_to_table(
        table,
        data,
        _engine
):
    for k, v in data.items():
        if isinstance(data[k], list):
            data[k] = np.array(data[k]).tobytes()

    with Session(_engine) as session:
        stmt = insert(table).values(**data)
        session.execute(stmt)
        session.commit()


async def __load_data_from_table(
        table,
        conditions,
        _engine
):
    stmt = select(table)
    for column_name, value in conditions.items():
        stmt = stmt.where(getattr(table, column_name) == value)

    with Session(_engine) as session:
        result = session.scalars(stmt).first()

    return result

@validate_call
async def load_id_roof_angle(
        angle,
        _engine,
        _db_url_server

):
    check_type_engine(_engine)
    result = await __load_data_from_table(RoofPitches, {"pitch": angle}, _engine)
    if result is None:
        server_engine = create_engine(_db_url_server)
        result = await __load_data_from_table(RoofPitches, {"pitch": angle}, server_engine)
        server_engine.dispose()
        if result is None:
            return None
        else:
            await __write_id_data_to_table(RoofPitches, {'id_pitch': result.id_pitch,
                                                          'pitch': result.pitch}, _engine)

    return result.id_pitch


@validate_call
async def load_id_wind_azimuth(
        angle,
        _engine,
        _db_url_server

):
    check_type_engine(_engine)
    result = await __load_data_from_table(WindAzimuths, {"wind_azimuth": angle}, _engine)
    if result is None:
        server_engine = create_engine(_db_url_server)
        result = await __load_data_from_table(WindAzimuths, {"wind_azimuth": angle}, server_engine)
        server_engine.dispose()
        if result is None:
            return None
        else:
            await __write_id_data_to_table(WindAzimuths, {'id_wind_azimuth': result.id_wind_azimuth,
                                                          'wind_azimuth': result.wind_azimuth}, _engine)

    return result.id_wind_azimuth


async def __write_experiment(
        experiment,
        _engine,
):
    attributes = experiment.__dict__
    del attributes['_sa_instance_state']
    attributes['pressure_coefficients'] = np.array(attributes['pressure_coefficients']).tobytes()

    with Session(_engine) as session:
        stmt = insert(BuildingWithoutEaves).values(**attributes)
        session.execute(stmt)
        session.commit()


@validate_call
async def __load_experiment(
        id_roof,
        angle,
        id_angle_roof,
        breadth,
        depth,
        height,
        _engine
):
    print(id_roof,
        angle,
        id_angle_roof,
        breadth,
        depth,
        height)
    stmt = select(BuildingWithoutEaves).where(BuildingWithoutEaves.id_roof == id_roof,
                                           BuildingWithoutEaves.angle == angle,
                                           BuildingWithoutEaves.id_pitch == id_angle_roof,
                                           BuildingWithoutEaves.breadth == breadth,
                                           BuildingWithoutEaves.depth == depth,
                                           BuildingWithoutEaves.height == height)

    with Session(_engine) as session:
        experiment = session.scalars(stmt).first()

    return experiment


@validate_call
async def load_experiment(
        id_roof,
        angle,
        id_angle_roof,
        breadth,
        depth,
        height,
        _engine,
        _db_url_server
):
    local_db = True
    experiment = await __load_experiment(id_roof,
                                         angle,
                                         id_angle_roof,
                                         breadth,
                                         depth,
                                         height,
                                         _engine)
    if experiment is None:
        local_db = False
        server_engine = create_engine(_db_url_server)
        experiment = await __load_experiment(id_roof,
                                             angle,
                                             id_angle_roof,
                                             breadth,
                                             depth,
                                             height,
                                             server_engine)
        server_engine.dispose()
        if experiment is None:
            return None
        else:
            await __write_experiment(experiment, _engine)
    return experiment, local_db


@validate_call
async def load_pressure_coefficients(
        id_roof,
        angle,
        id_angle_roof,
        breadth,
        depth,
        height,
        _engine,
        _db_url_server
):
    check_type_engine(_engine)
    experiment, local_db = await load_experiment(
        id_roof,
        angle,
        id_angle_roof,
        breadth,
        depth,
        height,
        _engine,
        _db_url_server
    )
    fc_result = dict()

    fc_result[experiment.angle] = np.frombuffer(experiment.pressure_coefficients, dtype=int).reshape(
        9000, -1) / __SENSOR_VALUES_DISCARD

    return fc_result


@validate_call
async def load_id_coordinates_and_surface(
        angle, id_roof, id_angle_roof, breadth, depth, height,
        _engine,
        _db_url_server
):
    check_type_engine(_engine)

    experiment, local_db = await load_experiment(
        id_roof,
        angle,
        id_angle_roof,
        breadth, depth, height,
        _engine,
        _db_url_server
    )

    return experiment.id_x_coordinates, experiment.id_y_coordinates, experiment.id_surface


@validate_call
async def load_coordinates(
        id_x_coordinates,
        id_y_coordinates,
        _engine,
        _db_url_server
):
    check_type_engine(_engine)

    check_type_engine(_engine)

    result = await __load_data_from_table(XCoordinatesRoof, {"id_x_coordinates": id_x_coordinates}, _engine)

    if result is None:
        server_engine = create_engine(_db_url_server)
        result = await __load_data_from_table(XCoordinatesRoof, {"id_x_coordinates": id_x_coordinates}, server_engine)
        server_engine.dispose()

        if result is None:
            return None
        else:
            await __write_id_data_to_table(XCoordinatesRoof, {'id_x_coordinates': result.id_x_coordinates,
                                                              'x_coordinates': result.x_coordinates}, _engine)

        x = result.x_coordinates
    else:
        x = np.frombuffer(result.x_coordinates, dtype=int)

    result = await __load_data_from_table(YCoordinatesRoof, {"id_y_coordinates": id_y_coordinates}, _engine)

    if result is None:
        server_engine = create_engine(_db_url_server)
        result = await __load_data_from_table(YCoordinatesRoof, {"id_y_coordinates": id_y_coordinates}, server_engine)
        server_engine.dispose()

        if result is None:
            return None
        else:
            await __write_id_data_to_table(YCoordinatesRoof, {'id_y_coordinates': result.id_y_coordinates,
                                                              'y_coordinates': result.y_coordinates}, _engine)

        z = result.y_coordinates

    else:
        z = np.frombuffer(result.y_coordinates, dtype=int)

    return x, z


@validate_call
async def load_face_number(
        id_surface,
        _engine,
        _db_url_server
):
    check_type_engine(_engine)

    result = await __load_data_from_table(SurfacesRoof, {"id_surface": id_surface}, _engine)

    if result is None:
        server_engine = create_engine(_db_url_server)
        result = await __load_data_from_table(SurfacesRoof, {"id_surface": id_surface}, server_engine)
        server_engine.dispose()

        if result is None:
            return None
        else:
            await __write_id_data_to_table(SurfacesRoof, {'id_surface': result.id_surface,
                                                          'surface': result.surface}, _engine)
        # result = np.frombuffer(result.surface, dtype=int)
        face_number = result.surface
    else:
        # result = result.surface
        face_number = np.frombuffer(result.surface, dtype=int)

    return face_number


async def __load_triplets(
id_roof,
        _engine
):
    stmt = select(BuildingWithoutEaves.breadth, BuildingWithoutEaves.depth, BuildingWithoutEaves.height).where(BuildingWithoutEaves.id_roof == id_roof)

    with Session(_engine) as session:
        result = session.execute(stmt).unique().all()

    return result


async def load_triplets(
        id_roof,
        _db_url_server
):
    server_engine = create_engine(_db_url_server)
    result = await __load_triplets(id_roof, server_engine)
    server_engine.dispose()

    return result


if __name__ == "__main__":
    import asyncio

    from sqlalchemy import create_engine

    local_engine = create_engine(r'sqlite:///D:\WindSpectrum\WindSpectrum\windspectrum.db')
    db_url_server = "postgresql://postgres:1234@localhost/postgres"

    # res = asyncio.run(load_pressure_coefficients(1, 1, 16, 24, 18, local_engine, db_url_server))
    res = asyncio.run(load_triplets(db_url_server))
    # res = asyncio.run(load_id_wind_azimuth(0, local_engine, db_url_server))
    print(res)
