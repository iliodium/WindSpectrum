import json
from typing import (Any,
                    Sequence, )

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
                                   check_type_engine, AngleType, )
from src.submodules.databasetoolkit.orm.models import (ExperimentsAlpha4,
                                                       ExperimentsAlpha6,
                                                       t_models_alpha_4,
                                                       t_models_alpha_6, )

__SENSOR_VALUES_DISCARD = 1000


@validate_call
async def find_experiment_by_model_name(
        model_name: ModelNameIsolatedType,
        alpha: AlphaType,
        _engine,
        _db_url_server
) -> ExperimentsAlpha4 | ExperimentsAlpha6 | None:
    check_type_engine(_engine)

    experiment = await __load_experiments_alpha_by_model_name(model_name, alpha, _engine, _db_url_server)

    return experiment


@validate_call
async def load_experiment_by_id(
        experiment_id: ExperimentIdType,
        alpha: AlphaType,
        _engine
) -> ExperimentsAlpha4 | ExperimentsAlpha6 | None:
    check_type_engine(_engine)

    models_type = ExperimentsAlpha4 if alpha == 4 else ExperimentsAlpha6

    with Session(_engine) as session:
        _m = session.query(models_type).get(experiment_id)
    return _m


# @dataclass
class ExperimentsList(BaseModel):
    # чтобы работали кастомные типы
    model_config = ConfigDict(arbitrary_types_allowed=True)

    alpha_4: Sequence[ExperimentsAlpha4]
    alpha_6: Sequence[ExperimentsAlpha6]


@validate_call
async def list_experiments(
        _engine
) -> ExperimentsList:
    check_type_engine(_engine)

    with Session(_engine) as session:
        result4 = session.scalars(
            select(ExperimentsAlpha4)
            .order_by(ExperimentsAlpha4.model_name)
        ).fetchall()

        result6 = session.scalars(
            select(ExperimentsAlpha6)
            .order_by(ExperimentsAlpha6.model_name)
        ).fetchall()

    return ExperimentsList(alpha_4=result4, alpha_6=result6)


@dataclass(slots=True)
class __FilterPressureCoefficients:
    __experiment_id: ExperimentIdType
    __engine: Any
    __alpha: AlphaType
    __face_number: FaceOrNoneType
    __position_x: PositionXOrNoneType
    __position_y: PositionYOrNoneType

    def __call__(
            self,
            *args,
            **kwargs
    ) -> numpy.ndarray:
        if len(args) != 1 or len(kwargs) != 0:
            raise NotImplementedError(f"Unexpected args: {args} kwargs: {kwargs}")

        value = args[0]

        if not isinstance(value, numpy.ndarray):
            raise NotImplementedError(f"Unexpected value: {value}. Should be numpy.ndarray")

        effective_fn = self.__face_number.value

        experiment_description: ExperimentsAlpha4 | ExperimentsAlpha6

        match self.__alpha:
            case 4:
                stmt = select(ExperimentsAlpha4).where(ExperimentsAlpha4.model_id == self.__experiment_id)

                with Session(self.__engine) as session:
                    experiment_description: ExperimentsAlpha4 = session.scalars(stmt).first()

                if experiment_description is None:
                    raise ValueError(f"Experiment with id {self.__experiment_id} not found")
            case 6:
                stmt = select(ExperimentsAlpha6).where(ExperimentsAlpha6.model_id == self.__experiment_id)

                with Session(self.__engine) as session:
                    experiment_description: ExperimentsAlpha6 = session.scalars(stmt).first()

                if experiment_description is None:
                    raise ValueError(f"Experiment with id {self.__experiment_id} not found")

        fn = numpy.array(experiment_description.face_number)
        positions_x = numpy.array(experiment_description.x_coordinates)
        positions_y = numpy.array(experiment_description.z_coordinates)

        _numpy_query = None

        if effective_fn is not None:
            _numpy_query = fn == effective_fn

        if self.__position_x is not None:
            _numpy_query = positions_x == self.__position_x if _numpy_query is None else _numpy_query & (
                    positions_x == self.__position_x)

        if self.__position_y is not None:
            _numpy_query = positions_y == self.__position_y if _numpy_query is None else _numpy_query & (
                    positions_y == self.__position_y)

        if _numpy_query is None:
            return value

        _res = value[:, _numpy_query]

        if len(_res.shape) == 2:
            if _res.shape[1] == 1:
                return _res[:, 0]

        return _res


def __identity(
        el
):
    return el


@validate_call
async def load_positions(
        experiment_id: ExperimentIdType,
        alpha: AlphaType,
        _engine,
        _db_url_server,
        *,
        load_x: bool = True,
        load_y: bool = True,
) -> CoordinatesType:
    check_type_engine(_engine)

    if not (load_x or load_y):
        raise ValueError("load_x or load_y must be True")

    experiment = __load_experiments_alpha(experiment_id, alpha, _engine, _db_url_server, local_db=True)

    if load_x and load_y:
        return experiment.x_coordinates, experiment.z_coordinates

    if load_x:
        return experiment.x_coordinates

    if load_y:
        return experiment.z_coordinates


def __load_experiments_alpha(
        experiment_id: ExperimentIdType,
        alpha,
        _engine,
        _db_url_server,
        local_db: bool = False

):
    match alpha:
        case 4:
            experiments_alpha = ExperimentsAlpha4
        case 6:
            experiments_alpha = ExperimentsAlpha6

    stmt = select(experiments_alpha).where(experiments_alpha.model_id == experiment_id)

    with Session(_engine) as session:
        experiment = session.scalars(stmt).first()

    if experiment is None:
        server_engine = create_engine(_db_url_server)
        with Session(server_engine) as session:
            experiment = session.scalars(stmt).first()
        server_engine.dispose()

        if experiment is not None:
            __write_experiments_alpha(experiments_alpha, experiment, _engine)

    if experiment is None:
        raise ValueError(f"Experiment with id {experiment_id} not found")

    if local_db:
        experiment.x_coordinates = np.frombuffer(experiment.x_coordinates, dtype=float)
        experiment.z_coordinates = np.frombuffer(experiment.z_coordinates, dtype=float)
        experiment.face_number = np.frombuffer(experiment.face_number, dtype=int)

    return experiment


async def __load_experiments_alpha_by_model_name(
        model_name: ModelNameIsolatedType,
        alpha,
        _engine,
        _db_url_server,
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
        server_engine = create_engine(_db_url_server)
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
    stmt = select(models_type).where(models_type.c.model_id == experiment_id,
                                     models_type.c.angle == angle)

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
async def load_pressure_coefficients(
        experiment_id: ExperimentIdType,
        alpha: AlphaType,
        _engine,
        _db_url_server,
        *,
        angle: AngleType = None,
        face_number: FaceOrNoneType = None,
        position_x: PositionXOrNoneType = None,
        position_y: PositionYOrNoneType = None
):
    check_type_engine(_engine)

    match alpha:
        case 4:
            models_alpha = t_models_alpha_4
        case 6:
            models_alpha = t_models_alpha_6

    result = await __load_pressure_coefficients_for_type_and_alpha(
        experiment_id,
        models_alpha,
        alpha,
        _engine=_engine,
        angle=angle,
        face_number=face_number,
        position_x=position_x,
        position_y=position_y,
        local_db=True
    )

    if result is None:
        server_engine = create_engine(_db_url_server)
        result = await __load_pressure_coefficients_for_type_and_alpha(
            experiment_id,
            models_alpha,
            alpha,
            _engine=server_engine,
            angle=angle,
            face_number=face_number,
            position_x=position_x,
            position_y=position_y,
            local_db=False
        )
        server_engine.dispose()

        if result is None:
            return None
        else:
            __write_models_alpha(models_alpha, experiment_id, angle, result[angle], _engine)

    return result


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


@validate_call
async def load_face_number(
        experiment_id: ExperimentIdType,
        alpha: AlphaType,
        _engine,
        db_url_server

):
    check_type_engine(_engine)

    experiment = __load_experiments_alpha(experiment_id, alpha, _engine, db_url_server, local_db=True)

    return experiment.face_number


if __name__ == "__main__":
    import asyncio

    from sqlalchemy import create_engine

    local_engine = create_engine('sqlite:///windspectrum.db')
    # server_engine = create_engine(_db_url_server)
    # for angle in range(0,50, 5):
    #     res = asyncio.run(load_pressure_coefficients(1, 4, local_engine, angle=angle))
    #     print(res)
    #     # res = asyncio.run(load_pressure_coefficients(1, 4, local_engine))
    #
    # print(res)
    # print(len(res))

    # for i in range(1, 14):
    #     res = asyncio.run(load_positions(i, 4, local_engine))
    #     print(res)

    res = asyncio.run(find_experiment_by_model_name(112, 4, local_engine))
    print(res.model_id)
