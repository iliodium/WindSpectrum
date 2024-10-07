from typing import (Any,
                    Sequence,)

import numpy
from pydantic import validate_call
from pydantic.dataclasses import dataclass
from sqlalchemy import select
from sqlalchemy.orm import Session
from src.common.annotation import (AlphaType, AngleOrNoneType,
                                   ExperimentIdType, FaceOrNoneType,
                                   ModelNameIsolatedType,
                                   PositionXOrNoneType,
                                   PositionXType,
                                   PositionYOrNoneType,
                                   PositionYType,
                                   check_type_engine,)
from src.common.FaceType import FaceType
from src.submodules.databasetoolkit.orm.models import (ExperimentsAlpha4,
                                                       ExperimentsAlpha6,
                                                       t_models_alpha_4,
                                                       t_models_alpha_6,)

__SENSOR_VALUES_DISCARD = 1000


@validate_call
async def find_experiment_by_model_name(
        model_name: ModelNameIsolatedType,
        alpha: AlphaType,
        _engine
) -> ExperimentsAlpha4 | ExperimentsAlpha6 | None:
    check_type_engine(_engine)

    models_type = ExperimentsAlpha4 if alpha == 4 else ExperimentsAlpha6

    stmt = select(models_type).where(models_type.model_name == model_name)

    with Session(_engine) as session:
        result = list(session.execute(stmt).scalars())

    if len(result) == 0:
        return None

    return result[0]


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


@dataclass(slots=True)
class ExperimentsList:
    __alpha_4: Sequence[ExperimentsAlpha4]
    __alpha_6: Sequence[ExperimentsAlpha6]


async def list_experiments(_engine) -> ExperimentsList:
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

    return ExperimentsList(result4, result6)


@dataclass(slots=True)
class __FilterPressureCoefficients:
    __experiment_id: ExperimentIdType
    __engine: Any
    __alpha: AlphaType
    __face_number: FaceType
    __position_x: PositionXType
    __position_y: PositionYType

    def __call__(self, *args,
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


def __identity(el):
    return el


@validate_call
async def __load_pressure_coefficients_for_type_and_alpha(
        experiment_id: ExperimentIdType,
        models_type,
        alpha: AlphaType,
        _engine,
        *,
        angle: AngleOrNoneType = None,
        face_number: FaceOrNoneType = None,
        position_x: PositionXOrNoneType = None,
        position_y: PositionYOrNoneType = None
) -> dict[int, numpy.ndarray] | None:
    stmt = select(models_type).where(models_type.c.model_id == experiment_id)

    if angle is not None:
        stmt = stmt.where(models_type.c.angle == angle)

    with Session(_engine) as session:
        result = session.execute(stmt).fetchall()

    if result is None:
        return None

    if len(result) == 0:
        return None

    _mapper = __identity

    if face_number is not None or position_x is not None or position_y is not None:
        _mapper = __FilterPressureCoefficients(experiment_id, _engine, alpha, face_number, position_x, position_y)

    fc_result = dict()

    for row in result:
        fc_result[row.angle] = _mapper(numpy.array(row.pressure_coefficients, dtype=float)) / __SENSOR_VALUES_DISCARD

    return fc_result


@validate_call
async def __load_pressure_coefficients_alpha_4(
        experiment_id: ExperimentIdType,
        _engine,
        *,
        angle: AngleOrNoneType = None,
        face_number: FaceOrNoneType = None,
        position_x: PositionXOrNoneType = None,
        position_y: PositionYOrNoneType = None
):
    return await __load_pressure_coefficients_for_type_and_alpha(
        experiment_id,
        t_models_alpha_4,
        4,
        _engine,
        angle=angle,
        face_number=face_number,
        position_x=position_x,
        position_y=position_y
    )


@validate_call
async def load_positions(
        experiment_id: ExperimentIdType,
        alpha: AlphaType,
        _engine,
        *,
        load_x: bool = True,
        load_y: bool = True
) -> tuple[numpy.ndarray, numpy.ndarray] | numpy.ndarray:
    check_type_engine(_engine)

    if not (load_x or load_y):
        raise ValueError("load_x or load_y must be True")

    experiment_description: ExperimentsAlpha4 | ExperimentsAlpha6

    if alpha == 4:
        stmt = select(ExperimentsAlpha4).where(ExperimentsAlpha4.model_id == experiment_id)

        with Session(_engine) as session:
            experiment_description: ExperimentsAlpha4 = session.scalars(stmt).first()

        if experiment_description is None:
            raise ValueError(f"Experiment with id {experiment_id} not found")
    elif alpha == 6:
        stmt = select(ExperimentsAlpha6).where(ExperimentsAlpha6.model_id == experiment_id)

        with Session(_engine) as session:
            experiment_description: ExperimentsAlpha6 = session.scalars(stmt).first()

        if experiment_description is None:
            raise ValueError(f"Experiment with id {experiment_id} not found")

    if load_x and load_y:
        return experiment_description.x_coordinates, experiment_description.z_coordinates

    if load_x:
        return experiment_description.x_coordinates

    if load_y:
        return experiment_description.z_coordinates


@validate_call
async def __load_pressure_coefficients_alpha_6(
        experiment_id: ExperimentIdType,
        _engine,
        *,
        angle: AngleOrNoneType = None,
        face_number: FaceOrNoneType = None,
        position_x: PositionXOrNoneType = None,
        position_y: PositionYOrNoneType = None
):
    return await __load_pressure_coefficients_for_type_and_alpha(
        experiment_id,
        t_models_alpha_6,
        6,
        _engine,
        angle=angle,
        face_number=face_number,
        position_x=position_x,
        position_y=position_y
    )


@validate_call
async def load_pressure_coefficients(
        experiment_id: ExperimentIdType,
        alpha: AlphaType,
        _engine,
        *,
        angle: AngleOrNoneType = None,
        face_number: FaceOrNoneType = None,
        position_x: PositionXOrNoneType = None,
        position_y: PositionYOrNoneType = None
):
    check_type_engine(_engine)

    if alpha == 4:
        return await __load_pressure_coefficients_alpha_4(
            experiment_id,
            _engine,
            angle=angle,
            face_number=face_number,
            position_x=position_x,
            position_y=position_y
        )
    elif alpha == 6:
        return await __load_pressure_coefficients_alpha_6(
            experiment_id,
            _engine,
            angle=angle,
            face_number=face_number,
            position_x=position_x,
            position_y=position_y
        )
    else:
        raise RuntimeError(f"alpha must be 4 or 6, not {alpha}")


if __name__ == "__main__":
    import asyncio

    from sqlalchemy import create_engine

    # engine = create_engine("postgresql://postgres:password@localhost:15432/postgres")
    # engine = create_engine("postgresql://postgres:dSJJNjkn42384*$(#@92.246.143.110:5432/windspectrum_db")
    engine = create_engine("postgresql://postgres:1234@localhost/postgres")

    res = asyncio.run(
        load_pressure_coefficients(1, 6, engine, angle=0, face_number=FaceType.ON_WIND, position_x=0.05,
                                   position_y=0.07))

    for i in res.keys():
        print(i, res[i], res[i].shape)

    res = asyncio.run(
        load_pressure_coefficients(1, 4, engine, angle=0, face_number=FaceType.ON_WIND, position_x=0.05))

    for i in res.keys():
        print(i, res[i], res[i].shape)
