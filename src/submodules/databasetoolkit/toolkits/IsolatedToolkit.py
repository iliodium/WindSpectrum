import asyncio

from pydantic import validate_call
from sqlalchemy import Engine

from src.common.annotation import (AlphaType,
                                   AngleOrNoneType,
                                   ExperimentIdType,
                                   FaceOrNoneType,
                                   PositionXOrNoneType,
                                   PositionYOrNoneType,
                                   check_type_engine,)
from src.submodules.databasetoolkit.isolated import (list_experiments,
                                                     load_positions,
                                                     load_pressure_coefficients,)


class IsolatedToolkit:
    __engine: Engine

    __slots__ = [
        '__engine'
    ]

    def __init__(self, engine: Engine):
        assert isinstance(engine, Engine), "engine must be an instance of sqlalchemy.engine.Engine"
        # На самом деле эта штука не thread-safe, но мы пока закроем на это глаза
        self.__engine = engine

    def list_experiments(self):
        return asyncio.run(
            list_experiments(
                self.__engine
            )
        )

    @validate_call
    def list_positions(self,
                       experiment_id: ExperimentIdType,
                       alpha: AlphaType,
                       *,
                       load_x: bool = True,
                       load_y: bool = True):
        return asyncio.run(
            load_positions(
                experiment_id,
                alpha,
                self.__engine,
                load_x=load_x,
                load_y=load_y
            )
        )

    @validate_call
    def load_pressure_coefficients(
            self,
            experiment_id: ExperimentIdType,
            alpha: AlphaType,
            *,
            angle: AngleOrNoneType,
            face_number: FaceOrNoneType = None,
            position_x: PositionXOrNoneType = None,
            position_y: PositionYOrNoneType = None
    ):
        return asyncio.run(
            load_pressure_coefficients(
                experiment_id,
                alpha,
                self.__engine,
                angle=angle,
                face_number=face_number,
                position_x=position_x,
                position_y=position_y
            )
        )


class AsyncIsolatedToolkit:
    __engine: Engine

    __slots__ = [
        '__engine'
    ]

    def __init__(self, engine: Engine):
        check_type_engine(engine)
        # На самом деле эта штука не thread-safe, но мы пока закроем на это глаза
        self.__engine = engine

    async def list_experiments(self):
        return list_experiments(
            self.__engine
        )

    @validate_call
    async def list_positions(self,
                             experiment_id: ExperimentIdType,
                             alpha: AlphaType,
                             *,
                             load_x: bool = True,
                             load_y: bool = True):
        return load_positions(
            experiment_id,
            alpha,
            self.__engine,
            load_x=load_x,
            load_y=load_y
        )

    @validate_call
    async def load_pressure_coefficients(
            self,
            experiment_id: ExperimentIdType,
            alpha: AlphaType,
            *,
            angle: AngleOrNoneType = None,
            face_number: FaceOrNoneType = None,
            position_x: PositionXOrNoneType = None,
            position_y: PositionYOrNoneType = None
    ):
        return load_pressure_coefficients(
            experiment_id,
            alpha,
            self.__engine,
            angle=angle,
            face_number=face_number,
            position_x=position_x,
            position_y=position_y
        )
