import asyncio

from sqlalchemy import Engine

from src.common.FaceType import FaceType
from src.submodules.databasetoolkit.isolated import list_experiments, load_pressure_coefficients, load_positions


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

    def list_positions(self, experiment_id: int, alpha: int | str, *, load_x: bool = True, load_y: bool = True):
        return asyncio.run(
            load_positions(
                experiment_id,
                alpha,
                self.__engine,
                load_x=load_x,
                load_y=load_y
            )
        )

    def load_pressure_coefficients(
            self, experiment_id: int, alpha: int | str, *, angle: int = None,
            face_number: int | FaceType = None,
            position_x: int | float = None,
            position_y: int | float = None
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
        assert isinstance(engine, Engine), "engine must be an instance of sqlalchemy.engine.Engine"
        # На самом деле эта штука не thread-safe, но мы пока закроем на это глаза
        self.__engine = engine

    async def list_experiments(self):
        return list_experiments(
            self.__engine
        )

    async def list_positions(self, experiment_id: int, alpha: int | str, *, load_x: bool = True, load_y: bool = True):
        return load_positions(
            experiment_id,
            alpha,
            self.__engine,
            load_x=load_x,
            load_y=load_y
        )

    async def load_pressure_coefficients(
            self, experiment_id: int, alpha: int | str, *, angle: int = None,
            face_number: int | FaceType = None,
            position_x: int | float = None,
            position_y: int | float = None
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
