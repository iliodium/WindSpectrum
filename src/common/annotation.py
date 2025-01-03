from typing import (Annotated,
                    Literal,
                    Union, )

from pydantic import Field
from sqlalchemy.engine.base import Engine

from src.common.FaceType import FaceType
from src.common.constants import (alpha_standards,
                                  ks10,
                                  wind_regions, )
from src.submodules.plot.common.ChartMode import ChartMode

type AlphaType = Literal[4, 6]
type AlphaOrNoneType = Union[AlphaType | None]

type ExperimentIdType = Annotated[int, Field(gt=0, le=13)]

type AngleType = Annotated[int, Field(ge=0, lt=360)]
type AngleOrNoneType = Union[AngleType | None]

type PositionXType = float
type PositionXOrNoneType = Union[PositionXType, None]

type PositionYType = float
type PositionYOrNoneType = Union[PositionYType, None]

type FaceOrNoneType = Union[FaceType | None]

type ModelSizeType = tuple[float, float, float]
type ModelSizeOrNoneType = Union[ModelSizeType | None]

type ModelNameType = str
type ModelNameIsolatedType = Literal[
    111,
    112,
    113,
    114,
    115,

    212,
    213,
    214,
    215,

    312,
    313,
    314,
    315,
]

type ModelNameIsolatedOrNoneType = Union[ModelNameIsolatedType | None]

type BuildingSizeType = float

type AlphaStandardsType = Literal[*alpha_standards]
type Ks10Type = Literal[*ks10]
type WindRegionsType = Literal[*wind_regions]

type CoordinatesType = Union[tuple[tuple, tuple] | tuple]
type ChartModeType = tuple[ChartMode, ...]

type ModelSizeType = tuple[float, float, float]
type ModelSizeOrNoneType = Union[ModelSizeType | None]


def check_type_engine(engine):
    if not isinstance(engine, Engine):
        raise Exception("engine must be an instance of sqlalchemy.engine.Engine")
