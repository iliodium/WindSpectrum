from typing import (Literal,
                    Tuple,
                    Union,)

from sqlalchemy.engine.base import Engine

from src.common.constants import (alpha_standards,
                                  ks10,
                                  wind_regions,)
from src.common.FaceType import FaceType

type AlphaType = Literal[4, 6]
type ExperimentIdType = int

type AngleType = int
type AngleOrNoneType = Union[AngleType | None]

type PositionXType = float
type PositionXOrNoneType = Union[PositionXType, None]

type PositionYType = float
type PositionYOrNoneType = Union[PositionYType, None]

type FaceOrNoneType = Union[FaceType | None]

type ModelSizeType = Tuple[float, float, float]
type ModelSizeOrNoneType = Union[ModelSizeType | None]

type ModelNameType = str
type ModelNameIsolatedType = Literal[
    '111',
    '112',
    '113',
    '114',
    '115',

    '212',
    '213',
    '214',
    '215',

    '312',
    '313',
    '314',
    '315',
]

type ModelNameIsolatedOrNoneType = Union[ModelNameIsolatedType | None]

type BuildingSizeType = float

type AlphaStandardsType = Literal[*alpha_standards]
type Ks10Type = Literal[*ks10]
type WindRegionsType = Literal[*wind_regions]


def check_type_engine(engine):
    if not isinstance(engine, Engine):
        raise Exception("engine must be an instance of sqlalchemy.engine.Engine")


'''
from src.common.annotation import AngleType


        check_type_engine(_engine)

        experiment_id: ExperimentIdType,
        alpha: AlphaType,
        _engine,
        *,
        angle: AngleType = None,
        face_number: FaceType = None,
        position_x: PositionXType = None,
        position_y: PositionYType = None

@validate_call
'''
'''
__FilterPressureCoefficients src/submodules/databasetoolkit/isolated.py
типы переписал на свои, но не бросает ошибку, если что то не так при создании тк класс
для чего он нужен ?

get_view_permutation_data src/submodules/utils/permutations.py
нужно ли ограничение что угол 0<=angle<360 ?
тк если пользователь вводит >=360, то мы угол % 360 всегда
тк 0 == 360 == 720 и тд
можно так то это ограничение глобально задать 
type AngleType = int
Annotated[AngleType, Field(ge=0, lt=360)]

calculate_cmz src/submodules/science/integration.py
не работает валидация на _pressure_coefficients: np.ndarray,










'''
