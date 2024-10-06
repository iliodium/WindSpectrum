from typing import Annotated

from pydantic import (Field,
                      validate_call,)

from src.common.annotation import AngleType
from src.common.PermutationView import PermutationView
from src.common.TypeOfBasement import TypeOfBasement


@validate_call
def get_view_permutation_data(type_base: TypeOfBasement,
                              angle: AngleType) -> PermutationView:
    """Определяет порядок данных для перестановки.
    reverse = нужно перевернуть [1, 2, 3] -> [3, 2, 1]
    forward = не нужно перевернуть [1, 2, 3] -> [1, 2, 3]"""

    if type_base == TypeOfBasement.SQUARE:  # в основании квадрат
        if any([45 < angle < 90, 135 < angle < 180, 225 < angle < 270, 315 < angle < 360]):
            return PermutationView.REVERSE
        else:
            return PermutationView.FORWARD

    elif type_base == TypeOfBasement.RECTANGLE:  # в основании прямоугольник
        if any([90 < angle < 180, 270 < angle < 360]):
            return PermutationView.REVERSE
        else:
            return PermutationView.FORWARD


@validate_call
def get_sequence_permutation_data(type_base: TypeOfBasement,
                                  permutation_view: PermutationView,
                                  angle: Annotated[AngleType, Field(ge=0, lt=360)]):
    """Определяет как менять расстановку датчиков"""

    if type_base == TypeOfBasement.SQUARE:
        if permutation_view == PermutationView.REVERSE:
            if 45 < angle < 90:
                return (1, 0, 3, 2)
            elif 135 < angle < 180:
                return (2, 1, 0, 3)
            elif 225 < angle < 270:
                return (3, 2, 1, 0)
            elif 315 < angle < 360:
                return (0, 3, 1, 2)
            else:
                raise ValueError('angle must be in one of the following ranges: 45 < angle < 90,'
                                 ' 135 < angle < 180, 225 < angle < 270, 315 < angle < 360')
        elif permutation_view == PermutationView.FORWARD:
            if 0 <= angle <= 45:
                return (0, 1, 2, 3)
            elif 90 <= angle <= 135:
                return (3, 0, 1, 2)
            elif 180 <= angle <= 225:
                return (2, 3, 0, 1)
            elif 270 <= angle <= 315:
                return (1, 2, 3, 0)
            else:
                raise ValueError('angle must be in one of the following ranges: 0 <= angle <= 45,'
                                 ' 90 <= angle <= 135, 180 <= angle <= 225, 270 <= angle <= 315')

    elif type_base == TypeOfBasement.RECTANGLE:
        if permutation_view == PermutationView.REVERSE:
            if 90 < angle < 180:
                return (2, 1, 0, 3)
            elif 270 < angle < 360:
                return (0, 3, 2, 1)
            else:
                raise ValueError('angle must be in one of the following ranges: 90 < angle < 180,'
                                 ' 270 < angle < 360')
        elif permutation_view == PermutationView.FORWARD:
            if 0 <= angle <= 90:
                return (0, 1, 2, 3)
            elif 180 <= angle <= 270:
                return (2, 3, 0, 1)
            else:
                raise ValueError('angle must be in one of the following ranges: 0 <= angle <= 90,'
                                 ' 180 <= angle <= 270')
