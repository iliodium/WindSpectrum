from typing import Tuple, List

import numpy as np

from src.common.PermutationView import PermutationView
from src.common.TypeOfBasement import TypeOfBasement


def get_base_angle(angle: int, permutation_view: PermutationView, type_base: TypeOfBasement = TypeOfBasement.SQUARE):
    if permutation_view == PermutationView.REVERSE:
        return 90 * (angle // 90 + 1) - angle
    elif permutation_view == PermutationView.FORWARD:
        if type_base == TypeOfBasement.SQUARE:
            return angle % 90
        elif type_base == TypeOfBasement.RECTANGLE and angle == 270:
            return 90
        else:
            return angle % 90
    else:
        raise ValueError('permutation_view must be one of PermutationView enum')


def changer_sequence_coefficients(coefficients: np.ndarray,
                                  permutation_view: PermutationView,
                                  model_name: str,
                                  sequence_permutation: Tuple[int, int, int, int]):
    """Меняет порядок следования датчиков тем самым генерируя не существующие углы"""
    assert isinstance(coefficients, np.ndarray), 'coefficients must be np.ndarray'
    assert isinstance(permutation_view, PermutationView), 'permutation_view must be PermutationView'
    f1, f2, f3, f4 = sequence_permutation

    count_sensors_on_middle = int(model_name[0]) * 5
    count_sensors_on_side = int(model_name[1]) * 5

    count_sensors_on_model = len(coefficients[0])
    count_row = count_sensors_on_model // (2 * (count_sensors_on_middle + count_sensors_on_side))

    count_of_times_in_seria = coefficients.shape[0]

    if permutation_view == PermutationView.FORWARD:
        arr = coefficients.reshape((count_of_times_in_seria, count_row, -1))
        arr = np.split(arr, [count_sensors_on_middle,
                             count_sensors_on_middle + count_sensors_on_side,
                             2 * count_sensors_on_middle + count_sensors_on_side,
                             2 * (count_sensors_on_middle + count_sensors_on_side)
                             ], axis=2)
        coefficients = np.concatenate((arr[f1],
                                       arr[f2],
                                       arr[f3],
                                       arr[f4]), axis=2).reshape((count_of_times_in_seria, count_sensors_on_model))
    else:
        arr = coefficients.reshape((count_of_times_in_seria, count_row, -1))
        arr = np.split(arr, [count_sensors_on_middle,
                             count_sensors_on_middle + count_sensors_on_side,
                             2 * count_sensors_on_middle + count_sensors_on_side,
                             2 * (count_sensors_on_middle + count_sensors_on_side)
                             ], axis=2)
        coefficients = np.concatenate((np.flip(arr[f1], axis=2),
                                       np.flip(arr[f2], axis=2),
                                       np.flip(arr[f3], axis=2),
                                       np.flip(arr[f4], axis=2)), axis=2).reshape(
            (count_of_times_in_seria, count_sensors_on_model))

    return coefficients


def changer_sequence_numbers(numbers: List[int],
                             model_name: str,
                             sequence_permutation: Tuple[int, int, int, int]):
    f1, f2, f3, f4 = sequence_permutation
    count_sensors_on_middle = int(model_name[1]) * 5
    count_sensors_on_side = int(model_name[0]) * 5
    count_sensors_on_model = len(numbers)
    count_row = count_sensors_on_model // (2 * (count_sensors_on_middle + count_sensors_on_side))

    arr = np.array(numbers).reshape((count_row, -1))
    arr = np.split(arr, [count_sensors_on_middle,
                         count_sensors_on_middle + count_sensors_on_side,
                         2 * count_sensors_on_middle + count_sensors_on_side,
                         2 * (count_sensors_on_middle + count_sensors_on_side)
                         ], axis=1)

    numbers = np.concatenate((arr[f1],
                              arr[f2],
                              arr[f3],
                              arr[f4]), axis=1).reshape(count_sensors_on_model)

    return numbers


if __name__ == "__main__":
    from sqlalchemy import create_engine
    from src.submodules.databasetoolkit.isolated import load_pressure_coefficients
    import asyncio

    engine = create_engine("postgresql://postgres:password@localhost:15432/postgres")

    res = asyncio.run(
        load_pressure_coefficients(5, 6, engine, angle=0))

    pressure_coefficients = res[0]

    print(
        changer_sequence_coefficients(
            pressure_coefficients,
            PermutationView.REVERSE,
            '115',
            (2, 3, 0, 1)
        )
    )
