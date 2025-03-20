from typing import Union

import numpy as np
from pydantic import validate_call
from src.common.annotation import ModelNameIsolatedType
from src.common.DbType import DbType


@validate_call
def get_size_tpu_and_count_sensors(
        count_sensors_on_model: int,
        model_name: Union[int | None] = None
):
    """

    :param count_sensors_on_model: len(coordinates[0]) or pressure_coefficients.shape[1]
    :param model_name:
    :return:
    """
    model_name_list = [i for i in list(str(model_name))]
    breadth, depth, height = [int(i) / 10 for i in model_name_list]

    count_sensors_on_middle_row = int(model_name_list[0]) * 5
    count_sensors_on_side_row = int(model_name_list[1]) * 5

    return ((breadth,
             depth,
             height),
            (count_sensors_on_model,
             count_sensors_on_middle_row,
             count_sensors_on_side_row))


def converter_coordinates(
        x_old,
        breadth: float,
        depth: float,
        face_number,
        count_sensors: int,
        accuracy: int = 1
):
    """Возвращает из (x_old) -> (x,y)"""
    x = []
    y = []
    for i in range(count_sensors):
        if face_number[i] == 1:
            x.append(float('%.5f' % (-depth / 2)))
            y.append(float('%.5f' % (breadth / 2 - x_old[i])))
        elif face_number[i] == 2:
            x.append(float('%.5f' % (- depth / 2 + x_old[i] - breadth)))
            y.append(float('%.5f' % (-breadth / 2)))
        elif face_number[i] == 3:
            x.append(float('%.5f' % (depth / 2)))
            y.append(float('%.5f' % (-3 * breadth / 2 + x_old[i] - depth)))
        else:
            x.append(float('%.5f' % (3 * depth / 2 - x_old[i] + 2 * breadth)))
            y.append(float('%.5f' % (breadth / 2)))

    x = np.array(x).round(accuracy)
    y = np.array(y).round(accuracy)

    return x, y


def tpu_size_to_real(
        size,
        building_size,
        tpu_size
):
    return (size / tpu_size) * building_size
