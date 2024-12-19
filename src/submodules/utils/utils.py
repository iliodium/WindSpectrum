from typing import (Union, )

from pydantic import validate_call

from src.common.DbType import DbType
from src.common.annotation import (ModelNameIsolatedType, )


@validate_call
def get_size_and_count_sensors(
        pressure_coefficients_shape: int,
        model_name: Union[ModelNameIsolatedType | None] = None,
        height: float | None = None,
        db: DbType = DbType.ISOLATED
):
    match db:
        case DbType.ISOLATED:
            model_name_list = [i for i in list(str(model_name))]
            breadth, depth, height = [int(i) / 10 for i in model_name_list]

            count_sensors_on_model = pressure_coefficients_shape
            count_sensors_on_middle_row = int(model_name_list[0]) * 5
            count_sensors_on_side_row = int(model_name_list[1]) * 5

        case DbType.INTERFERENCE:
            assert height is not None and isinstance(height, float), \
                'height must be not None float when db == DbType.INTERFERENCE'
            breadth, depth = 0.07, 0.07

            count_sensors_on_model = pressure_coefficients_shape
            count_sensors_on_middle_row = 7
            count_sensors_on_side_row = 7

    return ((breadth,
             depth,
             height),
            (count_sensors_on_model,
             count_sensors_on_middle_row,
             count_sensors_on_side_row))
