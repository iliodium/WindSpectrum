import os

import numba
import numpy as np
from numba import jit
from numba.pycc import CC

# получить имя файла
file_name = os.path.splitext(os.path.basename(__file__))[0]
pyd_name = f'aot_{file_name}'

cc = CC(pyd_name)
cc.output_dir = os.path.dirname(os.path.realpath(__file__))
cc.output_file = f'{pyd_name}.pyd'


@jit
@cc.export('_add_borders_to_coordinates_x',
           numba.float64[:, ::1](
               numba.float64[::1],  # coordinates
               numba.int64,  # count_sensors_on_row
               numba.int64,  # count_row
               numba.float64,  # breadth
               numba.float64  # depth
           ))
def _add_borders_to_coordinates_x(
        coordinates,
        count_sensors_on_row,
        count_row,
        breadth,
        depth
):
    x = np.reshape(coordinates, (-1, count_sensors_on_row))
    # добавление 0 в начало каждой строки и добавление 2 * (breadth + depth) в конец каждой строки
    x = np.column_stack((np.zeros(count_row), x, np.full(count_row, 2 * (breadth + depth))))

    return x


@jit
@cc.export('_add_borders_to_coordinates_y',
           numba.float64[:, ::1](
               numba.float64[::1],  # coordinates
               numba.int64,  # count_sensors_on_row
               numba.float64  # height
           ))
def _add_borders_to_coordinates_y(
        coordinates,
        count_sensors_on_row,
        height
):
    y = np.reshape(coordinates, (-1, count_sensors_on_row))
    # добавление _height в начало каждого вектора и добавление 0 в конец каждого вектора
    y = np.vstack((np.full((1, count_sensors_on_row), height), y, np.zeros((1, count_sensors_on_row))))

    return y


# Рассчитываем область влияние(площадь) сенсора на модели


@jit
@cc.export('_calculate_sensors_area_effect',
           (
                   numba.float64[:, ::1],  # x
                   numba.float64[:, ::1],  # y
                   numba.int64,  # count_row
                   numba.int64,  # count_sensors_on_row
                   numba.int64,  # count_sensors_on_middle_row
                   numba.int64  # count_sensors_on_side_row

           ))
def _calculate_sensors_area_effect(
        x,
        y,
        count_row,
        count_sensors_on_row,
        count_sensors_on_middle_row,
        count_sensors_on_side_row
):
    squares_faces = np.zeros((count_row, count_sensors_on_row))
    for y_i in range(count_row):
        for x_i in range(count_sensors_on_row):
            y_t = y[y_i][x_i]
            y_m = y[y_i + 1][x_i]
            y_b = y[y_i + 2][x_i]

            if y_i == 0:
                dy = y_t - y_m + (y_m - y_b) / 2
            elif y_i == count_row - 1:
                dy = (y_t - y_m) / 2 + y_m - y_b
            else:
                dy = (y_t - y_m) / 2 + (y_m - y_b) / 2

            x_l = x[y_i][x_i]
            x_m = x[y_i][x_i + 1]
            x_r = x[y_i][x_i + 2]

            if x_i == 0:
                dx = x_m - x_l + (x_r - x_m) / 2
            elif x_i == count_sensors_on_row - 1:
                dx = (x_m - x_l) / 2 + x_r - x_m
            else:
                dx = (x_m - x_l) / 2 + (x_r - x_m) / 2

            squares_faces[y_i, x_i] += dy * dx

    squares_faces = np.split(squares_faces, [count_sensors_on_middle_row,
                                             count_sensors_on_middle_row + count_sensors_on_side_row,
                                             2 * count_sensors_on_middle_row + count_sensors_on_side_row,
                                             2 * (count_sensors_on_middle_row + count_sensors_on_side_row)
                                             ], axis=1)

    return squares_faces[0], squares_faces[1]


@jit
@cc.export('_split_pressure_coefficients',
           (
                   numba.int64,  # count_sensors_on_model
                   numba.int64,  # count_sensors_on_middle_row
                   numba.int64,  # count_sensors_on_side_row
                   numba.float64[:, ::1]  # _pressure_coefficients
           ))
def _split_pressure_coefficients(
        count_sensors_on_model,
        count_sensors_on_middle_row,
        count_sensors_on_side_row,
        pressure_coefficients
):
    count_row = count_sensors_on_model // (2 * (count_sensors_on_middle_row + count_sensors_on_side_row))

    pc_copy = np.reshape(pressure_coefficients, (pressure_coefficients.shape[0], count_row, -1))
    pc_copy = np.split(pc_copy, [count_sensors_on_middle_row,
                                 count_sensors_on_middle_row + count_sensors_on_side_row,
                                 2 * count_sensors_on_middle_row + count_sensors_on_side_row,
                                 2 * (count_sensors_on_middle_row + count_sensors_on_side_row)
                                 ], axis=2)

    return pc_copy[0], pc_copy[1], pc_copy[2], pc_copy[3]


@jit
@cc.export('_calculate_projection_on_the_axis',
           numba.float64(
               numba.float64,  # breadth
               numba.float64,  # depth
               numba.int64  # angle
           ))
def _calculate_projection_on_the_axis(
        breadth,
        depth,
        angle
):
    projection_on_the_axis = np.cos(np.deg2rad(angle)) * breadth + np.sin(np.deg2rad(angle)) * depth
    return projection_on_the_axis


@jit
@cc.export('_get_mid_of__face',
           (
                   numba.float64,  # breadth
                   numba.float64,  # depth
           ))
def _get_mid_of_face(
        breadth,
        depth,
):
    # центры граней
    mid13_x = breadth / 2
    mid24_x = depth / 2

    return mid13_x, mid24_x


@jit
@cc.export('_calculate_mxs',
           (
                   numba.int64,  # count_sensors_on_model
                   numba.int64,  # count_sensors_on_middle_row
                   numba.int64,  # count_sensors_on_side_row
                   numba.float64[::1],  # _coordinate_x_for_mxs
                   numba.float64,  # breadth
                   numba.float64  # depth
           ))
def _calculate_mxs(
        count_sensors_on_model,
        count_sensors_on_middle_row,
        count_sensors_on_side_row,
        coordinate_x,
        breadth,
        depth,
):
    count_row = count_sensors_on_model // (2 * (count_sensors_on_middle_row + count_sensors_on_side_row))
    x_for_mxs = np.reshape(coordinate_x, (count_row, -1))
    x_for_mxs = np.split(x_for_mxs, [count_sensors_on_middle_row,
                                     count_sensors_on_middle_row + count_sensors_on_side_row,
                                     2 * count_sensors_on_middle_row + count_sensors_on_side_row,
                                     2 * (count_sensors_on_middle_row + count_sensors_on_side_row)
                                     ], axis=1)

    del x_for_mxs[4]

    # центры граней
    mid13_x, mid24_x = _get_mid_of_face(breadth, depth)

    # плечи
    # разделение плечей на 4 массива, тк возможны варианты когда векторы массива разной длины,
    # а в np.array все векторы должны быть одинаковой длины
    mxs1 = x_for_mxs[0] - mid13_x
    mxs2 = x_for_mxs[1] - breadth - mid24_x

    return mxs1, mxs2


# Площадь поверхности грани(лицевой и боковой)


@jit
@cc.export('_calculate_square_of_face',
           (
                   numba.float64,  # breadth
                   numba.float64,  # depth
                   numba.float64  # height
           ))
def _calculate_square_of_face(
        breadth,
        depth,
        height
):
    s13 = breadth * height
    s24 = depth * height
    return s13, s24


@jit
@cc.export('_calculate_cmz',
           numba.float64[:](
               numba.float64,  # projection_on_the_axis

               numba.float64,  # square13
               numba.float64,  # square24

               numba.float64[:, :],  # sensors_area_effect13
               numba.float64[:, :],  # sensors_area_effect24

               numba.float64[:, :],  # mxs13
               numba.float64[:, :],  # mxs24

               numba.float64[:, :, :],  # pc13
               numba.float64[:, :, :],  # pc13
               numba.float64[:, :, :],  # pc13
               numba.float64[:, :, :],  # pc13
           ))
def _calculate_cmz(
        projection_on_the_axis,
        square13,
        square24,
        sensors_area_effect13,
        sensors_area_effect24,
        mxs13,
        mxs24,
        pc1,
        pc2,
        pc3,
        pc4,
):
    cmz = np.zeros(pc1.shape[0])

    divisor13 = square13 * projection_on_the_axis
    divisor24 = square24 * projection_on_the_axis

    for pc in (pc1, pc3):
        cmz += (np.sum(np.sum(pc * sensors_area_effect13 * mxs13, axis=1), axis=1) / divisor13)

    for pc in (pc2, pc4):
        cmz += (np.sum(np.sum(pc * sensors_area_effect24 * mxs24, axis=1), axis=1) / divisor24)

    return cmz


@jit
@cc.export('_calculate_cx_cy',
           (
                   numba.float64,  # square13
                   numba.float64,  # square24

                   numba.float64[:, :],  # sensors_area_effect13
                   numba.float64[:, :],  # sensors_area_effect24

                   numba.float64[:, ::1],  # pc1
                   numba.float64[:, ::1],  # pc2
                   numba.float64[:, ::1],  # pc3
                   numba.float64[:, ::1],  # pc4
           ))
def _calculate_cx_cy(
        square13,
        square24,
        sensors_area_effect13,
        sensors_area_effect24,
        pc1,
        pc2,
        pc3,
        pc4
):
    cx = np.zeros(pc1.shape[0])
    cy = np.zeros(pc1.shape[0])

    cx += (np.sum(np.sum(pc1 * sensors_area_effect13, axis=1), axis=1) / square13)
    cx -= (np.sum(np.sum(pc3 * sensors_area_effect13, axis=1), axis=1) / square13)

    cy += (np.sum(np.sum(pc2 * sensors_area_effect24, axis=1), axis=1) / square24)
    cy -= (np.sum(np.sum(pc4 * sensors_area_effect24, axis=1), axis=1) / square24)

    return cx, cy


@jit
@cc.export('calculate_cmz_aot',
           numba.float64[:](
               numba.int64,  # count_sensors_on_model
               numba.int64,  # count_sensors_on_middle_row
               numba.int64,  # count_sensors_on_side_row
               numba.int64,  # angle
               numba.float64,  # breadth
               numba.float64,  # depth
               numba.float64[::1],  # _x
               numba.float64[::1],  # _y
               numba.float64,  # height
               numba.float64[:, ::1]  # _pressure_coefficients
           ))
def calculate_cmz_aot(
        count_sensors_on_model,
        count_sensors_on_middle_row,
        count_sensors_on_side_row,
        angle,
        breadth,
        depth,
        _x,
        _y,
        height,
        pressure_coefficients
):
    count_row = count_sensors_on_model // (2 * (count_sensors_on_middle_row + count_sensors_on_side_row))
    count_sensors_on_row = 2 * (count_sensors_on_middle_row + count_sensors_on_side_row)

    x = _add_borders_to_coordinates_x(
        _x,
        count_sensors_on_row,
        count_row,
        breadth,
        depth
    )

    y = _add_borders_to_coordinates_y(
        _y,
        count_sensors_on_row,
        height
    )

    sensors_area_effect = _calculate_sensors_area_effect(
        x,
        y,
        count_row,
        count_sensors_on_row,
        count_sensors_on_middle_row,
        count_sensors_on_side_row
    )

    pc = _split_pressure_coefficients(
        count_sensors_on_model,
        count_sensors_on_middle_row,
        count_sensors_on_side_row,
        pressure_coefficients
    )

    mxs = _calculate_mxs(
        count_sensors_on_model,
        count_sensors_on_middle_row,
        count_sensors_on_side_row,
        _x,
        breadth,
        depth,
    )

    projection_on_the_axis = _calculate_projection_on_the_axis(
        breadth,
        depth,
        angle
    )

    square13, square24 = _calculate_square_of_face(
        breadth,
        depth,
        height
    )

    return _calculate_cmz(
        projection_on_the_axis,
        square13,
        square24,
        *sensors_area_effect,
        *mxs,
        *pc
    )


@jit
@cc.export('calculate_cx_cy_aot',
           (
                   numba.int64,  # count_sensors_on_model
                   numba.int64,  # count_sensors_on_middle_row
                   numba.int64,  # count_sensors_on_side_row
                   numba.float64,  # breadth
                   numba.float64,  # depth
                   numba.float64[::1],  # _x
                   numba.float64[::1],  # _y
                   numba.float64,  # height
                   numba.float64[:, ::1]  # _pressure_coefficients
           ))
def calculate_cx_cy_aot(
        count_sensors_on_model,
        count_sensors_on_middle_row,
        count_sensors_on_side_row,
        breadth,
        depth,
        _x,
        _y,
        height,
        pressure_coefficients
):
    count_row = count_sensors_on_model // (2 * (count_sensors_on_middle_row + count_sensors_on_side_row))
    count_sensors_on_row = 2 * (count_sensors_on_middle_row + count_sensors_on_side_row)

    x = _add_borders_to_coordinates_x(
        _x,
        count_sensors_on_row,
        count_row,
        breadth,
        depth
    )

    y = _add_borders_to_coordinates_y(
        _y,
        count_sensors_on_row,
        height
    )

    square13, square24 = _calculate_square_of_face(
        breadth,
        depth,
        height
    )

    sensors_area_effect = _calculate_sensors_area_effect(
        x,
        y,
        count_row,
        count_sensors_on_row,
        count_sensors_on_middle_row,
        count_sensors_on_side_row
    )

    pc = _split_pressure_coefficients(
        count_sensors_on_model,
        count_sensors_on_middle_row,
        count_sensors_on_side_row,
        pressure_coefficients
    )

    return _calculate_cx_cy(
        square13,
        square24,
        *sensors_area_effect,
        *pc
    )


@jit
@cc.export('_calculate_cmz_floor_integration',
           numba.float64[:](
               numba.float64,  # count_row
               numba.float64[:, :],  # sensors_area_effect13
               numba.float64[:, :],  # sensors_area_effect24
               numba.float64[:, :],  # mx13
               numba.float64[:, :],  # mx24
               numba.float64,  # square13
               numba.float64,  # square24
               numba.float64,  # projection_on_the_axis
               numba.float64[:, ::1],  # pc1
               numba.float64[:, ::1],  # pc2
               numba.float64[:, ::1],  # pc3
               numba.float64[:, ::1],  # pc4
           ))
def _calculate_cmz_floor_integration(
        count_row,
        sensors_area_effect13,
        sensors_area_effect24,
        mx13,
        mx24,
        square13,
        square24,
        projection_on_the_axis,
        pc1,
        pc2,
        pc3,
        pc4,
):
    div13 = (square13 / count_row) * projection_on_the_axis
    div24 = (square24 / count_row) * projection_on_the_axis

    cmz1 = np.sum(mx13 * pc1 * sensors_area_effect13, axis=2)
    cmz3 = np.sum(mx13 * pc3 * sensors_area_effect13, axis=2)

    cmz2 = np.sum(mx24 * pc2 * sensors_area_effect24, axis=2)
    cmz4 = np.sum(mx24 * pc4 * sensors_area_effect24, axis=2)

    cmz13 = (cmz1 + cmz3) / div13
    cmz24 = (cmz2 + cmz4) / div24

    cmz = (cmz13 + cmz24).T / count_row

    return np.round(np.flipud(cmz), decimals=5)


@jit
@cc.export('_calculate_cx_cy_floor_integration',
           (
                   numba.float64[:, :],  # sensors_area_effect13
                   numba.float64[:, :],  # sensors_area_effect24
                   numba.float64,  # square13
                   numba.float64,  # square24
                   numba.float64[:, ::1],  # pc1
                   numba.float64[:, ::1],  # pc2
                   numba.float64[:, ::1],  # pc3
                   numba.float64[:, ::1],  # pc4
           ))
def _calculate_cx_cy_floor_integration(
        sensors_area_effect13,
        sensors_area_effect24,
        square13,
        square24,
        pc1,
        pc2,
        pc3,
        pc4
):
    count_row = pc1.shape[1]

    divisor13 = square13 / count_row
    divisor24 = square24 / count_row

    cx1 = np.sum(pc1 * sensors_area_effect13, axis=2)
    cx2 = np.sum(pc3 * sensors_area_effect13, axis=2)
    cx = np.flipud(((cx1 - cx2) / divisor13).T / count_row)

    cy1 = np.sum(pc2 * sensors_area_effect24, axis=2)
    cy2 = np.sum(pc4 * sensors_area_effect24, axis=2)
    cy = np.flipud(((cy1 - cy2) / divisor24).T / count_row)

    return np.round(cx, decimals=5), np.round(cy, decimals=5)


@jit
@cc.export('aot_height_integration_cx_cy_cmz_floors_to_txt',
           (
                   numba.int64,  # count_sensors_on_model
                   numba.int64,  # count_sensors_on_middle_row
                   numba.int64,  # count_sensors_on_side_row
                   numba.int64,  # angle
                   numba.float64,  # breadth
                   numba.float64,  # depth
                   numba.float64[::1],  # _x
                   numba.float64[::1],  # _y
                   numba.float64,  # height
                   numba.float64[:, ::1]  # pressure_coefficients
           ))
def aot_height_integration_cx_cy_cmz_floors_to_txt(
        count_sensors_on_model,
        count_sensors_on_middle_row,
        count_sensors_on_side_row,
        angle,
        breadth,
        depth,
        _x,
        _y,
        height,
        pressure_coefficients
):
    count_row = count_sensors_on_model // (2 * (count_sensors_on_middle_row + count_sensors_on_side_row))
    count_sensors_on_row = 2 * (count_sensors_on_middle_row + count_sensors_on_side_row)

    projection_on_the_axis = _calculate_projection_on_the_axis(
        breadth,
        depth,
        angle
    )

    mx = _calculate_mxs(
        count_sensors_on_model,
        count_sensors_on_middle_row,
        count_sensors_on_side_row,
        _x,
        breadth,
        depth,
    )

    square = _calculate_square_of_face(
        breadth,
        depth,
        height
    )

    x = _add_borders_to_coordinates_x(
        _x,
        count_sensors_on_row,
        count_row,
        breadth,
        depth
    )

    y = _add_borders_to_coordinates_y(
        _y,
        count_sensors_on_row,
        height
    )

    sensors_area_effect = _calculate_sensors_area_effect(
        x,
        y,
        count_row,
        count_sensors_on_row,
        count_sensors_on_middle_row,
        count_sensors_on_side_row
    )

    pc = _split_pressure_coefficients(
        count_sensors_on_model,
        count_sensors_on_middle_row,
        count_sensors_on_side_row,
        pressure_coefficients
    )

    cx, cy = _calculate_cx_cy_floor_integration(
        *sensors_area_effect,
        *square,
        *pc
    )

    cmz = _calculate_cmz_floor_integration(
        count_row,
        *sensors_area_effect,
        *mx,
        *square,
        projection_on_the_axis,
        *pc
    )

    return cx, cy, cmz


if __name__ == "__main__":
    cc.compile()
