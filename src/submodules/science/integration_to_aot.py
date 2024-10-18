import os

import numba
import numpy as np
from numba.pycc import CC

cc = CC('aot_integration')
cc.output_dir = os.path.dirname(os.path.realpath(__file__))
cc.output_file = 'aot_integration.pyd'


@cc.export('add_borders_to_coordinates_x',
           numba.float64[:, ::1](
               numba.float64[::1],  # coordinates
               numba.int64,  # count_sensors_on_row
               numba.int64,  # count_row
               numba.float64,  # breadth
               numba.float64  # depth
           ))
def _aot_add_borders_to_coordinates_x(
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


@cc.export('add_borders_to_coordinates_y',
           numba.float64[:, ::1](
               numba.float64[::1],
               numba.int64,
               numba.float64
           ))
def _aot_add_borders_to_coordinates_y(
        coordinates,
        count_sensors_on_row,
        height
):
    y = np.reshape(coordinates, (-1, count_sensors_on_row))
    # добавление _height в начало каждого вектора и добавление 0 в конец каждого вектора
    y = np.vstack((np.full((1, count_sensors_on_row), height), y, np.zeros((1, count_sensors_on_row))))

    return y


@cc.export('aot_calculate_cmz',
           numba.float64[:](
               numba.int64,  # count_sensors_on_model
               numba.int64,  # count_sensors_on_middle_row
               numba.int64,  # count_sensors_on_side_row
               numba.int64,  # _angle
               numba.float64,  # breadth
               numba.float64,  # depth
               numba.float64[::1],  # _coordinate_x_for_mxs
               numba.float64[:, ::1],  # x
               numba.float64[:, ::1],  # y
               numba.float64,  # _height
               numba.float64[:, ::1]  # _pressure_coefficients
           ))
def _aot_calculate_cmz(
        count_sensors_on_model,
        count_sensors_on_middle_row,
        count_sensors_on_side_row,
        angle,
        breadth,
        depth,
        coordinate_x_for_mxs,
        x,
        y,
        height,
        pressure_coefficients
):
    count_row = count_sensors_on_model // (2 * (count_sensors_on_middle_row + count_sensors_on_side_row))

    count_sensors_on_row = 2 * (count_sensors_on_middle_row + count_sensors_on_side_row)

    x_for_mxs = np.reshape(coordinate_x_for_mxs, (count_row, -1))
    x_for_mxs = np.split(x_for_mxs, [count_sensors_on_middle_row,
                                     count_sensors_on_middle_row + count_sensors_on_side_row,
                                     2 * count_sensors_on_middle_row + count_sensors_on_side_row,
                                     2 * (count_sensors_on_middle_row + count_sensors_on_side_row)
                                     ], axis=1)

    del x_for_mxs[4]

    # центры граней
    mid13_x = breadth / 2
    mid24_x = depth / 2

    # плечи
    # разделение плечей на 4 массива, тк возможны варианты когда векторы массива разной длины
    # а в np.array все векторы должны быть одинаковой длины
    mxs1 = x_for_mxs[0] - mid13_x
    mxs2 = x_for_mxs[1] - breadth - mid24_x
    mxs3 = x_for_mxs[2] - (breadth + depth) - mid13_x
    mxs4 = x_for_mxs[3] - (2 * breadth + depth) - mid24_x

    # Площадь
    s13 = breadth * height
    s24 = depth * height

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

    pc_copy = np.reshape(pressure_coefficients, (pressure_coefficients.shape[0], count_row, -1))
    pc_copy = np.split(pc_copy, [count_sensors_on_middle_row,
                                 count_sensors_on_middle_row + count_sensors_on_side_row,
                                 2 * count_sensors_on_middle_row + count_sensors_on_side_row,
                                 2 * (count_sensors_on_middle_row + count_sensors_on_side_row)
                                 ], axis=2)

    projection_on_the_axis = np.cos(np.deg2rad(angle)) * breadth + np.sin(np.deg2rad(angle)) * depth

    divisors = np.array(
        [
            s13,
            s24,
            s13,
            s24
        ]
    ) * projection_on_the_axis

    result = np.zeros(pressure_coefficients.shape[0])

    for i, mxs in enumerate((mxs1, mxs2, mxs3, mxs4)):
        result += (np.sum(np.sum(pc_copy[i] * squares_faces[i] * mxs, axis=1), axis=1) / divisors[i])

    return result


@cc.export('aot_calculate_cx_cy',
           (
                   numba.int64,  # count_sensors_on_model
                   numba.int64,  # count_sensors_on_middle_row
                   numba.int64,  # count_sensors_on_side_row
                   numba.float64,  # breadth
                   numba.float64,  # depth
                   numba.float64[:, ::1],  # x
                   numba.float64[:, ::1],  # y
                   numba.float64,  # _height
                   numba.float64[:, ::1]  # _pressure_coefficients
           ))
def _aot_calculate_cx_cy(
        count_sensors_on_model,
        count_sensors_on_middle_row,
        count_sensors_on_side_row,
        breadth,
        depth,
        x,
        y,
        height,
        _pressure_coefficients
):
    count_row = count_sensors_on_model // (2 * (count_sensors_on_middle_row + count_sensors_on_side_row))
    count_sensors_on_row = 2 * (count_sensors_on_middle_row + count_sensors_on_side_row)

    # Площадь
    s13 = breadth * height
    s24 = depth * height

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

    cx = np.zeros(_pressure_coefficients.shape[0])
    cy = np.zeros(_pressure_coefficients.shape[0])

    pc_copy = np.reshape(_pressure_coefficients, (_pressure_coefficients.shape[0], count_row, -1))
    pc_copy = np.split(pc_copy, [count_sensors_on_middle_row,
                                 count_sensors_on_middle_row + count_sensors_on_side_row,
                                 2 * count_sensors_on_middle_row + count_sensors_on_side_row,
                                 2 * (count_sensors_on_middle_row + count_sensors_on_side_row)
                                 ], axis=2)

    cx += (np.sum(np.sum(pc_copy[0] * squares_faces[0], axis=1), axis=1) / s13)
    cx -= (np.sum(np.sum(pc_copy[2] * squares_faces[2], axis=1), axis=1) / s13)

    cy += (np.sum(np.sum(pc_copy[1] * squares_faces[1], axis=1), axis=1) / s24)
    cy -= (np.sum(np.sum(pc_copy[3] * squares_faces[3], axis=1), axis=1) / s24)

    return cx, cy


if __name__ == "__main__":
    cc.compile()
