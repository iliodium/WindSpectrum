import os

import numba
from numba import jit
import numpy as np
from numba.pycc import CC

cc = CC('aot_integration')
cc.output_dir = os.path.dirname(os.path.realpath(__file__))
cc.output_file = 'aot_integration.pyd'

@jit
@cc.export('add_borders_to_coordinates_x',
           numba.float64[:, ::1](
               numba.float64[::1],  # coordinates
               numba.int64,  # count_sensors_on_row
               numba.int64,  # count_row
               numba.float64,  # breadth
               numba.float64  # depth
           ))
def add_borders_to_coordinates_x(
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
@cc.export('add_borders_to_coordinates_y',
           numba.float64[:, ::1](
               numba.float64[::1],
               numba.int64,
               numba.float64
           ))
def add_borders_to_coordinates_y(
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
@cc.export('calculate_sensors_area_effect',
           (
                   numba.float64[:, ::1],  # x
                   numba.float64[:, ::1],  # y
                   numba.int64,  # count_row
                   numba.int64,  # count_sensors_on_row
                   numba.int64,  # count_sensors_on_middle_row
                   numba.int64  # count_sensors_on_side_row

           ))
def calculate_sensors_area_effect(
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
    # squares_faces13 = np.stack(
    #     (
    #         squares_faces[0],
    #         squares_faces[2],
    #     )
    # )
    # squares_faces24 = np.stack(
    #     (
    #         squares_faces[1],
    #         squares_faces[3],
    #     )
    # )
    #
    # return squares_faces13, squares_faces24
    return squares_faces[0], squares_faces[1], squares_faces[2], squares_faces[3]

@jit
@cc.export('split_pressure_coefficients',
           (
                   numba.int64,  # count_sensors_on_model
                   numba.int64,  # count_sensors_on_middle_row
                   numba.int64,  # count_sensors_on_side_row
                   numba.float64[:, ::1]  # _pressure_coefficients
           ))
def split_pressure_coefficients(
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

    # pc13 = np.stack(
    #     (
    #         pc_copy[0],
    #         pc_copy[2],
    #     )
    # )
    # pc24 = np.stack(
    #     (
    #         pc_copy[1],
    #         pc_copy[3],
    #     )
    # )

    # return pc13,pc24
    return pc_copy[0], pc_copy[1], pc_copy[2], pc_copy[3]

@jit
@cc.export('calculate_projection_on_the_axis',
           numba.float64(
               numba.float64,  # breadth
               numba.float64,  # depth
               numba.int64  # angle
           ))
def calculate_projection_on_the_axis(
        breadth,
        depth,
        angle
):
    projection_on_the_axis = np.cos(np.deg2rad(angle)) * breadth + np.sin(np.deg2rad(angle)) * depth
    return projection_on_the_axis

@jit
@cc.export('calculate_mxs',
           (
                   numba.int64,  # count_sensors_on_model
                   numba.int64,  # count_sensors_on_middle_row
                   numba.int64,  # count_sensors_on_side_row
                   numba.float64[::1],  # _coordinate_x_for_mxs
                   numba.float64,  # breadth
                   numba.float64  # depth
           ))
def calculate_mxs(
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
    mid13_x = breadth / 2
    mid24_x = depth / 2

    # плечи
    # разделение плечей на 4 массива, тк возможны варианты когда векторы массива разной длины
    # а в np.array все векторы должны быть одинаковой длины
    mxs1 = x_for_mxs[0] - mid13_x
    mxs2 = x_for_mxs[1] - breadth - mid24_x
    mxs3 = x_for_mxs[2] - (breadth + depth) - mid13_x
    mxs4 = x_for_mxs[3] - (2 * breadth + depth) - mid24_x

    # mxs13 = np.stack(
    #     (
    #         mxs1,
    #         mxs3,
    #     )
    # )
    # mxs24 = np.stack(
    #     (
    #         mxs2,
    #         mxs4,
    #     )
    # )
    #
    # return mxs13, mxs24
    return mxs1, mxs2, mxs3, mxs4


# Площадь поверхности грани(лицевой и боковой)
@jit
@cc.export('calculate_square_of_face',
           (
                   numba.float64,  # breadth
                   numba.float64,  # depth
                   numba.float64  # height
           ))
def calculate_square_of_face(
        breadth,
        depth,
        height
):
    s13 = breadth * height
    s24 = depth * height
    return s13, s24


# @cc.export('aot_calculate_cmz',
#            numba.float64[:](
#                numba.float64,  # projection_on_the_axis
#
#                numba.float64,  # square13
#                numba.float64[:, :, :],  # sensors_area_effect13
#                numba.float64[:, :, :],  # mxs13
#                numba.float64[:, :, :, :],  # pc13
#
#                numba.float64,  # square13
#                numba.float64[:, :, :],  # sensors_area_effect13
#                numba.float64[:, :, :],  # mxs13
#                numba.float64[:, :, :, :],  # pc13
#
#            ))
# def aot_calculate_cmz(
#         projection_on_the_axis,
#         square13,
#         sensors_area_effect13,
#         mxs13,
#         pc13,
#         square24,
#         sensors_area_effect24,
#         mxs24,
#         pc24
# ):
@jit
@cc.export('aot_calculate_cmz',
           numba.float64[:](
               numba.float64,  # projection_on_the_axis

               numba.float64,  # square13
               numba.float64,  # square24

               numba.float64[:, :],  # sensors_area_effect13
               numba.float64[:, :],  # sensors_area_effect13
               numba.float64[:, :],  # sensors_area_effect13
               numba.float64[:, :],  # sensors_area_effect13

               numba.float64[:, :],  # mxs13
               numba.float64[:, :],  # mxs13
               numba.float64[:, :],  # mxs13
               numba.float64[:, :],  # mxs13

               numba.float64[:, :, :],  # pc13
               numba.float64[:, :, :],  # pc13
               numba.float64[:, :, :],  # pc13
               numba.float64[:, :, :],  # pc13
           ))
def aot_calculate_cmz(
        projection_on_the_axis,
        square13,
        square24,
        sensors_area_effect1,
        sensors_area_effect2,
        sensors_area_effect3,
        sensors_area_effect4,
        mxs1,
        mxs2,
        mxs3,
        mxs4,
        pc1,
        pc2,
        pc3,
        pc4,
):
    # cmz = np.zeros(pc13.shape[1])
    #
    # for square, sensors, mxs, pc in zip(
    #         (square13, square24),
    #         (sensors_area_effect13, sensors_area_effect24),
    #         (mxs13, mxs24),
    #         (pc13, pc24),
    # ):
    #     for i in range(2):
    #         cmz += (np.sum(np.sum(pc[i] * sensors[i] * mxs[i], axis=1), axis=1) /
    #                 (square * projection_on_the_axis))
    #
    # return cmz
    cmz = np.zeros(pc1.shape[0])

    # for square, sensors, mxs, pc in zip(
    #         (square13, square24),
    #         (sensors_area_effect13, sensors_area_effect24),
    #         (mxs13, mxs24),
    #         (pc13, pc24),
    # ):
    #     for i in range(2):
    #         cmz += (np.sum(np.sum(pc[i] * sensors[i] * mxs[i], axis=1), axis=1) /
    #                 (square * projection_on_the_axis))
    cmz += (np.sum(np.sum(pc1 * sensors_area_effect1 * mxs1, axis=1), axis=1) /
            (square13 * projection_on_the_axis))

    cmz += (np.sum(np.sum(pc2 * sensors_area_effect2 * mxs2, axis=1), axis=1) /
            (square24 * projection_on_the_axis))

    cmz += (np.sum(np.sum(pc3 * sensors_area_effect3 * mxs3, axis=1), axis=1) /
            (square13 * projection_on_the_axis))

    cmz += (np.sum(np.sum(pc4 * sensors_area_effect4 * mxs4, axis=1), axis=1) /
            (square24 * projection_on_the_axis))
    #
    # cmz += (np.sum(pc1 * sensors_area_effect1 * mxs1, axis=(1, 2)) /
    #         (square13 * projection_on_the_axis))
    #
    # cmz += (np.sum(pc2 * sensors_area_effect2 * mxs2, axis=(1, 2)) /
    #         (square24 * projection_on_the_axis))
    #
    # cmz += (np.sum(pc3 * sensors_area_effect3 * mxs3, axis=(1, 2)) /
    #         (square13 * projection_on_the_axis))
    #
    # cmz += (np.sum(pc4 * sensors_area_effect4 * mxs4, axis=(1, 2)) /
    #         (square24 * projection_on_the_axis))
    return cmz

    # result = np.zeros(_pressure_coefficients.shape[0])
    #
    #     for i in range(4):
    #         result += (np.sum(pc_copy[i] * squares_faces[i] * mxs[i], axis=(1, 2)) / divisors[i])

    # return result


@cc.export('aot_calculate_cmz_old',
           numba.float64[:](
               numba.int64,  # count_sensors_on_model
               numba.int64,  # count_sensors_on_middle_row
               numba.int64,  # count_sensors_on_side_row
               numba.int64,  # _angle
               numba.float64,  # breadth
               numba.float64,  # depth
               numba.float64[::1],  # _coordinate_x_for_mxs
               numba.float64[::1],  # x
               numba.float64[::1],  # y
               numba.float64,  # _height
               numba.float64[:, ::1]  # _pressure_coefficients
           ))
def aot_calculate_cmz_old(
        count_sensors_on_model,
        count_sensors_on_middle_row,
        count_sensors_on_side_row,
        angle,
        breadth,
        depth,
        coordinate_x_for_mxs,
        _x,
        _y,
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
    y = np.reshape(_y, (-1, count_sensors_on_row))
    # добавление _height в начало каждого вектора и добавление 0 в конец каждого вектора
    y = np.vstack((np.full((1, count_sensors_on_row), height), y, np.zeros((1, count_sensors_on_row))))

    x = np.reshape(_x, (-1, count_sensors_on_row))
    # добавление 0 в начало каждой строки и добавление 2 * (breadth + depth) в конец каждой строки
    x = np.column_stack((np.zeros(count_row), x, np.full(count_row, 2 * (breadth + depth))))
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


@cc.export('calculate_cmz_artem',
           (
                   numba.int64,  # count_sensors_on_model
                   numba.int64,  # count_sensors_on_middle_row
                   numba.int64,  # count_sensors_on_side_row
                   numba.int64,  # _angle
                   numba.float64,  # breadth
                   numba.float64,  # depth
                   numba.float64[::1],  # x
                   numba.float64[::1],  # y
                   numba.float64,  # _height
                   numba.float64[:, ::1]  # _pressure_coefficients
           ))
def calculate_cmz_artem(
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

    x = add_borders_to_coordinates_x(
        _x,
        count_sensors_on_row,
        count_row,
        breadth,
        depth
    )

    y = add_borders_to_coordinates_y(
        _y,
        count_sensors_on_row,
        height
    )

    sensors_area_effect = calculate_sensors_area_effect(
        # sensors_area_effect13, sensors_area_effect24 = calculate_sensors_area_effect(
        x,
        y,
        count_row,
        count_sensors_on_row,
        count_sensors_on_middle_row,
        count_sensors_on_side_row
    )

    pc = split_pressure_coefficients(
        # pc13, pc24 = split_pressure_coefficients(
        count_sensors_on_model,
        count_sensors_on_middle_row,
        count_sensors_on_side_row,
        pressure_coefficients
    )

    mxs = calculate_mxs(
        # mxs13, mxs24 = calculate_mxs(
        count_sensors_on_model,
        count_sensors_on_middle_row,
        count_sensors_on_side_row,
        _x,
        breadth,
        depth,
    )

    projection_on_the_axis = calculate_projection_on_the_axis(
        breadth,
        depth,
        angle
    )

    square13, square24 = calculate_square_of_face(
        breadth,
        depth,
        height
    )

    result = aot_calculate_cmz(
        projection_on_the_axis,
        square13,
        square24,
        *sensors_area_effect,
        *mxs,
        *pc
    )

    return result

# @cc.export('aot_calculate_cx_cy',
#            (
#                    numba.float64,  # s13
#                    numba.float64,  # s24
#                    numba.float64[:, ::1],  # squares_faces
#                    numba.float64[:, ::1]  # _pressure_coefficients
#            ))
# def aot_calculate_cx_cy(
#         s13,
#         s24,
#         squares_faces,
#         _pressure_coefficients
# ):
#     cx = np.zeros(_pressure_coefficients.shape[0])
#     cy = np.zeros(_pressure_coefficients.shape[0])
#
#     cx += (np.sum(np.sum(pc_copy[0] * squares_faces[0], axis=1), axis=1) / s13)
#     cx -= (np.sum(np.sum(pc_copy[2] * squares_faces[2], axis=1), axis=1) / s13)
#
#     cy += (np.sum(np.sum(pc_copy[1] * squares_faces[1], axis=1), axis=1) / s24)
#     cy -= (np.sum(np.sum(pc_copy[3] * squares_faces[3], axis=1), axis=1) / s24)
#
#     return cx, cy


# @cc.export('aot_calculate_cmz',
#            numba.float64[:](
#                numba.int64,  # count_sensors_on_model
#                numba.int64,  # count_sensors_on_middle_row
#                numba.int64,  # count_sensors_on_side_row
#                numba.int64,  # _angle
#                numba.float64,  # breadth
#                numba.float64,  # depth
#                numba.float64[::1],  # _coordinate_x_for_mxs
#                numba.float64[:, ::1],  # x
#                numba.float64[:, ::1],  # y
#                numba.float64,  # _height
#                numba.float64[:, ::1]  # _pressure_coefficients
#            ))
# def aot_height_integration_cx_cy_cmz_floors_to_txt(
#         count_sensors_on_model,
#         count_sensors_on_middle_row,
#         count_sensors_on_side_row,
#         angle,
#         breadth,
#         depth,
#         coordinate_x_for_mxs,
#         x,
#         y,
#         height,
#         pressure_coefficients
# ):
#     shirina = np.cos(np.deg2rad(_angle)) * breadth + np.sin(np.deg2rad(_angle)) * depth
#
#     # центры граней
#     mid13_x = breadth / 2
#     mid24_x = depth / 2
#
#     x1 = coordinates[0]
#     x1 = np.reshape(x1, (count_row, -1))
#     x1 = np.split(x1, [count_sensors_on_middle_row,
#                        count_sensors_on_middle_row + count_sensors_on_side_row,
#                        2 * count_sensors_on_middle_row + count_sensors_on_side_row,
#                        2 * (count_sensors_on_middle_row + count_sensors_on_side_row)
#                        ], axis=1)
#
#     v2 = breadth
#     v3 = breadth + depth
#     v4 = 2 * breadth + depth
#     x1[1] -= v2
#     x1[2] -= v3
#     x1[3] -= v4
#
#     # mx плечи для каждого сенсора
#     mx13 = np.array([
#         x1[0] - mid13_x,
#         x1[2] - mid13_x,
#     ])
#
#     mx24 = np.array([
#         x1[1] - mid24_x,
#         x1[3] - mid24_x,
#     ])
#
#     # Площадь
#     s13 = breadth * height
#     s24 = depth * height
#     z_levels = sorted(set(y), reverse=True)
#
#     x = coordinates[0]
#     x = np.reshape(x, (-1, count_sensors_on_row))
#     x = np.append(x, np.full((len(x), 1), 2 * (breadth + depth)), axis=1)
#     x = np.insert(x, 0, 0, axis=1)
#
#     y = coordinates[1]
#     y = np.append(np.full(count_sensors_on_row, height), y)
#     y = np.reshape(y, (-1, count_sensors_on_row))
#     y = np.append(y, np.zeros((count_sensors_on_row, 1)))
#     y = np.reshape(y, (-1, count_sensors_on_row))
#
#     squares = []
#     for y_i in range(count_row):
#         for x_i in range(count_sensors_on_row):
#             y_t = y[y_i][x_i]
#             y_m = y[y_i + 1][x_i]
#             y_b = y[y_i + 2][x_i]
#             if y_i == 0:
#                 dy = y_t - y_m + (y_m - y_b) / 2
#             elif y_i == count_row - 1:
#                 dy = (y_t - y_m) / 2 + y_m - y_b
#             else:
#                 dy = (y_t - y_m) / 2 + (y_m - y_b) / 2
#
#             x_l = x[y_i][x_i]
#             x_m = x[y_i][x_i + 1]
#             x_r = x[y_i][x_i + 2]
#
#             if x_i == 0:
#                 dx = x_m - x_l + (x_r - x_m) / 2
#             elif x_i == count_sensors_on_row - 1:
#                 dx = (x_m - x_l) / 2 + x_r - x_m
#             else:
#                 dx = (x_m - x_l) / 2 + (x_r - x_m) / 2
#
#             squares.append(dy * dx)
#
#     squares_faces = np.reshape(squares, (count_row, -1))
#     squares_faces = np.split(squares_faces, [count_sensors_on_middle_row,
#                                              count_sensors_on_middle_row + count_sensors_on_side_row,
#                                              2 * count_sensors_on_middle_row + count_sensors_on_side_row,
#                                              2 * (count_sensors_on_middle_row + count_sensors_on_side_row)
#                                              ], axis=1)
#
#     cx = [
#         [] for _ in range(count_row)
#     ]
#     cy = [
#         [] for _ in range(count_row)
#     ]
#     cmz = [
#         [] for _ in range(count_row)
#     ]
#     for pr in pressure_coefficients:
#         pr = np.reshape(pr, (count_row, -1))
#         pr = np.split(pr, [count_sensors_on_middle_row,
#                            count_sensors_on_middle_row + count_sensors_on_side_row,
#                            2 * count_sensors_on_middle_row + count_sensors_on_side_row,
#                            2 * (count_sensors_on_middle_row + count_sensors_on_side_row)
#                            ], axis=1)
#
#         for row_i in range(count_row):
#             faces_x = []
#             faces_y = []
#
#             for face in range(4):
#                 if face in [0, 2]:
#                     faces_x.append(np.sum(pr[face][row_i] * squares_faces[face][row_i]) / (s13 / count_row))
#                 else:
#                     faces_y.append(np.sum(pr[face][row_i] * squares_faces[face][row_i]) / (s24 / count_row))
#
#             cx[row_i].append(faces_x[0] - faces_x[1])
#             cy[row_i].append(faces_y[0] - faces_y[1])
#
#             t1 = np.sum(mx13[0][row_i] * pr[0][row_i] * squares_faces[0][row_i]) / ((s13 / count_row) * shirina)
#             t3 = np.sum(mx13[1][row_i] * pr[2][row_i] * squares_faces[2][row_i]) / ((s13 / count_row) * shirina)
#
#             t2 = np.sum(mx24[0][row_i] * pr[1][row_i] * squares_faces[1][row_i]) / ((s24 / count_row) * shirina)
#             t4 = np.sum(mx24[1][row_i] * pr[3][row_i] * squares_faces[3][row_i]) / ((s24 / count_row) * shirina)
#
#             cmz[row_i] = np.append(cmz[row_i], sum([t1, t2, t3, t4]))
#
#     cx = (np.array(list(reversed(cx))) / count_row).round(5)
#     cy = (np.array(list(reversed(cy))) / count_row).round(5)
#     cmz = (np.array(list(reversed(cmz))) / count_row).round(5)
#
#     # fig, ax = plt.subplots(dpi=Plot.dpi, num='snfgkjdsnfkjsdnf', clear=True)
#     # ax.plot(list(range(32768)), np.sum(cx, axis=0))
#
#     angle = _angle
#
#     time = np.linspace(0, 32.768, 32768).round(5)
#
#     if _alpha == 4:
#         speed_2_3 = np.round(interp_025_tpu([height])[0], 3)
#     elif _alpha == 6:
#         speed_2_3 = np.round(interp_016_tpu([height])[0], 3)
#
#     f = io.StringIO()
#     alpha_temp = '0.25' if _alpha == 4 else '0.16'
#     f.write(
#         f'{model_name} Вариант модели\n{breadth}, {depth}, {height} м размеры модели b d h\n'
#         f'{alpha_temp} альфа\n{angle} угол\n{uh_speed} Uh скорость на высоте H\n'
#         f'{speed_2_3} скорость на высоте 2/3 H\n'
#         f'{count_row} количество этажей\n'
#     )
#
#     f.write('time, ')
#
#     # for ind in range(1, count_row + 1):
#     #     temp_name_str += f'cx{count_row - ind}, cy{count_row - ind}, cmz{count_row - ind}, '
#
#     for ind in range(1, count_row + 1):
#         f.write(f'cx{ind}, cy{ind}, cmz{ind}, ')
#
#     f.write('cxsum, cysum, cmzsum\n')
#
#     # enumerate_str = ', '.join(map(str, reversed(range(count_row * 3 + 1 + 3)))) + '\n'
#     f.write(', '.join(map(str, range(count_row * 3 + 1 + 3))) + '\n')
#
#     f.write('-1, ')
#     for z in reversed(z_levels):
#         z /= height
#         f.write(f'{z}, {z}, {z}, ')
#     f.write('1, 1, 1\n')
#
#     data_to_txt = np.array([time])
#
#     for ind in range(count_row):
#         data_to_txt = np.append(data_to_txt, [cx[ind]], axis=0)
#         data_to_txt = np.append(data_to_txt, [cy[ind]], axis=0)
#         data_to_txt = np.append(data_to_txt, [cmz[ind]], axis=0)
#
#     data_to_txt = np.append(data_to_txt, [np.sum(cx, axis=0).round(5)], axis=0)
#     data_to_txt = np.append(data_to_txt, [np.sum(cy, axis=0).round(5)], axis=0)
#     data_to_txt = np.append(data_to_txt, [np.sum(cmz, axis=0).round(5)], axis=0)
#
#     np.savetxt(f, data_to_txt.T, newline="\n", delimiter=',', fmt='%.5f')
#
#     res = str(f.getvalue())
#
#     f.close()
#
#     return res


if __name__ == "__main__":
    cc.compile()
