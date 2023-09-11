import time
from typing import Tuple

from utils.utils import interpolator as intp
from utils.utils import ks10, alpha_standards, wind_regions

import toml
import numpy as np
from scipy.signal import welch
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.tri as mtri
from matplotlib.axis import rcParams
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.colors import Normalize
from matplotlib.ticker import MultipleLocator, ScalarFormatter

plt.switch_backend('Agg')


class Plot:
    """
    Класс отвечает за отрисовку графиков
       Виды графиков:
           Изополя:
               -Максимальные значения
               -Средние значения
               -Минимальные значения
               -Среднее квадратичное отклонение
           Огибающие:
               -Максимальные значения
               -Минимальные значения
           Коэффициенты обеспеченности:
               -Максимальные значения
               -Минимальные значения
           Спектры
           Нестационарные сигналы
       Нумерация:
       Нумерация графиков нужная для очищения памяти
       num = alpha_model_name_angle_mode
       """

    config = toml.load('config.toml')

    dpi = config['plots']['dpi']  # качество графиков

    def isofields_coefficients_flat_roof(self, **kwargs):
        db = kwargs['db']

        breadth = kwargs['breadth']
        depth = kwargs['depth']
        height = kwargs['height']

        x_coordinates = kwargs['x_coordinates']
        y_coordinates = kwargs['y_coordinates']
        surface = kwargs['surface']
        pressure_coefficients = kwargs['pressure_coefficients']
        pressure_coefficients = np.mean(pressure_coefficients, axis=0)

        count_faces = len(set(surface))
        count_levels = 11

        data = dict()
        for face in set(surface):
            data[face] = {
                "x_coordinates": list(),
                "y_coordinates": list(),
                "pressure_coefficients": list()}

        for ind, face in enumerate(surface):
            data[face]["x_coordinates"].append(x_coordinates[ind])
            data[face]["y_coordinates"].append(y_coordinates[ind])
            data[face]["pressure_coefficients"].append(pressure_coefficients[ind])

        width_plot = 2 * height + depth

        range_height = [i for i in range(height + 1)]
        range_breadth = [i for i in range(breadth + 1)]
        range_depth = [i for i in range(depth + 1)]

        len_range_breadth = len(range_breadth)
        len_range_depth = len(range_depth)
        len_range_height = len(range_height)

        if db == 'without_eaves':
            position_plot_on_grid = {
                1: [1, 0],
                2: [2, 1],
                3: [1, 2],
                4: [0, 1],
                5: [1, 1],
            }
            x_labels = {
                1: range_height,
                2: range_depth,
                3: range_height,
                4: range_depth,
                5: range_depth,
            }
            y_labels = {
                1: range_breadth,
                2: range_height,
                3: range_breadth,
                4: range_height,
                5: range_breadth,
            }
            left_right_face = (1, 3)
            len_x_labels = {
                1: len_range_height,
                2: len_range_depth,
                3: len_range_height,
                4: len_range_depth,
                5: len_range_depth,
            }
            len_y_labels = {
                1: len_range_breadth,
                2: len_range_height,
                3: len_range_breadth,
                4: len_range_height,
                5: len_range_breadth,
            }
        elif db == 'non_isolated':
            position_plot_on_grid = {
                1: [1, 1],
                2: [1, 0],
                3: [2, 1],
                4: [1, 2],
                5: [0, 1],
            }
            x_labels = {
                1: range_depth,
                2: range_height,
                3: range_depth,
                4: range_height,
                5: range_depth,
            }
            y_labels = {
                1: range_breadth,
                2: range_breadth,
                3: range_height,
                4: range_breadth,
                5: range_height,
            }
            left_right_face = (2, 4)
            len_x_labels = {
                1: len_range_depth,
                2: len_range_height,
                3: len_range_depth,
                4: len_range_height,
                5: len_range_depth,
            }
            len_y_labels = {
                1: len_range_breadth,
                2: len_range_breadth,
                3: len_range_height,
                4: len_range_breadth,
                5: len_range_height,
            }

        depth_ratios = depth / width_plot
        height_ratios = height / width_plot

        fig = plt.figure(figsize=(12, 12), dpi=80)
        grid = plt.GridSpec(3, 3, hspace=0.2, wspace=0.2,
                            width_ratios=[height_ratios, depth_ratios, height_ratios],
                            height_ratios=[height_ratios, height_ratios, height_ratios])

        cmap = cm.get_cmap(name="jet")

        levels_for_data = np.linspace(
            np.min([np.min(data[i]['pressure_coefficients']) for i in range(1, count_faces + 1)]),
            np.max([np.max(data[i]['pressure_coefficients']) for i in range(1, count_faces + 1)]),
            count_levels)

        for face in range(1, count_faces + 1):
            x_coordinates = data[face]['x_coordinates']
            y_coordinates = data[face]['y_coordinates']
            pressure_coefficients = data[face]['pressure_coefficients']

            set_x_coordinates = sorted(list(set(x_coordinates)))
            set_y_coordinates = sorted(list(set(y_coordinates)))

            min_x_coordinate = min(set_x_coordinates)
            max_x_coordinate = max(set_x_coordinates)

            new_min_x_coordinate = min_x_coordinate - 1
            new_max_x_coordinate = max_x_coordinate + 1

            min_y_coordinate = min(set_y_coordinates)
            max_y_coordinate = max(set_y_coordinates)

            new_min_y_coordinate = min_y_coordinate - 1
            new_max_y_coordinate = max_y_coordinate + 1

            boundary_coordinates_x = []
            boundary_coordinates_y = []
            if face in left_right_face:
                for x in set_x_coordinates:
                    local_max_y = float('-inf')
                    local_min_y = float('+inf')

                    for x_coord, y_coord in zip(x_coordinates, y_coordinates):
                        if x == x_coord:
                            local_max_y = y_coord if y_coord > local_max_y else local_max_y
                            local_min_y = y_coord if y_coord < local_min_y else local_min_y

                    boundary_coordinates_x.extend([x, x])
                    boundary_coordinates_y.extend([local_min_y - 1, local_max_y + 1])

                for y in set_y_coordinates:
                    local_max_x = float('-inf')
                    local_min_x = float('+inf')

                    for x_coord, y_coord in zip(x_coordinates, y_coordinates):
                        if y == y_coord:
                            local_max_x = x_coord if x_coord > local_max_x else local_max_x
                            local_min_x = x_coord if x_coord < local_min_x else local_min_x

                    boundary_coordinates_x.extend([local_min_x - 1, local_max_x + 1])
                    boundary_coordinates_y.extend([y, y])

                for x, y in zip(x_coordinates, y_coordinates):
                    if x == min_x_coordinate and y == min_y_coordinate:
                        boundary_coordinates_x.append(new_min_x_coordinate)
                        boundary_coordinates_y.append(new_min_y_coordinate)

                    elif x == min_x_coordinate and y == max_y_coordinate:
                        boundary_coordinates_x.append(new_min_x_coordinate)
                        boundary_coordinates_y.append(new_max_y_coordinate)

                    elif x == max_x_coordinate and y == min_y_coordinate:
                        boundary_coordinates_x.append(new_max_x_coordinate)
                        boundary_coordinates_y.append(new_min_y_coordinate)

                    elif x == max_x_coordinate and y == max_y_coordinate:
                        boundary_coordinates_x.append(new_max_x_coordinate)
                        boundary_coordinates_y.append(new_max_y_coordinate)

            else:
                x_coordinates_left_border = [new_min_x_coordinate for _ in range(len(set_y_coordinates) + 2)]
                x_coordinates_right_border = [new_max_x_coordinate for _ in range(len(set_y_coordinates) + 2)]

                boundary_coordinates_x = x_coordinates_left_border + set_x_coordinates + \
                                         x_coordinates_right_border + set_x_coordinates

                y_coordinates_top_border = [new_max_y_coordinate for _ in range(len(set_x_coordinates) + 2)]
                y_coordinates_bottom_border = [new_min_y_coordinate for _ in range(len(set_x_coordinates) + 1)]

                boundary_coordinates_y = [new_min_y_coordinate] + set_y_coordinates + y_coordinates_top_border + \
                                         set_y_coordinates + y_coordinates_bottom_border

            coords = [[i1, j1] for i1, j1 in zip(x_coordinates, y_coordinates)]  # Старые координаты
            # Интерполятор полученный на основе имеющихся данных
            interpolator = intp(coords, pressure_coefficients)

            data_from_interpolator = [float(interpolator([[X, Y]])) for X, Y in
                                      zip(boundary_coordinates_x, boundary_coordinates_y)]

            full_x_coordinates = x_coordinates + boundary_coordinates_x
            full_y_coordinates = y_coordinates + boundary_coordinates_y
            full_data = pressure_coefficients + data_from_interpolator

            temp_triang = mtri.Triangulation(full_x_coordinates, full_y_coordinates)

            temp_refiner = mtri.UniformTriRefiner(temp_triang)

            temp_grid, temp_value = temp_refiner.refine_field(full_data, subdiv=4)

            temp_xticks = np.linspace(new_min_x_coordinate, new_max_x_coordinate, len_x_labels[face])
            temp_yticks = np.linspace(new_min_y_coordinate, new_max_y_coordinate, len_y_labels[face])

            temp_plot = fig.add_subplot(grid[position_plot_on_grid[face][0], position_plot_on_grid[face][1]],
                                        xticklabels=x_labels[face], xticks=temp_xticks,
                                        yticklabels=y_labels[face], yticks=temp_yticks)

            data_colorbar = temp_plot.tricontourf(temp_grid, temp_value, cmap=cmap, extend='both',
                                                  levels=levels_for_data)
            temp_contour = temp_plot.tricontour(temp_grid, temp_value, linewidths=1, linestyles='solid', colors='black',
                                                levels=levels_for_data)

            temp_plot.clabel(temp_contour, fontsize=10)
            temp_plot.plot(x_coordinates, y_coordinates, '.k', **dict(markersize=3))

        fig.colorbar(data_colorbar, ax=fig.axes[:], cmap=cmap, ticks=levels_for_data, shrink=1).ax.tick_params(
            labelsize=4)

        return fig

    def isofields_coefficients_gable_roof(self, **kwargs):
        db = kwargs['db']

        breadth = kwargs['breadth']
        depth = kwargs['depth']
        height = kwargs['height']
        pitch = kwargs['pitch']

        x_coordinates = kwargs['x_coordinates']
        y_coordinates = kwargs['y_coordinates']
        surface = kwargs['surface']
        pressure_coefficients = kwargs['pressure_coefficients']
        pressure_coefficients = np.mean(pressure_coefficients, axis=0)

        count_faces = len(set(surface))
        count_levels = 11

        data = dict()
        for face in set(surface):
            data[face] = {
                "x_coordinates": list(),
                "y_coordinates": list(),
                "pressure_coefficients": list()}

        for ind, face in enumerate(surface):
            data[face]["x_coordinates"].append(x_coordinates[ind])
            data[face]["y_coordinates"].append(y_coordinates[ind])
            data[face]["pressure_coefficients"].append(pressure_coefficients[ind])

        width_plot = 2 * height + depth
        height_plot = 2 * height + breadth

        range_height = [i for i in range(height + 1)]
        range_breadth = [i for i in range(breadth + 1)]
        range_depth = [i for i in range(depth + 1)]

        len_range_breadth = len(range_breadth)
        len_range_depth = len(range_depth)
        len_range_height = len(range_height)

        roof_length = breadth / (2 * np.cos(np.deg2rad(pitch)))
        height_in_triangle = np.sin(np.deg2rad(pitch)) * roof_length

        full_height_side_faces = (height_in_triangle + height).round(2)
        range_full_height_side_faces = [i for i in range(int(full_height_side_faces) + 1)]
        if not float(range_full_height_side_faces[-1]) == full_height_side_faces:
            range_full_height_side_faces.append(full_height_side_faces)

        if db == 'without_eaves':
            left_right_face = (1, 3)
            border_face = (1, 2, 3, 4)
            main_face = (5,)

            position_plot_on_grid = {
                1: [1, 0],
                2: [2, 1],
                3: [1, 2],
                4: [0, 1],
                5: [1, 1],
            }
            x_labels = {
                1: range_full_height_side_faces,
                2: range_depth,
                3: range_full_height_side_faces[::-1],
                4: range_depth,
                5: range_depth,
            }

            y_labels = {
                1: range_breadth,
                2: range_height,
                3: range_breadth,
                4: range_height,
                5: range_breadth,
            }
            len_x_labels = {
                1: len_range_height,
                2: len_range_depth,
                3: len_range_height,
                4: len_range_depth,
                5: len_range_depth,
            }
            len_y_labels = {
                1: len_range_breadth,
                2: len_range_height,
                3: len_range_breadth,
                4: len_range_height,
                5: len_range_breadth,
            }
        elif db in ('non_isolated', 'with_eaves'):
            left_right_face = (2, 4)
            border_face = (2, 3, 4, 5)
            # main_face = (1,)

            position_plot_on_grid = {
                1: [1, 1],
                2: [1, 0],
                3: [2, 1],
                4: [1, 2],
                5: [0, 1],
            }
            x_labels = {
                1: range_depth,
                2: range_full_height_side_faces,
                3: range_depth,
                4: range_full_height_side_faces[::-1],
                5: range_depth,
            }

            y_labels = {
                1: range_breadth,
                2: range_breadth,
                3: range_height,
                4: range_breadth,
                5: range_height,
            }
            left_right_face = (2, 4)
            len_x_labels = {
                1: len_range_depth,
                2: len_range_height,
                3: len_range_depth,
                4: len_range_height,
                5: len_range_depth,
            }
            len_y_labels = {
                1: len_range_breadth,
                2: len_range_breadth,
                3: len_range_height,
                4: len_range_breadth,
                5: len_range_height,
            }
        breadth_ratios = breadth / width_plot
        depth_ratios = depth / width_plot
        height_ratios = height / width_plot

        fig = plt.figure(figsize=(12, 12), dpi=80)
        grid = plt.GridSpec(3, 3, hspace=0.2, wspace=0.2,
                            width_ratios=[height_ratios, depth_ratios, height_ratios],
                            height_ratios=[height_ratios, height_ratios, height_ratios])

        cmap = cm.get_cmap(name="jet")

        levels_for_data = np.linspace(
            np.min([np.min(data[i]['pressure_coefficients']) for i in range(1, count_faces + 1)]),
            np.max([np.max(data[i]['pressure_coefficients']) for i in range(1, count_faces + 1)]),
            count_levels)

        for face in border_face:
            x_coordinates = data[face]['x_coordinates']
            y_coordinates = data[face]['y_coordinates']
            pressure_coefficients = data[face]['pressure_coefficients']

            set_x_coordinates = sorted(list(set(x_coordinates)))
            set_y_coordinates = sorted(list(set(y_coordinates)))

            min_x_coordinate = min(set_x_coordinates)
            max_x_coordinate = max(set_x_coordinates)

            new_min_x_coordinate = min_x_coordinate - 1
            new_max_x_coordinate = max_x_coordinate + 1

            min_y_coordinate = min(set_y_coordinates)
            max_y_coordinate = max(set_y_coordinates)

            new_min_y_coordinate = min_y_coordinate - 1
            new_max_y_coordinate = max_y_coordinate + 1

            boundary_coordinates_x = []
            boundary_coordinates_y = []
            if face in left_right_face:
                if face == 2:
                    boundary_coordinates_x.append(max(set_x_coordinates) + 2)
                    boundary_coordinates_y.append(sum(set_y_coordinates) / len(set_y_coordinates))
                    new_max_x_coordinate += 1
                else:
                    boundary_coordinates_x.append(min_x_coordinate - 2)
                    boundary_coordinates_y.append(sum(set_y_coordinates) / len(set_y_coordinates))
                    new_min_x_coordinate -= 1

                for x in set_x_coordinates:
                    local_max_y = float('-inf')
                    local_min_y = float('+inf')

                    for x_coord, y_coord in zip(x_coordinates, y_coordinates):
                        if x == x_coord:
                            local_max_y = y_coord if y_coord > local_max_y else local_max_y
                            local_min_y = y_coord if y_coord < local_min_y else local_min_y

                    boundary_coordinates_x.extend([x, x])
                    boundary_coordinates_y.extend([local_min_y - 1, local_max_y + 1])

                for y in set_y_coordinates:
                    local_max_x = float('-inf')
                    local_min_x = float('+inf')

                    for x_coord, y_coord in zip(x_coordinates, y_coordinates):
                        if y == y_coord:
                            local_max_x = x_coord if x_coord > local_max_x else local_max_x
                            local_min_x = x_coord if x_coord < local_min_x else local_min_x

                    boundary_coordinates_x.extend([local_min_x - 1, local_max_x + 1])
                    boundary_coordinates_y.extend([y, y])

                for x, y in zip(x_coordinates, y_coordinates):
                    if x == min_x_coordinate and y == min_y_coordinate:
                        boundary_coordinates_x.append(new_min_x_coordinate)
                        boundary_coordinates_y.append(new_min_y_coordinate)

                    elif x == min_x_coordinate and y == max_y_coordinate:
                        boundary_coordinates_x.append(new_min_x_coordinate)
                        boundary_coordinates_y.append(new_max_y_coordinate)

                    elif x == max_x_coordinate and y == min_y_coordinate:
                        boundary_coordinates_x.append(new_max_x_coordinate)
                        boundary_coordinates_y.append(new_min_y_coordinate)

                    elif x == max_x_coordinate and y == max_y_coordinate:
                        boundary_coordinates_x.append(new_max_x_coordinate)
                        boundary_coordinates_y.append(new_max_y_coordinate)

            else:
                x_coordinates_left_border = [new_min_x_coordinate for _ in range(len(set_y_coordinates) + 2)]
                x_coordinates_right_border = [new_max_x_coordinate for _ in range(len(set_y_coordinates) + 2)]

                boundary_coordinates_x = x_coordinates_left_border + set_x_coordinates + \
                                         x_coordinates_right_border + set_x_coordinates

                y_coordinates_top_border = [new_max_y_coordinate for _ in range(len(set_x_coordinates) + 2)]
                y_coordinates_bottom_border = [new_min_y_coordinate for _ in range(len(set_x_coordinates) + 1)]

                boundary_coordinates_y = [new_min_y_coordinate] + set_y_coordinates + y_coordinates_top_border + \
                                         set_y_coordinates + y_coordinates_bottom_border

            coords = [[i1, j1] for i1, j1 in zip(x_coordinates, y_coordinates)]  # Старые координаты
            # Интерполятор полученный на основе имеющихся данных
            interpolator = intp(coords, pressure_coefficients)

            data_from_interpolator = [float(interpolator([[X, Y]])) for X, Y in
                                      zip(boundary_coordinates_x, boundary_coordinates_y)]

            full_x_coordinates = x_coordinates + boundary_coordinates_x
            full_y_coordinates = y_coordinates + boundary_coordinates_y
            full_data = pressure_coefficients + data_from_interpolator

            temp_triang = mtri.Triangulation(full_x_coordinates, full_y_coordinates)

            temp_refiner = mtri.UniformTriRefiner(temp_triang)

            temp_grid, temp_value = temp_refiner.refine_field(full_data, subdiv=4)

            temp_yticks = np.linspace(new_min_y_coordinate, new_max_y_coordinate, len(y_labels[face]))
            temp_xticks = np.linspace(new_min_x_coordinate, new_max_x_coordinate, len(x_labels[face]))

            temp_plot = fig.add_subplot(grid[position_plot_on_grid[face][0], position_plot_on_grid[face][1]],
                                        xticklabels=x_labels[face], xticks=temp_xticks,
                                        yticklabels=y_labels[face], yticks=temp_yticks)

            temp_plot.tricontourf(temp_grid, temp_value, cmap=cmap, extend='both', levels=levels_for_data)
            temp_contour = temp_plot.tricontour(temp_grid, temp_value, linewidths=1, linestyles='solid', colors='black',
                                                levels=levels_for_data)

            temp_plot.clabel(temp_contour, fontsize=10)

            temp_plot.plot(x_coordinates, y_coordinates, '.k', **dict(markersize=3))

        if db == 'without_eaves':
            set_x_coordinates = sorted(list(set(data[5]['x_coordinates'])))
            set_y_coordinates_5 = sorted(list(set(data[5]['y_coordinates'])))
            set_y_coordinates_6 = sorted(list(set(data[6]['y_coordinates'])))

            union_plot = fig.add_subplot(grid[position_plot_on_grid[5][0], position_plot_on_grid[5][1]],
                                         xticklabels=x_labels[5], xticks=[i for i in range(min(set_x_coordinates) - 1,
                                                                                           max(set_x_coordinates) + 2)],
                                         yticklabels=y_labels[5], yticks=[i for i in range(min(set_y_coordinates_5) - 1,
                                                                                           max(set_y_coordinates_6) + 2)])

            for face in (5, 6):
                x_coordinates = data[face]['x_coordinates']
                y_coordinates = data[face]['y_coordinates']
                pressure_coefficients = data[face]['pressure_coefficients']

                set_x_coordinates = sorted(list(set(x_coordinates)))
                set_y_coordinates = sorted(list(set(y_coordinates)))

                min_x_coordinate = min(set_x_coordinates)
                max_x_coordinate = max(set_x_coordinates)

                new_min_x_coordinate = min_x_coordinate - 1
                new_max_x_coordinate = max_x_coordinate + 1

                min_y_coordinate = min(set_y_coordinates)
                max_y_coordinate = max(set_y_coordinates)

                new_min_y_coordinate = min_y_coordinate - 1
                new_max_y_coordinate = max_y_coordinate + 1

                x_coordinates_left_border = [new_min_x_coordinate for _ in range(len(set_y_coordinates) + 2)]
                x_coordinates_right_border = [new_max_x_coordinate for _ in range(len(set_y_coordinates) + 2)]

                boundary_coordinates_x = x_coordinates_left_border + set_x_coordinates + \
                                         x_coordinates_right_border + set_x_coordinates

                y_coordinates_top_border = [new_max_y_coordinate for _ in range(len(set_x_coordinates) + 2)]
                y_coordinates_bottom_border = [new_min_y_coordinate for _ in range(len(set_x_coordinates) + 1)]

                boundary_coordinates_y = [new_min_y_coordinate] + set_y_coordinates + y_coordinates_top_border + \
                                         set_y_coordinates + y_coordinates_bottom_border

                coords = [[i1, j1] for i1, j1 in zip(x_coordinates, y_coordinates)]  # Старые координаты
                # Интерполятор полученный на основе имеющихся данных
                extrapolator = intp(coords, pressure_coefficients)

                data_from_interpolator = [float(extrapolator([[X, Y]])) for X, Y in
                                          zip(boundary_coordinates_x, boundary_coordinates_y)]

                full_x_coordinates = x_coordinates + boundary_coordinates_x
                full_y_coordinates = y_coordinates + boundary_coordinates_y
                full_data = pressure_coefficients + data_from_interpolator

                temp_triang = mtri.Triangulation(full_x_coordinates, full_y_coordinates)

                temp_refiner = mtri.UniformTriRefiner(temp_triang)

                temp_grid, temp_value = temp_refiner.refine_field(full_data, subdiv=4)

                union_plot.tricontourf(temp_grid, temp_value, cmap=cmap, extend='both', levels=levels_for_data)
                temp_contour = union_plot.tricontour(temp_grid, temp_value, linewidths=1, linestyles='solid',
                                                     colors='black',
                                                     levels=levels_for_data)

                union_plot.clabel(temp_contour, fontsize=10)

                union_plot.plot(x_coordinates, y_coordinates, '.k', **dict(markersize=3))

        elif db in ('non_isolated', 'with_eaves'):
            union_face = 1
            set_x_coordinates = sorted(list(set(data[union_face]['x_coordinates'])))
            set_y_coordinates = sorted(list(set(data[union_face]['y_coordinates'])))

            union_plot = fig.add_subplot(
                grid[position_plot_on_grid[union_face][0], position_plot_on_grid[union_face][1]],
                xticklabels=x_labels[union_face],
                xticks=[i for i in range(min(set_x_coordinates) - 1, max(set_x_coordinates) + 2)],
                yticklabels=y_labels[union_face],
                yticks=[i for i in range(min(set_y_coordinates) - 1, max(set_y_coordinates) + 2)])

            for face in (1,):
                x_coordinates = data[face]['x_coordinates']
                y_coordinates = data[face]['y_coordinates']
                pressure_coefficients = data[face]['pressure_coefficients']

                set_x_coordinates = sorted(list(set(x_coordinates)))
                set_y_coordinates = sorted(list(set(y_coordinates)))

                min_x_coordinate = min(set_x_coordinates)
                max_x_coordinate = max(set_x_coordinates)

                new_min_x_coordinate = min_x_coordinate - 1
                new_max_x_coordinate = max_x_coordinate + 1

                min_y_coordinate = min(set_y_coordinates)
                max_y_coordinate = max(set_y_coordinates)

                new_min_y_coordinate = min_y_coordinate - 1
                new_max_y_coordinate = max_y_coordinate + 1

                x_coordinates_left_border = [new_min_x_coordinate for _ in range(len(set_y_coordinates) + 2)]
                x_coordinates_right_border = [new_max_x_coordinate for _ in range(len(set_y_coordinates) + 2)]

                boundary_coordinates_x = x_coordinates_left_border + set_x_coordinates + \
                                         x_coordinates_right_border + set_x_coordinates

                y_coordinates_top_border = [new_max_y_coordinate for _ in range(len(set_x_coordinates) + 2)]
                y_coordinates_bottom_border = [new_min_y_coordinate for _ in range(len(set_x_coordinates) + 1)]

                boundary_coordinates_y = [new_min_y_coordinate] + set_y_coordinates + y_coordinates_top_border + \
                                         set_y_coordinates + y_coordinates_bottom_border

                coords = [[i1, j1] for i1, j1 in zip(x_coordinates, y_coordinates)]  # Старые координаты
                # Интерполятор полученный на основе имеющихся данных
                extrapolator = intp(coords, pressure_coefficients)

                data_from_interpolator = [float(extrapolator([[X, Y]])) for X, Y in
                                          zip(boundary_coordinates_x, boundary_coordinates_y)]

                full_x_coordinates = x_coordinates + boundary_coordinates_x
                full_y_coordinates = y_coordinates + boundary_coordinates_y
                full_data = pressure_coefficients + data_from_interpolator

                temp_triang = mtri.Triangulation(full_x_coordinates, full_y_coordinates)

                temp_refiner = mtri.UniformTriRefiner(temp_triang)

                temp_grid, temp_value = temp_refiner.refine_field(full_data, subdiv=4)

                union_plot.tricontourf(temp_grid, temp_value, cmap=cmap, extend='both', levels=levels_for_data)
                temp_contour = union_plot.tricontour(temp_grid, temp_value, linewidths=1, linestyles='solid',
                                                     colors='black',
                                                     levels=levels_for_data)

                union_plot.clabel(temp_contour, fontsize=10)

                union_plot.plot(x_coordinates, y_coordinates, '.k', **dict(markersize=3))

        return fig

    @staticmethod
    def pseudocolor_coefficients(model_name: str, mode: str, angle: str, alpha: str, pressure_coefficients,
                                 coordinates):
        """Отрисовка дискретных изополей"""
        # Виды изополей
        mods = {
            'max': np.max(pressure_coefficients, axis=0),
            'mean': np.mean(pressure_coefficients, axis=0),
            'min': np.min(pressure_coefficients, axis=0),
            'std': np.std(pressure_coefficients, axis=0),
        }
        pressure_coefficients = mods[mode]
        breadth, depth, height = int(model_name[0]) / 10, int(model_name[1]) / 10, int(model_name[2]) / 10
        count_sensors_on_model = len(pressure_coefficients)
        count_sensors_on_middle = int(model_name[0]) * 5
        count_sensors_on_side = int(model_name[1]) * 5
        count_row = count_sensors_on_model // (2 * (count_sensors_on_middle + count_sensors_on_side))
        pressure_coefficients = np.reshape(pressure_coefficients, (count_row, -1))
        pressure_coefficients = np.split(pressure_coefficients,
                                         [count_sensors_on_middle,
                                          count_sensors_on_middle + count_sensors_on_side,
                                          2 * count_sensors_on_middle + count_sensors_on_side,
                                          2 * (count_sensors_on_middle + count_sensors_on_side)
                                          ], axis=1)
        del pressure_coefficients[4]

        x, z = np.array(coordinates)
        z = np.array(z[::2 * (count_sensors_on_middle + count_sensors_on_side)])[::-1]

        num_fig = f'Дискретные изополя {model_name} {mode} {alpha} {angle}'
        fig, ax = plt.subplots(1, 4, num=num_fig, dpi=Plot.dpi, clear=True)
        cmap = cm.get_cmap(name="jet")

        min_v = np.min([np.min(pressure_coefficients[i]) for i in range(4)])
        max_v = np.max([np.max(pressure_coefficients[i]) for i in range(4)])

        normalizer = Normalize(min_v, max_v)
        im = cm.ScalarMappable(norm=normalizer, cmap=cmap)
        for i in range(4):
            data_old = pressure_coefficients[i]
            ax[i].pcolormesh(np.flip(data_old, axis=0), cmap=cmap, norm=normalizer)
            labels = np.arange(0, height + 0.01, 0.05).round(2)
            ax[i].set_yticks(np.linspace(0, count_row, len(labels)))
            ax[i].set_yticklabels(labels=labels, fontsize=5)
            if i in [0, 2]:
                ax[i].set_xticks([i for i in range(0, count_sensors_on_middle + 1, 5)])
                ax[i].set_xticklabels(labels=list(map(str, np.arange(0, breadth + 0.1, 0.1).round(2))), fontsize=5)
            else:
                ax[i].set_xticks([i for i in range(0, count_sensors_on_side + 1, 5)])
                ax[i].set_xticklabels(labels=list(map(str, np.arange(0, depth + 0.1, 0.1).round(2))), fontsize=5)
            # x, y = np.meshgrid(np.arange(0.5, count_sensors_on_middle + 0.5, 1), z * count_row / height)
            # ax[i].plot(x, y, '.k')

        if breadth == depth == height:
            [ax[i].set_aspect('equal') for i in range(4)]
        ticks = np.arange(np.round(min_v, 1), np.round(max_v, 1) + 0.1, 0.1)
        fig.colorbar(im, ax=ax, location='bottom', cmap=cmap, ticks=ticks).ax.tick_params(labelsize=4)

        return fig

    @staticmethod
    def isofields_coefficients(db='isolated', **kwargs):
        """Отрисовка интегральных изополей"""

        model_name = kwargs['model_scale']
        model_size = kwargs['model_size']
        scale_factors = kwargs['scale_factors']
        mode = kwargs['mode']
        angle = kwargs['angle']
        pressure_coefficients = kwargs['pressure_coefficients']
        coordinates = kwargs['coordinates']
        pressure_plot_parameters = kwargs['pressure_plot_parameters']
        coefficient_for_pressure = None

        # Виды изополей
        mods = {
            'mean': lambda coefficients: np.mean(coefficients, axis=0).round(4),
            'rms': lambda coefficients: np.array([np.sqrt(i.dot(i) / i.size) for i in coefficients.T]).round(4),
            'std': lambda coefficients: np.std(coefficients, axis=0).round(4),
            'max': lambda coefficients: np.max(coefficients, axis=0).round(4),
            'min': lambda coefficients: np.min(coefficients, axis=0).round(4),

        }

        size_x, size_y, size_z = map(float, model_size)
        x_scale_factor, y_scale_factor, z_scale_factor = scale_factors

        pressure_coefficients = mods[mode](pressure_coefficients)
        if db == 'isolated':
            alpha = kwargs['alpha']
            breadth, depth, height = int(model_name[0]) / 10, int(model_name[1]) / 10, int(model_name[2]) / 10
            count_sensors_on_middle = int(model_name[0]) * 5
            count_sensors_on_side = int(model_name[1]) * 5
            # Шаги для изополей коэффициентов
            steps = {
                'max': 0.2,
                'mean': 0.2 if alpha == '6' else 0.1,
                'min': 0.2,
                'std': 0.05,
            }
            num_fig = f'Коэффициенты изополя {model_name} {model_size} {mode} {alpha} {angle}'

        elif db == 'interference':
            model_scale = kwargs['model_scale']
            case = kwargs['case']

            height = model_scale / 1000
            breadth, depth = 0.07, 0.07
            count_sensors_on_middle = 7
            count_sensors_on_side = 7
            # Шаги для изополей коэффициентов
            steps = {
                'max': 0.2,
                'mean': 0.2,
                'min': 0.2,
                'std': 0.05,
                'rms': 0.1,
            }
            num_fig = f'Коэффициенты изополя {model_name} {model_size} {mode} {case} {angle}'

        count_sensors_on_model = len(pressure_coefficients)
        count_row = count_sensors_on_model // (2 * (count_sensors_on_middle + count_sensors_on_side))

        x, z = np.array(coordinates)
        # Уровни для изополей коэффициентов
        if not pressure_plot_parameters:
            min_v = np.min(pressure_coefficients)
            max_v = np.max(pressure_coefficients)
            levels = np.arange(np.round(min_v, 1), np.round(max_v, 1) + 0.1, steps[mode]).round(2)

        pressure_coefficients = np.reshape(pressure_coefficients, (count_row, -1))
        pressure_coefficients = np.split(pressure_coefficients, [count_sensors_on_middle,
                                                                 count_sensors_on_middle + count_sensors_on_side,
                                                                 2 * count_sensors_on_middle + count_sensors_on_side,
                                                                 2 * (count_sensors_on_middle + count_sensors_on_side)
                                                                 ], axis=1)
        x = np.reshape(x, (count_row, -1))
        x = np.split(x, [count_sensors_on_middle,
                         count_sensors_on_middle + count_sensors_on_side,
                         2 * count_sensors_on_middle + count_sensors_on_side,
                         2 * (count_sensors_on_middle + count_sensors_on_side)
                         ], axis=1)

        z = np.reshape(z, (count_row, -1))
        z = np.split(z, [count_sensors_on_middle,
                         count_sensors_on_middle + count_sensors_on_side,
                         2 * count_sensors_on_middle + count_sensors_on_side,
                         2 * (count_sensors_on_middle + count_sensors_on_side)
                         ], axis=1)

        del pressure_coefficients[4]
        del x[4]
        del z[4]

        # x это тензор со всеми координатами граней по ширине,x1...x4 координаты отдельных граней
        x1, x2, x3, x4 = list(x[0]), list(x[1]), list(x[2]), list(x[3])
        # z это тензор со всеми координатами граней по высоте,z1...z4 координаты отдельных граней
        z1, z2, z3, z4 = list(z[0]), list(z[1]), list(z[2]), list(z[3])

        # Расширение матрицы координат по бокам
        for i in range(len(x1)):
            x1[i] = np.append(np.insert(x1[i], 0, 0), breadth)
            x2[i] = np.append(np.insert(x2[i], 0, breadth), breadth + depth)
            x3[i] = np.append(np.insert(x3[i], 0, breadth + depth), 2 * breadth + depth)
            x4[i] = np.append(np.insert(x4[i], 0, 2 * breadth + depth), 2 * (breadth + depth))

        x1.append(x1[0])
        x2.append(x2[0])
        x3.append(x3[0])
        x4.append(x4[0])

        x1.insert(0, x1[0])
        x2.insert(0, x2[0])
        x3.insert(0, x3[0])
        x4.insert(0, x4[0])

        # Расширение матрицы координат по бокам
        for i in range(len(z1)):
            z1[i] = np.append(np.insert(z1[i], 0, z1[i][0]), z1[i][0])
            z2[i] = np.append(np.insert(z2[i], 0, z2[i][0]), z2[i][0])
            z3[i] = np.append(np.insert(z3[i], 0, z3[i][0]), z3[i][0])
            z4[i] = np.append(np.insert(z4[i], 0, z4[i][0]), z4[i][0])

        z1.append(np.array([0 for _ in range(len(z1[0]))]))
        z2.append(np.array([0 for _ in range(len(z2[0]))]))
        z3.append(np.array([0 for _ in range(len(z3[0]))]))
        z4.append(np.array([0 for _ in range(len(z4[0]))]))

        z1.insert(0, np.array([height for _ in range(len(z1[0]))]))
        z2.insert(0, np.array([height for _ in range(len(z2[0]))]))
        z3.insert(0, np.array([height for _ in range(len(z3[0]))]))
        z4.insert(0, np.array([height for _ in range(len(z4[0]))]))

        # Расширенные координаты для изополей
        z_extended = [np.array(z1), np.array(z2), np.array(z3), np.array(z4)]
        x_extended = [np.array(x1), np.array(x2), np.array(x3), np.array(x4)]

        #fig, ax = plt.subplots(1, 4, num=num_fig, dpi=Plot.dpi, clear=True)
        sum_size = size_x+size_y+size_z

        breadth_ratios = size_x / sum_size
        depth_ratios = size_y / sum_size
        height_ratios = size_z / sum_size

        fig = plt.figure(num=num_fig, dpi=Plot.dpi, clear=True)
        grid = plt.GridSpec(1, 4,
                            width_ratios=[breadth_ratios, depth_ratios, breadth_ratios, depth_ratios],
                            height_ratios=[height_ratios])

        ax = [fig.add_subplot(grid[0, i]) for i in range(4)]

        cmap = cm.get_cmap(name="jet")
        data_colorbar = None

        h_scaled = height * z_scale_factor
        b_scaled = breadth * x_scale_factor
        d_scaled = depth * y_scale_factor

        x_new_list = []
        z_new_list = []

        x_old_list = []
        z_old_list = []

        data_new_list = []
        data_old_list = []

        if pressure_plot_parameters:
            type_area = pressure_plot_parameters['type_area']
            wind_region = pressure_plot_parameters['wind_region']
            alpha_standard = alpha_standards[type_area]
            k10 = ks10[type_area]
            wind_pressure = wind_regions[wind_region]

            PO = 1.225
            yf = 1.4
            ks = 1
            w0 = wind_pressure * 1000
            u10 = np.sqrt(2 * yf * k10 * w0 / PO)
            wind_profile = u10 * (ks * size_z / 10) ** alpha_standard
            coefficient_for_pressure = wind_profile ** 2 * PO / 2

        for i in range(4):
            x_new = x_extended[i].reshape(1, -1)[0]
            x_old = x[i].reshape(1, -1)[0]

            z_new = z_extended[i].reshape(1, -1)[0]
            z_old = z[i].reshape(1, -1)[0]
            # Вычитаем чтобы все координаты по x находились в интервале [0, 1]
            if i == 1:
                x_old -= breadth
                x_new -= breadth
            elif i == 2:
                x_old -= (breadth + depth)
                x_new -= (breadth + depth)
            elif i == 3:
                x_old -= (2 * breadth + depth)
                x_new -= (2 * breadth + depth)

            # Масштабирование координат
            z_new = z_new * z_scale_factor
            z_old = z_old * z_scale_factor

            if i in [0, 2]:
                x_new = x_new * x_scale_factor
                x_old = x_old * x_scale_factor

            else:
                x_new = x_new * y_scale_factor
                x_old = x_old * y_scale_factor

            data_old = pressure_coefficients[i].reshape(1, -1)[0]

            if pressure_plot_parameters:
                data_old = [coefficient_for_pressure * coefficient for coefficient in data_old]

            # data_old_integer.append(data_old)
            coords = [[i1, j1] for i1, j1 in zip(x_old, z_old)]  # Старые координаты
            # Интерполятор полученный на основе имеющихся данных
            interpolator = intp(coords, data_old)

            # Получаем данные для несуществующих датчиков
            data_new = [float(interpolator([[X, Y]])) for X, Y in zip(x_new, z_new)]

            x_new_list.append(x_new)
            z_new_list.append(z_new)

            x_old_list.append(x_old)
            z_old_list.append(z_old)

            data_new_list.append(data_new)
            # data_old_list.append(data_old)

        # Уровни для изополей давления
        if pressure_plot_parameters:
            min_0_2 = np.min(np.array([data_new_list[0], data_new_list[2]]))
            min_1_3 = np.min(np.array([data_new_list[1], data_new_list[3]]))
            min_v = np.min(np.array(min_0_2, min_1_3))

            max_0_2 = np.max(np.array([data_new_list[0], data_new_list[2]]))
            max_1_3 = np.max(np.array([data_new_list[1], data_new_list[3]]))
            max_v = np.max(np.array(max_0_2, max_1_3))

            all_levels = np.arange(min_v, max_v, (np.abs(min_v) + np.abs(max_v)) * 0.1).round(2)

        if db == 'isolated' and not pressure_plot_parameters:
            all_levels = np.linspace(np.min([np.min(data_new_list[i]) for i in range(4)]),
                                     np.max([np.max(data_new_list[i]) for i in range(4)]), 11)
        elif db == 'interference' and not pressure_plot_parameters:
            if mode in ('max', 'min'):
                all_levels = np.arange(-6, 6.2, .2)
            elif mode == 'mean':
                all_levels = np.arange(-2, 2.1, .1)
            elif mode in ('rms', 'std'):
                all_levels = np.arange(-1, 1.1, .1)

        for i in range(4):
            triang = mtri.Triangulation(x_new_list[i], z_new_list[i])
            refiner = mtri.UniformTriRefiner(triang)
            grid, value = refiner.refine_field(data_new_list[i], subdiv=4)

            # min_value = np.min(data_new_list[i])
            # max_value = np.max(data_new_list[i])
            # temp_levels = np.linspace(min_value, max_value, 11)
            # print(all_levels)
            # data_colorbar = ax[i].tricontourf(grid, value, cmap=cmap, extend='both', levels=11)
            # print(data_colorbar.levels)
            data_colorbar = ax[i].tricontourf(grid, value, cmap=cmap, extend='both', levels=all_levels)
            # data_colorbar = ax[i].tricontourf(grid, value, cmap=cmap, extend='both', levels=temp_levels)
            # data_colorbar = ax[i].tricontourf(grid, value, cmap=cmap, extend='both', levels=levels)

            # aq = ax[i].tricontour(grid, value, linewidths=1, linestyles='solid', colors='black', levels=11)
            # print(aq.levels)
            aq = ax[i].tricontour(grid, value, linewidths=1, linestyles='solid', colors='black', levels=all_levels)
            # aq = ax[i].tricontour(grid, value, linewidths=1, linestyles='solid', colors='black', levels=temp_levels)
            # aq = ax[i].tricontour(grid, value, linewidths=1, linestyles='solid', colors='black', levels=levels)

            ax[i].clabel(aq, fontsize=10, fmt='%.2f')

            ax[i].set_ylim([0, h_scaled])
            ax[i].set_yticks(np.arange(0, h_scaled + h_scaled * 0.01, h_scaled * 0.2))
            ax[i].set_yticklabels(labels=np.arange(0, size_z + size_z * 0.01, size_z * 0.2).round(2), fontsize=5)

            if i in [0, 2]:
                xlim = b_scaled
                label_b = size_x
            else:
                xlim = d_scaled
                label_b = size_y

            ax[i].set_xlim([0, xlim])
            ax[i].set_xticks(ticks=np.arange(0, xlim + xlim * 0.01, xlim * 0.2))
            ax[i].set_xticklabels(labels=np.arange(0, label_b + label_b * 0.01, label_b * 0.2).round(2), fontsize=5)

            if not pressure_plot_parameters:
                x_dots, y_dots = np.meshgrid(x_old_list[i], z_old_list[i])
                ax[i].plot(x_dots, y_dots, '.k', **dict(markersize=3))

        # print(all_levels)
        # fig.colorbar(data_colorbar, ax=ax, location='bottom', cmap=cmap, ticks=levels).ax.tick_params(labelsize=4)
        fig.colorbar(data_colorbar, ax=ax, location='bottom', cmap=cmap, ticks=all_levels).ax.tick_params(labelsize=4)

        return fig

    @staticmethod
    def summary_coefficients(db='isolated',
                             **kwargs):
        """Графики суммарных аэродинамических коэффициентов в декартовой системе координат
        data = {name:array,
                ...
                }
        """
        data = kwargs['data']
        model_scale = kwargs['model_scale']
        angle = kwargs['angle']
        if db == 'isolated':
            alpha = kwargs['alpha']
            num_fig = f'Суммарные коэффициенты декартовая система координат {model_scale} {alpha} {angle} isolated'
            fig, ax = plt.subplots(dpi=Plot.dpi, num=num_fig, clear=True)
            ax.set_xlim(0, 32.768)
            ox = np.linspace(0, 32.768, 32768)
        elif db == 'interference':
            case = kwargs['case']
            num_fig = f'Суммарные коэффициенты декартовая система координат {model_scale} {case} {angle} interference'
            fig, ax = plt.subplots(dpi=Plot.dpi, num=num_fig, clear=True)
            ax.set_xlim(0, 7.5)
            ox = np.linspace(0, 7.5, 5858)

        ax.grid()
        ax.set_ylabel('Суммарные аэродинамические коэффициенты')
        ax.set_xlabel('Время, с', labelpad=.3)
        for name in data.keys():
            if data[name] is not None:
                ax.plot(ox, data[name], label=name)
        ax.legend(loc='upper right', fontsize=9)

        return fig

    @staticmethod
    def scaling_data(x, y=None, angle_border=50):
        """Масштабирование данных до 360 градусов"""
        if angle_border == 50:
            if y is not None:
                a = np.array(y)
                b = np.append(a, np.flip(x)[1:])
                c = np.append(b, np.flip(b)[1:])
                x_scale = np.append(c, np.flip(c)[1:])

                a = np.array(x)
                b = np.append(a, np.flip(y)[1:])
                c = np.append(b, np.flip(b)[1:])
                y_scale = np.append(c, np.flip(c)[1:])

                return x_scale, y_scale

            else:
                a = np.array(x)
                b = np.append(a, np.flip(x)[1:])
                c = np.append(b, np.flip(b)[1:])
                x_scale = np.append(c, np.flip(c)[1:])
                return x_scale

        elif angle_border == 95:
            if y is not None:
                a = np.array(y)
                b = np.append(a, np.flip(x)[1:])
                x_scale = np.append(b, np.flip(b)[1:])

                a = np.array(x)
                b = np.append(a, np.flip(y)[1:])
                y_scale = np.append(b, np.flip(b)[1:])

                return x_scale, y_scale

            else:
                a = np.array(x)
                b = np.append(a, np.flip(x)[1:])
                x_scale = np.append(b, np.flip(b)[1:])
                return x_scale

    @staticmethod
    def polar_plot(db='isolated', **kwargs):
        """Графики суммарных аэродинамических коэффициентов в полярной системе координат.
        data = {name:array,
                ...
                }
        """
        data = kwargs['data']
        title = kwargs['title']
        model_size = kwargs['model_size']
        if db == 'isolated':
            alpha = kwargs['alpha']
            num_fig = f'Суммарные коэффициенты декартовая система координат {title} {" ".join(model_size)} {alpha}'
        elif db == 'interference':
            case = kwargs['case']
            num_fig = f'Суммарные коэффициенты декартовая система координат {title} {" ".join(model_size)} {case}'

        angles = np.array([angle for angle in range(0, 365, 5)]) * np.pi / 180.0

        fig, ax = plt.subplots(dpi=Plot.dpi, num=num_fig, clear=True, subplot_kw={'projection': 'polar'})

        for name in data.keys():
            ax.plot(angles, data[name], label=name)

        ax.set_theta_direction(-1)
        ax.set_theta_zero_location('N')
        ax.set_thetagrids([i for i in range(0, 360, 15)])
        ax.legend(loc='upper right', fontsize=9)
        ax.set_title(title)

        return fig

    @staticmethod
    def model_pic(model_size, model_scale, coordinates):
        """Отрисовка развертки модели"""
        breadth_real, depth_real, height_real = float(model_size[0]), float(model_size[1]), float(model_size[2])

        size_x = 2 * (breadth_real + depth_real)
        x, z = coordinates

        num_fig = f'Развертка модели {" ".join(model_size)}'
        fig, ax = plt.subplots(figsize=(16, 9), num=num_fig, dpi=Plot.dpi, clear=True)
        ax.set_title('Развертка датчиков по модели', fontweight='semibold', fontsize=8)
        ax.set_xlabel('Горизонтальная развертка /м', fontweight='semibold', fontsize=8)
        ax.set_ylabel('Высота модели /м', fontweight='semibold', fontsize=8)
        ax.set_ylim(0, height_real)
        ax.set_xlim(0, size_x)

        xticks = np.array([0, breadth_real, breadth_real + depth_real, 2 * breadth_real + depth_real, size_x])
        yticks = np.arange(0, height_real + height_real * 0.01, height_real * 0.2)

        ax.set_xticks(ticks=xticks)
        ax.set_yticks(ticks=yticks)

        ax.set_xticklabels(labels=(xticks / 10).round(3))
        ax.set_yticklabels(labels=(yticks / 10).round(3))

        ax.xaxis.set_minor_locator(MultipleLocator(size_x * 0.03125))
        ax.yaxis.set_minor_locator(MultipleLocator(height_real * 0.05))

        ax.xaxis.set_minor_formatter(ScalarFormatter())
        ax.yaxis.set_minor_formatter(ScalarFormatter())

        minor_xticks = (np.array(ax.get_xticks(minor=True)) / 10).round(3)
        minor_yticks = (np.array(ax.get_yticks(minor=True)) / 10).round(3)

        ax.set_xticklabels(labels=minor_xticks, minor=True)
        ax.set_yticklabels(labels=minor_yticks, minor=True)

        ax.tick_params(axis='x', which='minor', pad=5, labelsize=7)
        ax.tick_params(axis='x', which='major', pad=10, labelsize=10)

        ax.tick_params(axis='y', which='minor', pad=5, labelsize=7)
        ax.tick_params(axis='y', which='major', pad=10, labelsize=10)

        count_sensors_on_model = len(x)
        count_sensors_on_middle = int(model_scale[0]) * 5
        count_sensors_on_side = int(model_scale[1]) * 5
        count_row = count_sensors_on_model // (2 * (count_sensors_on_middle + count_sensors_on_side))

        x = np.reshape(x, (count_row, -1))
        x = np.split(x, [count_sensors_on_middle,
                         count_sensors_on_middle + count_sensors_on_side,
                         2 * count_sensors_on_middle + count_sensors_on_side,
                         2 * (count_sensors_on_middle + count_sensors_on_side)
                         ], axis=1)

        z = np.reshape(z, (count_row, -1))
        z = np.split(z, [count_sensors_on_middle,
                         count_sensors_on_middle + count_sensors_on_side,
                         2 * count_sensors_on_middle + count_sensors_on_side,
                         2 * (count_sensors_on_middle + count_sensors_on_side)
                         ], axis=1)
        del x[4]
        del z[4]

        for i in (breadth_real, breadth_real + depth_real, breadth_real + depth_real, 2 * breadth_real + depth_real):
            ax.plot([i, i], [0, height_real], linestyle='--', color='black')

        for i in range(4):
            ax.plot(x[i], z[i], 'b+')

        labels = [str(i) for i in range(1, count_sensors_on_model + 1)]
        labels = np.reshape(labels, (count_row, -1))
        labels = np.split(labels, [count_sensors_on_middle,
                                   count_sensors_on_middle + count_sensors_on_side,
                                   2 * count_sensors_on_middle + count_sensors_on_side,
                                   2 * (count_sensors_on_middle + count_sensors_on_side)
                                   ], axis=1)

        d = height_real * 0.02

        for f in range(4):
            for i in range(len(x[f])):
                for j in range(len(x[f][i])):
                    ax.text(x[f][i][j], z[f][i][j] - d, labels[f][i][j], fontsize=8)

        return fig

    @staticmethod
    def model_polar(model_size):
        """Отрисовка модели в полярной системе координат"""
        x, y, _ = float(model_size[0]), float(model_size[1]), float(model_size[2])
        min_size = min(x, y)
        b_scale, d_scale = x / min_size, y / min_size

        num_fig = f'Модель в полярной системе координат {" ".join(model_size)}'
        fig = plt.figure(num=num_fig, dpi=Plot.dpi, clear=True)
        pos = [0.1, 0.1, 0.8, 0.8]
        polar = fig.add_axes(pos, projection='polar')
        polar.patch.set_alpha(0)
        polar.set_theta_zero_location('N')
        polar.set_theta_direction(-1)
        polar.set_thetagrids([i for i in range(0, 360, 15)], fontsize=7)
        polar.set_yticks([0, 1, 2, 3])
        polar.set_yticklabels([0, 1, 2, 3], visible=False)
        polar.set_autoscale_on(False)
        angles = np.array([angle for angle in range(0, 365, 5)]) * np.pi / 180.0
        polar.annotate("", xy=(angles[0], 2), xytext=(0, 0),
                       arrowprops=dict(arrowstyle="->",
                                       linewidth=2.5))
        polar.annotate("", xy=(angles[18], 2), xytext=(0, 0),
                       arrowprops=dict(arrowstyle="->",
                                       linewidth=2.5))
        polar.annotate("y", xy=(angles[0], 2))
        polar.annotate("x", xy=(angles[18], 2))

        ax = fig.add_subplot(111, position=pos)
        ax.set_visible(True)
        ax.set_autoscale_on(False)
        ax.set_aspect(1)
        ax.axis('off')
        dx = dy = (0.7 - 0.3) / 3
        b = dx * b_scale
        d = dy * d_scale
        mid = 0.5
        x0 = mid - b / 2
        x1 = mid + b / 2
        y0 = mid - d / 2
        y1 = mid + d / 2
        ax.fill_between([x0, x1], [y0, y0], [y1, y1], color='grey', alpha=0.5)

        return fig

    @staticmethod
    def model_3d(model_size):
        """Отрисовка модели в трехмерном виде"""
        x, y, z = float(model_size[0]), float(model_size[1]), float(model_size[2])
        min_size = min(x, y, z)
        b_scale, d_scale, h_scale = x / min_size, y / min_size, z / min_size

        num_fig = f'Модель в трехмерном представлении {" ".join(model_size)}'
        fig = plt.figure(num=num_fig, dpi=Plot.dpi, clear=True)
        ax = fig.add_subplot(1, 1, 1, projection='3d')

        count_nodes = 2
        x = np.linspace(1, b_scale + 1, count_nodes)
        y = np.linspace(1, d_scale + 1, count_nodes)
        z = np.linspace(0, h_scale, count_nodes)

        ones1d = np.ones(count_nodes)
        zeros2d = np.zeros((count_nodes, count_nodes))

        X_top, Y_top = np.meshgrid(x, y)
        Z_top = zeros2d + h_scale
        ax.plot_surface(X_top, Y_top, Z_top, color='grey')

        X_face, Y_face = np.meshgrid(x, ones1d)
        Z_face = (zeros2d + z).T
        ax.plot_surface(X_face, Y_face, Z_face, color='grey')

        Y_side, X_side = np.meshgrid(y, ones1d + b_scale)
        ax.plot_surface(X_side, Y_side, Z_face, color='grey')

        ax.set_xlabel('Depth')
        ax.set_ylabel('Breadth')
        ax.set_zlabel('Height')

        # Get rid of the panes
        ax.w_xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        ax.w_yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        ax.w_zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))

        # Get rid of the spines
        ax.w_xaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
        ax.w_yaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
        ax.w_zaxis.line.set_color((1.0, 1.0, 1.0, 0.0))

        # Get rid of the ticks
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])

        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False

        # Now set color to white (or whatever is "invisible")
        ax.xaxis.pane.set_edgecolor('w')
        ax.yaxis.pane.set_edgecolor('w')
        ax.zaxis.pane.set_edgecolor('w')

        ax.grid(False)
        max_range = np.array([b_scale, d_scale, h_scale]).max() / 2 + 1

        mid_x = (b_scale + 1) * 0.5
        mid_y = (d_scale + 1) * 0.5
        mid_z = (h_scale + 1) * 0.5
        ax.set_xlim(mid_x - max_range, mid_x + max_range)
        ax.set_ylim(mid_y - max_range, mid_y + max_range)
        ax.set_zlim(mid_z - max_range, mid_z + max_range)

        return fig

    @staticmethod
    def envelopes(pressure_coefficients, alpha: str, model_scale: str, angle, mods):
        """Отрисовка огибающих"""

        colors = ('b', 'g', 'r', 'c', 'y')[:len(mods)]
        names = [i for i in ('mean', 'rms', 'std', 'max', 'min') if i in mods]
        lambdas = {
            'mean': lambda coefficients: np.mean(coefficients, axis=0).round(4),
            'rms': lambda coefficients: np.array([np.sqrt(i.dot(i) / i.size) for i in coefficients.T]).round(4),
            'std': lambda coefficients: np.std(coefficients, axis=0).round(4),
            'max': lambda coefficients: np.max(coefficients, axis=0).round(4),
            'min': lambda coefficients: np.min(coefficients, axis=0).round(4),
        }

        figs = []  # массив для графиков так как на 1 графике максимум 100 датчиков
        step_x = 20
        step_x_minor = 5
        step_y = 0.4

        step_sens = 100
        count_sensors_plot = len(pressure_coefficients[0])

        sensors = list(range(0, count_sensors_plot, step_sens))

        for q in sensors:
            num_fig = f'Огибающие {model_scale} {alpha} {angle} {q}_{"_".join(names)}'
            fig, ax = plt.subplots(dpi=Plot.dpi, num=num_fig, clear=True)
            ax.grid(visible=True, which='minor', color='black', linestyle='--')
            ax.grid(visible=True, which='major', color='black', linewidth=1.5)

            coefficients = pressure_coefficients.T[q:q + step_sens].T
            data_for_plot = [lambdas[name](coefficients) for name in names]

            if q == sensors[-1]:
                len_data = len(data_for_plot[0])
            else:
                len_data = step_sens

            ox = [i for i in range(q + 1, q + len_data + 1)]

            for i, j, c in zip(data_for_plot, names, colors):
                ax.plot(ox, i, '-', label=j, linewidth=3, color=c)

            ax.set_xlim([q + 1, q + len_data])

            all_data = []
            [all_data.extend(i) for i in data_for_plot]

            yticks = np.arange(np.min(all_data) - step_y, np.max(all_data) + step_y, step_y).round(2)
            ax.set_ylim(np.min(yticks) + 0.2, np.max(yticks) + 0.2)
            ax.set_yticks(yticks)

            ax.set_xticks([q + 1] + [i for i in range(q + step_x, q + len_data + 1, 20)])

            ax.xaxis.set_minor_locator(MultipleLocator(step_x_minor))
            ax.xaxis.set_minor_formatter(ScalarFormatter())
            ax.xaxis.set_tick_params(which='major', labelsize=10)
            ax.xaxis.set_tick_params(which='minor', labelsize=7)

            ax.legend(loc='upper right', fontsize=9)
            ax.set_title('Огибающие')

            figs.append(fig)

        return figs

    @staticmethod
    def welch_graphs(db='isolated', **kwargs):
        """Отрисовка графиков спектральной плотности мощности"""
        data = kwargs['data']
        if db == 'isolated':
            model_size = kwargs['model_size']
            alpha = kwargs['alpha']
            angle = kwargs['angle']
            num_fig = f'Спектральная плотность мощности {model_size} {alpha} {angle}'
            fs = 1000
            counts = 32768
        elif db == 'interference':
            model_size = kwargs['model_size']
            case = kwargs['case']
            angle = kwargs['angle']
            fs = 781
            counts = 5858
            num_fig = f'Спектральная плотность мощности {model_size} {case} {angle}'

        fig, ax = plt.subplots(dpi=Plot.dpi, num=num_fig, clear=True)

        ax.set_xlim([10 ** -2, 10 ** 3])
        ax.set_xscale('log')
        ax.set_yscale('log')

        ax.grid()
        ax.set_xlabel('Frequency')
        ax.set_ylabel('PSD, V**2/Hz')

        for name in data.keys():
            if data[name] is not None:
                temp, psd = welch(data[name], fs=fs, nperseg=int(counts / 5))
                ax.plot(temp, psd, label=name)
                print(psd.max())
                print(np.where(psd == psd.max()))
                print(temp[np.where(psd == psd.max())])

        ax.legend(loc='upper right', fontsize=9)

        return fig
