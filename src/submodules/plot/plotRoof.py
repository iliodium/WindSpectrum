import matplotlib
import matplotlib.tri as mtri
import numpy as np
from matplotlib import pyplot as plt
from src.submodules.plot.plot import Plot
from src.submodules.plot.utils import interpolator as intp


class PlotRoof(Plot):
    @staticmethod
    def isofields_coefficients_C(
            coordinates,
            _pressure_coefficients,
            size_tpu,
    ):
        roof_angle = 26.7
        _x_coordinates, _z_coordinates = coordinates
        breadth_tpu, depth_tpu, height_tpu = size_tpu

        count_sensors_on_eave_y = len(set(_z_coordinates[6]))
        count_sensors_on_face_1 = len(set(_z_coordinates[1]))

        length_eave = (breadth_tpu * count_sensors_on_eave_y * 2) / (count_sensors_on_face_1 * 2)

        count_faces = 9
        count_levels = 11

        width_plot = 2 * height_tpu + depth_tpu
        height_plot = 2 * height_tpu + breadth_tpu

        roof_length = breadth_tpu / (2 * np.cos(np.deg2rad(roof_angle)))
        height_in_triangle = np.sin(np.deg2rad(roof_angle)) * roof_length

        position_plot_on_grid = {
            1: [2, 2],
            2: [2, 0],
            3: [4, 2],
            4: [2, 4],
            5: [0, 2],
            6: [1, 2],
            7: [3, 2],
            8: [2, 1],
            9: [2, 3],

        }

        range_height = [i for i in range(height_tpu + 1)]
        range_breadth = [i for i in range(breadth_tpu + 1)]
        range_depth = [i for i in range(depth_tpu + 1)]

        full_height_side_faces = (height_in_triangle + height_tpu).round(2)
        range_full_height_side_faces = [i for i in range(int(full_height_side_faces) + 1)]
        range_full_height_side_faces.append(full_height_side_faces)

        range_breadth_eave = [i for i in range(int(length_eave) + 1)]
        range_breadth_eave.append(length_eave)

        x_labels = {
            1: range_depth,
            2: range_full_height_side_faces,
            3: range_depth,
            4: range_full_height_side_faces[::-1],
            5: range_depth,
            6: range_depth,
            7: range_depth,
            8: range_breadth_eave,
            9: range_breadth_eave,
        }

        range_breadth_and_eaves = [i for i in range(breadth_tpu + int(length_eave * 2) + 1)]
        range_breadth_and_eaves.append(breadth_tpu + length_eave * 2)

        y_labels = {
            1: range_breadth_and_eaves,
            2: range_breadth,
            3: range_height,
            4: range_breadth,
            5: range_height,
            6: range_breadth_eave,
            7: range_breadth_eave,
            8: range_breadth_and_eaves,
            9: range_breadth_and_eaves,

        }

        breadth_ratios = breadth_tpu / width_plot
        depth_ratios = depth_tpu / width_plot
        height_ratios = height_tpu / width_plot
        breadth_eave_ratios = length_eave / width_plot

        fig = plt.figure(figsize=(12, 12), dpi=80)
        grid = plt.GridSpec(5, 5, hspace=0.2, wspace=0.2,
                            width_ratios=[height_ratios, breadth_eave_ratios, depth_ratios, breadth_eave_ratios,
                                          height_ratios],
                            height_ratios=[height_ratios, breadth_eave_ratios, height_ratios, breadth_eave_ratios,
                                           height_ratios])

        cmap = matplotlib.colormaps.get_cmap("jet")

        levels_for_data = np.linspace(
            np.min([np.min(_pressure_coefficients[i]) for i in range(1, count_faces + 1)]),
            np.max([np.max(_pressure_coefficients[i]) for i in range(1, count_faces + 1)]),
            count_levels)

        left_right_face = (2, 4)

        for face in range(2, 10):
            x_coordinates = _x_coordinates[face]
            z_coordinates = _z_coordinates[face]
            pressure_coefficients = _pressure_coefficients[face]

            set_x_coordinates = sorted(list(set(x_coordinates)))
            set_y_coordinates = sorted(list(set(z_coordinates)))

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

                    for x_coord, y_coord in zip(x_coordinates, z_coordinates):
                        if x == x_coord:
                            local_max_y = y_coord if y_coord > local_max_y else local_max_y
                            local_min_y = y_coord if y_coord < local_min_y else local_min_y

                    boundary_coordinates_x.extend([x, x])
                    boundary_coordinates_y.extend([local_min_y - 1, local_max_y + 1])

                for y in set_y_coordinates:
                    local_max_x = float('-inf')
                    local_min_x = float('+inf')

                    for x_coord, y_coord in zip(x_coordinates, z_coordinates):
                        if y == y_coord:
                            local_max_x = x_coord if x_coord > local_max_x else local_max_x
                            local_min_x = x_coord if x_coord < local_min_x else local_min_x

                    boundary_coordinates_x.extend([local_min_x - 1, local_max_x + 1])
                    boundary_coordinates_y.extend([y, y])

                for x, y in zip(x_coordinates, z_coordinates):
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

            if face in (6, 7):
                coords = [[i1, j1] for i1, j1 in zip(x_coordinates, z_coordinates)]  # Старые координаты
                pressure_coefficients_for_int = list(pressure_coefficients)
                if face == 6:
                    coords.extend([[i1, j1] for i1, j1 in zip(_x_coordinates[5], _z_coordinates[5])])
                    pressure_coefficients_for_int.extend(_pressure_coefficients[5])

                elif face == 7:
                    coords.extend([[i1, j1] for i1, j1 in zip(_x_coordinates[3], _z_coordinates[3])])
                    pressure_coefficients_for_int.extend(_pressure_coefficients[3])

                # Интерполятор полученный на основе имеющихся данных
                interpolator = intp(coords, pressure_coefficients_for_int)
            elif face in (8, 9):
                coords = [[i1, j1] for i1, j1 in zip(x_coordinates, z_coordinates)]  # Старые координаты
                pressure_coefficients_for_int = list(pressure_coefficients)
                if face == 8:
                    coords.extend([[i1, j1] for i1, j1 in zip(_x_coordinates[2], _z_coordinates[2])])
                    pressure_coefficients_for_int.extend(_pressure_coefficients[2])

                elif face == 9:
                    coords.extend([[i1, j1] for i1, j1 in zip(_x_coordinates[4], _z_coordinates[4])])
                    pressure_coefficients_for_int.extend(_pressure_coefficients[4])

                # Интерполятор полученный на основе имеющихся данных
                interpolator = intp(coords, pressure_coefficients_for_int)
            else:
                coords = [[i1, j1] for i1, j1 in zip(x_coordinates, z_coordinates)]  # Старые координаты
                # Интерполятор полученный на основе имеющихся данных
                interpolator = intp(coords, pressure_coefficients)

            data_from_interpolator = [float(interpolator([[X, Y]])) for X, Y in
                                      zip(boundary_coordinates_x, boundary_coordinates_y)]

            full_x_coordinates = x_coordinates + boundary_coordinates_x
            full_y_coordinates = z_coordinates + boundary_coordinates_y
            full_data = pressure_coefficients + data_from_interpolator

            #     full_x_coordinates = x_coordinates
            #     full_y_coordinates = y_coordinates
            #     full_data = pressure_coefficients

            temp_triang = mtri.Triangulation(full_x_coordinates, full_y_coordinates)

            temp_refiner = mtri.UniformTriRefiner(temp_triang)

            temp_grid, temp_value = temp_refiner.refine_field(full_data, subdiv=4)

            temp_yticks = [j for j in range(new_min_y_coordinate, new_max_y_coordinate + 1)]
            temp_xticks = [j for j in range(new_min_x_coordinate, new_max_x_coordinate + 1)]

            temp_plot = fig.add_subplot(grid[position_plot_on_grid[face][0], position_plot_on_grid[face][1]],
                                        xticklabels=x_labels[face], xticks=temp_xticks,
                                        yticklabels=y_labels[face], yticks=temp_yticks)

            temp_plot.tricontourf(temp_grid, temp_value, cmap=cmap, extend='both', levels=levels_for_data)
            temp_contour = temp_plot.tricontour(temp_grid, temp_value, linewidths=1, linestyles='solid', colors='black',
                                                levels=levels_for_data)

            temp_plot.clabel(temp_contour, fontsize=10)

            temp_plot.plot(x_coordinates, z_coordinates, '.k', **dict(markersize=3))

            # temp_plot.axis('off')
            # temp_plot.scatter(x_coordinates, y_coordinates)
            # temp_plot.scatter(boundary_coordinates_x,boundary_coordinates_y)

        union_face = 1
        set_x_coordinates = sorted(list(set(_x_coordinates[union_face])))
        set_y_coordinates = sorted(list(set(_z_coordinates[union_face])))

        union_plot = fig.add_subplot(grid[position_plot_on_grid[union_face][0], position_plot_on_grid[union_face][1]],
                                     xticklabels=x_labels[union_face],
                                     xticks=[i for i in range(min(set_x_coordinates) - 1, max(set_x_coordinates) + 2)],
                                     yticklabels=y_labels[union_face],
                                     yticks=[i for i in range(min(set_y_coordinates) - 1, max(set_y_coordinates) + 2)])

        for face in (1,):
            x_coordinates = _x_coordinates[face]
            z_coordinates = _z_coordinates[face]
            pressure_coefficients = _pressure_coefficients[face]

            set_x_coordinates = sorted(list(set(x_coordinates)))
            set_y_coordinates = sorted(list(set(z_coordinates)))

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

            coords = [[i1, j1] for i1, j1 in zip(x_coordinates, z_coordinates)]  # Старые координаты
            # Интерполятор полученный на основе имеющихся данных
            extrapolator = intp(coords, pressure_coefficients)

            data_from_interpolator = [float(extrapolator([[X, Y]])) for X, Y in
                                      zip(boundary_coordinates_x, boundary_coordinates_y)]

            full_x_coordinates = x_coordinates + boundary_coordinates_x
            full_y_coordinates = z_coordinates + boundary_coordinates_y
            full_data = pressure_coefficients + data_from_interpolator

            temp_triang = mtri.Triangulation(full_x_coordinates, full_y_coordinates)

            temp_refiner = mtri.UniformTriRefiner(temp_triang)

            temp_grid, temp_value = temp_refiner.refine_field(full_data, subdiv=4)

            union_plot.tricontourf(temp_grid, temp_value, cmap=cmap, extend='both', levels=levels_for_data)
            temp_contour = union_plot.tricontour(temp_grid, temp_value, linewidths=1, linestyles='solid',
                                                 colors='black',
                                                 levels=levels_for_data)

            union_plot.clabel(temp_contour, fontsize=10)

            union_plot.plot(x_coordinates, z_coordinates, '.k', **dict(markersize=3))

        return fig

    @staticmethod
    def isofields_coefficients_A_B(
            coordinates,
            _pressure_coefficients,
            size_tpu,
    ):
        roof_angle = 26.7

        _x_coordinates, _z_coordinates = coordinates
        breadth_tpu, depth_tpu, height_tpu = size_tpu

        count_sensors_on_eave_y = len(set(_x_coordinates[6]))
        count_sensors_on_face_1 = len(set(_z_coordinates[1]))

        length_eave = (breadth_tpu * count_sensors_on_eave_y * 2) / (count_sensors_on_face_1 * 2)

        count_faces = 7
        count_levels = 11

        width_plot = 2 * height_tpu + depth_tpu
        height_plot = 2 * height_tpu + breadth_tpu

        roof_length = breadth_tpu / (2 * np.cos(np.deg2rad(roof_angle)))
        height_in_triangle = np.sin(np.deg2rad(roof_angle)) * roof_length

        position_plot_on_grid = {
            1: [2, 1],
            2: [2, 0],
            3: [4, 1],
            4: [2, 2],
            5: [0, 1],
            6: [1, 1],
            7: [3, 1]
        }

        range_height = [i for i in range(height_tpu + 1)]
        range_breadth = [i for i in range(breadth_tpu + 1)]
        range_depth = [i for i in range(depth_tpu + 1)]

        full_height_side_faces = (height_in_triangle + height_tpu).round(2)
        range_full_height_side_faces = [i for i in range(int(full_height_side_faces) + 1)]
        range_full_height_side_faces.append(full_height_side_faces)

        range_breadth_eave = [i for i in range(int(length_eave) + 1)]
        range_breadth_eave.append(length_eave)

        x_labels = {
            1: range_depth,
            2: range_full_height_side_faces,
            3: range_depth,
            4: range_full_height_side_faces[::-1],
            5: range_depth,
            6: range_depth,
            7: range_depth,

        }

        range_breadth_and_eaves = [i for i in range(breadth_tpu + int(length_eave * 2) + 1)]
        range_breadth_and_eaves.append(breadth_tpu + length_eave * 2)

        y_labels = {
            1: range_breadth_and_eaves,
            2: range_breadth,
            3: range_height,
            4: range_breadth,
            5: range_height,
            6: range_breadth_eave,
            7: range_breadth_eave,
        }

        breadth_ratios = breadth_tpu / width_plot
        depth_ratios = depth_tpu / width_plot
        height_ratios = height_tpu / width_plot
        breadth_eave_ratios = length_eave / width_plot

        fig = plt.figure(figsize=(12, 12), dpi=PlotRoof.DPI)
        grid = plt.GridSpec(5, 3, hspace=0.2, wspace=0.2,
                            width_ratios=[height_ratios, depth_ratios, height_ratios],
                            height_ratios=[height_ratios, breadth_eave_ratios, height_ratios, breadth_eave_ratios,
                                           height_ratios])
        # fig, axs = plt.subplots(nrows=5, ncols=3, layout='constrained')
        cmap = matplotlib.colormaps.get_cmap("jet")

        levels_for_data = np.linspace(
            np.min([np.min(_pressure_coefficients[i]) for i in range(1, count_faces + 1)]),
            np.max([np.max(_pressure_coefficients[i]) for i in range(1, count_faces + 1)]),
            count_levels)

        left_right_face = (2, 4)

        for face in range(2, 8):
            # for face in range(6, 8):
            x_coordinates = _x_coordinates[face]
            z_coordinates = _z_coordinates[face]
            pressure_coefficients = _pressure_coefficients[face]

            set_x_coordinates = sorted(list(set(x_coordinates)))
            set_y_coordinates = sorted(list(set(z_coordinates)))

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

                    for x_coord, y_coord in zip(x_coordinates, z_coordinates):
                        if x == x_coord:
                            local_max_y = y_coord if y_coord > local_max_y else local_max_y
                            local_min_y = y_coord if y_coord < local_min_y else local_min_y

                    boundary_coordinates_x.extend([x, x])
                    boundary_coordinates_y.extend([local_min_y - 1, local_max_y + 1])

                for y in set_y_coordinates:
                    local_max_x = float('-inf')
                    local_min_x = float('+inf')

                    for x_coord, y_coord in zip(x_coordinates, z_coordinates):
                        if y == y_coord:
                            local_max_x = x_coord if x_coord > local_max_x else local_max_x
                            local_min_x = x_coord if x_coord < local_min_x else local_min_x

                    boundary_coordinates_x.extend([local_min_x - 1, local_max_x + 1])
                    boundary_coordinates_y.extend([y, y])

                for x, y in zip(x_coordinates, z_coordinates):
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

            if face in (6, 7):
                coords = [[i1, j1] for i1, j1 in zip(x_coordinates, z_coordinates)]  # Старые координаты
                pressure_coefficients_for_int = list(pressure_coefficients)
                if face == 6:
                    coords.extend([[i1, j1] for i1, j1 in zip(_x_coordinates[5], _z_coordinates[5])])
                    pressure_coefficients_for_int.extend(_pressure_coefficients[5])

                elif face == 7:
                    coords.extend([[i1, j1] for i1, j1 in zip(_x_coordinates[3], _z_coordinates[3])])
                    pressure_coefficients_for_int.extend(_pressure_coefficients[3])

                # Интерполятор полученный на основе имеющихся данных
                interpolator = intp(coords, pressure_coefficients_for_int)
            else:
                coords = [[i1, j1] for i1, j1 in zip(x_coordinates, z_coordinates)]  # Старые координаты
                # Интерполятор полученный на основе имеющихся данных
                interpolator = intp(coords, pressure_coefficients)

            data_from_interpolator = [float(interpolator([[X, Y]])) for X, Y in
                                      zip(boundary_coordinates_x, boundary_coordinates_y)]

            full_x_coordinates = x_coordinates + boundary_coordinates_x
            full_y_coordinates = z_coordinates + boundary_coordinates_y
            full_data = pressure_coefficients + data_from_interpolator

            #     full_x_coordinates = x_coordinates
            #     full_y_coordinates = y_coordinates
            #     full_data = pressure_coefficients

            temp_triang = mtri.Triangulation(full_x_coordinates, full_y_coordinates)

            temp_refiner = mtri.UniformTriRefiner(temp_triang)

            temp_grid, temp_value = temp_refiner.refine_field(full_data, subdiv=4)

            temp_yticks = [j for j in range(new_min_y_coordinate, new_max_y_coordinate + 1)]
            temp_xticks = [j for j in range(new_min_x_coordinate, new_max_x_coordinate + 1)]

            temp_plot = fig.add_subplot(grid[position_plot_on_grid[face][0], position_plot_on_grid[face][1]],
                                        xticklabels=x_labels[face], xticks=temp_xticks,
                                        yticklabels=y_labels[face], yticks=temp_yticks)

            temp_plot.tricontourf(temp_grid, temp_value, cmap=cmap, extend='both', levels=levels_for_data)
            temp_contour = temp_plot.tricontour(temp_grid, temp_value, linewidths=1, linestyles='solid', colors='black',
                                                levels=levels_for_data)

            temp_plot.clabel(temp_contour, fontsize=10)

            temp_plot.plot(x_coordinates, z_coordinates, '.k', **dict(markersize=3))

            # temp_plot.axis('off')
            # temp_plot.scatter(x_coordinates, y_coordinates)
            # temp_plot.scatter(boundary_coordinates_x,boundary_coordinates_y)

        union_face = 1
        set_x_coordinates = sorted(list(set(_x_coordinates[union_face])))
        set_y_coordinates = sorted(list(set(_z_coordinates[union_face])))

        union_plot = fig.add_subplot(grid[position_plot_on_grid[union_face][0], position_plot_on_grid[union_face][1]],
                                     xticklabels=x_labels[union_face],
                                     xticks=[i for i in range(min(set_x_coordinates) - 1, max(set_x_coordinates) + 2)],
                                     yticklabels=y_labels[union_face],
                                     yticks=[i for i in range(min(set_y_coordinates) - 1, max(set_y_coordinates) + 2)])

        for face in (1,):
            x_coordinates = _x_coordinates[face]
            z_coordinates = _z_coordinates[face]
            pressure_coefficients = _pressure_coefficients[face]

            set_x_coordinates = sorted(list(set(x_coordinates)))
            set_y_coordinates = sorted(list(set(z_coordinates)))

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

            coords = [[i1, j1] for i1, j1 in zip(x_coordinates, z_coordinates)]  # Старые координаты
            # Интерполятор полученный на основе имеющихся данных
            extrapolator = intp(coords, pressure_coefficients)

            data_from_interpolator = [float(extrapolator([[X, Y]])) for X, Y in
                                      zip(boundary_coordinates_x, boundary_coordinates_y)]

            full_x_coordinates = x_coordinates + boundary_coordinates_x
            full_y_coordinates = z_coordinates + boundary_coordinates_y
            full_data = pressure_coefficients + data_from_interpolator

            temp_triang = mtri.Triangulation(full_x_coordinates, full_y_coordinates)

            temp_refiner = mtri.UniformTriRefiner(temp_triang)

            temp_grid, temp_value = temp_refiner.refine_field(full_data, subdiv=4)

            union_plot.tricontourf(temp_grid, temp_value, cmap=cmap, extend='both', levels=levels_for_data)
            temp_contour = union_plot.tricontour(temp_grid, temp_value, linewidths=1, linestyles='solid',
                                                 colors='black',
                                                 levels=levels_for_data)

            union_plot.clabel(temp_contour, fontsize=10)

            union_plot.plot(x_coordinates, z_coordinates, '.k', **dict(markersize=3))

        return fig

    @staticmethod
    def isofields_coefficients_O(
            coordinates,
            _pressure_coefficients,
            size_tpu,
    ):
        roof_angle = 26.7
        _x_coordinates, _z_coordinates = coordinates
        breadth_tpu, depth_tpu, height_tpu = size_tpu

        count_faces = 5
        count_levels = 11

        width_plot = 2 * height_tpu + depth_tpu
        height_plot = 2 * height_tpu + breadth_tpu

        roof_length = breadth_tpu / (2 * np.cos(np.deg2rad(roof_angle)))
        height_in_triangle = np.sin(np.deg2rad(roof_angle)) * roof_length

        position_plot_on_grid = {
            1: [1, 1],
            2: [1, 0],
            3: [2, 1],
            4: [1, 2],
            5: [0, 1],
        }

        range_height = [i for i in range(height_tpu + 1)]
        range_breadth = [i for i in range(breadth_tpu + 1)]
        range_depth = [i for i in range(depth_tpu + 1)]

        full_height_side_faces = (height_in_triangle + height_tpu).round(2)
        range_full_height_side_faces = [i for i in range(int(full_height_side_faces) + 1)]
        range_full_height_side_faces.append(full_height_side_faces)

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

        breadth_ratios = breadth_tpu / width_plot
        depth_ratios = depth_tpu / width_plot
        height_ratios = height_tpu / width_plot

        fig = plt.figure(figsize=(12, 12), dpi=80)
        grid = plt.GridSpec(3, 3, hspace=0.2, wspace=0.2,
                            width_ratios=[height_ratios, depth_ratios, height_ratios],
                            height_ratios=[height_ratios, height_ratios, height_ratios])

        cmap = matplotlib.colormaps.get_cmap("jet")

        levels_for_data = np.linspace(
            np.min([np.min(_pressure_coefficients[i]) for i in range(1, count_faces + 1)]),
            np.max([np.max(_pressure_coefficients[i]) for i in range(1, count_faces + 1)]),
            count_levels)

        left_right_face = (2, 4)

        for face in range(2, 6):
            x_coordinates = _x_coordinates[face]
            y_coordinates = _z_coordinates[face]
            pressure_coefficients = _pressure_coefficients[face]

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

            #     full_x_coordinates = x_coordinates
            #     full_y_coordinates = y_coordinates
            #     full_data = pressure_coefficients

            temp_triang = mtri.Triangulation(full_x_coordinates, full_y_coordinates)

            temp_refiner = mtri.UniformTriRefiner(temp_triang)

            temp_grid, temp_value = temp_refiner.refine_field(full_data, subdiv=4)

            temp_yticks = [j for j in range(new_min_y_coordinate, new_max_y_coordinate + 1)]
            temp_xticks = [j for j in range(new_min_x_coordinate, new_max_x_coordinate + 1)]

            temp_plot = fig.add_subplot(grid[position_plot_on_grid[face][0], position_plot_on_grid[face][1]],
                                        xticklabels=x_labels[face], xticks=temp_xticks,
                                        yticklabels=y_labels[face], yticks=temp_yticks)

            temp_plot.tricontourf(temp_grid, temp_value, cmap=cmap, extend='both', levels=levels_for_data)
            temp_contour = temp_plot.tricontour(temp_grid, temp_value, linewidths=1, linestyles='solid', colors='black',
                                                levels=levels_for_data)

            temp_plot.clabel(temp_contour, fontsize=10)

            temp_plot.plot(x_coordinates, y_coordinates, '.k', **dict(markersize=3))
            # temp_plot.axis('off')
            # temp_plot.scatter(boundary_coordinates_x,boundary_coordinates_y)

        union_face = 1
        set_x_coordinates = sorted(list(set(_x_coordinates[union_face])))
        set_y_coordinates = sorted(list(set(_z_coordinates[union_face])))

        union_plot = fig.add_subplot(grid[position_plot_on_grid[union_face][0], position_plot_on_grid[union_face][1]],
                                     xticklabels=x_labels[union_face],
                                     xticks=[i for i in range(min(set_x_coordinates) - 1, max(set_x_coordinates) + 2)],
                                     yticklabels=y_labels[union_face],
                                     yticks=[i for i in range(min(set_y_coordinates) - 1, max(set_y_coordinates) + 2)])

        for face in (1,):
            x_coordinates = _x_coordinates[face]
            y_coordinates = _z_coordinates[face]
            pressure_coefficients = _pressure_coefficients[face]

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
    def isofields_coefficients_flat_roof(
            coordinates,
            _pressure_coefficients,
            size_tpu,
    ):

        _x_coordinates, _z_coordinates = coordinates
        breadth_tpu, depth_tpu, height_tpu = size_tpu
        count_faces = 5
        count_levels = 11

        width_plot = 2 * height_tpu + depth_tpu
        height_plot = 2 * height_tpu + breadth_tpu

        position_plot_on_grid = {
            1: [1, 0],
            2: [2, 1],
            3: [1, 2],
            4: [0, 1],
            5: [1, 1],
        }

        range_height = [i for i in range(height_tpu + 1)]
        range_breadth = [i for i in range(breadth_tpu + 1)]
        range_depth = [i for i in range(depth_tpu + 1)]

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

        breadth_ratios = breadth_tpu / width_plot
        depth_ratios = depth_tpu / width_plot
        height_ratios = height_tpu / width_plot

        fig = plt.figure(figsize=(12, 12), dpi=80)
        grid = plt.GridSpec(3, 3, hspace=0.2, wspace=0.2,
                            width_ratios=[height_ratios, depth_ratios, height_ratios],
                            height_ratios=[height_ratios, height_ratios, height_ratios])

        cmap = matplotlib.colormaps.get_cmap("jet")

        levels_for_data = np.linspace(
            np.min([np.min(_pressure_coefficients[i]) for i in range(1, count_faces + 1)]),
            np.max([np.max(_pressure_coefficients[i]) for i in range(1, count_faces + 1)]),
            count_levels)

        for face in range(1, count_faces + 1):
            x_coordinates = _x_coordinates[face]
            z_coordinates = _z_coordinates[face]
            pressure_coefficients = _pressure_coefficients[face]

            set_x_coordinates = sorted(list(set(x_coordinates)))
            set_y_coordinates = sorted(list(set(z_coordinates)))

            max_x_coordinate = max(set_x_coordinates)
            min_x_coordinate = min(set_x_coordinates)
            new_max_x_coordinate = max_x_coordinate + 1
            new_min_x_coordinate = min_x_coordinate - 1

            max_y_coordinate = max(set_y_coordinates)
            min_y_coordinate = min(set_y_coordinates)
            new_max_y_coordinate = max_y_coordinate + 1
            new_min_y_coordinate = min_y_coordinate - 1

            x_coordinates_left_border = [new_min_x_coordinate for _ in range(len(set_y_coordinates) + 2)]
            x_coordinates_right_border = [new_max_x_coordinate for _ in range(len(set_y_coordinates) + 2)]

            boundary_coordinates_x = x_coordinates_left_border + set_x_coordinates + \
                                     x_coordinates_right_border + set_x_coordinates

            y_coordinates_top_border = [new_max_y_coordinate for _ in range(len(set_x_coordinates) + 2)]
            y_coordinates_bottom_border = [new_min_y_coordinate for _ in range(len(set_x_coordinates) + 1)]

            boundary_coordinates_y = [new_min_y_coordinate] + set_y_coordinates + y_coordinates_top_border + \
                                     set_y_coordinates + y_coordinates_bottom_border

            coords = [[i1, j1] for i1, j1 in zip(x_coordinates, z_coordinates)]  # Старые координаты
            # Интерполятор полученный на основе имеющихся данных
            interpolator = intp(coords, pressure_coefficients)

            data_from_interpolator = [float(interpolator([[X, Y]])) for X, Y in
                                      zip(boundary_coordinates_x, boundary_coordinates_y)]

            full_x_coordinates = x_coordinates + boundary_coordinates_x
            full_y_coordinates = z_coordinates + boundary_coordinates_y
            full_data = pressure_coefficients + data_from_interpolator

            temp_triang = mtri.Triangulation(full_x_coordinates, full_y_coordinates)

            temp_refiner = mtri.UniformTriRefiner(temp_triang)

            temp_grid, temp_value = temp_refiner.refine_field(full_data, subdiv=4)

            temp_xticks = [j for j in range(new_min_x_coordinate, new_max_x_coordinate + 1)]
            temp_yticks = [j for j in range(new_min_y_coordinate, new_max_y_coordinate + 1)]

            temp_plot = fig.add_subplot(grid[position_plot_on_grid[face][0], position_plot_on_grid[face][1]],
                                        xticklabels=x_labels[face], xticks=temp_xticks,
                                        yticklabels=y_labels[face], yticks=temp_yticks)

            temp_plot.tricontourf(temp_grid, temp_value, cmap=cmap, extend='both', levels=levels_for_data)
            temp_contour = temp_plot.tricontour(temp_grid, temp_value, linewidths=1, linestyles='solid', colors='black',
                                                levels=levels_for_data)

            temp_plot.clabel(temp_contour, fontsize=10)

            x_dots, y_dots = np.meshgrid(x_coordinates, z_coordinates)
            temp_plot.plot(x_dots, y_dots, '.k', **dict(markersize=3))

            # temp_plot.scatter(extrapolated_x,extrapolated_y)

        return fig

    @staticmethod
    def isofields_coefficients_flat_roof_non_isolated(
            coordinates,
            _pressure_coefficients,
            size_tpu
    ):
        roof_angle = 0
        _x_coordinates, _z_coordinates = coordinates
        breadth_tpu, depth_tpu, height_tpu = size_tpu

        count_faces = 5
        count_levels = 11

        width_plot = 2 * height_tpu + depth_tpu
        height_plot = 2 * height_tpu + breadth_tpu

        roof_length = breadth_tpu / (2 * np.cos(np.deg2rad(roof_angle)))
        height_in_triangle = np.sin(np.deg2rad(roof_angle)) * roof_length

        position_plot_on_grid = {
            1: [1, 1],
            2: [1, 0],
            3: [2, 1],
            4: [1, 2],
            5: [0, 1],
        }

        range_height = [i for i in range(height_tpu + 1)]
        range_breadth = [i for i in range(breadth_tpu + 1)]
        range_depth = [i for i in range(depth_tpu + 1)]

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

        breadth_ratios = breadth_tpu / width_plot
        depth_ratios = depth_tpu / width_plot
        height_ratios = height_tpu / width_plot

        fig = plt.figure(figsize=(12, 12), dpi=80)
        grid = plt.GridSpec(3, 3, hspace=0.2, wspace=0.2,
                            width_ratios=[height_ratios, depth_ratios, height_ratios],
                            height_ratios=[height_ratios, height_ratios, height_ratios])

        cmap = matplotlib.colormaps.get_cmap("jet")

        levels_for_data = np.linspace(
            np.min([np.min(_pressure_coefficients[i]) for i in range(1, count_faces + 1)]),
            np.max([np.max(_pressure_coefficients[i]) for i in range(1, count_faces + 1)]),
            count_levels)

        left_right_face = (2, 4)
        top_bottom_face = (5, 3)

        for face in range(2, 6):

            x_coordinates = _x_coordinates[face]
            z_coordinates = _z_coordinates[face]
            pressure_coefficients = _pressure_coefficients[face]

            set_x_coordinates = sorted(list(set(x_coordinates)))
            set_y_coordinates = sorted(list(set(z_coordinates)))

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

                    for x_coord, y_coord in zip(x_coordinates, z_coordinates):
                        if x == x_coord:
                            local_max_y = y_coord if y_coord > local_max_y else local_max_y
                            local_min_y = y_coord if y_coord < local_min_y else local_min_y

                    boundary_coordinates_x.extend([x, x])
                    boundary_coordinates_y.extend([local_min_y - 1, local_max_y + 1])

                for y in set_y_coordinates:
                    local_max_x = float('-inf')
                    local_min_x = float('+inf')

                    for x_coord, y_coord in zip(x_coordinates, z_coordinates):
                        if y == y_coord:
                            local_max_x = x_coord if x_coord > local_max_x else local_max_x
                            local_min_x = x_coord if x_coord < local_min_x else local_min_x

                    boundary_coordinates_x.extend([local_min_x - 1, local_max_x + 1])
                    boundary_coordinates_y.extend([y, y])

                for x, y in zip(x_coordinates, z_coordinates):
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

            coords = [[i1, j1] for i1, j1 in zip(x_coordinates, z_coordinates)]  # Старые координаты
            # Интерполятор полученный на основе имеющихся данных
            interpolator = intp(coords, pressure_coefficients)

            data_from_interpolator = [float(interpolator([[X, Y]])) for X, Y in
                                      zip(boundary_coordinates_x, boundary_coordinates_y)]

            full_x_coordinates = x_coordinates + boundary_coordinates_x
            full_y_coordinates = z_coordinates + boundary_coordinates_y
            full_data = pressure_coefficients + data_from_interpolator

            temp_triang = mtri.Triangulation(full_x_coordinates, full_y_coordinates)

            temp_refiner = mtri.UniformTriRefiner(temp_triang)

            temp_grid, temp_value = temp_refiner.refine_field(full_data, subdiv=4)

            #     temp_xticks = [j for j in range(min(full_x_coordinates), max(full_x_coordinates) + 1)]
            #     temp_yticks = [j for j in range(min(full_y_coordinates), max(full_y_coordinates) + 1)]

            temp_xticks = np.linspace(new_min_x_coordinate, new_max_x_coordinate, len(x_labels[face]))
            temp_yticks = np.linspace(new_min_y_coordinate, new_max_y_coordinate, len(y_labels[face]))

            temp_plot = fig.add_subplot(grid[position_plot_on_grid[face][0], position_plot_on_grid[face][1]],
                                        xticklabels=x_labels[face], xticks=temp_xticks,
                                        yticklabels=y_labels[face], yticks=temp_yticks)

            temp_plot.tricontourf(temp_grid, temp_value, cmap=cmap, extend='both', levels=levels_for_data)
            temp_contour = temp_plot.tricontour(temp_grid, temp_value, linewidths=1, linestyles='solid', colors='black',
                                                levels=levels_for_data)

            temp_plot.clabel(temp_contour, fontsize=10)

            temp_plot.plot(x_coordinates, z_coordinates, '.k', **dict(markersize=3))
            # temp_plot.axis('off')
            # temp_plot.scatter(boundary_coordinates_x,boundary_coordinates_y)

        union_face = 1
        set_x_coordinates = sorted(list(set(_x_coordinates[union_face])))
        set_y_coordinates = sorted(list(set(_z_coordinates[union_face])))

        union_plot = fig.add_subplot(grid[position_plot_on_grid[union_face][0], position_plot_on_grid[union_face][1]],
                                     xticklabels=x_labels[union_face],
                                     xticks=[i for i in range(min(set_x_coordinates) - 1, max(set_x_coordinates) + 2)],
                                     yticklabels=y_labels[union_face],
                                     yticks=[i for i in range(min(set_y_coordinates) - 1, max(set_y_coordinates) + 2)])

        for face in (1,):
            x_coordinates = _x_coordinates[face]
            z_coordinates = _z_coordinates[face]
            pressure_coefficients = _pressure_coefficients[face]

            set_x_coordinates = sorted(list(set(x_coordinates)))
            set_y_coordinates = sorted(list(set(z_coordinates)))

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

            coords = [[i1, j1] for i1, j1 in zip(x_coordinates, z_coordinates)]  # Старые координаты
            # Интерполятор полученный на основе имеющихся данных
            extrapolator = intp(coords, pressure_coefficients)

            data_from_interpolator = [float(extrapolator([[X, Y]])) for X, Y in
                                      zip(boundary_coordinates_x, boundary_coordinates_y)]

            full_x_coordinates = x_coordinates + boundary_coordinates_x
            full_y_coordinates = z_coordinates + boundary_coordinates_y
            full_data = pressure_coefficients + data_from_interpolator

            temp_triang = mtri.Triangulation(full_x_coordinates, full_y_coordinates)

            temp_refiner = mtri.UniformTriRefiner(temp_triang)

            temp_grid, temp_value = temp_refiner.refine_field(full_data, subdiv=4)

            union_plot.tricontourf(temp_grid, temp_value, cmap=cmap, extend='both', levels=levels_for_data)
            temp_contour = union_plot.tricontour(temp_grid, temp_value, linewidths=1, linestyles='solid',
                                                 colors='black',
                                                 levels=levels_for_data)

            union_plot.clabel(temp_contour, fontsize=10)

            union_plot.plot(x_coordinates, z_coordinates, '.k', **dict(markersize=3))

        return fig

    @staticmethod
    def isofields_coefficients_gable_roof(
            coordinates,
            _pressure_coefficients,
            size_tpu,
            roof_angle
    ):
        _x_coordinates, _z_coordinates = coordinates
        breadth_tpu, depth_tpu, height_tpu = size_tpu

        count_faces = 6
        count_levels = 11

        width_plot = 2 * height_tpu + depth_tpu
        height_plot = 2 * height_tpu + breadth_tpu

        roof_length = breadth_tpu / (2 * np.cos(np.deg2rad(roof_angle)))
        height_in_triangle = np.sin(np.deg2rad(roof_angle)) * roof_length

        position_plot_on_grid = {
            1: [1, 0],
            2: [2, 1],
            3: [1, 2],
            4: [0, 1],
            5: [1, 1],
        }

        range_height = [i for i in range(height_tpu + 1)]
        range_breadth = [i for i in range(breadth_tpu + 1)]
        range_depth = [i for i in range(depth_tpu + 1)]

        full_height_side_faces = (height_in_triangle + height_tpu).round(2)
        range_full_height_side_faces = [i for i in range(int(full_height_side_faces) + 1)]
        range_full_height_side_faces.append(full_height_side_faces)

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

        breadth_ratios = breadth_tpu / width_plot
        depth_ratios = depth_tpu / width_plot
        height_ratios = height_tpu / width_plot

        fig = plt.figure(figsize=(12, 12), dpi=80)
        grid = plt.GridSpec(3, 3, hspace=0.2, wspace=0.2,
                            width_ratios=[height_ratios, depth_ratios, height_ratios],
                            height_ratios=[height_ratios, height_ratios, height_ratios])

        cmap = matplotlib.colormaps.get_cmap("jet")

        levels_for_data = np.linspace(
            np.min([np.min(_pressure_coefficients[i]) for i in range(1, count_faces + 1)]),
            np.max([np.max(_pressure_coefficients[i]) for i in range(1, count_faces + 1)]),
            count_levels)

        left_right_face = (1, 3)
        top_bottom_face = (2, 4)

        for face in range(1, 5):
            x_coordinates = _x_coordinates[face]
            z_coordinates = _z_coordinates[face]
            pressure_coefficients = _pressure_coefficients[face]

            set_x_coordinates = sorted(list(set(x_coordinates)))
            set_y_coordinates = sorted(list(set(z_coordinates)))

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
                if face == 1:
                    boundary_coordinates_x.append(max(set_x_coordinates) + 2)
                    boundary_coordinates_y.append(sum(set_y_coordinates) / len(set_y_coordinates))
                else:
                    boundary_coordinates_x.append(min_x_coordinate - 2)
                    boundary_coordinates_y.append(sum(set_y_coordinates) / len(set_y_coordinates))

                for x in set_x_coordinates:
                    local_max_y = float('-inf')
                    local_min_y = float('+inf')

                    for x_coord, y_coord in zip(x_coordinates, z_coordinates):
                        if x == x_coord:
                            local_max_y = y_coord if y_coord > local_max_y else local_max_y
                            local_min_y = y_coord if y_coord < local_min_y else local_min_y

                    boundary_coordinates_x.extend([x, x])
                    boundary_coordinates_y.extend([local_min_y - 1, local_max_y + 1])

                for y in set_y_coordinates:
                    local_max_x = float('-inf')
                    local_min_x = float('+inf')

                    for x_coord, y_coord in zip(x_coordinates, z_coordinates):
                        if y == y_coord:
                            local_max_x = x_coord if x_coord > local_max_x else local_max_x
                            local_min_x = x_coord if x_coord < local_min_x else local_min_x

                    boundary_coordinates_x.extend([local_min_x - 1, local_max_x + 1])
                    boundary_coordinates_y.extend([y, y])

                for x, y in zip(x_coordinates, z_coordinates):
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

            coords = [[i1, j1] for i1, j1 in zip(x_coordinates, z_coordinates)]  # Старые координаты
            # Интерполятор полученный на основе имеющихся данных
            interpolator = intp(coords, pressure_coefficients)

            data_from_interpolator = [float(interpolator([[X, Y]])) for X, Y in
                                      zip(boundary_coordinates_x, boundary_coordinates_y)]

            full_x_coordinates = x_coordinates + boundary_coordinates_x
            full_y_coordinates = z_coordinates + boundary_coordinates_y
            full_data = pressure_coefficients + data_from_interpolator

            temp_triang = mtri.Triangulation(full_x_coordinates, full_y_coordinates)

            temp_refiner = mtri.UniformTriRefiner(temp_triang)

            temp_grid, temp_value = temp_refiner.refine_field(full_data, subdiv=4)

            temp_xticks = [j for j in range(min(full_x_coordinates), max(full_x_coordinates) + 1)]
            temp_yticks = [j for j in range(min(full_y_coordinates), max(full_y_coordinates) + 1)]

            temp_plot = fig.add_subplot(grid[position_plot_on_grid[face][0], position_plot_on_grid[face][1]],
                                        xticklabels=x_labels[face], xticks=temp_xticks,
                                        yticklabels=y_labels[face], yticks=temp_yticks)

            temp_plot.tricontourf(temp_grid, temp_value, cmap=cmap, extend='both', levels=levels_for_data)
            temp_contour = temp_plot.tricontour(temp_grid, temp_value, linewidths=1, linestyles='solid', colors='black',
                                                levels=levels_for_data)

            temp_plot.clabel(temp_contour, fontsize=10)

            temp_plot.plot(x_coordinates, z_coordinates, '.k', **dict(markersize=3))
            # temp_plot.axis('off')
            # temp_plot.scatter(boundary_coordinates_x,boundary_coordinates_y)

        set_x_coordinates = sorted(list(set(_x_coordinates[5])))
        set_y_coordinates_5 = sorted(list(set(_z_coordinates[5])))
        set_y_coordinates_6 = sorted(list(set(_z_coordinates[6])))

        union_plot = fig.add_subplot(grid[position_plot_on_grid[5][0], position_plot_on_grid[5][1]],
                                     xticklabels=x_labels[5],
                                     xticks=[i for i in range(min(set_x_coordinates) - 1, max(set_x_coordinates) + 2)],
                                     yticklabels=y_labels[5], yticks=[i for i in range(min(set_y_coordinates_5) - 1,
                                                                                       max(set_y_coordinates_6) + 2)])

        for face in (5, 6):
            x_coordinates = _x_coordinates[face]
            z_coordinates = _z_coordinates[face]
            pressure_coefficients = _pressure_coefficients[face]

            set_x_coordinates = sorted(list(set(x_coordinates)))
            set_y_coordinates = sorted(list(set(z_coordinates)))

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

            coords = [[i1, j1] for i1, j1 in zip(x_coordinates, z_coordinates)]  # Старые координаты
            # Интерполятор полученный на основе имеющихся данных
            extrapolator = intp(coords, pressure_coefficients)

            data_from_interpolator = [float(extrapolator([[X, Y]])) for X, Y in
                                      zip(boundary_coordinates_x, boundary_coordinates_y)]

            full_x_coordinates = x_coordinates + boundary_coordinates_x
            full_y_coordinates = z_coordinates + boundary_coordinates_y
            full_data = pressure_coefficients + data_from_interpolator

            temp_triang = mtri.Triangulation(full_x_coordinates, full_y_coordinates)

            temp_refiner = mtri.UniformTriRefiner(temp_triang)

            temp_grid, temp_value = temp_refiner.refine_field(full_data, subdiv=4)

            union_plot.tricontourf(temp_grid, temp_value, cmap=cmap, extend='both', levels=levels_for_data)
            temp_contour = union_plot.tricontour(temp_grid, temp_value, linewidths=1, linestyles='solid',
                                                 colors='black',
                                                 levels=levels_for_data)

            union_plot.clabel(temp_contour, fontsize=10)

            union_plot.plot(x_coordinates, z_coordinates, '.k', **dict(markersize=3))

        return fig

    @staticmethod
    def isofields_coefficients_gable_roof_non_isolated(
            coordinates,
            _pressure_coefficients,
            size_tpu,
            roof_angle
    ):
        _x_coordinates, _z_coordinates = coordinates
        breadth_tpu, depth_tpu, height_tpu = size_tpu

        count_faces = 5
        count_levels = 11

        width_plot = 2 * height_tpu + depth_tpu
        height_plot = 2 * height_tpu + breadth_tpu

        roof_length = breadth_tpu / (2 * np.cos(np.deg2rad(roof_angle)))
        height_in_triangle = np.sin(np.deg2rad(roof_angle)) * roof_length

        position_plot_on_grid = {
            1: [1, 1],
            2: [1, 0],
            3: [2, 1],
            4: [1, 2],
            5: [0, 1],
        }

        range_height = [i for i in range(height_tpu + 1)]
        range_breadth = [i for i in range(breadth_tpu + 1)]
        range_depth = [i for i in range(depth_tpu + 1)]

        full_height_side_faces = (height_in_triangle + height_tpu).round(2)
        range_full_height_side_faces = [i for i in range(int(full_height_side_faces) + 1)]
        if not float(range_full_height_side_faces[-1]) == full_height_side_faces:
            range_full_height_side_faces.append(full_height_side_faces)

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

        breadth_ratios = breadth_tpu / width_plot
        depth_ratios = depth_tpu / width_plot
        height_ratios = height_tpu / width_plot

        fig = plt.figure(figsize=(12, 12), dpi=80)
        grid = plt.GridSpec(3, 3, hspace=0.2, wspace=0.2,
                            width_ratios=[height_ratios, depth_ratios, height_ratios],
                            height_ratios=[height_ratios, height_ratios, height_ratios])

        cmap = matplotlib.colormaps.get_cmap("jet")

        levels_for_data = np.linspace(
            np.min([np.min(_pressure_coefficients[i]) for i in range(1, count_faces + 1)]),
            np.max([np.max(_pressure_coefficients[i]) for i in range(1, count_faces + 1)]),
            count_levels)

        left_right_face = (2, 4)

        for face in range(2, 6):
            x_coordinates = _x_coordinates[face]
            z_coordinates = _z_coordinates[face]
            pressure_coefficients = _pressure_coefficients[face]

            set_x_coordinates = sorted(list(set(x_coordinates)))
            set_y_coordinates = sorted(list(set(z_coordinates)))

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

                    for x_coord, y_coord in zip(x_coordinates, z_coordinates):
                        if x == x_coord:
                            local_max_y = y_coord if y_coord > local_max_y else local_max_y
                            local_min_y = y_coord if y_coord < local_min_y else local_min_y

                    boundary_coordinates_x.extend([x, x])
                    boundary_coordinates_y.extend([local_min_y - 1, local_max_y + 1])

                for y in set_y_coordinates:
                    local_max_x = float('-inf')
                    local_min_x = float('+inf')

                    for x_coord, y_coord in zip(x_coordinates, z_coordinates):
                        if y == y_coord:
                            local_max_x = x_coord if x_coord > local_max_x else local_max_x
                            local_min_x = x_coord if x_coord < local_min_x else local_min_x

                    boundary_coordinates_x.extend([local_min_x - 1, local_max_x + 1])
                    boundary_coordinates_y.extend([y, y])

                for x, y in zip(x_coordinates, z_coordinates):
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

            coords = [[i1, j1] for i1, j1 in zip(x_coordinates, z_coordinates)]  # Старые координаты
            # Интерполятор полученный на основе имеющихся данных
            interpolator = intp(coords, pressure_coefficients)

            data_from_interpolator = [float(interpolator([[X, Y]])) for X, Y in
                                      zip(boundary_coordinates_x, boundary_coordinates_y)]

            full_x_coordinates = x_coordinates + boundary_coordinates_x
            full_y_coordinates = z_coordinates + boundary_coordinates_y
            full_data = pressure_coefficients + data_from_interpolator

            #     full_x_coordinates = x_coordinates
            #     full_y_coordinates = y_coordinates
            #     full_data = pressure_coefficients

            temp_triang = mtri.Triangulation(full_x_coordinates, full_y_coordinates)

            temp_refiner = mtri.UniformTriRefiner(temp_triang)

            temp_grid, temp_value = temp_refiner.refine_field(full_data, subdiv=4)

            temp_yticks = [j for j in range(new_min_y_coordinate, new_max_y_coordinate + 1)]
            # temp_xticks = [j for j in range(new_min_x_coordinate, new_max_x_coordinate + 1)]
            temp_xticks = np.linspace(new_min_x_coordinate, new_max_x_coordinate, len(x_labels[face]))

            temp_plot = fig.add_subplot(grid[position_plot_on_grid[face][0], position_plot_on_grid[face][1]],
                                        xticklabels=x_labels[face], xticks=temp_xticks,
                                        yticklabels=y_labels[face], yticks=temp_yticks)

            temp_plot.tricontourf(temp_grid, temp_value, cmap=cmap, extend='both', levels=levels_for_data)
            temp_contour = temp_plot.tricontour(temp_grid, temp_value, linewidths=1, linestyles='solid', colors='black',
                                                levels=levels_for_data)

            temp_plot.clabel(temp_contour, fontsize=10)

            temp_plot.plot(x_coordinates, z_coordinates, '.k', **dict(markersize=3))
            # temp_plot.axis('off')
            # temp_plot.scatter(boundary_coordinates_x,boundary_coordinates_y)

        union_face = 1
        set_x_coordinates = sorted(list(set(_x_coordinates[union_face])))
        set_y_coordinates = sorted(list(set(_z_coordinates[union_face])))

        union_plot = fig.add_subplot(grid[position_plot_on_grid[union_face][0], position_plot_on_grid[union_face][1]],
                                     xticklabels=x_labels[union_face],
                                     xticks=[i for i in range(min(set_x_coordinates) - 1, max(set_x_coordinates) + 2)],
                                     yticklabels=y_labels[union_face],
                                     yticks=[i for i in range(min(set_y_coordinates) - 1, max(set_y_coordinates) + 2)])

        for face in (1,):
            x_coordinates = _x_coordinates[face]
            z_coordinates = _z_coordinates[face]
            pressure_coefficients = _pressure_coefficients[face]

            set_x_coordinates = sorted(list(set(x_coordinates)))
            set_y_coordinates = sorted(list(set(z_coordinates)))

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

            coords = [[i1, j1] for i1, j1 in zip(x_coordinates, z_coordinates)]  # Старые координаты
            # Интерполятор полученный на основе имеющихся данных
            extrapolator = intp(coords, pressure_coefficients)

            data_from_interpolator = [float(extrapolator([[X, Y]])) for X, Y in
                                      zip(boundary_coordinates_x, boundary_coordinates_y)]

            full_x_coordinates = x_coordinates + boundary_coordinates_x
            full_y_coordinates = z_coordinates + boundary_coordinates_y
            full_data = pressure_coefficients + data_from_interpolator

            temp_triang = mtri.Triangulation(full_x_coordinates, full_y_coordinates)

            temp_refiner = mtri.UniformTriRefiner(temp_triang)

            temp_grid, temp_value = temp_refiner.refine_field(full_data, subdiv=4)

            union_plot.tricontourf(temp_grid, temp_value, cmap=cmap, extend='both', levels=levels_for_data)
            temp_contour = union_plot.tricontour(temp_grid, temp_value, linewidths=1, linestyles='solid',
                                                 colors='black',
                                                 levels=levels_for_data)

            union_plot.clabel(temp_contour, fontsize=10)

            union_plot.plot(x_coordinates, z_coordinates, '.k', **dict(markersize=3))

        return fig

    @staticmethod
    def isofields_coefficients_hip_roof(
            coordinates,
            _pressure_coefficients,
            size_tpu,
            roof_angle
    ):

        _x_coordinates, _z_coordinates = coordinates
        breadth_tpu, depth_tpu, height_tpu = size_tpu

        count_faces = 8
        count_levels = 11

        width_plot = 2 * height_tpu + depth_tpu
        height_plot = 2 * height_tpu + breadth_tpu

        roof_length = breadth_tpu / (2 * np.cos(np.deg2rad(roof_angle)))
        height_in_triangle = np.sin(np.deg2rad(roof_angle)) * roof_length

        position_plot_on_grid = {
            1: [1, 0],
            2: [2, 1],
            3: [1, 2],
            4: [0, 1],
            5: [1, 1],
        }

        range_height = [i for i in range(height_tpu + 1)]
        range_breadth = [i for i in range(breadth_tpu + 1)]
        range_depth = [i for i in range(depth_tpu + 1)]

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

        breadth_ratios = breadth_tpu / width_plot
        depth_ratios = depth_tpu / width_plot
        height_ratios = height_tpu / width_plot

        fig = plt.figure(figsize=(12, 12), dpi=80)
        grid = plt.GridSpec(3, 3, hspace=0.2, wspace=0.2,
                            width_ratios=[height_ratios, depth_ratios, height_ratios],
                            height_ratios=[height_ratios, height_ratios, height_ratios])

        cmap = matplotlib.colormaps.get_cmap("jet")

        levels_for_data = np.linspace(
            np.min([np.min(_pressure_coefficients[i]) for i in range(1, count_faces + 1)]),
            np.max([np.max(_pressure_coefficients[i]) for i in range(1, count_faces + 1)]),
            count_levels)

        left_right_face = (1, 3)
        top_bottom_face = (2, 4)

        for face in range(1, 5):
            x_coordinates = _x_coordinates[face]
            z_coordinates = _z_coordinates[face]
            pressure_coefficients = _pressure_coefficients[face]

            set_x_coordinates = sorted(list(set(x_coordinates)))
            set_y_coordinates = sorted(list(set(z_coordinates)))

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

                    for x_coord, y_coord in zip(x_coordinates, z_coordinates):
                        if x == x_coord:
                            local_max_y = y_coord if y_coord > local_max_y else local_max_y
                            local_min_y = y_coord if y_coord < local_min_y else local_min_y

                    boundary_coordinates_x.extend([x, x])
                    boundary_coordinates_y.extend([local_min_y - 1, local_max_y + 1])

                for y in set_y_coordinates:
                    local_max_x = float('-inf')
                    local_min_x = float('+inf')

                    for x_coord, y_coord in zip(x_coordinates, z_coordinates):
                        if y == y_coord:
                            local_max_x = x_coord if x_coord > local_max_x else local_max_x
                            local_min_x = x_coord if x_coord < local_min_x else local_min_x

                    boundary_coordinates_x.extend([local_min_x - 1, local_max_x + 1])
                    boundary_coordinates_y.extend([y, y])

                for x, y in zip(x_coordinates, z_coordinates):
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

            coords = [[i1, j1] for i1, j1 in zip(x_coordinates, z_coordinates)]  # Старые координаты
            # Интерполятор полученный на основе имеющихся данных
            interpolator = intp(coords, pressure_coefficients)

            data_from_interpolator = [float(interpolator([[X, Y]])) for X, Y in
                                      zip(boundary_coordinates_x, boundary_coordinates_y)]

            full_x_coordinates = x_coordinates + boundary_coordinates_x
            full_y_coordinates = z_coordinates + boundary_coordinates_y
            full_data = pressure_coefficients + data_from_interpolator

            temp_triang = mtri.Triangulation(full_x_coordinates, full_y_coordinates)

            temp_refiner = mtri.UniformTriRefiner(temp_triang)

            temp_grid, temp_value = temp_refiner.refine_field(full_data, subdiv=4)

            temp_xticks = [j for j in range(min(full_x_coordinates), max(full_x_coordinates) + 1)]
            temp_yticks = [j for j in range(min(full_y_coordinates), max(full_y_coordinates) + 1)]

            temp_plot = fig.add_subplot(grid[position_plot_on_grid[face][0], position_plot_on_grid[face][1]],
                                        xticklabels=x_labels[face], xticks=temp_xticks,
                                        yticklabels=y_labels[face], yticks=temp_yticks)

            temp_plot.tricontourf(temp_grid, temp_value, cmap=cmap, extend='both', levels=levels_for_data)
            temp_contour = temp_plot.tricontour(temp_grid, temp_value, linewidths=1, linestyles='solid', colors='black',
                                                levels=levels_for_data)

            temp_plot.clabel(temp_contour, fontsize=10)

            temp_plot.plot(x_coordinates, z_coordinates, '.k', **dict(markersize=3))
            # temp_plot.axis('off')
            # temp_plot.scatter(boundary_coordinates_x,boundary_coordinates_y)

        set_x_coordinates_6 = sorted(list(set(_x_coordinates[6])))
        set_y_coordinates_5 = sorted(list(set(_z_coordinates[5])))

        union_xticks = [i for i in range(min(set_x_coordinates_6) - 2, max(set_x_coordinates_6) + 3)]
        union_yticks = [i for i in range(min(set_y_coordinates_5) - 2, max(set_y_coordinates_5) + 3)]

        union_plot = fig.add_subplot(grid[position_plot_on_grid[5][0], position_plot_on_grid[5][1]],
                                     xticklabels=x_labels[5], xticks=union_xticks,
                                     yticklabels=y_labels[5], yticks=union_yticks)

        for face in (5, 6, 7, 8):
            boundary_coordinates_x = []
            boundary_coordinates_y = []

            x_coordinates = _x_coordinates[face]
            z_coordinates = _z_coordinates[face]
            pressure_coefficients = _pressure_coefficients[face]

            set_x_coordinates = sorted(list(set(x_coordinates)))
            set_y_coordinates = sorted(list(set(z_coordinates)))

            min_x_coordinate = min(set_x_coordinates)
            max_x_coordinate = max(set_x_coordinates)

            new_min_x_coordinate = min_x_coordinate - 1
            new_max_x_coordinate = max_x_coordinate + 1

            min_y_coordinate = min(set_y_coordinates)
            max_y_coordinate = max(set_y_coordinates)

            new_min_y_coordinate = min_y_coordinate - 1
            new_max_y_coordinate = max_y_coordinate + 1

            for x in set_x_coordinates:
                local_max_y = float('-inf')
                local_min_y = float('+inf')

                for x_coord, y_coord in zip(x_coordinates, z_coordinates):
                    if x == x_coord:
                        local_max_y = y_coord if y_coord > local_max_y else local_max_y
                        local_min_y = y_coord if y_coord < local_min_y else local_min_y

                boundary_coordinates_x.extend([x, x])
                boundary_coordinates_y.extend([local_min_y - 1, local_max_y + 1])

            for y in set_y_coordinates:
                local_max_x = float('-inf')
                local_min_x = float('+inf')

                for x_coord, y_coord in zip(x_coordinates, z_coordinates):
                    if y == y_coord:
                        local_max_x = x_coord if x_coord > local_max_x else local_max_x
                        local_min_x = x_coord if x_coord < local_min_x else local_min_x

                boundary_coordinates_x.extend([local_min_x - 1, local_max_x + 1])
                boundary_coordinates_y.extend([y, y])

            for x, y in zip(x_coordinates, z_coordinates):
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

            if face == 5:
                boundary_coordinates_x.extend([new_min_x_coordinate, new_min_x_coordinate])
                boundary_coordinates_y.extend([new_min_y_coordinate - 1, new_max_y_coordinate + 1])

                boundary_coordinates_x.extend([new_min_x_coordinate + 1, new_min_x_coordinate + 1])
                boundary_coordinates_y.extend([new_min_y_coordinate - 1, new_max_y_coordinate + 1])

            elif face == 7:
                boundary_coordinates_x.extend([new_max_x_coordinate, new_max_x_coordinate])
                boundary_coordinates_y.extend([new_min_y_coordinate - 1, new_max_y_coordinate + 1])

                boundary_coordinates_x.extend([new_max_x_coordinate - 1, new_max_x_coordinate - 1])
                boundary_coordinates_y.extend([new_min_y_coordinate - 1, new_max_y_coordinate + 1])

            coords = [[i1, j1] for i1, j1 in zip(x_coordinates, z_coordinates)]  # Старые координаты
            # Интерполятор полученный на основе имеющихся данных
            extrapolator = intp(coords, pressure_coefficients)

            data_from_extrapolator = [float(extrapolator([[X, Y]])) for X, Y in
                                      zip(boundary_coordinates_x, boundary_coordinates_y)]

            full_x_coordinates = x_coordinates + boundary_coordinates_x
            full_y_coordinates = z_coordinates + boundary_coordinates_y
            full_data = pressure_coefficients + data_from_extrapolator

            temp_triang = mtri.Triangulation(full_x_coordinates, full_y_coordinates)

            temp_refiner = mtri.UniformTriRefiner(temp_triang)

            temp_grid, temp_value = temp_refiner.refine_field(full_data, subdiv=4)

            union_plot.tricontourf(temp_grid, temp_value, cmap=cmap, extend='both', levels=levels_for_data)
            temp_contour = union_plot.tricontour(temp_grid, temp_value, linewidths=1, linestyles='solid',
                                                 colors='black',
                                                 levels=levels_for_data)

            union_plot.clabel(temp_contour, fontsize=10)

            union_plot.plot(x_coordinates, z_coordinates, '.k', **dict(markersize=3))

        return fig

    @staticmethod
    def isofields_coefficients_hip_roof_non_isolated(
            coordinates,
            _pressure_coefficients,
            size_tpu,
            roof_angle
    ):

        _x_coordinates, _z_coordinates = coordinates
        breadth_tpu, depth_tpu, height_tpu = size_tpu

        count_faces = 5
        count_levels = 11

        width_plot = 2 * height_tpu + depth_tpu
        height_plot = 2 * height_tpu + breadth_tpu

        roof_length = breadth_tpu / (2 * np.cos(np.deg2rad(roof_angle)))
        height_in_triangle = np.sin(np.deg2rad(roof_angle)) * roof_length

        range_height = [i for i in range(height_tpu + 1)]
        range_breadth = [i for i in range(breadth_tpu + 1)]
        range_depth = [i for i in range(depth_tpu + 1)]

        left_right_face = (2, 4)
        border_face = (2, 3, 4, 5)
        main_face = (1,)
        position_plot_on_grid = {
            1: [1, 1],
            2: [1, 0],
            3: [2, 1],
            4: [1, 2],
            5: [0, 1],
        }
        x_labels = {
            1: range_height,
            2: range_depth,
            3: range_height[::-1],
            4: range_depth,
            5: range_depth,
        }

        y_labels = {
            1: range_breadth,
            2: range_height,
            3: range_breadth,
            4: range_height,
        }

        breadth_ratios = breadth_tpu / width_plot
        depth_ratios = depth_tpu / width_plot
        height_ratios = height_tpu / width_plot

        fig = plt.figure(figsize=(12, 12), dpi=80)
        grid = plt.GridSpec(3, 3, hspace=0.2, wspace=0.2,
                            width_ratios=[height_ratios, depth_ratios, height_ratios],
                            height_ratios=[height_ratios, height_ratios, height_ratios])

        cmap = matplotlib.colormaps.get_cmap("jet")

        levels_for_data = np.linspace(np.min([np.min(_pressure_coefficients[i]) for i in range(1, count_faces + 1)]),
                                      np.max([np.max(_pressure_coefficients[i]) for i in range(1, count_faces + 1)]),
                                      count_levels)

        # left_right_face = (2, 4)
        # top_bottom_face = (5, 3)

        for face in border_face:
            x_coordinates = _x_coordinates[face]
            z_coordinates = _z_coordinates[face]
            pressure_coefficients = _pressure_coefficients[face]

            set_x_coordinates = sorted(list(set(x_coordinates)))
            set_y_coordinates = sorted(list(set(z_coordinates)))

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

                    for x_coord, y_coord in zip(x_coordinates, z_coordinates):
                        if x == x_coord:
                            local_max_y = y_coord if y_coord > local_max_y else local_max_y
                            local_min_y = y_coord if y_coord < local_min_y else local_min_y

                    boundary_coordinates_x.extend([x, x])
                    boundary_coordinates_y.extend([local_min_y - 1, local_max_y + 1])

                for y in set_y_coordinates:
                    local_max_x = float('-inf')
                    local_min_x = float('+inf')

                    for x_coord, y_coord in zip(x_coordinates, z_coordinates):
                        if y == y_coord:
                            local_max_x = x_coord if x_coord > local_max_x else local_max_x
                            local_min_x = x_coord if x_coord < local_min_x else local_min_x

                    boundary_coordinates_x.extend([local_min_x - 1, local_max_x + 1])
                    boundary_coordinates_y.extend([y, y])

                for x, y in zip(x_coordinates, z_coordinates):
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

            coords = [[i1, j1] for i1, j1 in zip(x_coordinates, z_coordinates)]  # Старые координаты
            # Интерполятор полученный на основе имеющихся данных
            interpolator = intp(coords, pressure_coefficients)

            data_from_interpolator = [float(interpolator([[X, Y]])) for X, Y in
                                      zip(boundary_coordinates_x, boundary_coordinates_y)]

            full_x_coordinates = x_coordinates + boundary_coordinates_x
            full_y_coordinates = z_coordinates + boundary_coordinates_y
            full_data = pressure_coefficients + data_from_interpolator

            temp_triang = mtri.Triangulation(full_x_coordinates, full_y_coordinates)

            temp_refiner = mtri.UniformTriRefiner(temp_triang)

            temp_grid, temp_value = temp_refiner.refine_field(full_data, subdiv=4)

            temp_xticks = [j for j in range(min(full_x_coordinates), max(full_x_coordinates) + 1)]
            temp_yticks = [j for j in range(min(full_y_coordinates), max(full_y_coordinates) + 1)]

            temp_plot = fig.add_subplot(grid[position_plot_on_grid[face][0], position_plot_on_grid[face][1]],
                                        xticklabels=x_labels[face], xticks=temp_xticks,
                                        yticklabels=y_labels[face], yticks=temp_yticks)

            temp_plot.tricontourf(temp_grid, temp_value, cmap=cmap, extend='both', levels=levels_for_data)
            temp_contour = temp_plot.tricontour(temp_grid, temp_value, linewidths=1, linestyles='solid', colors='black',
                                                levels=levels_for_data)

            temp_plot.clabel(temp_contour, fontsize=10)

            temp_plot.plot(x_coordinates, z_coordinates, '.k', **dict(markersize=3))
            # temp_plot.axis('off')
            # temp_plot.scatter(boundary_coordinates_x,boundary_coordinates_y)

        union_face = 1
        set_x_coordinates = sorted(list(set(_x_coordinates[union_face])))
        set_y_coordinates = sorted(list(set(_z_coordinates[union_face])))

        union_plot = fig.add_subplot(grid[position_plot_on_grid[union_face][0], position_plot_on_grid[union_face][1]],
                                     xticklabels=x_labels[union_face],
                                     xticks=[i for i in range(min(set_x_coordinates) - 1, max(set_x_coordinates) + 2)],
                                     yticklabels=y_labels[union_face],
                                     yticks=[i for i in range(min(set_y_coordinates) - 1, max(set_y_coordinates) + 2)])

        for face in (1,):
            x_coordinates = _x_coordinates[face]
            z_coordinates = _z_coordinates[face]
            pressure_coefficients = _pressure_coefficients[face]

            set_x_coordinates = sorted(list(set(x_coordinates)))
            set_y_coordinates = sorted(list(set(z_coordinates)))

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

            coords = [[i1, j1] for i1, j1 in zip(x_coordinates, z_coordinates)]  # Старые координаты
            # Интерполятор полученный на основе имеющихся данных
            extrapolator = intp(coords, pressure_coefficients)

            data_from_interpolator = [float(extrapolator([[X, Y]])) for X, Y in
                                      zip(boundary_coordinates_x, boundary_coordinates_y)]

            full_x_coordinates = x_coordinates + boundary_coordinates_x
            full_y_coordinates = z_coordinates + boundary_coordinates_y
            full_data = pressure_coefficients + data_from_interpolator

            temp_triang = mtri.Triangulation(full_x_coordinates, full_y_coordinates)

            temp_refiner = mtri.UniformTriRefiner(temp_triang)

            temp_grid, temp_value = temp_refiner.refine_field(full_data, subdiv=4)

            union_plot.tricontourf(temp_grid, temp_value, cmap=cmap, extend='both', levels=levels_for_data)
            temp_contour = union_plot.tricontour(temp_grid, temp_value, linewidths=1, linestyles='solid',
                                                 colors='black',
                                                 levels=levels_for_data)

            union_plot.clabel(temp_contour, fontsize=10)

            union_plot.plot(x_coordinates, z_coordinates, '.k', **dict(markersize=3))

        return fig
