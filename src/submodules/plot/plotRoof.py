from pydantic import validate_call

from src.submodules.plot.plot import Plot


class PlotRoof(Plot):
    @staticmethod
    @validate_call
    def isofields_coefficients_gable_roof(
            model_size: ModelSizeType,
            model_name: ModelNameIsolatedType,
            parameter: ChartMode,
            pressure_coefficients,
            coordinates: CoordinatesType
    ):
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
    @validate_call
    def isofields_coefficients_flat_roof(**kwargs):
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