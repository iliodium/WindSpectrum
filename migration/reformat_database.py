"""

PROOF OF CONCEPT THAT CURRENT DATABASE STRUCTURE IS OPTIMAL

"""

import sys
import uuid

import numpy as np
from sqlalchemy import create_engine, Row, Connection
from sqlalchemy import text

PATH_TO_DATABASE = "localhost:15432"
DB_NAME = "postgres"
DB_USER = "postgres"
DB_PASSWORD = "password"
# For scale factor = 1
SENSORS_COUNT_PER_1 = 5
DATA_MULTIPLIER = 1000

TIME_INDEXES_PER_INSERT = {
    1: 30,
    2: 20,
    3: 15,
    4: 12,
    5: 10,
    6: 15,
    7: 12,
    8: 10,
    9: 8,
    10: 15,
    11: 12,
    12: 10,
    13: 8
}


def database_setup_script(model_name):
    return f"""
create table if not exists possible_x_coordinate_{model_name}
(
    x_coordinate_id bigint generated always as identity (minvalue 0)
        constraint fk_{str(uuid.uuid4()).replace("-", "")}
            primary key,
    experiment_id   bigint not null
        constraint fk_{str(uuid.uuid4()).replace("-", "")}
            references experiments
            on delete cascade,
    x_coordinate    NUMERIC(4, 2)    not null,
    constraint fk_{str(uuid.uuid4()).replace("-", "")}
        unique (experiment_id, x_coordinate)
);

alter table possible_x_coordinate_{model_name}
    owner to postgres;


create table if not exists possible_y_coordinate_{model_name}
(
    y_coordinate_id bigint generated always as identity (minvalue 0)
        constraint fk_{str(uuid.uuid4()).replace("-", "")}
            primary key,
    experiment_id   bigint not null
        constraint fk_{str(uuid.uuid4()).replace("-", "")}
            references experiments
            on delete cascade,
    y_coordinate    NUMERIC(4, 2)    not null,
    constraint fk_{str(uuid.uuid4()).replace("-", "")}
        unique (experiment_id, y_coordinate)
);

alter table possible_y_coordinate_{model_name}
    owner to postgres;


create table if not exists sensor_values_{model_name}
(
    experiment_id            bigint   not null
        constraint sensor_values_{model_name}_experiments_experiment_id_fk
            references experiments
            on delete cascade,
    sensor_value_id          bigint generated always as identity (minvalue 0)
        constraint sensor_values_{model_name}_pk
            primary key,
    x_coordinate_for_face_id bigint   not null
        constraint fk_{str(uuid.uuid4()).replace("-", "")}
            references possible_x_coordinate_{model_name},
    y_coordinate_for_face_id bigint   not null
        constraint fk_{str(uuid.uuid4()).replace("-", "")}
            references possible_y_coordinate_{model_name},
    face_number              NUMERIC(1, 0) not null,
    sensor_value             NUMERIC(4, 3)     not null,
    time_indx                NUMERIC(6, 0)   not null,
    angle                    NUMERIC(3, 0)  not null,
    constraint sensor_values_{model_name}_pk2
        unique (x_coordinate_for_face_id, y_coordinate_for_face_id, experiment_id, time_indx, face_number, angle)
);

alter table sensor_values_{model_name}
    owner to postgres;
"""


class TaskIterator(object):
    def __init__(
            self, model_name, x_coordinates, x_coordinates_from_db_mapped,
            z_coordinates, y_coordinates_from_db_mapped, time_indx,
            face_number, pressure_coefficients, experiment_id, angle,
            *,
            start_from: int = 0, per_insert: int = None
    ):
        self.__model_name = model_name
        self.__start_from = start_from
        self.__per_insert = per_insert
        self.__idx = start_from - per_insert
        self.__x_coordinates = x_coordinates
        self.__x_coordinates_from_db_mapped = x_coordinates_from_db_mapped
        self.__z_coordinates = z_coordinates
        self.__y_coordinates_from_db_mapped = y_coordinates_from_db_mapped
        self.__time_indx = time_indx
        self.__face_number = face_number
        self.__pressure_coefficients = pressure_coefficients
        self.__experiment_id = experiment_id
        self.__angle = angle

    def __iter__(self):
        return self

    def __next__(self):
        self.__idx += self.__per_insert
        if self.__idx >= self.__time_indx:
            raise StopIteration
        try:
            return self.__model_name, self.__x_coordinates, self.__x_coordinates_from_db_mapped, \
                self.__z_coordinates, self.__y_coordinates_from_db_mapped, \
                self.__idx, self.__per_insert, self.__time_indx, self.__face_number, self.__pressure_coefficients, \
                self.__experiment_id, self.__angle
        except IndexError:
            self.idx = self.__start_from
            raise StopIteration  # Done iterating.


class Tasks(object):
    def __init__(
            self, model_name, angle, experiment_id, time_indx_count,
            mapped_x_coord, x_coordinates,
            mapped_y_coord, z_coordinates,
            face_numbers, pressure_coefficients,
            start_from, per_insert
    ):
        self.__model_name = model_name
        self.__start_from = start_from
        self.__per_insert = per_insert
        self.__angle = angle
        self.__experiment_id = experiment_id
        self.__time_indx_count = time_indx_count
        self.__mapped_x_coord = mapped_x_coord
        self.__x_coordinates = x_coordinates
        self.__mapped_y_coord = mapped_y_coord
        self.__z_coordinates = z_coordinates
        self.__face_numbers = face_numbers
        self.__pressure_coefficients = pressure_coefficients

    def __iter__(self):
        return TaskIterator(
            self.__model_name,
            self.__x_coordinates,
            self.__mapped_x_coord,
            self.__z_coordinates,
            self.__mapped_y_coord,
            self.__time_indx_count,
            self.__face_numbers,
            self.__pressure_coefficients,
            self.__experiment_id,
            self.__angle,
            start_from=self.__start_from,
            per_insert=self.__per_insert
        )


def process_tasks(
        args
):
    model_name, x_coordinates, x_coordinates_from_db_mapped, \
        z_coordinates, y_coordinates_from_db_mapped, from_time_indx, per_insert, total_times, \
        face_number, pressure_coefficients, experiment_id, angle = args
    _engine = create_engine(
        f"postgresql://{DB_USER}:{DB_PASSWORD}@{PATH_TO_DATABASE}/{DB_NAME}"
    )
    lp = pressure_coefficients[from_time_indx:min(from_time_indx + per_insert, total_times)]
    PRESSURE_COEFFICIENTS_TO_INSERT = list[str]()
    for time_indx_r in range(lp.shape[0]):
        time_indx = time_indx_r + from_time_indx
        for i in range(lp.shape[1]):
            x_coord = x_coordinates[i]
            z_coord = z_coordinates[i]
            face_number_v = face_number[i]
            value = pressure_coefficients[time_indx][i]
            PRESSURE_COEFFICIENTS_TO_INSERT.append(
                f"""
                ({experiment_id}, {x_coordinates_from_db_mapped[x_coord]}, {y_coordinates_from_db_mapped[z_coord]}, {face_number_v}, {value}, {time_indx}, {angle})
                """
            )

    with _engine.connect() as _connection:
        _connection.execute(
            text(
                f"INSERT INTO sensor_values_{model_name}(experiment_id, x_coordinate_for_face_id, y_coordinate_for_face_id, face_number, sensor_value, time_indx, angle) VALUES {','.join(PRESSURE_COEFFICIENTS_TO_INSERT)}")
        )

        _connection.commit()

    _engine.dispose()


def process_experiment(experiment_info: Row, alpha: int, connection: Connection) -> None:
    assert alpha == 4 or alpha == 6, "Alpha must be either 4 or 6"

    model_name = str(experiment_info.model_name)
    x_coordinates = np.array(experiment_info.x_coordinates)
    z_coordinates = np.array(experiment_info.z_coordinates)
    face_number = np.array(experiment_info.face_number)

    check = connection.execute(
        text(
            "SELECT exp_.experiment_id FROM experiments exp_ WHERE exp_.model_name = :model_name"
        ),
        {"model_name": model_name}
    )

    connection.execute(
        text(
            database_setup_script(model_name)
        )
    )

    if check.rowcount != 0:
        experiment_id = check.first().experiment_id
    else:
        experiment_id = connection.execute(
            text(
                """
        INSERT INTO experiments(
                        model_name,
                        breadth,
                        depth,
                        height,
                        sample_frequency,
                        sample_period,
                        uh_averagewindspeed,
                        alpha
        ) VALUES (
            :model_name,
            :breadth,
            :depth,
            :height,
            :sample_frequency,
            :sample_period,
            :uh_averagewindspeed,
            :alpha
        ) RETURNING experiment_id"""
            ),
            {
                "model_name": model_name,
                "breadth": experiment_info.breadth,
                "depth": experiment_info.depth,
                "height": experiment_info.height,
                "sample_frequency": experiment_info.sample_frequency,
                "sample_period": experiment_info.sample_period,
                "uh_averagewindspeed": experiment_info.uh_averagewindspeed,
                "alpha": alpha
            }
        ).first().experiment_id

    print(
        f"Got experiment '{model_name}' [id = {experiment_info.model_id}] with"
        f" model({experiment_info.breadth}, {experiment_info.depth}, {experiment_info.height})"
    )
    print(f"Sample freq = {experiment_info.sample_frequency}, period = {experiment_info.sample_period}")
    print(f"Average uh wind speed = {experiment_info.uh_averagewindspeed}")
    experiment_results = connection.execute(
        text(
            """
            SELECT
                mod_.angle,
                mod_.pressure_coefficients
            FROM models_alpha_4 mod_
            WHERE mod_.model_id = :model_id
            """
        ),
        {
            "model_id": experiment_info.model_id
        }
    )

    x_coordinates_to_insert = list()

    existing_x_coordinates = list(map(lambda el: float(el.x_coordinate), connection.execute(
        text(
            f"""
            SELECT
                pos_x_.x_coordinate
            FROM possible_x_coordinate_{model_name} pos_x_
            WHERE pos_x_.experiment_id = :experiment_id"""
        ),
        {
            "experiment_id": experiment_id
        }
    )))

    print("Existing x coordinates: ", existing_x_coordinates)

    for x_coordinate in sorted(set(x_coordinates)):
        if existing_x_coordinates.count(x_coordinate) == 0:
            x_coordinates_to_insert.append(
                f"({experiment_id}, {x_coordinate})"
            )

    if len(x_coordinates_to_insert) > 0:
        connection.execute(
            text(
                f"INSERT INTO possible_x_coordinate_{model_name}(experiment_id, x_coordinate) VALUES {','.join(x_coordinates_to_insert)}")
        )

    z_coordinates_to_insert = list()

    existing_z_coordinates = list(map(lambda el: float(el.y_coordinate), connection.execute(
        text(
            f"""
            SELECT
                pos_y_.y_coordinate
            FROM possible_y_coordinate_{model_name} pos_y_
            WHERE pos_y_.experiment_id = :experiment_id"""
        ),
        {
            "experiment_id": experiment_id
        }
    )))

    for z_coordinate in sorted(set(z_coordinates)):
        if existing_z_coordinates.count(z_coordinate) == 0:
            z_coordinates_to_insert.append(
                f"({experiment_id}, {z_coordinate})"
            )

    if len(z_coordinates_to_insert) > 0:
        connection.execute(
            text(
                f"INSERT INTO possible_y_coordinate_{model_name}(experiment_id, y_coordinate) VALUES {','.join(z_coordinates_to_insert)}")
        )

    x_coordinates_from_db = connection.execute(
        text(
            f"""
            SELECT
                pos_x_.x_coordinate_id,
                pos_x_.x_coordinate
            FROM possible_x_coordinate_{model_name} pos_x_
            WHERE pos_x_.experiment_id = :experiment_id"""
        ),
        {
            "experiment_id": experiment_id
        }
    )

    x_coordinates_from_db_mapped = {float(i.x_coordinate): i.x_coordinate_id for i in x_coordinates_from_db}

    y_coordinates_from_db = connection.execute(
        text(
            f"""
            SELECT
                pos_y_.y_coordinate_id,
                pos_y_.y_coordinate
            FROM possible_y_coordinate_{model_name} pos_y_
            WHERE pos_y_.experiment_id = :experiment_id"""
        ),
        {
            "experiment_id": experiment_id
        }
    )

    y_coordinates_from_db_mapped = {float(i.y_coordinate): i.y_coordinate_id for i in y_coordinates_from_db}

    check = {i.angle: int(i.MAXIMUM_UPLOADED) for i in connection.execute(
        text(
            f"""
            SELECT
                sv_.angle,
                MAX(sv_.time_indx) AS "MAXIMUM_UPLOADED"
            FROM sensor_values_{model_name} sv_
            WHERE sv_.experiment_id = :experiment_id
            GROUP BY sv_.angle
            """
        ),
        {
            "experiment_id": experiment_id
        }
    )}

    print("Possible x values:", x_coordinates_from_db_mapped)
    print("Possible y values:", y_coordinates_from_db_mapped)

    for row in experiment_results:
        angle = row.angle
        pressure_coefficients: np.ndarray = np.array(row.pressure_coefficients) / DATA_MULTIPLIER

        start_from = 0

        # Если не всё время загружено для угла
        if angle in check.keys():
            if check[angle] + 1 < pressure_coefficients.shape[0]:
                # Начинаем с того, что не загрузили
                start_from = check[angle] + 1
            else:
                continue

        current_shape = pressure_coefficients.shape

        print(f"Got angle = {angle}")
        print(f"Got data with shape = {current_shape} starting from {start_from}")

        connection.commit()

        list(
            map(
                process_tasks,
                Tasks(
                    model_name,
                    angle,
                    experiment_id,
                    current_shape[0],
                    x_coordinates_from_db_mapped,
                    x_coordinates,
                    y_coordinates_from_db_mapped,
                    z_coordinates,
                    face_number,
                    pressure_coefficients,
                    start_from,
                    TIME_INDEXES_PER_INSERT[experiment_info.model_id]
                )
            )
        )


if __name__ == "__main__":

    alpha = sys.argv[1]
    model_id = sys.argv[2]


    engine = create_engine(
        f"postgresql://{DB_USER}:{DB_PASSWORD}@{PATH_TO_DATABASE}/{DB_NAME}"
    )

    with engine.connect() as connection:
        if alpha == '4':
            result = connection.execute(
                text(
                    """
            select
                exp_.model_id,
                exp_.breadth,
                exp_.depth,
                exp_.height,
                exp_.sample_frequency,
                exp_.sample_period,
                exp_.uh_averagewindspeed,
                exp_.model_name,
                exp_.x_coordinates,
                exp_.z_coordinates,
                exp_.face_number
            from experiments_alpha_4 exp_
            WHERE exp_.model_id = :model_id
            """
                ),
                {
                    "model_id": model_id
                }
            )
            for row in result:
                process_experiment(row, 4, connection)

        if alpha == '6':
            result = connection.execute(
                text(
                    """
                select
                    exp_.model_id,
                    exp_.breadth,
                    exp_.depth,
                    exp_.height,
                    exp_.sample_frequency,
                    exp_.sample_period,
                    exp_.uh_averagewindspeed,
                    exp_.model_name
                from experiments_alpha_6 exp_
                WHERE exp_.model_id = :model_id
                """
                ),
                {
                    "model_id": model_id
                }
            )
            for row in result:
                process_experiment(row, 6, connection)
