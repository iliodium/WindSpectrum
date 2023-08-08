import os
import time

import toml
from psycopg2.pool import ThreadedConnectionPool, PoolError

from utils import get_logger

interference_id_building = {
    140: 2,
    196: 5,
    280: 1,
    420: 3,
    560: 4,
}


class DataBaseToolkit:
    """Класс для запросов в БД"""
    config = toml.load('config.toml')

    _min_count_connections = config['databasetoolkit']['min_count_connections']  # минимальное число соединений с БД
    _max_count_connections = config['databasetoolkit']['max_count_connections']  # максимальное число соединений с БД
    _time_sleep = config['databasetoolkit']['time_sleep']

    def __init__(self,
                 min_conn=None,
                 max_conn=None):
        """Создание пула подключений к БД"""
        self.logger = get_logger('DataBaseToolkit')
        self.logger.info("Подключение к БД")
        if min_conn is None:
            min_conn = DataBaseToolkit._min_count_connections

        if max_conn is None:
            max_conn = DataBaseToolkit._max_count_connections

        # self.connection_pool = ThreadedConnectionPool(min_conn,
        #                                               max_conn,
        #                                               user=os.getenv("WINDSPECTRUM_USER"),
        #                                               password=os.getenv("WINDSPECTRUM_PASSWORD"),
        #                                               host=os.getenv("WINDSPECTRUM_HOST"),
        #                                               port=os.getenv("WINDSPECTRUM_PORT"),
        #                                               database=os.getenv("WINDSPECTRUM_DATABASE_NAME"))
        self.connection_pool = ThreadedConnectionPool(min_conn,
                                                      max_conn,
                                                      user='postgres',
                                                      password='1234',
                                                      host='127.0.0.1',
                                                      port='5432',
                                                      database='postgres')

        self.logger.info("Подключение к БД успешно создано")

    def get_connection(self):
        """Запрос свободного подключения к БД с таймаутом 1 секунда"""
        try:
            connection = self.connection_pool.getconn()
        except PoolError:
            self.logger.info("Ожидание свободного подключения")
            time.sleep(self._time_sleep)
            connection = self.get_connection()

        return connection

    def get_experiments(self):
        """Возвращает всё доступные эксперименты из БД
        manager для того чтобы буфер был доступен во всех процессах.
        """

        self.logger.info("Запрос экспериментов из БД")

        connection = self.connection_pool.getconn()
        cursor = connection.cursor()

        experiments = dict({'4': dict(),
                            '6': dict()
                            })
        # Альфа 4
        cursor.execute("""
                            select model_name
                            from experiments_alpha_4
                                """)
        model_names_4 = cursor.fetchall()

        for model_name in model_names_4:
            experiments['4'][str(model_name[0])] = dict()

        # Альфа 6
        cursor.execute("""
                            select model_name
                            from experiments_alpha_6
                                        """)
        model_names_6 = cursor.fetchall()

        for model_name in model_names_6:
            experiments['6'][str(model_name[0])] = dict()

        self.logger.info("Запрос экспериментов из БД успешно выполнен")

        cursor.close()
        self.connection_pool.putconn(connection)

        return experiments

    def get_pressure_coefficients(self, db, **kwargs):
        """Возвращает коэффициенты давления эксперимента из БД"""

        connection = self.get_connection()
        cursor = connection.cursor()

        if db == 'isolated':
            alpha = kwargs['alpha']
            model_name = kwargs['model_name']
            angle = kwargs['angle']

            self.logger.info(
                f"Запрос коэффициентов давления модель = {model_name} альфа = {alpha} угол = {angle.rjust(2, '0')} из БД")

            if alpha == '4':
                cursor.execute("""
                                   select model_id
                                   from experiments_alpha_4
                                   where model_name = (%s) limit 1
                               """, (model_name,)
                               )
                model_id = str(cursor.fetchall()[0][0])
                cursor.execute("""
                                   select pressure_coefficients
                                   from models_alpha_4
                                   where model_id = (%s) and angle = (%s) limit 1
                               """, (model_id, angle))

            elif alpha == '6':
                cursor.execute("""
                                   select model_id
                                   from experiments_alpha_6
                                   where model_name = (%s) limit 1
                               """, (model_name,)
                               )

                model_id = str(cursor.fetchall()[0][0])

                cursor.execute("""
                                   select pressure_coefficients
                                   from models_alpha_6
                                   where model_id = (%s) and angle = (%s) limit 1
                               """, (model_id, angle))

            self.logger.info(f"Запрос коэффициентов давления модель = {model_name} "
                             f"альфа = {alpha} угол = {angle.rjust(2, '0')} из БД успешно выполнен"
                             )

            pressure_coefficients = [i[0] for i in cursor.fetchall()][0]
        elif db == 'interference':
            case = kwargs['case']
            model_name = kwargs['model_name']
            angle = kwargs['angle']

            cursor.execute("""select pressure_coefficients
                from interference
                join mean_wind_speeds mws on mws.id_mean_wind_speed = interference.id_mean_wind_speed
                join sample_periods sp on interference.id_sample_period = sp.id_sample_period
                where instance = (%s) and id_interfering_building=(%s) and angle=(%s)""",
                           (case, interference_id_building[model_name], angle))

            pressure_coefficients = [i[0] for i in cursor.fetchall()][0]

        cursor.close()
        self.connection_pool.putconn(connection)

        return pressure_coefficients

    def get_coordinates(self, alpha: str, model_name: str, ):
        """Возвращает координаты датчиков эксперимента из БД"""

        self.logger.info(f"Запрос координат датчиков модель = {model_name} альфа = {alpha} из БД")

        connection = self.get_connection()
        cursor = connection.cursor()

        if alpha == '4':
            cursor.execute("""
                           select x_coordinates, z_coordinates
                           from experiments_alpha_4
                           where model_name = (%s) limit 1
                       """, (model_name,))

        elif alpha == '6':
            cursor.execute("""
                           select x_coordinates, z_coordinates
                           from experiments_alpha_6
                           where model_name = (%s) limit 1
                       """, (model_name,))

        self.logger.info(f"Запрос координат датчиков модель = {model_name} альфа = {alpha} из БД успешно выполнен")

        x, z = cursor.fetchall()[0]

        cursor.close()
        self.connection_pool.putconn(connection)

        return x, z

    def get_uh_average_wind_speed(self, alpha: str, model_name: str, ):
        """Возвращает среднюю скорость ветра эксперимента из БД"""

        self.logger.info(f"Запрос средней скорости ветра модель = {model_name} альфа = {alpha} из БД")

        connection = self.get_connection()
        cursor = connection.cursor()

        if alpha == '4':
            cursor.execute("""
                        select uh_averagewindspeed
                        from experiments_alpha_4
                        where model_name = (%s) limit 1
                    """, (model_name,))

        elif alpha == '6':
            cursor.execute("""
                        select uh_averagewindspeed
                        from experiments_alpha_6
                        where model_name = (%s) limit 1
                    """, (model_name,))

        self.logger.info(f"Запрос средней скорости ветра модель = {model_name} альфа = {alpha} из БД успешно выполнен")

        speed = cursor.fetchall()[0][0]

        cursor.close()
        self.connection_pool.putconn(connection)

        return speed

    def get_face_number(self, alpha: str, model_name: str, ):
        """Возвращает нумерацию датчиков эксперимента из БД"""

        self.logger.info(f"Запрос нумерации датчиков модель = {model_name} альфа = {alpha} из БД")

        connection = self.get_connection()
        cursor = connection.cursor()

        if alpha == '4':
            cursor.execute("""
                        select face_number
                        from experiments_alpha_4
                        where model_name = (%s) limit 1
                    """, (model_name,))

        elif alpha == '6':
            cursor.execute("""
                        select face_number
                        from experiments_alpha_6
                        where model_name = (%s) limit 1
                    """, (model_name,))

        self.logger.info(f"Запрос нумерации датчиков модель = {model_name} альфа = {alpha} из БД успешно выполнен")

        face_number = cursor.fetchall()[0][0]

        cursor.close()
        self.connection_pool.putconn(connection)

        return face_number


if __name__ == '__main__':
    d = DataBaseToolkit()
    # d.get_experiments()
    # for model_name, angel in a:
    #     print(model_name, angel)
    # d1 = d.get_connection()
    # d1 = d.get_connection()
    # d1 = d.get_connection()
    # d1 = d.get_connection()
    # d1 = d.get_connection()
    # d1 = d.get_connection1(1, '1')
    # d.get_pressure_coefficients('6', '111', '0', d.connection_pool)
    # print(type(d.connection_pool))
    # import time
    #
    # args_pres = [('6', '115', str(i)) for i in range(0, 50, 5)]
    # # list_pres = []
    # t1 = time.time()
    # with ThreadPoolExecutor(max_workers=20) as executor:
    #     list_pres = executor.map(lambda i: DataBaseToolkit.get_pressure_coefficients(*i), args_pres)
    # # for i in args_pres:
    # #     list_pres.append(DataBaseToolkit.get_pressure_coefficients(*i))
    # print(time.time() - t1)
    # # time.sleep(10)
    # # print(list(list_pres))
    # # d = DataBaseToolkit()
    # # print(d.get_experiments())
    # # print(d.get_uh_average_wind_speed('4', '111')) 118 сек
