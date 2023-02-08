import time

#from multiprocessing import Manager
from psycopg2.pool import ThreadedConnectionPool, PoolError
from concurrent.futures import ThreadPoolExecutor


class DataBaseToolkit:
    _min_count_connections = 15  # минимальное число соединений с БД
    _max_count_connections = 15  # максимальное число соединений с БД

    def __init__(self):
        """Создание пула подключений к БД"""
        #self.manager = Manager()
        #self.manager = None
        self.connection_pool = ThreadedConnectionPool(DataBaseToolkit._min_count_connections,
                                                      DataBaseToolkit._max_count_connections,
                                                      user='postgres',
                                                      password='2325070307',
                                                      host='127.0.0.1',
                                                      port='5432',
                                                      database='tpu')

        # user = 'postgres',
        #        password = '08101430',
        #                   host = '26.148.227.16',
        #                          port = '5432',
        #                                 database = 'tpu')

    def get_connection(self):
        """Запрос свободного подключения к БД с таймаутом 1 секунда"""
        try:
            connection = self.connection_pool.getconn()
        except PoolError:
            print("Ожидание свободного подключения")
            time.sleep(1)
            connection = self.get_connection()

        return connection

    def get_experiments(self):
        """Возвращает всё доступные эксперименты из БД"""

        connection = self.connection_pool.getconn()
        cursor = connection.cursor()

        experiments = {'4': dict(),
                       '6': dict()
                       }
        cursor.execute("""
                            select model_name
                            from experiments_alpha_4
                            """)
        experiments['4'] = {str(i[0]): dict() for i in cursor.fetchall()}
        for model_name in experiments['4'].keys():
            cursor.execute("""
                                select angle
                                from models_alpha_4
                                where
                                model_id = (
                                    select model_id
                                from experiments_alpha_4
                                where model_name = (%s))
                                """, (model_name,))
            experiments['4'][model_name] = {str(i[0]): dict() for i in cursor.fetchall()}

        cursor.execute("""
                            select model_name
                            from experiments_alpha_6
                            """)
        experiments['6'] = {str(i[0]): dict() for i in cursor.fetchall()}
        for model_name in experiments['6'].keys():
            cursor.execute("""
                                select angle
                                from models_alpha_6
                                where
                                model_id = (
                                    select model_id
                                from experiments_alpha_6
                                where model_name = (%s))
                                """, (model_name,))
            experiments['6'][model_name] = {str(i[0]): dict() for i in cursor.fetchall()}

        cursor.close()
        self.connection_pool.putconn(connection)

        return experiments

    def get_pressure_coefficients(self, alpha: str, model_name: str, angle: str, ):
        """Возвращает коэффициенты давления эксперимента из БД"""

        print(f"Запрос коэффициентов давления модель = {model_name} альфа = {alpha} угол = {angle} из БД")

        connection = self.get_connection()
        cursor = connection.cursor()

        if alpha == '4':
            cursor.execute("""
                               select model_id
                               from experiments_alpha_4
                               where model_name = (%s)
                           """, (model_name,)
                           )
            model_id = str(cursor.fetchall()[0][0])
            cursor.execute("""
                               select pressure_coefficients
                               from models_alpha_4
                               where model_id = (%s) and angle = (%s)
                           """, (model_id, angle))

        elif alpha == '6':
            cursor.execute("""
                               select model_id
                               from experiments_alpha_6
                               where model_name = (%s)
                           """, (model_name,)
                           )

            model_id = str(cursor.fetchall()[0][0])

            cursor.execute("""
                               select pressure_coefficients
                               from models_alpha_6
                               where model_id = (%s) and angle = (%s)
                           """, (model_id, angle))

        print(
            f"Запрос коэффициентов давления модель = {model_name} альфа = {alpha} угол = {angle} из БД успешно выполнен"
        )

        pressure_coefficients = [i[0] for i in cursor.fetchall()][0]

        cursor.close()
        self.connection_pool.putconn(connection)

        return pressure_coefficients

    def get_coordinates(self, alpha: str, model_name: str, ):
        """Возвращает координаты датчиков эксперимента из БД"""

        print(f"Запрос координат датчиков модель = {model_name} альфа = {alpha} из БД")

        connection = self.get_connection()
        cursor = connection.cursor()

        if alpha == '4':
            cursor.execute("""
                           select x_coordinates, z_coordinates
                           from experiments_alpha_4
                           where model_name = (%s)
                       """, (model_name,))

        elif alpha == '6':
            cursor.execute("""
                           select x_coordinates, z_coordinates
                           from experiments_alpha_6
                           where model_name = (%s)
                       """, (model_name,))

        print(f"Запрос координат датчиков модель = {model_name} альфа = {alpha} из БД успешно выполнен")

        x, z = cursor.fetchall()[0]

        cursor.close()
        self.connection_pool.putconn(connection)

        return x, z

    def get_uh_average_wind_speed(self, alpha: str, model_name: str, ):
        """Возвращает среднюю скорость ветра эксперимента из БД"""

        print(f"Запрос средней скорости ветра модель = {model_name} альфа = {alpha} из БД")

        connection = self.get_connection()
        cursor = connection.cursor()

        if alpha == '4':
            cursor.execute("""
                        select uh_averagewindspeed
                        from experiments_alpha_4
                        where model_name = (%s)
                    """, (model_name,))

        elif alpha == '6':
            cursor.execute("""
                        select uh_averagewindspeed
                        from experiments_alpha_6
                        where model_name = (%s)
                    """, (model_name,))

        print(f"Запрос средней скорости ветра модель = {model_name} альфа = {alpha} из БД успешно выполнен")

        speed = cursor.fetchall()[0][0]

        cursor.close()
        self.connection_pool.putconn(connection)

        return speed

    def get_face_number(self, alpha: str, model_name: str, ):
        """Возвращает нумерацию датчиков эксперимента из БД"""

        print(f"Запрос нумерации датчиков модель = {model_name} альфа = {alpha} из БД")

        connection = self.get_connection()
        cursor = connection.cursor()

        if alpha == '4':
            cursor.execute("""
                        select face_number
                        from experiments_alpha_4
                        where model_name = (%s)
                    """, (model_name,))

        elif alpha == '6':
            cursor.execute("""
                        select face_number
                        from experiments_alpha_6
                        where model_name = (%s)
                    """, (model_name,))

        print(f"Запрос нумерации датчиков модель = {model_name} альфа = {alpha} из БД успешно выполнен")

        face_number = cursor.fetchall()[0][0]

        cursor.close()
        self.connection_pool.putconn(connection)

        return face_number


if __name__ == '__main__':
    d = DataBaseToolkit()
    d1 = d.get_connection1(1, '1')
    print(d1)
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
