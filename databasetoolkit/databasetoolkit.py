from psycopg2.pool import ThreadedConnectionPool
from concurrent.futures import ThreadPoolExecutor


class DataBaseToolkit:

    def __init__(self):
        self.connection_pool = ThreadedConnectionPool(15, 15,
                                                      user='postgres',
                                                      password='08101430',
                                                      host='26.148.227.16',
                                                      port='5432',
                                                      database='tpu')

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

        connection = self.connection_pool.getconn()
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
        pressure_coefficients = [i[0] for i in cursor.fetchall()][0]

        cursor.close()
        self.connection_pool.putconn(connection)

        return pressure_coefficients

    def get_coordinates(self, alpha: str, model_name: str, ):
        """Возвращает координаты датчиков эксперимента из БД"""

        connection = self.connection_pool.getconn()
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
        x, z = cursor.fetchall()[0]

        cursor.close()
        self.connection_pool.putconn(connection)

        return x, z

    def get_uh_average_wind_speed(self, alpha: str, model_name: str, ):

        connection = self.connection_pool.getconn()
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
        wind_speed = cursor.fetchall()[0][0]

        cursor.close()
        self.connection_pool.putconn(connection)

        return wind_speed

    def get_face_number(self, alpha: str, model_name: str, ):

        connection = self.connection_pool.getconn()
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
        face_number = cursor.fetchall()[0][0]

        cursor.close()
        self.connection_pool.putconn(connection)

        return face_number


if __name__ == '__main__':
    import time

    args_pres = [('6', '115', str(i)) for i in range(0, 50, 5)]
    # list_pres = []
    t1 = time.time()
    with ThreadPoolExecutor(max_workers=20) as executor:
        list_pres = executor.map(lambda i: DataBaseToolkit.get_pressure_coefficients(*i), args_pres)
    # for i in args_pres:
    #     list_pres.append(DataBaseToolkit.get_pressure_coefficients(*i))
    print(time.time() - t1)
    # time.sleep(10)
    # print(list(list_pres))
    # d = DataBaseToolkit()
    # print(d.get_experiments())
    # print(d.get_uh_average_wind_speed('4', '111')) 118 сек
