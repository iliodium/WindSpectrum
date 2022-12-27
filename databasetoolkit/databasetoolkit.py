import psycopg2


class DataBaseToolkit:

    def __init__(self):
        self.connection = psycopg2.connect(user='postgres',
                                           password='08101430',
                                           host='127.0.0.1',
                                           port='5432',
                                           database='tpu')
        #2325070307
        # self.connection = psycopg2.connect(user='postgres',
        #                                    password='08101430',
        #                                    host='26.148.227.16',
        #                                    port='5432',
        #                                    database='tpu')
        self.cursor = self.connection.cursor()
        print("Соединение с PostgreSQL открыто")

    def get_experiments(self):
        """Возвращает всё доступные эксперименты из БД"""
        experiments = {'4': dict(),
                       '6': dict()
                       }
        self.cursor.execute("""
                            select model_name
                            from experiments_alpha_4
                            """)
        experiments['4'] = {str(i[0]): dict() for i in self.cursor.fetchall()}
        for model_name in experiments['4'].keys():
            self.cursor.execute("""
                                select angle
                                from models_alpha_4
                                where
                                model_id = (
                                    select model_id
                                from experiments_alpha_4
                                where model_name = (%s))
                                """, (model_name,))
            experiments['4'][model_name] = {str(i[0]): dict() for i in self.cursor.fetchall()}

        self.cursor.execute("""
                            select model_name
                            from experiments_alpha_6
                            """)
        experiments['6'] = {str(i[0]): dict() for i in self.cursor.fetchall()}
        for model_name in experiments['6'].keys():
            self.cursor.execute("""
                                select angle
                                from models_alpha_6
                                where
                                model_id = (
                                    select model_id
                                from experiments_alpha_6
                                where model_name = (%s))
                                """, (model_name,))
            experiments['6'][model_name] = {str(i[0]): dict() for i in self.cursor.fetchall()}
        self.connection.commit()

        return experiments

    def get_pressure_coefficients(self, alpha, model_name, angle):
        """Возвращает коэффициенты давления эксперимента из БД"""
        if alpha == '4':
            self.cursor.execute("""
                                select pressure_coefficients
                                from models_alpha_4
                                where model_id = (
                                select model_id
                                from experiments_alpha_4
                                where model_name = (%s)
                                ) and angle = (%s)
                            """, (model_name, angle))

        elif alpha == '6':
            self.cursor.execute("""
                                select pressure_coefficients
                                from models_alpha_6
                                where model_id = (
                                select model_id
                                from experiments_alpha_6
                                where model_name = (%s)
                                ) and angle = (%s)
                            """, (model_name, angle))
        self.connection.commit()
        pressure_coefficients = [i[0] for i in self.cursor.fetchall()][0]
        return pressure_coefficients

    def get_coordinates(self, alpha, model_name):
        """Возвращает координаты датчиков эксперимента из БД"""
        if alpha == '4':
            self.cursor.execute("""
                           select x_coordinates, z_coordinates
                           from experiments_alpha_4
                           where model_name = (%s)
                       """, (model_name,))

        elif alpha == '6':
            self.cursor.execute("""
                           select x_coordinates, z_coordinates
                           from experiments_alpha_6
                           where model_name = (%s)
                       """, (model_name,))
        self.connection.commit()
        x, z = self.cursor.fetchall()[0]
        return x, z

    def get_uh_average_wind_speed(self, alpha, model_name):
        if alpha == '4':
            self.cursor.execute("""
                        select uh_averagewindspeed
                        from experiments_alpha_4
                        where model_name = (%s)
                    """, (model_name,))

        elif alpha == '6':
            self.cursor.execute("""
                        select uh_averagewindspeed
                        from experiments_alpha_6
                        where model_name = (%s)
                    """, (model_name,))
        self.connection.commit()

        return self.cursor.fetchall()[0][0]

    def get_face_number(self, alpha, model_name):
        if alpha == '4':
            self.cursor.execute("""
                        select face_number
                        from experiments_alpha_4
                        where model_name = (%s)
                    """, (model_name,))

        elif alpha == '6':
            self.cursor.execute("""
                        select face_number
                        from experiments_alpha_6
                        where model_name = (%s)
                    """, (model_name,))
        self.connection.commit()

        return self.cursor.fetchall()[0][0]


if __name__ == '__main__':
    d = DataBaseToolkit()
    # print(d.get_experiments())
    print(d.get_uh_average_wind_speed('4', '111'))
