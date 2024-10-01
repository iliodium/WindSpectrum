# import logging
# import time
#
# import toml
# from sqlalchemy import create_engine
#
# interference_id_building = {
#     140: 2,
#     196: 5,
#     280: 1,
#     420: 3,
#     560: 4,
# }
#
#
# class DataBaseToolkit:
#     """Класс для запросов в БД"""
#     config = toml.load('config.toml')
#
#     _min_count_connections = config['databasetoolkit']['min_count_connections']  # минимальное число соединений с БД
#     _max_count_connections = config['databasetoolkit']['max_count_connections']  # максимальное число соединений с БД
#     _time_sleep = config['databasetoolkit']['time_sleep']
#
#     def __init__(self,
#                  min_conn=None,
#                  max_conn=None):
#         engine = create_engine(
#             f"postgresql://{DB_USER}:{DB_PASSWORD}@{PATH_TO_DATABASE}/{DB_NAME}"
#         )
#         """Создание пула подключений к БД"""
#         self.__logger = logging.getLogger('submodules.DataBaseToolkit')
#         self.__logger.info("Подключение к БД")
#
#         if min_conn is None:
#             min_conn = DataBaseToolkit._min_count_connections
#
#         if max_conn is None:
#             max_conn = DataBaseToolkit._max_count_connections
#
#         # self.connection_pool = ThreadedConnectionPool(min_conn,
#         #                                               max_conn,
#         #                                               user=os.getenv("WINDSPECTRUM_USER"),
#         #                                               password=os.getenv("WINDSPECTRUM_PASSWORD"),
#         #                                               host=os.getenv("WINDSPECTRUM_HOST"),
#         #                                               port=os.getenv("WINDSPECTRUM_PORT"),
#         #                                               database=os.getenv("WINDSPECTRUM_DATABASE_NAME"))
#         self.connection_pool = ThreadedConnectionPool(
#             min_conn,
#             max_conn,
#             user='postgres',
#             password='password',
#             host='127.0.0.1',
#             port='15432',
#             database='postgres',
#             options="-c search_path=windspectrum"
#         )
#
#         self.__logger.info("Подключение к БД успешно создано")
#
#     def get_connection(self, *, _timeout: int = 0):
#         """Запрос свободного подключения к БД с таймаутом 1 секунда"""
#         try:
#             connection = self.connection_pool.getconn()
#         except PoolError:
#             if _timeout == 3:
#                 return None
#             self.__logger.info("Ожидание свободного подключения")
#             time.sleep(self._time_sleep)
#             connection = self.get_connection(_timeout=_timeout + 1)
#
#         return connection
#
#     def get_experiments(self):
#         """Возвращает всё доступные эксперименты из БД
#         manager для того чтобы буфер был доступен во всех процессах.
#         """
#
#         self.__logger.info("Запрос экспериментов из БД")
#
#         connection = self.connection_pool.getconn()
#         cursor = connection.cursor()
#
#         experiments = dict({'4': dict(),
#                             '6': dict()
#                             })
#         # Альфа 4
#         cursor.execute("""
#                             select model_name
#                             from experiments_alpha_4
#                                 """)
#         model_names_4 = cursor.fetchall()
#
#         for model_name in model_names_4:
#             experiments['4'][str(model_name[0])] = dict()
#
#         # Альфа 6
#         cursor.execute("""
#                             select model_name
#                             from experiments_alpha_6
#                                         """)
#         model_names_6 = cursor.fetchall()
#
#         for model_name in model_names_6:
#             experiments['6'][str(model_name[0])] = dict()
#
#         self.__logger.info("Запрос экспериментов из БД успешно выполнен")
#
#         cursor.close()
#         self.connection_pool.putconn(connection)
#
#         return experiments
#
#     def get_pressure_coefficients(self, db, **kwargs):
#         """Возвращает коэффициенты давления эксперимента из БД"""
#
#         connection = self.get_connection()
#         cursor = connection.cursor()
#         angle = kwargs['angle']
#
#         if db == 'isolated':
#             alpha = kwargs['alpha']
#             model_name = kwargs['model_name']
#
#             self.__logger.info(
#                 f"Запрос коэффициентов давления модель = {model_name} альфа = {alpha} угол = {angle.rjust(2, '0')} из БД")
#
#             if alpha == '4':
#                 cursor.execute("""
#                                    select model_id
#                                    from experiments_alpha_4
#                                    where model_name = (%s) limit 1
#                                """, (model_name,)
#                                )
#                 model_id = str(cursor.fetchall()[0][0])
#                 cursor.execute("""
#                                    select pressure_coefficients
#                                    from models_alpha_4
#                                    where model_id = (%s) and angle = (%s) limit 1
#                                """, (model_id, angle))
#
#             elif alpha == '6':
#                 cursor.execute("""
#                                    select model_id
#                                    from experiments_alpha_6
#                                    where model_name = (%s) limit 1
#                                """, (model_name,)
#                                )
#
#                 model_id = str(cursor.fetchall()[0][0])
#
#                 cursor.execute("""
#                                    select pressure_coefficients
#                                    from models_alpha_6
#                                    where model_id = (%s) and angle = (%s) limit 1
#                                """, (model_id, angle))
#
#             self.__logger.info(f"Запрос коэффициентов давления модель = {model_name} "
#                              f"альфа = {alpha} угол = {angle.rjust(2, '0')} из БД успешно выполнен"
#                                )
#
#             pressure_coefficients = [i[0] for i in cursor.fetchall()][0]
#         elif db == 'interference':
#             case = kwargs['case']
#             model_name = kwargs['model_name']
#
#             cursor.execute("""select pressure_coefficients
#                 from interference
#                 join mean_wind_speeds mws on mws.id_mean_wind_speed = interference.id_mean_wind_speed
#                 join sample_periods sp on interference.id_sample_period = sp.id_sample_period
#                 where instance = (%s) and id_interfering_building=(%s) and angle=(%s)""",
#                            (case, interference_id_building[model_name], angle))
#
#             pressure_coefficients = [i[0] for i in cursor.fetchall()][0]
#         elif db == 'without_eaves':
#             breadth = kwargs['breadth']
#             depth = kwargs['depth']
#             height = kwargs['height']
#             roof_type = kwargs['roof_type']
#             pitch = kwargs['pitch']
#
#             cursor.execute("""select pressure_coefficients
#             from building_without_eaves
#             join roof_types rt on building_without_eaves.id_roof = rt.id_roof
#             join roof_pitches rp on building_without_eaves.id_pitch = rp.id_pitch
#             where roof_type = (%s) and angle =  (%s) and breadth = (%s) and depth = (%s) and height = (%s) and pitch = (%s)::real""",
#                            (roof_type, angle, breadth, depth, height, pitch))
#             pressure_coefficients = [i[0] for i in cursor.fetchall()][0]
#         elif db == 'with_eaves':
#             breadth = kwargs['breadth']
#             depth = kwargs['depth']
#             height = kwargs['height']
#             eave = kwargs['eave']
#
#             cursor.execute("""select pressure_coefficients
#                             from building_with_eaves
#                             join eave_types et on et.id_eave = building_with_eaves.id_eave
#                             join wind_azimuths wa on wa.id_wind_azimuth = building_with_eaves.id_wind_azimuth
#                             where wind_azimuth =  (%s) and eave = (%s) and breadth = (%s) and depth = (%s) and height = (%s)""",
#                            (angle, eave, breadth, depth, height))
#             pressure_coefficients = [i[0] for i in cursor.fetchall()][0]
#         elif db == 'non_isolated':
#             breadth = kwargs['breadth']
#             depth = kwargs['depth']
#             height = kwargs['height']
#             roof_type = kwargs['roof_type']
#             pitch = kwargs['pitch']
#             arrange_order = kwargs['arrange_order']
#             area_density = kwargs['area_density']
#
#             cursor.execute("""select pressure_coefficients
#             from non_isolated_building
#             join roof_types rt on non_isolated_building.id_roof = rt.id_roof
#             join roof_pitches rp on non_isolated_building.id_pitch = rp.id_pitch
#             join arrange_orders ao on ao.id_arrange_order = non_isolated_building.id_arrange_order
#             join areas_density ad on ad.id_area_density = non_isolated_building.id_area_density
#             join wind_azimuths wa on wa.id_wind_azimuth = non_isolated_building.id_wind_azimuth
#             where roof_type = (%s) and wind_azimuth = (%s) and breadth = (%s) and depth = (%s) and height = (%s) and pitch = (%s) and arrange_order = (%s) and area_density = (%s)""",
#                            (roof_type, angle, breadth, depth, height, pitch, arrange_order, area_density))
#
#             pressure_coefficients = [i[0] for i in cursor.fetchall()][0]
#
#         cursor.close()
#         self.connection_pool.putconn(connection)
#
#         return pressure_coefficients
#
#     def get_coordinates(self, alpha: str, model_name: str, ):
#         """Возвращает координаты датчиков эксперимента из БД"""
#
#         self.__logger.info(f"Запрос координат датчиков модель = {model_name} альфа = {alpha} из БД")
#
#         connection = self.get_connection()
#         cursor = connection.cursor()
#
#         if alpha == '4':
#             cursor.execute("""
#                            select x_coordinates, z_coordinates
#                            from experiments_alpha_4
#                            where model_name = (%s) limit 1
#                        """, (model_name,))
#
#         elif alpha == '6':
#             cursor.execute("""
#                            select x_coordinates, z_coordinates
#                            from experiments_alpha_6
#                            where model_name = (%s) limit 1
#                        """, (model_name,))
#
#         self.__logger.info(f"Запрос координат датчиков модель = {model_name} альфа = {alpha} из БД успешно выполнен")
#
#         x, z = cursor.fetchall()[0]
#
#         cursor.close()
#         self.connection_pool.putconn(connection)
#
#         return x, z
#
#     def get_uh_average_wind_speed(self, alpha: str, model_name: str, ):
#         """Возвращает среднюю скорость ветра эксперимента из БД"""
#
#         self.__logger.info(f"Запрос средней скорости ветра модель = {model_name} альфа = {alpha} из БД")
#
#         connection = self.get_connection()
#         cursor = connection.cursor()
#
#         if alpha == '4':
#             cursor.execute("""
#                         select uh_averagewindspeed
#                         from experiments_alpha_4
#                         where model_name = (%s) limit 1
#                     """, (model_name,))
#
#         elif alpha == '6':
#             cursor.execute("""
#                         select uh_averagewindspeed
#                         from experiments_alpha_6
#                         where model_name = (%s) limit 1
#                     """, (model_name,))
#
#         self.__logger.info(f"Запрос средней скорости ветра модель = {model_name} альфа = {alpha} из БД успешно выполнен")
#
#         speed = cursor.fetchall()[0][0]
#
#         cursor.close()
#         self.connection_pool.putconn(connection)
#
#         return speed
#
#     def get_face_number(self, alpha: str, model_name: str, ):
#         """Возвращает нумерацию датчиков эксперимента из БД"""
#
#         self.__logger.info(f"Запрос нумерации датчиков модель = {model_name} альфа = {alpha} из БД")
#
#         connection = self.get_connection()
#         cursor = connection.cursor()
#
#         if alpha == '4':
#             cursor.execute("""
#                         select face_number
#                         from experiments_alpha_4
#                         where model_name = (%s) limit 1
#                     """, (model_name,))
#
#         elif alpha == '6':
#             cursor.execute("""
#                         select face_number
#                         from experiments_alpha_6
#                         where model_name = (%s) limit 1
#                     """, (model_name,))
#
#         self.__logger.info(f"Запрос нумерации датчиков модель = {model_name} альфа = {alpha} из БД успешно выполнен")
#
#         face_number = cursor.fetchall()[0][0]
#
#         cursor.close()
#         self.connection_pool.putconn(connection)
#
#         return face_number
#
#     def get_x_y_surface(self, db, **kwargs):
#         connection = self.get_connection()
#         cursor = connection.cursor()
#         angle = kwargs['angle']
#
#         if db == 'without_eaves':
#             breadth = kwargs['breadth']
#             depth = kwargs['depth']
#             height = kwargs['height']
#             roof_type = kwargs['roof_type']
#             pitch = kwargs['pitch']
#
#             cursor.execute("""select id_x_coordinates,id_y_coordinates, id_surface
#             from building_without_eaves
#             join roof_types rt on building_without_eaves.id_roof = rt.id_roof
#             join roof_pitches rp on building_without_eaves.id_pitch = rp.id_pitch
#             where roof_type = (%s) and angle = (%s) and breadth = (%s) and depth = (%s) and height = (%s) and pitch = (%s)::real""",
#                            (roof_type, angle, breadth, depth, height, pitch))
#             id_x, id_y, id_surface = cursor.fetchall()[0]
#         elif db == 'with_eaves':
#             breadth = kwargs['breadth']
#             depth = kwargs['depth']
#             height = kwargs['height']
#             eave = kwargs['eave']
#
#             cursor.execute("""select id_x_coordinates,id_y_coordinates, id_surface
#             from building_with_eaves
#             where id_wind_azimuth =  (%s) and id_eave = (%s) and breadth = (%s) and depth = (%s) and height = (%s)""",
#                            (angle, eave, breadth, depth, height))
#
#             id_x, id_y, id_surface = cursor.fetchall()[0]
#
#         elif db == 'non_isolated':
#             breadth = kwargs['breadth']
#             depth = kwargs['depth']
#             height = kwargs['height']
#             roof_type = kwargs['roof_type']
#             pitch = kwargs['pitch']
#             arrange_order = kwargs['arrange_order']
#             area_density = kwargs['area_density']
#
#             cursor.execute("""select id_x_coordinates,id_y_coordinates, id_surface
#             from non_isolated_building
#             join roof_types rt on non_isolated_building.id_roof = rt.id_roof
#             join roof_pitches rp on non_isolated_building.id_pitch = rp.id_pitch
#             where roof_type = (%s) and id_wind_azimuth =  (%s) and breadth = (%s) and depth = (%s) and height = (%s) and pitch = (%s)::real and id_arrange_order = (%s) and id_area_density = (%s)""",
#                            (roof_type, angle, breadth, depth, height, pitch, arrange_order, area_density))
#
#             id_x, id_y, id_surface = cursor.fetchall()[0]
#
#         cursor.execute("""select x_coordinates
#         from x_coordinates_roof
#         where id_x_coordinates = (%s)""", (id_x,))
#         x_coordinates = cursor.fetchall()[0][0]
#         cursor.execute("""select y_coordinates
#         from y_coordinates_roof
#         where id_y_coordinates = (%s)""", (id_y,))
#         y_coordinates = cursor.fetchall()[0][0]
#         cursor.execute("""select surface
#         from surfaces_roof
#         where id_surface = (%s)""", (id_surface,))
#         surface = cursor.fetchall()[0][0]
#
#         cursor.close()
#         self.connection_pool.putconn(connection)
#
#         return x_coordinates, y_coordinates, surface
#
#
# if __name__ == '__main__':
#     d = DataBaseToolkit()
