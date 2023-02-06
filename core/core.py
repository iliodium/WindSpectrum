import os
import time
import glob
import matplotlib.pyplot as plt
import numpy as np
from typing import Tuple
from docx import Document
from multiprocessing import Process
from docx.shared import Inches, Pt, Mm
from docx.enum.text import WD_ALIGN_PARAGRAPH
from concurrent.futures import ThreadPoolExecutor

import utils.utils
# local imports
from plot.plot import Plot
from utils.utils import *

# local imports
from clipboard.clipboard import Clipboard
from plot.plot import Plot


class Core:
    _count_threads = 1  # количество запускаемых потоков при работе

    def __init__(self):
        """Создание объекта для работы с буфером"""
        self.clipboard_obj = Clipboard()

    def get_plot_isofields(self,
                           alpha: str,
                           model_size: Tuple[str, str, str],
                           angle: str,
                           mode: str,
                           type_plot: str):
        """Функция возвращает изополя.
        Если изополя отсутствуют в буфере, запускается отрисовка.
        """
        print(f'Запрос {type_plot} альфа = {alpha} '
              f'размер = {" ".join(model_size)} угол = {angle} режим = {mode} из буфера')
        fig = None
        model_scale, scale_factors = get_model_and_scale_factors(*model_size, alpha)

        id_fig = f'{type_plot}_{mode}_{"_".join(model_size)}'
        print(123)
        if not self.clipboard_obj.clipboard_dict[alpha][model_scale][angle].get(id_fig):
            print(f'Отрисовка {type_plot} альфа = {alpha} '
                  f'размер = {" ".join(model_size)} угол = {angle} режим = {mode}')
            pressure_coefficients = self.clipboard_obj.get_pressure_coefficients(alpha, model_scale, angle)
            coordinates = self.clipboard_obj.get_coordinates(alpha, model_scale)
            print(321)
            if type_plot == 'discrete_isofields':
                fig = Plot.discrete_isofield(model_scale, mode, angle, alpha, pressure_coefficients, coordinates)
            elif type_plot == 'integral_isofields':
                fig = Plot.integral_isofield(model_scale,
                                             model_size,
                                             scale_factors,
                                             alpha,
                                             mode,
                                             angle,
                                             pressure_coefficients,
                                             coordinates,
                                             )
            print(222)
            self.clipboard_obj.clipboard_dict[alpha][model_scale][angle][id_fig] = fig
        print(333)
        fig = self.clipboard_obj.clipboard_dict[alpha][model_scale][angle][id_fig]
        print(444)
        print(f'Запрос {type_plot} альфа = {alpha} '
              f'размер = {" ".join(model_size)} угол = {angle} режим = {mode} из буфера успешно выполнен')

        return fig

    def get_plot_summary_spectres(self,
                                  alpha: str,
                                  model_size: Tuple[str, str, str],
                                  angle: str,
                                  mode: str,
                                  scale: str,
                                  type_plot: str):
        """Функция возвращает графики суммарных спектров.
        Если графики суммарных спектров отсутствуют в буфере, запускается отрисовка.
        """
        message = f'{" ".join(list(model_size))} {alpha} {int(angle):02} {mode}'
        print(f'Запрос суммарных спектров {message} из буфера')

        mode = mode.replace(' ', '_')
        model_scale, scale_factors = get_model_and_scale_factors(*model_size, alpha)
        id_fig = f'{type_plot}_{mode}_{scale}_{"_".join(model_size)}'
        data = dict()

        if not self.clipboard_obj.clipboard_dict[alpha][model_scale][angle].get(id_fig):
            print(f'Отрисовка суммарных спектров {message}')

            if 'Cx' in mode or 'Cy' in mode:
                cx, cy = self.clipboard_obj.get_cx_cy(alpha, model_scale, angle)
                if 'Cx' in mode:
                    data['Cx'] = cx
                if 'Cy' in mode:
                    data['Cy'] = cy

            if 'CMz' in mode:
                data['CMz'] = self.clipboard_obj.get_cmz(alpha, model_scale, angle, model_size)

            fig = Plot.welch_graphs(data, model_size, alpha, angle)
            self.clipboard_obj.clipboard_dict[alpha][model_scale][angle][id_fig] = fig
        fig = self.clipboard_obj.clipboard_dict[alpha][model_scale][angle][id_fig]

        print(f'Запрос суммарных спектров {message} из буфера успешно выполнен')

        return fig

    def get_plot_summary_coefficients(self,
                                      alpha: str,
                                      model_size: Tuple[str, str, str],
                                      angle: str,
                                      mode: str, ):
        """Функция возвращает графики суммарных коэффициентов.
        Если графики суммарных коэффициентов отсутствуют в буфере, запускается отрисовка.
        """
        print(f'Запрос суммарных коэффициентов размеры = {" ".join(list(model_size))} '
              f'альфа = {alpha} угол = {int(angle):02}  режим = {mode} из буфера')

        model_scale, scale_factors = get_model_and_scale_factors(*model_size, alpha)
        type_plot = 'summary_coefficients'
        id_fig = f'{type_plot}_{mode}_{"_".join(model_size)}'
        data = dict()

        if not self.clipboard_obj.clipboard_dict[alpha][model_scale][angle].get(id_fig):
            print(f'Отрисовка суммарных коэффициентов {" ".join(list(model_size))} {alpha} {int(angle):02} {mode}')
            if 'Cx' in mode or 'Cy' in mode:
                cx, cy = self.clipboard_obj.get_cx_cy(alpha, model_scale, angle)
                if 'Cx' in mode:
                    data['Cx'] = cx
                if 'Cy' in mode:
                    data['Cy'] = cy

            if 'CMz' in mode:
                data['CMz'] = self.clipboard_obj.get_cmz(alpha, model_scale, angle, model_size)

            fig = Plot.summary_coefficients(data, model_scale, alpha, angle)
            self.clipboard_obj.clipboard_dict[alpha][model_scale][angle][id_fig] = fig

        fig = self.clipboard_obj.clipboard_dict[alpha][model_scale][angle][id_fig]

        print(f'Запрос суммарных коэффициентов размеры = {" ".join(list(model_size))} '
              f'альфа = {alpha} угол = {int(angle):02}  режим = {mode} из буфера успешно выполнен')

        return fig

    def get_plot_summary_coefficients_polar(self,
                                            alpha: str,
                                            model_scale: str,
                                            model_size: Tuple[str, str, str],
                                            angle_border: int,
                                            mode: str):
        """Функция возвращает графики суммарных коэффициентов в полярной системе координат.
        Если графики суммарных коэффициентов в полярной системе координат отсутствуют в буфере, запускается отрисовка.
        """
        print(f'Запрос суммарных коэффициентов в полярной системе координат размеры = {" ".join(list(model_size))} '
              f'альфа = {alpha} режим = {mode} из буфера')

        mods = {
            'MEAN': np.mean,
            'RMS': rms,
            'STD': np.std,
            'MAX': np.max,
            'MIN': np.min,
            'РАСЧЕТНОЕ': rach,
            'ОБЕСП+': obes_m,
            'ОБЕСП-': obes_p
        }

        plot_name = f'summary_coefficients_Cx_Cy_CMz_polar_{"_".join(model_size)}_{mode}'
        if not self.clipboard_obj.clipboard_dict[alpha][model_scale]['model_attributes'].get(plot_name):
            args_cmz = [(alpha, model_scale, str(angle), model_size) for angle in range(0, angle_border, 5)]
            with ThreadPoolExecutor(max_workers=Core._count_threads) as executor:
                list_cmz = list(executor.map(lambda i: self.clipboard_obj.get_cmz(*i), args_cmz))

            args_cx_cy = [(alpha, model_scale, str(angle)) for angle in range(0, angle_border, 5)]
            with ThreadPoolExecutor(max_workers=Core._count_threads) as executor:
                cx_cy = np.array(list(executor.map(lambda i: self.clipboard_obj.get_cx_cy(*i), args_cx_cy)))

            list_cx = cx_cy[:, 0]
            list_cy = cx_cy[:, 1]

            list_norm_cmz = list(map(mods[mode], list_cmz))
            list_norm_cx = list(map(mods[mode], list_cx))
            list_norm_cy = list(map(mods[mode], list_cy))

            x_scale, y_scale = Plot.scaling_data(list_norm_cx, list_norm_cy)
            cmz_scale = Plot.scaling_data(list_norm_cmz)

            data = {
                'Cx': x_scale,
                'Cy': y_scale,
                'CMz': cmz_scale,
            }

            fig = Plot.polar_plot(data, mode, model_size, alpha)
            self.clipboard_obj.clipboard_dict[alpha][model_scale]['model_attributes'][plot_name] = fig

        fig = self.clipboard_obj.clipboard_dict[alpha][model_scale]['model_attributes'].get(plot_name)

        print(f'Запрос суммарных коэффициентов в полярной системе координат размеры = {" ".join(list(model_size))} '
              f'альфа = {alpha} режим = {mode} из буфера успешно выполнен')

        return fig

    def get_envelopes(self,
                      alpha: str,
                      model_scale: str,
                      angle: str):
        """Функция возвращает графики огибающих.
        Если огибающие отсутствуют в буфере, запускается отрисовка.
        """
        print(f'Запрос огибающих параметры = {" ".join(list(model_scale))} '
              f'альфа = {alpha} угол = {int(angle):02} из буфера')
        if not self.clipboard_obj.clipboard_dict[alpha][model_scale][angle].get('envelopes'):
            print(f'Отрисовка огибающих {" ".join(list(model_scale))} {alpha} {int(angle):02}')

            pressure_coefficients = self.clipboard_obj.get_pressure_coefficients(alpha, model_scale, angle)
            figs = Plot.envelopes(pressure_coefficients, alpha, model_scale, angle)

            for ind, val in enumerate(figs):
                self.clipboard_obj.clipboard_dict[alpha][model_scale][angle][f'envelopes_{ind}'] = val
            self.clipboard_obj.clipboard_dict[alpha][model_scale][angle]['envelopes'] = True
            print(f'Отрисовка огибающих {" ".join(list(model_scale))} {alpha} {int(angle):02} завершена')

        figs = []
        i = 0
        while True:
            if not self.clipboard_obj.clipboard_dict[alpha][model_scale][angle].get(f'envelopes_{i}'):
                break
            else:
                figs.append(self.clipboard_obj.clipboard_dict[alpha][model_scale][angle][f'envelopes_{i}'])
                i += 1

        print(f'Запрос огибающих параметры = {" ".join(list(model_scale))} '
              f'альфа = {alpha} угол = {int(angle):02} из буфера успешно выполнен')

        return figs

    def draw_envelopes(self,
                       alpha: str,
                       model_scale: str,
                       angle_border: int,
                       path_report: str):
        """Функция запускает отрисовку всех огибающих выбранной модели.
        Отрисовка происходит в многопоточном режиме.
        """
        print(f'Отрисовка огибающих параметры = {" ".join(model_scale)} альфа = {alpha}')

        args_envelopes = [(alpha, model_scale, str(angle), path_report) for angle in range(0, angle_border, 5)]

        with ThreadPoolExecutor(max_workers=Core._count_threads) as executor:
            executor.map(lambda i: self.envelopes_thread(*i), args_envelopes)

        print(f'Отрисовка огибающих параметры = {" ".join(model_scale)} альфа = {alpha} завершена')

    def envelopes_thread(self,
                         alpha: str,
                         model_scale: str,
                         angle: str,
                         path_report: str):
        """Функция для запуска потока генерации графиков огибающих модели и их сохранения."""
        path_folder = f'{path_report}\\Огибающие\\Огибающие {model_scale} {alpha} {int(angle):02}'

        if not os.path.exists(path_folder):
            os.mkdir(path_folder)

        figs = self.get_envelopes(alpha, model_scale, angle)

        for i in range(len(figs)):
            file_name = f'Огибающие {model_scale} {alpha} {int(angle):02} {i * 100} - {(i + 1) * 100}.png'
            figs[i].savefig(f'{path_folder}\\{file_name}')

    def draw_welch_graphs(self,
                          alpha: str,
                          model_scale: str,
                          model_size: Tuple[str, str, str],
                          angle_border: int,
                          path_report: str):
        """Функция запускает отрисовку всех граффиков спектральной плотности мощности для выбранной модели.
        Отрисовка происходит в многопоточном режиме.
        """
        print(f'Отрисовка спектров суммарных коэффициентов параметры = {" ".join(model_scale)} альфа = {alpha}')

        args_welch_graphs = [(alpha, model_scale, str(angle), model_size, path_report)
                             for angle in range(0, angle_border, 5)]

        with ThreadPoolExecutor(max_workers=Core._count_threads) as executor:
            executor.map(lambda i: self.welch_graphs_thread(*i), args_welch_graphs)

        print(f'Отрисовка спектров суммарных коэффициентов '
              f'параметры = {" ".join(model_scale)} альфа = {alpha} завершена')

    def welch_graphs_thread(self,
                            alpha: str,
                            model_scale: str,
                            angle: str,
                            model_size: Tuple[str, str, str],
                            path_report: str):
        """Функция для запуска потока генерации графиков спектральной плотности мощности и их сохранения."""

        path_folder = f'{path_report}\\Спектральная плотность мощности\\Логарифмическая шкала'
        file_name = f'Спектральная плотность мощности {model_scale} {alpha} {int(angle):02}.png'

        fig = self.get_plot_summary_spectres(alpha, model_size, angle, 'Cx Cy CMz', 'log', 'summary_spectres')
        fig.savefig(f'{path_folder}\\{file_name}')

    def draw_isofields(self,
                       alpha: str,
                       model_size: Tuple[str, str, str],
                       angle_border: int,
                       path_report: str):
        """Функция запускает отрисовку всех изополей выбранной модели.
        Отрисовка происходит в многопоточном режиме.
        """
        print(f'Отрисовка изополей размеры = {" ".join(model_size)} альфа = {alpha}')

        mods = ('max',
                'mean',
                'min',
                'std',
                )

        args_isofields = [(alpha, model_size, str(angle), mode, path_report)
                          for angle in range(0, angle_border, 5) for mode in mods]

        # for i in range(len(args_isofields)):
        #     self.isofields_thread(*args_isofields[i])

        with ThreadPoolExecutor(max_workers=Core._count_threads) as executor:
            executor.map(lambda i: self.isofields_thread(*i), args_isofields)

        print(f'Отрисовка изополей размеры = {" ".join(model_size)} альфа = {alpha} завершена')

    def isofields_thread(self,
                         alpha: str,
                         model_size: Tuple[str, str, str],
                         angle: str,
                         mode: str,
                         path_report: str):
        """Функция для запуска потока генерации изополей и их сохранения."""

        file_name = f'Изополя {" ".join(model_size)} {alpha} {int(angle):02} {mode}.png'

        # if not os.path.exists(f'{path_report}\\Изополя ветровых нагрузок и воздействий\\Непрерывные\\'
        #                       f'{mode.upper()}\\{file_name}'):
            # fig = self.get_plot_isofields(alpha, model_size, str(angle), mode, type_plot)
            # fig.savefig(f'{path_report}\\Изополя ветровых нагрузок и воздействий\\Непрерывные\\'
            #             f'{mode.upper()}\\{file_name}')
        print(2)
        fig = self.get_plot_isofields(alpha, model_size, str(angle), mode, 'integral_isofields')
        print(1)

        fig.savefig(
            f'{path_report}\\Изополя ветровых нагрузок и воздействий\\Непрерывные\\{mode.upper()}\\{file_name}')

        fig = self.get_plot_isofields(alpha, model_size, str(angle), mode, 'discrete_isofields')
        print(1)

        fig.savefig(
            f'{path_report}\\Изополя ветровых нагрузок и воздействий\\Дискретные\\{mode.upper()}\\{file_name}')

    def draw_model(self,
                   alpha: str,
                   model_size: Tuple[str, str, str],
                   model_scale: str,
                   path_report: str):
        """Функция запускает отрисовку  выбранной модели.
        - развертка модели
        - модель в полярной системе координат
        - модель в трехмерном виде
        """
        print(f'Отрисовка модели размеры = {" ".join(model_size)} альфа = {alpha}')

        print('Генерация развертки модели')
        coordinates = self.clipboard_obj.get_coordinates(alpha, model_scale)
        fig = Plot.model_pic(model_size, model_scale, coordinates)
        fig.savefig(f'{path_report}\\Модель\\Развертка модели.png')
        print('Генерация развертки модели завершена')

        print('Генерация модели в полярной системе координат')
        fig = Plot.model_polar(model_size)
        fig.savefig(f'{path_report}\\Модель\\Модель в полярной системе координат.png')
        print('Генерация модели в полярной системе координат завершена')

        print('Генерация модели в трехмерном виде')
        fig = Plot.model_cube(model_size)
        fig.savefig(f'{path_report}\\Модель\\Модель 3D')
        print('Генерация модели в трехмерном виде завершена')

        print(f'Отрисовка модели размеры = {" ".join(model_size)} альфа = {alpha} завершена')

    def draw_summary_coefficients(self,
                                  alpha: str,
                                  model_size: Tuple[str, str, str],
                                  angle_border: int,
                                  path_report: str):
        """Функция запускает отрисовку всех графиков суммарных коэффициентов для выбранной модели.
        Отрисовка происходит в многопоточном режиме.
        """
        print(f'Отрисовка суммарных коэффициентов размеры = {" ".join(model_size)} альфа = {alpha}')

        args_summary_coefficients = [(alpha, model_size, str(angle), path_report)
                                     for angle in range(0, angle_border, 5)]

        with ThreadPoolExecutor(max_workers=Core._count_threads) as executor:
            executor.map(lambda i: self.summary_coefficients_thread(*i), args_summary_coefficients)

        print(f'Отрисовка суммарных коэффициентов размеры = {" ".join(model_size)} альфа = {alpha} завершена')

    def summary_coefficients_thread(self,
                                    alpha: str,
                                    model_size: Tuple[str, str, str],
                                    angle: str,
                                    path_report: str):
        """Функция для запуска потока генерации графиков суммарных аэродинамических коэффициентов модели
         и их сохранения."""

        file_name = f'Суммарные аэродинамические коэффициенты Cx_Cy_CMz ' \
                    f'{" ".join(model_size)} {alpha} {int(angle):02}.png'

        fig = self.get_plot_summary_coefficients(alpha, model_size, angle, 'Cx_Cy_CMz')

        fig.savefig(f'{path_report}\\Суммарные аэродинамические коэффициенты\\'
                    f'Декартовая система координат\\{file_name}')

    def draw_summary_coefficients_polar(self,
                                        alpha: str,
                                        model_scale: str,
                                        model_size: Tuple[str, str, str],
                                        angle_border: int,
                                        path_report: str):
        """Функция запускает отрисовку всех графиков суммарных коэффициентов в полярной системе для выбранной модели.
        Отрисовка происходит в многопоточном режиме.
        """
        print(f'Отрисовка суммарных аэродинамических коэффициентов в полярной системе координат '
              f'размеры = {" ".join(model_size)} альфа = {alpha}')
        mods = {
            'MEAN': np.mean,
            'RMS': rms,
            'STD': np.std,
            'MAX': np.max,
            'MIN': np.min,
            'РАСЧЕТНОЕ': rach,
            'ОБЕСП+': obes_m,
            'ОБЕСП-': obes_p
        }

        args_summary_polar = [(alpha, model_scale, model_size, angle_border, mode) for mode in mods.keys()]

        with ThreadPoolExecutor(max_workers=Core._count_threads) as executor:
            polar_plots = list(executor.map(lambda i: self.get_plot_summary_coefficients_polar(*i), args_summary_polar))

        for mode, plot in zip(mods.keys(), polar_plots):
            file_name = f'Суммарные аэродинамические коэффициенты Cx Cy CMz ' \
                        f'{" ".join(model_size)} {mode} в полярной системе координат.png'

            plot.savefig(f'{path_report}\\Суммарные аэродинамические коэффициенты\\Полярная система координат\\'
                         f'{file_name}')

        print(f'Отрисовка суммарных аэродинамических коэффициентов в полярной системе координат '
              f'размеры = {" ".join(model_size)} альфа = {alpha} завершена')

    def report(self, alpha: str, model_size: Tuple[str, str, str]):
        """Создание отчёта для выбранной конфигурации.
        Отчёт включает:
            - отображение модели
            - изополя
            - огибающие
            - суммарные аэродинамические коэффициенты
            - суммарные аэродинамические коэффициенты в полярной системе координат
            - спектральная плотность мощности суммарных аэродинамических коэффициентов
            - характеристика по датчикам
        """
        print(f'Начало формирования отчета размеры = {" ".join(model_size)} альфа = {alpha}')

        report_path = 'D:\\Projects\\WindSpectrum'

        model_scale, scale_factors = get_model_and_scale_factors(*model_size, alpha)
        breadth, depth, height = model_size
        name_report = f'Отчет ширина {breadth} глубина {depth} высота {height} альфа {alpha}'
        path_report = f'{report_path}\\{name_report}'
        generate_directory_for_report(path_report)

        if model_scale[0] == model_scale[1]:
            angle_border = 50
        else:
            angle_border = 95

        # Запрос данных из БД
        # Координаты являются одинаковыми для всех углов
        # и чтобы потоки не обращались к БД за ними
        # нужно запросить их заранее.
        self.clipboard_obj.get_coordinates(alpha, model_scale)

        # # Отображение модели
        # self.draw_model(alpha, model_size, model_scale, path_report)
        #
        # # Изополя
        # self.draw_isofields(alpha, model_size, angle_border, path_report)
        #
        # # Огибающие
        # self.draw_envelopes(alpha, model_scale, angle_border, path_report)

        # # Суммарные аэродинамические коэффициенты в декартовой системе координат
        # self.draw_summary_coefficients(alpha, model_size, angle_border, path_report)

        # # Суммарные аэродинамические коэффициенты в полярной системе координат
        # self.draw_summary_coefficients_polar(alpha, model_scale, model_size, angle_border, path_report)

        # # Спектральная плотность мощности суммарных аэродинамических коэффициентов
        # self.draw_welch_graphs(alpha, model_scale, model_size, angle_border, path_report)

        # Работа с word файлом
        doc = Document()
        style = doc.styles['Normal']
        style.font.size = Pt(14)
        style.font.name = 'Times New Roman'
        section = doc.sections[0]
        section.left_margin = Mm(30)
        section.right_margin = Mm(15)
        section.top_margin = Mm(20)
        section.bottom_margin = Mm(20)

        counter_plots = 1  # Счетчик графиков для нумерации
        counter_tables = 1  # Счетчик таблиц для нумерации

        # Раздел 1

        title = doc.add_heading().add_run(f'Отчет по зданию {breadth}x{depth}x{height}')
        doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
        title.font.size = Pt(24)
        title.bold = True

        for i in ('Параметры ветрового воздействия:',
                  'Ветровой район: None',
                  'Тип местности: None'
                  ):
            doc.add_paragraph().add_run(i)
        p = doc.add_paragraph()

        run = p.add_run()
        run.add_picture(f'{path_report}\\Модель\\3D Модель.png', width=Mm(82.5))
        run.add_picture(f'{path_report}\\Модель\\Модель в полярной системе координат.png', width=Mm(82.5))
        doc.add_paragraph().add_run(
            f'Рисунок {counter_plots}. Геометрические размеры и система координат направления ветровых потоков')
        counter_plots += 1

        # 1. Геометрические размеры здания

        doc.add_heading().add_run('1. Геометрические размеры здания').font.size = Pt(20)
        doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER

        header_model = (
            'Геометрический размер',
            'Значение, м'
        )
        table_model = doc.add_table(rows=1, cols=len(header_model))
        table_model.style = 'Table Grid'
        hdr_cells = table_model.rows[0].cells
        for i in range(len(header_model)):
            hdr_cells[i].add_paragraph().add_run(header_model[i])
        for i, j in zip((breadth, depth, height), ('Ширина:', 'Глубина:', 'Высота:')):
            row_cells = table_model.add_row().cells
            row_cells[0].add_paragraph().add_run(j)
            row_cells[1].add_paragraph().add_run(str(i))
        doc.add_picture(f'{path_report}\\Модель\\Развертка модели.png')
        doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
        doc.add_paragraph().add_run(f'Рисунок {counter_plots}. Система датчиков мониторинга')
        counter_plots += 1
        doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
        doc.add_page_break()

        # 2. Статистика по датчиках. Максимумы и огибающие

        doc.add_heading().add_run('2. Статистика по датчиках. Максимумы и огибающие').font.size = Pt(20)
        doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
        for angle in range(0, 50, 5):
            envelopes = glob.glob(
                f'{path_report}\\Огибающие\\Огибающие {model_scale}_{alpha} {angle:02}\\Огибающие *.png')
            for i in envelopes:
                doc.add_picture(i, height=Mm(80))
                doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
                doc.add_paragraph().add_run(
                    f'Рисунок {counter_plots}. Огибающая ветрового давления для здания '
                    f'{breadth}x{depth}x{height} угол {angle:02}º '
                    f'датчики {i[i[:i.rfind("-") - 1].rfind(" ") + 1:i.rfind(".")]}')
                doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
                counter_plots += 1
        doc.add_page_break()

        # 3. Изополя ветровых нагрузок и воздействий

        doc.add_heading().add_run('3. Изополя ветровых нагрузок и воздействий').font.size = Pt(20)
        doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
        doc.add_heading(level=2).add_run('3.1 Непрерывные изополя').font.size = Pt(16)
        doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER

        isofields = glob.glob(
            f'{path_report}\\Изополя ветровых нагрузок и воздействий\\Непрерывные\\Изополя *.png')
        for i in isofields:
            doc.add_picture(i)
            doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
            angle = i[i.rfind('_') + 1:i.rfind(' ')]
            mode = i[i.rfind(" ") + 1:i.rfind(".")]
            doc.add_paragraph().add_run(
                f'Рисунок {counter_plots}. Непрерывные изополя {mode} '
                f'для здания {breadth}x{depth}x{height} угол {angle}º')
            doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
            counter_plots += 1
        doc.add_heading(level=2).add_run('3.2 Дискретные изополя').font.size = Pt(16)
        doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
        isofields_disc = glob.glob(
            f'{path_report}\\Изополя ветровых нагрузок и воздействий\\Дискретные\\Изополя *.png')
        for i in isofields_disc:
            doc.add_picture(i)
            doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
            angle = i[i.rfind('_') + 1:i.rfind(' ')]
            mode = i[i.rfind(" ") + 1:i.rfind(".")]
            doc.add_paragraph().add_run(
                f'Рисунок {counter_plots}. Дискретные изополя {mode} '
                f'для здания {breadth}x{depth}x{height} угол {angle}º')
            doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
            counter_plots += 1
        doc.add_page_break()



        print(f'Формирование отчета размеры = {" ".join(model_size)} альфа = {alpha} завершено')


if __name__ == '__main__':
    c = Core()
    # fig = c.get_plot_summary_coefficients('6', ('0.1', '0.1', '0.1'), '0', 'Cx Cy CMz')
    # utils.utils.open_fig(fig)
    t1 = time.time()
    c.report('6', ('0.1', '0.1', '0.1'))
    print(time.time() - t1)
    # print('=========================================')
    # c.report('6', ('2', '2', '2'))
    # print(os.listdir('D:\\'))
