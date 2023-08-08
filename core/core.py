import os
import random
import time
import glob
import pickle
from typing import Tuple, List
from multiprocessing import Process, Manager, managers
from concurrent.futures import ThreadPoolExecutor
from utils.utils import ks10, alpha_standards, wind_regions
from utils.utils import interpolator as intp

import toml
import numpy as np
from docx import Document
from docx.shared import Pt, Mm
from docx.enum.text import WD_ALIGN_PARAGRAPH
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt

# Local imports
from plot.plot import Plot
from clipboard.clipboard import Clipboard
from utils.utils import (get_logger,
                         get_model_and_scale_factors,
                         rms,
                         rach,
                         obes_m,
                         obes_p,
                         converter_coordinates_to_real,
                         converter_coordinates,
                         generate_directory_for_report,
                         get_view_permutation_data,
                         get_sequence_permutation_data,
                         changer_sequence_coefficients,
                         get_base_angle,
                         to_dict,
                         to_multiprocessing_dict,
                         id_to_name,
                         get_model_and_scale_factors_interference,
                         )


class Core:
    config = toml.load('config.toml')

    _count_threads = config['core']['count_threads']  # количество запускаемых потоков при создании отчета

    def __init__(self, ex_clipboard=None):
        """Создание объекта для работы с буфером"""
        self.logger = get_logger('Core')
        self.logger.info('Создание ядра')
        self._manager = None
        self.clipboard_obj = Clipboard(ex_clipboard=ex_clipboard)
        self.logger.info('Ядро успешно создано')

    def save_clipboard(self):
        with open(f'{os.getcwd()}\\clipboard\\clipboard_binary', "wb") as clipboard_binary:
            dict_without_manager = to_dict(self.clipboard_obj.clipboard_dict)
            data_binary = pickle.dumps(dict_without_manager)
            clipboard_binary.write(data_binary)

    def get_plot_isofields(self,
                           db='isolated',
                           **kwargs):
        """Функция возвращает изополя"""
        if kwargs['pressure_plot_parameters']:
            type_plot = 'isofields_pressure'
        else:
            type_plot = 'isofields_coefficients'

        fig = self.clipboard_obj.get_isofields(db=db, type_plot=type_plot, **kwargs)

        return fig

    def get_plot_summary_spectres(self,
                                  db='interference',
                                  **kwargs):

        fig = self.clipboard_obj.get_summary_spectres(db=db, **kwargs)

        return fig

    def get_plot_summary_coefficients(self,
                                      db='isolated',
                                      **kwargs):

        fig = self.clipboard_obj.get_summary_coefficients(db, **kwargs)

        return fig

    def draw_welch_graphs(self, db, **kwargs):
        """Функция запускает отрисовку всех граффиков спектральной плотности мощности для выбранной модели.
        Отрисовка происходит в многопоточном режиме.
        """

        for mode in kwargs['mods']:
            args_welch_graphs = [(db, {'angle': str(angle), 'mode': mode}, kwargs) for angle in
                                 range(0, kwargs['angle_border'], 5)]
            for i in args_welch_graphs:
                self.welch_graphs_thread(i[0], **i[1], **i[2])

            with ThreadPoolExecutor(max_workers=Core._count_threads) as executor:
                executor.map(lambda i: self.welch_graphs_thread(i[0], **i[1], **i[2]), args_welch_graphs)

    def welch_graphs_thread(self, db, **kwargs):
        """Функция для запуска потока генерации графиков спектральной плотности мощности и их сохранения."""
        mode = kwargs['mode']
        path_report = kwargs['path_report']
        model_scale = kwargs['model_scale']
        angle = int(kwargs['angle'])

        mode_fig = ' '.join(mode.lower().title().split('_'))
        path_folder = f'{path_report}\\Спектральная плотность мощности\\Логарифмическая шкала'
        if db == 'isolated':
            alpha = kwargs['alpha']
            file_name = f'Спектральная плотность мощности {model_scale} {alpha} {mode_fig} {angle:02}.png'

        elif db == 'interference':
            case = kwargs['case']
            file_name = f'Спектральная плотность мощности {model_scale} вариант {case} {mode_fig} {angle:02}.png'

        fig = self.get_plot_summary_spectres(db, **kwargs, scale='log', type_plot='summary_spectres')
        fig.savefig(f'{path_folder}\\{file_name}')
        plt.close(fig)

    def get_plot_pressure_tap_locations(self, model_size, alpha):
        return self.clipboard_obj.get_plot_pressure_tap_locations(model_size, alpha)

    def get_plot_model_3d(self, model_size):
        return self.clipboard_obj.get_plot_model_3d(model_size)

    def get_model_polar(self, model_size):
        return self.clipboard_obj.get_model_polar(model_size)

    def get_pseudocolor_coefficients(self, alpha, model_size, angle, mode):
        return self.clipboard_obj.get_pseudocolor_coefficients(alpha, model_size, angle, mode)

    def draw_summary_coefficients(self,
                                  db,
                                  **kwargs):

        """Функция запускает отрисовку всех графиков суммарных коэффициентов для выбранной модели.
        Отрисовка происходит в многопоточном режиме.
        """

        for mode in kwargs['mods']:
            args_summary_coefficients = [(db, {'angle': str(angle), 'mode': mode}, kwargs)
                                         for angle in range(0, kwargs['angle_border'], 5)]

            # for i in args_summary_coefficients:
            #     self.summary_coefficients_thread(i[0], **i[1], **i[2])

            with ThreadPoolExecutor(max_workers=Core._count_threads) as executor:
                executor.map(lambda i: self.summary_coefficients_thread(i[0], **i[1], **i[2]),
                             args_summary_coefficients)

    def summary_coefficients_thread(self,
                                    db,
                                    **kwargs):
        """Функция для запуска потока генерации графиков суммарных аэродинамических коэффициентов модели
         и их сохранения."""
        angle = int(kwargs['angle'])
        mode = kwargs['mode']
        model_size = kwargs['model_size']
        path_report = kwargs['path_report']

        mode_fig = ' '.join(mode.lower().title().split('_'))
        if db == 'isolated':
            alpha = kwargs['alpha']
            file_name = f'Суммарные аэродинамические коэффициенты {mode_fig} ' \
                        f'{" ".join(model_size)} {alpha} {angle:02}.png'
        elif db == 'interference':
            case = kwargs['case']
            file_name = f'Суммарные аэродинамические коэффициенты {mode_fig} ' \
                        f'{" ".join(model_size)} вариант {case} {angle:02}.png'
        path_sum = f'{path_report}\\Суммарные аэродинамические коэффициенты\\Декартовая система координат\\' \
                   f'{angle:02}'
        if not os.path.isdir(path_sum):
            os.mkdir(path_sum)

        fig = self.get_plot_summary_coefficients(db, **kwargs)

        fig.savefig(f'{path_sum}\\{file_name}')
        plt.close(fig)

    def isofields_coefficients_thread(self,
                                      db,
                                      **kwargs):
        """Функция для запуска потока генерации изополей и их сохранения."""
        model_size = kwargs['model_size']
        angle = kwargs['angle']
        mode = kwargs['mode']
        path_report = kwargs['path_report']
        if db == 'isolated':
            alpha = kwargs['alpha']
            file_name = f'Изополя {" ".join(model_size)} {alpha} {int(angle):02} {mode}.png'

        elif db == 'interference':
            case = kwargs['case']
            file_name = f'Изополя {" ".join(model_size)} вариант {case} {int(angle):02} {mode}.png'

        fig = self.get_plot_isofields(db, pressure_plot_parameters=None, **kwargs)

        fig.savefig(
            f'{path_report}\\Изополя ветровых нагрузок и воздействий\\Коэффициенты\\{mode}\\{file_name}')

        plt.close(fig)

    def draw_isofields_coefficients(self,
                                    db,
                                    **kwargs, ):
        """Функция запускает отрисовку всех изополей выбранной модели.
        Отрисовка происходит в многопоточном режиме.
        """
        mods = kwargs['mods']
        angle_border = kwargs['angle_border']

        # if db == 'isolated':
        #     model_size = kwargs['model_size']
        #     alpha = kwargs['alpha']

        # self.logger.info(f'Отрисовка изополей размеры = {" ".join(model_size)} альфа = {alpha}')

        for mode in mods:
            args = [(db, {'angle': angle, 'mode': mode}, kwargs) for angle in range(0, angle_border, 5)]

            # for i in args:
            #     self.isofields_coefficients_thread(i[0], **i[1], **i[2])

            with ThreadPoolExecutor(max_workers=Core._count_threads) as executor:
                executor.map(lambda i: self.isofields_coefficients_thread(i[0], **i[1], **i[2]), args)

            # self.logger.info(f'Отрисовка изополей размеры = {" ".join(model_size)} альфа = {alpha} завершена')
        # elif db == 'interference':
        #     for mode in mods:
        #         args = [(alpha, model_size, str(angle), mode, path_report) for angle in
        #                 range(0, angle_border, 5)]
        #
        #         with ThreadPoolExecutor(max_workers=Core._count_threads) as executor:
        #             executor.map(lambda i: self.isofields_coefficients_thread(*i), args)

    def isofields_pressure_thread(self,
                                  db,
                                  **kwargs):
        model_size = kwargs['model_size']
        angle = kwargs['angle']
        mode = kwargs['mode']
        path_report = kwargs['path_report']

        if db == 'isolated':
            alpha = kwargs['alpha']
            file_name = f'Изополя давления {" ".join(model_size)} {alpha} {int(angle):02} {mode}.png'

        elif db == 'interference':
            case = kwargs['case']
            file_name = f'Изополя давления {" ".join(model_size)} вариант {case} {int(angle):02} {mode}.png'

        fig = self.get_plot_isofields(db, **kwargs)

        fig.savefig(
            f'{path_report}\\Изополя ветровых нагрузок и воздействий\\Давление\\{mode}\\{file_name}')
        plt.close(fig)

    def draw_isofields_pressure(self, db, **kwargs):
        mods = kwargs['mods']
        angle_border = kwargs['angle_border']

        for mode in mods:
            args = [(db, {'angle': angle, 'mode': mode}, kwargs) for angle in range(0, angle_border, 5)]

            with ThreadPoolExecutor(max_workers=Core._count_threads) as executor:
                executor.map(lambda i: self.isofields_pressure_thread(i[0], **i[1], **i[2]), args)

    def draw_pseudocolor_coefficients(self, alpha, model_size, angle_border, path_report, mods):
        for mode in mods:
            args = [(alpha, model_size, str(angle), mode, path_report) for angle in
                    range(0, angle_border, 5)]
            # for i in args:
            #     self.pseudocolor_coefficients_thread(*i)
            with ThreadPoolExecutor(max_workers=Core._count_threads) as executor:
                executor.map(lambda i: self.pseudocolor_coefficients_thread(*i), args)

    def pseudocolor_coefficients_thread(self, alpha, model_size, angle, mode, path_report):
        file_name = f'Мозаика коэффициентов {" ".join(model_size)} {alpha} {int(angle):02} {mode}.png'

        fig = self.get_pseudocolor_coefficients(alpha, model_size, angle, mode)

        fig.savefig(
            f'{path_report}\\Мозаика коэффициентов\\{mode}\\{file_name}')
        plt.close(fig)

    def draw_summary_coefficients_polar(self,
                                        db,
                                        **kwargs):
        """Функция запускает отрисовку всех графиков суммарных коэффициентов в полярной системе для выбранной модели.
        Отрисовка происходит в многопоточном режиме.
        """

        args = [(db, {'mode': mode}, kwargs) for mode in kwargs["mods"]]

        with ThreadPoolExecutor(max_workers=Core._count_threads) as executor:
            polar_plots = list(
                executor.map(lambda i: self.get_plot_summary_coefficients_polar(i[0], **i[1], **i[2]), args))
        path_folder = f'{kwargs["path_report"]}\\Суммарные аэродинамические коэффициенты\\Полярная система координат'

        for mode, fig in zip(kwargs["mods"], polar_plots):
            file_name = f'Суммарные аэродинамические коэффициенты Cx Cy CMz {" ".join(kwargs["model_size"])} {id_to_name[mode]}.png'
            fig.savefig(f'{path_folder}\\{file_name}')
            plt.close(fig)

    def get_plot_summary_coefficients_polar(self,
                                            db='isolated',
                                            **kwargs):
        """Функция возвращает графики суммарных коэффициентов в полярной системе координат.
        Если графики суммарных коэффициентов в полярной системе координат отсутствуют в буфере, запускается отрисовка.
        """
        if db == 'isolated':
            alpha = kwargs['alpha']
            mode = kwargs['mode']
            model_size = kwargs['model_size']
            model_scale = kwargs['model_scale']
            angle_border = kwargs['angle_border']

            self.logger.info(f'Запрос суммарных коэффициентов в '
                             f'полярной системе координат размеры = {" ".join(list(model_size))} '
                             f'альфа = {alpha} режим = {mode.ljust(4, " ")} из буфера')

            mods = {
                'mean': np.mean,
                'rms': rms,
                'std': np.std,
                'max': np.max,
                'min': np.min,
                'Расчетное': rach,
                'rach': rach,
                'obesP': obes_p,
                'obesM': obes_m,
                'Обеспеченность +': obes_m,
                'Обеспеченность -': obes_p
            }

            id_fig = f'summary_coefficients_Cx_Cy_CMz_polar_{"_".join(model_size)}_{mode}'

            if not self.clipboard_obj.clipboard_dict['isolated'][alpha][model_scale]['const_stuff'].get(id_fig):
                self.clipboard_obj.get_coordinates(db='isolated', alpha=alpha, model_scale=model_scale)

                args_cmz = [('isolated', {'alpha': alpha, 'model_scale': model_scale, 'angle': str(angle)}) for angle in
                            range(0, angle_border, 5)]
                with ThreadPoolExecutor(max_workers=Core._count_threads) as executor:
                    list_cmz = list(executor.map(lambda i: self.clipboard_obj.get_cmz(i[0], **i[1]), args_cmz))

                args_cx_cy = [('isolated', {'alpha': alpha, 'model_scale': model_scale, 'angle': str(angle)}) for angle
                              in range(0, angle_border, 5)]
                with ThreadPoolExecutor(max_workers=Core._count_threads) as executor:
                    cx_cy = np.array(
                        list(executor.map(lambda i: self.clipboard_obj.get_cx_cy(i[0], **i[1]), args_cx_cy)))

                list_cx = cx_cy[:, 0]
                list_cy = cx_cy[:, 1]

                list_norm_cmz = list(map(mods[mode], list_cmz))
                list_norm_cx = list(map(mods[mode], list_cx))
                list_norm_cy = list(map(mods[mode], list_cy))

                x_scale, y_scale = Plot.scaling_data(list_norm_cx, list_norm_cy, angle_border=angle_border)
                cmz_scale = Plot.scaling_data(list_norm_cmz, angle_border=angle_border)

                data = {
                    'Cx': x_scale,
                    'Cy': y_scale,
                    'CMz': cmz_scale,
                }

                fig = Plot.polar_plot(db, data=data, title=mode, model_size=model_size, alpha=alpha)
                self.clipboard_obj.clipboard_dict['isolated'][alpha][model_scale]['const_stuff'][id_fig] = fig

            fig = self.clipboard_obj.clipboard_dict['isolated'][alpha][model_scale]['const_stuff'].get(id_fig)

            self.logger.info(f'Запрос суммарных коэффициентов в полярной системе координат размеры = '
                             f'{" ".join(list(model_size))} альфа = {alpha} режим = {mode.ljust(4, " ")} из буфера успешно выполнен')
        elif db == 'interference':
            case = kwargs['case']
            mode = kwargs['mode']
            model_size = kwargs['model_size']
            model_scale = kwargs['model_scale']

            mods = {
                'mean': np.mean,
                'rms': rms,
                'std': np.std,
                'max': np.max,
                'min': np.min,
                'Расчетное': rach,
                'rach': rach,
                'obesP': obes_p,
                'obesM': obes_m,
                'Обеспеченность +': obes_m,
                'Обеспеченность -': obes_p
            }

            id_fig = f'summary_coefficients_Cx_Cy_CMz_polar_{"_".join(model_size)}_{mode}_{db}_{case}'

            if not self.clipboard_obj.clipboard_dict['interference'][model_scale][case]['const_stuff'].get(id_fig):
                self.clipboard_obj.get_coordinates(db='interference', case=case, model_scale=model_scale)
                args_cmz = [('interference', {'case': case, 'model_scale': model_scale, 'angle': angle}) for angle
                            in
                            range(0, 360, 5)]
                with ThreadPoolExecutor(max_workers=Core._count_threads) as executor:
                    list_cmz = list(executor.map(lambda i: self.clipboard_obj.get_cmz(i[0], **i[1]), args_cmz))

                args_cx_cy = [('interference', {'case': case, 'model_scale': model_scale, 'angle': angle}) for
                              angle in
                              range(0, 360, 5)]
                with ThreadPoolExecutor(max_workers=Core._count_threads) as executor:
                    cx_cy = np.array(
                        list(executor.map(lambda i: self.clipboard_obj.get_cx_cy(i[0], **i[1]), args_cx_cy)))

                list_cx = cx_cy[:, 0]
                list_cy = cx_cy[:, 1]

                list_norm_cmz = list(map(mods[mode], list_cmz))
                list_norm_cx = list(map(mods[mode], list_cx))
                list_norm_cy = list(map(mods[mode], list_cy))

                list_norm_cmz.append(list_norm_cmz[0])
                list_norm_cx.append(list_norm_cx[0])
                list_norm_cy.append(list_norm_cy[0])

                data = {
                    'Cx': list_norm_cx,
                    'Cy': list_norm_cy,
                    'CMz': list_norm_cmz,
                }

                fig = Plot.polar_plot(db, data=data, title=mode, model_size=model_size, case=case)
                self.clipboard_obj.clipboard_dict['interference'][model_scale][case]['const_stuff'][id_fig] = fig

            fig = self.clipboard_obj.clipboard_dict['interference'][model_scale][case]['const_stuff'].get(id_fig)

        return fig

    def get_envelopes(self,
                      db='isolated',
                      **kwargs
                      ):
        """Функция возвращает графики огибающих.
        Если огибающие отсутствуют в буфере, запускается отрисовка.
        """
        mods = ['mean', 'rms', 'std', 'max', 'min']

        if db == 'isolated':
            alpha = kwargs['alpha']
            model_scale = kwargs['model_scale']
            angle = kwargs['angle']
            mods = kwargs['mods']
            self.logger.info(f'Запрос огибающих параметры = {" ".join(list(model_scale))} '
                             f'альфа = {alpha} угол = {angle.rjust(2, "0")} из буфера')

            if not self.clipboard_obj.clipboard_dict['isolated'][alpha][model_scale][angle].get('envelopes'):
                self.logger.info(f'Отрисовка огибающих {" ".join(list(model_scale))} {alpha} {int(angle):02}')

                pressure_coefficients = self.clipboard_obj.get_pressure_coefficients('isolated', alpha=alpha,
                                                                                     model_name=model_scale,
                                                                                     angle=angle)
                figs = Plot.envelopes(pressure_coefficients, alpha, model_scale, angle, mods)

                for ind, val in enumerate(figs):
                    self.clipboard_obj.clipboard_dict['isolated'][alpha][model_scale][angle][
                        f'envelopes_{ind}_{"_".join(mods)}'] = val
                self.clipboard_obj.clipboard_dict['isolated'][alpha][model_scale][angle]['envelopes'] = True
                self.logger.info(f'Отрисовка огибающих {" ".join(list(model_scale))} {alpha} {int(angle):02} завершена')

            figs = []
            i = 0
            while True:
                if not self.clipboard_obj.clipboard_dict['isolated'][alpha][model_scale][angle].get(
                        f'envelopes_{i}_{"_".join(mods)}'):
                    break
                else:
                    figs.append(
                        self.clipboard_obj.clipboard_dict['isolated'][alpha][model_scale][angle][
                            f'envelopes_{i}_{"_".join(mods)}'])
                    i += 1

            self.logger.info(f'Запрос огибающих параметры = {" ".join(list(model_scale))} '
                             f'альфа = {alpha} угол = {angle.rjust(2, "0")} из буфера успешно выполнен')

        elif db == 'interference':
            case = kwargs['case']
            model_scale = kwargs['model_scale']
            angle = int(kwargs['angle'])
            if not self.clipboard_obj.clipboard_dict['interference'][model_scale][case][angle].get('envelopes'):
                pressure_coefficients = self.clipboard_obj.get_pressure_coefficients('interference', case=case,
                                                                                     model_name=model_scale,
                                                                                     angle=angle)

                figs = Plot.envelopes(pressure_coefficients, case, model_scale, angle, mods)

                for ind, val in enumerate(figs):
                    self.clipboard_obj.clipboard_dict['interference'][model_scale][case][angle][
                        f'envelopes_{ind}_{"_".join(mods)}'] = val
                self.clipboard_obj.clipboard_dict['interference'][model_scale][case][angle]['envelopes'] = True

            figs = []
            i = 0
            while True:
                if not self.clipboard_obj.clipboard_dict['interference'][model_scale][case][angle].get(
                        f'envelopes_{i}_{"_".join(mods)}'):
                    break
                else:
                    figs.append(
                        self.clipboard_obj.clipboard_dict['interference'][model_scale][case][angle][
                            f'envelopes_{i}_{"_".join(mods)}'])
                    i += 1

        return figs

    def draw_envelopes(self,
                       db, **kwargs):
        """Функция запускает отрисовку всех огибающих выбранной модели.
        Отрисовка происходит в многопоточном режиме.
        """
        args = [(db, {'angle': str(angle)}, kwargs) for angle in range(0, kwargs['angle_border'], 5)]

        # for i in args:
        #     self.envelopes_thread(i[0], **i[1], **i[2])

        # Для мощных пк, если есть 10ГБ оперативной памяти лишней
        with ThreadPoolExecutor(max_workers=Core._count_threads) as executor:
            executor.map(lambda i: self.envelopes_thread(i[0], **i[1], **i[2]), args)

    def envelopes_thread(self,
                         db,
                         **kwargs):
        """Функция для запуска потока генерации графиков огибающих модели и их сохранения."""
        figs = self.get_envelopes(db=db, **kwargs)
        path_report = kwargs['path_report']
        model_scale = kwargs['model_scale']
        angle = kwargs['angle']
        mods = kwargs['mods']

        if db == 'isolated':
            alpha = kwargs['alpha']
            path_envelopes = f'{path_report}\\Огибающие\\{model_scale} {alpha} {angle}'

        elif db == 'interference':
            case = kwargs['case']
            path_envelopes = f'{path_report}\\Огибающие\\{model_scale} {case} {angle}'

        if not os.path.isdir(path_envelopes):
            os.mkdir(path_envelopes)

        for i in range(len(figs)):
            if db == 'isolated':
                dbb = alpha
            elif db == 'interference':
                dbb = case

            file_name = f'Огибающие {model_scale} {dbb} {int(angle):02} ' \
                        f'{i * 100} - {(i + 1) * 100} {" ".join(mods)}.png'
            figs[i].savefig(f'{path_envelopes}\\{file_name}')
            plt.close(figs[i])

    @staticmethod
    def check_alive_proc(proc, button, spinner):
        while proc.is_alive():
            time.sleep(1)

        spinner.active = False
        button.disabled = False

    def wrapper_for_clipboard(self):
        if not self._manager:
            self._manager = Manager()

        self.clipboard_obj.clipboard_dict = to_multiprocessing_dict(self.clipboard_obj.clipboard_dict, self._manager)

    def draw_plot_model_3d(self, model_size, path_report):
        fig = self.get_plot_model_3d(model_size)
        fig.savefig(f'{path_report}\\Модель\\Модель трехмерная.png')
        plt.close(fig)

    def draw_model_polar(self, model_size, path_report):
        fig = self.get_model_polar(model_size)
        fig.savefig(f'{path_report}\\Модель\\Модель в полярной системе.png')
        plt.close(fig)

    def draw_plot_pressure_tap_locations(self, model_size, alpha, path_report):
        fig = self.get_plot_pressure_tap_locations(model_size, alpha)
        fig.savefig(f'{path_report}\\Модель\\Развертка модели.png')
        plt.close(fig)

    def get_summary_coefficients_statistics(self, db, **kwargs):
        mods = kwargs['mods']

        accuracy_values = 3
        statistics = []

        data = dict()

        cmz = self.clipboard_obj.get_cmz(db, **kwargs)
        data['cmz'] = cmz

        cx, cy = self.clipboard_obj.get_cx_cy(db, **kwargs)
        data['cx'] = cx
        data['cy'] = cy

        functions = {
            'max': lambda d: np.max(d, axis=0).round(accuracy_values),
            'mean': lambda d: np.mean(d, axis=0).round(accuracy_values),
            'min': lambda d: np.min(d, axis=0).round(accuracy_values),
            'std': lambda d: np.std(d, axis=0).round(accuracy_values),
            'rms': lambda d: rms(d).round(accuracy_values),
            'rach': lambda d: rach(d).round(accuracy_values),
            'obesP': lambda d: obes_p(d).round(accuracy_values),
            'obesM': lambda d: obes_m(d).round(accuracy_values)}

        for name in data.keys():
            statistics_name = []
            statistics_name.append(name.capitalize())

            for mode in mods:
                statistics_name.append(functions[mode](data[name]))

            statistics.append(statistics_name)

        return statistics

    def get_sensor_statistics(self, db, **kwargs):
        model_size = kwargs['model_size']
        mods = kwargs['mods']

        accuracy_values = 2

        breadth_real, depth_real, height_real = float(model_size[0]), float(model_size[1]), float(model_size[2])
        x, z = self.clipboard_obj.get_coordinates(db, **kwargs)
        sensors_on_model = len(x)

        if db == 'isolated':
            alpha = kwargs['alpha']
            model_scale = kwargs['model_scale']
            face_number = self.clipboard_obj.get_face_number(alpha, model_scale)
            kwargs['angle'] = str(kwargs['angle'])

        elif db == 'interference':
            kwargs['angle'] = int(kwargs['angle'])
            face_number = []
            for _ in range(sensors_on_model // 28):
                face_number.extend([1 for _ in range(7)])
                face_number.extend([2 for _ in range(7)])
                face_number.extend([3 for _ in range(7)])
                face_number.extend([4 for _ in range(7)])

        kwargs['model_name'] = kwargs['model_scale']

        x_new, y_new = converter_coordinates(x, breadth_real, depth_real, face_number, sensors_on_model)

        pressure_coefficients = self.clipboard_obj.get_pressure_coefficients(db, **kwargs)

        pressure_coefficients_t = pressure_coefficients.T

        functions = {'x': lambda: x_new,
                     'y': lambda: y_new,
                     'z': lambda: z,
                     'max': lambda: np.max(pressure_coefficients, axis=0).round(accuracy_values),
                     'mean': lambda: np.mean(pressure_coefficients, axis=0).round(accuracy_values),
                     'min': lambda: np.min(pressure_coefficients, axis=0).round(accuracy_values),
                     'std': lambda: np.std(pressure_coefficients, axis=0).round(accuracy_values),
                     'rms': lambda: [rms(i).round(accuracy_values) for i in pressure_coefficients_t],
                     'rach': lambda: [rach(i).round(accuracy_values) for i in pressure_coefficients_t],
                     'obesP': lambda: [obes_p(i).round(accuracy_values) for i in pressure_coefficients_t],
                     'obesM': lambda: [obes_m(i).round(accuracy_values) for i in pressure_coefficients_t]}

        statistics_of_angle = [[i for i in range(1, sensors_on_model + 1)]]

        for k in mods:
            statistics_of_angle.append(functions[k]())

        return np.array(statistics_of_angle).T

    def preparation_for_report(self,
                               pressure_plot_parameters,
                               content,
                               button,
                               spinner,
                               **kwargs):
        # if isinstance(self.clipboard_obj.clipboard_dict, dict):
        #     self.wrapper_for_clipboard()

        args = (self.clipboard_obj.clipboard_dict, pressure_plot_parameters, content)

        Core.start_report(*args, **kwargs)

        # proc = Process(target=Core.start_report, args=args, kwargs=kwargs)
        # proc.start()

        # executor = ThreadPoolExecutor(max_workers=1)
        # executor.map(lambda i: Core.check_alive_proc(*i), ((proc, button, spinner),))

    @staticmethod
    def start_report(clipboard_dict, pressure_plot_parameters, content, **kwargs):
        new_core = Core(clipboard_dict)
        new_core.report(pressure_plot_parameters, content, **kwargs)

    def report(self, pressure_plot_parameters, content, **kwargs):
        t1 = time.time()
        db = kwargs['db']
        del kwargs['db']
        model_size = kwargs['model_size']
        breadth, depth, height = model_size

        if db == 'isolated':
            alpha = kwargs['alpha']
            model_scale, scale_factors = get_model_and_scale_factors(*model_size, alpha)

            name_report = f'Отчет ширина {breadth} глубина {depth} высота {height} альфа {alpha}'
            if model_scale[0] == model_scale[1]:
                angle_border = 50
            else:
                angle_border = 95

        elif db == 'interference':
            case = kwargs['case']
            model_scale, scale_factors = get_model_and_scale_factors_interference(*model_size)

            name_report = f'Отчет ширина {breadth} глубина {depth} высота {height} вариант {case}'
            angle_border = 360

        current_path = os.getcwd()

        path_report = f'{current_path}\\Отчеты\\{name_report}'

        kwargs['model_scale'] = model_scale
        kwargs['angle_border'] = angle_border
        kwargs['path_report'] = path_report

        generate_directory_for_report(current_path, name_report)

        # Только эти графики используют координаты
        if any((content['isofieldsCoefficients'][0],
                # content['pressureTapLocations'][0] if db == 'isolated' else None,
                content['pseudocolorCoefficients'][0],
                content['isofieldsPressure'][0],
                content['summaryCoefficients'][0],
                content['statisticsSensors'][0],
                content['statisticsSummaryCoefficients'][0],
                content['summarySpectres'][0])):
            self.clipboard_obj.get_coordinates(db=db, **kwargs)
            args = [(db, {'angle': angle}, kwargs) for angle in range(0, angle_border, 5)]

            with ThreadPoolExecutor(max_workers=Core._count_threads) as executor:
                executor.map(lambda i: self.clipboard_obj.get_pressure_coefficients(i[0], **i[1], **i[2]), args)

        # # Отрисовка графиков
        # self.logger.info(f'Отрисовка графиков')
        # # isofieldsPressure
        # if content['isofieldsPressure'][0]:
        #     mods_isofieldsPressure = [k for k in content['isofieldsPressure'][1].keys() if
        #                               content['isofieldsPressure'][1][k]]
        #     self.draw_isofields_pressure(db, **kwargs, mods=mods_isofieldsPressure,
        #                                  pressure_plot_parameters=pressure_plot_parameters)
        #     self.logger.info(f'isofieldsPressure')
        #
        # # isofieldsCoefficients
        # if content['isofieldsCoefficients'][0]:
        #     mods_isofieldsCoefficients = [k for k in content['isofieldsCoefficients'][1].keys() if
        #                                   content['isofieldsCoefficients'][1][k]]
        #     self.draw_isofields_coefficients(db, **kwargs, mods=mods_isofieldsCoefficients)
        #     self.logger.info(f'isofieldsCoefficients')
        #
        # # pseudocolorCoefficients
        # if content['pseudocolorCoefficients'][0] and db == 'isolated':
        #     mods_pseudocolorCoefficients = [k for k in content['pseudocolorCoefficients'][1].keys() if
        #                                     content['pseudocolorCoefficients'][1][k]]
        #     self.draw_pseudocolor_coefficients(alpha=kwargs['alpha'], model_size=kwargs['model_size'],
        #                                        angle_border=kwargs['angle_border'], path_report=kwargs['path_report'],
        #                                        mods=mods_pseudocolorCoefficients)
        #     self.logger.info(f'pseudocolorCoefficients')
        # # envelopes
        # if content['envelopes'][0]:
        #     mods_envelopes = [k for k in content['envelopes'][1].keys() if content['envelopes'][1][k]]
        #     self.draw_envelopes(db, **kwargs, mods=mods_envelopes)
        #     self.logger.info(f'envelopes')
        #
        # # polarSummaryCoefficients
        # if content['polarSummaryCoefficients'][0]:
        #     mods_polarSummaryCoefficients = [k for k in content['polarSummaryCoefficients'][1].keys() if
        #                                      content['polarSummaryCoefficients'][1][k]]
        #     self.draw_summary_coefficients_polar(db, **kwargs, mods=mods_polarSummaryCoefficients)
        #     self.logger.info(f'polarSummaryCoefficients')
        #
        # # summaryCoefficients
        # if content['summaryCoefficients'][0]:
        #     mods_summaryCoefficients = [k for k in content['summaryCoefficients'][1].keys() if
        #                                 content['summaryCoefficients'][1][k]]
        #     self.draw_summary_coefficients(db, **kwargs, mods=mods_summaryCoefficients)
        #     self.logger.info(f'summaryCoefficients')
        #
        # # summarySpectres
        # if content['summarySpectres'][0]:
        #     mods_summarySpectres = [k for k in content['summarySpectres'][1].keys() if content['summarySpectres'][1][k]]
        #     self.draw_welch_graphs(db, **kwargs, mods=mods_summarySpectres)
        #     self.logger.info(f'summarySpectres')

        # pressureTapLocations
        # if db == 'isolated':
        #     if content['pressureTapLocations'][0]:
        #         self.draw_plot_pressure_tap_locations(model_size, alpha, path_report)
        #         self.logger.info(f'pressureTapLocations')

        if db == 'isolated':
            self.draw_plot_model_3d(model_size, path_report)
            self.draw_model_polar(model_size, path_report)

        self.logger.info(f'Отрисовка графиков все')

        # Получение статистики
        self.logger.info(f'Получение статистики')
        # statisticsSensors
        # statistics_sensors = [
        #     угол 0
        #     [
        #         [1 ...],
        #         [2 ...],
        #         [3 ...],
        #         ...
        #     ],
        #     угол 5
        #     [
        #         [1 ...],
        #         [2 ...],
        #         [3 ...],
        #         ...
        #     ],
        # ]
        if content['statisticsSensors'][0]:
            mods_statisticsSensors = [k for k in content['statisticsSensors'][1].keys() if
                                      content['statisticsSensors'][1][k]]
            if mods_statisticsSensors:
                args = [(db, {'mods': mods_statisticsSensors, 'angle': angle}, kwargs) for angle in
                        range(0, angle_border, 5)]
                with ThreadPoolExecutor(max_workers=self._count_threads) as executor:
                    statistics_sensors = list(
                        executor.map(lambda i: self.get_sensor_statistics(i[0], **i[1], **i[2]), args))

        # statisticsSummaryCoefficients
        # statistics_summary_coefficients = [
        #     угол 0
        #     [
        #         [cx ...],
        #         [cy ...],
        #         [cmz ...],
        #     ],
        #     угол 5
        #     [
        #         [cx ...],
        #         [cy ...],
        #         [cmz ...],
        #     ],
        # ]
        if content['statisticsSummaryCoefficients'][0]:
            mods_statisticsSummaryCoefficients = [k for k in content['statisticsSummaryCoefficients'][1].keys()
                                                  if content['statisticsSummaryCoefficients'][1][k]]

            args = [(db, {'mods': mods_statisticsSummaryCoefficients, 'angle': angle}, kwargs) for angle in
                    range(0, angle_border, 5)]
            with ThreadPoolExecutor(max_workers=self._count_threads) as executor:
                statistics_summary_coefficients = list(
                    executor.map(lambda i: self.get_summary_coefficients_statistics(i[0], **i[1], **i[2]), args))

        self.logger.info(f'Получение статистики все')

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
        # ширина A4 210 мм высота 297 мм
        fig_width = Mm(165)

        # Шрифт заголовков разного уровня
        head_lvl1 = Pt(20)
        head_lvl2 = Pt(16)

        counter_plots = 1  # Счетчик графиков для нумерации
        counter_tables = 1  # Счетчик таблиц для нумерации
        counter_head = 1  # Счетчик заголовков

        title = doc.add_heading()
        run = title.add_run(f'Отчет по зданию {breadth}x{depth}x{height}')
        title.alignment = WD_ALIGN_PARAGRAPH.CENTER
        run.font.size = Pt(24)
        run.bold = True

        for i in ('Параметры ветрового воздействия:',
                  f'Ветровой район: {pressure_plot_parameters["wind_region"]}',
                  f'Тип местности: {pressure_plot_parameters["type_area"]}'
                  ):
            doc.add_paragraph().add_run(i)
        if db == 'isolated':
            p = doc.add_paragraph()
            run = p.add_run()
            run.add_picture(f'{path_report}\\Модель\\Модель трехмерная.png', width=fig_width / 2)
            run.add_picture(f'{path_report}\\Модель\\Модель в полярной системе.png', width=fig_width / 2)
            run.add_text(f'Рисунок {counter_plots}. '
                         f'Геометрические размеры и система координат направления ветровых потоков')
            counter_plots += 1

        self.logger.info('1. Геометрические размеры здания')

        head = doc.add_heading()
        run = head.add_run(f'{counter_head}. Геометрические размеры здания')
        head.alignment = WD_ALIGN_PARAGRAPH.CENTER
        run.font.size = head_lvl1

        counter_head += 1

        header_model = (
            'Геометрический размер',
            'Значение, м'
        )

        table_model = doc.add_table(rows=1, cols=len(header_model))
        table_model.style = 'Table Grid'

        head_cells = table_model.rows[0].cells
        for i, name in enumerate(header_model):
            p = head_cells[i].paragraphs[0]
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            cell = p.add_run(name)
            cell.bold = True

        for i, j in zip((breadth, depth, height), ('Ширина:', 'Глубина:', 'Высота:')):
            row_cells = table_model.add_row().cells
            row_cells[0].text = j
            row_cells[1].text = str(i)

        # if content['pressureTapLocations'][0]:
        #     p = doc.add_paragraph()
        #     run = p.add_run()
        #     run.add_picture(f'{path_report}\\Модель\\Развертка модели.png', width=fig_width)
        #     run.add_text(f'Рисунок {counter_plots}. Система датчиков мониторинга')
        #     counter_plots += 1
        #     p.alignment = WD_ALIGN_PARAGRAPH.CENTER

        doc.add_page_break()

        self.logger.info(f'{counter_head}. Статистика по датчиках. Максимумы и огибающие')
        if content['envelopes'][0]:
            head = doc.add_heading()
            run = head.add_run(f'{counter_head}. Статистика по датчиках. Максимумы и огибающие')
            head.alignment = WD_ALIGN_PARAGRAPH.CENTER
            run.font.size = head_lvl1

            counter_head += 1
            for i1 in os.listdir(f'{path_report}\\Огибающие'):
                for i2 in os.listdir(f'{path_report}\\Огибающие\\{i1}'):
                    p = doc.add_paragraph()
                    run = p.add_run()
                    run.add_picture(f'{path_report}\\Огибающие\\{i1}\\{i2}', width=fig_width)
                    run.add_text(f'Рисунок {counter_plots}. {i2[:-4]}')
                    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
                    counter_plots += 1
            doc.add_page_break()

        self.logger.info(f'{counter_head}. Изополя ветровых нагрузок и воздействий')

        if content['isofieldsPressure'][0] or content['isofieldsCoefficients'][0]:
            head = doc.add_heading()
            run = head.add_run(f'{counter_head}. Изополя ветровых нагрузок и воздействий')
            run.font.size = head_lvl1
            head.alignment = WD_ALIGN_PARAGRAPH.CENTER

            counter_head += 1

        self.logger.info('3.1 Коэффициенты изополя')
        counter_head_lvl2 = 0.1
        if content['isofieldsCoefficients'][0]:
            head = doc.add_heading(level=2)
            run = head.add_run(f'{counter_head + counter_head_lvl2} Коэффициенты изополя')
            head.alignment = WD_ALIGN_PARAGRAPH.CENTER
            run.font.size = head_lvl2

            counter_head_lvl2 += 0.1
            path_temp = 'Изополя ветровых нагрузок и воздействий'
            for i1 in os.listdir(f'{path_report}\\{path_temp}\\Коэффициенты'):
                for i2 in os.listdir(f'{path_report}\\{path_temp}\\Коэффициенты\\{i1}'):
                    p = doc.add_paragraph()
                    run = p.add_run()
                    run.add_picture(f'{path_report}\\{path_temp}\\Коэффициенты\\{i1}\\{i2}')
                    run.add_text(f'Рисунок {counter_plots}. {i2[:-4]}')
                    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
                    counter_plots += 1

        self.logger.info('3.2 Изополя давления')
        if content['isofieldsPressure'][0]:
            head = doc.add_heading(level=2)
            run = head.add_run(f'{counter_head + counter_head_lvl2}. Изополя давления')
            head.alignment = WD_ALIGN_PARAGRAPH.CENTER
            run.font.size = head_lvl2

            counter_head_lvl2 += 0.1
            path_temp = 'Изополя ветровых нагрузок и воздействий'
            for i1 in os.listdir(f'{path_report}\\{path_temp}\\Давление'):
                for i2 in os.listdir(f'{path_report}\\{path_temp}\\Давление\\{i1}'):
                    p = doc.add_paragraph()
                    run = p.add_run()
                    run.add_picture(f'{path_report}\\{path_temp}\\Давление\\{i1}\\{i2}')
                    run.add_text(f'Рисунок {counter_plots}. {i2[:-4]}')
                    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
                    counter_plots += 1
            doc.add_page_break()

        self.logger.info('4. Статистика по датчикам в табличном виде')

        if content['statisticsSensors'][0]:
            head = doc.add_heading()
            run = head.add_run(f'{counter_head}. Статистика по датчикам в табличном виде ')
            run.font.size = head_lvl1
            head.alignment = WD_ALIGN_PARAGRAPH.CENTER

            counter_head += 1

            header_sensors = ['Датчик']
            header_sensors.extend([id_to_name[mode] for mode in mods_statisticsSensors])

            for angle in range(0, angle_border, 5):
                self.logger.info(f'4. Статистика по датчикам в табличном виде угол {angle}')
                p = doc.add_paragraph()
                p.add_run(
                    f'\nТаблица {counter_tables}. Аэродинамический коэффициент в датчиках для '
                    f'здания {breadth}x{depth}x{height} угол {angle:02}º')
                p.alignment = WD_ALIGN_PARAGRAPH.CENTER
                counter_tables += 1
                table_sensors = doc.add_table(rows=1, cols=len(header_sensors))
                table_sensors.style = 'Table Grid'
                head_cells = table_sensors.rows[0].cells
                for i, name in enumerate(header_sensors):
                    p = head_cells[i].paragraphs[0]
                    p.alignment = WD_ALIGN_PARAGRAPH.CENTER

                    cell = p.add_run(name)
                    cell.bold = True
                    cell.font.size = Pt(8)

                for row in statistics_sensors[angle // 5]:
                    cells = table_sensors.add_row().cells
                    for i, value in enumerate(row):
                        cells[i].text = str(value)

                        cells[i].paragraphs[0].runs[0].font.size = Pt(12)

            doc.add_page_break()

            del statistics_sensors

        self.logger.info(f'{counter_head}. Суммарные значения аэродинамических коэффициентов')
        if content['summaryCoefficients'][0]:
            head = doc.add_heading()
            run = head.add_run(f'{counter_head}. Графики суммарных аэродинамических коэффициентов')
            run.font.size = head_lvl1
            head.alignment = WD_ALIGN_PARAGRAPH.CENTER

            counter_head += 1
            path_temp = 'Суммарные аэродинамические коэффициенты'
            for i1 in os.listdir(f'{path_report}\\{path_temp}\\Декартовая система координат'):
                for i2 in os.listdir(f'{path_report}\\{path_temp}\\Декартовая система координат\\{i1}'):
                    p = doc.add_paragraph()
                    run = p.add_run()
                    run.add_picture(f'{path_report}\\{path_temp}\\Декартовая система координат\\{i1}\\{i2}')
                    run.add_text(f'Рисунок {counter_plots}. {i2[:-4]}')
                    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
                    counter_plots += 1

        if content['statisticsSummaryCoefficients'][0]:
            head = doc.add_heading()
            run = head.add_run(f'{counter_head}. Статистика суммарных аэродинамических коэффициентов')
            run.font.size = head_lvl1
            head.alignment = WD_ALIGN_PARAGRAPH.CENTER

            counter_head += 1

            header_sum = ['Сила']
            header_sum.extend([id_to_name[mode] for mode in mods_statisticsSummaryCoefficients])

            for angle in range(0, angle_border, 5):
                doc.add_paragraph().add_run(
                    f'Таблица {counter_tables}. Суммарные аэродинамические коэффициенты '
                    f'для здания {breadth}x{depth}x{height} угол {angle:02}º')
                counter_tables += 1

                table = doc.add_table(rows=1, cols=len(header_sum))
                table.style = 'Table Grid'
                head_cells = table.rows[0].cells
                for i, name in enumerate(header_sum):
                    p = head_cells[i].paragraphs[0]
                    p.alignment = WD_ALIGN_PARAGRAPH.CENTER

                    cell = p.add_run(name)
                    cell.bold = True
                    cell.font.size = Pt(8)

                for row in statistics_summary_coefficients[angle // 5]:
                    cells = table.add_row().cells
                    for i, value in enumerate(row):
                        cells[i].text = str(value)

                        cells[i].paragraphs[0].runs[0].font.size = Pt(12)

            del statistics_summary_coefficients
            doc.add_page_break()

        if content['polarSummaryCoefficients'][0]:
            head = doc.add_heading()
            run = head.add_run(f'{counter_head}. Cуммарные аэродинамических коэффициенты в полярной системе координат')
            run.font.size = head_lvl1
            head.alignment = WD_ALIGN_PARAGRAPH.CENTER

            counter_head += 1
            path_temp = 'Суммарные аэродинамические коэффициенты'
            for i1 in os.listdir(f'{path_report}\\{path_temp}\\Полярная система координат'):
                p = doc.add_paragraph()
                run = p.add_run()
                run.add_picture(f'{path_report}\\{path_temp}\\Полярная система координат\\{i1}')
                run.add_text(f'Рисунок {counter_plots}. {i1[:-4]}')
                p.alignment = WD_ALIGN_PARAGRAPH.CENTER
                counter_plots += 1
                doc.add_page_break()

        self.logger.info('6. Спектры cуммарных значений аэродинамических коэффициентов')

        if content['summarySpectres'][0]:
            head = doc.add_heading()
            run = head.add_run(f'{counter_head}. Спектры cуммарных значений аэродинамических коэффициентов')
            run.font.size = head_lvl1
            head.alignment = WD_ALIGN_PARAGRAPH.CENTER

            counter_head += 1
            path_temp = 'Спектральная плотность мощности'
            for i1 in os.listdir(f'{path_report}\\{path_temp}\\Логарифмическая шкала'):
                    p = doc.add_paragraph()
                    run = p.add_run()
                    run.add_picture(f'{path_report}\\{path_temp}\\Логарифмическая шкала\\{i1}')
                    run.add_text(f'Рисунок {counter_plots}. {i1[:-4]}')
                    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
                    counter_plots += 1
            doc.add_page_break()

        if db == 'isolated':
            doc.save(f'{path_report}\\Отчет ширина {breadth} глубина {depth} высота {height} альфа {alpha}.docx')
            os.startfile(f'{path_report}\\Отчет ширина {breadth} глубина {depth} высота {height} альфа {alpha}.docx')

        elif db == 'interference':
            doc.save(f'{path_report}\\Отчет ширина {breadth} глубина {depth} высота {height} вариант {case}.docx')
            os.startfile(f'{path_report}\\Отчет ширина {breadth} глубина {depth} высота {height} вариант {case}.docx')

        self.logger.info(f'{time.time() - t1} Затраченное время на создание отчета')

    # Интегрирование по высоте
    def height_integration(self,
                           alpha: str,
                           model_size: Tuple[str, str, str],
                           angle: str,
                           mode: str,
                           pressure_plot_parameters: dict,
                           faces: tuple,
                           step: tuple):

        COUNT_DOTS = 100

        model_scale, scale_factors = get_model_and_scale_factors(*model_size, alpha)
        pressure_coefficients = self.clipboard_obj.get_pressure_coefficients('isolated', alpha=alpha,
                                                                             model_name=model_scale,
                                                                             angle=angle)
        coordinates = self.clipboard_obj.get_coordinates(alpha, model_scale)
        model_name = model_scale

        # Виды изополей
        mods = {
            'max': np.max(pressure_coefficients, axis=0),
            'mean': np.mean(pressure_coefficients, axis=0),
            'min': np.min(pressure_coefficients, axis=0),
            'std': np.std(pressure_coefficients, axis=0),
        }

        size_x, size_y, size_z = map(float, model_size)
        x_scale_factor, y_scale_factor, z_scale_factor = scale_factors

        pressure_coefficients = mods[mode]
        breadth, depth, height = int(model_name[0]) / 10, int(model_name[1]) / 10, int(model_name[2]) / 10
        count_sensors_on_model = len(pressure_coefficients)
        count_sensors_on_middle = int(model_name[0]) * 5
        count_sensors_on_side = int(model_name[1]) * 5
        count_row = count_sensors_on_model // (2 * (count_sensors_on_middle + count_sensors_on_side))

        x, z = np.array(coordinates)

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

        z_marks = [0]
        scaled_height = height * z_scale_factor
        if len(step) == 1:
            pr = step[0] * 100 / size_z
            step_m = scaled_height * pr / 100
            z_marks = np.append(np.arange(0, height * z_scale_factor, step_m), scaled_height)
        else:
            for s in step:
                pr = s * 100 / size_z
                step_m = scaled_height * pr / 100
                z_marks.append(step_m)

            z_marks.append(scaled_height)
        l_z_marks = len(z_marks)

        x_marks_border = breadth * x_scale_factor
        y_marks_border = depth * y_scale_factor

        x_dots = None
        y_dots = None
        z_dots = []

        for i in [f - 1 for f in faces]:
            x_old = x[i].reshape(1, -1)[0]
            z_old = z[i].reshape(1, -1)[0]
            # Вычитаем чтобы все координаты по x находились в интервале [0, 1]
            if i == 1:
                x_old -= breadth
            elif i == 2:
                x_old -= (breadth + depth)
            elif i == 3:
                x_old -= (2 * breadth + depth)

            z_old = z_old * z_scale_factor
            if i in [0, 2]:
                x_old = x_old * x_scale_factor
                if not x_dots:
                    x_dots = [random.uniform(0, x_marks_border) for _ in range(COUNT_DOTS)]
            else:
                x_old = x_old * y_scale_factor
                if not y_dots:
                    y_dots = [random.uniform(0, y_marks_border) for _ in range(COUNT_DOTS)]

            if not z_dots:
                for b1 in range(l_z_marks - 1):
                    z_dots.append([random.uniform(z_marks[b1], z_marks[b1 + 1]) for _ in range(COUNT_DOTS)])

            data_old = pressure_coefficients[i].reshape(1, -1)[0]
            data_old = [coefficient_for_pressure * coefficient for coefficient in data_old]

            coords = [[i1, j1] for i1, j1 in zip(x_old, z_old)]  # Старые координаты
            # Интерполятор полученный на основе имеющихся данных
            interpolator = intp(coords, data_old)

            data_new = []
            if i in [0, 2]:
                for dots in z_dots:
                    data_new.append(np.mean([float(interpolator([[X, Y]])) for X, Y in zip(x_dots, dots)]))
            else:
                for dots in z_dots:
                    data_new.append(np.mean([float(interpolator([[X, Y]])) for X, Y in zip(y_dots, dots)]))

            print(data_new)

    def create_word_isolated(self):
        pass


if __name__ == '__main__':
    c = Core()
    c.report('6', ('3', '1', '5'))
    # run_proc(c.clipboard_obj.clipboard_dict, '4', ('1', '1', '1'))
    # fig = c.get_plot_summary_coefficients('6', ('0.1', '0.1', '0.1'), '0', 'Cx Cy CMz')
    # utils.utils.open_fig(fig)
    # t1 = time.time()
    # c.preparation_for_report('6', ('1', '4', '8'))
    # c.get_plot_isofields('6', ('1', '1', '1'),'0','mean','isofields_coefficients')
    # c.save_clipboard()
    # c.preparation_for_report('6', ('1', '1', '1'))
    # print(time.time() - t1)
    # print('=========================================')
    # c.report('6', ('2', '2', '2')) alpha: str,
    #                            model_size: Tuple[str, str, str],
    #                            angle: str,
    #                            mode: str,
    #                            type_plot: str):
    # print(os.listdir('D:\\'))
