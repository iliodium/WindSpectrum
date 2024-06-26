import os
from random import random
from typing import List, Union, Tuple, Any
from dataclasses import dataclass, field

import numpy as np
from kivy.uix.popup import Popup
from kivy.uix.label import Label
from kivy.metrics import dp
from kivymd.uix.menu import MDDropdownMenu
from kivymd.uix.screen import MDScreen
from matplotlib import pyplot as plt

from utils import open_fig, get_model_and_scale_factors, speed_sp_b, interp_025_tpu, speed_sp_a, interp_016_tpu


@dataclass
class IsolatedHighriseScreen(MDScreen):
    # приписка ws, чтобы не было конфликтов имен
    core_ws: Any = field(default=None, init=False)
    # Параметры модели
    _alpha_ws: str = field(default=None, init=False)
    angle_ws: str = field(default=None, init=False)
    _model_size_ws: Tuple[str] = field(default=None, init=False)
    # Ветровой район
    wind_region_ws: str = field(default=None, init=False)
    # Тип местности
    type_area_ws: str = field(default=None, init=False)
    # Параметры отображения изополей
    type_isofields_ws: str = field(default=None, init=False)
    # Выпадающие меню
    drop_down_menu_plots: MDDropdownMenu = field(default=None, init=False)
    drop_down_menu_mode_isofields: MDDropdownMenu = field(default=None, init=False)
    drop_down_menu_type_isofields: MDDropdownMenu = field(default=None, init=False)
    drop_down_menu_summary_coefficients: MDDropdownMenu = field(default=None, init=False)
    drop_down_menu_summary_spectrum: MDDropdownMenu = field(default=None, init=False)
    drop_down_menu_wind_regions: MDDropdownMenu = field(default=None, init=False)
    drop_down_menu_types_areas: MDDropdownMenu = field(default=None, init=False)
    drop_down_menu_mode_integration: MDDropdownMenu = field(default=None, init=False)
    # Настройки выпадающего меню
    drop_down_menu_plots_caller_ws: str = field(default=None, init=False)
    radius_ws: List[Union[int, float]] = field(default=list, init=False)
    width_mult_ws: [int, float] = field(default=4, init=False)
    max_height_ws: dp = field(default=dp(250), init=False)
    hor_growth_ws: str = field(default="right", init=False)
    ver_growth_ws: str = field(default="up", init=False)
    # Флаг инициализации выпадающих меню
    flag_init_drop_down_menus: bool = field(default=False, init=False)
    # Интегрирование по высоте
    _face_integration: Tuple[int] = field(default=None, init=False)
    _step_integration: Tuple[float] = field(default=None, init=False)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def on_pre_enter(self):
        if not self.flag_init_drop_down_menus:
            self.core_ws = self.manager.core
            self.radius_ws = [24, 0, 24, 0]
            self.drop_down_menu_plots_caller_ws = self.ids.plots
            self.init_drop_down_menus()
            self.flag_init_drop_down_menus = True

    def save_clipboard(self):
        self.core.save_clipboard()

    def report(self):
        if not self.update_model_parameters():
            return
        button = self.ids.report_button
        self.core.preparation_for_report(self._alpha_ws, self.model_size_ws, button)

    @staticmethod
    def popup(warning_text):
        Popup(
            title='Предупреждение',
            title_size=20,
            content=Label(text=warning_text, font_size=20),
            size_hint=(None, None),
            size=(350, 150),
            auto_dismiss=True
        ).open()

    # Проверки на правильность входных данных
    def check_parameters(parameter: str):
        """
        parameter =
        all проверить все параметры
        model проверить size, alpha, angle
        """

        def update_model_parameters(func):
            """Возвращает False если данные некорректны"""

            def wrapper_model(self, *args, **kwargs):
                self.dismiss_all_drop_down_menu()

                if parameter == 'all':
                    if all((
                            self.model_size_ws,
                            self.alpha_ws,
                            self.get_angle(),
                            self.get_type_area(),
                            self.get_wind_region(),
                    )):
                        func(self, *args, **kwargs)
                elif parameter == 'model':
                    if all((
                            self.model_size_ws,
                            self.alpha_ws,
                            self.get_angle(),
                    )):
                        func(self, *args, **kwargs)

            return wrapper_model

        return update_model_parameters

    # Блок геттеров
    @property
    def model_size_ws(self):
        size = self.ids.model_size.text
        model_size = []
        for i in size.split():
            try:
                var = str(float(i.replace(',', '.')))
            except ValueError:
                self.popup('Размеры здания некорректны')
                return False
            model_size.append(var)
        self.model_size_ws = tuple(model_size)
        return tuple(model_size)

    @model_size_ws.setter
    def model_size_ws(self, value):
        self._model_size_ws = value

    @property
    def alpha_ws(self):
        self.get_type_area()
        return self._alpha_ws
        # alpha = self.ids.alpha.text
        # if alpha not in ['4', '6']:
        #     self.popup('Параметр альфа 4 или 6')
        #     return False
        # else:
        #     self._alpha_ws = alpha
        #     return alpha

    @alpha_ws.setter
    def alpha_ws(self, value):
        self._alpha_ws = value

    def get_angle(self):
        angle = self.ids.angle.text
        if int(angle) % 5 != 0:
            self.popup('Углы должны быть кратными 5')
            return False
        else:
            self.angle_ws = str(int(angle) % 360)
            return True

    def get_type_area(self):
        type_area = self.ids.type_area_button.text
        if type_area == 'Выбор':
            self.popup('Выберите тип местности')
            return False
        else:
            self.type_area_ws = type_area
            if type_area == 'B':
                self.alpha_ws = '4'
            elif type_area == 'A':
                self.alpha_ws = '6'

            return type_area

    def get_wind_region(self):
        wind_region = self.ids.wind_region_button.text
        if wind_region == 'Выбор':
            self.popup('Выберите ветровой район')
            return False
        else:
            self.wind_region_ws = wind_region
            return wind_region

    @property
    def face_integration(self):
        values = self.ids.face_integration.text
        if len(values) == 1:
            values = int(values),
            self.face_integration = values

        else:
            values = tuple(map(int, values.split()))
            self.face_integration = values

        return values

    @face_integration.setter
    def face_integration(self, value: tuple):
        self._face_integration = value

    @property
    def parameters_integration(self):
        values = self.ids.parameters_integration.text
        if len(values) == 1:
            values = float(values),  # tuple
            self.parameters_integration = values

        else:
            values = tuple(map(float, values.split()))
            self.parameters_integration = values

        return values

    @parameters_integration.setter
    def parameters_integration(self, value: tuple):
        self._step_integration = value

    # Блок создания выпадающих меню
    def init_drop_down_menus(self):
        self.init_drop_down_menu_wind_regions()
        self.init_drop_down_menu_types_areas()
        self.init_drop_down_menu_summary_coefficients()
        self.init_drop_down_menu_mode_isofields()
        self.init_drop_down_menu_type_isofields()
        self.init_drop_down_menu_polar_summary_coefficients()
        self.init_drop_down_menu_mode_pseudocolor_coefficients()
        self.init_drop_down_menu_model_plots()
        self.init_drop_down_menu_summary_spectrum()
        self.init_drop_down_menu_mode_integration()
        self.init_drop_down_menu_mode_integration_spectre()

        self.init_drop_down_menu_plots()

    def init_drop_down_menu_wind_regions(self):
        regions = ('Iа', 'I', 'II', 'III', 'IV', 'V', 'VI', 'VII')

        items_drop_down_menu_wind_regions = [
            {
                "text": region,
                "viewclass": "OneLineListItem",
                "on_release": lambda x1=region: self.action_wind_region(x1),
            }
            for region in regions
        ]

        self.drop_down_menu_wind_regions = MDDropdownMenu(
            items=items_drop_down_menu_wind_regions,
            caller=self.ids.wind_region_button,
            radius=self.radius_ws,
            width_mult=self.width_mult_ws,
            max_height=self.max_height_ws,
            hor_growth=self.hor_growth_ws,
            ver_growth='down',
        )

    def init_drop_down_menu_types_areas(self):
        types_areas = ('A', 'B', 'C')

        items_types_areas = [
            {
                "text": type_area,
                "viewclass": "OneLineListItem",
                "on_release": lambda x=type_area: self.action_type_area(x),

            }
            for type_area in types_areas
        ]

        self.drop_down_menu_types_areas = MDDropdownMenu(
            items=items_types_areas,
            caller=self.ids.type_area_button,
            radius=self.radius_ws,
            width_mult=self.width_mult_ws,
            max_height=dp(150),
            hor_growth=self.hor_growth_ws,
            ver_growth='down',
        )

    def init_drop_down_menu_plots(self):
        items_plots_menu = [
            {
                "text": 'Изополя',
                "viewclass": "OneLineListItem",
                "on_release": self.drop_down_menu_type_isofields.open,
            },
            {
                "text": 'Огибающие',
                "viewclass": "OneLineListItem",
                "on_release": self.plot_envelopes,
            },
            {
                "text": 'Суммарные коэффициенты',
                "viewclass": "OneLineListItem",
                "on_release": self.drop_down_menu_summary_coefficients.open,
            },
            {
                "text": 'Спектры',
                "viewclass": "OneLineListItem",
                "on_release": self.drop_down_menu_summary_spectrum.open,
            },
            {
                "text": 'Модель',
                "viewclass": "OneLineListItem",
                "on_release": self.drop_down_menu_model_plots.open,
            },
        ]

        self.drop_down_menu_plots = MDDropdownMenu(
            items=items_plots_menu,
            caller=self.drop_down_menu_plots_caller_ws,
            radius=self.radius_ws,
            width_mult=self.width_mult_ws,
            max_height=self.max_height_ws,
            hor_growth=self.hor_growth_ws,
            ver_growth=self.ver_growth_ws,
        )

    def init_drop_down_menu_type_isofields(self):
        items_type_isofields = [
            {
                "text": i,
                "viewclass": "OneLineListItem",
                "on_release": lambda x=i: self.action_type_isofields(x),
            } for i in ('Давление',
                        'Коэффициенты',
                        )
        ]
        self.drop_down_menu_type_isofields = MDDropdownMenu(
            items=items_type_isofields,
            caller=self.drop_down_menu_plots_caller_ws,
            radius=self.radius_ws,
            width_mult=self.width_mult_ws,
            max_height=self.max_height_ws,
            hor_growth=self.hor_growth_ws,
            ver_growth=self.ver_growth_ws,
        )

    def init_drop_down_menu_mode_integration(self):
        items = [
            {
                "text": mode,
                "viewclass": "OneLineListItem",
                "on_release": lambda x=mode, y=ind, z=plot: self.height_integration(x, y, z),
            } for mode, ind, plot in (('по этажам', 0, 'summary'),
                                      ('по шагу', 1, 'summary'),
                                      ('по области', 2, 'summary'),
                                      ('по областям', 3, 'summary'),
                                      ('по количеству', 4, 'summary'),
                                      )
        ]
        self.drop_down_menu_mode_integration = MDDropdownMenu(
            items=items,
            caller=self.ids.height_integration,
            radius=self.radius_ws,
            width_mult=self.width_mult_ws,
            max_height=self.max_height_ws,
            hor_growth=self.hor_growth_ws,
            ver_growth=self.ver_growth_ws,

        )

    def init_drop_down_menu_mode_integration_spectre(self):
        items = [
            {
                "text": mode,
                "viewclass": "OneLineListItem",
                "on_release": lambda x=mode, y=ind, z=plot: self.height_integration(x, y, z),
            } for mode, ind, plot in (('по этажам', 0, 'spectre'),
                                      ('по шагу', 1, 'spectre'),
                                      ('по области', 2, 'spectre'),
                                      ('по областям', 3, 'spectre'),
                                      ('по количеству', 4, 'spectre'),
                                      )
        ]
        self.drop_down_menu_mode_integration_spectre = MDDropdownMenu(
            items=items,
            caller=self.ids.height_integration_spectre,
            radius=self.radius_ws,
            width_mult=self.width_mult_ws,
            max_height=self.max_height_ws,
            hor_growth=self.hor_growth_ws,
            ver_growth=self.ver_growth_ws,

        )

    def init_drop_down_menu_mode_isofields(self):
        items_mode_isofields = [
            {
                "text": mode,
                "viewclass": "OneLineListItem",
                "on_release": lambda x=mode: self.action_mode_isofields(x),
            } for mode in ('mean',
                           'min',
                           'max',
                           'std',
                           )
        ]
        self.drop_down_menu_mode_isofields = MDDropdownMenu(
            items=items_mode_isofields,
            caller=self.drop_down_menu_plots_caller_ws,
            radius=self.radius_ws,
            width_mult=self.width_mult_ws,
            max_height=self.max_height_ws,
            hor_growth=self.hor_growth_ws,
            ver_growth=self.ver_growth_ws,

        )

    def init_drop_down_menu_summary_coefficients(self):
        items_mode_summary_coefficients = [
            {
                "text": i,
                "viewclass": "OneLineListItem",
                "on_release": lambda x=i: self.action_summary_coefficients(x),
            } for i in ('Полярная система',
                        'Cx',
                        'Cy',
                        'CMz',
                        'Cx Cy',
                        'Cx CMz',
                        'Cy CMz',
                        'Cx Cy CMz',
                        )
        ]

        self.drop_down_menu_summary_coefficients = MDDropdownMenu(
            items=items_mode_summary_coefficients,
            caller=self.drop_down_menu_plots_caller_ws,
            radius=self.radius_ws,
            width_mult=self.width_mult_ws,
            max_height=self.max_height_ws,
            hor_growth=self.hor_growth_ws,
            ver_growth=self.ver_growth_ws,
        )

    def init_drop_down_menu_summary_spectrum(self):
        items_mode_summary_spectrum = [
            {
                "text": i,
                "viewclass": "OneLineListItem",
                "on_release": lambda x=i: self.plot_spectrum(x),
            } for i in ('Cx',
                        'Cy',
                        'CMz',
                        'Cx Cy',
                        'Cx CMz',
                        'Cy CMz',
                        'Cx Cy CMz',
                        )
        ]

        self.drop_down_menu_summary_spectrum = MDDropdownMenu(
            items=items_mode_summary_spectrum,
            caller=self.drop_down_menu_plots_caller_ws,
            radius=self.radius_ws,
            width_mult=self.width_mult_ws,
            max_height=self.max_height_ws,
            hor_growth=self.hor_growth_ws,
            ver_growth=self.ver_growth_ws,
        )

    def init_drop_down_menu_polar_summary_coefficients(self):
        items_mode_polar_summary_coefficients = [
            {
                "text": i,
                "viewclass": "OneLineListItem",
                "on_release": lambda x=i: self.plot_polar_summary_coefficients(x),
            } for i in ('mean',
                        'rms',
                        'std',
                        'max',
                        'min',
                        'Расчетное',
                        'Обеспеченность +',
                        'Обеспеченность -'
                        )
        ]

        self.drop_down_menu_polar_summary_coefficients = MDDropdownMenu(
            items=items_mode_polar_summary_coefficients,
            caller=self.drop_down_menu_plots_caller_ws,
            radius=self.radius_ws,
            width_mult=self.width_mult_ws,
            max_height=self.max_height_ws,
            hor_growth=self.hor_growth_ws,
            ver_growth=self.ver_growth_ws,
        )

    def init_drop_down_menu_mode_pseudocolor_coefficients(self):
        items_mode_pseudocolor_coefficients = [
            {
                "text": mode,
                "viewclass": "OneLineListItem",
                "on_release": lambda x=mode: self.action_mode_pseudocolor_coefficients(x),
            } for mode in ('mean',
                           'min',
                           'max',
                           'std',
                           )
        ]
        self.drop_down_menu_mode_pseudocolor_coefficients = MDDropdownMenu(
            items=items_mode_pseudocolor_coefficients,
            caller=self.drop_down_menu_plots_caller_ws,
            radius=self.radius_ws,
            width_mult=self.width_mult_ws,
            max_height=self.max_height_ws,
            hor_growth=self.hor_growth_ws,
            ver_growth=self.ver_growth_ws,

        )

    def init_drop_down_menu_model_plots(self):
        items_model_plots = [
            {
                "text": name,
                "viewclass": "OneLineListItem",
                "on_release": lambda x=func: self.action_func_and_dismiss(x),
            } for name, func in (('Модель трехмерная', self.plot_model_3d),
                                 ('Модель в полярной системе', self.plot_model_polar),
                                 ('Развертка модели', self.plot_pressure_tap_locations),
                                 )
        ]
        items_model_plots.append(
            {
                "text": 'Мозаика коэффициентов',
                "viewclass": "OneLineListItem",
                "on_release": self.drop_down_menu_mode_pseudocolor_coefficients.open,
            })

        self.drop_down_menu_model_plots = MDDropdownMenu(
            items=items_model_plots,
            caller=self.drop_down_menu_plots_caller_ws,
            radius=self.radius_ws,
            width_mult=self.width_mult_ws,
            max_height=self.max_height_ws,
            hor_growth=self.hor_growth_ws,
            ver_growth=self.ver_growth_ws,
        )

    # Блок действий для выпадающего меню
    def action_type_area(self, type_area):
        self.ids.type_area_button.text = type_area
        self.get_type_area()
        self.drop_down_menu_types_areas.dismiss()

    def action_wind_region(self, region):
        self.ids.wind_region_button.text = region
        self.get_wind_region()
        self.drop_down_menu_wind_regions.dismiss()

    def action_mode_pseudocolor_coefficients(self, mode):
        self.dismiss_all_drop_down_menu()
        self.mode_pseudocolor_coefficients = mode

    def action_type_isofields(self, type_isofields):
        self.type_isofields_ws = type_isofields
        self.drop_down_menu_mode_isofields.open()

    def action_mode_isofields(self, mode_isofields):
        if self.type_isofields_ws == 'Давление':
            self.plot_isofields_pressure(mode_isofields)

        elif self.type_isofields_ws == 'Коэффициенты':
            self.plot_isofields_coefficients(mode_isofields)

    def action_summary_coefficients(self, mode):
        if mode == 'Полярная система':
            self.drop_down_menu_polar_summary_coefficients.open()
        else:
            self.plot_summary_coefficients(mode)

    def action_func_and_dismiss(self, func):
        self.dismiss_all_drop_down_menu()
        func()

    def dismiss_all_drop_down_menu(self):
        menus = [
            self.drop_down_menu_plots,
            self.drop_down_menu_mode_isofields,
            self.drop_down_menu_type_isofields,
            self.drop_down_menu_summary_coefficients,
            self.drop_down_menu_summary_spectrum,
            self.drop_down_menu_model_plots,
            self.drop_down_menu_mode_pseudocolor_coefficients,
            self.drop_down_menu_mode_integration,
            self.drop_down_menu_mode_integration_spectre,
        ]
        for i in menus:
            i.dismiss()

    # Блок отрисовки графиков
    @check_parameters('model')
    def plot_spectrum(self, mode: str):
        fig = self.core_ws.get_plot_summary_spectres(db='isolated', alpha=self._alpha_ws, model_size=self.model_size_ws,
                                                     angle=self.angle_ws, mode=mode, scale=None, type_plot=None)
        open_fig(fig)

    @check_parameters('model')
    def plot_envelopes(self):
        model_scale, _ = get_model_and_scale_factors(*self.model_size_ws, self.alpha_ws)
        figs = self.core_ws.get_envelopes(db='isolated', alpha=self.alpha_ws, model_scale=model_scale,
                                          angle=self.angle_ws)
        open_fig(figs)

    @check_parameters('all')
    def plot_isofields_pressure(self, mode: str):
        fig = self.core_ws.get_plot_isofields(db='isolated',
                                              alpha=self._alpha_ws,
                                              model_size=self.model_size_ws,
                                              angle=self.angle_ws,
                                              mode=mode,
                                              pressure_plot_parameters=
                                              {'type_area': self.type_area_ws,
                                               'wind_region': self.wind_region_ws,
                                               })

        open_fig(fig)

    @check_parameters('model')
    def plot_isofields_coefficients(self, mode: str):
        fig = self.core_ws.get_plot_isofields(db='isolated',
                                              alpha=self._alpha_ws,
                                              model_size=self.model_size_ws,
                                              angle=self.angle_ws,
                                              mode=mode,
                                              pressure_plot_parameters=None)

        open_fig(fig)

    @check_parameters('model')
    def plot_summary_coefficients(self, mode: str):
        fig = self.core_ws.get_plot_summary_coefficients(db='isolated',
                                                         alpha=self.alpha_ws,
                                                         model_size=self.model_size_ws,
                                                         mode=mode,
                                                         angle=self.angle_ws,
                                                         )
        open_fig(fig)

    @check_parameters('model')
    def plot_polar_summary_coefficients(self, mode: str):
        model_scale, scale_factors = get_model_and_scale_factors(*self.model_size_ws, self._alpha_ws)
        if model_scale[0] == model_scale[1]:
            angle_border = 50
        else:
            angle_border = 95

        fig = self.core_ws.get_plot_summary_coefficients_polar(db='isolated',
                                                               alpha=self.alpha_ws,
                                                               model_scale=model_scale,
                                                               model_size=self.model_size_ws,
                                                               mode=mode,
                                                               angle_border=angle_border,
                                                               )
        open_fig(fig)

    @check_parameters('model')
    def plot_model_3d(self):
        fig = self.core_ws.get_plot_model_3d(self.model_size_ws)
        open_fig(fig)

    @check_parameters('model')
    def plot_pressure_tap_locations(self):
        fig = self.core_ws.get_plot_pressure_tap_locations(self.model_size_ws, self._alpha_ws)
        open_fig(fig)

    @check_parameters('model')
    def plot_pseudocolor_coefficient(self, mode):
        fig = self.core_ws.get_pseudocolor_coefficients(self._alpha_ws, self.model_size_ws, self.angle_ws, mode)
        open_fig(fig)

    @check_parameters('model')
    def plot_model_polar(self):
        fig = self.core_ws.get_model_polar(self.model_size_ws)
        open_fig(fig)

    # Интегрирование по высоте
    @check_parameters('all')
    def height_integration(self, mode, ind, plot):
        steps = []
        figs = []
        labels = []

        flag_save = self.ids.checkbox_integration_save_plots.active if plot == 'summary' else self.ids.checkbox_integration_spectre_save_plots.active
        flag_open = self.ids.checkbox_integration_open_plots.active if plot == 'summary' else self.ids.checkbox_integration_spectre_open_plots.active

        model_size = self.model_size_ws
        angle = self.angle_ws
        alpha = self._alpha_ws
        parameter = self.parameters_integration

        breadth, depth, height = model_size

        breadth = float(breadth)
        depth = float(depth)
        height = float(height)
        print(ind)
        if ind == 0:
            # figs, labels = self.core_ws.height_integration_cx_cy_cmz_floors(db='isolated',
            #                                                                 alpha=alpha,
            #                                                                 model_size=model_size,
            #                                                                 angle=angle,
            #                                                                 plot=plot
            #                                                                 )

            # data -> (cx, cy, cmz)
            # data = self.core_ws.get_height_integration_cx_cy_cmz_floors(db='isolated',
            #                                                             alpha=alpha,
            #                                                             model_size=model_size,
            #                                                             angle=angle,
            #                                                             plot=plot
            #                                                             )

            # TO TXT
            # self.core_ws.height_integration_cx_cy_cmz_floors_to_txt(db='isolated',
            #                                                         alpha=alpha,
            #                                                         model_size=model_size,
            #                                                         angle=angle,
            #                                                         )
            #
            for x_t, y_t, z_t, alp_t in zip(
                    ('0.1', '0.3', '0.1', '0.3', '0.2', '0.2'),
                    ('0.1', '0.1', '0.1', '0.1', '0.1', '0.1'),
                    ('0.5', '0.5', '0.5', '0.5', '0.4', '0.5'),
                    ('4', '4', '6', '6', '4', '4'),
            ):
                t_mode_size = x_t, y_t, z_t
                for t_angle in range(0, 105, 15):
                    self.core_ws.height_integration_cx_cy_cmz_floors_to_txt(db='isolated',
                                                                            alpha=alp_t,
                                                                            model_size=t_mode_size,
                                                                            angle=str(t_angle),
                                                                            )
                    print(t_mode_size, alp_t, t_angle)
                    # return

            # from openpyxl import Workbook, load_workbook
            # from openpyxl.styles import Alignment
            #
            # press_tap = [i for i in range(1, 21)]
            #
            # for mode in ('mean', 'std'):
            #     workbook = Workbook()
            #     sheet = workbook.active
            #     sheet.append(press_tap)
            #     for t_angle in range(0, 105, 15):
            #         sheet.append([t_angle])
            #
            #         data_new_out = self.core_ws.get_coeff_for_melbourne(db='isolated',
            #                                                             alpha=alpha,
            #                                                             model_size=model_size,
            #                                                             angle=str(t_angle),
            #                                                             mode=mode
            #                                                             )
            #         # sheet.append(data_new_out)
            #         # print(data_new_out[0])
            #         # print(data_new_out[0]+data_new_out[3]+data_new_out[2]+data_new_out[1])
            #         sheet.append(data_new_out[0] + data_new_out[3] + data_new_out[2] + data_new_out[1])
            #
            #     workbook.save(filename=f'veronika\\tpu_coeff_{mode}.xlsx')

            if plot == 'spectre':
                angle = int(angle)
                if alpha == '4':
                    speed_tpu = interp_025_tpu(height)

                elif alpha == '6':
                    speed_tpu = interp_016_tpu(height)

                l_m = breadth * np.cos(np.deg2rad(angle)) + depth * np.sin(np.deg2rad(angle))
                sh = lambda f: f * l_m / speed_tpu

                figs, labels = self.plot_integrated_summary_sh(sh, *data)
            else:
                figs, labels = self.plot_integrated_summary(*data)

        elif ind == 1:
            steps_temp = np.arange(0, height, parameter[0])
            steps_temp = np.append(steps_temp, height)

            for i in range(0, len(steps_temp) - 1):
                steps += (steps_temp[i], steps_temp[i + 1]),  # tuple

        elif ind == 2:
            steps += parameter,  # tuple

        elif ind == 3:
            for i in range(0, len(parameter), 2):
                steps += parameter[i:i + 2],  # tuple

        elif ind == 4:
            steps_temp = np.linspace(0, height, parameter)

            for i in range(0, len(steps_temp) - 1):
                steps += (steps_temp[i], steps_temp[i + 1]),  # tuple

        if ind != 0:
            print(steps)
            figs, labels = self.core_ws.height_integration_cx_cy_cmz(db='isolated',
                                                                     alpha=alpha,
                                                                     model_size=model_size,
                                                                     angle=angle,
                                                                     steps=steps,
                                                                     plot=plot
                                                                     )

        if flag_open:
            open_fig(figs)

        breadth = int(breadth) if breadth.is_integer() else f'{round(breadth, 2):.2f}'
        depth = int(depth) if depth.is_integer() else f'{round(depth, 2):.2f}'
        height = int(height) if height.is_integer() else f'{round(height, 2):.2f}'

        if flag_save:
            path = 'Интегрирование' if plot == 'summary' else 'Спектры'

            if not os.path.isdir(path):
                os.mkdir(path)

            path = f'{path}\\{breadth} {depth} {height} {angle} {mode}'

            if not os.path.isdir(path):
                os.mkdir(path)

            for fig, label in zip(figs, labels):
                fig.savefig(f'{path}\\{label}', bbox_inches='tight')

        for fig in figs:
            plt.close(fig)

        # figs = self.core_ws.sh_floors(db='isolated',
        #                               alpha=self._alpha_ws,
        #                               model_size=self.model_size_ws,
        #                               angle=self.angle_ws,
        #                               mode=mode,
        #                               faces=self.face_integration,
        #                               step=self.step_integration)

    @check_parameters('all')
    def run_FEA(self):
        count_floors = self.ids.FEA_count_floors.text
        try:
            count_floors = int(count_floors) + 1
        except Exception as e:
            self.popup('Неверное количество этажей для FEA')
            return

        steps = []

        model_size = self.model_size_ws
        angle = self.angle_ws
        alpha = self._alpha_ws

        _, _, height = model_size

        steps_temp = np.linspace(0, float(height), count_floors)

        for i in range(0, len(steps_temp) - 1):
            steps += (steps_temp[i], steps_temp[i + 1]),  # tuple

        #self.core_ws.FEA(model_size=model_size, angle=angle, alpha=alpha, steps=steps)

        figs,_ =  self.core_ws.FEA(model_size=model_size, angle=angle, alpha=alpha, steps=steps)
        open_fig(figs)
