from typing import List, Union, Tuple, Any
from dataclasses import dataclass, field

import numpy as np
from kivy.uix.popup import Popup
from kivy.uix.label import Label
from kivy.metrics import dp
from kivymd.uix.menu import MDDropdownMenu
from kivymd.uix.screen import MDScreen

from utils import open_fig, get_model_and_scale_factors
from utils import interpolator as intp


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
    def step_integration(self):
        values = self.ids.step_integration.text
        if len(values) == 1:
            values = float(values),
            self.step_integration = values

        else:
            values = tuple(map(float, values.split()))
            self.step_integration = values

        return values

    @step_integration.setter
    def step_integration(self, value: tuple):
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
                "on_release": lambda x=mode: self.height_integration(x),
            } for mode in ('mean',
                           'min',
                           'max',
                           'std',
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
        ]
        for i in menus:
            i.dismiss()

    # Блок отрисовки графиков
    @check_parameters('model')
    def plot_spectrum(self, mode: str):
        fig = self.core_ws.get_plot_summary_spectres(self._alpha_ws, self.model_size_ws, self.angle_ws, mode)
        open_fig(fig)

    @check_parameters('model')
    def plot_envelopes(self):
        model_scale, _ = get_model_and_scale_factors(*self.model_size_ws, self.alpha_ws)
        figs = self.core_ws.get_envelopes(self.alpha_ws, model_scale, self.angle_ws)
        open_fig(figs)

    @check_parameters('all')
    def plot_isofields_pressure(self, mode: str):
        fig = self.core_ws.get_plot_isofields(self._alpha_ws,
                                              self.model_size_ws,
                                              self.angle_ws,
                                              mode,
                                              {'type_area': self.type_area_ws,
                                               'wind_region': self.wind_region_ws,
                                               })

        open_fig(fig)

    @check_parameters('model')
    def plot_isofields_coefficients(self, mode: str):
        fig = self.core_ws.get_plot_isofields(self._alpha_ws,
                                              self.model_size_ws,
                                              self.angle_ws,
                                              mode,
                                              None)

        open_fig(fig)

    @check_parameters('model')
    def plot_summary_coefficients(self, mode: str):
        fig = self.core_ws.get_plot_summary_coefficients(self._alpha_ws, self.model_size_ws, self.angle_ws, mode, )
        open_fig(fig)

    @check_parameters('model')
    def plot_polar_summary_coefficients(self, mode: str):
        model_scale, scale_factors = get_model_and_scale_factors(*self.model_size_ws, self._alpha_ws)
        if model_scale[0] == model_scale[1]:
            angle_border = 50
        else:
            angle_border = 95

        fig = self.core_ws.get_plot_summary_coefficients_polar(self._alpha_ws,
                                                               model_scale,
                                                               self.model_size_ws,
                                                               angle_border,
                                                               mode,
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
    def height_integration(self, mode):
        self.core_ws.height_integration(self._alpha_ws,
                                        self.model_size_ws,
                                        self.angle_ws,
                                        mode,
                                        {'type_area': self.type_area_ws,
                                         'wind_region': self.wind_region_ws,
                                         },
                                        self.face_integration,
                                        self.step_integration)
