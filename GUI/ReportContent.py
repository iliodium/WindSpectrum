from kivymd.uix.selectioncontrol.selectioncontrol import MDCheckbox
from kivymd.uix.label.label import MDLabel
from kivymd.uix.button.button import MDFlatButton

from kivymd.uix.gridlayout import MDGridLayout
from kivymd.uix.screen import MDScreen

from dataclasses import dataclass, field
from typing import Any

from utils.utils import id_to_name


@dataclass
class ReportContent(MDScreen):
    # приписка ws, чтобы не было конфликтов имен
    core_ws: Any = field(default=None, init=False)
    report_content: dict = field(default=dict, init=False)
    # Флаг инициализации меню
    flag_init_menu: bool = field(default=False, init=False)
    # Словарь с чекбоксами id_checkbox : checkbox
    checkboxes_ws: dict = field(default=dict, init=False)
    # Флаг выбора всего содержимого
    flag_all_content: bool = field(default=False, init=False)

    """
    id = раздел
    или
    id = раздел_параметр
    """
    size_checkbox_x = '24dp'
    size_checkbox_y = '24dp'
    font_size = 15

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        mode_4 = {
            'max': False,
            'mean': False,
            'min': False,
            'std': False,
        }
        mode_5 = mode_4.copy()
        mode_5['rms'] = False

        mode_8 = mode_5.copy()
        mode_8['rach'] = False
        mode_8['obesP'] = False
        mode_8['obesM'] = False

        mode_coefficients = {
            'cx': False,
            'cy': False,
            'cmz': False,
        }
        mode_coefficients_all = mode_coefficients.copy()
        mode_coefficients_all.update({
            'cx_cy': False,
            'cx_cmz': False,
            'cy_cmz': False,
            'cx_cy_cmz': False,
        })

        sensors = {
            'x': False,
            'y': False,
            'z': False,
        }
        sensors.update(mode_8)

        self.report_content = {
            'isofieldsPressure': [False, mode_4.copy()],
            'isofieldsCoefficients': [False, mode_4.copy()],
            'pseudocolorCoefficients': [False, mode_4.copy()],
            'envelopes': [False, mode_5.copy()],
            'polarSummaryCoefficients': [False, mode_8.copy()],
            'summaryCoefficients': [False, mode_coefficients_all.copy()],
            'summarySpectres': [False, mode_coefficients_all.copy()],
            'pressureTapLocations': [False],
            'statisticsSensors': [False, sensors.copy()],
            'statisticsSummaryCoefficients': [False, mode_8.copy()],
        }

    def on_pre_enter(self):
        if not self.flag_init_menu:
            self.core_ws = self.manager.core
            self.init_report_content_menu()
            self.flag_init_menu = True

    def init_report_content_menu(self):
        self.checkboxes_ws = dict()
        for section in self.report_content.keys():
            checkbox = MDCheckbox(
                id=section,
                size_hint=(None, None),
                size=(ReportContent.size_checkbox_x, ReportContent.size_checkbox_y),
            )
            checkbox.bind(active=self.on_checkbox_active)
            self.ids.menu_layout.add_widget(checkbox)
            self.ids.menu_layout.add_widget(MDLabel(
                adaptive_height=True,
                font_size=ReportContent.font_size,
                text=id_to_name[section],
                size_hint_x=None,
                width=500,
            ))
            self.checkboxes_ws[section] = checkbox
            if len(self.report_content[section]) == 2:
                gridlayout = MDGridLayout(
                    cols=2,
                    padding=[20, 0, 0, 0],
                    adaptive_height=True,
                )
                self.ids.menu_layout.add_widget(gridlayout)

                for parameter in self.report_content[section][1].keys():
                    checkbox = MDCheckbox(
                        id=f'{section}_{parameter}',
                        size_hint=(None, None),
                        size=(ReportContent.size_checkbox_x, ReportContent.size_checkbox_y),
                    )
                    checkbox.bind(active=self.on_checkbox_active)
                    gridlayout.add_widget(checkbox)
                    gridlayout.add_widget(MDLabel(
                        adaptive_width=True,
                        font_size=ReportContent.font_size,
                        text=id_to_name[parameter],
                        size_hint_x=None,
                        width=300,
                    ))
                    self.checkboxes_ws[f'{section}_{parameter}'] = checkbox

                self.ids.menu_layout.add_widget(MDFlatButton(disabled=True))

    def brute_force_sections(self, value):
        for section in self.report_content.keys():
            self.report_content[section][0] = value
            self.checkboxes_ws[section].active = value

    def brute_force_parameters(self, section, value):
        if len(self.report_content[section]) == 2:
            for parameter in self.report_content[section][1].keys():
                self.report_content[section][1][parameter] = value
                self.checkboxes_ws[f'{section}_{parameter}'].active = value
        else:
            self.report_content[section] = [value]

    def on_checkbox_active(self, checkbox, value):
        id_cb = checkbox.id

        section = id_cb[:id_cb.find('_')] if '_' in id_cb else id_cb
        parameter = id_cb[id_cb.find('_') + 1:] if '_' in id_cb else None

        if parameter is None:
            self.report_content[section][0] = value
            self.brute_force_parameters(section, value)
        else:
            self.report_content[section][1][parameter] = value

    def chose_all_content(self):
        self.brute_force_sections(not self.flag_all_content)
        self.flag_all_content = not self.flag_all_content

    def report(self):
        ish = self.manager.get_screen('IsolatedHighriseScreen_screen')
        alpha = ish.alpha_ws
        model_size = ish.model_size_ws

        pressure_plot_parameters = {'type_area': ish.get_type_area(),
                                    'wind_region': ish.get_wind_region() if self.report_content['isofieldsPressure'][0] else None,
                                    }
        if False in pressure_plot_parameters.values():
            return

        button = self.ids.report_button
        spinner = ish.ids.report_spinner

        button.disabled = True
        spinner.active = True

        self.core_ws.preparation_for_report(alpha,
                                            model_size,
                                            pressure_plot_parameters,
                                            self.report_content,
                                            button,
                                            spinner)
