# coding:utf-8
from abc import abstractmethod

import numpy as np
import scipy
from PySide6 import QtGui, QtCore
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QWidget, QHBoxLayout, QStackedLayout, QVBoxLayout
from qfluentwidgets import PushButton, TitleLabel, ComboBox, \
    StrongBodyLabel, LineEdit

from compiled_functions import aot_calculations
from src.common.annotation import ModelSizeType
from src.common.constants import wind_regions, alpha_standards, Uz_a_0_16_z, Uz_a_0_16_x, Uz_a_0_25_x, Uz_a_0_25_z
from src.submodules.plot.plotBuilding import PlotBuilding
from src.submodules.plot.utils import scaling_data
from src.submodules.utils.data_features import polar_lambdas
from src.submodules.utils.scaling import get_model_and_scale_factors
from src.submodules.utils.speed_sp import speed_sp_region
from src.ui.common.Buttons import Buttons
from src.ui.common.CartesianModelSummaryCoefficients import CartesianModelSummaryCoefficients
from src.ui.common.ChartMode import ChartMode
from src.ui.common.ChartType import ChartType
from src.ui.common.CoordinateSystem import CoordinateSystem
from src.ui.common.IsofieldsType import IsofieldsType
from src.ui.common.StyleSheet import StyleSheet
from src.ui.components.MultiSelectComboBox import MultiSelectComboBox
from src.ui.view.widgets.MatplotlibWidget import MatplotlibWidget
from src.ui.view.widgets.SensorWidget import SensorWidget


class Interface(QWidget):
    """Interface"""

    SAMPLE_PERIOD = 7.5
    SAMPLE_FREQUENCY = 781
    NUMBER_OF_TIME_COUNTS = 5858

    def __init__(
            self,
            parent=None,
            engine=None
    ):
        super().__init__(parent=parent)
        self.engine = engine
        self.view = self
        self.plotFlag = False

        # Set style
        StyleSheet.MAIN_INTERFACE.apply(self)

        self.hBoxLayoutMain = QHBoxLayout(self)

        # Initialization left menu
        self._init_general_information()

        self.StackedLayoutMainMenu = QStackedLayout()
        self.hBoxLayoutMain.addLayout(self.StackedLayoutMainMenu)

        # Initialization chart menu
        WidgetChartMenu = QWidget()
        self.vBoxLayoutPlot = QVBoxLayout(WidgetChartMenu)
        self.vBoxLayoutPlot.setAlignment(Qt.AlignTop | Qt.AlignLeft)

        self.StackedLayoutMainMenu.addWidget(WidgetChartMenu)

        self._init_chart_menu()

        WidgetSensorsOverview = QWidget()
        self.vBoxLayoutSensorsOverview = QVBoxLayout(WidgetSensorsOverview)
        self.vBoxLayoutSensorsOverview.setAlignment(Qt.AlignTop | Qt.AlignLeft)

        self.StackedLayoutMainMenu.addWidget(WidgetSensorsOverview)

        self._init_sensors_overview()

        # An array to store references to objects (like envelopes)
        # If you do not store the objects, they will be deleted by garbage collector
        self.plots = []

        self.fig_spectrum_sensors_overview = None
        self.fig_summary_coefficients_sensors_overview = None

    def _switch_stacked_layout_sensors_overview(
            self
    ):
        current_index = self.StackedLayoutMainMenu.currentIndex()
        new_index = 1 if current_index == 0 else 0
        if new_index == 1:
            self.PushButtonSensorsOverview.setText(Buttons.PLOTS)
            model_size = self._get_model_size()

            if model_size != self.SensorWidget.model_size:
                self.SensorWidget.model_size = model_size

                breadth, depth, _ = model_size
                length = 2 * (breadth + depth)

                lines_pos = [i / length for i in (breadth, breadth + depth, 2 * breadth + depth)]

                self.SensorWidget.lines_pos = lines_pos
                self.SensorWidget.update_lines()

                alpha = self._get_alpha()
                model_name, _ = get_model_and_scale_factors(*self._get_model_size(), alpha)
                model_name_str = str(model_name)
                model_id = self.get_model_id(model_name, alpha)
                coordinates = self._get_coordinates(model_id, alpha)

                length_x = 2 * (int(model_name_str[0]) + int(model_name_str[1])) / 10
                length_y = int(model_name_str[2]) / 10

                x = [i / length_x for i in coordinates[0]]
                y = [i / length_y for i in coordinates[1]]

                for b in self.SensorWidget.buttons:
                    self.SensorWidget.scene.removeItem(b)
                    b.deleteLater()  # Уничтожаем объект

                self.SensorWidget.buttons = []

                self.SensorWidget.buttons_pos = [(i, j) for i, j in zip(x, y)]
                self.SensorWidget.add_buttons()

            else:
                pass
        else:
            self.PushButtonSensorsOverview.setText(Buttons.SENSORS)

        self.StackedLayoutMainMenu.setCurrentIndex(new_index)

    def sensors_overview_button_action(self, sensor_id):
        alpha = self._get_alpha()
        model_size = self._get_model_size()
        model_name, _ = get_model_and_scale_factors(*model_size, alpha)
        angle = self._get_angle()

        model_id = self.get_model_id(model_name, alpha)

        pressure_coefficients = self.get_pressure_coefficients_for_the_sensor(model_id, alpha, angle, str(model_name),
                                                                              sensor_id)
        type_plot = ChartType(self.ComboBoxSensorsOverview.currentText())
        match type_plot:
            case ChartType.SUMMARY_COEFFICIENTS:
                if self.fig_summary_coefficients_sensors_overview is None:
                    fig = PlotBuilding.summary_coefficients({f"Датчик {sensor_id + 1}": pressure_coefficients})
                else:
                    data_to_plot = {f"Датчик {sensor_id + 1}": pressure_coefficients}
                    # Перенос данных из первой фигуры
                    for ax in self.fig_summary_coefficients_sensors_overview.get_axes():
                        for line in ax.get_lines():
                            data_to_plot[line.get_label()] = line.get_ydata()
                    self.windows_plot_sensors_overview_summary_coefficients.close()

                    fig = PlotBuilding.summary_coefficients(data_to_plot)

                self.fig_summary_coefficients_sensors_overview = fig
                self.open_plot_in_new_window_sensors_overview_summary_coefficients(fig, ChartType.SUMMARY_COEFFICIENTS)
            case ChartType.SPECTRUM:
                height = model_size[2]
                alpha_str = self._get_alpha(area_type=True)
                wind_region = self._get_wind_region()

                speed_sp = speed_sp_region(height, alpha_str, wind_region)

                if self.fig_spectrum_sensors_overview is None:
                    fig = PlotBuilding.welch_graph({f"Датчик {sensor_id + 1}": pressure_coefficients}, height, speed_sp)
                else:
                    data_to_plot = {f"Датчик {sensor_id + 1}": pressure_coefficients}
                    # Перенос данных из первой фигуры
                    for ax in self.fig_spectrum_sensors_overview.get_axes():
                        for line in ax.get_lines():
                            label = line.get_label()
                            last_space_index = label.rfind(" ")
                            sensor_id = int(label[last_space_index + 1:]) - 1
                            data_to_plot[label] = self.get_pressure_coefficients_for_the_sensor(model_id, alpha, angle,
                                                                                                str(model_name),
                                                                                                sensor_id)
                    self.windows_plot_sensors_overview_summary_coefficients.close()

                    fig = PlotBuilding.welch_graph(data_to_plot, height, speed_sp)

                self.fig_spectrum_sensors_overview = fig
                self.open_plot_in_new_window_sensors_overview_summary_coefficients(fig, ChartType.SPECTRUM)

    def _init_general_information(
            self
    ):
        self.generalInformationContainer = QWidget()
        self.generalInformationContainer.setFixedWidth(275)
        self.generalInformationContainer.setFixedHeight(300)

        self.vBoxLayoutGenInf = QVBoxLayout(self.generalInformationContainer)

        # Add label to grid layout
        self.vBoxLayoutGenInf.addWidget(TitleLabel(Buttons.GENERAL_INFORMATION))

        # Wind regions
        # Create horizontal box layout
        self.hBoxLayoutWindRegions = QHBoxLayout()
        # Add label to horizontal box layout
        self.hBoxLayoutWindRegions.addWidget(StrongBodyLabel(Buttons.WIND_REGION))
        # Create combo box
        self.ComboBoxWindRegions = ComboBox()
        # Fill the combo box
        self.ComboBoxWindRegions.addItems([
            self.tr(i) for i in [*wind_regions]
        ])
        # set fixed width of combobox
        self.ComboBoxWindRegions.setFixedWidth(75)
        # Add combo box to horizontal box layout
        self.hBoxLayoutWindRegions.addWidget(self.ComboBoxWindRegions)
        self.vBoxLayoutGenInf.addLayout(self.hBoxLayoutWindRegions)

        # Type of area
        self.hBoxLayoutTypeOfArea = QHBoxLayout(self.view)
        self.hBoxLayoutTypeOfArea.addWidget(StrongBodyLabel(Buttons.TYPE_OF_AREA))
        self.ComboBoxTypeOfArea = ComboBox()
        self.ComboBoxTypeOfArea.addItems([
            self.tr(i) for i in [*alpha_standards]
        ])
        self.ComboBoxTypeOfArea.setFixedWidth(75)
        self.hBoxLayoutTypeOfArea.addWidget(self.ComboBoxTypeOfArea)
        self.vBoxLayoutGenInf.addLayout(self.hBoxLayoutTypeOfArea)

        # Wind angle
        self.hBoxLayoutWindAngle = QHBoxLayout(self.view)
        self.hBoxLayoutWindAngle.addWidget(StrongBodyLabel(Buttons.WIND_ANGLE))
        # Create text input widget
        self.lineEditWindAngle = LineEdit()
        # Set default text
        self.lineEditWindAngle.setText(self.tr('0'))
        # Set clear button
        self.lineEditWindAngle.setClearButtonEnabled(True)
        self.lineEditWindAngle.setFixedWidth(75)
        # Add text input widget to horizontal box layout
        self.hBoxLayoutWindAngle.addWidget(self.lineEditWindAngle)
        self.vBoxLayoutGenInf.addLayout(self.hBoxLayoutWindAngle)

        # Building size
        self.hBoxLayoutBuildingSizeInterfering = QHBoxLayout(self.view)
        self.hBoxLayoutBuildingSizeInterfering.addWidget(StrongBodyLabel(Buttons.BUILDING_SIZE))
        self.lineEditBuildingSize = LineEdit()
        self.lineEditBuildingSize.setText(self.tr('10 10 20'))
        self.lineEditBuildingSize.setClearButtonEnabled(True)
        self.lineEditBuildingSize.setFixedWidth(125)
        self.hBoxLayoutBuildingSizeInterfering.addWidget(self.lineEditBuildingSize)
        self.vBoxLayoutGenInf.addLayout(self.hBoxLayoutBuildingSizeInterfering)

        PushButtonReport = PushButton(Buttons.REPORT)
        PushButtonReport.clicked.connect(self.create_report)
        self.vBoxLayoutGenInf.addWidget(PushButtonReport)

        self.PushButtonSensorsOverview = PushButton(Buttons.SENSORS)
        self.PushButtonSensorsOverview.clicked.connect(self._switch_stacked_layout_sensors_overview)
        self.vBoxLayoutGenInf.addWidget(self.PushButtonSensorsOverview)

        self.PushButtonInterferingInformation = PushButton(Buttons.FEA)
        # self.PushButtonFiniteElementMethod.clicked.connect(self._switch_stacked_layout_sensors_overview)
        self.vBoxLayoutGenInf.addWidget(self.PushButtonInterferingInformation)

        self.hBoxLayoutMain.addWidget(self.generalInformationContainer)

    def _init_chart_menu(
            self
    ):
        container = QWidget()
        container.setFixedWidth(775)
        container.setFixedHeight(100)

        # Chart menu
        hBoxLayoutChartMenu = QHBoxLayout(container)

        self.ComboBoxChartMenu = ComboBox()
        self.ComboBoxChartMenu.addItems([
            self.tr(i) for i in ChartType
        ])
        self.ComboBoxChartMenu.currentTextChanged.connect(self._switch_stacked_layout_type_chart)
        self.ComboBoxChartMenu.setFixedWidth(225)
        hBoxLayoutChartMenu.addWidget(self.ComboBoxChartMenu)

        self.StackedLayoutTypeChart = QStackedLayout()

        self._init_chart_isofields()
        self._init_chart_envelopes()
        self._init_chart_summary_coefficients()
        self._init_chart_spectrum()
        self._init_chart_pseudocolor_coefficients()

        hBoxLayoutChartMenu.addLayout(self.StackedLayoutTypeChart)

        PushButtonCreatePlot = PushButton(Buttons.BUILD_PLOT)
        PushButtonCreatePlot.clicked.connect(self.create_plot)
        PushButtonCreatePlot.setFixedWidth(100)

        hBoxLayoutChartMenu.addWidget(PushButtonCreatePlot)
        self.vBoxLayoutPlot.addWidget(container)

    def _init_sensors_overview(self):
        container = QWidget()
        container.setFixedWidth(775)
        container.setFixedHeight(100)

        # Chart menu
        hBoxLayoutChartMenu = QHBoxLayout(container)
        hBoxLayoutChartMenu.setAlignment(Qt.AlignLeft)
        self.ComboBoxSensorsOverview = ComboBox()
        self.ComboBoxSensorsOverview.addItems([
            self.tr(i) for i in (ChartType.SUMMARY_COEFFICIENTS, ChartType.SPECTRUM)
        ])
        self.ComboBoxSensorsOverview.setFixedWidth(350)
        hBoxLayoutChartMenu.addWidget(self.ComboBoxSensorsOverview)
        self.vBoxLayoutSensorsOverview.addWidget(container)

        self.SensorWidget = SensorWidget(self.sensors_overview_button_action)
        self.vBoxLayoutSensorsOverview.addWidget(self.SensorWidget.view)

    def create_plot(
            self
    ):
        match self.StackedLayoutTypeChart.currentIndex():
            case 0:
                self.plot_isofields()
            case 1:
                self.plot_envelopes()
            case 2:
                self.plot_summary_coefficients()
            case 3:
                self.plot_welch_graph()
            case 4:
                self.plot_pseudocolor_coefficients()

    def _init_chart_isofields(
            self
    ):
        self.WidgetIsofields = QWidget()
        self.hBoxLayoutIsofields = QHBoxLayout(self.WidgetIsofields)

        self.ComboBoxTypesIsofields = ComboBox()
        self.ComboBoxTypesIsofields.addItems([
            self.tr(i) for i in (IsofieldsType.PRESSURE,
                                 IsofieldsType.COEFFICIENT,
                                 )
        ])
        self.ComboBoxTypesIsofields.setFixedWidth(140)
        self.hBoxLayoutIsofields.addWidget(self.ComboBoxTypesIsofields)

        self.isofieldsParameters = ComboBox()
        self.isofieldsParameters.addItems([
            self.tr(i) for i in (ChartMode.MAX,
                                 ChartMode.MEAN,
                                 ChartMode.MIN,
                                 ChartMode.RMS,
                                 ChartMode.STD,
                                 )])
        self.hBoxLayoutIsofields.addWidget(self.isofieldsParameters)

        self.StackedLayoutTypeChart.addWidget(self.WidgetIsofields)

    def _init_chart_envelopes(
            self
    ):
        self.WidgetEnvelopes = QWidget()
        self.hBoxLayoutEnvelopes = QHBoxLayout(self.WidgetEnvelopes)
        self.envelopesParameters = MultiSelectComboBox(placeholderText=Buttons.PARAMETERS)
        self.envelopesParameters.addItems([ChartMode.MAX,
                                           ChartMode.MEAN,
                                           ChartMode.MIN,
                                           ChartMode.RMS,
                                           ChartMode.STD,
                                           ])
        # self.envelopesParameters.checkAllItems()
        self.hBoxLayoutEnvelopes.addWidget(self.envelopesParameters)
        self.StackedLayoutTypeChart.addWidget(self.WidgetEnvelopes)

    def _init_chart_summary_coefficients(
            self
    ):
        self.WidgetSummaryCoefficients = QWidget()
        self.hBoxLayoutSummaryCoefficients = QHBoxLayout(self.WidgetSummaryCoefficients)

        # Combo box with coordinate system
        self.ComboBoxCoordinateSystemSummaryCoefficients = ComboBox()
        self.ComboBoxCoordinateSystemSummaryCoefficients.addItems([
            self.tr(i) for i in (CoordinateSystem.CARTESIAN,
                                 CoordinateSystem.POLAR,
                                 )
        ])
        self.ComboBoxCoordinateSystemSummaryCoefficients.currentTextChanged.connect(
            self._switch_stacked_layout_summary_coefficients)
        self.ComboBoxCoordinateSystemSummaryCoefficients.setFixedWidth(140)
        self.hBoxLayoutSummaryCoefficients.addWidget(self.ComboBoxCoordinateSystemSummaryCoefficients)
        # Stacked layout which varies depending on the coordinate system
        self.StackedLayoutSummaryCoefficients = QStackedLayout()
        self.hBoxLayoutSummaryCoefficients.addLayout(self.StackedLayoutSummaryCoefficients)
        # Cartesian system
        self.WidgetCartesianSummaryCoefficients = QWidget()
        self.hBoxLayoutCartesianCoordinateSystem = QHBoxLayout(self.WidgetCartesianSummaryCoefficients)
        self.cartesianModelSummaryCoefficients = ComboBox()
        self.cartesianModelSummaryCoefficients.addItems([
            self.tr(i) for i in (CartesianModelSummaryCoefficients.REAL,
                                 CartesianModelSummaryCoefficients.TPU,
                                 )
        ])

        self.cartesianParameters = MultiSelectComboBox(placeholderText=Buttons.PARAMETERS)
        self.cartesianParameters.addItems([ChartMode.CX,
                                           ChartMode.CY,
                                           ChartMode.CMZ,
                                           ])
        self.hBoxLayoutCartesianCoordinateSystem.addWidget(self.cartesianModelSummaryCoefficients)
        self.hBoxLayoutCartesianCoordinateSystem.addWidget(self.cartesianParameters)
        self.StackedLayoutSummaryCoefficients.addWidget(self.WidgetCartesianSummaryCoefficients)
        # Polar system
        self.WidgetPolarSummaryCoefficients = QWidget()
        self.hBoxLayoutPolarCoordinateSystem = QHBoxLayout(self.WidgetPolarSummaryCoefficients)
        self.polarView = MultiSelectComboBox(placeholderText=Buttons.VIEW)
        self.polarView.addItems([ChartMode.CX,
                                 ChartMode.CY,
                                 ChartMode.CMZ,
                                 ])
        self.polarParameters = MultiSelectComboBox(placeholderText=Buttons.PARAMETERS)
        self.polarParameters.addItems([ChartMode.MAX,
                                       ChartMode.MEAN,
                                       ChartMode.MIN,
                                       ChartMode.RMS,
                                       ChartMode.STD,
                                       ChartMode.CALCULATED,
                                       ChartMode.WARRANTY_PLUS,
                                       ChartMode.WARRANTY_MINUS,
                                       ])
        self.hBoxLayoutPolarCoordinateSystem.addWidget(self.polarView)
        self.hBoxLayoutPolarCoordinateSystem.addWidget(self.polarParameters)
        self.StackedLayoutSummaryCoefficients.addWidget(self.WidgetPolarSummaryCoefficients)
        # Add widget to main layout
        self.StackedLayoutTypeChart.addWidget(self.WidgetSummaryCoefficients)

    def _init_chart_spectrum(
            self
    ):
        self.WidgetSpectrum = QWidget()
        self.hBoxLayoutSpectrum = QHBoxLayout(self.WidgetSpectrum)
        self.spectrumParameters = MultiSelectComboBox(placeholderText=Buttons.PARAMETERS)
        self.spectrumParameters.addItems([ChartMode.CX,
                                          ChartMode.CY,
                                          ChartMode.CMZ,
                                          ])
        self.hBoxLayoutSpectrum.addWidget(self.spectrumParameters)
        self.StackedLayoutTypeChart.addWidget(self.WidgetSpectrum)

    def _init_chart_pseudocolor_coefficients(
            self
    ):
        self.WidgetDiscreteIsofields = QWidget()
        self.hBoxLayoutDiscreteIsofields = QHBoxLayout(self.WidgetDiscreteIsofields)
        self.discreteIsofieldsParameters = ComboBox()
        self.discreteIsofieldsParameters.addItems([
            self.tr(i) for i in (ChartMode.MAX,
                                 ChartMode.MEAN,
                                 ChartMode.MIN,
                                 ChartMode.RMS,
                                 ChartMode.STD)
        ])
        self.hBoxLayoutDiscreteIsofields.addWidget(self.discreteIsofieldsParameters)
        self.StackedLayoutTypeChart.addWidget(self.WidgetDiscreteIsofields)

    def _switch_stacked_layout_type_chart(
            self,
            chart_type
    ):
        match chart_type:
            case ChartType.ISOFIELDS:
                self.StackedLayoutTypeChart.setCurrentIndex(0)
            case ChartType.ENVELOPES:
                self.StackedLayoutTypeChart.setCurrentIndex(1)
            case ChartType.SUMMARY_COEFFICIENTS:
                self.StackedLayoutTypeChart.setCurrentIndex(2)
            case ChartType.SPECTRUM:
                self.StackedLayoutTypeChart.setCurrentIndex(3)
            case ChartType.DISCRETE_ISOFIELDS:
                self.StackedLayoutTypeChart.setCurrentIndex(4)

    def _switch_stacked_layout_summary_coefficients(
            self,
            system
    ):
        match system:
            case CoordinateSystem.CARTESIAN:
                self.StackedLayoutSummaryCoefficients.setCurrentIndex(0)
            case CoordinateSystem.POLAR:
                self.StackedLayoutSummaryCoefficients.setCurrentIndex(1)

    def _get_model_size(
            self
    ) -> ModelSizeType:
        return tuple(map(float, self.lineEditBuildingSize.text().replace(',', '.').split(' ')))

    def _get_alpha(
            self,
            area_type=False
    ):
        if area_type:
            return self.ComboBoxTypeOfArea.text()

        else:
            type_alpha = {
                'A': 4,
                'C': 6,
            }
            return type_alpha[self.ComboBoxTypeOfArea.text()]

    def _get_wind_region(
            self
    ):

        return self.ComboBoxWindRegions.text()

    def _get_angle(
            self
    ):
        """получаем угол и возвращаем ближайшее кратное число на 5"""
        angle = int(self.lineEditWindAngle.text()) % 360

        return 5 * round(angle / 5)

    def _icon(
            self,
            path
    ):
        """
        link to the original ->
        matplotlib.backends.backend_qt.NavigationToolbar2QT._icon
        """
        pm = QtGui.QPixmap(path)
        pm.setDevicePixelRatio(
            self.devicePixelRatioF() or 1)  # rarely, devicePixelRatioF=0
        if self.palette().color(self.backgroundRole()).value() < 128:
            icon_color = self.palette().color(self.foregroundRole())
            mask = pm.createMaskFromColor(
                QtGui.QColor('black'),
                QtCore.Qt.MaskMode.MaskOutColor)
            pm.fill(icon_color)
            pm.setMask(mask)
        return QtGui.QIcon(pm)

    def open_in_new_window(
            self
    ):
        window = QWidget()
        window.setWindowTitle(self.plotWidget.plotTitle)
        self.plots.append(window)
        layout = QVBoxLayout(window)

        layout.addWidget(self.plotWidget.toolbar)
        layout.addWidget(self.plotWidget)

        self.plotFlag = False

        window.show()

    def del_plot(self):
        self.vBoxLayoutPlot.removeWidget(self.containerPlot)
        self.plotWidget.setParent(None)  # Убираем из иерархии
        self.plotWidget.toolbar.deleteLater()  # Уничтожаем объект
        self.plotWidget.deleteLater()  # Уничтожаем объект
        self.containerPlot = None

    def add_plot_on_screen(
            self,
            fig,
            title='График',
            open_in_new_window_button=True
    ):
        if self.plotFlag:
            self.del_plot()

        self.containerPlot = QWidget()
        self.containerPlot.setStyleSheet("border: 0px;")
        self.vBoxLayoutPlot1 = QVBoxLayout(self.containerPlot)

        fig.patch.set_facecolor('#f7f9fc')

        # Create plot widget
        self.plotWidget = MatplotlibWidget(fig, title)

        if open_in_new_window_button:
            path = r"src\ui\resource\images\fl_icon.png"
            act = self.plotWidget.toolbar.addAction(self._icon(path), 'Открыть в новом окне', self.open_in_new_window)
            # if you set the value to -1, the button will be in the rightmost position
            self.plotWidget.toolbar.insertAction(self.plotWidget.toolbar.actions()[-2], act)

        self.vBoxLayoutPlot1.addWidget(self.plotWidget.toolbar)
        self.vBoxLayoutPlot1.addWidget(self.plotWidget)
        self.vBoxLayoutPlot1.setAlignment(Qt.AlignTop | Qt.AlignLeft)

        self.vBoxLayoutPlot.addWidget(self.containerPlot)

        self.plotFlag = True

    def plot_welch_graph(
            self
    ):

        parameters = [ChartMode(i) for i in self.spectrumParameters.getCurrentOptions()]
        if not parameters:
            return

        data_to_plot = {}

        coordinates = self._get_coordinates()
        angle = self._get_angle()
        pressure_coefficients = self._get_pressure_coefficients()

        size_tpu, count_sensors = self._get_size_and_count_sensors(len(coordinates[0]))

        model_size = self._get_model_size()
        height = model_size[2]

        if ChartMode.CX in parameters or ChartMode.CY in parameters:
            cx, cy = aot_calculations.calculate_cx_cy(
                *count_sensors,
                *size_tpu,
                np.array(coordinates[0]),
                np.array(coordinates[1]),
                pressure_coefficients
            )
            if ChartMode.CX in parameters:
                data_to_plot[ChartMode.CX] = cx
            if ChartMode.CY in parameters:
                data_to_plot[ChartMode.CY] = cy

        if ChartMode.CMZ in parameters:
            cmz = aot_calculations.calculate_cmz(
                *count_sensors,
                angle,
                *size_tpu,
                np.array(coordinates[0]),
                np.array(coordinates[1]),
                pressure_coefficients
            )
            data_to_plot[ChartMode.CMZ] = cmz

        alpha_str = self._get_alpha(area_type=True)
        wind_region = self._get_wind_region()
        speed_sp = speed_sp_region(height, alpha_str, wind_region)

        fig = PlotBuilding.welch_graph(data_to_plot, height, speed_sp, sample_frequency=self.SAMPLE_FREQUENCY,
                                       number_of_time_counts=self.NUMBER_OF_TIME_COUNTS)
        self.add_plot_on_screen(fig, ChartType.ISOFIELDS)

    def open_plot_in_new_window(
            self,
            fig,
            title='График'
    ):
        window = QWidget()
        window.setWindowTitle(title)

        self.plots.append(window)
        layout = QVBoxLayout(window)
        plotWidget = MatplotlibWidget(fig)

        layout.addWidget(plotWidget.toolbar)
        layout.addWidget(plotWidget)
        window.show()

    def open_plot_in_new_window_sensors_overview_summary_coefficients(
            self,
            fig,
            title='График'
    ):
        self.windows_plot_sensors_overview_summary_coefficients = QWidget()
        self.windows_plot_sensors_overview_summary_coefficients.setWindowTitle(title)

        def closeEvent(event):
            match self.windows_plot_sensors_overview_summary_coefficients.windowTitle():
                case ChartType.SUMMARY_COEFFICIENTS:
                    self.fig_summary_coefficients_sensors_overview = None
                case ChartType.SPECTRUM:
                    self.fig_spectrum_sensors_overview = None
            event.accept()  # Закрываем окно

        # Привязываем кастомный closeEvent к окну
        self.windows_plot_sensors_overview_summary_coefficients.closeEvent = closeEvent

        self.plots.append(self.windows_plot_sensors_overview_summary_coefficients)
        layout = QVBoxLayout(self.windows_plot_sensors_overview_summary_coefficients)
        plotWidget = MatplotlibWidget(fig)

        layout.addWidget(plotWidget.toolbar)
        layout.addWidget(plotWidget)
        self.windows_plot_sensors_overview_summary_coefficients.show()

    def open_plot_in_new_window_sensors_overview_spectrum(
            self,
            fig,
            title='График'
    ):
        self.windows_plot_sensors_overview_spectrum = QWidget()
        self.windows_plot_sensors_overview_spectrum.setWindowTitle(title)

        def closeEvent(event):
            match self.windows_plot_sensors_overview_spectrum.windowTitle():
                case ChartType.SUMMARY_COEFFICIENTS:
                    self.fig_summary_coefficients_sensors_overview = None
                case ChartType.SPECTRUM:
                    self.fig_spectrum_sensors_overview = None
            event.accept()  # Закрываем окно

        # Привязываем кастомный closeEvent к окну
        self.windows_plot_sensors_overview_spectrum.closeEvent = closeEvent

        self.plots.append(self.windows_plot_sensors_overview_spectrum)
        layout = QVBoxLayout(self.windows_plot_sensors_overview_spectrum)
        plotWidget = MatplotlibWidget(fig)

        layout.addWidget(plotWidget.toolbar)
        layout.addWidget(plotWidget)
        self.windows_plot_sensors_overview_spectrum.show()

    def _plot_envelopes(
            self,
            pressure_coefficients,
            mods
    ):
        figs = PlotBuilding.envelopes(pressure_coefficients, mods)

        return figs

    def plot_envelopes(
            self
    ):
        mods = [ChartMode(i) for i in self.envelopesParameters.getCurrentOptions()]
        pressure_coefficients = self._get_pressure_coefficients()

        figs = self._plot_envelopes(pressure_coefficients, mods)

        for fig in figs:
            self.open_plot_in_new_window(fig, ChartType.ENVELOPES)

    @abstractmethod
    def _get_pressure_coefficients(
            self,
            *args,
            **kwargs
    ):
        print('Необходимо переопределить')
        pass

    def get_pressure_coefficients_for_the_sensor(
            self,
            *args,
            **kwargs
    ):
        print('Необходимо переопределить')
        pass

    @abstractmethod
    def get_model_id(self,
                     model_name,
                     alpha
                     ):
        print('Необходимо переопределить')
        pass

    @abstractmethod
    def _get_coordinates(
            self,
            *args,
            **kwargs
    ):
        print('Необходимо переопределить')
        pass

    @abstractmethod
    def get_face_number(
            self,
            model_id,
            alpha
    ):
        print('Необходимо переопределить')
        pass

    @abstractmethod
    def create_report(
            self
    ):
        print('Необходимо переопределить')
        pass

    @abstractmethod
    def _get_size_and_count_sensors(
            self,
            *args,
            **kwargs
    ):
        pass

    @abstractmethod
    def get_model_name(
            self,
            *args,
            **kwargs
    ):
        pass

    def _plot_isofields(
            self,
            model_size,
            size,
            count_sensors,
            parameter,
            pressure_coefficients,
            coordinates
    ):
        match self.ComboBoxTypesIsofields.text():
            case IsofieldsType.PRESSURE:
                area_type = self._get_alpha(area_type=True)
                wind_region = self._get_wind_region()
                fig = PlotBuilding.isofields_coefficients(model_size,
                                                          size,
                                                          count_sensors,
                                                          parameter,
                                                          pressure_coefficients,
                                                          coordinates,
                                                          area_type,
                                                          wind_region
                                                          )
            case IsofieldsType.COEFFICIENT:
                fig = PlotBuilding.isofields_coefficients(model_size,
                                                          size,
                                                          count_sensors,
                                                          parameter,
                                                          pressure_coefficients,
                                                          coordinates
                                                          )

        return fig

    def plot_isofields(
            self
    ):
        pressure_coefficients = self._get_pressure_coefficients()
        coordinates = self._get_coordinates()
        size, count_sensors = self._get_size_and_count_sensors(pressure_coefficients.shape[1])
        model_size = self._get_model_size()
        parameter = ChartMode(self.isofieldsParameters.currentText())

        fig = self._plot_isofields(model_size, size, count_sensors, parameter, pressure_coefficients, coordinates)

        self.add_plot_on_screen(fig, ChartType.ISOFIELDS)

    def plot_summary_coefficients_cartesian(
            self,
            angle,
            pressure_coefficients_storage: dict,
            coordinates,
            size_tpu,
            count_sensors,
            model_size
    ):

        data_to_plot = {}

        parameters = [ChartMode(i) for i in self.cartesianParameters.getCurrentOptions()]
        pressure_coefficients = pressure_coefficients_storage[angle]

        if ChartMode.CX in parameters or ChartMode.CY in parameters:
            cx, cy = aot_calculations.calculate_cx_cy(
                *count_sensors,
                *size_tpu,
                np.array(coordinates[0]),
                np.array(coordinates[1]),
                pressure_coefficients
            )
            if ChartMode.CX in parameters:
                data_to_plot[ChartMode.CX] = cx
            if ChartMode.CY in parameters:
                data_to_plot[ChartMode.CY] = cy
        if ChartMode.CMZ in parameters:
            cmz = aot_calculations.calculate_cmz(
                *count_sensors,
                angle,
                *size_tpu,
                np.array(coordinates[0]),
                np.array(coordinates[1]),
                pressure_coefficients
            )

            data_to_plot[ChartMode.CMZ] = cmz

        match CartesianModelSummaryCoefficients(self.cartesianModelSummaryCoefficients.currentText()):
            case CartesianModelSummaryCoefficients.TPU:
                kt = 1
            case CartesianModelSummaryCoefficients.REAL:
                breadth, depth, height = model_size
                breadth_tpu, depth_tpu, height_tpu = size_tpu
                alpha_str = self._get_alpha(area_type=True)
                wind_region = self._get_wind_region()

                kz = height / height_tpu

                match alpha_str:
                    case 'A':
                        Uz_a_x = Uz_a_0_16_x
                        Uz_a_z = np.array(Uz_a_0_16_z)
                    case 'C':
                        Uz_a_x = Uz_a_0_25_x
                        Uz_a_z = np.array(Uz_a_0_25_z)

                Uz_a_z_scaled = Uz_a_z * kz
                speed_tpu_function = scipy.interpolate.interp1d(Uz_a_z_scaled, Uz_a_x)
                speed_tpu = speed_tpu_function(height)

                speed_sp = speed_sp_region(height, alpha_str, wind_region)

                l_m = aot_calculations.calculate_projection_on_the_axis(breadth, depth, angle)
                l_tpu = aot_calculations.calculate_projection_on_the_axis(breadth_tpu, depth_tpu, angle)

                kv = speed_sp / speed_tpu
                km = l_m / l_tpu

                kt = km / kv

        fig = PlotBuilding.summary_coefficients(data_to_plot,
                                                kt,
                                                sample_period=self.SAMPLE_PERIOD,
                                                number_of_time_counts=self.NUMBER_OF_TIME_COUNTS)

        self.add_plot_on_screen(fig, ChartType.SUMMARY_COEFFICIENTS)

    def plot_summary_coefficients_polar(
            self,
            angle_border,
            pressure_coefficients_storage: dict,
            coordinates,
            size_tpu,
            count_sensors
    ):

        data_to_plot = {}

        views = [ChartMode(i) for i in self.polarView.getCurrentOptions()]
        parameters = [ChartMode(i) for i in self.polarParameters.getCurrentOptions()]

        x = np.array(coordinates[0])
        y = np.array(coordinates[1])

        cx_flag = ChartMode.CX in views
        cy_flag = ChartMode.CY in views
        cmz_flag = ChartMode.CMZ in views

        if cx_flag or cy_flag:
            for v in [ChartMode.CX, ChartMode.CY]:
                data_to_plot[v] = {}
                for p in parameters:
                    data_to_plot[v][p] = []

        if cmz_flag:
            data_to_plot[ChartMode.CMZ] = {}
            for p in parameters:
                data_to_plot[ChartMode.CMZ][p] = []

        for angle in range(0, angle_border + 5, 5):
            pressure_coefficients = pressure_coefficients_storage[angle]

            if cx_flag or cy_flag:
                cx, cy = aot_calculations.calculate_cx_cy(
                    *count_sensors,
                    *size_tpu,
                    x,
                    y,
                    pressure_coefficients
                )
                for p in parameters:
                    data_to_plot[ChartMode.CX][p].append(polar_lambdas[p](cx))
                    data_to_plot[ChartMode.CY][p].append(polar_lambdas[p](cy))

            if cmz_flag:
                cmz = aot_calculations.calculate_cmz(
                    *count_sensors,
                    angle,
                    *size_tpu,
                    x,
                    y,
                    pressure_coefficients
                )
                for p in parameters:
                    data_to_plot[ChartMode.CMZ][p].append(polar_lambdas[p](cmz))

        if len(pressure_coefficients_storage) != 72:
            if cx_flag or cy_flag:
                for p in parameters:
                    cx_scale, cy_scale = scaling_data(data_to_plot[ChartMode.CX][p], data_to_plot[ChartMode.CY][p],
                                                      angle_border=angle_border)
                    data_to_plot[ChartMode.CX][p] = cx_scale
                    data_to_plot[ChartMode.CY][p] = cy_scale

            if cmz_flag:
                for p in parameters:
                    cmz_scale = scaling_data(data_to_plot[ChartMode.CMZ][p], angle_border=angle_border)
                    data_to_plot[ChartMode.CMZ][p] = cmz_scale
        else:
            if cx_flag:
                for p in parameters:
                    data_to_plot[ChartMode.CX][p] = np.append(data_to_plot[ChartMode.CX][p],
                                                              data_to_plot[ChartMode.CX][p][0])

            if cy_flag:
                for p in parameters:
                    data_to_plot[ChartMode.CY][p] = np.append(data_to_plot[ChartMode.CY][p],
                                                              data_to_plot[ChartMode.CY][p][0])

            if cmz_flag:
                for p in parameters:
                    data_to_plot[ChartMode.CMZ][p] = np.append(data_to_plot[ChartMode.CMZ][p],
                                                               data_to_plot[ChartMode.CMZ][p][0])

        if not cx_flag and ChartMode.CX in data_to_plot:
            del data_to_plot[ChartMode.CX]

        if not cy_flag and ChartMode.CY in data_to_plot:
            del data_to_plot[ChartMode.CY]

        if not cmz_flag and ChartMode.CMZ in data_to_plot:
            del data_to_plot[ChartMode.CMZ]

        fig = PlotBuilding.polar_plot(data_to_plot)

        self.add_plot_on_screen(fig, ChartType.SUMMARY_COEFFICIENTS)

    def plot_summary_coefficients(
            self
    ):
        if not ([ChartMode(i) for i in self.cartesianParameters.getCurrentOptions()] or
                ([ChartMode(i) for i in self.polarView.getCurrentOptions()] and
                 [ChartMode(i) for i in self.polarView.getCurrentOptions()])):
            return

        type_plot = CoordinateSystem(self.ComboBoxCoordinateSystemSummaryCoefficients.currentText())

        coordinates = self._get_coordinates()

        size_tpu, count_sensors = self._get_size_and_count_sensors(len(coordinates[0]))

        pressure_coefficients_storage = {}

        match type_plot:
            case CoordinateSystem.CARTESIAN:
                model_size = self._get_model_size()
                angle = self._get_angle()

                pressure_coefficients = self._get_pressure_coefficients()
                pressure_coefficients_storage[angle] = pressure_coefficients

                self.plot_summary_coefficients_cartesian(angle, pressure_coefficients_storage, coordinates, size_tpu,
                                                         count_sensors, model_size)

            case CoordinateSystem.POLAR:
                pressure_coefficients_storage = self._get_pressure_coefficients_for_polar_plot()
                angle_border = self._get_angle_border()
                self.plot_summary_coefficients_polar(angle_border, pressure_coefficients_storage, coordinates,
                                                     size_tpu, count_sensors)

    @abstractmethod
    def _get_pressure_coefficients_for_polar_plot(
            self,
            *args,
            **kwargs
    ):
        pass

    @abstractmethod
    def _get_angle_border(
            self,
            *args,
            **kwargs
    ):
        pass

    def _plot_pseudocolor_coefficients(
            self,
            model_size,
            count_sensors,
            parameter,
            pressure_coefficients
    ):
        fig = PlotBuilding.pseudocolor_coefficients(model_size,
                                                    count_sensors,
                                                    parameter,
                                                    pressure_coefficients)

        return fig

    def plot_pseudocolor_coefficients(
            self
    ):
        pressure_coefficients = self._get_pressure_coefficients()
        size, count_sensors = self._get_size_and_count_sensors(pressure_coefficients.shape[1])
        model_size = self._get_model_size()
        parameter = ChartMode(self.discreteIsofieldsParameters.currentText())

        fig = self._plot_pseudocolor_coefficients(model_size, count_sensors, parameter, pressure_coefficients)

        self.add_plot_on_screen(fig, ChartType.DISCRETE_ISOFIELDS)
