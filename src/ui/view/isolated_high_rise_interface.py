# coding:utf-8
import asyncio

import numpy as np
from PySide6 import QtGui, QtCore
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QWidget, QGridLayout, QHBoxLayout, QStackedLayout, QVBoxLayout, QSpacerItem, QSizePolicy
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT
from matplotlib.figure import Figure
from qfluentwidgets import ScrollArea, PushButton, TitleLabel, ComboBox, \
    StrongBodyLabel, LineEdit

from compiled_aot.integration import aot_integration
from src.common.DbType import DbType
from src.submodules.databasetoolkit.isolated import load_pressure_coefficients, find_experiment_by_model_name, \
    load_positions
from src.submodules.plot.plotBuilding import PlotBuilding
from src.submodules.plot.utils import scaling_data
from src.submodules.utils import utils
from src.submodules.utils.data_features import polar_lambdas
from src.submodules.utils.scaling import get_model_and_scale_factors
from src.ui.common.ChartMode import ChartMode
from src.ui.common.ChartType import ChartType
from src.ui.common.CoordinateSystem import CoordinateSystem
from src.ui.common.IsofieldsType import IsofieldsType
from src.ui.common.StyleSheet import StyleSheet
from src.ui.components.MultiSelectComboBox import MultiSelectComboBox


class MatplotlibWidget(ScrollArea):
    def __init__(
            self,
            fig,
            title='График',
            parent=None
    ):
        super().__init__(parent)
        container = QWidget()
        self.setWidget(container)
        self.setWidgetResizable(True)  # Обеспечивает изменение размеров содержимого

        self.fig = fig
        self.canvas = FigureCanvasQTAgg(fig)
        # Создаем Toolbar
        self.toolbar = NavigationToolbar2QT(self.canvas, self)
        # Добавляем компоновку для отображения FigureCanvas
        self.plotLayout = QVBoxLayout(container)
        self.plotLayout.addWidget(self.canvas)
        self.plotTitle = title


class IsolatedHighRiseInterface(QWidget):
    """Isolated High Rise Interface"""

    def __init__(
            self,
            parent=None,
            engine=None
    ):
        super().__init__(parent=parent)
        self.engine = engine
        self.setObjectName('IsolatedHighRiseInterface')
        self.view = self
        self.plotFlag = False

        # Set style
        StyleSheet.MAIN_INTERFACE.apply(self)
        # Create grid layout
        # self.hBoxLayoutMain = QGridLayout(self.view)
        self.hBoxLayoutMain = QHBoxLayout(self)
        self.hBoxLayoutMain.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        # Initialization left menu
        self._init_general_information()
        # self.hBoxLayoutMain.addWidget(PushButton())

        # Initialization chart menu
        self.vBoxLayoutPlot = QVBoxLayout(self)
        # self.vBoxLayoutPlot.addStretch()
        self.vBoxLayoutPlot.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        self.hBoxLayoutMain.addLayout(self.vBoxLayoutPlot)

        # self.vBoxLayoutPlot.addWidget(PushButton())
        self._init_chart_menu()

        # An array to store references to objects (like envelopes)
        # If you do not store the objects, they will be deleted by garbage collector
        self.plots = []

    def _init_general_information(
            self
    ):
        container = QWidget()
        container.setFixedWidth(275)
        container.setFixedHeight(225)

        vBoxLayoutGenInf = QVBoxLayout(container)
        # Add label to grid layout
        vBoxLayoutGenInf.addWidget(TitleLabel('Общие сведения'))

        # Wind regions
        # Create horizontal box layout
        self.hBoxLayoutWindRegions = QHBoxLayout()
        # Add label to horizontal box layout
        self.hBoxLayoutWindRegions.addWidget(StrongBodyLabel('Ветровой район'))
        # Create combo box
        self.ComboBoxWindRegions = ComboBox()
        # Fill the combo box
        self.ComboBoxWindRegions.addItems([
            self.tr(i) for i in ('Iа', 'I', 'II', 'III', 'IV', 'V', 'VI', 'VII')
        ])
        # set fixed width of combobox
        self.ComboBoxWindRegions.setFixedWidth(75)
        # Add combo box to horizontal box layout
        self.hBoxLayoutWindRegions.addWidget(self.ComboBoxWindRegions)
        vBoxLayoutGenInf.addLayout(self.hBoxLayoutWindRegions)

        # Type of area
        self.hBoxLayoutTypeOfArea = QHBoxLayout(self.view)
        self.hBoxLayoutTypeOfArea.addWidget(StrongBodyLabel('Тип местности'))
        self.ComboBoxTypeOfArea = ComboBox()
        self.ComboBoxTypeOfArea.addItems([
            self.tr(i) for i in ('A', 'B', 'C')
        ])
        self.ComboBoxTypeOfArea.setFixedWidth(75)
        self.hBoxLayoutTypeOfArea.addWidget(self.ComboBoxTypeOfArea)
        vBoxLayoutGenInf.addLayout(self.hBoxLayoutTypeOfArea)

        # Wind angle
        self.hBoxLayoutWindAngle = QHBoxLayout(self.view)
        self.hBoxLayoutWindAngle.addWidget(StrongBodyLabel('Угол атаки ветра'))
        # Create text input widget
        self.lineEditWindAngle = LineEdit()
        # Set default text
        self.lineEditWindAngle.setText(self.tr('0'))
        # Set clear button
        self.lineEditWindAngle.setClearButtonEnabled(True)
        self.lineEditWindAngle.setFixedWidth(75)
        # Add text input widget to horizontal box layout
        self.hBoxLayoutWindAngle.addWidget(self.lineEditWindAngle)
        vBoxLayoutGenInf.addLayout(self.hBoxLayoutWindAngle)

        # Building size
        self.hBoxLayoutBuildingSize = QHBoxLayout(self.view)
        self.hBoxLayoutBuildingSize.addWidget(StrongBodyLabel('Размеры здания'))
        self.lineEditBuildingSize = LineEdit()
        self.lineEditBuildingSize.setText(self.tr('0.1 0.1 0.1'))
        self.lineEditBuildingSize.setClearButtonEnabled(True)
        self.lineEditBuildingSize.setFixedWidth(125)
        self.hBoxLayoutBuildingSize.addWidget(self.lineEditBuildingSize)
        vBoxLayoutGenInf.addLayout(self.hBoxLayoutBuildingSize)
        # self.spacer = QSpacerItem(20, 40, QSizePolicy.Minimum, QSizePolicy.Expanding)
        # vBoxLayoutGenInf.addItem(self.spacer)

        self.hBoxLayoutMain.addWidget(container)

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

        self.PushButtonCreatePlot = PushButton('Построить')
        self.PushButtonCreatePlot.clicked.connect(self.create_plot)
        self.PushButtonCreatePlot.setFixedWidth(100)

        hBoxLayoutChartMenu.addWidget(self.PushButtonCreatePlot)
        self.vBoxLayoutPlot.addWidget(container)

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
                print(3)
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
        self.envelopesParameters = MultiSelectComboBox(placeholderText='Параметры')
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
        self.ComboBoxСoordinateSystemSummaryCoefficients = ComboBox()
        self.ComboBoxСoordinateSystemSummaryCoefficients.addItems([
            self.tr(i) for i in (CoordinateSystem.CARTESIAN,
                                 CoordinateSystem.POLAR,
                                 )
        ])
        self.ComboBoxСoordinateSystemSummaryCoefficients.currentTextChanged.connect(
            self._switch_stacked_layout_summary_coefficients)
        self.ComboBoxСoordinateSystemSummaryCoefficients.setFixedWidth(140)
        self.hBoxLayoutSummaryCoefficients.addWidget(self.ComboBoxСoordinateSystemSummaryCoefficients)
        # Stacked layout which varies depending on the coordinate system
        self.StackedLayoutSummaryCoefficients = QStackedLayout()
        self.hBoxLayoutSummaryCoefficients.addLayout(self.StackedLayoutSummaryCoefficients)
        # Cartesian system
        self.WidgetCartesianSummaryCoefficients = QWidget()
        self.hBoxLayoutCartesianCoordinateSystem = QHBoxLayout(self.WidgetCartesianSummaryCoefficients)
        self.cartesianParameters = MultiSelectComboBox(placeholderText='Параметры')
        self.cartesianParameters.addItems([ChartMode.CX,
                                           ChartMode.CY,
                                           ChartMode.CMZ,
                                           ])
        self.hBoxLayoutCartesianCoordinateSystem.addWidget(self.cartesianParameters)
        self.StackedLayoutSummaryCoefficients.addWidget(self.WidgetCartesianSummaryCoefficients)
        # Polar system
        self.WidgetPolarSummaryCoefficients = QWidget()
        self.hBoxLayoutPolarCoordinateSystem = QHBoxLayout(self.WidgetPolarSummaryCoefficients)
        self.polarView = MultiSelectComboBox(placeholderText='Вид')
        self.polarView.addItems([ChartMode.CX,
                                 ChartMode.CY,
                                 ChartMode.CMZ,
                                 ])
        self.polarParameters = MultiSelectComboBox(placeholderText='Параметры')
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
        self.spectrumParameters = MultiSelectComboBox(placeholderText='Параметры')
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
    ):
        return tuple(map(float, self.lineEditBuildingSize.text().replace(',', '.').split(' ')))

    def _get_alpha(
            self
    ):
        type_alpha = {
            'A': 4,
            'C': 6,
        }
        return type_alpha[self.ComboBoxTypeOfArea.text()]

    def _icon(
            self,
            path
    ):
        '''
        link to the original ->
        matplotlib.backends.backend_qt.NavigationToolbar2QT._icon
        '''
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
        self.vBoxLayoutPlot1 = QVBoxLayout(self.containerPlot)

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

    def plot_envelopes(
            self
    ):
        mods = [ChartMode(i) for i in self.envelopesParameters.getCurrentOptions()]

        if not mods:
            return

        alpha = self._get_alpha()
        model_name, _ = get_model_and_scale_factors(*self._get_model_size(), alpha)
        angle = int(self.lineEditWindAngle.text())

        model_id = asyncio.run(find_experiment_by_model_name(model_name, alpha, self.engine)).model_id
        pressure_coefficients = asyncio.run(load_pressure_coefficients(model_id, alpha, self.engine, angle=angle))[
            angle]
        figs = PlotBuilding.envelopes(pressure_coefficients, mods)

        for fig in figs:
            self.open_plot_in_new_window(fig, ChartType.ENVELOPES)

    def plot_isofields(
            self
    ):
        alpha = self._get_alpha()
        model_size = self._get_model_size()
        model_name, _ = get_model_and_scale_factors(*model_size, alpha)
        angle = int(self.lineEditWindAngle.text())

        model_id = asyncio.run(find_experiment_by_model_name(model_name, alpha, self.engine)).model_id
        pressure_coefficients = asyncio.run(load_pressure_coefficients(model_id, alpha, self.engine, angle=angle))[
            angle]
        coordinates = asyncio.run(load_positions(model_id, alpha, self.engine))

        parameter = ChartMode(self.isofieldsParameters.currentText())

        fig = PlotBuilding.isofields_coefficients(model_size,
                                                  model_name,
                                                  parameter,
                                                  pressure_coefficients,
                                                  coordinates)

        self.add_plot_on_screen(fig, ChartType.ISOFIELDS)

    def plot_summary_coefficients(
            self
    ):
        if not ([ChartMode(i) for i in self.cartesianParameters.getCurrentOptions()] or
                ([ChartMode(i) for i in self.polarView.getCurrentOptions()] and
                 [ChartMode(i) for i in self.polarView.getCurrentOptions()])):
            return

        type_plot = CoordinateSystem(self.ComboBoxСoordinateSystemSummaryCoefficients.currentText())
        alpha = self._get_alpha()

        model_size = self._get_model_size()
        model_name, scale_factors = get_model_and_scale_factors(*model_size, alpha)
        angle = int(self.lineEditWindAngle.text())
        model_id = asyncio.run(find_experiment_by_model_name(model_name, alpha, self.engine)).model_id
        coordinates = asyncio.run(load_positions(model_id, alpha, self.engine))

        size, count_sensors = utils.get_size_and_count_sensors(len(coordinates[0]),
                                                               model_name,
                                                               )
        data_to_plot = {}
        match type_plot:
            case CoordinateSystem.CARTESIAN:
                parameters = [ChartMode(i) for i in self.cartesianParameters.getCurrentOptions()]
                pressure_coefficients = \
                    asyncio.run(load_pressure_coefficients(model_id, alpha, self.engine, angle=angle))[
                        angle]

                if ChartMode.CX in parameters or ChartMode.CY in parameters:
                    cx, cy = aot_integration.calculate_cx_cy(
                        *count_sensors,
                        *size,
                        np.array(coordinates[0]),
                        np.array(coordinates[1]),
                        pressure_coefficients
                    )
                    if ChartMode.CX in parameters:
                        data_to_plot[ChartMode.CX] = cx
                    if ChartMode.CY in parameters:
                        data_to_plot[ChartMode.CY] = cy
                if ChartMode.CMZ in parameters:
                    cmz = aot_integration.calculate_cmz(
                        *count_sensors,
                        angle,
                        *size,
                        np.array(coordinates[0]),
                        np.array(coordinates[1]),
                        pressure_coefficients
                    )

                    data_to_plot[ChartMode.CMZ] = cmz
                fig = PlotBuilding.summary_coefficients(data_to_plot, DbType.ISOLATED)

            case CoordinateSystem.POLAR:
                views = [ChartMode(i) for i in self.polarView.getCurrentOptions()]
                parameters = [ChartMode(i) for i in self.polarParameters.getCurrentOptions()]
                model_scale_str = str(model_name)

                if model_scale_str[0] == model_scale_str[1]:
                    angle_border = 50
                else:
                    angle_border = 95

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

                for angle in range(0, angle_border, 5):
                    pressure_coefficients = \
                        asyncio.run(load_pressure_coefficients(model_id, alpha, self.engine, angle=angle))[angle]

                    if cx_flag or cy_flag:
                        cx, cy = aot_integration.calculate_cx_cy(
                            *count_sensors,
                            *size,
                            x,
                            y,
                            pressure_coefficients
                        )
                        for p in parameters:
                            data_to_plot[ChartMode.CX][p].append(polar_lambdas[p](cx))
                            data_to_plot[ChartMode.CY][p].append(polar_lambdas[p](cy))

                    if cmz_flag:
                        cmz = aot_integration.calculate_cmz(
                            *count_sensors,
                            angle,
                            *size,
                            x,
                            y,
                            pressure_coefficients
                        )
                        for p in parameters:
                            data_to_plot[ChartMode.CMZ][p].append(polar_lambdas[p](cmz))

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

                fig = PlotBuilding.polar_plot(data_to_plot)

        self.add_plot_on_screen(fig, ChartType.SUMMARY_COEFFICIENTS)

    def plot_pseudocolor_coefficients(
            self
    ):
        alpha = self._get_alpha()
        model_name, _ = get_model_and_scale_factors(*self._get_model_size(), alpha)
        angle = int(self.lineEditWindAngle.text())

        model_id = asyncio.run(find_experiment_by_model_name(model_name, alpha, self.engine)).model_id
        pressure_coefficients = asyncio.run(load_pressure_coefficients(model_id, alpha, self.engine, angle=angle))[
            angle]
        model_size = self._get_model_size()
        parameter = ChartMode(self.discreteIsofieldsParameters.currentText())

        fig = PlotBuilding.pseudocolor_coefficients(model_size,
                                                    model_name,
                                                    parameter,
                                                    pressure_coefficients)
        self.add_plot_on_screen(fig, ChartType.DISCRETE_ISOFIELDS)
