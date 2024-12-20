# coding:utf-8
import asyncio

import numpy as np
from PySide6 import QtGui, QtCore
from PySide6.QtGui import QAction
from PySide6.QtWidgets import QWidget, QGridLayout, QHBoxLayout, QStackedLayout, QVBoxLayout
from matplotlib.backend_tools import ToolBase
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT
from matplotlib.figure import Figure
from qfluentwidgets import ScrollArea, PushButton, TitleLabel, ComboBox, \
    StrongBodyLabel, LineEdit

from src.submodules.databasetoolkit.isolated import load_pressure_coefficients, find_experiment_by_model_name, \
    load_positions
from src.submodules.plot.plot import Plot
from src.submodules.plot.plotBuilding import PlotBuilding
from src.submodules.utils import utils
from src.submodules.utils.scaling import get_model_and_scale_factors
from src.ui.common.ChartMode import ChartMode
from src.ui.common.ChartType import ChartType
from src.ui.common.CoordinateSystem import CoordinateSystem
from src.ui.common.IsofieldsType import IsofieldsType
from src.ui.common.StyleSheet import StyleSheet
from src.ui.components.MultiSelectionComboBox import MultiSelectionComboBox
from compiled_aot.integration import aot_integration

from matplotlib.backend_tools import ToolBase
from matplotlib.backend_managers import ToolManager


class MatplotlibWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Secondary Window")
        self.setGeometry(100, 100, 300, 200)

        # Создаем объект Figure и добавляем его в FigureCanvas
        self.figure = Figure()
        self.canvas = FigureCanvasQTAgg(self.figure)
        # Создаем Toolbar
        self.toolbar = NavigationToolbar2QT(self.canvas, self)

        # Добавляем пользовательскую кнопку
        self.add_custom_button()
        # Добавляем компоновку для отображения FigureCanvas
        layout = QVBoxLayout(self)
        layout.addWidget(self.canvas)

        # Рисуем график
        # self.plot()

    def add_custom_button(self):
        path = r"src\ui\resource\images\fl_icon.png"
        act = self.toolbar.addAction(self._icon(path), 'Открыть в новом окне', self.show)
        # if you set the value to -1, the button will be in the rightmost position
        self.toolbar.insertAction(self.toolbar.actions()[-2], act)

    def _icon(self, path):
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

    def show(self):
        self.window = QWidget()
        layout = QVBoxLayout(self.window)

        layout.addWidget(self.toolbar)
        layout.addWidget(self)
        self.window.show()

    def plot(self):
        # Создаем оси и строим график
        ax = self.figure.add_subplot(111)
        x = np.linspace(0, 2 * np.pi, 100)
        y = np.sin(x)
        ax.plot(x, y, label="Sine Wave")
        ax.set_title("Matplotlib with PySide6")
        ax.legend()
        self.canvas.draw()


class IsolatedHighRiseInterface(ScrollArea):
    """ Isolated High Rise Interface"""

    def __init__(
            self,
            parent=None,
            engine=None
    ):
        super().__init__(parent=parent)
        self.engine = engine
        self.setObjectName('IsolatedHighRiseInterface')
        self.view = QWidget(self)
        # Set style
        StyleSheet.MAIN_INTERFACE.apply(self)
        # Create grid layout
        self.grigLayout = QGridLayout(self.view)
        # Add label to grid layout
        self.grigLayout.addWidget(TitleLabel('Общие сведения'), 0, 0)
        # Initialization left menu
        self._init_general_information()
        # Initialization chart menu
        self._init_chart_menu()
        # Create plot widget
        self.plotWidget = MatplotlibWidget()
        # Add toolbar to grid layout
        self.grigLayout.addWidget(self.plotWidget.toolbar, 1, 1)
        # Add plot to grid layout
        # widget, row, column, rowSpan, columnSpan
        self.grigLayout.addWidget(self.plotWidget, 2, 1, 5, 5)

    def _init_general_information(
            self
    ):
        # Wind regions
        # Create horizontal box layout
        self.hBoxLayoutWindRegions = QHBoxLayout(self.view)
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
        # Add horizontal box layout to grid layout
        self.grigLayout.addLayout(self.hBoxLayoutWindRegions, 1, 0)

        # Type of area
        self.hBoxLayoutTypeOfArea = QHBoxLayout(self.view)
        self.hBoxLayoutTypeOfArea.addWidget(StrongBodyLabel('Тип местности'))
        self.ComboBoxTypeOfArea = ComboBox()
        self.ComboBoxTypeOfArea.addItems([
            self.tr(i) for i in ('A', 'B', 'C')
        ])
        self.ComboBoxTypeOfArea.setFixedWidth(75)
        self.hBoxLayoutTypeOfArea.addWidget(self.ComboBoxTypeOfArea)
        self.grigLayout.addLayout(self.hBoxLayoutTypeOfArea, 2, 0)

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
        self.grigLayout.addLayout(self.hBoxLayoutWindAngle, 3, 0)

        # Building size
        self.hBoxLayoutBuildingSize = QHBoxLayout(self.view)
        self.hBoxLayoutBuildingSize.addWidget(StrongBodyLabel('Размеры здания'))
        self.lineEditBuildingSize = LineEdit()
        self.lineEditBuildingSize.setText(self.tr('0.1 0.1 0.1'))
        self.lineEditBuildingSize.setClearButtonEnabled(True)
        self.lineEditBuildingSize.setFixedWidth(125)
        self.hBoxLayoutBuildingSize.addWidget(self.lineEditBuildingSize)
        self.grigLayout.addLayout(self.hBoxLayoutBuildingSize, 4, 0)

    def _init_chart_menu(
            self
    ):
        # Chart menu
        self.hBoxLayoutChartMenu = QHBoxLayout(self.view)
        self.ComboBoxChartMenu = ComboBox()
        self.ComboBoxChartMenu.addItems([
            self.tr(i) for i in (ChartType.ISOFIELDS,
                                 ChartType.ENVELOPES,
                                 ChartType.SUMMARY_COEFFICIENTS,
                                 ChartType.SPECTRUM
                                 )
        ])
        self.ComboBoxChartMenu.currentTextChanged.connect(self._switch_stacked_layout_type_chart)
        self.ComboBoxChartMenu.setFixedWidth(225)
        self.hBoxLayoutChartMenu.addWidget(self.ComboBoxChartMenu)

        self.StackedLayoutTypeChart = QStackedLayout()

        self._init_chart_isofields()
        self._init_chart_envelopes()
        self._init_chart_summary_coefficients()
        self._init_chart_spectrum()

        self.hBoxLayoutChartMenu.addLayout(self.StackedLayoutTypeChart)

        self.PushButtonCreatePlot = PushButton('Построить')
        self.PushButtonCreatePlot.clicked.connect(self.create_plot)
        self.PushButtonCreatePlot.setFixedWidth(100)

        self.hBoxLayoutChartMenu.addWidget(self.PushButtonCreatePlot)
        self.grigLayout.addLayout(self.hBoxLayoutChartMenu, 0, 1)

    def create_plot(self):
        match self.StackedLayoutTypeChart.currentIndex():
            case 0:
                print(0)
            case 1:
                print(1)
                self.plot_envelopes()
            case 2:
                print(2)
            case 3:
                print(3)

        # self.plot_widget.show()  # Открыть график в новом окне

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

        self.isofieldsParameters = MultiSelectionComboBox(placeholder_text='Параметры')
        self.isofieldsParameters.addItems((ChartMode.MAX,
                                           ChartMode.MEAN,
                                           ChartMode.MIN,
                                           ChartMode.RMS,
                                           ChartMode.STD,
                                           ))
        self.hBoxLayoutIsofields.addWidget(self.isofieldsParameters)

        self.StackedLayoutTypeChart.addWidget(self.WidgetIsofields)

    def _init_chart_envelopes(
            self
    ):
        self.WidgetEnvelopes = QWidget()
        self.hBoxLayoutEnvelopes = QHBoxLayout(self.WidgetEnvelopes)
        self.envelopesParameters = MultiSelectionComboBox(placeholder_text='Параметры')
        self.envelopesParameters.addItems((ChartMode.MAX,
                                           ChartMode.MEAN,
                                           ChartMode.MIN,
                                           ChartMode.RMS,
                                           ChartMode.STD,
                                           ))
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
        self.cartesianParameters = MultiSelectionComboBox(placeholder_text='Параметры')
        self.cartesianParameters.addItems((ChartMode.CX,
                                           ChartMode.CY,
                                           ChartMode.CMZ,
                                           ))
        self.hBoxLayoutCartesianCoordinateSystem.addWidget(self.cartesianParameters)
        self.StackedLayoutSummaryCoefficients.addWidget(self.WidgetCartesianSummaryCoefficients)
        # Polar system
        self.WidgetPolarSummaryCoefficients = QWidget()
        self.hBoxLayoutPolarCoordinateSystem = QHBoxLayout(self.WidgetPolarSummaryCoefficients)
        self.polarParameters = MultiSelectionComboBox(placeholder_text='Параметры')
        self.polarParameters.addItems((ChartMode.MAX,
                                       ChartMode.MEAN,
                                       ChartMode.MIN,
                                       ChartMode.RMS,
                                       ChartMode.STD,
                                       ChartMode.SETTLEMENT,
                                       ChartMode.WARRANTY_PLUS,
                                       ChartMode.WARRANTY_MINUS,
                                       ))
        self.hBoxLayoutPolarCoordinateSystem.addWidget(self.polarParameters)
        self.StackedLayoutSummaryCoefficients.addWidget(self.WidgetPolarSummaryCoefficients)
        # Add widget to main layout
        self.StackedLayoutTypeChart.addWidget(self.WidgetSummaryCoefficients)

    def _init_chart_spectrum(
            self
    ):
        self.WidgetSpectrum = QWidget()
        self.hBoxLayoutSpectrum = QHBoxLayout(self.WidgetSpectrum)
        self.spectrumParameters = MultiSelectionComboBox(placeholder_text='Параметры')
        self.spectrumParameters.addItems((ChartMode.CX,
                                          ChartMode.CY,
                                          ChartMode.CMZ,
                                          ))
        self.spectrumParameters.checkAllItems()
        self.hBoxLayoutSpectrum.addWidget(self.spectrumParameters)
        self.StackedLayoutTypeChart.addWidget(self.WidgetSpectrum)

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
        return map(float, self.lineEditBuildingSize.text().replace(',', '.').split(' '))

    def _get_alpha(
            self
    ):
        type_alpha = {
            'A': 4,
            'C': 6,
        }
        return type_alpha[self.ComboBoxTypeOfArea.text()]

    def plot_envelopes(
            self
    ):
        alpha = self._get_alpha()
        model_name, _ = get_model_and_scale_factors(*self._get_model_size(), alpha)
        angle = int(self.lineEditWindAngle.text())

        mods = self.envelopesParameters.getSelected()

        print(mods)
        print(alpha)
        print(model_name)
        print(angle)
        model_id = asyncio.run(find_experiment_by_model_name(model_name, alpha, self.engine)).model_id
        pressure_coefficients = asyncio.run(load_pressure_coefficients(model_id, alpha, self.engine, angle=angle))[
            angle]
        figs = PlotBuilding.envelopes(pressure_coefficients, mods)
        print(123)
        self.plotWidget.canvas = FigureCanvasQTAgg(figs[0])
        self.plotWidget.canvas.draw()

    def plot_isofields(
            self
    ):
        alpha = self._get_alpha()
        model_name, _ = get_model_and_scale_factors(*self._get_model_size(), alpha)
        angle = int(self.lineEditWindAngle.text())

        mods = self.isofieldsParameters.getSelected()

        model_id = asyncio.run(find_experiment_by_model_name(model_name, alpha, self.engine)).model_id
        pressure_coefficients = asyncio.run(load_pressure_coefficients(model_id, alpha, self.engine, angle=angle))[
            angle]
        coordinates = asyncio.run(load_positions(model_id, alpha, self.engine))

        PlotBuilding.isofields_coefficients((10, 10, 10),
                                            model_name,
                                            *mods,
                                            pressure_coefficients,
                                            coordinates)

    def plot_pseudocolor_coefficients(
            self
    ):
        alpha = self._get_alpha()
        model_name, _ = get_model_and_scale_factors(*self._get_model_size(), alpha)
        angle = int(self.lineEditWindAngle.text())

        mods = self.isofieldsParameters.getSelected()

        model_id = asyncio.run(find_experiment_by_model_name(model_name, alpha, self.engine)).model_id
        pressure_coefficients = asyncio.run(load_pressure_coefficients(model_id, alpha, self.engine, angle=angle))[
            angle]
        coordinates = asyncio.run(load_positions(model_id, alpha, self.engine))

        PlotBuilding.pseudocolor_coefficients(pressure_coefficients,
                                              coordinates,
                                              model_name,
                                              *mods)

    def plot_summary_coefficients(
            self
    ):
        alpha = self._get_alpha()
        model_name, _ = get_model_and_scale_factors(*self._get_model_size(), alpha)
        angle = int(self.lineEditWindAngle.text())

        mods = self.sum.getSelected()

        model_id = asyncio.run(find_experiment_by_model_name(model_name, alpha, self.engine)).model_id
        pressure_coefficients = asyncio.run(load_pressure_coefficients(model_id, alpha, self.engine, angle=angle))[
            angle]
        coordinates = asyncio.run(load_positions(model_id, alpha, self.engine))

        size, count_sensors = utils.get_size_and_count_sensors(pressure_coefficients.shape[1],
                                                               model_name,
                                                               )

        cx, cy = aot_integration.calculate_cx_cy(
            *count_sensors,
            *size,
            np.array(coordinates[0]),
            np.array(coordinates[1]),
            pressure_coefficients
        )
        cmz = aot_integration.calculate_cmz(
            *count_sensors,
            angle,
            *size,
            np.array(coordinates[0]),
            np.array(coordinates[1]),
            pressure_coefficients
        )

        PlotBuilding.summary_coefficients({
            'Cx': cx,
            'Cy': cy,
            'CMz': cmz,
        }, DbType.ISOLATED)
