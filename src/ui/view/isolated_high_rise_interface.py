# coding:utf-8
from PySide6.QtWidgets import QWidget, QGridLayout, QHBoxLayout, QStackedLayout
from qfluentwidgets import ScrollArea, PushButton, TitleLabel, CheckBox, ComboBox, \
    StrongBodyLabel, LineEdit

from src.ui.common.ChartMode import ChartMode
from src.ui.common.ChartType import ChartType
from src.ui.common.CoordinateSystem import CoordinateSystem
from src.ui.common.IsofieldsType import IsofieldsType
from src.ui.common.StyleSheet import StyleSheet


class IsolatedHighRiseInterface(ScrollArea):
    """ Isolated High Rise Interface"""

    def __init__(
            self,
            parent=None
    ):
        super().__init__(parent=parent)
        self.setObjectName('IsolatedHighRiseInterface')
        self.view = QWidget(self)
        # Set style
        StyleSheet.MAIN_INTERFACE.apply(self)
        # Create grid layout
        self.GrigLayout = QGridLayout(self.view)
        # Add label to grid layout
        self.GrigLayout.addWidget(TitleLabel('Общие сведения'), 0, 0)
        # Initialization left menu
        self._init_general_information()
        # Initialization chart menu
        self._init_chart_menu()

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
        # Add combo box to horizontal box layout
        self.hBoxLayoutWindRegions.addWidget(self.ComboBoxWindRegions)
        # Add horizontal box layout to grid layout
        self.GrigLayout.addLayout(self.hBoxLayoutWindRegions, 1, 0)

        # Type of area
        self.hBoxLayoutTypeOfArea = QHBoxLayout(self.view)
        self.hBoxLayoutTypeOfArea.addWidget(StrongBodyLabel('Тип местности'))
        self.ComboBoxTypeOfArea = ComboBox()
        self.ComboBoxTypeOfArea.addItems([
            self.tr(i) for i in ('A', 'B', 'C')
        ])
        self.hBoxLayoutTypeOfArea.addWidget(self.ComboBoxTypeOfArea)
        self.GrigLayout.addLayout(self.hBoxLayoutTypeOfArea, 2, 0)

        # Wind angle
        self.hBoxLayoutWindAngle = QHBoxLayout(self.view)
        self.hBoxLayoutWindAngle.addWidget(StrongBodyLabel('Угол атаки ветра'))
        # Create text input widget
        self.lineEditWindAngle = LineEdit()
        # Set default text
        self.lineEditWindAngle.setText(self.tr('0'))
        # Set clear button
        self.lineEditWindAngle.setClearButtonEnabled(True)
        # Add text input widget to horizontal box layout
        self.hBoxLayoutWindAngle.addWidget(self.lineEditWindAngle)
        self.GrigLayout.addLayout(self.hBoxLayoutWindAngle, 3, 0)

        # Building size
        self.hBoxLayoutBuildingSize = QHBoxLayout(self.view)
        self.hBoxLayoutBuildingSize.addWidget(StrongBodyLabel('Размеры здания'))
        self.lineEditBuildingSize = LineEdit()
        self.lineEditBuildingSize.setText(self.tr('0.1 0.1 0.3'))
        self.lineEditBuildingSize.setClearButtonEnabled(True)
        self.hBoxLayoutBuildingSize.addWidget(self.lineEditBuildingSize)
        self.GrigLayout.addLayout(self.hBoxLayoutBuildingSize, 4, 0)

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
        self.hBoxLayoutChartMenu.addWidget(self.ComboBoxChartMenu)

        self.StackedLayoutTypeChart = QStackedLayout()

        self._init_chart_isofields()
        self._init_chart_envelopes()
        self._init_chart_summary_coefficients()
        self._init_chart_spectrum()

        self.hBoxLayoutChartMenu.addLayout(self.StackedLayoutTypeChart)

        self.PushButtonCreatePlot = PushButton('Построить')
        self.hBoxLayoutChartMenu.addWidget(self.PushButtonCreatePlot)
        self.GrigLayout.addLayout(self.hBoxLayoutChartMenu, 0, 1)

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
        self.hBoxLayoutIsofields.addWidget(self.ComboBoxTypesIsofields)

        self.ComboBoxModeIsofields = ComboBox()
        self.ComboBoxModeIsofields.addItems([
            self.tr(i) for i in (ChartMode.MAX,
                                 ChartMode.MEAN,
                                 ChartMode.MIN,
                                 ChartMode.RMS,
                                 ChartMode.STD,
                                 )
        ])
        self.hBoxLayoutIsofields.addWidget(self.ComboBoxModeIsofields)

        self.StackedLayoutTypeChart.addWidget(self.WidgetIsofields)

    def _init_chart_envelopes(
            self
    ):
        self.WidgetEnvelopes = QWidget()
        self.hBoxLayoutEnvelopes = QHBoxLayout(self.WidgetEnvelopes)

        buttons = [CheckBox(i) for i in (ChartMode.MAX,
                                         ChartMode.MEAN,
                                         ChartMode.MIN,
                                         ChartMode.RMS,
                                         ChartMode.STD,
                                         )
                   ]

        for b in buttons:
            self.hBoxLayoutEnvelopes.addWidget(b)

        buttons[0].setChecked(True)

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
        self.hBoxLayoutSummaryCoefficients.addWidget(self.ComboBoxСoordinateSystemSummaryCoefficients)
        # Stacked layout which varies depending on the coordinate system
        self.StackedLayoutSummaryCoefficients = QStackedLayout()
        self.hBoxLayoutSummaryCoefficients.addLayout(self.StackedLayoutSummaryCoefficients)
        # Cartesian system
        self.WidgetCartesianSummaryCoefficients = QWidget()
        self.hBoxLayoutCartesianCoordinateSystem = QHBoxLayout(self.WidgetCartesianSummaryCoefficients)
        cartesian_buttons = [CheckBox(i) for i in (ChartMode.CX,
                                                   ChartMode.CY,
                                                   ChartMode.CMZ,
                                                   )]
        for b in cartesian_buttons:
            self.hBoxLayoutCartesianCoordinateSystem.addWidget(b)
        self.StackedLayoutSummaryCoefficients.addWidget(self.WidgetCartesianSummaryCoefficients)
        # Polar system
        self.WidgetPolarSummaryCoefficients = QWidget()
        self.hBoxLayoutPolarCoordinateSystem = QHBoxLayout(self.WidgetPolarSummaryCoefficients)

        polar_buttons = [CheckBox(i) for i in (ChartMode.MAX,
                                               ChartMode.MEAN,
                                               ChartMode.MIN,
                                               ChartMode.RMS,
                                               ChartMode.STD,
                                               ChartMode.SETTLEMENT,
                                               ChartMode.WARRANTY_PLUS,
                                               ChartMode.WARRANTY_MINUS,
                                               )
                         ]
        for b in polar_buttons:
            self.hBoxLayoutPolarCoordinateSystem.addWidget(b)
        self.StackedLayoutSummaryCoefficients.addWidget(self.WidgetPolarSummaryCoefficients)
        # Add widget to main layout
        self.StackedLayoutTypeChart.addWidget(self.WidgetSummaryCoefficients)

    def _init_chart_spectrum(
            self
    ):
        self.WidgetSpectrum = QWidget()
        self.hBoxLayoutSpectrum = QHBoxLayout(self.WidgetSpectrum)

        buttons = [CheckBox(i) for i in (ChartMode.CX,
                                         ChartMode.CY,
                                         ChartMode.CMZ,
                                         )
                   ]

        for b in buttons:
            self.hBoxLayoutSpectrum.addWidget(b)

        buttons[0].setChecked(True)

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
