# coding:utf-8
import os
from abc import abstractmethod
from concurrent.futures import ProcessPoolExecutor

import matplotlib
import numpy as np
from compiled_functions import aot_calculations
from docx import Document
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import (Mm,
                         Pt,)
from matplotlib import pyplot as plt
from openpyxl import Workbook
from PySide6 import (QtCore,
                     QtGui,)
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (QHBoxLayout,
                               QStackedLayout,
                               QVBoxLayout,
                               QWidget,)
from qfluentwidgets import (ComboBox,
                            LineEdit,
                            PushButton,
                            StrongBodyLabel,
                            TitleLabel,)
from sqlalchemy import create_engine
from src.common.annotation import ModelSizeType
from src.common.constants import (alpha_standards,
                                  wind_regions,)
from src.submodules.plot.plotBuilding import PlotBuilding
from src.submodules.plot.utils import scaling_data
from src.submodules.report_tools.reportFolder import ReportFolder
from src.submodules.report_tools.utils import (
    create_directory_to_report_building,)
from src.submodules.report_tools.wordBuilder import WordBuilder
from src.submodules.utils.data_features import (calculated,
                                                lambdas,
                                                polar_lambdas,
                                                warranty_minus,
                                                warranty_plus,)
from src.submodules.utils.scaling import calculate_kt
from src.submodules.utils.speed_sp import speed_sp_region
from src.submodules.utils.utils import (converter_coordinates,
                                        tpu_size_to_real,)
from src.ui.common.Buttons import Buttons
from src.ui.common.CartesianModelSummaryCoefficients import (
    CartesianModelSummaryCoefficients,)
from src.ui.common.ChartMode import ChartMode
from src.ui.common.ChartType import ChartType
from src.ui.common.CoordinateSystem import CoordinateSystem
from src.ui.common.IsofieldsType import IsofieldsType
from src.ui.common.StyleSheet import StyleSheet
from src.ui.components.MultiSelectComboBox import MultiSelectComboBox
from src.ui.view.Interface import Interface
from src.ui.view.widgets.MatplotlibWidget import MatplotlibWidget
from src.ui.view.widgets.SensorWidget import SensorWidget


class BuildingInterface(Interface):
    def _init_general_information(
            self
    ):
        super()._init_general_information()
        self.PushButtonInterferingInformation = PushButton(Buttons.FEA)
        # self.PushButtonFiniteElementMethod.clicked.connect(self._switch_stacked_layout_sensors_overview)
        self.vBoxLayoutGenInf.addWidget(self.PushButtonInterferingInformation)
