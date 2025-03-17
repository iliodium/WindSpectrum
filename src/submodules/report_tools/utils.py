import os

from src.submodules.report_tools.reportFolder import ReportFolder
from src.ui.common.ChartMode import ChartMode
from src.ui.common.ChartType import ChartType
from src.ui.common.CoordinateSystem import CoordinateSystem
from src.ui.common.IsofieldsType import IsofieldsType


def create_directory_to_report(
        name
):
    folder = os.path.join(os.getcwd(), ReportFolder.WORD_REPORT, name)

    # folders for all types of plots
    for i in ChartType:
        os.makedirs(os.path.join(folder, i), exist_ok=True)

    # folders for isofields
    for i in IsofieldsType:
        for j in (ChartMode.MAX, ChartMode.MEAN, ChartMode.MIN, ChartMode.RMS, ChartMode.STD):
            os.makedirs(os.path.join(folder, ChartType.ISOFIELDS, i, j), exist_ok=True)

    # folders for discrete isofields
    for i in (ChartMode.MAX, ChartMode.MEAN, ChartMode.MIN, ChartMode.RMS, ChartMode.STD):
        os.makedirs(os.path.join(folder, ChartType.DISCRETE_ISOFIELDS, i), exist_ok=True)

    # folders for summary coefficients
    for i in CoordinateSystem:
        os.makedirs(os.path.join(folder, ChartType.SUMMARY_AERODYNAMIC_COEFFICIENTS, i), exist_ok=True)

    # folders for polar summary coefficients
    for i in (ChartMode.CX, ChartMode.CY, ChartMode.CMZ):
        os.makedirs(os.path.join(folder, ChartType.SUMMARY_AERODYNAMIC_COEFFICIENTS, CoordinateSystem.POLAR, i), exist_ok=True)

    # folders for spectrum
    for i in (ChartMode.CX, ChartMode.CY, ChartMode.CMZ):
        os.makedirs(os.path.join(folder, ChartType.SPECTRUM, i), exist_ok=True)
