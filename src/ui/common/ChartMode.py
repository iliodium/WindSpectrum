import enum


class ChartMode(enum.StrEnum):
    MAX = 'max'
    MEAN = 'mean'
    MIN = 'min'
    RMS = 'rms'
    STD = 'std'
    CX = 'Cx'
    CY = 'Cy'
    CMZ = 'CMz'
    SETTLEMENT = 'Расчетное'
    WARRANTY_PLUS = 'Обеспеченность +'
    WARRANTY_MINUS = 'Обеспеченность -'
