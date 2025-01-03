import enum


class ChartType(enum.StrEnum):
    ISOFIELDS = 'Изополя'
    ENVELOPES = 'Огибающие'
    SUMMARY_COEFFICIENTS = 'Суммарные коэффициенты'
    SPECTRUM = 'Спектры'
