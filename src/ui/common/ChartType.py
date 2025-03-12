import enum


class ChartType(enum.StrEnum):
    ISOFIELDS = 'Изополя'
    ENVELOPES = 'Огибающие'
    SUMMARY_COEFFICIENTS = 'Суммарные аэродинамические коэффициенты'
    SPECTRUM = 'Спектры'
