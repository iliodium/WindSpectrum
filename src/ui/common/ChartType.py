import enum


class ChartType(enum.StrEnum):
    ISOFIELDS = 'Изополя'
    DISCRETE_ISOFIELDS = 'Дискретные изополя'
    ENVELOPES = 'Огибающие'
    SUMMARY_COEFFICIENTS = 'Суммарные аэродинамические коэффициенты'
    SPECTRUM = 'Спектры'
