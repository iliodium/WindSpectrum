import enum


class ChartType(enum.StrEnum):
    ISOFIELDS = 'Изополя'
    DISCRETE_ISOFIELDS = 'Дискретные изополя'
    ENVELOPES = 'Огибающие'
    SUMMARY_AERODYNAMIC_COEFFICIENTS = 'Суммарные аэродинамические коэффициенты'
    AERODYNAMIC_COEFFICIENTS = 'Аэродинамические коэффициенты'
    SPECTRUM = 'Спектры'
