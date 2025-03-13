import enum


class Buttons(enum.StrEnum):
    SENSORS = "Датчики"
    PLOTS = "Графики"
    BUILDING_SIZE = "Размеры здания"
    WIND_ANGLE = "Угол атаки ветра"
    BUILD_PLOT = "Построить"
    INTERFERING_INFORMATION = "Схема окружающей застройки"
    INTERFERING_POSITION = "Позиция застройки"
    INTERFERING_SIZE = "Размеры застройки"
    GENERAL_INFORMATION = "Общие сведения"
    WIND_REGION = "Ветровой район"
    TYPE_OF_AREA = "Тип местности"
    REPORT = "Отчет"
    FEA = "МКЭ"

    PARAMETERS = "Параметры"
    VIEW = "Вид"
