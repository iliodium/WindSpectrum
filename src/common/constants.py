from src.common.TypeOfArea import TypeOfArea
from src.common.WindRegion import WindRegion

alpha_standards = {
    TypeOfArea.A: 0.15,
    'B': 0.2,
    TypeOfArea.C: 0.25
}
ks10 = {
    TypeOfArea.A: 1,
    'B': 0.65,
    TypeOfArea.C: 0.4
}
wind_regions = {
    WindRegion.FIRST: 0.17,
    WindRegion.FIRST_A: 0.23,
    WindRegion.SECOND: 0.30,
    WindRegion.THIRD: 0.38,
    WindRegion.FOURTH: 0.48,
    WindRegion.FIFTH: 0.60,
    WindRegion.SIXTH: 0.73,
    WindRegion.SEVENTH: 0.85
}
