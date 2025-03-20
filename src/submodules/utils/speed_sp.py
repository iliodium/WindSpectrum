"""
СП 20. Нагрузки и воздействия
"""

from pydantic import validate_call
from src.common.annotation import (AlphaStandardsOrKs10Type,
                                   WindRegionsType,)
from src.common.constants import (alpha_standards,
                                  ks10,
                                  wind_regions,)


@validate_call
def speed_sp(
        z: int | float,
        scale_ks: int | float,
        a: float,
        kS: float,
        po: float,
        y: float,
        w0: int | float,
        *,
        uS_power: float = 0.5,
        scaling_divisor: int | float = 10
) -> int | float:
    """
    :return: ((2 * y * kS * w0 / po) ** uS_power) * ((scale_ks * z / scaling_divisor) ** a)
    """

    uS = (2 * y * kS * w0 / po) ** uS_power
    return uS * (scale_ks * z / scaling_divisor) ** a


@validate_call
def speed_sp_region(
        z: int | float,
        area_type: AlphaStandardsOrKs10Type,
        wind_region: WindRegionsType,
        *,
        p0: int | float = 1.225,
        yf: int | float = 1.4,
        scale_ks: int | float = 1,
        uS_power: float = 0.5,
        scaling_divisor: int | float = 10
) -> int | float:
    alpha = alpha_standards[area_type]
    kS = ks10[area_type]
    w0 = wind_regions[wind_region] * 1000
    a = speed_sp(z, scale_ks, alpha, kS, p0, yf, w0, uS_power=uS_power, scaling_divisor=scaling_divisor)

    return a


@validate_call
def pressure_coefficient_for_region(
        z: int | float,
        area_type: AlphaStandardsOrKs10Type,
        wind_region: WindRegionsType,
        *,
        p0: int | float = 1.225,
        yf: int | float = 1.4,
        scale_ks: int | float = 1,
        uS_power: float = 0.5,
        scaling_divisor: int | float = 10
) -> int | float:
    return (speed_sp_region(
        z, area_type, wind_region,
        p0=p0, yf=yf, scale_ks=scale_ks,
        uS_power=uS_power, scaling_divisor=scaling_divisor
    ) ** 2) * p0 / 2
