"""
СП 20. Нагрузки и воздействия
"""
import numpy as np

from src.common.constants import wind_regions, alpha_standards, ks10


def speed_sp(
        z: int | float | np.ndarray, scale_ks: int | float,
        a: float, kS: float, po: float, y: float, w0: int | float, *,
        uS_power: float = 0.5, scaling_divisor: int | float = 10
) -> int | float | np.ndarray:
    """
    :return: ((2 * y * kS * w0 / po) ** uS_power) * ((scale_ks * z / scaling_divisor) ** a)
    """
    assert isinstance(z, (int, float, np.ndarray)), "z must be int, float or np.ndarray"
    assert isinstance(scale_ks, (int, float)), "scale_ks must be int or float"
    assert isinstance(a, float), "a must be float"
    assert isinstance(kS, float), "kS must be float"
    assert isinstance(po, float), "po must be float"
    assert isinstance(y, float), "y must be float"
    assert isinstance(w0, (int, float)), "W0 must be int or float"
    assert isinstance(uS_power, float), "uS_power must be float"
    assert isinstance(scaling_divisor, (int, float)), "scaling_divisor must be int or float"

    uS = (2 * y * kS * w0 / po) ** uS_power
    return uS * (scale_ks * z / scaling_divisor) ** a


def speed_sp_region(
        z: int | float | np.ndarray, area_type: str, wind_region: str, *,
        p0: int | float = 1.225, yf: int | float = 1.4, scale_ks: int | float = 1,
        uS_power: float = 0.5, scaling_divisor: int | float = 10
) -> int | float | np.ndarray:
    assert isinstance(z, (int, float, np.ndarray)), "z must be int, float or np.ndarray"
    assert isinstance(area_type, str), "region must be str"
    assert area_type in (alpha_standards.keys() & ks10.keys()), "region must be in alpha_standards"
    assert isinstance(wind_region, str), "wind_region must be str"
    assert wind_region in wind_regions.keys(), "wind_region must be in wind_regions"
    assert isinstance(p0, (int, float)), "p0 must be int or float"
    assert isinstance(yf, (int, float)), "yf must be int or float"
    assert isinstance(scale_ks, (int, float)), "ks must be int or float"
    assert isinstance(uS_power, float), "uS_power must be float"
    assert isinstance(scaling_divisor, (int, float)), "scaling_divisor must be int or float"

    alpha = alpha_standards[area_type]
    kS = ks10[area_type]
    w0 = wind_regions[wind_region] * 1000

    return speed_sp(z, scale_ks, alpha, kS, p0, yf, w0, uS_power=uS_power, scaling_divisor=scaling_divisor)


def pressure_coefficient_for_region(
        z: int | float | np.ndarray, area_type: str, wind_region: str, *,
        p0: int | float = 1.225, yf: int | float = 1.4, scale_ks: int | float = 1,
        uS_power: float = 0.5, scaling_divisor: int | float = 10
) -> int | float | np.ndarray:
    assert isinstance(z, (int, float, np.ndarray)), "z must be int, float or np.ndarray"
    assert isinstance(area_type, str), "region must be str"
    assert area_type in (alpha_standards.keys() & ks10.keys()), "region must be in alpha_standards"
    assert isinstance(wind_region, str), "wind_region must be str"
    assert wind_region in wind_regions.keys(), "wind_region must be in wind_regions"
    assert isinstance(p0, (int, float)), "p0 must be int or float"
    assert isinstance(yf, (int, float)), "yf must be int or float"
    assert isinstance(scale_ks, (int, float)), "ks must be int or float"
    assert isinstance(uS_power, float), "uS_power must be float"
    assert isinstance(scaling_divisor, (int, float)), "scaling_divisor must be int or float"
    return (speed_sp_region(
        z, area_type, wind_region,
        p0=p0, yf=yf, scale_ks=scale_ks,
        uS_power=uS_power, scaling_divisor=scaling_divisor
    ) ** 2) * p0 / 2
