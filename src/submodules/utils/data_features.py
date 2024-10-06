import numpy as np
from pydantic import validate_call


@validate_call
def rms(data: np.ndarray) -> float:
    """Среднеквадратичное отклонение"""
    return np.sqrt(np.array(data).dot(np.array(data)) / np.array(data).size).round(2)


@validate_call
def rach(data: np.ndarray) -> float:
    """Расчетное"""
    return np.max([np.abs(np.min(data)), np.abs(np.max(data))]).round(2)


@validate_call
def obes_p(data: np.ndarray) -> float:
    """Обеспеченность +"""
    return (np.abs(np.max(data) - np.mean(data)) / np.std(data)).round(2)


@validate_call
def obes_m(data: np.ndarray) -> float:
    """Обеспеченность -"""
    return (np.abs(np.min(data) - np.mean(data)) / np.std(data)).round(2)
