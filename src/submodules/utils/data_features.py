import numpy as np
from pydantic import validate_call

from src.ui.common.ChartMode import ChartMode


@validate_call
def rms(data) -> float:
    """Среднеквадратичное отклонение"""
    return np.sqrt(np.array(data).dot(np.array(data)) / np.array(data).size).round(2)


@validate_call
def calculated(data) -> float:
    """Расчетное"""
    return np.max([np.abs(np.min(data)), np.abs(np.max(data))]).round(2)


@validate_call
def warranty_plus(data) -> float:
    """Обеспеченность +"""
    return (np.abs(np.max(data) - np.mean(data)) / np.std(data)).round(2)


@validate_call
def warranty_minus(data) -> float:
    """Обеспеченность -"""
    return (np.abs(np.min(data) - np.mean(data)) / np.std(data)).round(2)


lambdas = {
    ChartMode.MAX: lambda coefficients: np.max(coefficients, axis=0),
    ChartMode.MEAN: lambda coefficients: np.mean(coefficients, axis=0),
    ChartMode.MIN: lambda coefficients: np.min(coefficients, axis=0),
    ChartMode.STD: lambda coefficients: np.std(coefficients, axis=0),
    ChartMode.RMS: lambda coefficients: np.array([np.sqrt(i.dot(i) / i.size) for i in coefficients.T]),
}
