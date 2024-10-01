import numpy as np


def rms(data: np.ndarray) -> float:
    """Среднеквадратичное отклонение"""
    return np.sqrt(np.array(data).dot(np.array(data)) / np.array(data).size).round(2)


def rach(data: np.ndarray) -> float:
    """Расчетное"""
    return np.max([np.abs(np.min(data)), np.abs(np.max(data))]).round(2)


def obes_p(data: np.ndarray) -> float:
    """Обеспеченность +"""
    return (np.abs(np.max(data) - np.mean(data)) / np.std(data)).round(2)


def obes_m(data: np.ndarray) -> float:
    """Обеспеченность -"""
    return (np.abs(np.min(data) - np.mean(data)) / np.std(data)).round(2)
