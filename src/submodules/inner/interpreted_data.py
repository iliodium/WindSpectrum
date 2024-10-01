import pathlib

import numpy as np
import scipy.interpolate

current_data_path = pathlib.Path(__file__) \
    .absolute() \
    .parent \
    .parent \
    .parent \
    .parent \
    .joinpath("resources") \
    .joinpath("data")

x_025 = list()
y_025 = list()

with open(current_data_path.joinpath('Uz_a_0_25.txt'), mode='r') as file:
    f_025 = file.read().split('\n')

for i in f_025:
    x, y = list(map(float, i.split()))
    x_025.append(x)
    y_025.append(y)

with open(current_data_path.joinpath('Uz_a_0_16.txt'), mode='r') as file:
    f_016 = file.read().split('\n')

x_016 = list()
y_016 = list()

for i in f_016:
    x, y = list(map(float, i.split()))
    x_016.append(x)
    y_016.append(y)

del f_016
del f_025

UH = 11
x_016 = np.array(x_016) * UH
x_025 = np.array(x_025) * UH

y_016 = np.array(y_016) / 100
y_025 = np.array(y_025) / 100

interp_016_tpu = scipy.interpolate.interp1d(y_016, x_016)
interp_025_tpu = scipy.interpolate.interp1d(y_025, x_025)

interp_016_tpu_400 = scipy.interpolate.interp1d(y_016 * 400, x_016)
interp_025_tpu_400 = scipy.interpolate.interp1d(y_025 * 400, x_025)

interp_016_real_tpu = scipy.interpolate.interp1d(y_016 / 100, x_016)
interp_025_real_tpu = scipy.interpolate.interp1d(y_025 / 100, x_025)
