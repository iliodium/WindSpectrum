import os
import enum
import json
import subprocess
import dataclasses
import os.path as path
from typing import List

import numpy as np


@dataclasses.dataclass()
class SuperElement:
    alpha: float = 1.
    beta: float = 1.
    rho: float = 1.
    A: float = 1.
    L: float = 1.
    J: float = 1.
    I_y: float = 1.
    I_z: float = 1.
    E: float = 1.
    G: float = 1.


class DynamicsExecutionMode(enum.IntEnum):
    MCD = 1
    NEWMARK = 2


@dataclasses.dataclass()
class DynamicsExecutionUnit:
    f_dat_file_path: str
    method: DynamicsExecutionMode = DynamicsExecutionMode.NEWMARK
    dt: float = 0.01
    N: int = 1
    elements: List[SuperElement] = dataclasses.field(default_factory=list)

    def __add__(self, other):
        if isinstance(other, SuperElement):
            self.elements.append(other)
            return self
        else:
            raise ValueError(f"Unsupported operand of addition of type: {None if other is None else other.__class__}")

    def __len__(self):
        return len(self.elements)

    def __call__(self, *args, **kwargs):
        raise ValueError("Not implemented!")


@dataclasses.dataclass()
class NewmarkExecutionUnit(DynamicsExecutionUnit):
    newmark_alpha: float = 1
    newmark_delta: float = 1

    def __add__(self, other):
        return super(NewmarkExecutionUnit, self).__add__(other)

    def __call__(self, *args, **kwargs) -> subprocess.Popen:
        assert len(args) == 2, "Count of arguments to start execution must be exactly one"
        if not isinstance(args[0], str):
            raise ValueError("Path to executable must be of str type!")
        if not path.exists(path.abspath(args[0])):
            raise ValueError(f"Not found executable given by path: {args[0]}")
        if not isinstance(args[1], str):
            raise ValueError("Path to config must be of str type!")
        path_to_executable = path.abspath(args[0])
        filename_of_config = path.abspath(args[1])
        print(f"Attempt to start execution {path_to_executable} with config {filename_of_config}")
        with open(filename_of_config, mode='w') as json_file:
            json.dump(dataclasses.asdict(self), json_file)
        return subprocess.Popen((path_to_executable, filename_of_config), stdin=subprocess.DEVNULL,
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE)


class FDMExecutionUtils:

    @staticmethod
    def start_execution(deu: DynamicsExecutionUnit, fs_dat: np.array, path_to_executable: str,
                        config_filename: str) -> subprocess.Popen:
        deu.f_dat_file_path = path.abspath(deu.f_dat_file_path)
        FDMExecutionUtils.write_dat_file(deu, fs_dat)
        return deu(path_to_executable, config_filename)

    @staticmethod
    def write_dat_file(deu: DynamicsExecutionUnit, fs_dat: np.array):
        lse = len(deu)
        sx = deu.N
        sy = 6 * lse
        shape = fs_dat.shape
        if len(shape) == 2:
            if shape[0] == sx and shape[1] == sy:
                np.savetxt(deu.f_dat_file_path, fs_dat, delimiter=' ')
            else:
                raise ValueError(f"fs_dat of wrong shape. Expected ({sx}, {sy}). Got {shape}")
        else:
            raise ValueError("fs_dat dimensions mismatch")


if __name__ == "__main__":
    import sys

    print("Test suite for class SuperElement")
    se = SuperElement()
    se.alpha = 0.5
    se.beta = 0.5
    se.rho = 1e5
    se.A = 1
    se.L = 5
    se.J = 1
    se.I_y = 0.1
    se.I_z = 0.1
    se.E = 1.75e7
    se.G = 1.05e11

    neu = NewmarkExecutionUnit(r"./test.dat")

    neu.dt = 0.05
    neu += se
    neu += se
    neu.N = 100000
    subprocess_opened = FDMExecutionUtils.start_execution(
        neu,
        np.array([list(range(12)) for _ in range(neu.N)]),
        os.getenv("MECHANICS_SOLVER"),
        "./example.json"
    )

    # hack to wait till finish of subprocess
    while subprocess_opened.poll():
        pass

    # last line is about time elapsed. contained as binary string!
    print(subprocess_opened.stdout.readlines()[-1].decode('ascii'))
