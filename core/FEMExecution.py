import logging
import os.path as path
import numpy as np


class SuperElement(dict):
    __slots__ = [
        'alpha', 'beta', 'rho',
        'a', 'length',
        'j', 'i_y', 'i_z',
        'e', 'g'
    ]

    def __init__(
            self,
            alpha: float = 0., beta: float = 0., rho: float = 0.,
            a: float = 1., length: float = 1.,
            j: float = 1., i_y: float = 0.1, i_z: float = 0.1,
            e: float = 1.75e7, g: float = 1.05e11
    ):
        super().__init__(self)
        self['alpha'] = alpha
        self['beta'] = beta
        self['rho'] = rho
        self['A'] = a
        self['L'] = length
        self['J'] = j
        self['I_y'] = i_y
        self['I_z'] = i_z
        self['E'] = e
        self['G'] = g

    def __getattr__(self, item):
        if item == 'a':
            return self['A']
        elif item == 'length':
            return self['L']
        elif item == 'j':
            return self['J']
        elif item == 'i_y':
            return self['I_y']
        elif item == 'i_z':
            return self['I_z']
        elif item == 'e':
            return self['E']
        elif item == 'g':
            return self['G']
        return self[item]

    def __setattr__(self, item, value):
        super().__setattr__(item, value)
        if item == 'a':
            self['A'] = value
        elif item == 'length':
            self['L'] = value
        elif item == 'j':
            self['J'] = value
        elif item == 'i_y':
            self['I_y'] = value
        elif item == 'i_z':
            self['I_z'] = value
        elif item == 'e':
            self['E'] = value
        elif item == 'g':
            self['G'] = value
        else:
            self[item] = value

    def __call__(self, *args, **kwargs) -> bool:
        return self.__validate()

    def __validate(self) -> bool:
        if not isinstance(self.alpha, (float, int)):
            return False
        elif not isinstance(self.beta, (float, int)):
            return False
        elif not isinstance(self.rho, (float, int)):
            return False
        elif not isinstance(self.a, (float, int)):
            return False
        elif not isinstance(self.length, (float, int)):
            return False
        elif not isinstance(self.j, (float, int)):
            return False
        elif not isinstance(self.i_y, (float, int)):
            return False
        elif not isinstance(self.i_z, (float, int)):
            return False
        elif not isinstance(self.e, (float, int)):
            return False
        elif not isinstance(self.g, (float, int)):
            return False
        return True


class DynamicsExecutionMode:

    MCD = 1
    NEWMARK = 2


class DynamicsExecutionUnit(dict):

    __slots__ = [
        'method',
        'dt',
        'n',
        'f_dat',
        'super_elements'
    ]

    def __init__(
            self,
            method=DynamicsExecutionMode.NEWMARK,
            dt: float = 0.01, n: int = 1, f_dat: str = None,
            initial_super_elements: None | list[SuperElement] = None
    ):
        super().__init__()
        if initial_super_elements is None:
            self['elements'] = list()
        else:
            self['elements'] = list(initial_super_elements)
        self['method'] = method
        self['dt'] = dt
        self['N'] = n
        self['f_dat_file_path'] = f_dat

    def __setattr__(self, key, value):
        super().__setattr__(key, value)
        if key == 'n':
            self['N'] = value
        elif key == 'f_dat':
            self['f_dat_file_path'] = value
        else:
            self[key] = value

    def __getattr__(self, key):
        if key == 'n':
            return self['N']
        elif key == 'f_dat':
            return self['f_dat_file_path']
        else:
            return self[key]

    def __add__(self, item):
        if not isinstance(item, SuperElement):
            raise IndexError("Attempt to add non super element")
        self['elements'].append(item)
        return self

    def __getitem__(self, item):
        if isinstance(item, int):
            return self['elements'][item]
        elif isinstance(item, str):
            return super().__getitem__(item)
        raise IndexError("Invalid index passed!")

    def __delitem__(self, key):
        if isinstance(key, int):
            if key >= 0 and key < len(self['elements']):
                del self['elements'][key]
            else:
                logging.warning("Index({0}) out of bounds for list of length {1}", )
        else:
            raise IndexError("Not supported indexing other than by int index!")

    def _validate(self) -> bool:
        if self['method'] != DynamicsExecutionMode.MCD and self['method'] != DynamicsExecutionMode.NEWMARK:
            logging.error("Method not determined")
            return False
        elif not isinstance(self['dt'], float):
            logging.error("dt is not of type float")
            return False
        elif not isinstance(self['N'], int):
            logging.error("N is not of type int")
            return False
        elif self['f_dat_file_path'] is None:
            logging.error("f_dat not specified")
            return False
        elif not path.exists(self['f_dat_file_path']):
            logging.error("f_dat path does not exist. f_dat = %s", self["f_dat_file_path"])
            return False
        elif not path.isfile(self['f_dat_file_path']):
            logging.error("f_dat path is not path to file")
            return False
        return True

    def __len__(self):
        return len(self['elements'])

    def __call__(self, *args, **kwargs):
        return self._validate()


class NewmarkExecutionUnit(DynamicsExecutionUnit):

    __slots__ = [
        'newmark_alpha',
        'newmark_delta'
    ]

    def __init__(self, newmark_alpha=1, newmark_delta=1, **kwargs):
        super().__init__(**kwargs, method=DynamicsExecutionMode.NEWMARK)
        self['newmark_alpha'] = newmark_alpha
        self['newmark_delta'] = newmark_delta

    def _validate(self) -> bool:
        if self['newmark_alpha'] is None:
            return False
        elif self['newmark_delta'] is None:
            return False
        return super()._validate()

    def __call__(self, *args, **kwargs):
        if self._validate():
            print("Can run")
        else:
            raise ValueError("Check configuration!")


class FDMExecutionUtils:

    @staticmethod
    def write_dat_file(deu: DynamicsExecutionUnit, fs_dat: np.array, filename: str):
        lse = len(deu)
        sx = deu.n
        sy = 6 * lse
        shape = fs_dat.shape
        if len(shape) == 2:
            if shape[0] == sx and shape[1] == sy:
                np.savetxt(filename, fs_dat, delimiter=' ')
            else:
                raise ValueError(f"fs_dat of wrong shape. Expected ({sx}, {sy}). Got {shape}")
        else:
            raise ValueError("fs_dat dimensions mismatch")


if __name__ == "__main__":
    import json
    import sys

    print("Test suite for class SuperElement")
    se = SuperElement()
    se.alpha = 0.5
    se.beta = 0.5
    se.rho = 1e5
    se.a = 1
    se.length = 5
    se.j = 1
    se.i_y = 0.1
    se.i_z = 0.1
    se.e = 1.75e7
    se.g = 1.05e11
    try:
        se.fake = 3
        raise IndexError("Fake field indexed")
    except IndexError:
        exit(2)
    except AttributeError as e:
        print("Fake field test success!")
    print(se.g)
    print(json.dumps(se))
    print(se())

    print(sys.getsizeof(se))

    neu = NewmarkExecutionUnit()
    neu.dt = 0.05
    neu.f_dat = r"C:\Users\Sergio\Desktop\mechanics-master\fs.dat"
    neu += se
    neu += se
    print(len(neu))
    print(neu.dt)
    print(json.dumps(neu))
    print(neu())
    FDMExecutionUtils.write_dat_file(neu, np.array([[1, 2, 3], [4, 5, 6]]), "./test.dat")

