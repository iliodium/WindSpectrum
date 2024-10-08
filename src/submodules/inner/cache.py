import os
from functools import (cache,
                       lru_cache,
                       wraps, )

import numpy as np


# https://stackoverflow.com/questions/52331944/cache-decorator-for-numpy-arrays
def np_cache(function):
    @lru_cache()
    def cached_wrapper(*args, **kwargs):
        args = [np.array(a) if isinstance(a, tuple) else a for a in args]
        kwargs = {
            k: np.array(v) if isinstance(v, tuple) else v for k, v in kwargs.items()
        }

        return function(*args, **kwargs)

    @wraps(function)
    def wrapper(*args, **kwargs):
        '''
        Конвертирует вложенный np.array во вложенный tuple
        '''
        args = [tuple(tuple(sub) for sub in a) if isinstance(a, np.ndarray) else a for a in args]
        kwargs = {
            k: tuple(tuple(sub) for sub in v) if isinstance(v, np.ndarray) else v for k, v in kwargs.items()
        }
        return cached_wrapper(*args, **kwargs)

    # copy lru_cache attributes over too
    wrapper.cache_info = cached_wrapper.cache_info
    wrapper.cache_clear = cached_wrapper.cache_clear

    return wrapper


def cache_decorator(
        *,
        disabled=os.environ.get("DISABLE_CACHE", False)
):
    if isinstance(disabled, str):
        match disabled:
            case '1' | 'y' | 'Y':
                disabled = True
            case '0' | 'n' | 'N':
                disabled = False
            case _:
                disabled = bool(disabled)
    elif not isinstance(disabled, bool):
        disabled = bool(disabled)

    if disabled:
        return lambda _f: _f
    return lru_cache(typed=True, maxsize=32)
