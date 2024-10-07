import os
from functools import lru_cache


def cache_decorator(
        *,
        disabled=os.environ.get("DISABLE_CACHE", False)
):
    if isinstance(disabled, str):
        if disabled == '1':
            disabled = True
        elif disabled == '0':
            disabled = False
        elif disabled == 'y' or disabled == 'Y':
            disabled = True
        elif disabled == 'n' or disabled == 'N':
            disabled = False
        else:
            disabled = bool(disabled)
    if not isinstance(disabled, bool):
        disabled = bool(disabled)
    if disabled:
        return lambda _f: _f
    return lru_cache(typed=True, maxsize=32)
