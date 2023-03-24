import ctypes
import math
from pathlib import Path

import numpy as np


_modelib = ctypes.cdll.LoadLibrary(
    (Path(__file__).parent / "lib/build/libmode.so").resolve()
)
_cxxmode = _modelib.half_width_mode
_cxxmode.restype = ctypes.c_double
_cxxmode.argtypes = [
    np.ctypeslib.ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),
    ctypes.c_size_t,
]

_cxxmode_f = _modelib.half_width_mode_float
_cxxmode_f.restype = ctypes.c_float
_cxxmode_f.argtypes = [
    np.ctypeslib.ndpointer(ctypes.c_float, flags="C_CONTIGUOUS"),
    ctypes.c_size_t,
]


def mode(x, is_sorted=False):
    def dispatch_dtype(array):
        if array.dtype == np.float64:
            return _cxxmode(array, array.size)
        elif array.dtype == np.float32:
            return _cxxmode_f(array, array.size)
        else:
            raise RuntimeError("mode only accepts float64 and float32")

    x_array = np.asanyarray(x)
    if not is_sorted:
        x_array.sort()

    return dispatch_dtype(x_array)


def rms(x):
    """
    Root mean square of a sample
    """
    return np.sqrt(np.mean(np.square(x)))
