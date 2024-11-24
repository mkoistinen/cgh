import numpy as np
from typing_extensions import deprecated


@deprecated("No longer to be used.")
def get_numpy_precision_types(force_64bit=False):
    supported_types = {}
    try:
        _ = np.float128(0)
        supported_types['float128'] = True
    except AttributeError:
        supported_types['float128'] = False

    try:
        _ = np.complex256(0)
        supported_types['complex256'] = True
    except AttributeError:
        supported_types['complex256'] = False

    if (
        not force_64bit and
        supported_types["float128"] and
        supported_types["complex256"]
    ):
        return np.float128, np.complex256
    else:
        return np.float64, np.complex128


FLOAT, COMPLEX = np.float64, np.complex128
