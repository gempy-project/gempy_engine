﻿def require_tensorflow():
    try:
        import tensorflow as tf
    except ImportError:
        raise ImportError("The tensorflow library is required to use this function.")
    return tf


def require_pandas():
    try:
        import pandas as pd
    except ImportError:
        raise ImportError("The pandas library is required to use this function.")
    return pd


def require_pooch():
    try:
        import pooch
    except ImportError:
        raise ImportError("The pooch library is required to use this function.")
    return pooch


def require_gempy_legacy():
    try:
        import gempy_legacy
    except ImportError:
        raise ImportError("The gempy_legacy library is required to use this function.")
    return gempy_legacy


def require_gempy_viewer():
    try:
        import gempy_viewer
    except ImportError:
        raise ImportError("The gempy_viewer package is required to run this function.")
    return gempy_viewer


def require_skimage():
    try:
        import skimage
    except ImportError:
        raise ImportError("The skimage package is required to run this function.")
    return skimage


def require_scipy():
    try:
        import scipy
    except ImportError:
        raise ImportError("The scipy package is required to run this function.")
    return scipy