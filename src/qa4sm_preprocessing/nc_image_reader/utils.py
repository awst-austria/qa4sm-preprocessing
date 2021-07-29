import datetime
from typing import Sequence
from math import floor, sqrt
import numpy as np


def mkdate(datestring):
    if len(datestring) == 10:
        return datetime.datetime.strptime(datestring, "%Y-%m-%d")
    elif len(datestring) == 16:
        return datetime.datetime.strptime(datestring, "%Y-%m-%dT%H:%M")
    else:  # pragma: no cover
        raise ValueError(f"Invalid date: {datestring}")


def str2bool(val):
    if val in ["True", "true", "t", "T", "1", "yes", "y"]:
        return True
    else:
        return False


def infer_chunks(dim_sizes: Sequence, target_size: float, dtype) -> tuple:
    """
    Calculates a good chunk size for the given dimensions using a continuous
    chunk for the last dimension.

    Parameters
    ----------
    dim_sizes : Sequence
        Dimension sizes.
    target_size : float
        Target size of a single chunk in MB. Note that for NetCDF files on
        disk, the chunk size should be <32 MB (so rather in single digit MBs),
        while for dask it should be probably around a few hundred MBs.
    dtype : dtype
        Datatype of the array, e.g. np.float32

    Returns
    -------
    chunks : tuple
    """
    dtype_size = np.dtype(dtype).itemsize
    image_size = target_size / dtype_size / dim_sizes[-1] * 1024 * 1024
    chunk_size = int(floor(sqrt(image_size)))
    chunks = [min(chunk_size, dim_sizes[i]) for i in range(len(dim_sizes) - 1)]
    chunks.append(dim_sizes[-1])
    return tuple(chunks)
