import datetime
import logging
from math import floor
import numpy as np
from typing import Sequence
import xarray as xr


def mkdate(datestring):
    if len(datestring) == 10:
        return datetime.datetime.strptime(datestring, "%Y-%m-%d")
    elif len(datestring) == 16:
        return datetime.datetime.strptime(datestring, "%Y-%m-%dT%H:%M")
    elif len(datestring) == 19:
        return datetime.datetime.strptime(datestring, "%Y-%m-%dT%H:%M:%S")
    else:  # pragma: no cover
        raise ValueError(f"Invalid date: {datestring}")


def str2bool(val):  # pragma: no cover
    if val in ["True", "true", "t", "T", "1", "yes", "y"]:
        return True
    else:
        return False


def nimages_for_memory(img: xr.Dataset, memory: float):
    """
    Infers the number of images that fits into memory.

    Parameters
    ----------
    img : xr.Dataset
    memory : float
        Size of available memory in GB
    """
    imagesize = img.nbytes / 1024**3
    return int(np.floor(memory / imagesize))


def infer_chunksizes(dimsizes: Sequence, target_size: float, dtype) -> tuple:
    """
    Calculates a good chunk size for the given dimensions using a contiguous
    chunk for the last dimension.

    Parameters
    ----------
    dimsizes : Sequence
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
    if isinstance(dtype, int):
        dtype_size = dtype
    else:
        dtype_size = np.dtype(dtype).itemsize
    # if we imagine a chunk to be a cuboid, and the last (contiguous) dimension
    # the height, then 'chunk_size' is the size of the base area (in number of
    # elements)
    chunk_size = target_size / dtype_size / dimsizes[-1] * 1024 * 1024
    # we can get the size of a single base side by taking the square root,
    # i.e. **(1/2), or in general, **(1/(d-1))
    d = len(dimsizes)
    base_size = int(floor(chunk_size ** (1 / (d - 1))))
    # use base size or dim size, whichever is smaller
    chunks = [min(base_size, dimsizes[i]) for i in range(d - 1)]
    # the last chunks should be contiguous
    chunks.append(dimsizes[-1])
    return tuple(chunks)


def numpy_timeoffsetunit(unit):
    assert isinstance(unit, str)
    unit = unit.lower()[0]
    if unit == "d":
        unit = "D"
    assert unit in ["s", "m", "h", "D"]
    return unit


def infer_cellsize(grid):
    deltalat = np.max(grid.activearrlat) - np.min(grid.activearrlat)
    deltalon = np.max(grid.activearrlon) - np.min(grid.activearrlon)
    cellsize = 30 * np.sqrt(deltalat * deltalon / len(grid.activegpis))
    logging.info(f"Inferred cell size for cell grid: {cellsize:.3f}°")
    if cellsize == 0. :
        cellsize = 180
    return cellsize
