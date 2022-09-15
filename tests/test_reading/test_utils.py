import datetime
import numpy as np
import xarray as xr

from qa4sm_preprocessing.reading.utils import (
    mkdate,
    nimages_for_memory,
    str2bool,
    infer_chunksizes,
    numpy_timeoffsetunit,
)


def test_mkdate():
    assert mkdate("2000-12-17") == datetime.datetime(2000, 12, 17)
    assert mkdate("2000-12-17T01:56") == datetime.datetime(2000, 12, 17, 1, 56)
    assert mkdate("2000-12-17T01:56:33") == datetime.datetime(2000, 12, 17, 1, 56, 33)


def test_str2bool():
    for val in ["True", "true", "t", "T", "1", "yes", "y"]:
        assert str2bool(val)
    for val in ["False", "false", "f", "F", "0", "no", "n"]:
        assert not str2bool(val)


def test_nimages_for_memory():

    nx, ny, nz = 100, 100, 10
    ds = xr.Dataset(
        {
            "X": (("x", "y", "z"), np.random.randn(nx, ny, nz).astype(np.float64)),
            "Y": (("x", "y", "z"), np.random.randn(nx, ny, nz).astype(np.float32)),
            "Z": (("x", "y", "z"), np.random.randn(nx, ny, nz).astype(np.float32)),
        }
    )

    size_X = nx * ny * nz * 8
    size_Y = nx * ny * nz * 4
    totalsize = size_X + 2 * size_Y

    # set memory to get 8 images when using full dataset
    memory = totalsize * 8 / 1024 ** 3
    nimages = nimages_for_memory(ds, memory)
    assert nimages == 8

    # when using only X, we can get twice as many images into the same memory
    nimages = nimages_for_memory(ds[["X"]], memory)
    assert nimages == 16
    # same when using Y and Z
    nimages = nimages_for_memory(ds[["Y", "Z"]], memory)
    assert nimages == 16
    # when using only Y, 32 images should be possible
    nimages = nimages_for_memory(ds[["Y"]], memory)
    assert nimages == 32


def test_infer_chunksizes():

    dimsizes = (1000, 1000, 1000)
    chunksizes = (100, 100, 1000)

    target_size = 8 * np.prod(chunksizes) / 1024 ** 2
    assert infer_chunksizes(dimsizes, target_size, np.float64) == chunksizes
    assert infer_chunksizes(dimsizes, target_size, 8) == chunksizes

    target_size = 4 * np.prod(chunksizes) / 1024 ** 2
    assert infer_chunksizes(dimsizes, target_size, np.float32) == chunksizes
    assert infer_chunksizes(dimsizes, target_size, 4) == chunksizes


def test_numpy_timeoffsetunit():
    assert numpy_timeoffsetunit("seconds") == "s"
    assert numpy_timeoffsetunit("minutes") == "m"
    assert numpy_timeoffsetunit("hours") == "h"
    assert numpy_timeoffsetunit("days") == "D"
