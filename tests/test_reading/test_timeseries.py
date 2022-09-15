import numpy as np
import pandas as pd
import shutil
import xarray as xr

from qa4sm_preprocessing.reading import StackTs, GriddedNcOrthoMultiTs, StackImageReader

import pytest
from pytest import test_data_path


def test_StackTs(regular_test_dataset):
    reader = StackTs(regular_test_dataset, "X")
    gpis, lons, lats, _ = reader.grid.get_grid_points()
    for gpi, lon, lat in zip(gpis, lons, lats):
        ref = regular_test_dataset.X.sel(lat=lat, lon=lon)
        ts = reader.read(lon, lat)["X"]
        assert np.all(ts == ref)
        ts = reader.read(gpi)["X"]
        assert np.all(ts == ref)


def test_StackTs_locdim(unstructured_test_dataset):
    reader = StackTs(unstructured_test_dataset, "X", locdim="location")
    gpis, lons, lats, _ = reader.grid.get_grid_points()
    for gpi, lon, lat in zip(gpis, lons, lats):
        ts = reader.read(gpi)["X"]
        ref = unstructured_test_dataset.X.isel(location=gpi)
        assert np.all(ts == ref)
        ts = reader.read(lon, lat)["X"]
        assert np.all(ts == ref)


def test_GriddedNcOrthoMultiTs(synthetic_test_args):

    ds, kwargs = synthetic_test_args
    stack = StackImageReader(ds, ["X", "Y"], **kwargs)

    tspath = test_data_path / "ts_test_path"
    tsreader = stack.repurpose(tspath, overwrite=True)

    gpis, lons, lats, _ = tsreader.grid.get_grid_points()
    for gpi, lon, lat in zip(gpis, lons, lats):
        for var in ["X", "Y"]:
            ref = ds[var].where((ds.lat == lat) & (ds.lon == lon), drop=True).squeeze()
            ts = tsreader.read(gpi)[var]
            assert np.all(ts == ref)
            ts = tsreader.read(lon, lat)[var]
            assert np.all(ts == ref)

    # manually create tsreader and test read_bulk logic
    assert tsreader.ioclass_kws["read_bulk"] is True
    tsreader = GriddedNcOrthoMultiTs(tspath, read_bulk=False)
    assert tsreader.ioclass_kws["read_bulk"] is False
    tsreader = GriddedNcOrthoMultiTs(tspath, ioclass_kws={"read_bulk": False})
    assert tsreader.ioclass_kws["read_bulk"] is False
    tsreader = GriddedNcOrthoMultiTs(
        tspath, ioclass_kws={"read_bulk": False}, read_bulk=False
    )
    assert tsreader.ioclass_kws["read_bulk"] is False
    with pytest.warns(UserWarning, match="read_bulk=False but"):
        tsreader = GriddedNcOrthoMultiTs(
            tspath, ioclass_kws={"read_bulk": True}, read_bulk=False
        )
        assert tsreader.ioclass_kws["read_bulk"] is False


def test_StackTs_timeoffset(synthetic_test_args):
    ds, kwargs = synthetic_test_args
    ds["time_offset"] = xr.ones_like(ds.X)
    reader = StackTs(
        ds, timeoffsetvarname="time_offset", timeoffsetunit="seconds", **kwargs
    )

    df = reader.read(0)
    assert list(df.columns) == ["X", "Y"]
    assert np.all(df.index == (ds.indexes["time"] + pd.Timedelta("1s")))


def test_StackTs_timevar(synthetic_test_args):
    ds, kwargs = synthetic_test_args
    newtime = xr.ones_like(ds.X)
    newtime = (newtime.T + np.arange(len(ds.time))*86400).T
    newtime.attrs["units"] = "seconds since 2000-01-01"
    newtime.attrs["long_name"] = "exact observation time"
    ds["exact_time"] = newtime
    # compared to test_StackTs_timeoffset we specify the varnames explicitly
    # here, so we cover also the case where varnames are given but the
    # timevarname is not included
    reader = StackTs(
        ds, ["X", "Y"], timevarname="exact_time", **kwargs
    )
    df = reader.read(0)
    assert list(df.columns) == ["X", "Y"]
    assert np.all(df.index == (ds.indexes["time"] + pd.Timedelta("1s")))


def test_GriddedNcOrthoMultiTs_timeoffset(synthetic_test_args):

    ds, kwargs = synthetic_test_args
    ds["time_offset"] = xr.ones_like(ds.X)
    stack = StackImageReader(ds, **kwargs)

    tspath = test_data_path / "ts_test_path"
    # time_offset should be detected by default
    tsreader = stack.repurpose(tspath, overwrite=True)

    df = tsreader.read(0)
    assert list(df.columns) == ["X", "Y"]
    assert np.all(df.index == (ds.indexes["time"] + pd.Timedelta("1s")))


def test_GriddedNcOrthoMultiTs_timevar(synthetic_test_args):

    ds, kwargs = synthetic_test_args
    newtime = xr.ones_like(ds.X)
    newtime = (newtime.T + np.arange(len(ds.time))*86400).T
    newtime.attrs["units"] = "seconds since 2000-01-01"
    newtime.attrs["long_name"] = "exact observation time"
    ds["exact_time"] = newtime
    stack = StackImageReader(ds, **kwargs)

    tspath = test_data_path / "ts_test_path"
    tsreader = stack.repurpose(tspath, timevarname="exact_time", overwrite=True)

    df = tsreader.read(0)
    assert list(df.columns) == ["X", "Y"]
    assert np.all(df.index == (ds.indexes["time"] + pd.Timedelta("1s")))
