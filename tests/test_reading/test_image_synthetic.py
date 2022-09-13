# This module contains tests for the DirectoryImageReader based on synthetic datasets
#
# The tests typically follow this pattern:
# - take fixture as input dataset
# - modify a copy of the input dataset
# - write to image files
# - read image files so that the original input dataset is returned
# - check if reader returns images/blocks as expected (data + metadata)
#
# Features to test
# - [X] basic functionality: directory, fmt, basics of other grids
# - [X] not all variables
# - [X] renaming of variables
# - [X] skipping missing variables
# - [ ] skip missing with levels and rename
# - [X] not reading files that don't match the pattern
# - [X] using a regex pattern for getting the time information
# - [X] extract a level (multiple options need to be tested)
# - [X] discard attributes
# - [X] fill_value: replace some values by -9999 and make it a fill value
# - [X] non-default coordinate names (lat, lon, time)
# - [X] get lat/lon from array
# - [X] construct_grid
# - [X] averaging of subdaily timestamps
# - [X] multiple timestamps in a file
# - [X] transpose dataset so time is first when there are multiple timestamps
#       in a file
#
# Untested features
# - get lat/lon from (start, stop, step) tuple: this is tested with the
#   lis-noahmp test images
# - landmask, bbox, cellsize: covered in StackImageReader tests

import numpy as np
import pandas as pd
import pytest
import shutil
import xarray as xr


from qa4sm_preprocessing.reading import (
    DirectoryImageReader,
)
from qa4sm_preprocessing.reading.write import write_images

# this is defined in conftest.py
from pytest import test_data_path

from .utils import validate_reader


def make_clean_testdir():
    path = test_data_path / "synthetic"
    if path.exists():
        shutil.rmtree(path)



def test_directory_image_reader_basic(synthetic_test_args):
    ds, kwargs = synthetic_test_args
    make_clean_testdir()
    write_images(ds, test_data_path / "synthetic", "synthetic")
    reader = DirectoryImageReader(
        test_data_path / "synthetic",
        ["X", "Y"],
        fmt="synthetic_%Y%m%dT%H%M.nc",
        **kwargs
    )
    validate_reader(reader, ds)
    make_clean_testdir()


def test_directory_image_reader_incomplete_varnames(synthetic_test_args):
    # tests whether reading a single variable also works as expected
    ds, kwargs = synthetic_test_args
    ds = ds[["X"]]
    make_clean_testdir()
    write_images(ds, test_data_path / "synthetic", "synthetic")
    reader = DirectoryImageReader(
        test_data_path / "synthetic", ["X"], fmt="synthetic_%Y%m%dT%H%M.nc", **kwargs
    )
    validate_reader(reader, ds)
    make_clean_testdir()


def test_directory_image_reader_renaming(synthetic_test_args):
    # tests whether renaming works
    ds, kwargs = synthetic_test_args
    test_ds = ds.rename({"X": "newX", "Y": "newY"})
    make_clean_testdir()
    write_images(test_ds, test_data_path / "synthetic", "synthetic")
    reader = DirectoryImageReader(
        test_data_path / "synthetic",
        ["X", "Y"],
        rename={"newX": "X", "newY": "Y"},
        fmt="synthetic_%Y%m%dT%H%M.nc",
        **kwargs
    )
    validate_reader(reader, ds)
    make_clean_testdir()


def test_directory_image_reader_renaming_level(synthetic_test_args):
    ds, kwargs = synthetic_test_args
    new_ds = xr.concat((ds, ds * 2), dim="level").transpose(..., "level")
    new_ds = new_ds.rename({"X": "newX", "Y": "newY"})
    make_clean_testdir()
    write_images(new_ds, test_data_path / "synthetic", "synthetic")
    reader = DirectoryImageReader(
        test_data_path / "synthetic",
        ["X", "Y"],
        fmt="synthetic_%Y%m%dT%H%M.nc",
        rename={"newX": "X", "newY_0": "Y"},
        # testing selecting single level ("0") and multiple levels ("[0]")
        level={"newX": {"level": 0}, "newY": {"level": [0]}},
        **kwargs
    )
    validate_reader(reader, ds)
    make_clean_testdir()


def test_directory_image_reader_missing_varname(synthetic_test_args):
    # tests whether skipping missing variables works
    ds, kwargs = synthetic_test_args
    ds = ds[["X"]]
    make_clean_testdir()
    write_images(ds, test_data_path / "synthetic", "synthetic")
    with pytest.warns(
        UserWarning, match="Skipping variable 'Y' because it does not exist!"
    ):
        # skipping a variable raises a warning
        reader = DirectoryImageReader(
            test_data_path / "synthetic",
            ["X", "Y"],
            fmt="synthetic_%Y%m%dT%H%M.nc",
            skip_missing=True,
            **kwargs
        )
    validate_reader(reader, ds)
    make_clean_testdir()


def test_directory_image_reader_missing_varname_level_rename(synthetic_test_args):
    # tests whether skipping missing variables works
    ds, kwargs = synthetic_test_args
    ds = ds[["X"]]
    new_ds = xr.concat((ds, ds * 2), dim="level").transpose(..., "level")
    new_ds = new_ds.rename({"X": "newX"})
    make_clean_testdir()
    write_images(new_ds, test_data_path / "synthetic", "synthetic")
    with pytest.warns(
        UserWarning, match="Skipping variable 'Y' because it does not exist!"
    ):
        # skipping a variable raises a warning
        reader = DirectoryImageReader(
            test_data_path / "synthetic",
            ["X", "Y"],
            fmt="synthetic_%Y%m%dT%H%M.nc",
            skip_missing=True,
            rename={"newX_0": "X", "newY": "Y"},
            level={"newX": {"level": [0]}, "newY": {"level": 0}},
            **kwargs
        )
    validate_reader(reader, ds)
    make_clean_testdir()


def test_directory_image_reader_pattern(synthetic_test_args):
    # tests whether skipping files that don't match the pattern works
    ds, kwargs = synthetic_test_args
    make_clean_testdir()
    write_images(ds, test_data_path / "synthetic", "synthetic")
    # add another netCDF file with a different filename that would make the
    # reader fail
    ds.to_netcdf(test_data_path / "synthetic" / "full_stack.nc")
    reader = DirectoryImageReader(
        test_data_path / "synthetic",
        ["X", "Y"],
        fmt="synthetic_%Y%m%dT%H%M.nc",
        pattern="synthetic*.nc",
        **kwargs
    )
    validate_reader(reader, ds)
    make_clean_testdir()


def test_directory_image_reader_time_regex_pattern(synthetic_test_args):
    # tests whether the time regex pattern argument works
    ds, kwargs = synthetic_test_args
    make_clean_testdir()
    write_images(ds, test_data_path / "synthetic", "synthetic")
    reader = DirectoryImageReader(
        test_data_path / "synthetic",
        ["X", "Y"],
        fmt="%Y%m%dT%H%M",
        time_regex_pattern="synthetic_([0-9T]+).nc",
        **kwargs
    )
    validate_reader(reader, ds)
    make_clean_testdir()


def test_directory_image_reader_select_level(synthetic_test_args):
    # tests whether selecting from a level dimension works
    ds, kwargs = synthetic_test_args
    ds = xr.concat((ds, ds * 2), dim="level").transpose(..., "level")
    make_clean_testdir()
    write_images(ds, test_data_path / "synthetic", "synthetic")
    reader = DirectoryImageReader(
        test_data_path / "synthetic",
        ["X", "Y"],
        fmt="synthetic_%Y%m%dT%H%M.nc",
        level={"X": {"level": 0}, "Y": {"level": 1}},
        **kwargs
    )
    X = ds.X.isel(level=0)
    Y = ds.Y.isel(level=1)
    newds = xr.merge((X, Y))
    newds.attrs = ds.attrs
    validate_reader(reader, newds)
    make_clean_testdir()


def test_directory_image_reader_select_level_single_var(synthetic_test_args):
    # tests whether selecting from a level dimension for a single variable with
    # shorthand notation works
    ds, kwargs = synthetic_test_args
    ds = xr.concat((ds, ds * 2), dim="level").transpose(..., "level")
    make_clean_testdir()
    write_images(ds, test_data_path / "synthetic", "synthetic")
    reader = DirectoryImageReader(
        test_data_path / "synthetic",
        ["X"],
        fmt="synthetic_%Y%m%dT%H%M.nc",
        level={"level": 0},
        **kwargs
    )
    validate_reader(reader, ds[["X"]].isel(level=0))
    make_clean_testdir()


def test_directory_image_reader_discard_attrs(synthetic_test_args):
    # tests whether discarding metadata works
    ds, kwargs = synthetic_test_args
    make_clean_testdir()
    write_images(ds, test_data_path / "synthetic", "synthetic")
    reader = DirectoryImageReader(
        test_data_path / "synthetic",
        ["X", "Y"],
        fmt="synthetic_%Y%m%dT%H%M.nc",
        discard_attrs=True,
        **kwargs
    )
    assert reader.global_attrs == {}
    assert reader.array_attrs == {"X": {}, "Y": {}}
    ds.attrs = {}
    ds["X"].attrs = {}
    ds["Y"].attrs = {}
    validate_reader(reader, ds)
    make_clean_testdir()


def test_directory_image_reader_fill_value(synthetic_test_args):
    # tests whether replacing fill values with NaN works
    ds, kwargs = synthetic_test_args
    ds["X"].values[ds.X.values < -0.5] = -9999
    ds["Y"].values[ds.Y.values < -0.5] = -9999

    make_clean_testdir()
    write_images(ds, test_data_path / "synthetic", "synthetic")
    reader = DirectoryImageReader(
        test_data_path / "synthetic",
        ["X", "Y"],
        fmt="synthetic_%Y%m%dT%H%M.nc",
        fill_value=-9999,
        **kwargs
    )
    ds["X"].values[ds.X.values == -9999] = np.nan
    ds["Y"].values[ds.Y.values == -9999] = np.nan
    validate_reader(reader, ds)
    make_clean_testdir()


def test_directory_image_reader_nondefault_names(synthetic_test_args):
    ds, kwargs = synthetic_test_args
    ds = ds.rename({"lat": "newlat", "lon": "newlon", "time": "newtime"})
    make_clean_testdir()
    write_images(ds, test_data_path / "synthetic", "synthetic", dim="newtime")
    reader = DirectoryImageReader(
        test_data_path / "synthetic",
        ["X", "Y"],
        fmt="synthetic_%Y%m%dT%H%M.nc",
        latname="newlat",
        lonname="newlon",
        timename="newtime",
        **kwargs
    )
    validate_reader(reader, ds, timename="newtime")
    make_clean_testdir()


def test_directory_image_reader_latlon_from_array(synthetic_test_args):
    ds, kwargs = synthetic_test_args
    make_clean_testdir()
    write_images(ds, test_data_path / "synthetic", "synthetic")
    dims = ds.X.dims
    if len(dims) == 3:
        # regular and curvilinear
        _, ydim, xdim = dims
    else:
        ydim = dims[1]
        xdim = dims[1]
    reader = DirectoryImageReader(
        test_data_path / "synthetic",
        ["X", "Y"],
        fmt="synthetic_%Y%m%dT%H%M.nc",
        lat=ds.lat.values,
        lon=ds.lon.values,
        ydim=ydim,
        xdim=xdim,
        **kwargs
    )
    validate_reader(reader, ds)
    make_clean_testdir()


def test_directory_image_reader_latlon_from_2d(regular_test_dataset):
    ds = regular_test_dataset
    orig_ds = ds.copy()
    LON, LAT = np.meshgrid(ds.lon, ds.lat)
    ds["LAT"] = (["lat", "lon"], LAT)
    ds["LON"] = (["lat", "lon"], LON)
    ds = ds.drop_vars(["lat", "lon"])
    make_clean_testdir()
    write_images(ds, test_data_path / "synthetic", "synthetic")
    reader = DirectoryImageReader(
        test_data_path / "synthetic",
        ["X", "Y"],
        fmt="synthetic_%Y%m%dT%H%M.nc",
        latname="LAT",
        xdim="lat",
        lonname="LON",
        ydim="lon",
    )
    validate_reader(reader, orig_ds)
    make_clean_testdir()


def test_directory_image_reader_landmask(synthetic_test_args):
    # tests whether replacing fill values with NaN works
    ds, kwargs = synthetic_test_args
    ds["landmask"] = ds.X.isel(time=0) > -0.5
    ds["X"] = ds.X.where(ds.landmask)
    ds["Y"] = ds.X.where(ds.landmask)
    make_clean_testdir()
    write_images(ds, test_data_path / "synthetic", "synthetic")

    # test with array as landmask
    reader = DirectoryImageReader(
        test_data_path / "synthetic",
        ["X", "Y"],
        fmt="synthetic_%Y%m%dT%H%M.nc",
        landmask=ds["landmask"],
        **kwargs
    )
    validate_reader(reader, ds[["X", "Y"]])

    # test with string as landmask
    reader = DirectoryImageReader(
        test_data_path / "synthetic",
        ["X", "Y"],
        fmt="synthetic_%Y%m%dT%H%M.nc",
        landmask="landmask",
        **kwargs
    )
    validate_reader(reader, ds[["X", "Y"]])

    make_clean_testdir()


def test_directory_image_reader_no_grid(synthetic_test_args):
    ds, kwargs = synthetic_test_args
    make_clean_testdir()
    write_images(ds, test_data_path / "synthetic", "synthetic")
    reader = DirectoryImageReader(
        test_data_path / "synthetic",
        ["X", "Y"],
        fmt="synthetic_%Y%m%dT%H%M.nc",
        construct_grid=False,
        **kwargs
    )
    validate_reader(reader, ds, grid=False)
    make_clean_testdir()


def test_directory_image_reader_averaging(synthetic_test_args):
    ds, kwargs = synthetic_test_args
    newtime = pd.date_range(
        ds.indexes["time"][0], periods=len(ds.indexes["time"]), freq="12H"
    )
    ds = ds.assign_coords({"time": newtime})
    make_clean_testdir()
    write_images(ds, test_data_path / "synthetic", "synthetic")
    reader = DirectoryImageReader(
        test_data_path / "synthetic",
        ["X", "Y"],
        fmt="synthetic_%Y%m%dT%H%M.nc",
        average="daily",
        **kwargs
    )
    newds = ds.resample(time="D").mean(keep_attrs=True)
    assert len(newds.time) == (len(ds.time) // 2)
    validate_reader(reader, newds)
    make_clean_testdir()


def test_directory_image_reader_multiple_timesteps(synthetic_test_args):
    # in this test we write out two timesteps per file and the attempt to read
    # them again, using the "timestamps" keyword argument
    ds, kwargs = synthetic_test_args
    newtime = pd.date_range(
        ds.indexes["time"][0], periods=len(ds.indexes["time"]), freq="12H"
    )
    newtime += pd.Timedelta("6H")
    ds = ds.assign_coords({"time": newtime})
    make_clean_testdir()
    write_images(
        ds, test_data_path / "synthetic", "synthetic", stepsize=2, fmt="%Y%m%d"
    )
    reader = DirectoryImageReader(
        test_data_path / "synthetic",
        ["X", "Y"],
        fmt="synthetic_%Y%m%d.nc",
        timestamps=[pd.Timedelta("6H"), pd.Timedelta("18H")],
        **kwargs
    )
    ntime = len(reader.timestamps)
    assert len(reader._file_tstamp_map) == ntime // 2
    for tstamps in reader._file_tstamp_map.values():
        assert len(tstamps) == 2
    validate_reader(reader, ds)
    make_clean_testdir()


def test_directory_image_reader_multiple_timesteps_transposed(synthetic_test_args):
    # same as multiple_timests test, but now with time as last dimension in the images
    ds, kwargs = synthetic_test_args
    newtime = pd.date_range(
        ds.indexes["time"][0], periods=len(ds.indexes["time"]), freq="12H"
    )
    newtime += pd.Timedelta("6H")
    ds = ds.assign_coords({"time": newtime})
    ds = ds.transpose(..., "time")
    make_clean_testdir()
    write_images(
        ds, test_data_path / "synthetic", "synthetic", stepsize=2, fmt="%Y%m%d"
    )
    reader = DirectoryImageReader(
        test_data_path / "synthetic",
        ["X", "Y"],
        fmt="synthetic_%Y%m%d.nc",
        transpose=("time", ...),  # we have to revert the transpose operation
        timestamps=[pd.Timedelta("6H"), pd.Timedelta("18H")],
        **kwargs
    )
    ds = ds.transpose("time", ...)
    validate_reader(reader, ds)
    make_clean_testdir()


def _write_multistep_files(ds, directory, drop_time=False):
    directory.mkdir(exist_ok=True)
    time = ds.indexes["time"]
    if drop_time:
        ds = ds.drop_vars("time")
    ds1 = ds.isel(time=slice(0, 4))
    ds1.to_netcdf(directory / time[0].strftime("synthetic_%Y%m%d.nc"))
    ds2 = ds.isel(time=slice(4, 8))
    ds2.to_netcdf(directory / time[4].strftime("synthetic_%Y%m%d.nc"))


def test_directory_image_reader_multiple_timesteps_subset(synthetic_test_args):
    # here we test if it works to only read a subset of timesteps from files
    # with multiple timesteps
    ds, kwargs = synthetic_test_args
    newtime = pd.date_range(
        ds.indexes["time"][0], periods=len(ds.indexes["time"]), freq="6H"
    )
    ds = ds.assign_coords(time=newtime)
    make_clean_testdir()
    _write_multistep_files(ds, test_data_path / "synthetic")
    reader = DirectoryImageReader(
        test_data_path / "synthetic",
        ["X", "Y"],
        fmt="synthetic_%Y%m%d.nc",
        timestamps=[
            pd.Timedelta("0H"),
            pd.Timedelta("6H"),
            pd.Timedelta("12H"),
            pd.Timedelta("18H"),
        ],
        **kwargs
    )
    ds = ds.isel(time=slice(0, 8))
    validate_reader(reader, ds)
    make_clean_testdir()


def test_directory_image_reader_multiple_timesteps_subset_notime(synthetic_test_args):
    # here we test if it works to only read a subset of timesteps from files
    # with multiple timesteps
    ds, kwargs = synthetic_test_args
    newtime = pd.date_range(
        ds.indexes["time"][0], periods=len(ds.indexes["time"]), freq="6H"
    )
    ds = ds.assign_coords(time=newtime)
    make_clean_testdir()
    _write_multistep_files(ds, test_data_path / "synthetic", drop_time=True)
    reader = DirectoryImageReader(
        test_data_path / "synthetic",
        ["X", "Y"],
        fmt="synthetic_%Y%m%d.nc",
        timestamps=[
            pd.Timedelta("0H"),
            pd.Timedelta("6H"),
            pd.Timedelta("12H"),
            pd.Timedelta("18H"),
        ],
        **kwargs
    )
    ds = ds.isel(time=slice(0, 8))
    validate_reader(reader, ds)
    make_clean_testdir()
