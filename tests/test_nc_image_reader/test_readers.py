import numpy as np
import pandas as pd
import pytest
import time
import xarray as xr

from qa4sm_preprocessing.nc_image_reader.readers import (
    DirectoryImageReader,
    XarrayImageReader,
    XarrayTSReader,
)

# this is defined in conftest.py
from pytest import test_data_path


def validate_reader(reader):

    expected_timestamps = pd.date_range(
        "2017-03-30 00:00", periods=6, freq="D"
    ).to_pydatetime()
    assert len(reader.timestamps) == 6
    assert np.all(list(reader.timestamps) == expected_timestamps)

    img = reader.read(expected_timestamps[0])
    true = xr.open_dataset(
        test_data_path
        / "lis_noahmp"
        / "201703"
        / "LIS_HIST_201703300000.d01.nc"
    )["SoilMoist_inst"].isel(SoilMoist_profiles=0)
    np.testing.assert_allclose(img.data["SoilMoist_inst"], true.values.ravel())
    true.attrs == img.metadata["SoilMoist_inst"]

    img = reader.read(expected_timestamps[-1])
    true = xr.open_dataset(
        test_data_path
        / "lis_noahmp"
        / "201704"
        / "LIS_HIST_201704040000.d01.nc"
    )["SoilMoist_inst"].isel(SoilMoist_profiles=0)
    np.testing.assert_allclose(img.data["SoilMoist_inst"], true.values.ravel())
    true.attrs == img.metadata["SoilMoist_inst"]


###############################################################################
# DirectoryImageReader
###############################################################################

# Optional features to test for DirectoryImageReader:
# - [X] level
#   - [X] with level: default_directory_reader
#   - [-] without level: covered in XarrayImageReader tests
# - [X] fmt: default_directory_reader
# - [X] pattern: default_directory_reader
# - [X] time_regex_pattern: test_time_regex
# - [-] timename, latname, lonname: covered in XarrayImageReader tests
# - [X] latdim, londim: default_directory_reader
# - [-] locdim: covered in XarrayImageReader tests
# - [-] landmask: covered in XarrayImageReader tests
# - [-] bbox: covered in XarrayImageReader tests
# - [-] cellsize None: covered in XarrayImageReader tests


def test_directory_reader_setup():

    # test "normal procedure", i.e. with given fmt
    pattern = "LIS_HIST*.nc"
    fmt = "LIS_HIST_%Y%m%d%H%M.d01.nc"

    # the LIS_HIST files have dimensions north_south and east_west instead of
    # lat/lon
    start = time.time()
    reader = DirectoryImageReader(
        test_data_path / "lis_noahmp",
        "SoilMoist_inst",
        fmt=fmt,
        pattern=pattern,
        latdim="north_south",
        londim="east_west",
        level={"SoilMoist_profiles": 0},
        lat=(29.875, 0.25),
        lon=(-11.375, 0.25),
    )
    runtime = time.time() - start
    print(f"Setup time with fmt string: {runtime:.2e}")
    validate_reader(reader)

    # test without fmt, requires opening all files
    start = time.time()
    reader = DirectoryImageReader(
        test_data_path / "lis_noahmp",
        "SoilMoist_inst",
        pattern=pattern,
        latdim="north_south",
        londim="east_west",
        level={"SoilMoist_profiles": 0},
        lat=(29.875, 0.25),
        lon=(-11.375, 0.25),
    )
    runtime2 = time.time() - start
    print(f"Setup time without fmt string: {runtime2:.2e}")
    validate_reader(reader)

    assert runtime < runtime2


def test_time_regex():
    # test with using a regex for the time string
    pattern = "LIS_HIST*.nc"
    fmt = "%Y%m%d%H%M"
    time_regex_pattern = r"LIS_HIST_(\d+)\..*\.nc"
    reader = DirectoryImageReader(
        test_data_path / "lis_noahmp",
        "SoilMoist_inst",
        fmt=fmt,
        pattern=pattern,
        time_regex_pattern=time_regex_pattern,
        latdim="north_south",
        londim="east_west",
        level={"SoilMoist_profiles": 0},
        lat=(29.875, 0.25),
        lon=(-11.375, 0.25),
    )
    validate_reader(reader)


def test_read_block(default_directory_reader):
    block = default_directory_reader.read_block()["SoilMoist_inst"]
    assert block.shape == (6, 100, 50)

    reader = XarrayImageReader(
        block.to_dataset(name="SoilMoist_inst"), "SoilMoist_inst"
    )
    validate_reader(reader)

    start_date = next(iter(default_directory_reader.timestamps))
    block = default_directory_reader.read_block(
        start=start_date, end=start_date
    )["SoilMoist_inst"]
    assert block.shape == (1, 100, 50)


###############################################################################
# XarrayImageReader
###############################################################################

# Optional features to test for DirectoryImageReader:
# - [X] level
#   - [-] with level: covered in DirectoryImageReader tests
#   - [X] without level: test_xarray_reader_basic
# - [X] timename, latname, lonname: test_nonstandard_names
# - [X] latdim, londim: covered in DirectoryImageReader tests
# - [X] locdim: test_locdim
# - [X] bbox, cellsize, landmask: test_landmask, test_bbox_cellsize


def test_xarray_reader_basic(default_xarray_reader):
    validate_reader(default_xarray_reader)


def test_nonstandard_names(test_dataset):
    ds = test_dataset.rename({"time": "tim", "lat": "la", "lon": "lo"})
    reader = XarrayImageReader(
        ds, "X", timename="tim", latname="la", lonname="lo"
    )
    block = reader.read_block()["X"]
    assert block.shape == (100, 10, 20)


def test_locdim(test_loc_dataset):
    reader = XarrayImageReader(
        test_loc_dataset, "X", locdim="location", latname="lat", lonname="lon"
    )
    block = reader.read_block()["X"]
    assert block.shape == (100, 200)


def test_bbox_landmask_cellsize(cmip_ds):
    """
    Tests the bounding box feature
    """
    num_gpis = cmip_ds["mrsos"].isel(time=0).size

    # normal reader without bbox or landmask
    reader = XarrayImageReader(cmip_ds, "mrsos", cellsize=5.0)
    assert len(reader.grid.activegpis) == num_gpis
    assert len(np.unique(reader.grid.activearrcell)) == 100
    block = reader.read_block()["mrsos"]
    np.testing.assert_allclose(block.values, cmip_ds.mrsos.values)

    # now with bbox
    min_lon = 90
    min_lat = 20
    max_lon = 100
    max_lat = 30
    bbox = [min_lon, min_lat, max_lon, max_lat]
    reader = XarrayImageReader(cmip_ds, "mrsos", bbox=bbox, cellsize=5.0)
    num_gpis_box = len(reader.grid.activegpis)
    assert num_gpis_box < num_gpis
    assert len(np.unique(reader.grid.activearrcell)) == 4
    assert not np.any(reader.grid.arrlon < min_lon)
    assert not np.any(reader.grid.arrlat < min_lat)
    assert not np.any(reader.grid.arrlon > max_lon)
    assert not np.any(reader.grid.arrlat > max_lat)
    block = reader.read_block()["mrsos"]
    # the sel-slice notation only works with regular grids
    europe_mrsos = cmip_ds.mrsos.sel(
        lat=slice(min_lat, max_lat), lon=slice(min_lon, max_lon)
    )
    np.testing.assert_allclose(block.values, europe_mrsos.values)

    # now additionally using a landmask
    landmask = ~np.isnan(cmip_ds.mrsos.isel(time=0))
    reader = XarrayImageReader(
        cmip_ds, "mrsos", bbox=bbox, landmask=landmask, cellsize=5.0
    )
    assert len(reader.grid.activegpis) < num_gpis
    assert len(np.unique(reader.grid.activearrcell)) == 4
    assert not np.any(reader.grid.arrlon < min_lon)
    assert not np.any(reader.grid.arrlat < min_lat)
    assert not np.any(reader.grid.arrlon > max_lon)
    assert not np.any(reader.grid.arrlat > max_lat)
    block = reader.read_block()["mrsos"]
    # the sel-slice notation only works with regular grids
    europe_mrsos = cmip_ds.mrsos.sel(
        lat=slice(min_lat, max_lat), lon=slice(min_lon, max_lon)
    )
    np.testing.assert_allclose(block.values, europe_mrsos.values)

    # with landmask as variable name
    ds = cmip_ds
    ds["landmask"] = landmask
    reader = XarrayImageReader(
        ds, "mrsos", bbox=bbox, landmask=landmask, cellsize=5.0
    )
    assert len(reader.grid.activegpis) < num_gpis
    assert len(np.unique(reader.grid.activearrcell)) == 4
    assert not np.any(reader.grid.arrlon < min_lon)
    assert not np.any(reader.grid.arrlat < min_lat)
    assert not np.any(reader.grid.arrlon > max_lon)
    assert not np.any(reader.grid.arrlat > max_lat)
    new_block = reader.read_block()["mrsos"]
    np.testing.assert_allclose(block.values, new_block.values)


###############################################################################
# XarrayTSReader
###############################################################################


def test_xarray_ts_reader(test_dataset):
    reader = XarrayTSReader(test_dataset, "X")
    _, lons, lats = reader.grid.get_grid_points()
    for lon, lat in zip(lons, lats):
        ts = reader.read(lon, lat)["X"]
        ref = test_dataset.X.sel(lat=lat, lon=lon)
        assert np.all(ts == ref)


def test_xarray_ts_reader_locdim(test_loc_dataset):
    reader = XarrayTSReader(test_loc_dataset, "X", locdim="location")
    gpis, _, _ = reader.grid.get_grid_points()
    for gpi in gpis:
        ts = reader.read(gpi)["X"]
        ref = test_loc_dataset.X.isel(location=gpi)
        assert np.all(ts == ref)
