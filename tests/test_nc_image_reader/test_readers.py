import numpy as np
from pathlib import Path
import pandas as pd
import pytest
import time
import xarray as xr

from repurpose.img2ts import Img2Ts

from qa4sm_preprocessing.nc_image_reader.readers import (
    DirectoryImageReader,
    XarrayImageStackReader,
    XarrayTSReader,
    GriddedNcOrthoMultiTs,
)
from qa4sm_preprocessing.nc_image_reader.utils import mkdate

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
#   - [-] without level: covered in XarrayImageStackReader tests
# - [X] fmt: default_directory_reader
# - [X] pattern: default_directory_reader
# - [X] time_regex_pattern: test_time_regex
# - [-] timename, latname, lonname: covered in XarrayImageStackReader tests
# - [X] latdim, londim: default_directory_reader
# - [-] locdim: covered in XarrayImageStackReader tests
# - [-] landmask: covered in XarrayImageStackReader tests
# - [-] bbox: covered in XarrayImageStackReader tests
# - [-] cellsize None: covered in XarrayImageStackReader tests


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

    reader = XarrayImageStackReader(
        block.to_dataset(name="SoilMoist_inst"), "SoilMoist_inst"
    )
    validate_reader(reader)

    start_date = next(iter(default_directory_reader.timestamps))
    block = default_directory_reader.read_block(
        start=start_date, end=start_date
    )["SoilMoist_inst"]
    assert block.shape == (1, 100, 50)


###############################################################################
# XarrayImageStackReader
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


def test_nonstandard_names(latlon_test_dataset):
    ds = latlon_test_dataset.rename({"time": "tim", "lat": "la", "lon": "lo"})
    reader = XarrayImageStackReader(
        ds, "X", timename="tim", latname="la", lonname="lo"
    )
    block = reader.read_block()["X"]
    assert block.shape == (100, 10, 20)


def test_unstructured(unstructured_test_dataset):
    reader = XarrayImageStackReader(
        unstructured_test_dataset,
        "X",
        locdim="location",
        latname="lat",
        lonname="lon",
    )
    block = reader.read_block()["X"]
    assert block.shape == (100, 200)
    assert list(block.coords) == ["time", "lat", "lon"]

    img = reader.read(reader.timestamps[0])
    assert img.data["X"].shape == (200,)
    assert np.all(img.data["X"] == block.isel(time=0).values)


def test_curvilinear(curvilinear_test_dataset):
    reader = XarrayImageStackReader(
        curvilinear_test_dataset,
        "X",
        latdim="y",
        londim="x",
        latname="lat",
        lonname="lon",
        curvilinear=True,
    )
    block = reader.read_block()["X"]
    assert block.shape == (100, 10, 20)
    assert list(block.coords) == ["time", "lat", "lon"]
    assert list(block.dims) == ["time", "y", "x"]
    assert block.lat.shape == (10, 20)
    assert block.lon.shape == (10, 20)

    img = reader.read(reader.timestamps[0])
    assert img.data["X"].shape == (200,)
    assert np.all(img.data["X"] == block.isel(time=0).values.ravel())


def test_bbox_landmask_cellsize(cmip_ds):
    """
    Tests the bounding box feature
    """
    num_gpis = cmip_ds["mrsos"].isel(time=0).size

    # normal reader without bbox or landmask
    reader = XarrayImageStackReader(cmip_ds, "mrsos", cellsize=5.0)
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
    reader = XarrayImageStackReader(cmip_ds, "mrsos", bbox=bbox, cellsize=5.0)
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
    reader = XarrayImageStackReader(
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
    reader = XarrayImageStackReader(
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


def test_xarray_ts_reader(latlon_test_dataset):
    reader = XarrayTSReader(latlon_test_dataset, "X")
    _, lons, lats = reader.grid.get_grid_points()
    for lon, lat in zip(lons, lats):
        ts = reader.read(lon, lat)["X"]
        ref = latlon_test_dataset.X.sel(lat=lat, lon=lon)
        assert np.all(ts == ref)


def test_xarray_ts_reader_locdim(unstructured_test_dataset):
    reader = XarrayTSReader(unstructured_test_dataset, "X", locdim="location")
    gpis, _, _ = reader.grid.get_grid_points()
    for gpi in gpis:
        ts = reader.read(gpi)["X"]
        ref = unstructured_test_dataset.X.isel(location=gpi)
        assert np.all(ts == ref)


###############################################################################
# Images with and without time dimension
###############################################################################


def test_image_time_dimension(test_output_path):
    # create test dataset with subdaily timestamps
    nlat, nlon, ntime = 5, 5, 20
    lat = np.linspace(0, 1, nlat)
    lon = np.linspace(0, 1, nlon)
    time = pd.date_range("2000", periods=ntime, freq="6H")

    X = np.ones((ntime, nlat, nlon), dtype=np.float32)
    X = (X.T * np.arange(ntime)).T

    ds = xr.Dataset(
        {"X": (["time", "lat", "lon"], X)},
        coords={"time": time, "lat": lat, "lon": lon},
    )

    def write_images(ds, outpath, image_with_time_dim):
        outpath = Path(outpath)
        outpath.mkdir(exist_ok=True, parents=True)
        for timestamp in ds.indexes["time"]:
            if image_with_time_dim:
                img = ds.sel(time=slice(timestamp, timestamp))
            else:
                img = ds.sel(time=timestamp)
            fname = timestamp.strftime("img_%Y%m%dT%H%M%S.nc")
            img.to_netcdf(outpath / fname)

    # Write images to netCDF files once with and once without time dimension,
    # and then try to read them with averaging
    for image_with_time_dim in [False, True]:
        outpath = (
            test_output_path / f"image_with_time_dim={image_with_time_dim}"
        )
        write_images(ds, outpath, image_with_time_dim)

        reader = DirectoryImageReader(
            outpath,
            varnames=["X"],
            fmt="img_%Y%m%dT%H%M%S.nc",
            daily_average=True,
        )
        assert reader.timestamps[0] < reader.timestamps[-1]
        for i, tstamp in enumerate(reader.timestamps):
            # assumes that ntime is a multiple of 4, otherwise edge cases can
            # occur
            expected = 1.5 + 4 * i
            img = reader.read_block(tstamp, tstamp)
            assert np.all(img.X.values == expected)


###############################################################################
# Full chain with time offset
###############################################################################


def test_SMOS(test_output_path):
    reader = DirectoryImageReader(
        test_data_path / "SMOS_L3",
        varnames=["Soil_Moisture", "Mean_Acq_Time_Seconds"],
        time_regex_pattern="SM_OPER_MIR_CLF31A_([0-9T]+)_.*.DBL.nc",
        fmt="%Y%m%dT%H%M%S",
    )

    outpath = test_output_path / "SMOS_ts"
    outpath.mkdir(exist_ok=True, parents=True)

    reshuffler = Img2Ts(
        input_dataset=reader,
        outputpath=str(outpath),
        startdate=reader.timestamps[0],
        enddate=reader.timestamps[-1],
        ts_attributes=reader.global_attrs,
        zlib=True,
        imgbuffer=3,
        cellsize_lat=5,
        cellsize_lon=5,
    )
    reshuffler.calc()

    ts_reader = GriddedNcOrthoMultiTs(
        outpath, time_offset_name="Mean_Acq_Time_Seconds"
    )
    df = ts_reader.read(ts_reader.grid.activegpis[100])
    expected_timestamps = list(
        map(
            mkdate,
            [
                "2015-05-06T03:50:13",
                "2015-05-07T03:11:27",
                "2015-05-08T02:33:02",
            ],
        )
    )
    expected_values = np.array([0.162236, 0.013245, np.nan])
    assert np.all(expected_timestamps == df.index)
    np.testing.assert_almost_equal(expected_values, df.Soil_Moisture.values, 6)
    assert np.all(df.columns == ["Soil_Moisture"])
