import numpy as np
import pandas as pd
import time
import xarray as xr

from qa4sm_preprocessing.reading import (
    DirectoryImageReader,
    StackImageReader,
    GriddedNcOrthoMultiTs
)
from qa4sm_preprocessing.reading.utils import mkdate

# this is defined in conftest.py
from pytest import test_data_path


def validate_reader(reader, metadata=True):

    expected_timestamps = pd.date_range(
        "2017-03-30 00:00", periods=6, freq="D"
    ).to_pydatetime()
    assert len(reader.timestamps) == 6
    assert np.all(list(reader.timestamps) == expected_timestamps)

    img = reader.read(expected_timestamps[0])
    true = xr.open_dataset(
        test_data_path / "lis_noahmp" / "201703" / "LIS_HIST_201703300000.d01.nc"
    )["SoilMoist_inst"].isel(SoilMoist_profiles=0)
    np.testing.assert_allclose(img.data["SoilMoist_inst"], true.values.ravel())
    if metadata:
        true.attrs == img.metadata["SoilMoist_inst"]
    else:
        assert img.metadata["SoilMoist_inst"] == {}

    img = reader.read(expected_timestamps[-1])
    true = xr.open_dataset(
        test_data_path / "lis_noahmp" / "201704" / "LIS_HIST_201704040000.d01.nc"
    )["SoilMoist_inst"].isel(SoilMoist_profiles=0)
    np.testing.assert_allclose(img.data["SoilMoist_inst"], true.values.ravel())
    if metadata:
        true.attrs == img.metadata["SoilMoist_inst"]
    else:
        assert img.metadata["SoilMoist_inst"] == {}

    # metadata for read_block
    block = reader.read_block(expected_timestamps[0], expected_timestamps[0])
    if metadata:
        true = xr.open_dataset(
            test_data_path / "lis_noahmp" / "201703" / "LIS_HIST_201703300000.d01.nc"
        )
        assert true.attrs.keys() == block.attrs.keys()
        assert all([np.all(block.attrs[key] == true.attrs[key]) for key in true.attrs])
        assert (
            true["SoilMoist_inst"].attrs.keys() == block["SoilMoist_inst"].attrs.keys()
        )
        assert all(
            [
                np.all(
                    block["SoilMoist_inst"].attrs[key]
                    == true["SoilMoist_inst"].attrs[key]
                )
                for key in true["SoilMoist_inst"].attrs
            ]
        )
    else:
        assert block.attrs == {}
        assert block["SoilMoist_inst"].attrs == {}


def test_directory_reader_setup():

    # test "normal procedure", i.e. with given fmt
    pattern = "**/LIS_HIST*.nc"
    fmt = "LIS_HIST_%Y%m%d%H%M.d01.nc"

    # the LIS_HIST files have dimensions north_south and east_west instead of
    # lat/lon
    start = time.time()
    reader = DirectoryImageReader(
        test_data_path / "lis_noahmp",
        "SoilMoist_inst",
        fmt=fmt,
        pattern=pattern,
        rename={"north_south": "lat", "east_west": "lon"},
        level={"SoilMoist_profiles": 0},
        lat=(29.875, 54.75, 0.25),
        lon=(-11.375, 1.0, 0.25),
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
        rename={"north_south": "lat", "east_west": "lon"},
        level={"SoilMoist_profiles": 0},
        lat=(29.875, 54.75, 0.25),
        lon=(-11.375, 1.0, 0.25),
    )
    runtime2 = time.time() - start
    print(f"Setup time without fmt string: {runtime2:.2e}")
    validate_reader(reader)

    assert runtime < runtime2


def test_read_block(lis_noahmp_directory_image_reader):
    block = lis_noahmp_directory_image_reader.read_block()
    assert block["SoilMoist_inst"].shape == (6, 100, 50)

    reader = StackImageReader(block, "SoilMoist_inst")
    validate_reader(reader)

    start_date = next(iter(lis_noahmp_directory_image_reader.timestamps))
    block = lis_noahmp_directory_image_reader.read_block(
        start=start_date, end=start_date
    )["SoilMoist_inst"]
    assert block.shape == (1, 100, 50)


def test_xarray_reader_basic(default_xarray_reader):
    validate_reader(default_xarray_reader)


def test_stack_reader_basic(cmip_ds):
    num_gpis = cmip_ds["mrsos"].isel(time=0).size

    reader = StackImageReader(cmip_ds, "mrsos", cellsize=5.0)
    assert len(reader.grid.activegpis) == num_gpis
    assert len(np.unique(reader.grid.activearrcell)) == 100
    block = reader.read_block()["mrsos"]
    np.testing.assert_allclose(block.values, cmip_ds.mrsos.values)


def test_bbox_landmask_cellsize(cmip_ds):
    """
    Tests the bounding box feature
    """
    num_gpis = cmip_ds["mrsos"].isel(time=0).size

    # now with bbox
    min_lon = 90
    min_lat = 20
    max_lon = 100
    max_lat = 30
    bbox = [min_lon, min_lat, max_lon, max_lat]
    reader = StackImageReader(cmip_ds, "mrsos", bbox=bbox, cellsize=5.0)
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
    reader = StackImageReader(
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
    reader = StackImageReader(
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


def test_SMOS(test_output_path):

    reader = DirectoryImageReader(
        test_data_path / "SMOS_L3",
        varnames=["Soil_Moisture", "Mean_Acq_Time"],
        rename={"Mean_Acq_Time_Seconds": "Mean_Acq_Time"},
        timeoffsetvarname="Mean_Acq_Time",
        timeoffsetunit="seconds",
        time_regex_pattern="SM_OPER_MIR_CLF31A_([0-9T]+)_.*.DBL.nc",
        fmt="%Y%m%dT%H%M%S",
    )

    outpath = test_output_path / "SMOS_ts"
    _ = reader.repurpose(outpath, overwrite=True, timevarname="Mean_Acq_Time")
    ts_reader = GriddedNcOrthoMultiTs(str(outpath), timevarname="Mean_Acq_Time")

    def validate(ts_reader):
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

    validate(ts_reader)

    # test overwriting again with an existing directory
    _ = reader.repurpose(outpath, overwrite=True, timevarname="Mean_Acq_Time")
    ts_reader = GriddedNcOrthoMultiTs(str(outpath), timevarname="Mean_Acq_Time")
    validate(ts_reader)

    # test reading without overwriting
    _ = reader.repurpose(outpath, overwrite=False, timevarname="Mean_Acq_Time")
    ts_reader = GriddedNcOrthoMultiTs(str(outpath), timevarname="Mean_Acq_Time")
    validate(ts_reader)
