import numpy as np
import pandas as pd
import xarray as xr

from qa4sm_preprocessing.reading import (
    StackTs,
    StackImageReader,
    GriddedNcOrthoMultiTs,
)

import pytest


def test_StackTs(regular_test_dataset):
    reader = StackTs(regular_test_dataset, "X")
    gpis, lons, lats, _ = reader.grid.get_grid_points()
    for gpi, lon, lat in zip(gpis, lons, lats):
        ref = regular_test_dataset.X.sel(lat=lat, lon=lon)
        ts = reader.read(lon, lat)["X"]
        assert np.all(ts == ref)
        ts = reader.read(gpi)["X"]
        assert np.all(ts == ref)


def test_StackTs_repurpose(regular_test_dataset, test_output_path):
    reader = StackTs(regular_test_dataset, "X")
    period = (reader.data.time.values[0], reader.data.time.values[-1])
    outpath = test_output_path / "gridded_ts"
    tsreader = reader.repurpose(outpath, overwrite=True, start=period[0], end=period[1])
    gpis, lons, lats, _ = tsreader.grid.get_grid_points()
    for gpi, lon, lat in zip(gpis, lons, lats):
        ref = regular_test_dataset.X.sel(lat=lat, lon=lon)
        ts = tsreader.read(lon, lat)["X"]
        assert np.all(ts == ref)
        ts = tsreader.read(gpi)["X"]
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


def test_GriddedNcOrthoMultiTs(synthetic_test_args, test_output_path):

    ds, kwargs = synthetic_test_args
    stack = StackImageReader(ds, ["X", "Y"], **kwargs)

    tspath = test_output_path / "ts_test_path"
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
    newtime = (newtime.T + np.arange(len(ds.time)) * 86400).T
    newtime.attrs["units"] = "seconds since 2000-01-01"
    newtime.attrs["long_name"] = "exact observation time"
    ds["exact_time"] = newtime
    # compared to test_StackTs_timeoffset we specify the varnames explicitly
    # here, so we cover also the case where varnames are given but the
    # timevarname is not included
    reader = StackTs(ds, ["X", "Y"], timevarname="exact_time", **kwargs)
    df = reader.read(0)
    assert list(df.columns) == ["X", "Y"]
    assert np.all(df.index == (ds.indexes["time"] + pd.Timedelta("1s")))


def test_GriddedNcOrthoMultiTs_timeoffset(synthetic_test_args, test_output_path):

    ds, kwargs = synthetic_test_args
    ds["time_offset"] = xr.ones_like(ds.X)
    stack = StackImageReader(ds, **kwargs)

    tspath = test_output_path / "ts_test_path"
    # time_offset should be detected by default
    tsreader = stack.repurpose(tspath, overwrite=True)

    df = tsreader.read(0)
    assert list(df.columns) == ["X", "Y"]
    assert np.all(df.index == (ds.indexes["time"] + pd.Timedelta("1s")))


def test_GriddedNcOrthoMultiTs_timevar(synthetic_test_args, test_output_path):

    ds, kwargs = synthetic_test_args
    newtime = xr.ones_like(ds.X)
    newtime = (newtime.T + np.arange(len(ds.time)) * 86400).T
    newtime.attrs["units"] = "seconds since 2000-01-01"
    newtime.attrs["long_name"] = "exact observation time"
    ds["exact_time"] = newtime
    stack = StackImageReader(ds, **kwargs)

    tspath = test_output_path / "ts_test_path"
    tsreader = stack.repurpose(tspath, timevarname="exact_time", overwrite=True)

    df = tsreader.read(0)
    assert list(df.columns) == ["X", "Y"]
    assert np.all(df.index == (ds.indexes["time"] + pd.Timedelta("1s")))


def test_GriddedNcOrthoMultiTs_period(synthetic_test_args, test_output_path):

    ds, kwargs = synthetic_test_args
    stack = StackImageReader(ds, **kwargs)

    tspath = test_output_path / "ts_test_path"
    start = "2000-01-01"
    end = "2000-01-04"
    tsreader = stack.repurpose(tspath, start=start, end=end, overwrite=True)

    df = tsreader.read(0)
    assert list(df.columns) == ["X", "Y"]
    idx = {v: 0 for v in set(ds.dims) - set(["time"])}
    assert df.equals(ds.isel(**idx).to_dataframe().loc[start:end, ["X", "Y"]])


def test_GriddedNcOrthoMultiTs(synthetic_test_args, test_output_path):

    ds, kwargs = synthetic_test_args
    stack = StackImageReader(ds, ["X", "Y"], **kwargs)

    tspath = test_output_path / "ts_test_path"
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




# def test_ContiguousRaggedTs():
#     # tests the full pipeline going from single timeseries files to
#     # a ragged timeseries to a pynetcf GriddedNcContiguousRaggedTs


#     # timeseries 1: daily timestamps
#     index = pd.date_range("2020-01-01 12:00", "2020-12-31 12:00", freq="D")
#     data = np.random.randn(len(index))
#     ts1 = pd.Series(data, index=index)
#     lat1 = 12
#     lon1 = 0.1

#     # timeseries 2: hourly timestamps, but shorter period
#     index = pd.date_range("2020-06-01", "2020-08-01", freq="H")
#     data = np.random.randn(len(index)) + 2
#     ts2 = pd.Series(data, index=index)
#     lat2 = 34
#     lon2 = 2.3

#     # timeseries 3: irregular timestamps
#     index = pd.DatetimeIndex(
#         np.datetime64("2020-01-01")
#         + np.random.rand(1000) * np.timedelta64(365 * 24 * 60, "m")
#     )
#     data = np.random.randn(len(index)) - 2
#     ts3 = pd.Series(data, index=index)
#     lat3 = 56
#     lon3 = 4.5

#     lats = [lat1, lat2, lat3]
#     lons = [lon1, lon2, lon3]
#     timeseries = [ts1, ts2, ts3]
#     cont_ragged_ts = make_contiguous_ragged_array(
#         timeseries, lons, lats, name="soil_moisture"
#     )

#     assert isinstance(cont_ragged_ts, xr.Dataset)
#     assert "loc" in cont_ragged_ts.dims
#     assert "loctime" in cont_ragged_ts.dims
#     for v in ["lat", "lon", "count", "cumulative_count"]:
#         assert v in cont_ragged_ts.data_vars
#         assert cont_ragged_ts[v].dims == ("loc",)
#     for v in ["time", "soil_moisture"]:
#         assert v in cont_ragged_ts.data_vars
#         assert cont_ragged_ts[v].dims == ("loctime",)
#     assert cont_ragged_ts.dims["loc"] == 3
#     assert cont_ragged_ts.dims["loctime"] == len(ts1) + len(ts2) + len(ts3)

#     # testing with a small cellsize here to have different output files
#     ragged_tsreader = ContiguousRaggedTs(cont_ragged_ts, cellsize=1)
#     tspath = test_output_path / "ts_test_path"
#     gridded_tsreader = ragged_tsreader.repurpose(tspath, overwrite=True)

#     assert isinstance(ragged_tsreader, ContiguousRaggedTs)
#     assert isinstance(gridded_tsreader, GriddedNcContiguousRaggedTs)

#     for i in range(3):
#         # test if ragged reader returns the same as we put in
#         assert ragged_tsreader.read(i)["soil_moisture"].equals(timeseries[i])
#         # test if ragged reader returns the same as gridded reader
#         assert ragged_tsreader.read(i).equals(gridded_tsreader.read(i))

#     cells = ragged_tsreader.grid.get_cells()
#     cellfnames = [f"{cell}.nc" for cell in cells]
#     assert sorted(os.listdir(tspath)) == sorted(cellfnames + ["grid.nc"])
#     for i, fname in enumerate(cellfnames):
#         ds = xr.open_dataset(tspath / fname)
#         assert "soil_moisture" in ds.data_vars
#         assert np.abs(ds.lat.values[0] - lats[i]) < 1e-6
#         assert np.abs(ds.lon.values[0] - lons[i]) < 1e-6

#     # this somehow does not work in the CI
#     # # test if overwrite works by capturing the log message
#     # stream = StringIO()
#     # handler = logging.StreamHandler(stream)
#     # logger = logging.getLogger("root")
#     # logger.setLevel(logging.INFO)
#     # logger.addHandler(handler)
#     # ragged_tsreader.repurpose(tspath, overwrite=False)
#     # handler.flush()
#     # output = stream.getvalue()
#     # assert output == f"Output path already exists: {str(tspath)}\n"

#     # test if repurposing only a shorter period also works
#     start = "2020-01-01"
#     end = "2020-06-01"
#     gridded_tsreader = ragged_tsreader.repurpose(
#         tspath, overwrite=True, start=start, end=end
#     )

#     for i in range(3):
#         ts = gridded_tsreader.read(i)["soil_moisture"]
#         assert ts.equals(ragged_tsreader.read(i).loc[start:end, "soil_moisture"])
