import glob
import numpy as np
import pandas as pd
from pathlib import Path
import yaml
import xarray as xr
import zipfile

from qa4sm_preprocessing.reading import (
    GriddedNcOrthoMultiTs,
    GriddedNcContiguousRaggedTs,
)
from qa4sm_preprocessing.reading.timeseries import ZippedCsvTs
from qa4sm_preprocessing.utils import (
    make_csv_dataset,
    make_gridded_contiguous_ragged_dataset,
    preprocess_user_data,
)

from pytest import test_data_user_upload_path


def make_test_timeseries(lats=[12, 34, 56], lons=[0.1, 2.3, 4.5]):
    # timeseries 1: daily timestamps
    index = pd.date_range("2020-01-01 12:00", "2020-12-31 12:00", freq="D")
    data = np.random.randn(len(index))
    ts1 = pd.Series(data, index=index, name="soil_moisture")

    # timeseries 2: hourly timestamps, but shorter period
    index = pd.date_range("2020-06-01", "2020-08-01", freq="h")
    data = np.random.randn(len(index)) + 2
    ts2 = pd.Series(data, index=index, name="soil_moisture")

    # timeseries 3: irregular timestamps
    index = pd.DatetimeIndex(
        np.datetime64("2020-01-01")
        + np.random.rand(1000) * np.timedelta64(365 * 24 * 60, "m")
    )
    data = np.random.randn(len(index)) - 2
    ts3 = pd.Series(data, index=index, name="soil_moisture")

    ts3.index = ts3.index.astype('datetime64[ns]')
    timeseries = [ts1, ts2, ts3]
    metadata = {"soil_moisture": {"long_name": "soil moisture", "units": "m^3/m^3"}}
    return timeseries, lats, lons, metadata


def zip_directory(inpath, outpath):
    inpath = Path(inpath)
    outpath = Path(outpath)
    if outpath.exists():
        outpath.unlink()
    with zipfile.ZipFile(outpath, mode="w") as zfile:
        for f in inpath.iterdir():
            zipped_name = Path(inpath.name) / Path(f).name
            zfile.write(f, arcname=zipped_name)


def test_csv_pipeline(test_output_path):
    # tests the full pipeline going from pandas timeseries to CSVs to zipped
    # directory to gridded contiguous ragged dataset
    timeseries, lats, lons, metadata = make_test_timeseries()
    csv_dir = test_output_path / "csv"

    make_csv_dataset(
        timeseries, lats, lons, csv_dir, name="test", metadata=metadata, only_ismn=False
    )

    # check if they all look okay
    for i in range(len(timeseries)):
        fname = f"test_gpi={i}_lat={lats[i]}_lon={lons[i]}.csv"
        ts = pd.read_csv(csv_dir / fname, index_col=0, parse_dates=True)[
            "soil_moisture"
        ]
        pd.testing.assert_series_equal(ts, timeseries[i], check_freq=False)

    # check metadata
    with open(csv_dir / "metadata.yml", "r") as f:
        mdata = yaml.load(f, Loader=yaml.SafeLoader)
    for var in metadata:
        assert metadata[var] == mdata[var]

    # make zip file
    # make zip file
    zfile = test_output_path / "csv.zip"
    zip_directory(csv_dir, zfile)

    # try to read with ZippedCsvTs
    reader = ZippedCsvTs(zfile)
    for i in range(len(timeseries)):
        ts = reader.read(i)["soil_moisture"]
        pd.testing.assert_series_equal(ts, timeseries[i], check_freq=False)
    assert reader.get_metadata("soil_moisture") == metadata["soil_moisture"]

    # do the preprocessing with the full preprocessing function
    outpath = test_output_path / "gridded_ts"
    reader = preprocess_user_data(zfile, outpath)

    assert isinstance(reader, GriddedNcContiguousRaggedTs)
    desc = {
        "soil_moisture": {
            "name": "soil_moisture",
            "long_name": "soil moisture",
            "units": "m^3/m^3",
        }
    }
    assert reader.variable_description() == desc
    for i in range(len(timeseries)):
        ts = reader.read(i)["soil_moisture"]
        pd.testing.assert_series_equal(ts, timeseries[i], check_freq=False)
    assert (
        reader.fid.dataset["soil_moisture"].units == metadata["soil_moisture"]["units"]
    )
    assert (
        reader.fid.dataset["soil_moisture"].long_name
        == metadata["soil_moisture"]["long_name"]
    )


def test_contiguous_ragged_pipeline(test_output_path):
    # tests the full pipeline going from pandas timeseries to zipped pynetcf
    # directory to gridded contiguous ragged dataset
    timeseries, lats, lons, metadata = make_test_timeseries()
    pynetcf_dir = test_output_path / "pynetcf"

    make_gridded_contiguous_ragged_dataset(
        timeseries, lats, lons, pynetcf_dir, metadata=metadata, only_ismn=False
    )

    def check_reader(reader):
        for i in range(len(timeseries)):
            ts = reader.read(i)["soil_moisture"]
            pd.testing.assert_series_equal(ts, timeseries[i], check_freq=False)
        assert (
            reader.fid.dataset["soil_moisture"].units
            == metadata["soil_moisture"]["units"]
        )
        assert (
            reader.fid.dataset["soil_moisture"].long_name
            == metadata["soil_moisture"]["long_name"]
        )

    reader = GriddedNcContiguousRaggedTs(pynetcf_dir)
    check_reader(reader)

    # make zip file
    zfile = test_output_path / "pynetcf.zip"
    zip_directory(pynetcf_dir, zfile)

    # do the preprocessing with the full preprocessing function
    outpath = test_output_path / "gridded_ts"
    reader = preprocess_user_data(zfile, outpath)
    assert isinstance(reader, GriddedNcContiguousRaggedTs)
    check_reader(reader)
    desc = {
        "soil_moisture": {
            "name": "soil_moisture",
            "long_name": "soil moisture",
            "units": "m^3/m^3",
        }
    }
    assert reader.variable_description() == desc


def test_stack_pipeline(synthetic_test_args, test_output_path):
    stackpath = test_output_path / "stack.nc"

    ds, kwargs = synthetic_test_args
    ds.to_netcdf(stackpath)

    reader = preprocess_user_data(stackpath, test_output_path / "stack_ts")
    assert isinstance(reader, GriddedNcOrthoMultiTs)

    gpis, lons, lats, _ = reader.grid.get_grid_points()
    for gpi, lon, lat in zip(gpis, lons, lats):
        for var in ["X", "Y"]:
            ref = ds[var].where((ds.lat == lat) & (ds.lon == lon), drop=True).squeeze()
            ts = reader.read(gpi)[var]
            assert np.all(ts == ref)
            ts = reader.read(lon, lat)[var]
            assert np.all(ts == ref)
    desc = {
        "X": {"long_name": "eks", "name": "X", "unit": "m"},
        "Y": {"long_name": "why", "name": "Y", "unit": "m"},
    }
    assert reader.variable_description() == desc


def test_csv_pipeline_no_metadata(test_output_path):
    # tests the full pipeline going from pandas timeseries to CSVs to zipped
    # directory to gridded contiguous ragged dataset
    timeseries, lats, lons, metadata = make_test_timeseries()
    csv_dir = test_output_path / "csv"

    make_csv_dataset(timeseries, lats, lons, csv_dir, name="test", only_ismn=False)

    # make zip file
    zfile = test_output_path / "csv.zip"
    zip_directory(csv_dir, zfile)

    # do the preprocessing with the full preprocessing function
    outpath = test_output_path / "gridded_ts"
    reader = preprocess_user_data(zfile, outpath)

    assert isinstance(reader, GriddedNcContiguousRaggedTs)
    for i in range(len(timeseries)):
        ts = reader.read(i)["soil_moisture"]
        pd.testing.assert_series_equal(ts, timeseries[i], check_freq=False)


def test_csv_pipeline_only_ismn(test_output_path):

    close_lats = [10.88, 41.36, 67.35]
    close_lons = [-1.07, -106.24, 26.68]
    apart_lats = [21.94, -16.30, 27.68]
    apart_lons = [-39.02, 78.75, 151.87]
    radius = 2

    timeseries_close, _, _, _ = make_test_timeseries(lats=close_lats, lons=close_lons)
    timeseries_apart, _, _, _ = make_test_timeseries(lats=apart_lats, lons=apart_lons)

    timeseries = timeseries_apart + timeseries_close
    lats = apart_lats + close_lats
    lons = apart_lons + close_lons

    csv_dir = test_output_path / "csv"
    make_csv_dataset(timeseries, lats, lons, csv_dir, name="test", radius=radius)

    # check that only the close_lats/lons are written and that they have the
    # correct gpis
    expected_gpis = list(range(len(apart_lats), len(lats)))
    expected_files = sorted(
        [
            str(csv_dir / f"test_gpi={i}_lat={lats[i]}_lon={lons[i]}.csv")
            for i in expected_gpis
        ]
    )
    existing_files = sorted(glob.glob(str(csv_dir / "*.csv")))
    assert expected_files == existing_files


def test_contiguous_ragged_pipeline_only_ismn(test_output_path):

    close_lats = [10.88, 41.36, 67.35]
    close_lons = [-1.07, -106.24, 26.68]
    apart_lats = [21.94, -16.30, 27.68]
    apart_lons = [-39.02, 78.75, 151.87]
    radius = 2

    timeseries_close, _, _, _ = make_test_timeseries(lats=close_lats, lons=close_lons)
    timeseries_apart, _, _, _ = make_test_timeseries(lats=apart_lats, lons=apart_lons)

    timeseries = timeseries_apart + timeseries_close
    lats = apart_lats + close_lats
    lons = apart_lons + close_lons

    pynetcf_dir = test_output_path / "pynetcf"
    make_gridded_contiguous_ragged_dataset(
        timeseries, lats, lons, pynetcf_dir, radius=radius
    )

    reader = GriddedNcContiguousRaggedTs(pynetcf_dir)
    grid = reader.grid

    expected_gpis = list(range(len(apart_lats), len(lats)))
    np.testing.assert_equal(np.sort(grid.activearrlat), np.sort(close_lats))
    np.testing.assert_equal(np.sort(grid.activearrlon), np.sort(close_lons))
    np.testing.assert_equal(np.sort(grid.activegpis), np.sort(expected_gpis))


def test_hr_cgls_preprocessing(test_output_path):
    orig = xr.open_dataset(test_data_user_upload_path / "teststack_hr_cgls_small.nc")
    assert "crs" in orig

    reader = preprocess_user_data(
        test_data_user_upload_path / "teststack_hr_cgls_small.nc",
        test_output_path / "pynetcf"
    )
    assert "crs" not in reader.variable_description().keys()
