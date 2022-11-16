import numpy as np
import pandas as pd
from pathlib import Path
import shutil
import yaml
import zipfile

from qa4sm_preprocessing.reading import (
    StackImageReader,
    GriddedNcOrthoMultiTs,
    GriddedNcContiguousRaggedTs,
)
from qa4sm_preprocessing.reading.timeseries import ZippedCsvTs
from qa4sm_preprocessing.utils import (
    make_csv_dataset,
    make_gridded_contiguous_ragged_dataset,
    preprocess_user_data,
)

from pytest import test_data_path


def make_test_timeseries():
    # timeseries 1: daily timestamps
    index = pd.date_range("2020-01-01 12:00", "2020-12-31 12:00", freq="D")
    data = np.random.randn(len(index))
    ts1 = pd.Series(data, index=index, name="soil_moisture")
    lat1 = 12
    lon1 = 0.1

    # timeseries 2: hourly timestamps, but shorter period
    index = pd.date_range("2020-06-01", "2020-08-01", freq="H")
    data = np.random.randn(len(index)) + 2
    ts2 = pd.Series(data, index=index, name="soil_moisture")
    lat2 = 34
    lon2 = 2.3

    # timeseries 3: irregular timestamps
    index = pd.DatetimeIndex(
        np.datetime64("2020-01-01")
        + np.random.rand(1000) * np.timedelta64(365 * 24 * 60, "m")
    )
    data = np.random.randn(len(index)) - 2
    ts3 = pd.Series(data, index=index, name="soil_moisture")
    lat3 = 56
    lon3 = 4.5

    lats = [lat1, lat2, lat3]
    lons = [lon1, lon2, lon3]
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



def test_csv_pipeline():
    # tests the full pipeline going from pandas timeseries to CSVs to zipped
    # directory to gridded contiguous ragged dataset
    timeseries, lats, lons, metadata = make_test_timeseries()
    csv_dir = test_data_path / "csv"
    if csv_dir.exists():
        shutil.rmtree(csv_dir)

    make_csv_dataset(timeseries, lats, lons, csv_dir, name="test", metadata=metadata)

    # check if they all look okay
    for i in range(len(timeseries)):
        fname = f"test_lat={lats[i]}_lon={lons[i]}.csv"
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
    zip_directory(csv_dir, test_data_path / "csv.zip")

    # try to read with ZippedCsvTs
    reader = ZippedCsvTs(test_data_path / "csv.zip")
    for i in range(len(timeseries)):
        ts = reader.read(i)["soil_moisture"]
        pd.testing.assert_series_equal(ts, timeseries[i], check_freq=False)
    assert reader.get_metadata("soil_moisture") == metadata["soil_moisture"]

    # do the preprocessing with the full preprocessing function
    outpath = test_data_path / "gridded_ts"
    if outpath.exists():
        shutil.rmtree(outpath)
    reader = preprocess_user_data(test_data_path / "csv.zip", outpath)

    assert isinstance(reader, GriddedNcContiguousRaggedTs)
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


def test_contiguous_ragged_pipeline():
    # tests the full pipeline going from pandas timeseries to zipped pynetcf
    # directory to gridded contiguous ragged dataset
    timeseries, lats, lons, metadata = make_test_timeseries()
    pynetcf_dir = test_data_path / "pynetcf"
    if pynetcf_dir.exists():
        shutil.rmtree(pynetcf_dir)

    make_gridded_contiguous_ragged_dataset(
        timeseries, lats, lons, pynetcf_dir, metadata=metadata
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
    zip_directory(pynetcf_dir, test_data_path / "pynetcf.zip")

    # do the preprocessing with the full preprocessing function
    outpath = test_data_path / "gridded_ts"
    if outpath.exists():
        shutil.rmtree(outpath)
    reader = preprocess_user_data(test_data_path / "pynetcf.zip", outpath)
    assert isinstance(reader, GriddedNcContiguousRaggedTs)
    check_reader(reader)


def test_stack_pipeline(synthetic_test_args):
    stackpath = test_data_path / "stack.nc"
    if stackpath.exists():
        stackpath.unlink()

    ds, kwargs = synthetic_test_args
    ds.to_netcdf(stackpath)

    reader = preprocess_user_data(stackpath, test_data_path / "stack_ts")
    assert isinstance(reader, GriddedNcOrthoMultiTs)

    gpis, lons, lats, _ = reader.grid.get_grid_points()
    for gpi, lon, lat in zip(gpis, lons, lats):
        for var in ["X", "Y"]:
            ref = ds[var].where((ds.lat == lat) & (ds.lon == lon), drop=True).squeeze()
            ts = reader.read(gpi)[var]
            assert np.all(ts == ref)
            ts = reader.read(lon, lat)[var]
            assert np.all(ts == ref)


def test_csv_pipeline_no_metadata():
    # tests the full pipeline going from pandas timeseries to CSVs to zipped
    # directory to gridded contiguous ragged dataset
    timeseries, lats, lons, metadata = make_test_timeseries()
    csv_dir = test_data_path / "csv"
    if csv_dir.exists():
        shutil.rmtree(csv_dir)

    make_csv_dataset(timeseries, lats, lons, csv_dir, name="test")

    # make zip file
    zip_directory(csv_dir, test_data_path / "csv.zip")

    # do the preprocessing with the full preprocessing function
    outpath = test_data_path / "gridded_ts"
    if outpath.exists():
        shutil.rmtree(outpath)
    reader = preprocess_user_data(test_data_path / "csv.zip", outpath)

    assert isinstance(reader, GriddedNcContiguousRaggedTs)
    for i in range(len(timeseries)):
        ts = reader.read(i)["soil_moisture"]
        pd.testing.assert_series_equal(ts, timeseries[i], check_freq=False)
