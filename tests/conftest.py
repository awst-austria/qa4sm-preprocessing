import numpy as np
import pandas as pd
from pathlib import Path
import pytest
import shutil
import xarray as xr

from qa4sm_preprocessing.nc_image_reader.readers import DirectoryImageReader, XarrayImageReader


here = Path(__file__).resolve().parent


def pytest_configure():
    pytest.test_data_path = here / "test-data" / "preprocessing"


@pytest.fixture
def test_output_path(tmpdir_factory):
    # see https://stackoverflow.com/questions/51593595
    # for reference
    tmpdir = Path(tmpdir_factory.mktemp("output"))
    yield tmpdir
    shutil.rmtree(str(tmpdir))


@pytest.fixture
def test_dataset():
    rng = np.random.default_rng(42)
    nlat, nlon, ntime = 10, 20, 100
    lat = np.linspace(0, 1, nlat)
    lon = np.linspace(0, 1, nlon)
    time = pd.date_range("2000", periods=ntime, freq="D")

    X = rng.normal(size=(ntime, nlat, nlon))

    ds = xr.Dataset(
        {"X": (["time", "lat", "lon"], X)},
        coords={"time": time, "lat": lat, "lon": lon},
    )
    return ds


@pytest.fixture
def test_loc_dataset(test_dataset):
    ds = test_dataset.stack({"location": ("lat", "lon")})
    ds["latitude"] = ds.lat
    ds["longitude"] = ds.lon
    ds = ds.drop_vars("location").rename(
        {"latitude": "lat", "longitude": "lon"}
    )
    return ds


@pytest.fixture
def default_directory_reader():
    pattern = "LIS_HIST*.nc"
    fmt = "LIS_HIST_%Y%m%d%H%M.d01.nc"
    reader = DirectoryImageReader(
        pytest.test_data_path / "lis_noahmp",
        "SoilMoist_inst",
        fmt=fmt,
        pattern=pattern,
        latdim="north_south",
        londim="east_west",
        level={"SoilMoist_profiles": 0},
        lat=(29.875, 0.25),
        lon=(-11.375, 0.25),
    )
    return reader


@pytest.fixture
def lis_noahmp_stacked(default_directory_reader):
    stack_path = pytest.test_data_path / "lis_noahmp_stacked.nc"
    if not stack_path.exists():
        block = default_directory_reader.read_block()
        block.to_netcdf(stack_path)
    return xr.open_dataset(stack_path)


@pytest.fixture
def default_xarray_reader(lis_noahmp_stacked):
    return XarrayImageReader(lis_noahmp_stacked, "SoilMoist_inst")


@pytest.fixture
def cmip_ds():
    return xr.open_dataset(
        pytest.test_data_path
        / "cmip6"
        / "mrsos_day_EC-Earth3-Veg_land-hist_r1i1p1f1_gr_19700101-19700131.nc"
    )
