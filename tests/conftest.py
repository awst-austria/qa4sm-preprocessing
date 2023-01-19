import numpy as np
import pandas as pd
from pathlib import Path
import pytest
import shutil
import xarray as xr

from qa4sm_preprocessing.reading import (
    DirectoryImageReader,
    StackImageReader,
)


here = Path(__file__).resolve().parent


def pytest_configure():
    pytest.test_data_path = here / "test-data" / "preprocessing"
    pytest.test_data_user_upload_path = here / "test-data" / "user_data"


@pytest.fixture
def test_output_path(tmpdir_factory):
    # see https://stackoverflow.com/questions/51593595
    # for reference
    tmpdir = Path(tmpdir_factory.mktemp("output"))
    yield tmpdir
    shutil.rmtree(str(tmpdir))


def make_regular_test_dataset():
    rng = np.random.default_rng(42)
    nlat, nlon, ntime = 2, 4, 8
    lat = np.linspace(0, 1, nlat)
    lon = np.linspace(0, 1, nlon)
    time = pd.date_range("2000", periods=ntime, freq="D")

    X = rng.normal(size=(ntime, nlat, nlon))
    Y = rng.normal(size=(ntime, nlat, nlon))

    ds = xr.Dataset(
        {"X": (["time", "lat", "lon"], X), "Y": (["time", "lat", "lon"], Y)},
        coords={"time": time, "lat": lat, "lon": lon},
    )
    ds.X.attrs["unit"] = "m"
    ds.X.attrs["long_name"] = "eks"
    ds.Y.attrs["unit"] = "m"
    ds.Y.attrs["long_name"] = "why"
    ds.attrs["description"] = "test dataset"
    return ds


@pytest.fixture
def regular_test_dataset():
    return make_regular_test_dataset()


def make_curvilinear_test_dataset():
    rng = np.random.default_rng(42)
    nlat, nlon, ntime = 2, 4, 8
    lat = np.linspace(0, 1, nlat)
    lon = np.linspace(0, 1, nlon)
    LON, LAT = np.meshgrid(lon, lat)
    time = pd.date_range("2000", periods=ntime, freq="D")

    X = rng.normal(size=(ntime, nlat, nlon))
    Y = rng.normal(size=(ntime, nlat, nlon))

    ds = xr.Dataset(
        {"X": (["time", "y", "x"], X), "Y": (["time", "y", "x"], Y)},
        coords={
            "time": time,
            "lat": (["y", "x"], LAT),
            "lon": (["y", "x"], LON),
        },
    )
    ds.X.attrs["unit"] = "m"
    ds.X.attrs["long_name"] = "eks"
    ds.Y.attrs["unit"] = "m"
    ds.Y.attrs["long_name"] = "why"
    ds.attrs["description"] = "test dataset"
    return ds


@pytest.fixture
def curvilinear_test_dataset():
    return make_curvilinear_test_dataset()


def make_unstructured_test_dataset():
    latlon_test_dataset = make_regular_test_dataset()
    ds = latlon_test_dataset.stack({"location": ("lat", "lon")})
    ds["latitude"] = ds.lat
    ds["longitude"] = ds.lon
    ds = (
        ds.drop_vars("location")
        .rename({"latitude": "lat", "longitude": "lon"})
        .assign_coords(
            {"lat": ("location", ds.lat.values), "lon": ("location", ds.lon.values)}
        )
    )
    return ds


@pytest.fixture
def unstructured_test_dataset():
    return make_unstructured_test_dataset()


@pytest.fixture(
    scope="function",
    params=["regular", "curvilinear", "unstructured"],
)
def synthetic_test_args(request):
    # this fixture can be used if a test should run with all synthetic test
    # datasets
    if request.param == "regular":
        ds = make_regular_test_dataset()
        kwargs = {}
    elif request.param == "curvilinear":
        ds = make_curvilinear_test_dataset()
        kwargs = {}
    elif request.param == "unstructured":
        ds = make_unstructured_test_dataset()
        kwargs = {"locdim": "location"}
    else:
        raise NotImplementedError
    return ds, kwargs


@pytest.fixture
def lis_noahmp_directory_image_reader():
    pattern = "**/LIS_HIST*.nc"
    fmt = "LIS_HIST_%Y%m%d%H%M.d01.nc"
    reader = DirectoryImageReader(
        pytest.test_data_path / "lis_noahmp",
        "SoilMoist_inst",
        fmt=fmt,
        pattern=pattern,
        rename={"north_south": "lat", "east_west": "lon"},
        level={"SoilMoist_profiles": 0},
        lat=(29.875, 54.75, 0.25),
        lon=(-11.375, 1.0, 0.25),
    )
    return reader


@pytest.fixture
def lis_noahmp_stacked(lis_noahmp_directory_image_reader):
    stack_path = pytest.test_data_path / "lis_noahmp_stacked.nc"
    if not stack_path.exists():
        block = lis_noahmp_directory_image_reader.read_block()
        block.to_netcdf(stack_path)
    return xr.open_dataset(stack_path)


@pytest.fixture
def default_xarray_reader(lis_noahmp_stacked):
    return StackImageReader(lis_noahmp_stacked, "SoilMoist_inst")


@pytest.fixture
def cmip_ds():
    return xr.open_dataset(
        pytest.test_data_path
        / "cmip6"
        / "mrsos_day_EC-Earth3-Veg_land-hist_r1i1p1f1_gr_19700101-19700131.nc"
    )
