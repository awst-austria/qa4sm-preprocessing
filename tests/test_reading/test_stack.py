import numpy as np
import xarray as xr

from qa4sm_preprocessing.reading import (
    StackImageReader,
)

from qa4sm_preprocessing.reading.base import _1d_coord_from_2d

from .utils import validate_reader


def test_cf_conventions(synthetic_test_args):

    ds, kwargs = synthetic_test_args

    # rename coordinates but give them CF convention units
    ds.lat.attrs["units"] = "degrees_north"
    ds.lat.attrs["long_name"] = "latitude"
    ds.lon.attrs["units"] = "degrees_east"
    ds.lon.attrs["long_name"] = "longitude"
    ds.time.attrs["long_name"] = "time"
    ds = ds.rename({"lat": "mylat", "lon": "mylon", "time": "mytime"})

    reader = StackImageReader(ds, ["X", "Y"], **kwargs)
    assert reader.latname == "mylat"
    assert reader.lonname == "mylon"
    assert reader.timename == "mytime"
    if reader.gridtype == "regular":
        assert reader.ydim == "mylat"
        assert reader.xdim == "mylon"
        assert reader.locdim == None
    elif reader.gridtype == "curvilinear":
        assert reader.ydim == "y"
        assert reader.xdim == "x"
        assert reader.locdim == None
    else:
        assert reader.ydim == None
        assert reader.xdim == None
        assert reader.locdim == "location"
    validate_reader(reader, ds)


def test_latitude_longitude(synthetic_test_args):
    ds, kwargs = synthetic_test_args
    ds = ds.rename({"lat": "latitude", "lon": "longitude"})
    reader = StackImageReader(ds, **kwargs)
    assert reader.latname == "latitude"
    assert reader.lonname == "longitude"
    validate_reader(reader, ds)


def test_LaTitude_LonGitude(synthetic_test_args):
    ds, kwargs = synthetic_test_args
    ds = ds.rename({"lat": "LaTitude", "lon": "LonGitude"})
    ds.LaTitude.attrs["long_name"] = "LaTitude"
    ds.LonGitude.attrs["long_name"] = "LonGitude"
    reader = StackImageReader(ds, **kwargs)
    assert reader.latname == "LaTitude"
    assert reader.lonname == "LonGitude"
    validate_reader(reader, ds)


def test_2d_to_1d(curvilinear_test_dataset):
    # this tests if we can infer the 1D coordinate arrays from 2D arrays if
    # the 2D arrays are tensor products of the 1D products
    ds = curvilinear_test_dataset
    lat = _1d_coord_from_2d(ds.lat.values, 0)
    lon = _1d_coord_from_2d(ds.lon.values, 1)
    ds_1d = (ds.copy()
         .assign_coords({"y": lat, "x": lon})
         .drop(["lat", "lon"])
         .rename({"y": "lat", "x": "lon"})
     )

    reader = StackImageReader(ds_1d)

    dims = dict(ds.dims)
    lat = np.linspace(0, 1, dims["y"])
    lon = np.linspace(0, 1, dims["x"])

    np.testing.assert_almost_equal(lat, reader.lat, 15)
    np.testing.assert_almost_equal(lon, reader.lon, 15)

    # transform the dataset to the regular grid structure
    ds = (
        ds.drop_vars(["lat", "lon"])
        .rename({"y": "lat", "x": "lon"})
        .assign_coords({"lat": ("lat", lat), "lon": ("lon", lon)})
    )
    validate_reader(reader, ds)


def test_no_time(synthetic_test_args):
    ds, kwargs = synthetic_test_args
    ds = ds.rename({"time": "z"})
    reader = StackImageReader(ds, **kwargs)
    assert reader.timename == "z"
    validate_reader(reader, ds)


def test_coordinate_metadata(synthetic_test_args):
    ds, kwargs = synthetic_test_args
    reader = StackImageReader(ds, **kwargs)

    block = reader.read_block()
    assert block.lat.attrs["units"] == "degrees_north"
    assert block.lon.attrs["units"] == "degrees_east"
    assert block.lat.attrs["standard_name"] == "latitude"
    assert block.lon.attrs["standard_name"] == "longitude"
    assert block.time.attrs["standard_name"] == "time"


def test_dimension_orders(synthetic_test_args):
    ds, kwargs = synthetic_test_args
    if "lat" in ds.X.dims:
        newds = ds.transpose("lon", "time", "lat")
        expected_dims = ("time", "lat", "lon")
    elif "x" in ds.X.dims:
        newds = ds.transpose("x", "time", "y")
        expected_dims = ("time", "y", "x")
    else:
        newds = ds.transpose("location", "time")
        expected_dims = ("time", "location")
    reader = StackImageReader(newds, **kwargs)
    assert reader.latname == "lat"
    assert reader.lonname == "lon"
    assert reader.timename == "time"
    assert reader.get_dims() == expected_dims
    validate_reader(reader, ds)


def test_small_cellsize(synthetic_test_args):
    ds, kwargs = synthetic_test_args
    reader = StackImageReader(ds, **kwargs, cellsize=0.01)
    assert len(np.unique(reader.grid.activearrcell)) > 1
