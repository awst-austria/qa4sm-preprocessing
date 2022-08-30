import xarray as xr

from qa4sm_preprocessing.reading import StackImageReader
from qa4sm_preprocessing.reading.transpose import write_transposed_dataset

from pytest import test_data_path

def test_write_transposed_dataset(synthetic_test_args):

    ds, kwargs = synthetic_test_args
    stack = StackImageReader(ds, ["X", "Y"], **kwargs)

    transposed_path = test_data_path / "transposed.zarr"
    write_transposed_dataset(stack, transposed_path)
    transposed = xr.open_zarr(transposed_path, consolidated=True)
    xr.testing.assert_equal(ds.transpose(..., "time"), transposed)
    

def test_write_transposed_dataset_given_chunks(synthetic_test_args):

    ds, kwargs = synthetic_test_args
    stack = StackImageReader(ds, ["X", "Y"], **kwargs)

    if kwargs == {}:
        chunks = {"lat": 5, "lon": 5}
    elif "curvilinear" in kwargs:
        chunks = {"y": 5, "x": 5}
    else:
        chunks = {"location": 25}

    transposed_path = test_data_path / "transposed.zarr"
    write_transposed_dataset(stack, transposed_path, chunks=chunks)
    transposed = xr.open_zarr(transposed_path, consolidated=True)
    xr.testing.assert_equal(ds.transpose(..., "time"), transposed)

    if kwargs == {}:
        assert dict(transposed.chunks) == {"lat": (5,), "lon": (5, 5), "time": (20,)}
    elif "curvilinear" in kwargs:
        assert dict(transposed.chunks) == {"y": (5,), "x": (5, 5), "time": (20,)}
    else:
        assert dict(transposed.chunks) == {"location": (25, 25), "time": (20,)}


def test_write_transposed_dataset_fixed_stepsize(synthetic_test_args):

    ds, kwargs = synthetic_test_args
    stack = StackImageReader(ds, ["X", "Y"], **kwargs)

    transposed_path = test_data_path / "transposed.zarr"
    write_transposed_dataset(stack, transposed_path, stepsize=1)
    transposed = xr.open_zarr(transposed_path, consolidated=True)
    xr.testing.assert_equal(ds.transpose(..., "time"), transposed)
