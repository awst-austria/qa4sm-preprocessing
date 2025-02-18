import xarray as xr

from qa4sm_preprocessing.reading import StackImageReader
from qa4sm_preprocessing.reading.transpose import write_transposed_dataset

from tempfile import TemporaryDirectory
from pathlib import Path


def test_write_transposed_dataset(synthetic_test_args):

    ds, kwargs = synthetic_test_args
    stack = StackImageReader(ds, ["X", "Y"], **kwargs)

    with TemporaryDirectory() as tempdir:
        transposed_path = Path(tempdir) / "transposed.zarr"
        write_transposed_dataset(stack, transposed_path)
        transposed = xr.open_zarr(transposed_path, consolidated=True)
        xr.testing.assert_equal(ds.transpose(..., "time"), transposed)


def test_write_transposed_dataset_given_chunks(synthetic_test_args):

    ds, kwargs = synthetic_test_args
    stack = StackImageReader(ds, ["X", "Y"], **kwargs)

    if "lat" in ds.X.dims:
        chunks = {"lat": 2, "lon": 2}
    elif "y" in ds.X.dims:
        chunks = {"y": 2, "x": 2}
    else:
        chunks = {"location": 4}

    with TemporaryDirectory() as tempdir:
        transposed_path = Path(tempdir) / "transposed.zarr"
        write_transposed_dataset(stack, transposed_path, chunks=chunks)
        transposed = xr.open_zarr(transposed_path, consolidated=True)
        xr.testing.assert_equal(ds.transpose(..., "time"), transposed)

        if "lat" in ds.X.dims:
            assert dict(transposed.chunks) == {"lat": (2,), "lon": (2, 2), "time": (8,)}
        elif "y" in ds.X.dims:
            assert dict(transposed.chunks) == {"y": (2,), "x": (2, 2), "time": (8,)}
        else:
            assert dict(transposed.chunks) == {"location": (4, 4), "time": (8,)}


def test_write_transposed_dataset_fixed_stepsize(synthetic_test_args):

    ds, kwargs = synthetic_test_args
    stack = StackImageReader(ds, ["X", "Y"], **kwargs)

    with TemporaryDirectory() as tempdir:
        transposed_path = Path(tempdir) / "transposed.zarr"
        write_transposed_dataset(stack, transposed_path, stepsize=1)
        transposed = xr.open_zarr(transposed_path, consolidated=True)
        xr.testing.assert_equal(ds.transpose(..., "time"), transposed)
