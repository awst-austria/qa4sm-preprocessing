import numpy as np
import shutil

from qa4sm_preprocessing.reading import StackTs, GriddedNcOrthoMultiTs, StackImageReader

import pytest
from pytest import test_data_path


def test_StackTs(regular_test_dataset):
    reader = StackTs(regular_test_dataset, "X")
    gpis, lons, lats, _ = reader.grid.get_grid_points()
    for gpi, lon, lat in zip(gpis, lons, lats):
        ref = regular_test_dataset.X.sel(lat=lat, lon=lon)
        ts = reader.read(lon, lat)["X"]
        assert np.all(ts == ref)
        ts = reader.read(gpi)["X"]
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

        
def test_GriddedNcOrthoMultiTs(synthetic_test_args):

    ds, kwargs = synthetic_test_args
    stack = StackImageReader(ds, ["X", "Y"], **kwargs)

    tspath = test_data_path / "ts_test_path"
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
    tsreader = GriddedNcOrthoMultiTs(tspath, ioclass_kws={"read_bulk": False}, read_bulk=False)
    assert tsreader.ioclass_kws["read_bulk"] is False
    with pytest.warns(
        UserWarning, match="read_bulk=False but"
    ):
        tsreader = GriddedNcOrthoMultiTs(tspath, ioclass_kws={"read_bulk": True}, read_bulk=False)
        assert tsreader.ioclass_kws["read_bulk"] is False
