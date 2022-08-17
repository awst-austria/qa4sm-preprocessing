import numpy as np

from qa4sm_preprocessing.reading import XarrayTSReader


def test_xarray_ts_reader(regular_test_dataset):
    reader = XarrayTSReader(regular_test_dataset, "X")
    _, lons, lats = reader.grid.get_grid_points()
    for lon, lat in zip(lons, lats):
        ts = reader.read(lon, lat)["X"]
        ref = regular_test_dataset.X.sel(lat=lat, lon=lon)
        assert np.all(ts == ref)


def test_xarray_ts_reader_locdim(unstructured_test_dataset):
    reader = XarrayTSReader(unstructured_test_dataset, "X", locdim="location")
    gpis, _, _ = reader.grid.get_grid_points()
    for gpi in gpis:
        ts = reader.read(gpi)["X"]
        ref = unstructured_test_dataset.X.isel(location=gpi)
        assert np.all(ts == ref)
