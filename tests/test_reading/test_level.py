import pytest
import xarray as xr

from qa4sm_preprocessing.reading.base import LevelSelectionMixin


def test_normalize_level():
    # normalize_level should make a nice dictionary mapping varnames to level
    # selection dictionaries from incomplete dictionaries in case only a single
    # variable name is given

    selector = LevelSelectionMixin()

    full = {"X": {"level": 0}, "Y": {"level": [1, 2, 3]}}

    normalized = selector.normalize_level(full, ["X", "Y"])
    assert normalized == full
    normalized = selector.normalize_level(full["X"], ["X"])
    assert normalized == {"X": full["X"]}
    normalized = selector.normalize_level(full["Y"], ["Y"])
    assert normalized == {"Y": full["Y"]}


def test_select_levels(regular_test_dataset):
    ds = regular_test_dataset
    selector = LevelSelectionMixin()

    # try selecting level that does not exist, should raise warning
    selector.level = {"X": {"level1": [0]}, "Y": {"level1": [0, 1]}}
    with pytest.warns(UserWarning, match="Selection from level 'level1'"):
        selected = selector.select_levels(ds.copy())
    xr.testing.assert_allclose(ds, selected)

    # single level selection
    ds = xr.concat((ds, ds * 2), dim="level1").transpose(..., "level1")

    selector.level = {"X": {"level1": 0}, "Y": {"level1": [0]}}
    selected = selector.select_levels(ds.copy())
    xr.testing.assert_allclose(ds.X.isel(level1=0), selected.X)
    xr.testing.assert_allclose(ds.Y.isel(level1=0), selected.Y_0)

    selector.level = {"X": {"level1": [0]}, "Y": {"level1": [0, 1]}}
    selected = selector.select_levels(ds.copy())
    xr.testing.assert_allclose(ds.X.isel(level1=0), selected.X_0)
    xr.testing.assert_allclose(ds.Y.isel(level1=0), selected.Y_0)
    xr.testing.assert_allclose(ds.Y.isel(level1=1), selected.Y_1)

    # double level selection
    ds = xr.concat((ds, ds * 2), dim="level2").transpose(..., "level2")
    selector.level = {"X": {"level1": [0], "level2": [0]},
                      "Y": {"level1": [0, 1], "level2": [1]}}
    selected = selector.select_levels(ds.copy())
    xr.testing.assert_allclose(ds.X.isel(level1=0, level2=0), selected.X_0_0)
    xr.testing.assert_allclose(ds.Y.isel(level1=0, level2=1), selected.Y_0_1)
    xr.testing.assert_allclose(ds.Y.isel(level1=1, level2=1), selected.Y_1_1)

