import pandas as pd
import numpy as np
import pytest

from qa4sm_preprocessing.level2 import SMAPL2Reader, SMOSL2Reader

from pytest import test_data_path


@pytest.mark.slow
def test_SMOSL2(test_output_path):

    outpath = test_output_path / "SMOS_L2_ts"

    reader = SMOSL2Reader(test_data_path / "SMOS_L2")

    # test reader
    data = reader.read_block()
    expected_varnames = [
        "Soil_Moisture",
        "Soil_Moisture_DQX",
        "Science_Flags",
        "Confidence_Flags",
        "Processing_Flags",
        "Chi_2",
        "RFI_Prob",
        "N_RFI_X",
        "N_RFI_Y",
        "M_AVA0",
        "acquisition_time",
    ]

    assert sorted(expected_varnames) == sorted(list(data.data_vars))
    for vn in expected_varnames:
        assert data[vn].dims == ("time", "loc")
        assert data[vn].shape == (2, 2621450)
    assert np.isfinite(data.Soil_Moisture).any()

    # check if acquisition_time is set correctly
    assert data.acquisition_time.attrs["units"] == "seconds since 2000-01-01 00:00"
    tmin = int(data.acquisition_time.min())
    tmax = int(data.acquisition_time.min())
    for tm in [tmin, tmax]:
        t = np.timedelta64(tm, "s") + np.datetime64("2000-01-01 00:00")
        assert str(t.astype("datetime64[D]")) == "2010-06-01"

    # test repurpose
    tsreader = reader.repurpose(outpath, overwrite=True)

    idx = np.where(np.isfinite(data.Soil_Moisture))[1][0]
    gpi = reader.grid.activegpis[idx]
    ts = tsreader.read(gpi)
    assert len(ts) == 2
    assert ts.index[0] == pd.Timestamp("2010-06-01 08:18:35")
    np.testing.assert_almost_equal(
        ts.Soil_Moisture.values,
        data.isel(loc=idx).to_dataframe().Soil_Moisture.values
    )
    # acquisition_time should not be in the columns anymore
    assert sorted(list(ts.columns)) == sorted(expected_varnames[:-1])


@pytest.mark.slow
def test_SMAPL2(test_output_path):

    outpath = test_output_path / "SMAP_L2_ts"

    reader = SMAPL2Reader(test_data_path / "SMAP_L2")

    # test reader
    data = reader.read_block()
    expected_varnames = [
        "soil_moisture",
        "quality_flag",
        "acquisition_time",
    ]

    assert sorted(expected_varnames) == sorted(list(data.data_vars))
    for vn in expected_varnames:
        assert data[vn].dims == ("time", "y", "x")
        assert data[vn].shape == (2, 406, 964)
    assert np.isfinite(data.soil_moisture).any()

    # check if acquisition_time is set correctly
    assert data.acquisition_time.attrs["units"] == "seconds since 2000-01-01 12:00"
    tmin = int(data.acquisition_time.min())
    tmax = int(data.acquisition_time.min())
    for tm in [tmin, tmax]:
        t = np.timedelta64(tm, "s") + np.datetime64("2000-01-01 12:00")
        assert str(t.astype("datetime64[D]")) == "2015-08-11"

    # test repurpose
    tsreader = reader.repurpose(outpath, overwrite=True)

    _, yidx, xidx = np.where(np.isfinite(data.soil_moisture))
    # Note: using 100th entry here, because the first is out of the
    # valid_min/valid_max bounds, and therefore python-netcdf automatically
    # masks them
    lat = data.lat.values[yidx[100], xidx[100]]
    lon = data.lon.values[yidx[100], xidx[100]]
    ts = tsreader.read(lon, lat)
    assert len(ts) == 2
    assert ts.index[0] == pd.Timestamp("2015-08-11 02:16:58")
    np.testing.assert_almost_equal(
        ts.soil_moisture.values,
        data.isel(y=yidx[100], x=xidx[100]).to_dataframe().soil_moisture.values
    )
    # acquisition_time should not be in the columns anymore
    assert sorted(list(ts.columns)) == sorted(expected_varnames[:-1])
