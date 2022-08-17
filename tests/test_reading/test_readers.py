import numpy as np
from pathlib import Path
import pandas as pd
import pytest
import time
import xarray as xr

from repurpose.img2ts import Img2Ts

from qa4sm_preprocessing.reading import (
    DirectoryImageReader,
    XarrayImageStackReader,
    XarrayTSReader,
    GriddedNcOrthoMultiTs,
)
from qa4sm_preprocessing.reading.utils import mkdate

# this is defined in conftest.py
from pytest import test_data_path


# ###############################################################################
# # Images with and without time dimension
# ###############################################################################


# def test_image_time_dimension(test_output_path):
#     # create test dataset with subdaily timestamps
#     nlat, nlon, ntime = 5, 5, 20
#     lat = np.linspace(0, 1, nlat)
#     lon = np.linspace(0, 1, nlon)
#     time = pd.date_range("2000", periods=ntime, freq="6H")

#     X = np.ones((ntime, nlat, nlon), dtype=np.float32)
#     X = (X.T * np.arange(ntime)).T

#     ds = xr.Dataset(
#         {"X": (["time", "lat", "lon"], X)},
#         coords={"time": time, "lat": lat, "lon": lon},
#     )

#     def write_images(ds, outpath, image_with_time_dim):
#         outpath = Path(outpath)
#         outpath.mkdir(exist_ok=True, parents=True)
#         for timestamp in ds.indexes["time"]:
#             if image_with_time_dim:
#                 img = ds.sel(time=slice(timestamp, timestamp))
#             else:
#                 img = ds.sel(time=timestamp)
#             fname = timestamp.strftime("img_%Y%m%dT%H%M%S.nc")
#             img.to_netcdf(outpath / fname)

#     # Write images to netCDF files once with and once without time dimension,
#     # and then try to read them with averaging
#     for image_with_time_dim in [False, True]:
#         outpath = (
#             test_output_path / f"image_with_time_dim={image_with_time_dim}"
#         )
#         write_images(ds, outpath, image_with_time_dim)

#         reader = DirectoryImageReader(
#             outpath,
#             varnames=["X"],
#             fmt="img_%Y%m%dT%H%M%S.nc",
#             average="daily",
#         )
#         assert reader.timestamps[0] < reader.timestamps[-1]
#         for i, tstamp in enumerate(reader.timestamps):
#             # assumes that ntime is a multiple of 4, otherwise edge cases can
#             # occur
#             expected = 1.5 + 4 * i
#             img = reader.read_block(tstamp, tstamp)
#             assert np.all(img.X.values == expected)


###############################################################################
# Full chain with time offset
###############################################################################


def test_SMOS(test_output_path):
    reader = DirectoryImageReader(
        test_data_path / "SMOS_L3",
        varnames=["Soil_Moisture", "Mean_Acq_Time_Seconds"],
        time_regex_pattern="SM_OPER_MIR_CLF31A_([0-9T]+)_.*.DBL.nc",
        fmt="%Y%m%dT%H%M%S",
    )

    outpath = test_output_path / "SMOS_ts"
    outpath.mkdir(exist_ok=True, parents=True)

    reshuffler = Img2Ts(
        input_dataset=reader,
        outputpath=str(outpath),
        startdate=reader.timestamps[0],
        enddate=reader.timestamps[-1],
        ts_attributes=reader.global_attrs,
        zlib=True,
        imgbuffer=3,
        cellsize_lat=5,
        cellsize_lon=5,
    )
    reshuffler.orthogonal = True
    reshuffler.calc()

    ts_reader = GriddedNcOrthoMultiTs(
        outpath, time_offset_name="Mean_Acq_Time_Seconds"
    )
    df = ts_reader.read(ts_reader.grid.activegpis[100])
    expected_timestamps = list(
        map(
            mkdate,
            [
                "2015-05-06T03:50:13",
                "2015-05-07T03:11:27",
                "2015-05-08T02:33:02",
            ],
        )
    )
    expected_values = np.array([0.162236, 0.013245, np.nan])
    assert np.all(expected_timestamps == df.index)
    np.testing.assert_almost_equal(expected_values, df.Soil_Moisture.values, 6)
    assert np.all(df.columns == ["Soil_Moisture"])
