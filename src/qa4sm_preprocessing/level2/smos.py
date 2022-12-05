import xarray as xr
import logging
import numpy as np
from pathlib import Path
from typing import Union, List, Tuple, Mapping

from pygeogrids.netcdf import load_grid

from qa4sm_preprocessing.reading.base import GridInfo
from .base import L2Reader, _repurpose_level2_parse_cli_args


_smos_gridfile = Path(__file__).parent / "5deg_SMOSL2_grid.nc"


class SMOSL2Reader(L2Reader):
    def __init__(
        self, directory: Union[Path, str], varnames: Union[List[str], str] = None
    ):
        if varnames is None:  # pragma: no branc
            varnames = [
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
        super().__init__(
            directory,
            varnames=varnames,
        )

    @property
    def _time_regex_pattern(self):
        return r"SM_REPR_MIR_SMUDP2_([0-9T]+)_.*.nc"

    def _gridinfo(self):
        grid = load_grid(_smos_gridfile)
        return GridInfo.from_grid(grid, "unstructured")

    def _read_l2_file(
        self, fname: Union[Path, str]
    ) -> Tuple[Mapping[str, np.ndarray], Union[np.ndarray, Tuple[np.ndarray]]]:
        data = {}
        # we first have to open without mask_and_scale to figure out the right
        # datatype, because otherwise the fill values will lead to type conversion
        ds = xr.open_dataset(fname, decode_cf=False, mask_and_scale=False)

        # get acquisition time from undecoded
        acq_time = (
            ds.Days.astype("timedelta64[D]") + ds.Seconds.astype("timedelta64[s]")
        ).values
        # set zeros to NaN
        seconds_since_2000 = acq_time.astype("timedelta64[s]").astype(float)
        # convert to seconds as float
        seconds_since_2000[seconds_since_2000 == 0] = np.nan
        data["acquisition_time"] = seconds_since_2000

        # get data types
        varnames = [v for v in self.varnames if v != "acquisition_time"]
        dtypes = {v: ds[v].dtype for v in varnames}

        # now we decode and then convert back to the original types
        ds = xr.decode_cf(ds)
        data.update({v: ds[v].astype(dtypes[v]).values for v in varnames})
        gpi = ds["Grid_Point_ID"].values
        return data, gpi

    def _variable_metadata(self, fname: Union[Path, str]):
        ds = xr.open_dataset(fname, decode_cf=False, mask_and_scale=False)
        attrs = {
            var: dict(ds[var].attrs)
            for var in self.varnames
            if var != "acquisition_time"
        }
        attrs["acquisition_time"] = {
            "long_name": "Acquisition time in seconds",
            "units": "seconds since 2000-01-01 00:00",
        }
        return attrs

    def _global_metadata(self, fname: Union[Path, str]):
        return {
            "name": "SMOS_L2",
            "description": "L2 Soil Moisture Output User Data Product",
        }

    def repurpose(self, *args, **kwargs):
        return super().repurpose(*args, **kwargs, timevarname="acquisition_time")


def _repurpose_smosl2_cli():
    # to be called from the command-line via
    # repurpose_smosl2 <input_path> <output_path>
    args = _repurpose_level2_parse_cli_args(
        "Process SMOS level 2 orbit files to timeseries."
    )
    logging.basicConfig(level=logging.INFO, filename=args.logfile)
    reader = SMOSL2Reader(args.input_path, varnames=args.parameter)
    reader.repurpose(
        args.output_path,
        start=args.start,
        end=args.end,
        overwrite=args.overwrite,
        memory=args.memory,
    )
