import xarray as xr
import logging
import numpy as np
from pathlib import Path
from typing import Union, List, Tuple, Mapping
import warnings

from pygeogrids.netcdf import load_grid

from qa4sm_preprocessing.reading.base import GridInfo
from qa4sm_preprocessing.level2.base import L2Reader, _repurpose_level2_parse_cli_args



_smos_gridfile = Path(__file__).parent / "5deg_SMOSL2_grid.nc"
_smos_only_land_gridfile = Path(__file__).parent / "5deg_SMOSL2_grid_land.nc"


class SMOSL2Reader(L2Reader):
    def __init__(
        self,
        directory: Union[Path, str],
        varnames: Union[List[str], str] = None,
        add_overpass_flag=False,
        only_land=False,
    ):
        self.add_overpass_flag = add_overpass_flag
        self.only_land = only_land
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

        varnames = list(np.atleast_1d(varnames))
        if 'Days' in varnames:
            warnings.warn("`Days` will be included as `acquisition_time` "
                          "and are therefore ignored.")
            varnames.remove('Days')
        if 'Seconds' in varnames:
            warnings.warn("`Seconds` will be included as `acquisition_time` "
                          "and are therefore ignored.")
            varnames.remove('Seconds')

        if 'acquisition_time' not in varnames:
            warnings.warn("`acquisition_time` will be included automatically.")
            varnames.append('acquisition_time')


        if add_overpass_flag:
            add_attrs = {'Overpass': {'flag_masks': [1, 2],
                                      'flag_meanings': ['Ascending' 'Descending']}}
        else:
            add_attrs = None

        super().__init__(
            directory,
            varnames=varnames,
            add_attrs=add_attrs,
        )

        self.timekey = "acquisition_time"

    @property
    def _time_regex_pattern(self):
        return r"SM(?:_[A-Z]{4})?_MIR_SMUDP2_([0-9T]+)_.*.nc"

    def _build_filename(self, timestamp) -> str:
        filenames = list(self.blockreader.get_file_tstamp_map(np.atleast_1d(timestamp)).keys())
        assert len(filenames) == 1, "Got multiple filenames for time stamp"
        return filenames[0]

    def _gridinfo(self):
        if self.only_land:
            grid = load_grid(_smos_only_land_gridfile)
        else:
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

        # now we decode and then convert back to the original types
        ds = xr.decode_cf(ds)

        if self.add_overpass_flag:
            possible_attrs = [
                "VH:SPH:MI:TI:Ascending_Flag",
                "Variable_Header:Specific_Product_Header:Main_Info:Time_Info:Ascending_Flag",
            ]
            overpass = 0
            # Overpass 1= ASC, 2=DES
            for a in possible_attrs:
                if a in ds.attrs:
                    if ds.attrs[a] == 'A':
                        overpass = 0b1
                        break
                    if ds.attrs[a] == 'D':
                        overpass = 0b10
                        break

            if overpass == 0:
                raise ValueError("No matching overpass attr found in netcdf file.")

            ds['Overpass'] = xr.where(np.isfinite(ds['Soil_Moisture']),
                                      overpass, 0).astype(np.int8)

            if 'Overpass' not in self.varnames:
                self.varnames.append("Overpass")

        varnames = [v for v in self.varnames if v != "acquisition_time"]

        # get data types
        dtypes = {v: ds[v].dtype for v in varnames}
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
        if self.add_attrs is not None:
            for v in self.add_attrs.keys():
                attrs[v] = self.add_attrs[v]

        return attrs

    def _global_metadata(self, fname: Union[Path, str]):
        return {
            "name": "SMOS_L2",
            "description": "L2 Soil Moisture Output User Data Product",
        }

    def repurpose(self, *args, **kwargs):
        if not kwargs.get('img2ts_kwargs', None):
            kwargs['img2ts_kwargs'] = dict(backend='multiprocessing')
        return super().repurpose(*args, **kwargs)


def _repurpose_smosl2_cli():
    # to be called from the command-line via
    # repurpose_smosl2 <input_path> <output_path>
    args = _repurpose_level2_parse_cli_args(
        "Process SMOS level 2 orbit files to timeseries."
    )
    logging.basicConfig(level=logging.INFO, filename=args.logfile)
    reader = SMOSL2Reader(args.input_path, varnames=args.parameter, only_land=args.only_land)
    reader.repurpose(
        args.output_path,
        start=args.start,
        end=args.end,
        overwrite=args.overwrite,
        memory=args.memory,
    )