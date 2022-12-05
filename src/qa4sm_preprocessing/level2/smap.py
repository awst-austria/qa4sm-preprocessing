import numpy as np
import h5py
import logging
from pathlib import Path
from typing import Union, List
import xarray as xr

from qa4sm_preprocessing.reading.base import GridInfo
from .base import L2Reader, _repurpose_level2_parse_cli_args

_smapl2_gridfile = Path(__file__).parent / "NSIDC0772_LatLon_EASE2_M36km_v1.0.nc"


class SMAPL2Reader(L2Reader):
    def __init__(
        self, directory: Union[Path, str], varnames: Union[List[str], str] = None
    ):
        if varnames is None:
            varnames = list(self._variable_metadata(None).keys())
        super().__init__(
            directory,
            varnames=varnames,
        )

    @property
    def _time_regex_pattern(self):
        return r"SMAP_L2_SM_P_[0-9]+_[AD]+_([0-9T]+)_R.*.h5"

    def _read_l2_file(self, fname: Union[Path, str]):
        with h5py.File(fname) as f:
            sm = f["Soil_Moisture_Retrieval_Data"]["soil_moisture"][:]
            sm[sm == -9999] = np.nan
            flag = f["Soil_Moisture_Retrieval_Data"]["retrieval_qual_flag"][:]
            time = f["Soil_Moisture_Retrieval_Data"]["tb_time_seconds"][:]
            col_idx = f["Soil_Moisture_Retrieval_Data"]["EASE_column_index"][:]
            row_idx = f["Soil_Moisture_Retrieval_Data"]["EASE_row_index"][:]
        return {"soil_moisture": sm, "quality_flag": flag, "acquisition_time": time}, (
            col_idx,
            row_idx,
        )

    def _gridinfo(self):
        ncgrid = xr.open_dataset(_smapl2_gridfile)
        return GridInfo(ncgrid.latitude.values, ncgrid.longitude.values, "curvilinear")

    def _variable_metadata(self, fname: Union[Path, str]):
        return {
            "soil_moisture": {
                "long_name": "Volumetric soil moisture",
                "units": "m^3/m^3",
                "valid_max": 0.5,
                "valid_min": 0.02,
                "_FillValue": -9999.0,
            },
            "quality_flag": {
                "flag_masks": "1s, 2s, 4s, 8s",
                "flag_meanings": (
                    "Soil_moisture_retrieval_recommended"
                    " Soil_moisture_retrieval_attempted"
                    " Soil_moisture_retrieval_success"
                    " FT_retrieval_success",
                ),
                "long_name": "Retrieval quality bit flags",
            },
            "acquisition_time": {
                "long_name": "Tb acquisition time in seconds",
                "units": "seconds since 2000-01-01 12:00",
            },
        }

    def _global_metadata(self, fname: Union[Path, str]):
        return {
            "abstract": (
                "Passive soil moisture estimates onto a 36-km global Earth-fixed grid,"
                " based on radiometer measurements acquired when the SMAP spacecraft is"
                " travelling from North to South at approximately 6:00 AM local time."
            ),
            "credit": (
                "The software that generates the Level 2 Passive Soil Moisture product"
                " and the data system that automates its production were designed and"
                " implemented at the Jet Propulsion Laboratory, California Institute of"
                " Technology in Pasadena, California."
            ),
            "originatorOrganizationName": "Jet Propulsion Laboratory",
            "otherCitationDetails": (
                "The Calibration and Validation Version 2 of the SMAP Level 2 Passive"
                " Soil Moisture Science Processing Software."
            ),
            "purpose": (
                "The SMAP L2_SM_P effort provides soil moistures based on radiometer"
                " data on a 36 km grid."
            ),
            "shortName": "SPL2SMP",
            "CompositeReleaseID": "R18290",
            "ECSVersionID": "008",
            "SMAPShortName": "L2_SM_P",
            "UUID": "9fca619b-2d2c-40fb-a682-dcc4f1be0e20",
        }

    def repurpose(self, *args, **kwargs):
        return super().repurpose(*args, **kwargs, timevarname="acquisition_time")


def _repurpose_smapl2_cli():
    # to be called from the command-line via
    # repurpose_smapl2 <input_path> <output_path>
    args = _repurpose_level2_parse_cli_args(
        "Process SMAP level 2 orbit files to timeseries."
    )
    logging.basicConfig(level=logging.INFO, filename=args.logfile)
    reader = SMAPL2Reader(args.input_path, varnames=args.parameter)
    reader.repurpose(
        args.output_path,
        start=args.start,
        end=args.end,
        overwrite=args.overwrite,
        memory=args.memory,
    )
