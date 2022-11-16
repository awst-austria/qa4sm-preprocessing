import numpy as np
import pandas as pd
from pathlib import Path
import re
import shutil
from typing import Tuple, Union, Sequence, Mapping
import yaml
from zipfile import ZipFile

from qa4sm_preprocessing.reading import StackImageReader
from qa4sm_preprocessing.reading.timeseries import (
    ZippedCsvTs,
    TimeseriesListTs,
    GriddedNcContiguousRaggedTs,
)


def make_csv_dataset(
    timeseries: Sequence,
    lats: Union[Sequence, np.ndarray],
    lons: Union[Sequence, np.ndarray],
    outpath: Union[Path, str],
    name: str,
    metadata: Mapping[str, Mapping[str, str]] = None,
):
    """
    Parameters
    ----------
    timeseries : list of pd.Series/pd.DataFrame
        List of the time series to use
    lats, lons : array-like
        Coordinates of time series.
    outpath : path
        Path where the dataset is written to.
    name : str
        Name of the dataset used for creating the output file name. The output
        file name will be "<name>_lat=<lat>_lon=<lon>.csv".
    metadata : dict of dicts
        Dictionary mapping variable names in the dataset to metadata
        dictionaries, e.g. `{"sm1": {"long_name": "soil moisture 1", "units":
        "m^3/m^3"}, "sm2": {"long_name": "soil moisture 2", "units":
        "m^3/m^3"}}`.
    """
    nloc = len(timeseries)
    assert (
        nloc == len(lons) == len(lats)
    ), "'timeseries', 'lons', and 'lats' must all have the same length!"

    for gpi, (ts, lat, lon) in enumerate(zip(timeseries, lats, lons)):
        write_timeseries(ts, gpi, lat, lon, name, outpath)
    if metadata is not None:
        metadatafile = Path(outpath) / "metadata.yml"
        with open(metadatafile, "w") as f:
            yaml.dump(metadata, f)


def make_gridded_contiguous_ragged_dataset(
    timeseries: Sequence,
    lats: Union[Sequence, np.ndarray],
    lons: Union[Sequence, np.ndarray],
    outpath: Union[Path, str],
    metadata: Mapping[str, Mapping[str, str]] = None,
):
    """
    Parameters
    ----------
    timeseries : list of pd.Series/pd.DataFrame
        List of the time series to use
    lats, lons : array-like
        Coordinates of time series.
    outpath : path
        Path where the dataset is written to.
    metadata : dict of dicts
        Dictionary mapping variable names in the dataset to metadata
        dictionaries, e.g. `{"sm1": {"long_name": "soil moisture 1", "units":
        "m^3/m^3"}, "sm2": {"long_name": "soil moisture 2", "units":
        "m^3/m^3"}}`.
    """

    nloc = len(timeseries)
    assert (
        nloc == len(lons) == len(lats)
    ), "'timeseries', 'lons', and 'lats' must all have the same length!"

    reader = TimeseriesListTs(timeseries, lats, lons, metadata=metadata)
    reader.repurpose(outpath, overwrite=True)


def write_timeseries(
    ts: Union[pd.Series, pd.DataFrame],
    gpi: int,
    lat: float,
    lon: float,
    name: str,
    directory: Union[Path, str] = ".",
):
    """
    Parameters
    ----------
    ts : series or dataframe
        Data to write to CSV format.
    gpi : int
        Unique grid point index.
    lat, lon : float
        Coordinates of location.
    name : str
        Name of the station/dataset used for creating the output file name. The
        output file name will be "<name>_lat=<lat>_lon=<lon>.csv".
    directory : path, optional (default: ".")
        Directory where to store the data. Will be created if it doesn't exist
        yet.
    """
    directory = Path(directory)
    directory.mkdir(exist_ok=True, parents=True)
    path = directory / f"{name}_gpi={gpi}_lat={lat}_lon={lon}.csv"
    ts.to_csv(path)


def preprocess_user_data(uploaded, outpath, max_filesize=10):
    """
    Preprocesses user-uploaded data to the format required for QA4SM.

    Parameters
    ----------
    uploaded : path
        Path to uploaded data.
    outpath : path
        Path where the processed data should be stored.
    max_filesize : float, optional (default: 10GB)
        Maximum extracted size for uploaded zip files.

    Returns
    -------
    reader : GriddedNcContiguousRaggedTs or GriddedNcOrthoMultiTs
        Reader for the processed dataset.
    """
    uploaded = Path(uploaded)
    outpath = Path(outpath)
    if uploaded.name.endswith(".nc"):
        # user upload of netCDF stack
        reader = StackImageReader(uploaded)
        return reader.repurpose(outpath)
    elif uploaded.name.endswith(".zip"):
        zfile = ZipFile(uploaded)

        # test if file size is not too big
        filesize = sum(f.file_size for f in zfile.infolist()) / 1024 ** 3
        if filesize > max_filesize:
            raise ValueError(
                f"Unpacked file size is too large: {filesize}GB."
                " Maximum allowed unpacked file size is {max_filesize} GB"
            )

        filelist = zfile.namelist()

        nc_files_present = any(map(lambda s: s.endswith(".nc"), filelist))
        csv_files_present = any(map(lambda s: s.endswith(".csv"), filelist))

        if not nc_files_present and not csv_files_present:  # pragma: no cover
            raise ValueError(f"{str(uploaded)} does not contain CSV or netCDF files!")
        elif nc_files_present and csv_files_present:  # pragma: no cover
            raise ValueError(
                f"{str(uploaded)} contains CSV and netCDF, only one datatype"
                " is allowed."
            )
        elif nc_files_present:
            outpath.mkdir(exist_ok=True, parents=True)
            for f in filelist:
                # extracting is a bit complicated with zipfile, in order to
                # not recreate the whole directory structure we have to do
                # manual copying
                with zfile.open(f) as zf, open(outpath / Path(f).name, "wb") as tf:
                    shutil.copyfileobj(zf, tf)
            return GriddedNcContiguousRaggedTs(outpath)
        else:  # only csv files present
            reader = ZippedCsvTs(uploaded)
            return reader.repurpose(outpath)
    else:  # pragma: no cover
        raise ValueError(
            f"Unknown uploaded data format: {str(uploaded)},"
            " only *.nc and *.zip are supported."
        )
