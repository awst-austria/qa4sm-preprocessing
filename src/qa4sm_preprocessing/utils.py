import numpy as np
import pandas as pd
from pathlib import Path
import shutil
from typing import Tuple, Union, Sequence, Mapping
import yaml
from zipfile import ZipFile

from pygeogrids.grids import BasicGrid
from pygeogrids.netcdf import load_grid

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
    only_ismn: bool = True,
    only: Tuple = None,
    radius: float = None,
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
    only_ismn : bool, optional (default: True)
        Whether to select only timeseries within a given radius around ISMN
        stations. If this is selected, `radius` must also be passed.
    only : 2-tuple of array-like, optional (default: None)
        Latitude and longitude of target locations. If this is selected,
        `radius` must also be passed. Incompatible with `only_ismn`
    radius : float, optional (default: None)
        Search radius in km for the options `only_ismn` or `only`
    """
    nloc = len(timeseries)
    assert (
        nloc == len(lons) == len(lats)
    ), "'timeseries', 'lons', and 'lats' must all have the same length!"

    grid = _get_grid(only_ismn, only, radius)

    for gpi, (ts, lat, lon) in enumerate(zip(timeseries, lats, lons)):
        if grid is not None:
            nearest_gpi, _ = grid.find_k_nearest_gpi(
                lon, lat, k=1, max_dist=radius * 1000
            )
            if len(nearest_gpi) == 0:
                continue
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
    only_ismn: bool = True,
    only: Tuple = None,
    radius: float = None,
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
    only_ismn : bool, optional (default: True)
        Whether to select only timeseries within a given radius around ISMN
        stations. If this is selected, `radius` must also be passed.
    only : 2-tuple of array-like, optional (default: None)
        Latitude and longitude of target locations. If this is selected,
        `radius` must also be passed. Incompatible with `only_ismn`
    radius : float, optional (default: None)
        Search radius in km for the options `only_ismn` or `only`
    """

    nloc = len(timeseries)
    assert (
        nloc == len(lons) == len(lats)
    ), "'timeseries', 'lons', and 'lats' must all have the same length!"

    grid = _get_grid(only_ismn, only, radius)
    if grid is not None:
        newlats = []
        newlons = []
        newgpis = []
        newts = []
        for gpi, (ts, lat, lon) in enumerate(zip(timeseries, lats, lons)):
            nearest_gpi, _ = grid.find_k_nearest_gpi(
                lon, lat, k=1, max_dist=radius * 1000
            )
            if len(nearest_gpi) == 0:
                continue
            newlats.append(lat)
            newlons.append(lon)
            newgpis.append(gpi)
            newts.append(ts)
        assert len(newts) > 0, "No timeseries close to selected coordinates found!"
        reader = TimeseriesListTs(
            newts, newlats, newlons, metadata=metadata, gpi=newgpis
        )
    else:
        reader = TimeseriesListTs(timeseries, lats, lons, metadata=metadata)
    reader.repurpose(outpath, overwrite=True)


def _get_grid(only_ismn, only, radius):
    if only_ismn:
        assert only is None, "Only one of 'only_ismn' and 'only' can be used!"
        grid = ismn_grid()
        assert radius is not None, "If only_ism=True, 'radius' must not be None!"
    elif only:
        grid = BasicGrid(only[1], only[0])
        assert radius is not None, "If only != None, 'radius' must not be None!"
    else:
        grid = None
    return grid


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
    if uploaded.name.endswith(".nc") or uploaded.name.endswith(".nc4"):
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

        nc_files_present = any(
            map(lambda s: s.endswith(".nc") or s.endswith(".nc4"), filelist)
        )
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


def ismn_grid():
    """
    Generated from ISMN_v202202.csv via:

        import pandas as pd
        import numpy as np
        from pygeogrids.grids import BasicGrid
        from pygeogrids.netcdf import save_grid

        df = pd.read_csv("ISMN_v202202.csv", header=[0, 1])
        lat = df[("latitude", "val")]
        lon = df[("longitude", "val")]
        # to keep only unique combinations of lat and lon
        lat, lon = np.array(list(set(zip(lat, lon)))).T

        grid = BasicGrid(lon, lat)
        save_grid("ismn_grid.nc", grid)
    """
    gridfile = Path(__file__).parent / "ismn_grid.nc"
    return load_grid(gridfile)
