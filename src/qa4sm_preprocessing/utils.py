import numpy as np
import pandas as pd
from pathlib import Path
import shutil
from typing import Tuple, Union, Sequence, Mapping
import yaml
from zipfile import ZipFile
import netCDF4
import io

from pygeogrids.grids import BasicGrid
from pygeogrids.netcdf import load_grid

from qa4sm_preprocessing.reading import StackImageReader
from qa4sm_preprocessing.reading.timeseries import (
    ZippedCsvTs,
    TimeseriesListTs,
    GriddedNcContiguousRaggedTs,
)
from qa4sm_preprocessing.format_validator import NetCDFValidator, ZipValidator


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
    assert (nloc == len(lons) == len(lats)
           ), "'timeseries', 'lons', and 'lats' must all have the same length!"

    grid = _get_grid(only_ismn, only, radius)

    for gpi, (ts, lat, lon) in enumerate(zip(timeseries, lats, lons)):
        if grid is not None:
            nearest_gpi, dist = grid.find_k_nearest_gpi(
                lon, lat, k=1, max_dist=radius * 1000)
            if dist == np.inf:
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
    assert (nloc == len(lons) == len(lats)
           ), "'timeseries', 'lons', and 'lats' must all have the same length!"

    grid = _get_grid(only_ismn, only, radius)
    if grid is not None:
        newlats = []
        newlons = []
        newgpis = []
        newts = []
        for gpi, (ts, lat, lon) in enumerate(zip(timeseries, lats, lons)):
            nearest_gpi, dist = grid.find_k_nearest_gpi(
                lon, lat, k=1, max_dist=radius * 1000)
            if dist == np.inf:
                continue
            newlats.append(lat)
            newlons.append(lon)
            newgpis.append(gpi)
            newts.append(ts)
        assert len(
            newts) > 0, "No timeseries close to selected coordinates found!"
        reader = TimeseriesListTs(
            newts, newlats, newlons, metadata=metadata, gpi=newgpis)
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


def preprocess_user_data(uploaded, outpath, max_filesize=30):
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
        filesize = sum(f.file_size for f in zfile.infolist()) / 1e9
        if filesize > max_filesize:
            raise ValueError(
                f"Unpacked file size is too large: {filesize}GB."
                " Maximum allowed unpacked file size is {max_filesize} GB")

        filelist = zfile.namelist()

        nc_files_present = any(
            map(lambda s: s.endswith(".nc") or s.endswith(".nc4"), filelist))
        csv_files_present = any(map(lambda s: s.endswith(".csv"), filelist))

        if not nc_files_present and not csv_files_present:  # pragma: no cover
            raise ValueError(
                f"{str(uploaded)} does not contain CSV or netCDF files!")
        elif nc_files_present and csv_files_present:  # pragma: no cover
            raise ValueError(
                f"{str(uploaded)} contains CSV and netCDF, only one datatype"
                " is allowed.")
        elif nc_files_present:
            outpath.mkdir(exist_ok=True, parents=True)
            for f in filelist:
                # extracting is a bit complicated with zipfile, in order to
                # not recreate the whole directory structure we have to do
                # manual copying
                with zfile.open(f) as zf, open(outpath / Path(f).name,
                                               "wb") as tf:
                    shutil.copyfileobj(zf, tf)
            return GriddedNcContiguousRaggedTs(outpath)
        else:  # only csv files present
            reader = ZippedCsvTs(uploaded)
            return reader.repurpose(outpath)
    else:  # pragma: no cover
        raise ValueError(f"Unknown uploaded data format: {str(uploaded)},"
                         " only *.nc and *.zip are supported.")


def verify_file_extension(filename):
    """Check if file has a valid extension."""
    valid_extensions = ('.nc4', '.nc', '.zip')
    return filename.lower().endswith(valid_extensions)


def validate_file_upload(request, file, expected_filename):
    """
    Validate file upload with comprehensive checks.

    Returns:
        tuple: (is_valid, error_message, status_code)
    """
    # Check filename matches expectation
    if file.name != expected_filename:
        return False, f"Expected '{expected_filename}', got '{file.name}'", 400

    # Check file extension
    if not verify_file_extension(file.name):
        return False, "File must be .nc4, .nc, or .zip format", 400

    # Check file size against user's available space
    if hasattr(request.user, 'space_left') and request.user.space_left:
        if file.size > request.user.space_left:
            return False, f"File size ({file.size} bytes) exceeds available space ({request.user.space_left} bytes)", 413

    # NETCDF Case
    if file.name.endswith(".nc") or file.name.endswith(".nc4"):
        try:
            # Reset file pointer to beginning
            file.seek(0)

            # Read file content into memory
            file_content = file.read()
            file.seek(0)  # Reset for potential later use

            # Create in-memory dataset
            memory_file = io.BytesIO(file_content)

            # Open NetCDF dataset from memory
            with netCDF4.Dataset(
                    'dummy', mode='r', memory=memory_file.read()) as nc:
                validator = NetCDFValidator(nc)
                is_valid, results, status = validator.validate()

                if is_valid:
                    # Success case - extract relevant info for success message
                    vars = len(validator.get_variables())
                    message = f"NetCDF file validation successful. Found {vars} variables."

                    return True, message, 200
                else:
                    # Validation failed - compile error messages
                    error_messages = results.get('errors',
                                                 ['Unknown validation error'])
                    error_summary = "; ".join(
                        error_messages[:3])  # Limit to first 3 errors

                    if len(error_messages) > 3:
                        error_summary += f" (and {len(error_messages) - 3} more errors)"

                    return False, f"NetCDF validation failed: {error_summary}", status

        except Exception as e:
            return False, f"Error reading NetCDF file: {str(e)}", 500

    # ZIP Case
    elif file.name.endswith(".zip"):
        try:
            # Reset file pointer to beginning
            file.seek(0)

            # Read file content into memory
            file_content = file.read()
            file.seek(0)  # Reset for potential later use

            # Create in-memory zip file
            memory_file = io.BytesIO(file_content)

            validator = ZipValidator(memory_file)
            is_valid, results, status = validator.validate()

            if is_valid:
                # Success case - extract relevant info for success message
                message = "ZIP file validation successful."

                # Add specific details if available in results
                if results.get('file_count'):
                    message += f" Found {results['file_count']} files."

                # Include warnings if any
                if results.get('warnings'):
                    warning_count = len(results['warnings'])
                    message += f" ({warning_count} warning(s) found)"

                return True, message, status
            else:
                # Validation failed - compile error messages
                error_messages = results.get('errors',
                                             ['Unknown validation error'])
                error_summary = "; ".join(
                    error_messages[:3])  # Limit to first 3 errors

                if len(error_messages) > 3:
                    error_summary += f" (and {len(error_messages) - 3} more errors)"

                return False, f"ZIP validation failed: {error_summary}", status

        except Exception as e:
            return False, f"Error reading ZIP file: {str(e)}", 500

    # This should not be reached due to file extension check above
    return False, "Unsupported file format", 400


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


if __name__ == '__main__':
    preprocess_user_data("/tmp/test/ascat.zip", "/tmp")
