"""
Developer's guide
=================

The ``DirectoryImageReader`` aims to provide an easy to use class to read
directories of single images, to either create a single image stack file, a
transposed stack (via ``write_transposed_dataset``), or a cell-based timeseries
dataset (via the ``repurpose`` method).

The advantage over ``xr.open_mfdataset`` is that the reader typically does not
need to open each dataset to get information on coordinates. Instead, the
reader infers timestamps from the filenames via pattern matching, and infers
grid information from the first file in the dataset (earliest timestamp).

When calling ``read`` or ``read_block``, the reader goes through all files,
opens them, extracts the necessary information, and returns them in the
required format.

Internally, the reader opens the netCDF files with ``xr.open_dataset``, then
selects the correct variables and levels, and passes them on as dictionaries of
numpy arrays internally. When calling ``read``, the arrays are used to create
``Image`` instances, when calling ``read_block``, they are assembled again to
an ``xr.Dataset`` based on the inferred timestamps and coordinates.

The reader already provides a multitude of options and should work for many
different datasets "out of the box". Nevertheless, there are often situations
when it makes sense to subclass the reader to adapt certain parts.

The main routines that should be modified when subclassing are:

* ``_open_dataset``: basic routine for reading data from a file and returning
  it as xr.Dataset
* ``_latlon_from_dataset``: For manual adaption of how the grid is generated if
  it cannot be inferred directly from the file.
* ``_metadata_from_dataset``: If metadata cannot be read from the files.
* ``_tstamps_in_file``: If the timestamp cannot be inferred from the filename
  but via other info specific to the dataset this can be used to avoid having
  to read all files only to get the timestamps.
* ``_landmask_from_dataset``: If a landmask is required (only if the ``read``
  function is used), and it cannot be read with ``_open_dataset`` and also not
  with other options. Should not be necessary very often.
* ``_read_single_file``: normally this calls ``_open_dataset`` and then returns
  the data as dictionary that maps from variable names to 3d data arrays (numpy
  or dask). If it is hard to read the data as xr.Dataset, so that overriding
  `_open_dataset` is not feasible, this could be overriden instead, but then
  all the other routines for obtaining grid/metadata/landmask info also have to
  be overriden.

In the following some examples for subclassing are provided.

Subclassing for convenience
---------------------------

If a specific dataset requires many special settings (i.e. many keyword
arguments have to be set), and the reader for the dataset is used often, it
might be more convenient to create a custom subclass that sets all the
arguments, so that one only has to specify the directory and maybe a few
remaining options. In this case the subclass only needs to override
``__init__``.

As an example, consider that we have daily ERA5 images downloaded from
Copernicus and want to extract the surface soil moisture. The filenames follow
the pattern ".*AN_%Y%m%d_%H%M.nc", and we want to rename the original variable
name "swvl1" to "soil_moisture". Then our custom reader could look like this::

    from qa4sm_preprocessing.reading import DirectoryImageReader

    class ERA5Reader(DirectoryImageReader):

        def __init__(self, directory):
            super().__init__(
                directory,
                # we specify the new name, because the renaming
                # is applied early in the processing chain
                "soil_moisture",
                time_regex_pattern=".*AN_([0-9]+)_.*.nc",
                fmt="%Y%m%d",
                rename={"swvl1": "soil_moisture"},
            )

This can then be used in custom code, and only the directory to the data has to
be specified.


Subclassing for additional preprocessing
----------------------------------------

It is often necessary to preprocess the data before using it. For example, many
datasets contain quality flags as additional variable that need to be applied
to mask out unreliable data. Another example would be a case where one is
interested in a sum of multiple variables.  In these cases it is necessary to
override the ``_open_dataset`` method.

As an example, consider that we have images containing a field "soil_moisture"
and a field "quality_flag", and that we only want to use data where the first
bit in the quality_flag is 0.  Our new reader would then be::

    from qa4sm_preprocessing.reading import DirectoryImageReader

    class NewReader(DirectoryImageReader):

        def __init__(self, directory):
            super().__init__(
                directory,
                "soil_moisture",
                fmt="XXXX",  # needs to be specified
            )

        def _open_dataset(self, fname):
            ds = super()._open_dataset(fname)
            qc = ds["quality_flag"]
            # We check if the first bit is zero by doing a bitwise AND with 1.
            # The result is 0 if the first bit is zero, and 1 otherwise.
            valid = (qc & 2**0) == 0
            return ds[["soil_moisture"]].where(valid)


Subclassing for different file types and grid info
--------------------------------------------------

If the data is not available as netCDF, or if it has an uncovenient format
(e.g. grid information can not be inferred from the files and the keyword
options are not flexible enough), the subclassing can be a bit more
complicated.

For this example, consider that we have downloaded SMAP L3 retrievals in HDF5
format. Each file contains AM and PM retrievals in separate groups, and the
variable names for both are different. We also have to mask the data based on
quality flags, and the latitude and longitude are only provided as 2D arrays
with fill values (-9999) at non-land locations (but the grid is a regular
grid). A reader could look like this::

    class SmapSMReader(DirectoryImageReader):

        def __init__(self, directory):
            self.overpass_dict = {
                "AM": {
                    "group": "Soil_Moisture_Retrieval_Data_AM",
                    "sm": "soil_moisture",
                    "qc": "retrieval_qual_flag",
                },
                "PM": {
                    "group": "Soil_Moisture_Retrieval_Data_PM",
                    "sm": "soil_moisture_dca_pm",
                    "qc": "retrieval_qual_flag_dca_pm",
                }
            }

            super().__init__(
                directory,
                "SMAP_L3_SM",
                fmt="%Y%m%d",
                time_regex_pattern=r"SMAP_L3_SM_P_([0-9]+)_R.*.h5",
                pattern="**/*.h5",
                # there are 2 timestamps in each file
                timestamps=[pd.Timedelta("6H"), pd.Timedelta("18H")]
            )

        def _open_dataset(self, fname):
            sm_arrs = []
            with h5py.File(fname, "r") as f:
                for op in ["AM", "PM"]:
                    sm = self._read_overpass(f, op)
                    sm_arrs.append(sm)
            # Now we have read the AM and PM retrievals, but we still need to
            # concatenate them along a new time axis. We don't have to set the
            # actual time values though, since they will be inferred from the
            # filename in combination with the timestamps passed in the
            # constructor.
            sm = xr.concat(sm_arrs, dim="time")
            return sm.to_dataset(name="SMAP_L3_SM")

        def _read_overpass(self, f, op):
            names = self.overpass_dict[op]
            g = f[names["group"]]
            sm = np.ma.masked_equal(g[names["sm"]][...], -9999)
            qc = g[names["qc"]][...]
            valid = (qc & 1) == 0
            sm = np.ma.masked_where(~valid, sm).filled(np.nan)
            return xr.DataArray(sm, dims=["lat", "lon"])

        def _latlon_from_dataset(self, fname):
            with h5py.File(fname, "r") as f:
                g = f["Soil_Moisture_Retrieval_Data_AM"]
                lat = self.coord_from_2d(g["latitude"], 0, fill_value=-9999)
                lon = self.coord_from_2d(g["longitude"], 1, fill_value=-9999)
            return lat, lon

In this example we also adapted ``_latlon_from_dataset``. Instead, we could
have just read the latitude and longitude in ``_open_dataset`` and added them
as coordinates to the ``xr.Dataset``, but this way we don't have to read the
latitude and longitude arrays every time we open a file.

More advanced cases
-------------------

If even less information can be obtained from the files, e.g. if the filenames
don't contain timestamps, it might be necessary to also override
`_tstamps_in_file`, `_metadata_from_dataset`, or
`_landmask_from_dataset`. Since these are edge cases, they are not shown in
detail here, but it works similar to the other examples.
"""

import cftime
import dask
import datetime
import numpy as np
import glob
from pathlib import Path
import pandas as pd
import re
from tqdm.auto import tqdm
from typing import Union, Iterable, Sequence, Dict, Tuple, List
import warnings
import xarray as xr

from .imagebase import ImageReaderBase
from .base import LevelSelectionMixin
from .exceptions import ReaderError


class DirectoryImageReader(LevelSelectionMixin, ImageReaderBase):
    r"""
    Image reader for a directory containing netcdf files.

    This works for any datasets which are stored as single image files within a
    directory (and its subdirectories).

    It can be used to create a single image stack file, a transposed stack (via
    ``write_transposed_dataset``), or a cell-based timeseries dataset (via the
    repurpose package).

    The advantage over ``xr.open_mfdataset`` is that the reader typically does
    not need to open each dataset to get information on coordinates. Instead,
    the reader infers timestamps from the filenames via pattern matching, and
    infers grid information from the first file in the dataset (earliest
    timestamp).

    It also handles situations in which multiple timestamps are in each "image"
    file (which isn't really an image file anymore in this case), and is able
    to return averages from multiple files, e.g. if there are subdaily images,
    but daily average images are required.

    Parameters
    ----------
    directory : str or Path
        Directory in which the netcdf files are located. Any file matching
        `pattern` within this directory or any subdirectories is used.
    varnames : str or list of str
        Names of the variables that should be read. If `rename` is used, this
        should be the new names.
    fmt : str, optional but strongly recommended (default: None)
        Format string to deduce timestamp from filename (without directory
        name). If it is ``None`` (default), the timestamps will be obtained
        from the files (which requires opening all files and is therefore less
        efficient).
        This must not contain any wildcards, only the format specifiers
        from ``datetime.datetime.strptime`` (e.g. %Y for year, %m for month, %d
        for day, %H for hours, %M for minutes, ...).  If such a simple pattern
        does not work for you, you can additionally specify
        `time_regex_pattern` (see below).  `fmt` should only be used if the
        files only contain a single image.
    pattern : str, optional (default: "**/*.nc")
        Glob pattern to find all files to use. If all directories should be
        search recursively, the pattern should start with "**/", similar to the
        default pattern ("**/*.nc", looks for all paths that end with ".nc".
    time_regex_pattern : str, optional (default: None)
        A regex pattern to extract the part of the filename that contains the
        time information. It must contain a statement in parentheses that is
        extracted with ``re.findall``.
        If you are using this, make sure that `fmt` matches the the part of the
        pattern that is kept.
        Example: Consider that your filenames follow the strptime/glob pattern
        ``MY_DATASET_.%Y%m%d.%H%M.*.nc``, for example, one filename could be
        ``MY_DATASET_.20200101.1200.<random_string>.nc`` and
        ``<random_string>`` is not the same for all files.
        Then you would specify
        ``time_regex_pattern="MY_DATASET_\.([0-9.]+)\..*\.nc"``. The matched
        pattern from the above example filename would then be
        ``"20200101.1200"``, so you should set ``fmt="%Y%m%d.%H%M"``.
    rename : dict, optional (default: None)
        Dictionary to use to rename variables in the file. This is applied
        after level selection but before anything else, so all other parameters
        referring to variable names except 'level' should use the new names.
    timeoffsetvarname : str, optional (default: None)
        Sometimes an image is not really an image (i.e. a snapshot at a fixed
        time), but is composed of multiple observations at different times
        (e.g. satellite overpasses). In these cases, image files often contain
        a time offset variable, that gives the exact observation time.
        Time offset is calculated after applying `rename`, so
        `timeoffsetvarname` should be the renamed variable name.
    timeoffsetunit : str, optional (default: None)
        The unit of the time offset. Required if `timeoffsetvarname` is not
        ``None``. Valid values are "seconds"/, "minutes", "hours", "days".
    transpose: list, optional (default: None)
        By default, we assume that the order of coordinates is "time", "lat",
        "lon" (if all are present and on regular grids). If the time dimension
        is not the first dimension, this can be used to provide a new ordering,
        e.g. ``("time", ...)``.
    level : dict, optional (default: None)
        If a variable has more dimensions than latitude, longitude, time (or
        location, time), e.g. a level dimension, a single value for each
        remaining dimension must be chosen. They can be passed here as
        dictionary mapping dimension name to integer index (this will then be
        passed to ``xr.DataArray.isel``) for each variable. E.g., if you have
        two variables "X" and "Y", and "Y" has a level dimension, you would
        pass ``{"Y": {"level": 2}}``.
        In case you only want read a single variable, you can also pass the
        dictionary directly, e.g. ``{"level": 2}``.
        It is also possible to read multiple levels by passing a list instead
        of a single index, .e.g. ``{"Y": {"level": [2, 3]}}``. In this case the
        resulting variables are named ``Y_2`` and ``Y_3``. Level selection is
        applied before renaming, so you can rename the ugly level names to
        nicer ones. If multiple levels are selected, e.g. ``{"Y": {"level1":
        [0], "level2": [2, 3]}}``, the levels will be selected iteratively
        starting from the left. In this example the resulting names would be
        ``Y_0_2`` and ``Y_0_3``. These names can be specified in the `rename`
        argument to give more descriptive names, e.g. ``rename={"Y_0_2": "Y2",
        "Y_0_3": "Y3"}.
    skip_missing: bool, optional (default: False)
        Whether missing variables in the image files should be skipped or raise
        an error. Default is to raise an error.
    discard_attrs : bool, optional (default: False)
        Whether to discard the attributes of the input netCDF files (reduced
        data size).
    fill_value : float, optional (default: None)
        Fill values to be masked, e.g. -9999 as a common convention.
    latname : str, optional (default: "lat")
        If `locdim` is given (i.e. for non-rectangular grids), this must be the
        name of the latitude data variable, otherwise must be the name of the
        latitude coordinate.
    lonname : str, optional (default: "lon")
        If `locdim` is given (i.e. for non-rectangular grids), this must be the
        name of the longitude data variable, otherwise must be the name of the
        longitude coordinate.
    timename : str, optional (default: "time")
        The name of the time coordinate.
    latdim : str, optional (default: None)
        The name of the latitude dimension in case it's not the same as the
        latitude coordinate variable. For curvilinear grids it should be the
        first dimension of the coordinate dimensions.
    londim : str, optional (default: None)
        The name of the longitude dimension in case it's not the same as the
        longitude coordinate variable. For curvilinear grids it should be the
        first dimension of the coordinate dimensions.
    locdim : str, optional (default: None)
        The name of the location dimension for non-rectangular grids. If this
        is given, you *MUST* provide `lonname` and `latname`.
    lat : tuple or np.ndarray, optional (default: None)
        If the latitude can not be inferred from the dataset you can specify it
        by giving (start, stop, step) or an array of latitude values
    lon : tuple or np.ndarray, optional (default: None)
        If the longitude can not be inferred from the dataset you can specify
        it by giving (start, stop, step) or an array of longitude values.
    curvilinear : bool, optional (default: False)
        Whether the grid is curvilinear, i.e. is a 2D grid, but not a regular
        lat-lon grid. In this case, `latname` and `lonname` must be given, and
        must be names of the variables containing the 2D latitude and longitude
        values. Additionally, `latdim` and `londim` must be given and will be
        interpreted as vertical and horizontal dimension.
    landmask : xr.DataArray or str, optional (default: None)
        A land mask to be applied to reduce storage size. This can either be a
        xr.DataArray of the same shape as the dataset images with ``False`` at
        non-land points, or a string.
        If it is a string, it can either be the name of a variable that is also
        in the dataset, or it can follow the pattern
        "<filename>:<variable_name>". In the latter case, the part before the
        colon is interpreted as path to a netCDF file, the part after the colon
        as the variable name of the landmask within this file.
    bbox : Iterable, optional (default: None)
        (lonmin, latmin, lonmax, latmax) of a bounding box.
    cellsize : float, optional (default: None)
        Spatial coverage of a single cell file in degrees.
    construct_grid : bool, optional (default: True)
        Whether to construct a BasicGrid instance. For very large datasets it
        might be necessary to turn this off, because the grid requires too much
        memory.
    average: str, optional (default: None)
        If specified, average multiple images. Currently only "daily" is
        implemented.
    timestamps : list/tuple, optional (default: None)
        If there is data for multiple timesteps in a single file at a regular
        interval, this can be used to specify the timestamps of the steps
        relative to the base timestamp. For example, if each daily file
        contains a value for 6AM and 6PM, one could set this to
        ``[pd.Timedelta("6H"), pd.Timedelta("12H")]
    use_tqdm : bool, optional (default: True)
        Whether you want to have a nice progressbar.
    **open_dataset_kwargs : keyword arguments
       Additional keyword arguments passed to ``xr.open_dataset``.
    """

    def __init__(
        self,
        directory: Union[Path, str],
        varnames: Union[str, Sequence],
        fmt: str = None,
        pattern: str = "**/*.nc",
        time_regex_pattern: str = None,
        rename: dict = None,
        timeoffsetvarname: str = None,
        timeoffsetunit: str = None,
        transpose: Sequence = None,
        level: dict = None,
        skip_missing: bool = False,
        discard_attrs: bool = False,
        fill_value: float = None,
        latname: str = "lat",
        lonname: str = "lon",
        timename: str = "time",
        latdim: str = None,
        londim: str = None,
        locdim: str = None,
        lat: np.ndarray = None,
        lon: np.ndarray = None,
        curvilinear: bool = False,
        landmask: xr.DataArray = None,
        bbox: Iterable = None,
        cellsize: float = None,
        construct_grid: bool = True,
        average: str = None,
        timestamps: Sequence[pd.Timedelta] = None,
        use_tqdm: bool = True,
        **open_dataset_kwargs,
    ):

        # Before we do anything we have to assemble the files, because they are
        # necessary for the setup of the parent class
        directory = Path(directory).expanduser().resolve()
        if not directory.exists():  # pragma: no cover
            raise ReaderError(f"Directory does not exist: {str(directory)}")
        filepaths = sorted(glob.glob(str(directory / pattern), recursive=True))
        if not filepaths:  # pragma: no cover
            raise ReaderError(
                f"No files matching pattern {pattern} in directory "
                f"{str(directory)}."
            )
        self._example_file = filepaths[0]
        self.fmt = fmt
        if time_regex_pattern is not None:
            self.time_regex_pattern = re.compile(time_regex_pattern)
        else:
            self.time_regex_pattern = None

        # we also need the open_dataset kwargs, because they will be used to
        # open the example file
        self.open_dataset_kwargs = open_dataset_kwargs.copy()

        varnames, rename, level = self._fix_varnames_rename_level(
            varnames, timeoffsetvarname, rename, level, skip_missing
        )
        self.timeoffsetvarname = timeoffsetvarname
        if self.timeoffsetvarname is not None:
            assert timeoffsetunit is not None
            self.timeoffsetunit = timeoffsetunit.lower()[0]
            assert self.timeoffsetunit in ["s", "m", "h", "d"]
        self.rename = rename
        self.level = level
        self.transpose = transpose
        self.average = average
        self.use_tqdm = use_tqdm
        self.fill_value = fill_value

        super().__init__(
            self._example_file,
            varnames,
            timename=timename,
            latname=latname,
            lonname=lonname,
            latdim=latdim,
            londim=londim,
            locdim=locdim,
            lat=lat,
            lon=lon,
            landmask=landmask,
            bbox=bbox,
            cellsize=cellsize,
            curvilinear=curvilinear,
            construct_grid=construct_grid,
        )

        ######################################################################
        # Time information

        # The next step is to create a map that links filepaths to available
        # timestamps
        self._file_tstamp_map = {
            path: self._tstamps_in_file(
                path, timestamps=timestamps
            )
            for path in filepaths
        }
        # tstamp_file_map maps each timestamp to the file where it can be found
        self.tstamp_file_map = {}
        for fname, tstamps in self._file_tstamp_map.items():
            for t in tstamps:
                self.tstamp_file_map[t] = fname
        self._available_timestamps = sorted(list(self.tstamp_file_map))

        # If we do averaging, the timestamps exposed by the reader is not the
        # same as the timestamps available on file
        if self.average is not None:
            self._output_tstamp_map = {}
            for tstamp in self._available_timestamps:
                output_tstamp = self._calculate_averaging_timestamp(tstamp)
                if output_tstamp in self._output_tstamp_map:
                    self._output_tstamp_map[output_tstamp].append(tstamp)
                else:
                    self._output_tstamp_map[output_tstamp] = [tstamp]
            self._timestamps = sorted(list(self._output_tstamp_map))
        else:
            self._timestamps = self._available_timestamps

        if discard_attrs:
            self.discard_attrs()

        # Done
        ######################################################################

    def _open_dataset(self, fname: Union[Path, str]) -> xr.Dataset:
        """Returns data from file as xr.Dataset"""
        # can be overriden for custom datasets
        return xr.open_dataset(fname, **self.open_dataset_kwargs)

    def _latlon_from_dataset(
        self, fname: Union[Path, str]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Generates lat/lon arrays from a file"""
        # can be overriden for custom datasets
        if fname == self._example_file:
            ds = self.example_dataset
        else:  # pragma: no cover
            # Not tested because _open_datset(fname) is tested and this is not
            # a typical use case.
            ds = self._open_dataset(fname)
        return super()._latlon_from_dataset(ds)

    def _metadata_from_dataset(
        self, fname: Union[Path, str]
    ) -> Tuple[Dict, Dict]:
        """Loads the metadata from a file"""
        # can be overriden for custom datasets
        if fname == self._example_file:
            ds = self.example_dataset
        else:  # pragma: no cover
            # Not tested because _open_datset(fname) is tested and this is not
            # a typical use case.
            ds = self._open_dataset(fname)
        return super()._metadata_from_dataset(ds)

    def _landmask_from_dataset(
        self, fname: Union[Path, str], landmask: str
    ) -> xr.DataArray:
        """Loads the landmask from an image file"""
        # can be overriden for custom datasets
        if fname == self._example_file:
            ds = self.example_dataset
        else:  # pragma: no cover
            ds = self._open_dataset(fname)
        return super()._landmask_from_dataset(ds, landmask)

    def _tstamps_in_file(
        self,
        path: Union[Path, str],
        timestamps: Sequence = None,
    ) -> List[datetime.datetime]:
        """
        Creates a list of available timestamps in the files.
        """
        # can be overriden for custom datasets
        if self.fmt is not None:
            fname = Path(path).name
            if self.time_regex_pattern is not None:
                match = self.time_regex_pattern.findall(fname)
                if not match:  # pragma: no cover
                    raise ReaderError(
                        f"Pattern {self.time_regex_pattern} did not match "
                        f"{fname}"
                    )
                timestring = match[0]
            else:
                timestring = fname
            tstamp = datetime.datetime.strptime(timestring, self.fmt)
            if timestamps is not None:
                timestamps = [tstamp + dt for dt in timestamps]
            else:
                timestamps = [tstamp]
        else:
            # open file and read time
            ds = self._open_dataset(path)
            if self.timename not in ds.indexes:  # pragma: no cover
                raise ReaderError(
                    f"Time dimension {self.timename} does not exist in "
                    f"{str(path)}"
                )
            time = ds.indexes[self.timename]
            timestamps = [t.to_pydatetime() for t in time]
        return timestamps

    @property
    def example_dataset(self) -> xr.Dataset:
        if (
            not hasattr(self, "_example_dataset")
            or self._example_dataset is None
        ):
            self._example_dataset = self._open_nice_dataset(self._example_file)
        return self._example_dataset

    def _calculate_averaging_timestamp(self, tstamp) -> datetime.datetime:
        if self.average == "daily":
            date = tstamp.date()
            return datetime.datetime(date.year, date.month, date.day)
        else:
            raise NotImplementedError("only average='daily' is implemented")

    def discard_attrs(self):
        self.global_attrs = {}
        self.array_attrs = {v: {} for v in self.varnames}

    def _read_block(
        self, start, end
    ) -> Dict[str, Union[np.ndarray, dask.array.core.Array]]:

        # reading a block:
        # - find output timestamps in given range
        # - if output timestamps == available timestamps
        #   - for required each file, read the required arrays and stack the
        #     stacks onto each other
        # - if averaging has to be done:
        #   - for each output timestamp:
        #     - do same procedure as above, then take the mean
        #   - stack all images
        times = self.tstamps_for_daterange(start, end)
        if self.average is None:
            block_dict = self._read_all_files(times, self.use_tqdm)
        else:
            # If we have to average multiple images to a single image, we will
            # read image by image
            block_dict = {varname: [] for varname in self.varnames}
            if self.use_tqdm:  # pragma: no branch
                times = tqdm(times)
            for tstamp in times:
                # read all sub-images that have to be averaged later on
                times_to_read = self._output_tstamp_map[tstamp]
                tmp_block_dict = self._read_all_files(times_to_read, False)
                for varname in self.varnames:
                    block_dict[varname].append(
                        np.mean(tmp_block_dict[varname], axis=0)
                    )
            # now we just have to convert the lists of arrays to array stacks
            for varname in self.varnames:
                block_dict[varname] = np.stack(block_dict[varname], axis=0)
        return block_dict

    def _read_single_file(self, fname, tstamps) -> dict:
        """
        Reads a single file and returns a dictionary mapping variable names to
        numpy arrays. Can be overriden in case it's easier to provide this
        format than xarray datasets.
        """
        ds = self._open_nice_dataset(fname)
        block_dict = {}
        for varname in self.varnames:
            arr = ds[varname]
            if self.timename in arr.dims:
                # if there are multiple timestamps in each file, reading is
                # a bit more complicated. In the easiest case, we have as
                # many timestamps as there are in the file, and we can just
                # return it.
                # Otherwise, if time is a coordinate, we can use
                # .sel(tstamps). If this is also not the case, we need to
                # find the indices of tstamps in the file
                if len(tstamps) != len(arr[self.timename]):
                    if self.timename in arr.coords:
                        arr = arr.sel({self.timename: tstamps})
                    else:
                        all_tstamps = self._file_tstamp_map[fname]
                        ids = [all_tstamps.index(ts) for ts in tstamps]
                        arr = arr.isel({self.timename: ids})
                block_dict[varname] = arr.data
            else:
                block_dict[varname] = arr.data[np.newaxis, ...]
        return block_dict

    def _read_all_files(self, times, use_tqdm):
        # first we need to find all files that we have to visit, and remember
        # the timestamps that we need from this file
        file_tstamp_map = {}
        for tstamp in times:
            fname = self.tstamp_file_map[tstamp]
            if fname in file_tstamp_map:
                file_tstamp_map[fname].append(tstamp)
            else:
                file_tstamp_map[fname] = [tstamp]

        # now we can open each file and extract the timestamps we need
        block_dict = {varname: [] for varname in self.varnames}
        iterator = file_tstamp_map.items()
        if use_tqdm:  # pragma: no branch
            iterator = tqdm(iterator)
        for fname, tstamps in iterator:
            _blockdict = self._read_single_file(fname, tstamps)
            for varname in block_dict:
                block_dict[varname].append(_blockdict[varname])
        for varname in self.varnames:
            block_dict[varname] = np.vstack(block_dict[varname])
        return block_dict

    def _make_nicer_ds(self, ds: xr.Dataset) -> xr.Dataset:
        if self.fill_value is not None:
            ds = ds.where(ds != self.fill_value)
        if self.transpose is not None:
            ds = ds.transpose(*self.transpose)
        ds = self.select_levels(ds)
        if self.rename is not None:
            ds = ds.rename(self.rename)
        return ds

    def _convert_timeoffset(self, ds, fname) -> xr.Dataset:
        if self.timeoffsetvarname is not None:
            var = self.timeoffsetvarname
            assert "since" not in self.timeoffsetunit, (
                "time offset units must be relative to current timestamp"
            )
            timestamp = self._tstamps_in_file(fname)[0]
            start_date = cftime.num2date(0, f"days since {str(timestamp)}")
            start = cftime.date2num(start_date, "days since 1900-01-01")

            conversion = {
                "s": 86400, "m": 24*60, "h": 24, "d": 1
            }[self.timeoffsetunit]
            # start = pd.to_datetime(timestamp).to_julian_date()
            time = start + ds[var] / conversion
            ds[var] = time
            ds[var].attrs["units"] = "days since 1900-01-01"
            ds[var].attrs["long_name"] = "Observation time"
        return ds

    def _open_nice_dataset(self, fname) -> xr.Dataset:
        ds = self._make_nicer_ds(self._open_dataset(fname))
        ds = self._convert_timeoffset(ds, fname)
        return ds

    def _fix_varnames_rename_level(
        self, varnames, timeoffsetvarname, rename, level, skip_missing
    ):
        # To skip missing variables, we have to remove the variable names
        # from `varnames`, `rename`, and `level`.
        # Since `varnames` contains the final variable names, we first have
        # to create a mapping from original on file variable names to
        # variable names in `varnames`.
        # For example, if we want to read multiple soil moisture
        # levels,  runoff, and LAI from a LIS NOAHMP output file that does
        # not contain LAI, we would maybe have something like this:
        #
        # level={"SoilMoist_tavg": [0, 1]}
        # rename={"SoilMoist_tavg_0": "SSM", "SoilMoist_tavg_1": "RZSM"",
        #         "Qs_tavg": "RUNOFF", "LAI_tavg": "LAI"}
        # varnames=["SSM", "RZSM", "RUNOFF", "LAI"]
        #
        # We then first have to create a dictionary like this:
        #
        # namemapl1 = {"SoilMoist_tavg": ["SoilMoist_tavg_0",
        #                               "SoilMoist_tavg_1"],
        #              "Qs_tavg": ["Qs_tavg"]}
        #
        # (note the missing LAI, since we derive the map from the variables
        # in the file)
        if isinstance(varnames, str):
            varnames = [varnames]
        if timeoffsetvarname is not None and timeoffsetvarname not in varnames:
            varnames.append(timeoffsetvarname)
        level = self.normalize_level(level, varnames)
        if skip_missing:
            # Be careful: skip_missing only works if _open_dataset does not do
            # any selection of varnames, renaming, or selection of levels.
            # In the default setup this is the case, but in subclasses this
            # might be different.
            # However, then it's the resposibility of whoever subclasses to
            # make sure that skip_missing is always disabled.
            ds = self._open_dataset(self._example_file)

            if level is None:
                namemapl1 = {var: [var] for var in ds.data_vars}
            else:
                namemapl1 = {}
                for var in ds.data_vars:
                    if var in level:
                        levelvars = self.__class__._select_levels_iteratively(
                            var, ds[var], level[var]
                        )
                        namemapl1[var] = [name for name, _ in levelvars]
                    else:
                        namemapl1[var] = [var]

                # Now we can already check which of the keys in namemapl1 are
                # not available in the files and remove them from `level`.
                new_level = level.copy()
                for var in level:
                    if var not in namemapl1:
                        del new_level[var]
                level = new_level

            # In the next step we can apply the renaming:
            #
            # namemapl2 = {"SoilMoist_tavg": ["SSM", "RZSM"],
            #              "Qs_tavg": ["RUNOFF"]}
            #
            if rename is None:
                namemapl2 = namemapl1.copy()
            else:
                namemapl2 = {}
                for file_var in namemapl1:
                    namemapl2[file_var] = [
                        rename[orig_var] if orig_var in rename else orig_var
                        for orig_var in namemapl1[file_var]
                    ]
                # Now we can remove the entries in rename that do not have a
                # corresponding entry in namemapl1
                new_rename = rename.copy()
                l1names = []
                for file_var in namemapl1:
                    l1names += namemapl1[file_var]
                for l1name in rename:
                    if l1name not in l1names:
                        del new_rename[l1name]
                rename = new_rename

            # Now we can finally remove the non-existing varnames
            l2names = []
            for file_var in namemapl2:
                l2names += namemapl2[file_var]
            new_varnames = []
            for l2name in varnames:
                if l2name in l2names:
                    new_varnames.append(l2name)
                else:
                    warnings.warn(
                        f"Skipping variable '{l2name}' because it does"
                        " not exist!"
                    )
            varnames = new_varnames
        return varnames, rename, level
