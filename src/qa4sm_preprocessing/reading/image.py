"""
Developer's guide
=================

The ``DirectoryImageReader`` aims to provide an easy to use class to read
directories of single images, to either create a single image stack file, a
transposed stack (via ``write_transposed_dataset``), or a cell-based timeseries
dataset (via the ``repurpose`` method). At the same time, the
``DirectoryImageReader`` aims to be easy to subclass for specific, more
complicated datasets.

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
* ``_gridinfo_from_dataset``: For manual adaption of how the grid is generated
  if it cannot be inferred directly from the file.
* ``_metadata_from_dataset``: If metadata cannot be read from the files.
* ``_tstamps_in_file``: If the timestamp cannot be inferred from the filename
  but via other info specific to the dataset this can be used to avoid having
  to read all files only to get the timestamps.
* ``_landmask_from_dataset``: If a landmask is required (only if the ``read``
  function is used), and it cannot be read with ``_open_dataset`` and also not
  with other options. Should not be necessary very often.
* ``_read_file``: normally this calls ``_open_dataset`` and then returns the
  data as dictionary that maps from variable names to 3d data arrays (numpy or
  dask, dimensions should be time, lat, lon). If it is hard to read the data as
  xr.Dataset, so that overriding `_open_dataset` is not feasible, this could be
  overriden instead, but then all the other routines for obtaining
  grid/metadata/landmask info also have to be overriden.

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

    from qa4sm_preprocessing.reading.base import GridInfo, _1d_coord_from_2d


    class SmapSMReader(DirectoryImageReader):

        def __init__(self, directory):

            super().__init__(
                directory,
                "SMAP_L3_SM",
                fmt="%Y%m%d",
                time_regex_pattern=r"SMAP_L3_SM_P_([0-9]+)_R.*.h5",
                pattern="**/*.h5",
                # there are 2 timestamps in each file
                timestamps=[pd.Timedelta("6H"), pd.Timedelta("18H")]
            )

        def _read_file(self, fname):
            # This function only reads the actual data, but not the coordinates
            # to avoid having to read and construct the coordinates in every
            # step.
            sm_arrs = []
            with h5py.File(fname, "r") as f:
                for op in ["AM", "PM"]:
                    sm = self._read_overpass(f, op)
                    sm_arrs.append(sm)
            # Now we have read the AM and PM retrievals, but we still need to
            # concatenate them along the time axis.
            sm = np.vstack(sm_arrs)
            return {"SMAP_L3_SM": sm}

        def _read_overpass(self, f, op):
            # This function reads the data of a single overpass returns it as
            # numpy array
            names = self.overpass_dict[op]
            g = f[names["group"]]
            sm = np.ma.masked_equal(g[names["sm"]][...], -9999)
            qc = g[names["qc"]][...]
            valid = (qc & 1) == 0
            sm = np.ma.masked_where(~valid, sm).filled(np.nan)
            return sm[np.newaxis, ...]

        def _gridinfo_from_dataset(self, fname):
            # This method is called from the constructor of base.ReaderBase and
            # should return latitude and longitude as xr.DataArrays
            with h5py.File(fname, "r") as f:
                g = f["Soil_Moisture_Retrieval_Data_AM"]
                # reduces the 2D tensor product coordinates to 1D coordinates
                lat = _1d_coord_from_2d(g["latitude"], 0, fill_value=-9999)
                lon = _1d_coord_from_2d(g["longitude"], 1, fill_value=-9999)
            # since we now have 1D coordinates for a 2D array, the grid is
            # regular
            gridinfo = GridInfo(lat, lon, "regular")
            return gridinfo

        def _metadata_from_dataset(self, fname):
            array_attrs =  {"SMAP_L3_SM": {"long_name": "soil moisture",
                                           "units": "m^3/m^3"}}
            global_attrs = {"title": "SMAP level 3 soil moisture"}
            return global_attrs, array_attrs

        def _dtype_from_dataset(self, fname):
            return {"SMAP_L3_SM": float}

        @property
        def overpass_dict(self):
            return {
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

In this example we also adapted ``_gridinfo_from_dataset``,
``_metadata_from_dataset``, and ``_dtype_from_dataset``. Instead, we could have
just read the latitude and longitude in ``_open_dataset`` and added them as
coordinates to the ``xr.Dataset``, but this way we don't have to read the
latitude and longitude arrays every time we open a file.

More advanced cases
-------------------

If even less information can be obtained from the files, e.g. if the filenames
don't contain timestamps, it might be necessary to also override
`_tstamps_in_file`. Since this is an edge cases, it is not shown in detail
here, but it works similar to the other examples.
"""

import dask
import dask.array as da
import datetime
import numpy as np
import glob
from pathlib import Path
import pandas as pd
import re
from tqdm.auto import tqdm
from typing import Union, Iterable, Sequence, Mapping, Tuple, List
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
    ``repurpose`` method).

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
    latname : str, optional (default: None)
        Name of the latitude coordinate array in the dataset. If it is not
        given, it is inferred from the dataset using CF-conventions.
    lonname : str, optional (default: None)
        Name of the longitude coordinate array in the dataset. If it is not
        given, it is inferred from the dataset using CF-conventions.
    timename : str, optional (default: None)
        The name of the time coordinate. Default is "time".
    ydim : str, optional (default: None)
        The name of the latitude/y dimension in case it's not the same as the
        dimension on the latitude array of the dataset. Must be specified if
        `lat` and `lon` are passed explicitly.
    xdim : str, optional (default: None)
        The name of the longitude/x dimension in case it's not the same as the
        dimension on the longitude array of the dataset. Must be specified if
        `lat` and `lon` are passed explicitly.
    locdim : str, optional (default: None)
        The name of the location dimension for non-rectangular grids.
    lat : tuple or np.ndarray, optional (default: None)
        If the latitude can not be inferred from the dataset you can specify it
        by giving (start, stop, step) or an array of latitude values. In this
        case `lon` also has to be specified.
    lon : tuple or np.ndarray, optional (default: None)
        If the longitude can not be inferred from the dataset you can specify
        it by giving (start, stop, step) or an array of longitude values. In
        this case, `lat` also has to be specified.
    gridtype : str, optional (default: "infer")
        Type of the grid, one of "regular", "curvilinear", or "unstructured".
        By default, gridtype is inferred ("infer"). If `locdim` is passed, it
        is assumed that the grid is unstructured, and that latitude and
        longitude are 1D arrays. Otherwise, `gridtype` will be set to
        "curvilinear" if the coordinate arrays are 2-dimensional, and to
        "regular" if the coordinate arrays are 1-dimensional.
        Normally gridtype should be set to "infer". Only if the coordinate
        arrays are 2-dimensional but correspond to a tensor product of two
        1-dimensional coordinate arrays, it can be set to "regular" explicitly.
        In this case the 1-dimensional coordinate arrays are inferred from the
        2-dimensional arrays.
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
    add_attrs: dict, optional (default: None)
        Additional variable attributes that cannot be taken from the input data.
        {varname: {attr: val, ...}, ...}
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
        latname: str = None,
        lonname: str = None,
        timename: str = None,
        ydim: str = None,
        xdim: str = None,
        locdim: str = None,
        lat: np.ndarray = None,
        lon: np.ndarray = None,
        gridtype: str = "infer",
        construct_grid: bool = True,
        landmask: xr.DataArray = None,
        bbox: Iterable = None,
        cellsize: float = None,
        average: str = None,
        timestamps: Sequence[pd.Timedelta] = None,
        use_tqdm: bool = True,
        use_dask: bool = False,
        add_attrs: dict = None,
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
        self.directory = directory
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
        self.timeoffsetunit = timeoffsetunit
        self.rename = rename
        self.level = level
        self.transpose = transpose
        self.use_tqdm = use_tqdm
        self.use_dask = use_dask
        self.fill_value = fill_value

        super().__init__(
            self._example_file,
            varnames,
            timename=timename,
            latname=latname,
            lonname=lonname,
            ydim=ydim,
            xdim=xdim,
            locdim=locdim,
            lat=lat,
            lon=lon,
            gridtype=gridtype,
            construct_grid=construct_grid,
            landmask=landmask,
            bbox=bbox,
            cellsize=cellsize,
            add_attrs=add_attrs,
        )

        ######################################################################
        # Time information

        # The next step is to create a map that links filepaths to available
        # timestamps
        file_tstamp_map = {
            path: self._tstamps_in_file(path, timestamps=timestamps)
            for path in filepaths
        }

        if average is None:
            self.blockreader = _BlockDataReader(self, file_tstamp_map)
        else:
            self.blockreader = _AveragingBlockDataReader(
                self, file_tstamp_map, average
            )
        self._timestamps = self.blockreader._timestamps

        if discard_attrs:
            self.discard_attrs()

        # Done
        ######################################################################

    def _open_dataset(self, fname: Union[Path, str]) -> xr.Dataset:
        """Returns data from file as xr.Dataset"""
        # can be overriden for custom datasets
        return xr.load_dataset(fname, **self.open_dataset_kwargs)

    def _gridinfo_from_dataset(
        self, fname: Union[Path, str]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Generates lat/lon arrays from a file"""
        # can be overriden for custom datasets
        if fname == self._example_file:
            ds = self.example_dataset
        else:  # pragma: no cover
            # Not tested because _open_dataset(fname) is tested and this is not
            # a typical use case.
            ds = self._open_nice_dataset(fname)
        return super()._gridinfo_from_dataset(ds)

    def _metadata_from_dataset(
        self, fname: Union[Path, str]
    ) -> Tuple[Mapping, Mapping]:
        """Loads the metadata from a file"""
        # can be overriden for custom datasets
        if fname == self._example_file:
            ds = self.example_dataset
        else:  # pragma: no cover
            # Not tested because _open_dataset(fname) is tested and this is not
            # a typical use case.
            ds = self._open_nice_dataset(fname)
        return super()._metadata_from_dataset(ds)

    def _landmask_from_dataset(
        self, fname: Union[Path, str], landmask: str
    ) -> xr.DataArray:
        """Loads the landmask from an image file"""
        # can be overriden for custom datasets
        if fname == self._example_file:
            ds = self.example_dataset
        else:  # pragma: no cover
            ds = self._open_nice_dataset(fname)
        return super()._landmask_from_dataset(ds, landmask)

    def _dtype_from_dataset(self, fname: Union[Path, str]) -> Mapping:
        if fname == self._example_file:
            ds = self.example_dataset
        else:  # pragma: no cover
            ds = self._open_nice_dataset(fname)
        return super()._dtype_from_dataset(ds)

    def _shape_from_dataset(self, fname: Union[Path, str]) -> Mapping:
        if fname == self._example_file:
            ds = self.example_dataset
        else:  # pragma: no cover
            ds = self._open_nice_dataset(fname)
        return super()._shape_from_dataset(ds)

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

    def discard_attrs(self):
        self.global_attrs = {}
        self.array_attrs = {v: {} for v in self.varnames}

    def _read_block(
        self, start, end
    ) -> Mapping[str, Union[np.ndarray, da.core.Array]]:

        times = self.tstamps_for_daterange(start, end)
        return self.blockreader.read_timestamps(times)

    def _read_file(
        self, fname: Union[Path, str]
    ) -> Mapping[str, Union[np.ndarray, da.core.Array]]:
        # this function reads a xr.Dataset and converts it to the dictionary
        # format that is used internally
        dims = self.get_dims()
        ds = self._open_nice_dataset(fname)[self.varnames]
        # since we pass the data as dictionary of numpy arrays, we need to make
        # sure that the dimensions are in the order in which they are expected.
        actual_dims = [d for d in dims if d in list(ds.dims)]
        ds = ds.transpose(*actual_dims)
        data = {v: self._fix_ndim(ds[v].data) for v in self.varnames}
        return data

    def _make_nicer_ds(self, ds: xr.Dataset) -> xr.Dataset:
        if self.fill_value is not None:
            ds = ds.where(ds != self.fill_value)
        if self.transpose is not None:
            ds = ds.transpose(*self.transpose)
        ds = self.select_levels(ds)
        if self.rename is not None:
            ds = ds.rename(self.rename)
        return ds

    def _open_nice_dataset(self, fname) -> xr.Dataset:
        ds = self._make_nicer_ds(self._open_dataset(fname))
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
        varnames = self._maybe_add_varnames(varnames, [timeoffsetvarname])
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


class _BlockDataReader:
    # class for internal use to move some of the complexity of the mapping of
    # timestamps to files out of the reader, in the hope to make it easier for
    # people to subclass the DirectoryImageReader.

    def __init__(
        self, directoryreader, file_tstamp_map, use_tqdm: bool = True
    ):
        self.directoryreader = directoryreader
        self._file_tstamp_map = file_tstamp_map
        self._use_tqdm = use_tqdm  # can only be used to turn it off
        # invert file_tstamp_map
        self.tstamp_file_map = {}
        for fname, tstamps in self._file_tstamp_map.items():
            for t in tstamps:
                self.tstamp_file_map[t] = fname

    @property
    def _timestamps(self):
        return sorted(list(self.tstamp_file_map))

    @property
    def varnames(self):
        return self.directoryreader.varnames

    @property
    def use_dask(self):
        return self.directoryreader.use_dask

    @property
    def dtype(self):
        return self.directoryreader.dtype

    @property
    def use_tqdm(self):
        # using tqdm for a progressbar only makes sense if we do not use dask
        # otherwise it's better to use dask's diagnostic tools
        return (
            self.directoryreader.use_tqdm
            and self._use_tqdm
            and not self.use_dask
        )

    def get_file_tstamp_map(self, timestamps):
        # Here we need to get the file_tstamp_map limited to a selection of
        # timestamps. To get it, we use the inverse map (tstamp_file_map),
        # select our timestamps, and invert it again.
        file_tstamp_map = {}
        for tstamp in timestamps:
            fname = self.tstamp_file_map[tstamp]
            if fname in file_tstamp_map:
                file_tstamp_map[fname].append(tstamp)
            else:
                file_tstamp_map[fname] = [tstamp]
        return file_tstamp_map

    def read_timestamps(self, timestamps: Sequence[datetime.datetime]):
        block_dict = {v: [] for v in self.varnames}
        iterator = self.get_file_tstamp_map(timestamps).items()
        if self.use_tqdm:
            iterator = tqdm(iterator)
        for fname, tstamps in iterator:
            cur_blockdict = self.read_timestamps_from_file(fname, tstamps)
            for v in self.varnames:
                block_dict[v].append(cur_blockdict[v])
        return self._assemble_blockdict(block_dict)

    def read_timestamps_from_file(self, fname, timestamps):
        # this just handles dask, the actual implementation is in
        # self._read_single_file_timestamps
        if self.use_dask:
            ntime = len(timestamps)
            shape = self.directoryreader.get_blockshape(ntime)
            delayed_blockdict = dask.delayed(
                self._read_single_file_timestamps
            )(fname, timestamps)
            blockdict = {
                v: dask.array.from_delayed(
                    delayed_blockdict[v], shape=shape, dtype=self.dtype[v]
                )
                for v in self.varnames
            }
            return blockdict
        else:
            return self._read_single_file_timestamps(fname, timestamps)

    def _read_single_file_timestamps(self, fname, timestamps):
        data = self.directoryreader._read_file(fname)
        ntime_should = len(timestamps)
        actual_ntime = data[self.varnames[0]].shape[0]
        if ntime_should < actual_ntime:
            # There are some timestamps in the data that we are not interested
            # in. We assume that the timestamps in the file are ordered, so we
            # can just check the indices of the timestamps we want and extract
            # the corresponding data.
            tstamps_in_file = self._file_tstamp_map[fname]
            ids = [tstamps_in_file.index(ts) for ts in timestamps]
            data = {v: data[v][ids, ...] for v in data}
        return data

    def _assemble_blockdict(
        self,
        blockdict: Mapping[str, Sequence[Union[np.ndarray, da.core.Array]]],
    ) -> Mapping[str, Union[np.ndarray, da.core.Array]]:
        if self.use_dask:
            vstack = da.vstack
        else:
            vstack = np.vstack
        return {v: vstack(blockdict[v]) for v in blockdict}


class _AveragingBlockDataReader(_BlockDataReader):
    # The averaging reader does similar things than the block reader, but here
    # we modify the _file_tstamp_map, to map from output (averaged) timestamps
    # to timestamps on file. This means we will have the mapping the other way
    # around, so we adapt ``get_file_tstamp_map`` and ``_timestamps`` to return
    # the things we want.
    # In read_timestamps_from_file, we just call another block reader to do the
    # dirty work, and then average what we got from it.

    def __init__(self, directoryreader, file_tstamp_map, average):
        output_tstamp_map = {}
        available_timestamps = sum(
            [tstamps for tstamps in file_tstamp_map.values()], start=[]
        )
        for tstamp in available_timestamps:
            output_tstamp = self.calculate_averaging_timestamp(average, tstamp)
            if output_tstamp in output_tstamp_map:
                output_tstamp_map[output_tstamp].append(tstamp)
            else:
                output_tstamp_map[output_tstamp] = [tstamp]
        self._averaging_timestamps = sorted(list(output_tstamp_map))
        super().__init__(directoryreader, output_tstamp_map)

        # To remember the underlying mapping of actual timestamps to files, we
        # will additionally initialize a "normal" block reader, which can then
        # read the actual data.
        self.blockreader = _BlockDataReader(
            directoryreader, file_tstamp_map, use_tqdm=False
        )

    @property
    def _timestamps(self):
        return self._averaging_timestamps

    def get_file_tstamp_map(self, timestamps):
        # in this case we want to return all entries for the given timestamps,
        # which is easier than for the _BlockDataReader
        return {t: self._file_tstamp_map[t] for t in timestamps}

    def calculate_averaging_timestamp(
        self, average, tstamp
    ) -> datetime.datetime:
        if average == "daily":
            date = tstamp.date()
            return datetime.datetime(date.year, date.month, date.day)
        else:
            raise NotImplementedError("only average='daily' is implemented")

    def read_timestamps_from_file(self, fname, timestamps):

        # The trick here is the following: AveragedTimestampReader is a
        # BlockDataReader, therefore it will handle the looping over the
        # individual output timestamps using the read_method defined in the
        # BlockDataReader (which also handles all the dask stuff).
        # But it also has it's own BlockReader instance, which is now called to
        # do the dirty work.
        data = self.blockreader.read_timestamps(timestamps)

        # now we only have to do the averaging and return the data
        for v in self.varnames:
            # otherwise lots of warnings for mean of empty slice
            warnings.filterwarnings(
                action="ignore", message="Mean of empty slice"
            )
            # take mean along first dimension (time dimension), but add it
            # again so we have it in the shape the data is expected later on.
            data[v] = np.nanmean(data[v], axis=0)[np.newaxis, ...]
        return data
