"""
Timeseries and image readers for wrapping xarray.Datasets, compatible with
readers from the TUW-GEO python package universe (e.g. pygeobase, pynetcf).
"""

from abc import abstractmethod
import dask
import dask.array as da
import datetime
import logging
import numpy as np
import os
import glob
import pandas as pd
from pathlib import Path
import re
from tqdm.auto import tqdm
from typing import Union, Iterable, List, Tuple, Sequence, Dict
import warnings
import xarray as xr

from pygeobase.object_base import Image
from pygeogrids.grids import gridfromdims, BasicGrid, CellGrid
from pygeogrids.netcdf import load_grid
from pynetcf.time_series import GriddedNcOrthoMultiTs as _GriddedNcOrthoMultiTs

from .exceptions import ReaderError
from .utils import mkdate, infer_chunks


class XarrayReaderBase:
    """
    Base class for readers backed by xarray objects (images, image stacks,
    timeseries).

    This base class provides methods to infer grid information and metadata
    from the stack and set them up for use in reading methods.

    Note that the constructor needs a dataset from which to derive the grid,
    the metadata, and the land mask, so you might have to get this first in the
    child classes, before you can call the parent constructor.
    """

    def __init__(
        self,
        ds: Union[xr.Dataset, str],
        varnames: Union[str, Sequence],
        level: dict = None,
        timename: str = "time",
        latname: str = None,
        lonname: str = None,
        latdim: str = None,
        londim: str = None,
        locdim: str = None,
        lat: tuple = None,
        lon: tuple = None,
        landmask: Union[xr.DataArray, str] = None,
        bbox: Iterable = None,
        cellsize: float = None,
        curvilinear: bool = False,
        rename: dict = None,
    ):
        if isinstance(varnames, str):
            varnames = [varnames]
        self.varnames = list(varnames)
        if (
            level is not None
            and len(self.varnames) == 1
            and not isinstance(list(level.values())[0], dict)
        ):
            self.level = {self.varnames[0]: level}
        else:
            self.level = level
        self.rename = rename
        self.timename = timename
        if latname is None:
            self.latname = "lat"
        else:
            self.latname = latname
        if lonname is None:
            self.lonname = "lon"
        else:
            self.lonname = lonname
        self.latdim = latdim
        self.londim = londim
        self.locdim = locdim
        self.curvilinear = curvilinear
        if self.curvilinear and (
            latdim is None
            or londim is None
            or latname is None
            or lonname is None
            or locdim is not None
        ):
            raise ReaderError(
                "For curvilinear grids, lat/lon-dim and lat/lon-name must "
                "be given."
            )
        if locdim is not None and (
            latname is None or lonname is None
        ):  # pragma: no cover
            raise ReaderError(
                "If locdim is given, latname and lonname must also be given."
            )
        self._lat = lat
        self._lon = lon
        self._has_2d_grid = locdim is None
        self._has_regular_grid = self._has_2d_grid and not self.curvilinear
        self.bbox = bbox
        self.cellsize = cellsize
        # Landmask can be "<filename>:<variable_name>", "<variable_name>", or a
        # xr.DataArray. The latter two cases are handled elsewhere
        if isinstance(landmask, str):
            if ":" in landmask:
                fname, vname = landmask.split(":")
                self.landmask = xr.open_dataset(fname)[vname]
            else:
                self.landmask = ds[landmask]
        else:
            self.landmask = landmask

        ds = self._select_levels(ds)
        if self.rename is not None:
            ds = ds.rename(self.rename)
        (self.global_attrs, self.array_attrs,) = self._metadata_from_xarray(ds)

        self.lat = self._coord_from_xarray(ds, "lat")
        self.lon = self._coord_from_xarray(ds, "lon")
        self.grid = self._grid_from_xarray(ds)

    def _metadata_from_xarray(self, ds: xr.Dataset) -> Tuple[dict, dict]:
        global_attrs = dict(ds.attrs)
        array_attrs = {v: dict(ds[v].attrs) for v in self.varnames}
        return global_attrs, array_attrs

    def _grid_from_xarray(self, ds: xr.Dataset) -> CellGrid:

        # if using regular lat-lon grid, we can use gridfromdims
        if self._has_2d_grid:
            if self.curvilinear:
                grid = BasicGrid(
                    self.lon.values.ravel(), self.lat.values.ravel()
                )
            else:
                grid = gridfromdims(self.lon, self.lat)
        else:
            grid = BasicGrid(self.lon, self.lat)

        if hasattr(self, "landmask") and self.landmask is not None:
            if self._has_2d_grid:
                landmask = self.landmask.stack(
                    dimensions={"loc": (self.latname, self.lonname)}
                )
            else:
                landmask = self.landmask
            land_gpis = grid.get_grid_points()[0][landmask]
            grid = grid.subgrid_from_gpis(land_gpis)

        # bounding box
        if hasattr(self, "bbox") and self.bbox is not None:
            # given is: bbox = [lonmin, latmin, lonmax, latmax]
            lonmin, latmin, lonmax, latmax = (*self.bbox,)
            bbox_gpis = grid.get_bbox_grid_points(
                lonmin=lonmin, latmin=latmin, lonmax=lonmax, latmax=latmax,
            )
            grid = grid.subgrid_from_gpis(bbox_gpis)
        num_gpis = len(grid.activegpis)
        logging.debug(f"_grid_from_xarray: Number of active gpis: {num_gpis}")

        if hasattr(self, "cellsize") and self.cellsize is not None:
            grid = grid.to_cell_grid(cellsize=self.cellsize)
            num_cells = len(grid.get_cells())
            logging.debug(
                f"_grid_from_xarray: Number of grid cells: {num_cells}"
            )

        return grid

    def _coord_from_xarray(
        self, ds: xr.Dataset, coordname: str
    ) -> xr.DataArray:
        # coordname must be either "lat" or "lon", independent of what's in the
        # dataset, this only chooses attributes from this class based on the
        # choice
        cname = getattr(self, coordname + "name")
        dimname = getattr(self, coordname + "dim")
        _coord = getattr(self, "_" + coordname)
        if dimname is None:
            # coordinate is a dimension in dataset, so we can just return it
            return ds[cname]
        dimlen = len(ds[dimname])
        if self.curvilinear:
            # for curvilinear grids, we assume that latitude and longitude are
            # given via their coordinate name, and they will be 2D arrays
            return ds[cname]
        if _coord is not None:
            if isinstance(_coord, (list, tuple)) and len(_coord) == 2:
                start, step = _coord
                coord = np.array([start + i * step for i in range(dimlen)])
            elif isinstance(_coord, np.ndarray):
                coord = _coord
        else:
            # infer coordinate from variable in dataset
            coord = ds[cname]
            if len(coord.dims) > 1:
                othername = "lon" if coordname == "lat" else "lat"
                other_cname = getattr(self, othername + "dim")
                warnings.warn(
                    f"{cname} has more than one dimension, using values for"
                    f" {other_cname} index = 0"
                )
                coord = coord.isel({other_cname: 0})
            coord = coord.rename({dimname: cname})
            if np.any(np.isnan(coord)):  # pragma: no cover
                raise ReaderError(
                    f"Inferred coordinate values for {coordname}"
                    " contain NaN! Try using the 'lat' and 'lon'"
                    " keyword arguments to specify the coordinates"
                    " directly."
                )
        return xr.DataArray(
            coord, coords={cname: coord}, dims=[cname], name=cname,
        )

    def _select_levels(self, ds):
        if self.level is not None:
            for varname in self.level:
                variable_levels = self.__class__._select_levels_iteratively(
                    varname, ds[varname], self.level[varname]
                )
                if len(variable_levels) == 1:
                    ds[varname] = variable_levels[0][1]
                else:
                    del ds[varname]
                    for name, arr in variable_levels:
                        ds[name] = arr
        return ds

    @staticmethod
    def _select_levels_iteratively(name, arr, leveldict):
        # input list: list of (name, arr, leveldict)
        output_list = [(name, arr)]
        for levelname, idx in leveldict.items():
            if not isinstance(idx, list):
                idx = [idx]
            tmp_list = []
            for name, arr in output_list:
                for i in idx:
                    tmparr = arr.isel({levelname: i})
                    tmpname = name + "_" + str(i)
                    tmp_list.append((tmpname, tmparr))
            output_list = tmp_list
        return output_list


class XarrayImageReaderBase(XarrayReaderBase):
    """
    Base class for image readers backed by xarray objects (multiple single
    images or single stack of multiple images).

    Provides the methods
    - self.tstamps_for_daterange
    - self.read
    - self.read_block

    and therefore meets all prerequisites for Img2Ts.

    Child classes must override `_read_block` and need to set the attribute
    self.timestamps to an iterable of available timestamps.
    """

    def _validate_start_end(
        self,
        start: Union[datetime.datetime, str],
        end: Union[datetime.datetime, str],
    ) -> Tuple[datetime.datetime]:
        if start is None:
            start = self.timestamps[0]
        elif isinstance(start, str):
            start = mkdate(start)
        if end is None:
            end = self.timestamps[-1]
        elif isinstance(end, str):
            end = mkdate(end)
        return start, end

    def tstamps_for_daterange(
        self,
        start: Union[datetime.datetime, str],
        end: Union[datetime.datetime, str],
    ) -> List[datetime.datetime]:
        """
        Timestamps available within the given date range.

        Parameters
        ----------
        start: datetime, np.datetime64 or str
            start of date range
        end: datetime, np.datetime64 or str
            end of date range

        Returns
        -------
        timestamps : array_like
            Array of datetime.datetime timestamps of available images in the
            date range.
        """
        # evaluate the input to obtain the correct format
        start, end = self._validate_start_end(start, end)
        tstamps = list(filter(lambda t: start <= t <= end, self.timestamps))

        return tstamps

    @property
    def timestamps(self):
        return self._timestamps

    @abstractmethod
    def _read_block(
        self, start: datetime.datetime, end: datetime.datetime
    ) -> Dict[
        str, Union[np.ndarray, dask.array.core.Array]
    ]:  # pragma: no cover
        """
        Reads multiple images of a dataset as a numpy/dask array.

        Parameters
        ----------
        start, end : datetime.datetime

        Returns
        -------
        block : np.ndarray or dask.array.core.Array
            Block of data (data cube) with dimension order:
            - time, lat, lon for data on a regular 2D grid
            - time, y, x for data on a curvilinear 2D grid
            - time, loc for unstructured data
        """
        ...

    def read_block(
        self,
        start: Union[datetime.datetime, str] = None,
        end: Union[datetime.datetime, str] = None,
        _apply_landmask_bbox=True,
    ) -> xr.Dataset:
        """
        Reads a block of the image stack.

        Parameters
        ----------
        start : datetime.datetime or str, optional
            If not given, start at first timestamp in dataset.
        end : datetime.datetime or str, optional
            If not given, end at last timestamp in dataset.
        _apply_landmask_bbox : bool, optional
            For internal use only. Whether to apply the landmask and bounding
            box. Should be always set to True, except when calling from within
            `read`, because selection is then made based on the full grid.

        Returns
        -------
        block : xr.Dataset
            A block of the dataset. In case of a regular grid, this will have
            ``self.latname`` and ``self.lonname`` as dimensions.
        """
        start, end = self._validate_start_end(start, end)
        times = self.tstamps_for_daterange(start, end)
        block_dict = self._read_block(start, end)

        # we might need to apply the landmask
        if self.landmask is not None and _apply_landmask_bbox:
            mask = np.broadcast_to(
                ~self.landmask.values, [len(times)] + list(self.landmask.shape)
            )
            for var in block_dict:
                if isinstance(block_dict[var], np.ndarray):
                    masked_array = np.ma.masked_array
                elif isinstance(block_dict[var], da.core.Array):
                    masked_array = da.ma.masked_array
                else:
                    raise ReaderError("Unknown array type in read_block.")
                block_dict[var] = masked_array(block_dict[var], mask=mask)

        # Now we have to set the coordinates/dimensions correctly.  This works
        # differently depending on how the original data is structured:
        coords = {}
        coords[self.timename] = times
        if self._has_regular_grid:
            # we can simply wrap the data with time, lat, and lon
            coords[self.latname] = self.lat
            coords[self.lonname] = self.lon
            dims = (self.timename, self.latname, self.lonname)
        elif self.curvilinear:
            coords[self.latname] = ([self.latdim, self.londim], self.lat)
            coords[self.lonname] = ([self.latdim, self.londim], self.lon)
            dims = (self.timename, self.latdim, self.londim)
        else:  # unstructured grid
            coords[self.latname] = (self.locdim, self.lat)
            coords[self.lonname] = (self.locdim, self.lon)
            dims = (self.timename, self.locdim)

        arrays = {
            name: (dims, data, self.array_attrs[name])
            for name, data in block_dict.items()
        }
        block = xr.Dataset(arrays, coords=coords, attrs=self.global_attrs)

        # bounding box is applied after assigning the coordinates
        if self.bbox is not None and _apply_landmask_bbox:
            lonmin, latmin, lonmax, latmax = (*self.bbox,)
            block = block.where(
                (
                    (latmin <= block[self.latname])
                    & (block[self.latname] <= latmax)
                    & (
                        (lonmin <= block[self.lonname])
                        & (block[self.lonname] <= lonmax)
                    )
                ),
                drop=True,
            )
        return block

    def read(
        self, timestamp: Union[datetime.datetime, str], **kwargs
    ) -> Image:
        """
        Read a single image at a given timestamp. Raises `ReaderError` if
        timestamp is not available in the dataset.

        Parameters
        ----------
        timestamp : datetime.datetime or str
            Timestamp of image of interest

        Returns
        -------
        img_dict : dict
            Dictionary containing the image data as numpy array, using the
            parameter name as key.

        Raises
        ------
        KeyError
        """
        if isinstance(timestamp, str):
            timestamp = mkdate(timestamp)

        if timestamp not in self.timestamps:
            raise ReaderError(
                f"Timestamp {timestamp} is not available in the dataset!"
            )

        img = self.read_block(
            timestamp, timestamp, _apply_landmask_bbox=False
        ).isel({self.timename: 0})
        if self._has_2d_grid:
            latname = self.latname if not self.curvilinear else self.latdim
            lonname = self.lonname if not self.curvilinear else self.londim
            img = img.stack(dimensions={"loc": (latname, lonname)})
        data = {
            varname: img[varname].values[self.grid.activegpis]
            for varname in self.varnames
        }
        metadata = self.array_attrs
        return Image(
            self.grid.arrlon, self.grid.arrlat, data, metadata, timestamp
        )


class DirectoryImageReader(XarrayImageReaderBase):
    r"""
    Image reader for a directory containing netcdf files.

    This works for any datasets which are stored as single image files within a
    directory (and its subdirectories).

    Parameters
    ----------
    directory : str or Path
        Directory in which the netcdf files are located. Any file matching
        `pattern` within this directory or any subdirectories is used.
    varnames : str or list of str
        Names of the variables that should be read. If `rename` is used, this
        should be the new names.
    level : dict, optional
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
        nicer ones.
    fmt : str, optional
        Format string to deduce timestamp from filename (without directory
        name). If it is ``None`` (default), the timestamps will be obtained
        from the files (which requires opening all files and is therefore less
        efficient).
        This must not contain any wildcards, only the format specifiers
        from ``datetime.datetime.strptime`` (e.g. %Y for year, %m for month, %d
        for day, %H for hours, %M for minutes, ...).
        If such a simple pattern does not work for you, you can additionally
        specify `time_regex_pattern` (see below).
        `fmt` should only be used if the files only contain a single image.
    pattern : str, optional
        Glob pattern to find all files to use, default is "*.nc".
    time_regex_pattern : str, optional
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
    latname : str, optional
        If `locdim` is given (i.e. for non-rectangular grids), this must be the
        name of the latitude data variable, otherwise must be the name of the
        latitude coordinate. Default is "lat".
    lonname : str, optional
        If `locdim` is given (i.e. for non-rectangular grids), this must be the
        name of the longitude data variable, otherwise must be the name of the
        longitude coordinate. Default is "lon"
    latdim : str, optional
        The name of the latitude dimension in case it's not the same as the
        latitude coordinate variable.
    londim : str, optional
        The name of the longitude dimension in case it's not the same as the
        longitude coordinate variable.
    locdim : str, optional
        The name of the location dimension for non-rectangular grids. If this
        is given, you *MUST* provide `lonname` and `latname`.
    lat : tuple, optional
        If the latitude can not be inferred from the dataset you can specify it
        by giving a start and stepsize. This is only used if `latdim` is
        given.
    lon : tuple, optional
        If the longitude can not be inferred from the dataset you can specify
        it by giving a start and stepsize. This is only used if `londim` is
        given.
    curvilinear : bool, optional
        Whether the grid is curvilinear, i.e. is a 2D grid, but not a regular
        lat-lon grid. In this case, `latname` and `lonname` must be given, and
        must be names of the variables containing the 2D latitude and longitude
        values. Additionally, `latdim` and `londim` must be given and will be
        interpreted as vertical and horizontal dimension.
        Default is False.
    timename : str, optional
        The name of the time coordinate, default is "time".
    average: str, optional
        If specified, average multiple images. Currently only "daily" is
        implemented. Default is None.
    landmask : xr.DataArray or str, optional
        A land mask to be applied to reduce storage size. This can either be a
        xr.DataArray of the same shape as the dataset images with ``False`` at
        non-land points, or a string.
        If it is a string, it can either be the name of a variable that is also
        in the dataset, or it can follow the pattern
        "<filename>:<variable_name>". In the latter case, the part before the
        colon is interpreted as path to a netCDF file, the part after the colon
        as the variable name of the landmask within this file.
    bbox : Iterable, optional
        (lonmin, latmin, lonmax, latmax) of a bounding box.
    cellsize : float, optional
        Spatial coverage of a single cell file in degrees. Default is ``None``.
    rename : dict, optional
        Dictionary to use to rename variables in the file. This is applied
        after level selection, so all other parameters referring to variable
        names except 'level' should use the new names.
    use_dask : bool, optional
        Whether to open image files using dask. This might be useful in case
        you run into memory issues.
    discard_attrs : bool, optional
        Whether to discard the attributes of the input netCDF files (reduced
        data size).
    """

    def __init__(
        self,
        directory: Union[Path, str],
        varnames: Union[str, Sequence],
        level: dict = None,
        fmt: str = None,
        pattern: str = "*.nc",
        time_regex_pattern: str = None,
        timename: str = "time",
        latname: str = None,
        lonname: str = None,
        latdim: str = None,
        londim: str = None,
        locdim: str = None,
        lat: tuple = None,
        lon: tuple = None,
        average: str = None,
        landmask: xr.DataArray = None,
        bbox: Iterable = None,
        cellsize: float = None,
        rename: dict = None,
        use_dask: bool = False,
        cache: bool = False,
        curvilinear: bool = False,
        discard_attrs: bool = False,
        transpose: tuple = None,
        use_tqdm: bool = False,
        skip_missing: bool = False,
    ):
        if use_dask:
            self.chunks = -1
        else:
            self.chunks = None
        self.cache = cache
        self.average = average
        self.transpose = transpose
        self.use_tqdm = use_tqdm

        # first, we walk over the whole directory subtree and find any files
        # that match our pattern
        directory = Path(directory)
        filepaths = self._get_filepaths(directory, pattern)
        ds = self._get_example_dataset(filepaths)

        if skip_missing:
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
                        f"Skipping variable {l2name} because it does not exist"
                    )
            varnames = new_varnames

        # now we can call the parent constructor using the dataset from the
        # first file
        super().__init__(
            ds,
            varnames,
            level=level,
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
            rename=rename,
        )

        # file_tstamp_map maps a single file to 1 or many timestamps available
        # in this file
        self._file_tstamp_map = self._create_filepath_tstamp_map(
            fmt, time_regex_pattern, filepaths
        )

        # tstamp_file_map maps each timestamp to the file where it can be found
        self.tstamp_file_map = {}
        for fname, tstamps in self._file_tstamp_map.items():
            for t in tstamps:
                self.tstamp_file_map[t] = fname
        self._available_timestamps = sorted(list(self.tstamp_file_map))

        # output_tstamp_map maps a single output timestamp to 1 or many
        # timestamps available in this directory/dataset
        if self.average is not None:
            self._output_tstamp_map = self._create_output_tstamp_map()
            self._timestamps = sorted(list(self._output_tstamp_map))
        else:
            self._timestamps = self._available_timestamps

        if discard_attrs:
            self.discard_attrs()

    def _get_filepaths(self, directory, pattern):
        filepaths = {}
        for fpath in sorted(
            glob.glob(str(directory / f"**/{pattern}"), recursive=True)
        ):
            fname = Path(fpath).name
            filepaths[fname] = fpath

        if not filepaths:  # pragma: no cover
            raise ReaderError(
                f"No files matching pattern {pattern} in directory "
                f"{str(directory)}"
            )
        return filepaths

    def _get_example_dataset(self, filepaths):
        # We need to read the first file so that the parent constructor can
        # deduce the grid from it.
        fname = next(iter(filepaths.values()))
        logging.info(f"Opening {str(fname)} to infer metadata.")
        ds = self._open_dataset(fname)
        return ds

    def _open_dataset(self, fname):
        ds = xr.open_dataset(fname, chunks=self.chunks, cache=self.cache)
        return ds

    def _create_filepath_tstamp_map(self, fmt, time_regex_pattern, filepaths):
        filepath_tstamp_map = {}
        if fmt is not None:
            # the fastest way is to deduce the timestamps from the list
            # of filenames
            if time_regex_pattern is not None:
                time_pattern = re.compile(time_regex_pattern)
            for fname, path in filepaths.items():
                if time_regex_pattern is not None:
                    match = time_pattern.findall(fname)
                    if not match:  # pragma: no cover
                        raise ReaderError(
                            f"Pattern {time_regex_pattern} did not match "
                            f"{fname}"
                        )
                    timestring = match[0]
                else:
                    timestring = fname
                tstamp = datetime.datetime.strptime(timestring, fmt)
                filepath_tstamp_map[path] = [tstamp]
        else:
            for _, path in filepaths.items():
                ds = xr.open_dataset(path)
                if self.timename not in ds.indexes:  # pragma: no cover
                    raise ReaderError(
                        f"Time dimension {self.timename} does not exist in "
                        f"{str(path)}"
                    )
                time = ds.indexes[self.timename]
                tstamps = [t.to_pydatetime() for t in time]
                filepath_tstamp_map[path] = tstamps
        return filepath_tstamp_map

    def _create_output_tstamp_map(self):
        output_tstamp_map = {}
        for tstamp in self._available_timestamps:
            output_tstamp = self._calculate_averaging_timestamp(tstamp)
            if output_tstamp in output_tstamp_map:
                output_tstamp_map[output_tstamp].append(tstamp)
            else:
                output_tstamp_map[output_tstamp] = [tstamp]
        return output_tstamp_map

    def _calculate_averaging_timestamp(self, tstamp):
        if self.average == "daily":
            date = tstamp.date()
            return datetime.datetime(date.year, date.month, date.day)
        else:
            raise NotImplementedError("only average='daily' is implemented")

    def discard_attrs(self):
        self.global_attrs = {}
        self.array_attrs = {v: {} for v in self.varnames}

    @property
    def timestamps(self):
        return self._timestamps

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
            block_dict = self._read_all_files(times)
        else:
            # If we have to average multiple images to a single image, we will
            # read image by image
            block_dict = {varname: [] for varname in self.varnames}
            for tstamp in times:
                # read all sub-images that have to be averaged later on
                times_to_read = self._output_tstamp_map[tstamp]
                tmp_block_dict = self._read_all_files(times_to_read)
                for varname in self.varnames:
                    block_dict[varname].append(
                        np.mean(tmp_block_dict[varname], axis=0)
                    )
            # now we just have to convert the lists of arrays to array stacks
            for varname in self.varnames:
                block_dict[varname] = np.stack(block_dict[varname], axis=0)
        return block_dict

    def _read_all_files(self, times):
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
        if self.use_tqdm:
            iterator = tqdm(iterator)
        for fname, tstamps in iterator:
            ds = self._open_dataset(fname)
            if self.transpose is not None:
                ds = ds.transpose(*self.transpose)
            ds = self._select_levels(ds)
            if self.rename is not None:
                ds = ds.rename(self.rename)
            for varname in self.varnames:
                arr = ds[varname]
                if self.timename in arr.dims:
                    arr = arr.sel({self.timename: tstamps})
                    block_dict[varname].append(arr.data)
                else:
                    block_dict[varname].append(arr.data[np.newaxis, ...])
        for varname in self.varnames:
            block_dict[varname] = np.vstack(block_dict[varname])
        return block_dict


class XarrayImageStackReader(XarrayImageReaderBase):
    """
    Image reader that wraps a xarray.Dataset.

    This can be used as a generic image reader for netcdf stacks, e.g. for
    reformatting the data to timeseries format using the package ``repurpose``
    (which is implemented in ``nc_image_reader.reshuffle`` and can also be done
    using the supplied script ``repurpose-ncstack``.).

    Parameters
    ----------
    ds : xr.Dataset, Path or str
        Xarray dataset (or filename of a netCDF file). Must have a time
        coordinate and either `latname`/`latdim` and `lonname`/`latdim` (for a
        regular latitude-longitude grid) or `locdim` as additional
        coordinates/dimensions.
    varnames : str or list of str
        Names of the variable that should be read.
    level : dict, optional
        If a variable has more dimensions than latitude, longitude, time (or
        location, time), e.g. a level dimension, a single value for each
        remaining dimension must be chosen. They can be passed here as
        dictionary mapping dimension name to integer index (this will then be
        passed to ``xr.DataArray.isel``) for each variable. E.g., if you have
        two variables "X" and "Y", and "Y" has a level dimension, you would
        pass ``{"Y": {"level": 2}}``.
        In case you only want read a single variable, you can also pass the
        dictionary directly, e.g. ``{"level": 2}``.
    timename : str, optional
        The name of the time coordinate, default is "time".
    latname : str, optional
        If `locdim` is given (i.e. for non-rectangular grids), this must be the
        name of the latitude data variable, otherwise must be the name of the
        latitude coordinate. Default is "lat".
    lonname : str, optional
        If `locdim` is given (i.e. for non-rectangular grids), this must be the
        name of the longitude data variable, otherwise must be the name of the
        longitude coordinate. Default is "lon"
    latdim : str, optional
        The name of the latitude dimension in case it's not the same as the
        latitude coordinate variable.
    londim : str, optional
        The name of the longitude dimension in case it's not the same as the
        longitude coordinate variable.
    locdim : str, optional
        The name of the location dimension for non-rectangular grids. If this
        is given, you *MUST* provide `lonname` and `latname`.
    lat : tuple, optional
        If the latitude can not be inferred from the dataset you can specify it
        by giving a start and stepsize. This is only used if `latdim` is
        given.
    lon : tupl, optional
        If the longitude can not be inferred from the dataset you can specify
        it by giving a start and stepsize. This is only used if `londim` is
        given.
    curvilinear : bool, optional
        Whether the grid is curvilinear, i.e. is a 2D grid, but not a regular
        lat-lon grid. In this case, `latname` and `lonname` must be given, and
        must be names of the variables containing the 2D latitude and longitude
        values. Additionally, `latdim` and `londim` must be given and will be
        interpreted as vertical and horizontal dimension.
        Default is False.
    landmask : xr.DataArray or str, optional
        A land mask to be applied to reduce storage size. This can either be a
        xr.DataArray of the same shape as the dataset images with ``False`` at
        non-land points, or a string.
        If it is a string, it can either be the name of a variable that is also
        in the dataset, or it can follow the pattern
        "<filename>:<variable_name>". In the latter case, the part before the
        colon is interpreted as path to a netCDF file, the part after the colon
        as the variable name of the landmask within this file.
    bbox : Iterable, optional
        (lonmin, latmin, lonmax, latmax) of a bounding box.
    cellsize : float, optional
        Spatial coverage of a single cell file in degrees. Default is ``None``.
    use_dask: bool, optional
        Whether to open image files using dask. This might be useful in case
        you run into memory issues. Only used in case `ds` is only a pathname.
    """

    def __init__(
        self,
        ds: xr.Dataset,
        varnames: str,
        level: dict = None,
        timename: str = "time",
        latname: str = None,
        lonname: str = None,
        latdim: str = None,
        londim: str = None,
        locdim: str = None,
        lat: tuple = None,
        lon: tuple = None,
        landmask: xr.DataArray = None,
        bbox: Iterable = None,
        cellsize: float = None,
        use_dask: bool = False,
        curvilinear: bool = False,
    ):

        if isinstance(ds, (str, Path)):
            if use_dask:
                self.chunks = "auto"
            else:
                self.chunks = None
            ds = xr.open_dataset(ds, chunks=self.chunks)
        super().__init__(
            ds,
            varnames,
            level=level,
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
        )
        self.data = self._select_levels(ds)[self.varnames]
        self._timestamps = ds.indexes[self.timename].to_pydatetime()

    def _read_block(
        self, start: datetime.datetime, end: datetime.datetime
    ) -> xr.Dataset:
        block = self.data.sel({self.timename: slice(start, end)})
        block_dict = {var: block[var].data for var in self.varnames}
        return block_dict


class GriddedNcOrthoMultiTs(_GriddedNcOrthoMultiTs):
    def __init__(
        self,
        ts_path,
        grid_path=None,
        time_offset_name=None,
        time_offset_unit="S",
        **kwargs,
    ):
        """
        Class for reading time series in pynetcf format after reshuffling.

        Parameters
        ----------
        ts_path : str
            Directory where the netcdf time series files are stored
        grid_path : str, optional
            Path to grid file, that is used to organize the location of time
            series to read. If None is passed, grid.nc is searched for in the
            ts_path.
        time_offset_name : str, optional
            Name of the variable containing time offsets for a given location
            and time that is added to the timestamp. If given,
            `time_offset_units` must also be given. Default is None.
        time_offset_unit : str, optional
            Unit of the time offset. Default is "S" for seconds. Have a look at
            `pd.to_timedelta` for possible units.

        Additional keyword arguments
        ----------------------------
        parameters : list, optional (default: None)
            Specific variable names to read, if None are selected, all are
            read.
        offsets : dict, optional (default:None)
            Offsets (values) that are added to the parameters (keys)
        scale_factors : dict, optional (default:None)
            Offset (value) that the parameters (key) is multiplied with
        ioclass_kws: dict

        Optional keyword arguments to pass to OrthoMultiTs class:
        ---------------------------------------------------------
        read_bulk : boolean, optional (default:False)
            if set to True the data of all locations is read into memory,
            and subsequent calls to read_ts read from the cache and not from
            disk this makes reading complete files faster#
        read_dates : boolean, optional (default:False)
            if false dates will not be read automatically but only on specific
            request useable for bulk reading because currently the netCDF
            num2date routine is very slow for big datasets
        """
        if grid_path is None:  # pragma: no branch
            grid_path = os.path.join(ts_path, "grid.nc")
        grid = load_grid(grid_path)
        super().__init__(ts_path, grid, **kwargs)
        self.time_offset_name = time_offset_name
        self.time_offset_unit = time_offset_unit

    def read(self, *args, **kwargs) -> pd.DataFrame:
        df = super().read(*args, **kwargs)
        if self.time_offset_name is not None:
            delta = pd.to_timedelta(
                df[self.time_offset_name].values, unit=self.time_offset_unit
            )
            df.index = df.index + delta
            df.drop(self.time_offset_name, axis="columns", inplace=True)
        return df


class XarrayTSReader(XarrayReaderBase):
    """
    Wrapper for xarray.Dataset when timeseries of the data should be read.

    This is useful if you are using functions from the TUW-GEO package universe
    which require a timeseries reader, but you don't have the data in the
    pynetcf timeseries format.

    Since this is reading along the time dimension, you should make sure that
    the time dimension is either the last dimension in your netcdf (the fastest
    changing dimension), or that it is chunked in a way that makes timeseries
    access fast. To move the time dimension last, you can use the function
    ``nc_image_reader.transpose.write_transposed_dataset`` or programs like
    ``ncpdq``.


    Parameters
    ----------
    ds : xr.Dataset, Path or str
        Xarray dataset (or filename of a netCDF file). Must have a time
        coordinate and either `latname`/`latdim` and `lonname`/`latdim` (for a
        regular latitude-longitude grid) or `locdim` as additional
        coordinates/dimensions.
    varnames : str
        Names of the variable that should be read.
    level : dict, optional
        If a variable has more dimensions than latitude, longitude, time (or
        location, time), e.g. a level dimension, a single value for each
        remaining dimension must be chosen. They can be passed here as
        dictionary mapping dimension name to integer index (this will then be
        passed to ``xr.DataArray.isel``) for each variable. E.g., if you have
        two variables "X" and "Y", and "Y" has a level dimension, you would
        pass ``{"Y": {"level": 2}}``.
        In case you only want read a single variable, you can also pass the
        dictionary directly, e.g. ``{"level": 2}``.
    timename : str, optional
        The name of the time coordinate, default is "time".
    latname : str, optional
        If `locdim` is given (i.e. for non-rectangular grids), this must be the
        name of the latitude data variable, otherwise must be the name of the
        latitude coordinate. Default is "lat".
    lonname : str, optional
        If `locdim` is given (i.e. for non-rectangular grids), this must be the
        name of the longitude data variable, otherwise must be the name of the
        longitude coordinate. Default is "lon"
    latdim : str, optional
        The name of the latitude dimension in case it's not the same as the
        latitude coordinate variable.
    londim : str, optional
        The name of the longitude dimension in case it's not the same as the
        longitude coordinate variable.
    locdim : str, optional
        The name of the location dimension for non-rectangular grids. If this
        is given, you *MUST* provide `lonname` and `latname`.
    lat : tuple, optional
        If the latitude can not be inferred from the dataset you can specify it
        by giving a start and stepsize. This is only used if `latdim` is
        given.
    lon : tupl, optional
        If the longitude can not be inferred from the dataset you can specify
        it by giving a start and stepsize. This is only used if `londim` is
        given.
    landmask : xr.DataArray, optional
        A land mask to be applied to reduce storage size.
    bbox : Iterable, optional
        (lonmin, latmin, lonmax, latmax) of a bounding box.
    cellsize : float, optional
        Spatial coverage of a single cell file in degrees. Default is ``None``.
    """

    def __init__(
        self,
        ds: xr.Dataset,
        varnames: Union[str, Sequence],
        level: dict = None,
        timename: str = "time",
        latname: str = "lat",
        lonname: str = "lon",
        latdim: str = None,
        londim: str = None,
        locdim: str = None,
        lat: tuple = None,
        lon: tuple = None,
        landmask: xr.DataArray = None,
        bbox: Iterable = None,
        cellsize: float = None,
    ):
        if isinstance(varnames, str):
            varnames = [varnames]
        varnames = list(varnames)
        if isinstance(ds, (str, Path)):
            ds = xr.open_dataset(ds)

        # rechunk to good chunks for reading
        _latdim = latdim if latdim is not None else latname
        _londim = londim if londim is not None else lonname
        # with this construct I make sure that I select lat, lon, and time
        # in the right order
        ds_dims = dict(ds.dims)
        img_dims = ds[varnames[0]].dims
        dims = {}
        for dim in img_dims:
            if dim in [_latdim, _londim, timename]:
                dims[dim] = ds_dims[dim]
        shape = tuple(dims.values())
        chunks = infer_chunks(shape, 100, np.float32)
        ds = ds.chunk(dict(zip(list(dims), chunks)))

        if list(dims)[-1] != timename:
            warnings.warn("Time should be the last dimension!")

        super().__init__(
            ds,
            varnames,
            level=level,
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
        )

        if self._has_2d_grid:
            # we have to reshape the data
            latdim = self.latdim if self.latdim is not None else self.latname
            londim = self.londim if self.londim is not None else self.lonname
            self.orig_data = self._select_levels(ds)[self.varnames]
            self.data = self.orig_data.stack({"loc": (latdim, londim)})
            self.locdim = "loc"
        else:
            self.data = self._select_levels(ds)[self.varnames]

    def read(self, *args, **kwargs) -> pd.Series:
        """
        Reads variable timeseries from dataset.

        Parameters
        ----------
        args : tuple
            If a single argument, must be an integer denoting the grid point
            index at which to read the timeseries. If two arguments, it's
            longitude and latitude values at which to read the timeseries.

        Returns
        -------
        df : pd.DataFrame
        """
        if len(args) == 1:
            gpi = args[0]
            if self._has_2d_grid:
                lon, lat = self.grid.gpi2lonlat(gpi)

        elif len(args) == 2:
            lon = args[0]
            lat = args[1]
            if not self._has_2d_grid:
                gpi = self.grid.find_nearest_gpi(lon, lat)[0]
                if not isinstance(gpi, np.integer):  # pragma: no cover
                    raise ValueError(
                        f"No gpi near (lon={lon}, lat={lat}) found"
                    )
        else:  # pragma: no cover
            raise ValueError(
                f"args must have length 1 or 2, but has length {len(args)}"
            )

        if self._has_2d_grid:
            data = self.orig_data.sel(lat=lat, lon=lon)
        else:
            data = self.data[{self.locdim: gpi}]
        df = data.to_pandas()[self.varnames]
        return df
