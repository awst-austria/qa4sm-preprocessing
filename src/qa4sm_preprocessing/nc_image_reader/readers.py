"""
Timeseries and image readers for wrapping xarray.Datasets, compatible with
readers from the TUW-GEO python package universe (e.g. pygeobase, pynetcf).
"""

from abc import abstractmethod
import datetime
import fnmatch
import logging
import numpy as np
import os
import glob
import pandas as pd
from pathlib import Path
import re
from typing import Union, Iterable, List, Tuple, Sequence
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
    Base class for readers that either read a full netcdf stack or single files
    with xarray.

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
        if locdim is not None and (
            latname is None or lonname is None
        ):  # pragma: no cover
            raise ReaderError(
                "If locdim is given, latname and lonname must also be given."
            )
        self._lat = lat
        self._lon = lon
        self._has_regular_grid = locdim is None
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

        self.grid = self._grid_from_xarray(ds)
        (
            self.dataset_metadata,
            self.array_metadata,
        ) = self._metadata_from_xarray(ds)

    def _metadata_from_xarray(self, ds: xr.Dataset) -> Tuple[dict, dict]:
        dataset_metadata = dict(ds.attrs)
        array_metadata = {v: dict(ds[v].attrs) for v in self.varnames}
        return dataset_metadata, array_metadata

    def _grid_from_xarray(self, ds: xr.Dataset) -> CellGrid:

        # if using regular lat-lon grid, we can use gridfromdims
        self.lat = self._get_coord(ds, "lat")
        self.lon = self._get_coord(ds, "lon")
        if self._has_regular_grid:
            grid = gridfromdims(self.lon, self.lat)
        else:
            grid = BasicGrid(self.lon, self.lat)

        if hasattr(self, "landmask") and self.landmask is not None:
            if self._has_regular_grid:
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
                lonmin=lonmin,
                latmin=latmin,
                lonmax=lonmax,
                latmax=latmax,
            )
            grid = grid.subgrid_from_gpis(bbox_gpis)
        num_gpis = len(grid.activegpis)
        logging.info(f"_grid_from_xarray: Number of active gpis: {num_gpis}")

        if hasattr(self, "cellsize") and self.cellsize is not None:
            grid = grid.to_cell_grid(cellsize=self.cellsize)
            num_cells = len(grid.get_cells())
            logging.info(
                f"_grid_from_xarray: Number of grid cells: {num_cells}"
            )

        return grid

    def _get_coord(self, ds: xr.Dataset, coordname: str) -> xr.DataArray:
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
        if _coord is not None:
            start, step = _coord
            coord = np.array([start + i * step for i in range(dimlen)])
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
            coord,
            coords={cname: coord},
            dims=[cname],
            name=cname,
        )

    def _select_vars_levels(self, ds):
        ds = ds[self.varnames]
        if self.level is not None:
            for varname in self.level:
                ds[varname] = ds[varname].isel(self.level[varname])
        return ds


class XarrayImageReaderMixin:
    """
    Provides the methods
    - self.read
    - self.tstamps_for_daterange
    - self._grid_from_xarray
    - self.read_block

    and therefore meets all prerequisites for Img2Ts.

    Child classes must override `_read_image` and need to set self.timestamps.
    """

    @abstractmethod
    def _read_image(
        self, timestamp: datetime.datetime
    ) -> xr.Dataset:  # pragma: no cover
        """
        Returns a single image for the given timestamp
        """
        ...

    def read(
        self, timestamp: Union[datetime.datetime, str], **kwargs
    ) -> Image:
        """
        Read a single image at a given timestamp. Raises `KeyError` if
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

        if timestamp in self.timestamps:
            images = {}
            img = self._select_vars_levels(self._read_image(timestamp))
            # check if dimensions are as expected and potentially select from
            # non lat/lon/time dimensions
            latdim = self.latdim if self.latdim is not None else self.latname
            londim = self.londim if self.londim is not None else self.lonname
            expected_dims = [latdim, londim, self.timename]
            for d in img.dims:
                if d not in expected_dims:  # pragma: no cover
                    raise ReaderError(
                        f"Unexpected dimension {d} in image for {timestamp}."
                    )
            if self._has_regular_grid:
                img = img.stack(dimensions={"loc": (latdim, londim)})
            data = {
                varname: img[varname].values[self.grid.activegpis]
                for varname in self.varnames
            }
            metadata = self.array_metadata
            return Image(
                self.grid.arrlon, self.grid.arrlat, data, metadata, timestamp
            )
        else:  # pragma: no cover
            raise ReaderError(
                f"Timestamp {timestamp} is not available in the dataset!"
            )

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
            Array of datetime timestamps of available images in the date
            range.
        """
        if start is None:
            start = self.timestamps[0]
        elif isinstance(start, str):
            start = mkdate(start)
        if end is None:
            end = self.timestamps[-1]
        elif isinstance(end, str):
            end = mkdate(end)
        return list(filter(lambda t: t >= start and t <= end, self.timestamps))

    def _read_block(self, timestamps: Iterable) -> xr.Dataset:
        imgs = []
        for tstamp in timestamps:
            img = self._select_vars_levels(self._read_image(tstamp))
            imgs.append(img)
        block = xr.concat(imgs, dim=self.timename).assign_coords(
            {self.timename: timestamps}
        )
        for varname in self.array_metadata:
            block[varname].attrs.update(self.array_metadata[varname])
        return block

    def read_block(
        self,
        start: Union[datetime.datetime, str] = None,
        end: Union[datetime.datetime, str] = None,
    ) -> xr.Dataset:
        """
        Reads a block of the image stack.

        Parameters
        ----------
        start : datetime.datetime or str, optional
            If not given, start at first timestamp in dataset.
        end : datetime.datetime or str, optional
            If not given, end at last timestamp in dataset.

        Returns
        -------
        block : xr.Dataset
            A block of the dataset. In case of a regular grid, this will have
            ``self.latname`` and ``self.lonname`` as dimensions.
        """
        timestamps = self.tstamps_for_daterange(start, end)
        block = self._read_block(timestamps)
        # we might need to apply the landmask, this is applied before renaming
        # the coordinates, because it is in the original coordinates
        if self.landmask is not None:
            block = block.where(self.landmask)

        if self.latdim is not None:
            block = block.rename({self.latdim: self.latname}).assign_coords(
                {self.latname: self.lat.values}
            )
        if self.londim is not None:
            block = block.rename({self.londim: self.lonname}).assign_coords(
                {self.lonname: self.lon.values}
            )

        # bounding box is applied after assigning the coordinates
        if self.bbox is not None:
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


class DirectoryImageReader(XarrayReaderBase, XarrayImageReaderMixin):
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
    lon : tupl, optional
        If the longitude can not be inferred from the dataset you can specify
        it by giving a start and stepsize. This is only used if `londim` is
        given.
    timename : str, optional
        The name of the time coordinate, default is "time".
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
        before anything else, so all other parameters referring to variable
        names should use the new names.
    use_dask: bool, optional
        Whether to open image files using dask. This might be useful in case
        you run into memory issues.
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
        landmask: xr.DataArray = None,
        bbox: Iterable = None,
        cellsize: float = None,
        rename: dict = None,
        use_dask: bool = False,
    ):

        # first, we walk over the whole directory subtree and find any files
        # that match our pattern
        directory = Path(directory)
        filepaths = {}
        for fpath in glob.glob(str(Path(directory) / f"**/{pattern}"), recursive=True):
            fname = Path(fpath).name
            filepaths[fname] = fpath

        if not filepaths:  # pragma: no cover
            raise ReaderError(
                f"No files matching pattern {pattern} in directory "
                f"{str(directory)}"
            )

        # We need to read the first file so that the parent constructor can
        # deduce the grid from it.
        ds = xr.open_dataset(next(iter(filepaths.values())))
        self.rename = rename
        if self.rename is not None:
            ds = ds.rename(self.rename)

        if use_dask:
            self.chunks = -1
        else:
            self.chunks = None

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
        )

        # if possible, deduce the timestamps from the filenames and create a
        # dictionary mapping timestamps to file paths
        self.filepaths = {}
        if fmt is not None:
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
                self.filepaths[tstamp] = path
        else:
            for _, path in filepaths.items():
                ds = xr.open_dataset(path)
                if timename not in ds.indexes:  # pragma: no cover
                    raise ReaderError(
                        f"Time dimension {timename} does not exist in "
                        f"{str(path)}"
                    )
                time = ds.indexes[timename]
                if len(time) != 1:  # pragma: no cover
                    raise ReaderError(
                        f"Expected only a single timestamp, found {str(time)} "
                        f" in {str(path)}"
                    )
                tstamp = time[0].to_pydatetime()
                self.filepaths[tstamp] = path
        # sort the timestamps according to date, because we might have to
        # return them sorted
        self.timestamps = sorted(list(self.filepaths))

    def _read_image(self, timestamp):
        ds = xr.open_dataset(self.filepaths[timestamp], chunks=self.chunks)
        if self.rename is not None:
            ds = ds.rename(self.rename)
        return ds


class XarrayImageReader(XarrayReaderBase, XarrayImageReaderMixin):
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
    ):

        if isinstance(ds, (str, Path)):
            if use_dask:
                chunks = "auto"
            else:
                chunks = None
            ds = xr.open_dataset(ds, chunks=chunks)
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
        self.data = ds
        self.timestamps = ds.indexes[self.timename].to_pydatetime()

    def _read_image(self, timestamp):
        return self.data.sel({self.timename: timestamp})

    def _read_block(self, timestamps: Iterable) -> xr.DataArray:
        return self._select_vars_levels(
            self.data.sel({self.timename: timestamps})
        )


class GriddedNcOrthoMultiTs(_GriddedNcOrthoMultiTs):
    def __init__(self, ts_path, grid_path=None, **kwargs):
        """
        Class for reading time series after reshuffling.

        Parameters
        ----------
        ts_path : str
            Directory where the netcdf time series files are stored
        grid_path : str, optional (default: None)
            Path to grid file, that is used to organize the location of time
            series to read. If None is passed, grid.nc is searched for in the
            ts_path.

        Optional keyword arguments that are passed to the Gridded Base:
        ---------------------------------------------------------------
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
    ``nc_image_reader.transpose.create_transposed_netcdf`` or programs like
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

        if self._has_regular_grid:
            # we have to reshape the data
            latdim = self.latdim if self.latdim is not None else self.latname
            londim = self.londim if self.londim is not None else self.lonname
            self.orig_data = self._select_vars_levels(ds)
            self.data = self.orig_data.stack({"loc": (latdim, londim)})
            self.locdim = "loc"
        else:
            self.data = self._select_vars_levels(ds)

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
            if self._has_regular_grid:
                lon, lat = self.grid.gpi2lonlat(gpi)

        elif len(args) == 2:
            lon = args[0]
            lat = args[1]
            if not self._has_regular_grid:
                gpi = self.grid.find_nearest_gpi(lon, lat)[0]
                if not isinstance(gpi, np.integer):  # pragma: no cover
                    raise ValueError(
                        f"No gpi near (lon={lon}, lat={lat}) found"
                    )
        else:  # pragma: no cover
            raise ValueError(
                f"args must have length 1 or 2, but has length {len(args)}"
            )

        if self._has_regular_grid:
            data = self.orig_data.sel(lat=lat, lon=lon)
        else:
            data = self.data[{self.locdim: gpi}]
        df = data.to_pandas()[self.varnames]
        return df
