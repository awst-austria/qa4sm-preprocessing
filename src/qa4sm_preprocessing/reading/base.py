from abc import abstractmethod
import dask
import dask.array as da
import datetime
import logging
import numpy as np
from typing import Union, Iterable, List, Tuple, Sequence, Dict
import warnings
import xarray as xr

from pygeobase.object_base import Image
from pygeogrids.grids import gridfromdims, BasicGrid

from .exceptions import ReaderError
from .utils import mkdate


class XarrayReaderBase:
    """
    Base class for readers backed by xarray objects (images, image stacks,
    timeseries).

    This base class provides methods to infer grid information and metadata
    from the stack. The constructor tests the various combinations of keyword
    arguments common to all readers.
    """

    def __init__(
        self,
        ds: xr.Dataset,
        varnames: Union[str, Sequence],
        timename: str = "time",
        latname: str = "lat",
        lonname: str = "lon",
        latdim: str = None,
        londim: str = None,
        locdim: str = None,
        lat: Union[np.ndarray, tuple] = None,
        lon: Union[np.ndarray, tuple] = None,
        landmask: Union[xr.DataArray, str] = None,
        bbox: Iterable = None,
        cellsize: float = None,
        curvilinear: bool = False,
        grid: BasicGrid = None,
        construct_grid: bool = True,
    ):
        # variable names
        if isinstance(varnames, str):
            varnames = [varnames]
        self.varnames = list(varnames)

        # coordinate and dimension names
        self.timename = timename
        self.latname = latname
        self.lonname = lonname
        self.latdim = latdim
        self.londim = londim
        self.locdim = locdim
        if curvilinear and (
            latdim is None
            or londim is None
            or latname is None
            or lonname is None
            or locdim is not None
        ):
            raise ReaderError(
                "For curvilinear grids, lat/lon-dim and lat/lon-name must " "be given."
            )
        if locdim is not None and (
            latname is None or lonname is None
        ):  # pragma: no cover
            raise ReaderError(
                "If locdim is given, latname and lonname must also be given."
            )

        # infer gridtype from passed coordinates and dimensions
        if locdim is None:
            if curvilinear:
                self.gridtype = "curvilinear"
            else:
                self.gridtype = "regular"
        else:
            self.gridtype = "unstructured"

        # additional arguments related to grid setup
        self.bbox = bbox
        self.cellsize = cellsize
        if isinstance(landmask, str):
            if ":" in landmask:
                fname, vname = landmask.split(":")
                self.landmask = xr.open_dataset(fname)[vname]
            else:
                self.landmask = self._get_landmask(ds, landmask)
        else:
            self.landmask = landmask

        # metadata
        self.global_attrs, self.array_attrs = self._metadata_from_dataset(ds)

        # grid
        self.lat, self.lon, self.grid = self._gridinfo_from_dataset(
            ds, lat, lon, grid, construct_grid=construct_grid
        )

    def _landmask_from_dataset(self, ds: xr.Dataset, landmask):
        return ds[landmask]

    def _metadata_from_dataset(self, ds: xr.Dataset):
        global_attrs = dict(ds.attrs)
        array_attrs = {v: dict(ds[v].attrs) for v in self.varnames}
        return global_attrs, array_attrs

    def _gridinfo_from_dataset(
        self, ds: xr.Dataset, lat, lon, grid, construct_grid=True
    ):
        """
        Full setup of grid when a dataset is available.
        """
        # the landmask might be required for creating the grid, and it might
        # still be a string since we didn't have a dataset available yet

        # The grid can either be inferred from the arguments passed, or from
        # the first file in the dataset
        if grid is not None or lat is not None or lon is not None:
            lat, lon, grid = self.gridinfo_from_arguments(
                lat, lon, grid, construct_grid=construct_grid
            )
        else:
            lat, lon = self._latlon_from_dataset(ds)
            if construct_grid:
                grid = self.grid_from_latlon(lat, lon)
            else:
                grid = None
        return lat, lon, grid

    def gridinfo_from_arguments(self, lat, lon, grid, construct_grid=True):
        if grid is not None:
            grid = self.finalize_grid(grid)
            lat = grid.arrlat
            lon = grid.arrlon
        elif lat is not None or lon is not None:
            assert (
                lat is not None
            ), "If custom lon is given, custom lat must also be given."
            assert (
                lon is not None
            ), "If custom lat is given, custom lon must also be given."
            lat = self.coord_from_argument(lat)
            lon = self.coord_from_argument(lon)
            if construct_grid:
                grid = self.grid_from_latlon(lat, lon)
            else:
                grid = None
        return lat, lon, grid

    def _latlon_from_dataset(self, ds, construct_grid=True):
        lat = self.coord_from_dataset(ds, "lat")
        lon = self.coord_from_dataset(ds, "lon")
        return lat, lon

    def coord_from_argument(self, coord):
        if isinstance(coord, np.ndarray):
            # we already have it in the way we want
            return coord
        elif isinstance(coord, (list, tuple)) and len(coord) == 3:
            start, stop, step = coord
            return np.round(np.arange(start, stop, step), 5)
        else:
            raise ReaderError(f"Wrong specification of argument: {coord}")

    def coord_from_dataset(self, ds, coordname):
        cname = getattr(self, coordname + "name")
        dimname = getattr(self, coordname + "dim")
        if dimname is None:
            # latdim/londim is not specified, which means that dimname and
            # coordname coincide
            dimname = cname

        if self.gridtype != "regular":
            # for curvilinear and unstructured grids the coordinates have to be
            # given in the dataset
            return ds[cname].values
        if cname in ds.dims:
            # coordinate is also a dimension, so it is a 1D array
            return ds[cname].values
        # coordinate is not a dimension, so we have to infer it from a variable
        coord = ds[cname]
        if coord.ndim > 1:
            axis = ds.dims.index(dimname)
            coord = self._get_coord_from_2d(
                coord.values, axis, fill_value=coord.attrs["_FillValue"]
            )
        return coord.values

    def coord_from_2d(coord, axis, fill_value=-9999):
        # It happens often that coordinates of a regular grid are still given
        # as 2D arrays, often also with fill values at non-land locations.
        # To get the 1D array, we therefore take the nanmean along the
        # corresponding axis and hope that no masked values remain

        # if the coordinate is the first axis, we have to take the mean over
        # the second one and vice versa
        axis = (axis + 1) % 2
        coord = np.ma.masked_equal(coord, fill_value)
        coord = coord.mean(axis=axis).filled(np.nan)
        if np.any(np.isnan(coord)):  # pragma: no cover
            raise ReaderError("Inferred coordinate values contain NaN!")
        return coord

    def grid_from_latlon(self, lat: np.ndarray, lon: np.ndarray):
        if self.gridtype == "regular":
            grid = gridfromdims(lon, lat)
        elif self.gridtype == "curvilinear":
            grid = BasicGrid(lon.ravel(), lat.ravel())
        elif self.gridtype == "unstructured":
            grid = BasicGrid(lon, lat)
        else:
            raise ReaderError(
                "gridtype must be 'regular', 'curvilinear', or 'unstructured'"
            )
        return self.finalize_grid(grid)

    def finalize_grid(self, grid):
        """
        Applies landmask and bounding box to grid
        """

        if hasattr(self, "landmask") and self.landmask is not None:
            if self.gridtype != "unstructured":
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
        logging.debug(f"construct_grid: Number of active gpis: {num_gpis}")

        if hasattr(self, "cellsize") and self.cellsize is not None:
            grid = grid.to_cell_grid(cellsize=self.cellsize)
            num_cells = len(grid.get_cells())
            logging.debug(f"_grid_from_xarray: Number of grid cells: {num_cells}")

        return grid


class LevelSelectionMixin:
    def normalize_level(self, level, varnames):
        # levels
        if (
            level is not None
            and len(varnames) == 1
            and not isinstance(list(level.values())[0], dict)
        ):
            level = {varnames[0]: level}
        else:
            level = level
        return level

    def select_levels(self, ds: xr.Dataset):
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
            if levelname not in arr.dims:
                warnings.warn(
                    f"Selection from level {levelname} requested, but"
                    f" {levelname} is not an array dimension. Existing"
                    f" dimensions are {arr.dims}."
                )
                continue
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
    ) -> Dict[str, Union[np.ndarray, dask.array.core.Array]]:  # pragma: no cover
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
        if self.gridtype == "regular":
            # we can simply wrap the data with time, lat, and lon
            coords[self.latname] = self.lat
            coords[self.lonname] = self.lon
            dims = (self.timename, self.latname, self.lonname)
        elif self.gridtype == "curvilinear":
            coords[self.latname] = ([self.latdim, self.londim], self.lat)
            coords[self.lonname] = ([self.latdim, self.londim], self.lon)
            dims = (self.timename, self.latdim, self.londim)
        else:  # unstructured grid
            coords[self.latname] = (self.locdim, self.lat.data)
            coords[self.lonname] = (self.locdim, self.lon.data)
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

    def read(self, timestamp: Union[datetime.datetime, str], **kwargs) -> Image:
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
            raise ReaderError(f"Timestamp {timestamp} is not available in the dataset!")

        img = self.read_block(timestamp, timestamp, _apply_landmask_bbox=False).isel(
            {self.timename: 0}
        )
        if self.gridtype != "unstructured":
            if self.gridtype == "regular":
                latname = self.latname
                lonname = self.lonname
            elif self.gridtype == "curvilinear":
                latname = self.latdim
                lonname = self.londim
            img = img.stack(dimensions={"loc": (latname, lonname)})
        data = {
            varname: img[varname].values[self.grid.activegpis]
            for varname in self.varnames
        }
        metadata = self.array_attrs
        return Image(self.grid.arrlon, self.grid.arrlat, data, metadata, timestamp)
