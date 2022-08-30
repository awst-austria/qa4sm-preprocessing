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


class ReaderBase:
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
        ):  # pragma: no cover
            raise ReaderError(
                "For curvilinear grids, lat/lon-dim and lat/lon-name must"
                " be given."
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
                self.landmask = self._landmask_from_dataset(ds, landmask)
        else:
            self.landmask = landmask

        # metadata
        self.global_attrs, self.array_attrs = self._metadata_from_dataset(ds)

        # grid
        self.lat, self.lon, self.grid = self._gridinfo_from_dataset(
            ds, lat, lon, construct_grid
        )

    def _landmask_from_dataset(self, ds: xr.Dataset, landmask):
        return ds[landmask]

    def _metadata_from_dataset(self, ds: xr.Dataset):
        global_attrs = dict(ds.attrs)
        array_attrs = {v: dict(ds[v].attrs) for v in self.varnames}
        return global_attrs, array_attrs

    def _gridinfo_from_dataset(self, ds: xr.Dataset, lat, lon, construct_grid):
        """
        Full setup of grid when a dataset is available.
        """
        # the landmask might be required for creating the grid, and it might
        # still be a string since we didn't have a dataset available yet

        # The grid can either be inferred from the arguments passed, or from
        # the first file in the dataset
        if lat is not None or lon is not None:
            lat, lon = self._latlon_from_arguments(lat, lon)
        else:
            lat, lon = self._latlon_from_dataset(ds)
        if construct_grid:
            grid = self.grid_from_latlon(lat, lon)
        else:
            grid = None
        return lat, lon, grid

    def _latlon_from_arguments(self, lat, lon):
        assert (
            lat is not None
        ), "If custom lon is given, custom lat must also be given."
        assert (
            lon is not None
        ), "If custom lat is given, custom lon must also be given."
        lat = self.coord_from_argument(lat)
        lon = self.coord_from_argument(lon)
        return lat, lon

    def _latlon_from_dataset(self, ds):
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
        else:  # pragma: no cover
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
            axis = list(ds.dims).index(dimname)
            if "_FillValue" in coord.attrs:
                fill_value = coord.attrs["_FillValue"]
            elif hasattr(self, "fill_value"):
                fill_value = self.fill_value
            else:
                fill_value = None
            coord = self.coord_from_2d(
                coord.values, axis, fill_value=fill_value
            )
            return coord
        else:
            return coord.values

    def coord_from_2d(self, coord, axis, fill_value=-9999):
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
        else:  # pragma: no cover
            raise ReaderError(
                "gridtype must be 'regular', 'curvilinear', or 'unstructured'"
            )
        return self.finalize_grid(grid)

    def finalize_grid(self, grid):
        """
        Applies landmask and bounding box to grid
        """

        if hasattr(self, "landmask") and self.landmask is not None:
            landmask = self._stack(self.landmask)
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
        logging.debug(f"finalize_grid: Number of active gpis: {num_gpis}")

        if hasattr(self, "cellsize"):  # pragma: no branch
            if self.cellsize is None:
                # Automatically set a suitable cell size, aiming at cell sizes
                # of about 30**2 pixels.
                deltalat = np.max(grid.activearrlat) - np.min(grid.activearrlat)
                deltalon = np.max(grid.activearrlon) - np.min(grid.activearrlon)
                self.cellsize = 30 * np.sqrt(deltalat*deltalon/len(grid.activegpis))
            grid = grid.to_cell_grid(cellsize=self.cellsize)
            num_cells = len(grid.get_cells())
            logging.debug(
                f"finalize_grid: Number of grid cells: {num_cells}"
            )

        return grid

    def _stack(self, img):
        if self.gridtype != "unstructured":
            if self.gridtype == "regular":
                latname = self.latname
                lonname = self.lonname
            else:  # self.gridtype == "curvilinear"
                latname = self.latdim
                lonname = self.londim
            img = img.stack(dimensions={"loc": (latname, lonname)})
        return img


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
                    f"Selection from level '{levelname}' requested, but"
                    f" '{levelname}' is not an array dimension. Existing"
                    f" dimensions are {arr.dims}."
                )
                continue
            is_list = isinstance(idx, list)
            if not is_list:
                idx = [idx]
            tmp_list = []
            for name, arr in output_list:
                for i in idx:
                    tmparr = arr.isel({levelname: i})
                    if is_list:
                        tmpname = name + "_" + str(i)
                    else:
                        tmpname = name
                    tmp_list.append((tmpname, tmparr))
            output_list = tmp_list
        return output_list
