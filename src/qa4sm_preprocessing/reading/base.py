import logging
import numpy as np
from typing import Union, Iterable, Sequence
import warnings
import xarray as xr
import cf_xarray  # noqa

from pygeogrids.grids import gridfromdims, BasicGrid

from .exceptions import ReaderError
from .cf import get_coord, get_time


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
        varnames: Union[str, Sequence] = None,
        timename: str = None,
        latname: str = None,
        lonname: str = None,
        ydim: str = None,
        xdim: str = None,
        locdim: str = None,
        lat: Union[np.ndarray, tuple] = None,
        lon: Union[np.ndarray, tuple] = None,
        landmask: Union[xr.DataArray, str] = None,
        bbox: Iterable = None,
        cellsize: float = None,
        gridtype: str = "infer",
        construct_grid: bool = True,
    ):
        # variable names
        if varnames is None:
            varnames = list(ds.data_vars)
        elif isinstance(varnames, str):
            varnames = [varnames]
        self.varnames = list(varnames)
        if timename is None:
            if isinstance(ds, xr.Dataset):
                timename = get_time(ds).name
            else:
                timename = "time"
        self.timename = timename

        # we set the coordinate specification here, but the might be modified
        # later
        self.latname = latname
        self.lonname = lonname
        self.ydim = ydim
        self.xdim = xdim
        self.locdim = locdim
        self.gridtype = gridtype

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

        # infer metadata
        self.global_attrs, self.array_attrs = self._metadata_from_dataset(ds)

        # infer the coordinates and grid
        if lat is not None or lon is not None:
            # lat and lon are passed as arrays and we don't need to infer the
            # values
            (
                self.lat,
                self.lon,
                self.latname,
                self.lonname,
                self.ydim,
                self.xdim,
            ) = self._coordinfo_from_args(lat, lon)
        else:
            (
                self.lat,
                self.lon,
                self.latname,
                self.lonname,
                self.ydim,
                self.xdim,
            ) = self._coordinfo_from_dataset(ds)

        # now we can infer the gridtype
        if self.gridtype == "infer":
            if locdim is not None:
                self.gridtype = "unstructured"
            else:
                if self.lat.ndim == 2:
                    self.gridtype = "curvilinear"
                elif self.lat.ndim == 1:
                    self.gridtype = "regular"
                else:  # pragma: no cover
                    raise ReaderError(
                        "Coordinate array must have 1 or 2 dimensions!"
                    )

        if construct_grid:
            self.grid = self.grid_from_latlon(self.lat, self.lon, self.gridtype)
        else:
            self.grid = None

        # Done!

    def _landmask_from_dataset(self, ds: xr.Dataset, landmask):
        return ds[landmask]

    def _metadata_from_dataset(self, ds: xr.Dataset):
        global_attrs = dict(ds.attrs)
        array_attrs = {v: dict(ds[v].attrs) for v in self.varnames}
        return global_attrs, array_attrs

    def _coordinfo_from_args(self, lat, lon):
        assert (
            lat is not None and lon is not None
        ), "'lat' and 'lon' must both be specified or both be omitted!"
        assert not (self.ydim is None or self.xdim is None), (
            "If 'lat' and 'lon' values are passed, 'ydim' and 'xdim' must"
            " be specified!"
        )
        lat = self._coord_from_argument(lat)
        lon = self._coord_from_argument(lon)
        # lat, lon are np.ndarrays here, so we can't infer the name from
        # the metadata
        latname = self.latname if self.latname is not None else "lat"
        lonname = self.lonname if self.lonname is not None else "lon"
        return lat, lon, latname, lonname, self.ydim, self.xdim

    def _coordinfo_from_dataset(self, ds):
        # we have to infer the lat/lon values from the dataset
        lat, lon = self._latlon_from_dataset(ds)
        # infer latname/lonname and ydim/xdim from lat/lon coordinate arrays
        # if latname is not None, this effectively does nothing
        latname = lat.name
        lonname = lon.name
        assert lat.ndim in [1, 2], "Coordinates must have at most 2 dimensions."
        if self.ydim is None and self.xdim is None:
            # infer ydim and xdim from the coordinate arrays
            if lat.ndim == 2:  # pragma: no branch
                # for 2D coordinate arrays, both must have the same dims,
                # the first one is the y-dim, the second one the x-dim
                ydim, xdim = lat.dims
            elif lat.ndim == 1:  # pragma: no branch
                # for 1D coordinate arrays, the ydim is the latdim, the
                # xdim is the londim
                ydim = lat.dims[0]
                xdim = lon.dims[0]
        else:
            ydim = self.ydim
            xdim = self.xdim

        # sometimes the coordinate arrays are 2D, but actually represent a
        # regular grid that we want to infer. Then we have to transform them to
        # 1D. In this case the ydim is the latitude dimension and the xdim is
        # the longitude dimension
        if self.gridtype == "regular" and lat.ndim == 2:
            lat = self._1d_coord_from_2d(lat, ydim)
            lon = self._1d_coord_from_2d(lon, xdim)
        # we want the coordinate arrays to be numpy arrays
        lat = np.asarray(lat)
        lon = np.asarray(lon)
        return lat, lon, latname, lonname, ydim, xdim

    def _latlon_from_dataset(self, ds):
        if self.latname is None and self.lonname is None:
            # get specifications from CF conventions
            lat = get_coord(ds, "latitude", alternatives=["lat", "LAT"])
            lon = get_coord(ds, "longitude", alternatives=["lon", "LON"])
        elif self.latname is not None and self.lonname is not None:
            lat = ds[self.latname]
            lon = ds[self.lonname]
        else:  # pragma: no cover
            raise ReaderError(
                "'latname' and 'lonname' must either both be specified or"
                " both be omitted!"
            )
        return lat, lon

    def _coord_from_argument(self, coord):
        if isinstance(coord, np.ndarray):
            # we already have it in the way we want
            return coord
        elif isinstance(coord, (list, tuple)) and len(coord) == 3:
            start, stop, step = coord
            return np.round(np.arange(start, stop, step), 5)
        else:  # pragma: no cover
            raise ReaderError(f"Wrong specification of argument: {coord}")

    def _1d_coord_from_2d(self, coord, dimname, fill_value=-9999):
        # It happens often that coordinates of a regular grid are still given
        # as 2D arrays, often also with fill values at non-land locations.
        # To get the 1D array, we therefore take the nanmean along the
        # corresponding axis and hope that no masked values remain

        # if the coordinate is the first axis, we have to take the mean over
        # the second one and vice versa
        axis = list(coord.dims).index(dimname)
        axis = (axis + 1) % 2
        coord = np.ma.masked_equal(coord, fill_value)
        coord = coord.mean(axis=axis).filled(np.nan)
        if np.any(np.isnan(coord)):  # pragma: no cover
            raise ReaderError("Inferred coordinate values contain NaN!")
        return coord

    def grid_from_latlon(self, lat: np.ndarray, lon: np.ndarray, gridtype):
        if gridtype == "regular":
            grid = gridfromdims(lon, lat)
        elif gridtype == "curvilinear":
            grid = BasicGrid(lon.ravel(), lat.ravel())
        elif gridtype == "unstructured":
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
            land_gpis = grid.get_grid_points()[0][landmask.values]
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
                self.cellsize = 30 * np.sqrt(deltalat * deltalon / len(grid.activegpis))
            grid = grid.to_cell_grid(cellsize=self.cellsize)
            num_cells = len(grid.get_cells())
            logging.debug(f"finalize_grid: Number of grid cells: {num_cells}")

        return grid

    def _stack(self, img):
        if self.gridtype != "unstructured":
            if self.gridtype == "regular":
                img = img.stack(dimensions={"loc": (self.latname, self.lonname)})
            else:  # curvilinear grid
                img = img.stack(dimensions={"loc": (self.ydim, self.xdim)})
        return img

    def _maybe_add_varnames(self, varnames, to_add):
        if varnames is None:
            return varnames
        if isinstance(varnames, str):
            varnames = [varnames]
        for vname in to_add:
            if vname is not None and vname not in varnames:
                varnames.append(vname)
        return varnames


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
