import cftime
import numpy as np
import os
import pandas as pd
from pathlib import Path
from typing import Union, Iterable, Sequence
import warnings
import xarray as xr

from pygeogrids.netcdf import load_grid
from pynetcf.time_series import GriddedNcOrthoMultiTs as _GriddedNcOrthoMultiTs

from .base import ReaderBase


class StackTs(ReaderBase):
    """
    Wrapper for xarray.Dataset when timeseries of the data should be read.

    This is useful if you are using functions from the TUW-GEO package universe
    which require a timeseries reader, but you don't have the data in the
    pynetcf timeseries format.

    Since this is reading along the time dimension, you should make sure that
    the time dimension is either the last dimension in your netcdf (the fastest
    changing dimension), or that it is chunked in a way that makes timeseries
    access fast. To move the time dimension last, you can use the function
    ``reading.transpose.write_transposed_dataset`` or programs like
    ``ncpdq`` or ``ncks``.


    Parameters
    ----------
    ds : xr.Dataset, Path or str
        Xarray dataset (or filename of a netCDF file). Must have a time
        coordinate and either `latname`/`latdim` and `lonname`/`latdim` (for a
        regular latitude-longitude grid) or `locdim` as additional
        coordinates/dimensions.
    varnames : str
        Names of the variable that should be read.
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
    lat : tuple or np.ndarray, optional (default: None)
        If the latitude can not be inferred from the dataset you can specify it
        by giving (start, stop, step) or an array of latitude values
    lon : tuple or np.ndarray, optional (default: None)
        If the longitude can not be inferred from the dataset you can specify
        it by giving (start, stop, step) or an array of longitude values.
    landmask : xr.DataArray, optional
        A land mask to be applied to reduce storage size.
    bbox : Iterable, optional
        (lonmin, latmin, lonmax, latmax) of a bounding box.
    cellsize : float, optional
        Spatial coverage of a single cell file in degrees. Default is ``None``.
    construct_grid : bool, optional (default: True)
        Whether to construct a BasicGrid instance. For very large datasets it
        might be necessary to turn this off, because the grid requires too much
        memory.
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
        landmask: xr.DataArray = None,
        bbox: Iterable = None,
        cellsize: float = None,
        construct_grid: bool = True,
    ):
        if isinstance(ds, (str, Path)):
            ds = xr.open_dataset(ds)

        super().__init__(
            ds,
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
            construct_grid=construct_grid,
        )

        if self.gridtype == "unstructured":
            self.data = ds[self.varnames]
        else:
            # we have to reshape the data
            latdim = self.latdim if self.latdim is not None else self.latname
            londim = self.londim if self.londim is not None else self.lonname
            self.orig_data = ds[self.varnames]
            self.data = self.orig_data.stack({"loc": (latdim, londim)})
            self.locdim = "loc"

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
            if self.gridtype != "unstructured":
                lon, lat = self.grid.gpi2lonlat(gpi)

        elif len(args) == 2:
            lon = args[0]
            lat = args[1]
            if self.gridtype == "unstructured":
                gpi = self.grid.find_nearest_gpi(lon, lat)[0]
                if not isinstance(gpi, np.integer):  # pragma: no cover
                    raise ValueError(
                        f"No gpi near (lon={lon}, lat={lat}) found"
                    )
        else:  # pragma: no cover
            raise ValueError(
                f"args must have length 1 or 2, but has length {len(args)}"
            )

        if self.gridtype == "unstructured":
            data = self.data[{self.locdim: gpi}]
        else:
            data = self.orig_data.sel(lat=lat, lon=lon)
        df = data.to_pandas()[self.varnames]
        return df


class GriddedNcOrthoMultiTs(_GriddedNcOrthoMultiTs):
    def __init__(
        self,
        ts_path,
        grid_path=None,
        timevarname=None,
        read_bulk=None,
        kd_tree_name="pykdtree",
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
        read_bulk : boolean, optional (default: None)
            If set to True (default) the data of all locations is read into memory,
            and subsequent calls to read_ts read from the cache and not from
            disk this makes reading complete files faster.
        timevarname : str, optional (default: None)
            Name of the time variable to use instead of the original timestamps.
        kd_tree_name : str, optional (default: "pykdtree")
            Name of the Kd-tree engine used in the grid. Available options are
            "pykdtree" and "scipy".

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
        read_dates : boolean, optional (default:False)
            if false dates will not be read automatically but only on specific
            request useable for bulk reading because currently the netCDF
            num2date routine is very slow for big datasets
        """
        if grid_path is None:  # pragma: no branch
            grid_path = os.path.join(ts_path, "grid.nc")
        grid = load_grid(grid_path, kd_tree_name=kd_tree_name)

        ioclass_kws = kwargs.get("ioclass_kws", {})
        if "ioclass_kws" in kwargs:
            del kwargs["ioclass_kws"]
        # if read_bulk is not given, we use the value from ioclass_kws, or True
        # if this is also given. Otherwise we overwrite the value in ioclass_kws
        if read_bulk is None:
            read_bulk = ioclass_kws.get("read_bulk", True)
        else:
            if "read_bulk" in ioclass_kws and read_bulk != ioclass_kws["read_bulk"]:
                warnings.warn(
                    f"read_bulk={read_bulk} but ioclass_kws['read_bulk']="
                    f" {ioclass_kws['read_bulk']}. The first takes precedence."
                )
        ioclass_kws["read_bulk"] = read_bulk
        super().__init__(ts_path, grid, ioclass_kws=ioclass_kws, **kwargs)
        self.timevarname = timevarname


    def read(self, *args, **kwargs) -> pd.DataFrame:
        df = super().read(*args, **kwargs)
        if self.timevarname is not None:
            unit = self.fid.dataset.variables[self.timevarname].units
            times = df[self.timevarname].values
            index = pd.DatetimeIndex(
                cftime.num2pydate(times, unit)
            )
            df.index = index
            df.drop(self.timevarname, axis="columns", inplace=True)
        return df
