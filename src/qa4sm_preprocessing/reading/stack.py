import datetime
import numpy as np
from pathlib import Path
import shutil
from typing import Iterable, Union
import xarray as xr

from .imagebase import ImageReaderBase


class StackImageReader(ImageReaderBase):
    """
    Image reader that wraps a xarray.Dataset.

    This can be used as a generic image reader for netcdf stacks, e.g. for
    reformatting the data to timeseries format using the package ``repurpose``
    (which is implemented in ``reading.reshuffle`` and can also be done
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
    latname : str, optional (default: "lat")
        If `locdim` is given (i.e. for non-rectangular grids), this must be the
        name of the latitude data variable, otherwise must be the name of the
        latitude coordinate.
    lonname : str, optional (default: "lon")
        If `locdim` is given (i.e. for non-rectangular grids), this must be the
        name of the longitude data variable, otherwise must be the name of the
        longitude coordinate.
    latdim : str, optional
        The name of the latitude dimension in case it's not the same as the
        latitude coordinate variable.
    londim : str, optional
        The name of the longitude dimension in case it's not the same as the
        longitude coordinate variable.
    locdim : str, optional
        The name of the location dimension for non-rectangular grids. If this
        is given, you *MUST* provide `lonname` and `latname`.
    lat : tuple, optional (default: None)
        If the latitude can not be inferred from the dataset you can specify it
        by giving (start, stop, step).
    lon : tuple, optional (default: None)
        If the longitude can not be inferred from the dataset you can specify
        it by giving (start, stop, step).
    construct_grid : bool, optional (default: True)
        Whether to construct a BasicGrid instance. For very large datasets it
        might be necessary to turn this off, because the grid requires too much
        memory.
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
    """

    def __init__(
        self,
        ds: xr.Dataset,
        varnames: str,
        level: dict = None,
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
        curvilinear: bool = False,
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
            curvilinear=curvilinear,
            construct_grid=construct_grid,
        )
        self.data = ds
        self._timestamps = ds.indexes[self.timename].to_pydatetime()

    def _read_block(
        self, start: datetime.datetime, end: datetime.datetime
    ) -> xr.Dataset:
        block = self.data.sel({self.timename: slice(start, end)})
        block_dict = {var: block[var].data for var in self.varnames}
        return block_dict
