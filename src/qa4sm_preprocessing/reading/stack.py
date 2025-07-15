import datetime
import numpy as np
from pathlib import Path
from typing import Iterable, Union, Sequence
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
        Xarray dataset (or filename of a netCDF file).
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
    construct_grid : bool, optional (default: True)
        Whether to construct a BasicGrid instance. For very large datasets it
        might be necessary to turn this off, because the grid requires too much
        memory.
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
    construct_grid : bool, optional (default: True)
        Whether to construct a BasicGrid instance. For very large datasets it
        might be necessary to turn this off, because the grid requires too much
        memory.
    timeoffsetvarname : str, optional (default: None)
        Sometimes an image is not really an image (i.e. a snapshot at a fixed
        time), but is composed of multiple observations at different times
        (e.g. satellite overpasses). In these cases, image files often contain
        a time offset variable, that gives the exact observation time.
        Time offset is calculated after applying `rename`, so
        `timeoffsetvarname` should be the renamed variable name.
    timeoffsetunit : str, optional (default: None)
        The unit of the time offset. Required if `timeoffsetvarname` is not
        ``None``. Valid values are "seconds", "minutes", "hours", "days".
    **open_dataset_kwargs : keyword arguments
       Additional keyword arguments passed to ``xr.open_dataset`` in case `ds`
       is a filename.
    """

    def __init__(
        self,
        ds: Union[xr.Dataset, str, Path],
        varnames: Union[str, Sequence] = None,
        level: dict = None,
        timename: str = None,
        latname: str = None,
        lonname: str = None,
        ydim: str = None,
        xdim: str = None,
        locdim: str = None,
        lat: Union[np.ndarray, tuple] = None,
        lon: Union[np.ndarray, tuple] = None,
        gridtype: str = "infer",
        construct_grid: bool = True,
        landmask: xr.DataArray = None,
        bbox: Iterable = None,
        cellsize: float = None,
        timeoffsetvarname=None,
        timeoffsetunit=None,
        rename: dict = None,
        load: bool = True,
        **open_dataset_kwargs,
    ):

        if isinstance(ds, (str, Path)):
            ds = xr.open_dataset(ds, **open_dataset_kwargs)
        varnames = self._maybe_add_varnames(varnames, [timeoffsetvarname])
        if rename is not None:
            ds = ds.rename(rename)

        super(StackImageReader, self).__init__(
            ds,
            varnames,
            timename=timename,
            latname=latname,
            lonname=lonname,
            ydim=ydim,
            xdim=xdim,
            locdim=locdim,
            lat=lat,
            lon=lon,
            landmask=landmask,
            bbox=bbox,
            cellsize=cellsize,
            gridtype=gridtype,
            construct_grid=construct_grid,
        )
        self.data = ds
        if load:
            self.data.load()
        self._timestamps = ds.indexes[self.timename].to_pydatetime()
        self.timeoffsetvarname = timeoffsetvarname
        self.timeoffsetunit = timeoffsetunit

    def _read_block(
        self, start: datetime.datetime, end: datetime.datetime
    ) -> xr.Dataset:
        block = self.data.sel({self.timename: slice(start, end)})[self.varnames]
        block = block.transpose(*self.get_dims())
        block_dict = {v: self._fix_ndim(block[v].data) for v in self.varnames}
        return block_dict
