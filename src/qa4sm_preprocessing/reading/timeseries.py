from abc import abstractmethod
import datetime
import glob
import logging
import numpy as np
import os
import pandas as pd
from pathlib import Path
import re
import shutil
from tqdm.auto import tqdm
from typing import Union, Iterable, Sequence, Mapping
import warnings
import xarray as xr
import yaml
from zipfile import ZipFile

from pygeogrids.grids import BasicGrid
from pygeogrids.netcdf import load_grid, save_grid
from pynetcf.time_series import GriddedNcOrthoMultiTs as _GriddedNcOrthoMultiTs
from pynetcf.time_series import (
    GriddedNcContiguousRaggedTs as _GriddedNcContiguousRaggedTs,
)

from .base import ReaderBase
from .utils import numpy_timeoffsetunit, infer_cellsize
from .exceptions import ReaderError


class _TimeModificationMixin:
    @abstractmethod
    def _get_time_unit(self):  # pragma: no cover
        ...

    def _modify_time(self, df):
        if self.timevarname is not None and self.timevarname in df:
            cf_unit = self._get_time_unit()
            split = cf_unit.split(" ")
            unit = split[0]
            start_date = " ".join(split[2:])
            np_unit = numpy_timeoffsetunit(unit)
            start = np.datetime64(start_date)
            values = df[self.timevarname].values
            # careful: the conversion to timedelta64 will lose all decimals, so
            # make sure that the time unit is small enough to represent all
            # timestamps as integers
            times = start + values.astype(f"timedelta64[{np_unit}]")
            index = pd.DatetimeIndex(times)
            df.index = index
            df.drop(self.timevarname, axis="columns", inplace=True)
        if self.timeoffsetvarname is not None and self.timeoffsetvarname in df:
            times = df.index.values
            unit = numpy_timeoffsetunit(self.timeoffsetunit)
            offset = df[self.timeoffsetvarname].values
            delta = offset.astype(f"timedelta64[{unit}]")
            df.index = pd.DatetimeIndex(times + delta)
            df.drop(self.timeoffsetvarname, axis="columns", inplace=True)
        return df


class _TimeseriesRepurposeMixin:
    @property
    @abstractmethod
    def _ioclass(self):  # pragma: no cover
        ...

    def repurpose(
        self,
        outpath: Union[Path, str],
        start: Union[datetime.datetime, str] = None,
        end: Union[datetime.datetime, str] = None,
        overwrite: bool = False,
        **ioclass_kwargs,
    ):
        """
        Transforms the single contiguous ragged netCDF array to multiple files
        in the pynetcf format.

        Parameters
        ----------
        outpath : str or path
            Directory name where the timeseries files are written to.
        start : str or datetime.datetime, optional
            Processing start time
        end : str or datetime.datetime, optional
            Processing end time
        overwrite: bool, optional (default: False)
            Whether to overwrite existing directories. If set to False, the
            function will return a reader for the existing directory.
        **ioclass_kwargs: additional keyword arguments for the pynetcf
          timeseries reader.

        Returns
        -------
        reader : GriddedTs
            Reader for the timeseries files.

        Note
        ----
        Current implementation is not very fast because it loops over all grid
        points.
        """
        outpath = Path(outpath)
        if start is not None or end is not None:
            assert start is not None and end is not None, (
                "If 'start' or 'end' is given, the other one also"
                " needs to be set."
            )
            period = (start, end)
        else:
            period = None
        if (outpath / "grid.nc").exists() and overwrite:  # pragma: no branch
            shutil.rmtree(outpath)
        ioclass = self._ioclass
        if not (
            outpath / "grid.nc"
        ).exists():  # if overwrite=True, it was deleted now
            outpath.mkdir(exist_ok=True, parents=True)

            io = ioclass(outpath, self.grid, mode="w")
            attrs = {}
            for v in self.varnames:
                try:
                    attrs[v] = self.get_metadata(v)
                except KeyError:  # pragma: no cover
                    attrs[v] = {}
            # loop over cells and the write each cell separately
            for cell in tqdm(self.grid.get_cells()):
                (
                    cell_gpis,
                    cell_lons,
                    cell_lats,
                ) = self.grid.grid_points_for_cell(cell)
                for gpi in cell_gpis:
                    ts = self.read(gpi, period=period)
                    io._write_gp(gpi, ts, attributes=attrs)
            io.close()
            save_grid(outpath / "grid.nc", self.grid)
        else:  # pragma: no cover
            logging.info(f"Output path already exists: {str(outpath)}")
        io = ioclass(str(outpath), self.grid, **ioclass_kwargs, mode="r")
        return io

    def get_metadata(self, varname: str) -> Mapping:
        return dict(self.data[varname].attrs)


class _EasierGriddedNcMixin:

    def variable_description(self) -> Mapping[str, Mapping]:
        """
        Returns
        -------
        attrs : dict
            A dictionary mapping existing variables to metadata stored in the
            netCDF files.
        """
        ncfiles = glob.glob(str(Path(self.path) / "*.nc"))
        fname = next(filter(lambda s: not s.endswith("grid.nc"), ncfiles))

        with xr.open_dataset(fname) as ds:
            uninteresting = ["row_size", "location_id", "location_description"]
            attrs = {v: dict(ds[v].attrs) for v in list(ds.data_vars) if v not
                     in uninteresting}
        return attrs

    def _get_grid(self, ts_path, grid, kd_tree_name="pykdtree"):
        if grid is None:  # pragma: no branch
            grid = os.path.join(ts_path, "grid.nc")
        if isinstance(grid, (str, Path)):
            grid = load_grid(grid, kd_tree_name=kd_tree_name)
        assert isinstance(grid, BasicGrid)
        return grid

    def _modify_kwargs(self, read_bulk, kwargs):
        ioclass_kws = kwargs.get("ioclass_kws", {})
        if "ioclass_kws" in kwargs:
            del kwargs["ioclass_kws"]
        # if read_bulk is not given, we use the value from ioclass_kws, or True
        # if this is also given. Otherwise we overwrite the value in
        # ioclass_kws
        if read_bulk is None:
            read_bulk = ioclass_kws.get("read_bulk", True)
        else:
            if (
                "read_bulk" in ioclass_kws
                and read_bulk != ioclass_kws["read_bulk"]
            ):
                warnings.warn(
                    f"read_bulk={read_bulk} but ioclass_kws['read_bulk']="
                    f" {ioclass_kws['read_bulk']}. The first takes precedence."
                )
        ioclass_kws["read_bulk"] = read_bulk
        return kwargs, ioclass_kws


class GriddedNcOrthoMultiTs(
    _GriddedNcOrthoMultiTs, _TimeModificationMixin, _EasierGriddedNcMixin
):
    def __init__(
        self,
        ts_path,
        grid: Union[BasicGrid, Path, str] = None,
        timevarname: str = None,
        timeoffsetvarname: str = "time_offset",
        timeoffsetunit: str = "seconds",
        read_bulk: bool = None,
        kd_tree_name: str = "pykdtree",
        **kwargs,
    ):
        """
        Class for reading time series in pynetcf format after reshuffling.

        Parameters
        ----------
        ts_path : str
            Directory where the netcdf time series files are stored
        grid : BasicGrid or str, optional
            BasicGrid or path to grid file, that is used to organize the
            location of time series to read. If None is passed, grid.nc is
            searched for in the ts_path.
        read_bulk : boolean, optional (default: None)
            If set to True (default) the data of all locations is read into
            memory, and subsequent calls to read_ts read from the cache and not
            from disk this makes reading complete files faster.
        timevarname : str, optional (default: None)
            Name of the time variable (absolute time) to use instead of the
            original timestamps.
        timeoffsetvarname : str, optional (default: None)
            Name of the time offset variable name (relative to original
            timestamps).
        timeoffsetunit : str, optional (default: None)
            The unit of the time offset. Required if `timeoffsetvarname` is not
            ``None``. Valid values are "seconds"/, "minutes", "hours", "days".
        kd_tree_name : str, optional (default: "pykdtree")
            Name of the Kd-tree engine used in the grid. Available options are
            "pykdtree" and "scipy".

        Additional keyword arguments
        ----------------------------
        parameters : list, optional (default: None)
            Specific variable names to read, if None are selected, all are
            read.
        offsets : dict, optional (default: None)
            Offsets (values) that are added to the parameters (keys)
        scale_factors : dict, optional (default:None)
            Offset (value) that the parameters (key) is multiplied with
        ioclass_kws: dict

        Optional keyword arguments to pass to OrthoMultiTs class:
        ---------------------------------------------------------
        read_dates : boolean, optional (default: False)
            if false dates will not be read automatically but only on specific
            request useable for bulk reading because currently the netCDF
            num2date routine is very slow for big datasets
        """
        grid = self._get_grid(ts_path, grid, kd_tree_name)
        kwargs, ioclass_kws = self._modify_kwargs(read_bulk, kwargs)
        super().__init__(ts_path, grid, ioclass_kws=ioclass_kws, **kwargs)
        self.timevarname = timevarname
        self.timeoffsetvarname = timeoffsetvarname
        self.timeoffsetunit = timeoffsetunit

    def read(self, *args, **kwargs) -> pd.DataFrame:
        # already support the period kwarg
        df = super().read(*args, **kwargs)
        df = self._modify_time(df)
        return df

    def _get_time_unit(self):
        return self.dataset.variables[self.timevarname].units

    @property
    def dataset(self):
        if self.fid is not None:
            return self.fid.dataset
        else:  # pragma: no cover
            return None


class GriddedNcContiguousRaggedTs(
    _GriddedNcContiguousRaggedTs, _EasierGriddedNcMixin
):
    def __init__(
        self,
        ts_path,
        grid: Union[BasicGrid, Path, str] = None,
        read_bulk: bool = None,
        kd_tree_name: str = "pykdtree",
        **kwargs,
    ):
        """
        Class for reading time series in pynetcf format after reshuffling.

        Parameters
        ----------
        ts_path : str
            Directory where the netcdf time series files are stored
        grid : BasicGrid or str, optional
            BasicGrid or path to grid file, that is used to organize the
            location of time series to read. If None is passed, grid.nc is
            searched for in the ts_path.
        read_bulk : boolean, optional (default: None)
            If set to True (default) the data of all locations is read into
            memory, and subsequent calls to read_ts read from the cache and not
            from disk this makes reading complete files faster.
        kd_tree_name : str, optional (default: "pykdtree")
            Name of the Kd-tree engine used in the grid. Available options are
            "pykdtree" and "scipy".

        Additional keyword arguments
        ----------------------------
        parameters : list, optional (default: None)
            Specific variable names to read, if None are selected, all are
            read.
        offsets : dict, optional (default: None)
            Offsets (values) that are added to the parameters (keys)
        scale_factors : dict, optional (default:None)
            Offset (value) that the parameters (key) is multiplied with
        ioclass_kws: dict

        Optional keyword arguments to pass to OrthoMultiTs class:
        ---------------------------------------------------------
        read_dates : boolean, optional (default: False)
            if false dates will not be read automatically but only on specific
            request useable for bulk reading because currently the netCDF
            num2date routine is very slow for big datasets
        """
        grid = self._get_grid(ts_path, grid, kd_tree_name)
        kwargs, ioclass_kws = self._modify_kwargs(read_bulk, kwargs)
        super().__init__(ts_path, grid, ioclass_kws=ioclass_kws, **kwargs)


class StackTs(ReaderBase, _TimeModificationMixin, _TimeseriesRepurposeMixin):
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
        Xarray dataset (or filename of a netCDF file).
    varnames : str
        Names of the variable that should be read.
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
    timevarname : str, optional (default: None)
        Name of the time variable (absolute time) to use instead of the
        original timestamps.
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
        timevarname=None,
        timeoffsetvarname=None,
        timeoffsetunit=None,
        **open_dataset_kwargs,
    ):
        if isinstance(ds, (str, Path)):
            ds = xr.open_dataset(ds, **open_dataset_kwargs)
        varnames = self._maybe_add_varnames(
            varnames, [timevarname, timeoffsetvarname]
        )

        super().__init__(
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
            gridtype=gridtype,
            construct_grid=construct_grid,
            landmask=landmask,
            bbox=bbox,
            cellsize=cellsize,
        )

        if self.gridtype == "unstructured":
            self.data = ds[self.varnames]
        else:
            # we have to reshape the data
            self.orig_data = ds[self.varnames]
            self.data = self.orig_data.stack({"loc": (self.ydim, self.xdim)})
            self.locdim = "loc"

        self.timevarname = timevarname
        self.timeoffsetvarname = timeoffsetvarname
        self.timeoffsetunit = timeoffsetunit

    def read(self, *args, period=None, **kwargs) -> pd.Series:
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
            if self.gridtype == "regular":
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

        if self.gridtype == "regular":
            data = self.orig_data.sel(lat=lat, lon=lon)
        else:
            data = self.data[{self.locdim: gpi}]
        df = data.to_pandas()[self.varnames]
        df = self._modify_time(df)
        if period is not None:
            df = df.loc[period[0] : period[1]]
        return df

    def _get_time_unit(self):
        return self.data[self.timevarname].attrs["units"]

    @property
    def _ioclass(self):
        return GriddedNcOrthoMultiTs


# class ContiguousRaggedTs(_GriddedNcContiguousRaggedTs,
# _TimeseriesRepurposeMixin):
#     """
#     Reader for contiguous ragged timeseries arrays for QA4SM.

#     This is just an adaptation of the pynetcdf GriddedNcContiguousRaggedTs
#     class, and is not associated with the pynetcf ``ContiguousRaggedTs``.
#     """

#     def __init__(
#         self,
#         ds: Union[xr.Dataset, Path, str],
#         varnames: Union[str, Sequence] = None,
#         cellsize=None,
#         latname="lat",
#         lonname="lon",
#         timename="time",
#         countname="count",
#         cumulative_countname="cumulative_count",
#     ):
#         if isinstance(ds, (Path, str)):  # pragma: no cover
#             ds = xr.open_dataset(ds)
#         if varnames is None:
#             varnames = list(
#                 set(ds.data_vars)
#                 - set([latname, lonname, timename, countname,
#                 - cumulative_countname])
#             )
#         elif isinstance(varnames, str):  # pragma: no cover
#             varnames = [varnames]
#         self.varnames = list(varnames)

#         # infer the grid
#         lat = ds[latname].values
#         lon = ds[lonname].values
#         grid = BasicGrid(lon, lat)
#         if cellsize is None:  # pragma: no cover
#             cellsize = infer_cellsize(grid)
#         cellgrid = grid.to_cell_grid(cellsize=cellsize)
#         super().__init__(None, cellgrid, parameters=varnames)

#         self.timename = timename
#         self.locname = "loc"
#         self.data = ds

#     def _read_gp(self, gpi, period=None, **kwargs) -> pd.DataFrame:

#         # the main challenge is to find the start and end index of the values
#         # and the times, this is done via the cumulative count and the count
#         # variable
#         end = int(self.data["cumulative_count"][gpi])
#         start = end - int(self.data["count"][gpi])

#         time = self.data[self.timename][start:end]
#         values = np.array([self.data[p][start:end].values for p in
#         self.parameters]).T

#         df = pd.DataFrame(values, index=time, columns=self.parameters)
#         if period is not None:
#             df = df.loc[period[0] : period[1]]
#         return df

#     @property
#     def _ioclass(self):
#         return GriddedNcContiguousRaggedTs


class ZippedCsvTs(_GriddedNcContiguousRaggedTs, _TimeseriesRepurposeMixin):
    """
    Reader for zipped directory of CSV files in QA4SM format.

    Parameters
    ----------
    inputpath : path
        Path to the zip file.
    varnames : str
        Names of the variable that should be read.
    cellsize : float, optional (default: None)
        Size of cells for cell grid. If None, a heuristic will be used to
        estimate the size.
    """

    def __init__(
        self,
        inputpath: Union[Path, str],
        varnames: Union[str, Sequence] = None,
        cellsize=None,
    ):
        self.zfile = ZipFile(inputpath)

        # get coordinates
        namelist = self.zfile.namelist()
        self.fnames = list(filter(lambda s: s.endswith(".csv"), namelist))
        lats = np.empty(len(self.fnames))
        lons = np.empty(len(self.fnames))
        gpis = np.empty(len(self.fnames), dtype=int)
        self.gpi_index_map = {}
        for i, name in enumerate(self.fnames):
            match = re.findall(r".*gpi=([0-9]+).*\.csv", name)
            if len(match) != 1:
                raise ReaderError(f"Extracting gpi from filename {name} failed.")
            gpi = int(match[0])
            gpis[i] = gpi
            # mapping from gpi to index for  _read_gp
            self.gpi_index_map[gpi] = i
            match = re.findall(r".*lat=([-0-9.]+).*\.csv", name)
            if len(match) != 1:
                raise ReaderError(f"Extracting lat from filename {name} failed.")
            lats[i] = float(match[0])
            match = re.findall(r".*lon=([-0-9.]+).*\.csv", name)
            if len(match) != 1:
                raise ReaderError(f"Extracting lon from filename {name} failed.")
            lons[i] = float(match[0])

        # create grid object
        grid = BasicGrid(lons, lats, gpis=gpis)
        if cellsize is None:  # pragma: no cover
            cellsize = infer_cellsize(grid)
        cellgrid = grid.to_cell_grid(cellsize=cellsize)

        if varnames is None:
            # open first dataset to get the variable names
            df = self._read_file(self.fnames[0])
            varnames = df.columns
        elif isinstance(varnames, str):  # pragma: no cover
            varnames = [varnames]
        self.varnames = varnames

        metadata = list(filter(lambda s: s.endswith("metadata.yml"), namelist))
        if metadata:
            with self.zfile.open(metadata[0], "r") as f:
                metadata = yaml.load(f, Loader=yaml.SafeLoader)
        else:
            metadata = {v: {} for v in self.varnames}
        self.metadata = metadata

        # call super constructor (_GriddedNcContiguousRaggedTs)
        super().__init__(None, cellgrid, parameters=varnames)

    def _read_file(self, fname) -> pd.DataFrame:
        with self.zfile.open(fname) as f:
            df = pd.read_csv(f, index_col=0, parse_dates=True)
        return df

    def _read_gp(self, gpi, period=None, **kwargs) -> pd.DataFrame:
        fname = self.fnames[self.gpi_index_map[gpi]]
        df = self._read_file(fname)[self.varnames]
        if period is not None:
            df = df.loc[period[0] : period[1]]
        return df

    @property
    def _ioclass(self):
        return GriddedNcContiguousRaggedTs

    def get_metadata(self, varname: str) -> Mapping:
        return self.metadata[varname]


class TimeseriesListTs(
    _GriddedNcContiguousRaggedTs, _TimeseriesRepurposeMixin
):
    """
    Reader for list of timeseries

    Parameters
    ----------
    timeseries : list of pd.Series/pd.DataFrame
        List of the time series to use
    lat, lon : array-like
        Coordinates of time series.
    varnames : str
        Names of the variable that should be read.
    metadata : dict of dicts
        Dictionary mapping variable names in the dataset to metadata
        dictionaries, e.g. `{"sm1": {"long_name": "soil moisture 1", "units":
        "m^3/m^3"}, "sm2": {"long_name": "soil moisture 2", "units":
        "m^3/m^3"}}`.
    cellsize : float, optional (default: None)
        Size of cells for cell grid. If None, a heuristic will be used to
        estimate the size.
    """

    def __init__(
        self,
        timeseries: Sequence[Union[pd.Series, pd.DataFrame]],
        lat: Union[Sequence, np.ndarray],
        lon: Union[Sequence, np.ndarray],
        varnames: Union[str, Sequence] = None,
        metadata: Mapping[str, Mapping[str, str]] = None,
        cellsize=None,
        gpi: Union[Sequence, np.ndarray] = None,
    ):

        assert len(timeseries) > 0
        assert len(timeseries) == len(lat) == len(lon)

        # create grid object
        grid = BasicGrid(lon, lat, gpi)
        if cellsize is None:  # pragma: no cover
            cellsize = infer_cellsize(grid)
        cellgrid = grid.to_cell_grid(cellsize=cellsize)

        if gpi is None:
            gpi = np.arange(len(timeseries))
        self.timeseries = {idx: ts for idx, ts in zip(gpi, timeseries)}

        if varnames is None:
            # open first dataset to get the variable names
            ts = self.timeseries[gpi[0]]
            if isinstance(ts, pd.DataFrame):
                varnames = ts.columns
            elif ts.name is None:
                varnames = [0]
            else:
                varnames = [ts.name]
        elif isinstance(varnames, str):  # pragma: no cover
            varnames = [varnames]
        self.varnames = varnames

        if metadata is None:
            metadata = {v: {} for v in self.varnames}
        self.metadata = metadata

        # call super constructor (_GriddedNcContiguousRaggedTs)
        super().__init__(None, cellgrid, parameters=varnames)

    def _read_gp(self, gpi, period=None, **kwargs) -> pd.DataFrame:
        df = pd.DataFrame(self.timeseries[gpi])[self.varnames]
        if period is not None:
            df = df.loc[period[0] : period[1]]
        return df

    @property
    def _ioclass(self):
        return GriddedNcContiguousRaggedTs

    def get_metadata(self, varname: str) -> Mapping:
        return self.metadata[varname]
