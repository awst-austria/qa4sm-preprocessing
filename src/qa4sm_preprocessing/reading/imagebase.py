from abc import abstractmethod
import dask
import dask.array as da
import datetime
import logging
import numpy as np
from pathlib import Path
import shutil
from typing import Union, List, Tuple, Dict
import xarray as xr

from pygeobase.object_base import Image
from repurpose.img2ts import Img2Ts
from repurpose.process import ImageBaseConnection

from .exceptions import ReaderError
from .utils import mkdate, nimages_for_memory, numpy_timeoffsetunit
from .base import ReaderBase
from .timeseries import GriddedNcOrthoMultiTs
from pynetcf.time_series import GriddedNcIndexedRaggedTs
from pygeogrids.netcdf import load_grid


class ImageReaderBase(ReaderBase):
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
        elif isinstance(start, str):  # pragma: no cover
            start = mkdate(start)
        if end is None:
            end = self.timestamps[-1]
        elif isinstance(end, str):  # pragma: no cover
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

    @property
    def imgshape(self):
        if self.gridtype == "regular":
            return (len(self.lat), len(self.lon))
        else:
            # for curvilinear and unstructured grids, the latitude has the same
            # shape as the data
            return self.lat.shape

    @property
    def imgndim(self):
        if self.gridtype == "unstructured":
            return 1
        else:
            return 2

    def get_blockshape(self, ntime):
        return tuple([ntime] + list(self.imgshape))

    def _empty_blockdict(self, ntime):
        shape = self.get_blockshape(ntime)
        return {v: np.empty(shape, dtype=self.dtype[v]) for v in self.varnames}

    def _fix_ndim(self, arr: np.ndarray) -> np.ndarray:
        if arr.ndim == self.imgndim:
            # we need to add a time axis
            arr = arr[np.newaxis, ...]
        elif arr.ndim >= self.imgndim + 2:  # pragma: no cover
            raise ReaderError(
                f"Data with shape {arr.shape} has wrong number of dimensions"
            )
        return arr

    def get_dims(self):
        if self.gridtype == "regular":
            dims = (self.timename, self.latname, self.lonname)
        elif self.gridtype == "curvilinear":
            dims = (self.timename, self.ydim, self.xdim)
        else:  # unstructured grid
            dims = (self.timename, self.locdim)
        return dims

    def get_coords(self, times):
        coords = {}
        coords[self.timename] = times
        if self.gridtype == "regular":
            # we can simply wrap the data with time, lat, and lon
            coords[self.latname] = self.lat
            coords[self.lonname] = self.lon
        elif self.gridtype == "curvilinear":
            coords[self.latname] = ([self.ydim, self.xdim], self.lat)
            coords[self.lonname] = ([self.ydim, self.xdim], self.lon)
        else:  # unstructured grid
            coords[self.latname] = (self.locdim, self.lat.data)
            coords[self.lonname] = (self.locdim, self.lon.data)
        return coords

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
        array_attrs = self.array_attrs.copy()

        # if there is a time offset variable we will convert it to absolute
        # times relative to 1970
        if self.timeoffsetvarname is not None:
            vname = self.timeoffsetvarname
            values = block_dict[vname]
            exact_time = np.empty_like(values, dtype=float)
            for i, t in enumerate(times):
                offset = values[i, ...]
                np_unit = numpy_timeoffsetunit(self.timeoffsetunit)
                delta = offset.astype(f"timedelta64[{np_unit}]")
                # seconds hard-coded here, might be relaxed in the future
                start = np.datetime64(t, "s")
                exact_time[i, ...] = (start + delta).astype(float)
            block_dict[vname] = exact_time
            array_attrs[vname].update(
                {
                    "units": "seconds since 1970-01-01",
                    "description": "Exact acquisition time",
                }
            )

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
                else:  # pragma: no cover
                    raise ReaderError("Unknown array type in read_block.")
                block_dict[var] = masked_array(block_dict[var], mask=mask)

        # Now we have to set the coordinates/dimensions correctly.  This works
        # differently depending on how the original data is structured:
        coords = self.get_coords(times)
        dims = self.get_dims()
        arrays = {
            name: (dims, data, array_attrs[name]) for name, data in block_dict.items()
        }
        block = xr.Dataset(arrays, coords=coords, attrs=self.global_attrs)

        # add some nice CF convention attributes to the coordinates
        block[self.latname].attrs.update(
            {"units": "degrees_north", "standard_name": "latitude"}
        )
        block[self.lonname].attrs.update(
            {"units": "degrees_east", "standard_name": "longitude"}
        )
        block[self.timename].attrs.update({"standard_name": "time"})

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
        if isinstance(timestamp, str):  # pragma: no cover
            timestamp = mkdate(timestamp)

        if timestamp not in self.timestamps:  # pragma: no cover
            raise ReaderError(f"Timestamp {timestamp} is not available in the dataset!")

        img = self.read_block(timestamp, timestamp, _apply_landmask_bbox=False).isel(
            {self.timename: 0}
        )
        img = self._stack(img)

        if "parameter" in kwargs:
            varnames = kwargs["parameter"]
        else:
            varnames = self.varnames

        data = {}
        for varname in varnames:
            var = img[varname].values
            if len(var) == len(self.grid.activegpis):
                data[varname] = var
            else:
                data[varname] = var[self.grid.activegpis]
        metadata = {varname: img[varname].attrs.copy() for varname in varnames}

        img = Image(
            self.grid.arrlon,
            self.grid.arrlat,
            data,
            metadata,
            timestamp,
            timekey=self.timekey
        )

        return img

    def _testimg(self):
        if hasattr(self, "use_tqdm"):
            orig_tqdm = self.use_tqdm
            self.use_tqdm = False
        img = self.read_block(self.timestamps[0], self.timestamps[0])
        if hasattr(self, "use_tqdm"):
            self.use_tqdm = orig_tqdm
        return img

    def repurpose(
        self,
        outpath: Union[Path, str],
        start: Union[datetime.datetime, str] = None,
        end: Union[datetime.datetime, str] = None,
        overwrite: bool = False,
        memory: float = 2,
        drop_crs: bool = True,
        n_proc=1,
        cellsize=None,
        target_grid=None,
        img2ts_kwargs=None,
        imgbaseconnection=False,
        append=False,
        **reader_kwargs,
    ):
        """
        Transforms the netCDF stack to the pynetcf timeseries format.

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
        drop_crs : bool, optional (default: True)
            Some datasets have coordinate reference system info attached as a
            string variable. During repurposing, this can strongly grow in
            size. Therefore, by default, if a variable called "crs" is
            encountered that has a string dtype, it is dropped.
        cellsize: float, optional (default: None)
            Size of the cells to split time series into (affects the file size).
            If None, then automatically determined cellsize is used.
        n_proc: int, optional (default: 1)
            Number of parallel processes to use
        target_grid: Cellgrid, optional (default: None)
            To process a spatial subset pass a subgrid of the image grid
            here.
        img2ts_kwargs: dict, optional (default: None)
            additional keyword arguments to pass to Img2Ts
        imgbaseconnection: bool, optional (default: False)
            Apply wrapper to repeatedly try adding data in case a file is not
            accessible.
        append: bool, optional (default: False)
            If a time series dataset already exists, append new data.
        **reader_kwargs: additional keyword arguments for GriddedNcOrthoMultiTs

        Returns
        -------
        reader : GriddedNcOrthoMultiTs or GriddedNcIndexedRaggedTs or None
            Reader for the timeseries files or None if no reshuffling was
            performed
        """
        if overwrite and append:
            raise ValueError("You can not select both `overwrite` and `append` "
                             "at the same time")
        outpath = Path(outpath)
        start, end = self._validate_start_end(start, end)
        if (outpath / "grid.nc").exists() and overwrite:
            shutil.rmtree(outpath)

        # if overwrite=True, it was deleted now, otherwise append
        if (not (outpath / "grid.nc").exists()) or append:
            outpath.mkdir(exist_ok=True, parents=True)
            testimg = self._testimg()
            if (
                drop_crs
                and "crs" in self.varnames
                and testimg.crs.dtype.type is np.bytes_
            ):
                varnames = [v for v in self.varnames if v != "crs"]
            else:
                varnames = self.varnames

            array_attrs = {v: testimg[v].attrs.copy() for v in varnames}

            time_units = None
            if self.timekey in array_attrs:
                time_attrs = array_attrs.pop(self.timekey)
                if 'units' in time_attrs:
                    time_units = time_attrs['units']

            img2ts_kwargs = img2ts_kwargs or dict()
            if time_units is not None:
                img2ts_kwargs['time_units'] = time_units

            n = nimages_for_memory(testimg, memory)
            logging.info(f"Reading {n} images at once.")
            if hasattr(self, "use_tqdm"):  # pragma: no branch
                orig_tqdm = self.use_tqdm
                self.use_tqdm = False

            if img2ts_kwargs is None:
                img2ts_kwargs = dict()

            if imgbaseconnection:
                reader = ImageBaseConnection(self, attr_read='read',
                                             attr_path='directory')
            else:
                reader = self

            reshuffler = Img2Ts(
                reader,
                str(outpath),
                start,
                end,
                input_grid=self.grid,
                target_grid=target_grid,
                input_kwargs={"parameter": varnames},
                ts_attributes=array_attrs,
                cellsize_lat=self.cellsize if cellsize is None else cellsize,
                cellsize_lon=self.cellsize if cellsize is None else cellsize,
                global_attr=self.global_attrs,
                zlib=True,
                imgbuffer=n,
                n_proc=n_proc,
                **img2ts_kwargs
            )
            reshuffler.calc()
            if hasattr(self, "use_tqdm"):  # pragma: no branch
                self.use_tqdm = orig_tqdm
        else:
            reshuffler = None
            logging.info(f"Output path already exists: {str(outpath)}")

        if reshuffler is not None:
            if reshuffler.orthogonal:
                reader = GriddedNcOrthoMultiTs(str(outpath), **reader_kwargs)

            else:
                grid = load_grid(str(outpath / 'grid.nc'))
                reader = GriddedNcIndexedRaggedTs(str(outpath), grid=grid,
                                                  **reader_kwargs)
        else:
            reader = None

        return reader