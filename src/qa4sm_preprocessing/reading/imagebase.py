from abc import abstractmethod
import dask
import datetime
import numpy as np
from pathlib import Path
import shutil
from typing import Union, Iterable, List, Tuple, Sequence, Dict
import xarray as xr

from pygeobase.object_base import Image
from repurpose.img2ts import Img2Ts

from .exceptions import ReaderError
from .utils import mkdate, nimages_for_memory
from .base import ReaderBase
from .timeseries import GriddedNcOrthoMultiTs


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
                else:  # pragma: no cover
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
        if isinstance(timestamp, str):  # pragma: no cover
            timestamp = mkdate(timestamp)

        if timestamp not in self.timestamps:  # pragma: no cover
            raise ReaderError(
                f"Timestamp {timestamp} is not available in the dataset!"
            )

        img = self.read_block(
            timestamp, timestamp, _apply_landmask_bbox=False
        ).isel({self.timename: 0})
        img = self._stack(img)
        data = {
            varname: img[varname].values[self.grid.activegpis]
            for varname in self.varnames
        }
        metadata = self.array_attrs
        img = Image(
            self.grid.arrlon, self.grid.arrlat, data, metadata, timestamp,
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
            timevarname: str = None,
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

        Returns
        -------
        reader : GriddedNcOrthoMultiTs
            Reader for the timeseries files.
        """
        outpath = Path(outpath)
        start, end = self._validate_start_end(start, end)
        if (outpath / "grid.nc").exists() and overwrite:
            shutil.rmtree(outpath)
        if not (outpath / "grid.nc").exists():  # if overwrite=True, it was deleted now
            outpath.mkdir(exist_ok=True, parents=True)
            testimg = self._testimg()
            n = nimages_for_memory(testimg, memory)
            if hasattr(self, "use_tqdm"): # pragma: no branch
                orig_tqdm = self.use_tqdm
                self.use_tqdm = False
            reshuffler = Img2Ts(
                self,
                str(outpath),
                start,
                end,
                cellsize_lat=self.cellsize,
                cellsize_lon=self.cellsize,
                ts_attributes=self.array_attrs,
                global_attr=self.global_attrs,
                zlib=True,
                imgbuffer=n,
            )
            reshuffler.calc()
            if hasattr(self, "use_tqdm"):  # pragma: no branch
                self.use_tqdm = orig_tqdm
        reader = GriddedNcOrthoMultiTs(str(outpath), timevarname=timevarname, read_bulk=True)
        return reader
