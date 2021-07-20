# -*- coding: utf-8 -*-
from pygeobase.io_base import ImageBase, MultiTemporalImageBase
from netCDF4 import Dataset, num2date
import datetime
from pygeogrids.grids import CellGrid, gridfromdims
import os
import numpy as np
from typing import Union, Optional
import warnings
from pygeobase.object_base import Image

class S1Cgls1kmImage(ImageBase):
    """
    Reading a single S1 CGLS SSM or SWI file.
    """

    def __init__(self,
                 filename,
                 parameters=None,
                 grid=None,
                 flatten=False,
                 fillval=None,
                 mode='r'):
        """
        Read a single CGLS 1km  (SSM and SWI) image

        Parameters
        ----------
        filename: str
            Path to the image netcdf file to read
        parameters: str or list[str], optional (default: None)
            List of parameters to read from netcdf file. If None is passed
            all variables (except dimension variables are read)
        grid : pygeogrids.CellGrid
            Grid object that contains points to read, if None is passed,
            a grid is generated from the image data, and all points are read.
        flatten: bool, optional (default: False)
            Flatten the input image to a 1d data array, instead of reading
            the original 2d array. This is used e.g. when reshuffling to time
            series.
        fillval: dict or float, optional (default: None)
            Set a custom fill value for a variable (in a case a dict is passed
            the variable name is the key and the fill value the value), or use
            one fill value for all variables. None to use the original fill val.
            Note: When the dtype of the fill value does not match the dtype
             of the variable, the variable dtype changes!
        mode: str, optional (default: 'r')
            File access mode. Not used. Do not change.
            Expected when using this class as the ioclass.
        """

        self.path = os.path.dirname(filename)
        self.fname = os.path.basename(filename)


        super(S1Cgls1kmImage, self).__init__(
            filename=os.path.join(self.path, self.fname), mode=mode)

        parameters = np.atleast_1d(parameters if parameters is not None else [])
        self.parameters = parameters

        self.grid = grid
        self.flatten = flatten

        # set this after reading the image:
        self.img = None
        self.glob_attrs = None

        if isinstance(fillval, dict):
            self.fillval = fillval
            for p in self.parameters:
                if p not in self.fillval:
                    self.fillval[p] = None
        else:
            self.fillval = {p: fillval for p in self.parameters}

    @staticmethod
    def _ds_gen_grid(ds) -> CellGrid:
        grid = gridfromdims(londim=ds.variables['lon'][:],
                            latdim=ds.variables['lat'][:],
                            origin='bottom').to_cell_grid(5.)
        return grid

    @staticmethod
    def ds_gen_timestamp(ds) -> datetime.datetime:
        timestamp = num2date(ds['time'], ds['time'].units,
                             only_use_cftime_datetimes=True,
                             only_use_python_datetimes=False)
        assert len(timestamp) == 1, "Found more than 1 time stamps in image"
        timestamp = timestamp[0]

        return timestamp

    def read(self, timestamp: Optional[datetime.datetime]=None) -> Image:
        """
        Read data as 1d or 2d image. In case a 2d image is loaded, to image is
        NOT upside down! In case an 1d image image is read, the image is
        first flipped so that after flattening, gpis are stored in ascending
        order.
        """
        param_img, metadata, global_attrs, timestamp = \
            self._read_flat_img(timestamp)

        metadata['nc_global_attr'] = global_attrs

        if self.flatten:
            return Image(self.grid.activearrlon,
                         self.grid.activearrlat,
                         param_img,
                         metadata=metadata,
                         timestamp=timestamp)
        else:
            return Image(lon=np.flipud(self.grid.activearrlon.reshape(self.grid.shape)),
                         lat=np.flipud(self.grid.activearrlat.reshape(self.grid.shape)),
                         data={k: np.flipud(v.reshape(self.grid.shape)) for k, v in param_img.items()},
                         metadata=metadata,
                         timestamp=timestamp)


    def _read_flat_img(self, timestamp=None) -> (dict, dict, dict, datetime.datetime):
        """
        Reads a single C3S image, flat with gpi0 as first element
        """
        with Dataset(self.filename, mode='r') as ds:

            if self.grid is None:
                self.grid = self._ds_gen_grid(ds)

            if timestamp is None:
                timestamp = self.ds_gen_timestamp(ds)

            param_img = {}
            param_meta = {}

            if len(self.parameters) == 0:
                ignore_params = list(ds.dimensions.keys()) + ['crs']
                # all data vars, exclude coord vars
                self.parameters = [k for k in ds.variables.keys()
                                   if k not in ignore_params]

            parameters = list(self.parameters)

            for parameter in parameters:
                metadata = {}
                param = ds.variables[parameter]
                data = param[:][0] # there is only 1 time stamp in the image

                self.shape = (data.shape[0], data.shape[1])

                # read long name, FillValue and unit
                for attr in param.ncattrs():
                    metadata[attr] = param.getncattr(attr)

                if parameter in self.fillval:
                    if self.fillval[parameter] is None:
                        self.fillval[parameter] = data.fill_value

                    common_dtype = np.find_common_type(
                        array_types=[data.dtype],
                        scalar_types=[type(self.fillval[parameter])])
                    self.fillval[parameter] = np.array([self.fillval[parameter]],
                                                       dtype=common_dtype)[0]

                    data = data.astype(common_dtype)
                    data = data.filled(self.fillval[parameter])
                else:
                    self.fillval[parameter] = data.fill_value
                    data = data.filled()

                data = np.flipud(data)
                data = data.flatten()

                metadata['image_missing'] = 0

                param_img[parameter] = data[self.grid.activegpis]
                param_meta[parameter] = metadata

            global_attrs = ds.__dict__

        global_attrs['timestamp'] = str(timestamp)

        return param_img, param_meta, global_attrs, timestamp


class S1Cgls1kmDs(MultiTemporalImageBase):

    def __init__(self,
                 data_path: str,
                 parameters: Union[str, list],
                 grid: CellGrid = None,
                 fname_templ: str ="c_gls_SSM1km_{datetime}_CEURO_S1CSAR_V*.nc",
                 datetime_format: str = "%Y%m%d%H%M",
                 hours=(0,),
                 subpath_templ: list = None,
                 flatten: bool = False,
                 fillval: dict = None):
        """
        Parameters
        ----------
        data_path : str
            Path to folder where netcdf image files are stored.
        parameters: str or list
            Name of Paramter(s) to read from image files
        grid : CellGrid, optional (default: None)
            Grid with locations to read data for. If None is passed,
            the grid is generated from the first
        """

        if '{datetime}' not in fname_templ:
            warnings.warn("Filename template does not contain a placeholder "
                          "'{datetime}' to parse date")

        self.hours = hours

        ioclass_kws = {'parameters': parameters,
                       'grid': grid,
                       'fillval': fillval,
                       'flatten': flatten}

        super(S1Cgls1kmDs, self).__init__(data_path,
                                          S1Cgls1kmImage,
                                          fname_templ=fname_templ,
                                          datetime_format=datetime_format,
                                          subpath_templ=subpath_templ,
                                          exact_templ=False,
                                          ioclass_kws=ioclass_kws)
    
    def read(self, timestamp, **kwargs):
        return super(S1Cgls1kmDs, self).read(timestamp, **kwargs)

    def tstamps_for_daterange(self, start_date, end_date):
        """
        return timestamps for daterange,
        Parameters
        ----------
        start_date: datetime.datetime
            start of date range
        end_date: datetime.datetime
            end of date range

        Returns
        -------
        timestamps : list
            list of datetime objects of each available image between
            start_date and end_date
        """
        img_offsets = np.array([datetime.timedelta(hours=h) for h in self.hours])

        timestamps = []
        start_date = datetime.datetime(start_date.year, start_date.month, start_date.day)
        end_date = datetime.datetime(end_date.year, end_date.month, end_date.day)
        diff = end_date - start_date

        for i in range(diff.days + 1):
            daily_dates = start_date \
                          + datetime.timedelta(days=i) \
                          + img_offsets
            timestamps.extend(daily_dates.tolist())

        return timestamps

if __name__ == '__main__':

    input = "/home/wpreimes/shares/radar/Datapool/CGLS/01_raw/SWI1km/v1.0/product/"
    ds = S1Cgls1kmDs(input, flatten=False,
                     fname_templ="c_gls_SWI1km_{datetime}_CEURO_SCATSAR_V*.nc",
                     parameters=['SWI_005', 'SWI_040'])
    img = ds.read(datetime(2015,1,9, 12))