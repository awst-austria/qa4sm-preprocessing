# -*- coding: utf-8 -*-
import warnings

import pygeogrids
from netCDF4 import Dataset
from datetime import  timedelta
from pygeogrids.netcdf import load_grid
import os
import numpy as np
from pynetcf.time_series import GriddedNcOrthoMultiTs
import pandas as pd

"""
TODO:
At the moment the average Ts reader can only read a single variable.
In case that HR SSM/SWI time series should be flagged, we have to
    - allow reading multiple parameters here at once
    - probably do the masking outside of the pytesmo flagging adapter,
      because we want to give pytesmo the averaged time series already?
"""

class S1CglsTs(GriddedNcOrthoMultiTs):
    """
    Read CGLS SSM and SWI Time Series from reshuffled images.
    """

    def __init__(self, ts_path, parameters,
                 grid_path=None):
        """
        Parameters
        ----------
        ts_path : str
            Path to reshuffled time series netcdf files
        cache_surrounding_cells : bool, optional (default: True)
            This means that the up to 4 cells around a point will be kept in
            the cache. This can require a lot of memory (up to ~12 GB) but makes
            reading faster.
        parameters : str, optional (default: None)
            Parameter(s) to read from files. None reads all parameters.
        grid_path : str, optional (default: None)
            Path to the grid.nc file, if None is passed, grid.nc is searched
            in ts_path.
        """
        if grid_path is None:
            grid_path = os.path.join(ts_path, "grid.nc")

        grid = load_grid(grid_path, kd_tree_name='scipy')

        if not isinstance(parameters, str):
            raise NotImplementedError("Currently it is not possible to read "
                                      "more than 1 parameter")

        kwargs = {'ioclass_kws' : {'read_bulk': True}, 'parameters': parameters}
        super(S1CglsTs, self).__init__(ts_path, grid, **kwargs)

        self.grid: pygeogrids.CellGrid

        self.parameters = [parameters]

        # stores data for up to 6 cells, cell number as key
        # data as dataframe with gpis as columns for the param
        # not used when the read  function from the parent
        # class is used (when HR grid is reference)
        self.celldata = {}

    def _read_cell(self, cell, drop_nan_ts=True) -> pd.DataFrame:
        """
        Read all time series for a single variable in the selected cell.

        Parameters
        -------
        cell: int or list,
            Cell number(s) as in the cgls grid
        drop_nan_ts : bool, optional (default: True)
            Drop time series in cell were param is always nan
        """
        if cell not in self.celldata.keys():

            if len(self.celldata) >= 6: #  make some space in buffer
                self.celldata.pop(min(self.celldata.keys()))

            file_path = os.path.join(self.path, '{}.nc'.format("%04d" % (cell,)))
            with Dataset(file_path) as ncfile:
                loc_id = ncfile.variables['location_id'][:]
                time = ncfile.variables['time'][:]
                unit_time = ncfile.variables['time'].units
                delta = lambda t: timedelta(t)
                vfunc = np.vectorize(delta)
                since = pd.Timestamp(unit_time.split('since ')[1])
                time = since + vfunc(time)

                variable = ncfile.variables[self.parameters[0]][:]
                variable = np.transpose(variable)
                data = pd.DataFrame(variable, columns=loc_id, index=time)

            self.celldata[cell] = data

        if drop_nan_ts:
            return self.celldata[cell].dropna(how='all', axis=1)
        else:
            return self.celldata[cell]

    def _read_gps(self, gpis, applyf=None, **applyf_kwargs) -> pd.DataFrame:
        """
        Read multiple time series at once. Can also apply a function along
        the gpi time series after reading, e.g. to average them for an area.

        Parameters
        ----------
        gpis : np.array
            Array of gpis to read. Should be spatially close to avoid loading
            cell data unnecessarily.
        applyf : Callable, optional (default: None)
            Applied along axis=1 to merge multiple points, e.g. pd.DataFrame.mean
        **applyf_kwargs:
            Passed to applyf, e.g skipna=False mak

        Returns
        -------
        df : pd.DataFrame
            Contains the data, without applyf this will contain many columns.
            Otherwise whatever applyf returns.
        """

        gpis = sorted(gpis)

        if len(gpis) > 0:
            cells = np.unique(self.grid.gpi2cell(gpis))
        else:
            cells = np.array([])

        if len(cells) > 3:
            warnings.warn('Reading needs data from more than 3 cells!')

        data = []
        for cell in cells:
            celldata = self._read_cell(cell, drop_nan_ts=True)
            data.append(celldata[np.intersect1d(celldata.columns, gpis)])

        if len(data) == 0:
            data = pd.DataFrame(columns=self.parameters)
        else:
            data = pd.concat(data, axis=1)

        if applyf is not None and not data.empty:
            data = data.apply(applyf, axis=1, **applyf_kwargs)
            data = data.to_frame(name=self.parameters[0])
            data = pd.DataFrame(data)

        return data

    def read_area(self,
                  *args,
                  radius=10000,
                  area='circle',
                  average=False):
        """
        Read all points around a location (gpi or lon,lat)

        Parameters
        ----------
        *args: Used to read a single point.
        radius : float or None, optional (default: 0.1)
            Radius AROUND the passed coords, for a circle in M, for a square in DEG
            # todo: make it the same for both shapes
        area: Literal["square", "circle"] or None, optional (default: 'square')
            The shape of the area that radius defines to read.
        average: bool, optional (default: False)
            If selected, then all points are averaged via pd.DataFrame.mean
            and a single time series is returned otherwise the time series
            for all points in the area are returned.
        """
        if len(args) == 1:
            lon, lat = self.grid.gpi2lonlat(*args)
        elif len(args) == 2:
            lon, lat = args
        else:
            raise ValueError("Wrong number of args passed.")

        if (radius is None) or (radius == 0) or (area is None):
            gpi, d = self.grid.find_nearest_gpi(lon, lat)
            cell = self.grid.gpi2cell(gpi)
            try:
                if cell in self.celldata.keys():
                    # single point from internal cache
                    return self.celldata[cell][[gpi]].rename(columns={gpi: self.parameters[0]})
                else:
                    # single point when no cache yet exists
                    return self._read_gps([gpi]).rename(columns={gpi: self.parameters[0]})
            except KeyError:
                return self.read(*args)
        if area.lower() == 'square':
            gpis = self.grid.get_bbox_grid_points(lat-radius,
                                                  lat+radius,
                                                  lon-radius,
                                                  lon+radius)
        elif area.lower() == 'circle':
            # if self.grid.kdTree is None:
            #     self.grid._setup_kdtree()
            try:
                # find all points within radius
                gpis, dist  = self.grid.find_k_nearest_gpi(
                    lon, lat, max_dist=radius, k=None)
            except ValueError:  # when no points were found in radius
                gpis = np.array([])
        else:
            raise NotImplementedError(f"{area} is not supported.")

        if average:
            return self._read_gps(gpis, applyf=pd.DataFrame.mean, skipna=True)
        else:
            return self._read_gps(gpis)
