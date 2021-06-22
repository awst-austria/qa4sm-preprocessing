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

class S1CglsTs(GriddedNcOrthoMultiTs):
    """
    Read CGLS SSM and SWI Time Series from reshuffled images.
    """

    def __init__(self, ts_path,
                 parameter='ssm',
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
        parameter : str, optional (default: 'ssm')
            Parameters to read from files.
        grid_path : str, optional (default: None)
            Path to the grid.nc file, if None is passed, grid.nc is searched
            in ts_path.
        """
        if grid_path is None:
            grid_path = os.path.join(ts_path, "grid.nc")

        grid = load_grid(grid_path, kd_tree_name='scipy')

        kwargs = {'ioclass_kws' : {'read_bulk': True}, 'parameters': [parameter]}
        super(S1CglsTs, self).__init__(ts_path, grid, **kwargs)

        self.grid: pygeogrids.CellGrid

        self.parameter = parameter
        self.celldata = {}  # stores data for up to 6 cells, cell number as key
                            # data as dataframe with gpis as columns for the param

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

                variable = ncfile.variables[self.parameter][:]
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

        cells = np.unique(self.grid.gpi2cell(gpis))

        if len(np.unique(cells)) > 3:
            warnings.warn('Reading needs data from more than 3 cells!')

        data = []
        for cell in cells:
            celldata = self._read_cell(cell, drop_nan_ts=True)
            data.append(celldata[np.intersect1d(celldata.columns, gpis)])

        data = pd.concat(data, axis=1)

        if applyf is not None:
            data = data.apply(applyf, axis=1, **applyf_kwargs)
            data = data.to_frame(name=self.parameter)
            data = pd.DataFrame(data)

        return data

    def read_area(self,
                  *args,
                  radius=1000,
                  area='circle',
                  average=False):
        """
        Read all points around a location (gpi or lon,lat)

        Parameters
        ----------
        *args: Used to read a single point.
        radius : float, optional (default: 0.1)
            Radius AROUND the passed coords, for a circle in M, for a square in DEG
            # todo: make it the same for both shapes
        area: Literal["square", "circle"], optional (default: 'square')
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

        if radius == 0:
            gpi, d = self.grid.find_nearest_gpi(lon, lat)
            cell = self.grid.gpi2cell(gpi)
            try:
                if cell in self.celldata.keys():
                    return self.celldata[cell][[gpi]]
            except KeyError:
                return self.read(*args).rename(columns={self.parameter: gpi})

        if area.lower() == 'square':
            gpis = self.grid.get_bbox_grid_points(lat-radius,
                                                  lat+radius,
                                                  lon-radius,
                                                  lon+radius)
        elif area.lower() == 'circle':
            # if self.grid.kdTree is None:
            #     self.grid._setup_kdtree()
            gpis, dist  = self.grid.find_k_nearest_gpi(
                lon, lat, max_dist=radius, k=None)
        else:
            raise NotImplementedError(f"{area} is not supported.")

        if average:
            return self._read_gps(gpis, applyf=pd.DataFrame.mean, skipna=True)
        else:
            return self._read_gps(gpis)


if __name__ == '__main__':

    import matplotlib.pyplot as plt
    import cartopy.crs as ccrs

    from io_utils.plot.plot_maps import cp_scatter_map
    import time
    t0 = time.time()
    path = "/home/wpreimes/shares/radar/Projects/QA4SM_HR/07_data/CGLS_SSM1km_V1.1_ts/"
    cgls = S1CglsTs(path)

    ts = cgls.read_area(15,48, radius=0)

    fig, axs = plt.subplots(1,1, figsize=(15,15), subplot_kw={'projection': ccrs.Robinson()})
    import pandas as pd
    from io_utils.plot.plot_maps import cp_scatter_map
    df = pd.read_csv("/home/wpreimes/Temp/ismndata/hrplot.csv", index_col=00)
    r_lons, r_lats = cgls.grid.gpi2lonlat(df.index)

    llc= [v-v*0.1 for v in [ r_lons.min(), r_lats.min()]]
    urc= [v+v*0.1 for v in [ r_lons.max(), r_lats.max()]]
    cp_scatter_map(r_lons, r_lats, df['ISMN'].values, llc=llc, urc=urc, cbrange=(-1,1), imax=axs)


    #
    # ts = cgls.read_area(2.875, 45.125, area="circle", radius=10000)
    #
    # r_vals = ts.corr()[ts.columns[0]]
    # r_lons, r_lats = cgls.grid.gpi2lonlat(r_vals.columns)
    #
    # llc= [v*1.1 for v in [ r_lons.min(), r_lats.min()]]
    # urc= [v*1.1 for v in [ r_lons.max(), r_lats.max()]]
    # fig, imax, im = cp_scatter_map(r_lons, r_lats, r_vals, llc=llc, urc=urc)
    #
    # print(f"--- {time.time() - t0} seconds ---")


