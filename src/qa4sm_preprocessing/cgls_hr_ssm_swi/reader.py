# -*- coding: utf-8 -*-
import warnings

import pygeogrids
from pygeogrids.netcdf import load_grid
import os
from pynetcf.time_series import GriddedNcOrthoMultiTs

class S1CglsTs(GriddedNcOrthoMultiTs):
    """
    Read CGLS SSM and SWI Time Series from reshuffled images.
    """

    def __init__(self, ts_path, parameters=None, grid_path=None):
        """
        Parameters
        ----------
        ts_path : str
            Path to reshuffled time series netcdf files
        parameters : str, optional (default: None)
            Parameter(s) to read from files. None reads all parameters.
        grid_path : str, optional (default: None)
            Path to the grid.nc file, if None is passed, grid.nc is searched
            in ts_path.
        """
        if grid_path is None:
            grid_path = os.path.join(ts_path, "grid.nc")

        grid = load_grid(grid_path, kd_tree_name='scipy')

        kwargs = {'ioclass_kws' : {'read_bulk': True}}

        if parameters is not None:
            kwargs['parameters'] = parameters

        super(S1CglsTs, self).__init__(ts_path, grid, **kwargs)

        self.grid: pygeogrids.CellGrid

