
from pynetcf.time_series import GriddedNcOrthoMultiTs
from pygeogrids.netcdf import load_grid
import os
import numpy as np

path_new = "/home/wolfgang/data-read/temp/smos_sbpca/l2/ts/"
path_old = "/home/wolfgang/data-read/temp/smos_sbpca/old_l2_ts/"

reader_new = GriddedNcOrthoMultiTs(path_new, ioclass_kws={'read_bulk': True},
                                   grid=load_grid(os.path.join(path_new, 'grid.nc')))
reader_old = GriddedNcOrthoMultiTs(path_old, ioclass_kws={'read_bulk': True},
                                   grid=load_grid(os.path.join(path_old, 'grid.nc')))

lon, lat = 130.085, -27.459

ts_old = reader_old.read(lon, lat).loc['2012-01-02':'2012-01-02T23:59']
ts_new = reader_new.read(lon, lat)

# i = np.random.choice(np.arange(357))
# for i in range(357):
#     _, lons, lats = reader_old.grid.grid_points_for_cell(8953)
#     ts_old = reader_old.read(lons[i], lats[i]).loc['2012-01-02':'2012-01-02T23:59', "Soil_Moisture"].dropna()
#     if not ts_old.empty:
#         print(lons[i], lats[i])
#         break
