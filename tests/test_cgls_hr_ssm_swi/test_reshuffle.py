import os
import tempfile
import datetime
from qa4sm_preprocessing.cgls_hr_ssm_swi.reader import S1CglsTs
import numpy as np

from qa4sm_preprocessing.cgls_hr_ssm_swi.reshuffle import reshuffle

testdata_path = os.path.join(os.path.dirname(__file__), "..", "test-data", "preprocessing")

def test_reshuffle_swi():
    # here we actually re-define the time stamps, 0:00, not 12:00 as in the
    # original images.
    startdate = datetime.datetime(2017,6,1,12)
    enddate = datetime.datetime(2017,6,3,12)

    in_path = os.path.join(testdata_path, "CGLS_SWI1km_V1.0_img")

    with tempfile.TemporaryDirectory() as out_path:
        reshuffle(in_path,
                  out_path,
                  startdate=startdate,
                  enddate=enddate,
                  parameters=None,
                  bbox=[44.90625, 44.92411, -0.969, -0.9597],
                  # Note: we could also include time in datetime, change datetime format
                  #  and set hours to (12,), to resample the actual time stamps
                  #  in the images.
                  fname_templ="c_gls_SWI1km_{datetime}1200_CEURO_SCATSAR_V*.nc",
                  hours=(0,),
                  datetime_format = "%Y%m%d")
        assert len(os.listdir(out_path)) == 2
        ds = S1CglsTs(out_path, parameters='SWI_005')
        assert ds.grid.get_grid_points()[0].size == 4
        ts = ds.read(-0.95982, 44.9151)
        assert ts.loc['2017-06-02', 'SWI_005'] == 53.5

        # read all points
        ts_param = ds.read(-0.95982, 44.9151)
        ts_point = ds.read_area(-0.95982, 44.9151, radius=0)
        np.testing.assert_equal(ts_point.values,
                      ds.read_area(-0.95982, 44.9151, area=None).values)
        np.testing.assert_equal(ts_point.values, ts_param.values)

        ts_multi = ds.read_area(-0.962, 44.918, radius=10000, area='circle', average=False)
        assert len(ts_multi.columns) == 4
        ts_average = ds.read_area(-0.962, 44.918, radius=10000, area='circle', average=True)
        np.testing.assert_equal(ts_multi.mean(axis=1).values, ts_average['SWI_005'].values)

        # test case when there are no points in the radius
        empty_df = ds.read_area(-0.9, 45, radius=100, area='circle', average=False)
        also_empty_df = ds.read_area(-0.9, 45, radius=100, area='circle', average=True)
        assert all([empty_df.empty, also_empty_df.empty])




def test_reshuffle_ssm():
    startdate = datetime.datetime(2017,6,1)
    enddate = datetime.datetime(2017,6,3)

    in_path = os.path.join(testdata_path, "CGLS_SSM1km_V1.1_img")

    with tempfile.TemporaryDirectory() as out_path:
        reshuffle(in_path, out_path, startdate, enddate,
                  parameters=['ssm'],
                  bbox=[44.90625, 44.92411, -0.969, -0.9597],
                  fname_templ="c_gls_SSM1km_{datetime}_CEURO_S1CSAR_V*.nc",
                  hours=(0,),
                  datetime_format = "%Y%m%d%H%M")
        assert len(os.listdir(out_path)) == 2
        ds = S1CglsTs(out_path, parameters='ssm')
        assert ds.grid.get_grid_points()[0].size == 4
        ts = ds.read(-0.95982, 44.9151)
        assert ts.loc['2017-06-03', 'ssm'] == 57.5

        # read all points
        ts_param = ds.read(-0.95982, 44.9151)
        ts_point = ds.read_area(-0.95982, 44.9151, radius=0)
        np.testing.assert_equal(
            ts_point.values, ds.read_area(-0.95982, 44.9151, area=None).values)
        np.testing.assert_equal(ts_point.values, ts_param.values)

        ts_multi = ds.read_area(-0.962, 44.918, radius=10000, area='circle', average=False)
        assert len(ts_multi.columns) == 4
        ts_average = ds.read_area(-0.962, 44.918, radius=10000, area='circle', average=True)
        np.testing.assert_equal(ts_multi.mean(axis=1).values,
                                ts_average['ssm'].values)

        # test case when there are no points in the radius
        empty_df = ds.read_area(-0.9, 45, radius=100, area='circle', average=False)
        also_empty_df = ds.read_area(-0.9, 45, radius=100, area='circle', average=True)
        assert all([empty_df.empty, also_empty_df.empty])
