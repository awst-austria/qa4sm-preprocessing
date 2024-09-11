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
        assert len(os.listdir(out_path)) == 3
        ds = S1CglsTs(out_path, parameters='SWI_005')
        assert ds.grid.get_grid_points()[0].size == 4
        ts = ds.read(-0.95982, 44.9151)
        assert ts.columns == ['SWI_005']
        assert ts.loc['2017-06-02', 'SWI_005'] == 53.5


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
        assert len(os.listdir(out_path)) == 3
        params = ['ssm']
        ds = S1CglsTs(out_path, parameters=params)
        assert ds.grid.get_grid_points()[0].size == 4
        ts = ds.read(-0.95982, 44.9151)
        assert sorted(ts.columns) == sorted(params)
        assert ts.loc['2017-06-03', 'ssm'] == 57.5
