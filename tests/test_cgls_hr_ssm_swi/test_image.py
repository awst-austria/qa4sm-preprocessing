import pandas as pd

from qa4sm_preprocessing.cgls_hr_ssm_swi.s1cgls_nc import S1Cgls1kmDs, S1Cgls1kmImage
import unittest
import os
import numpy as np
import datetime

testdata_path = os.path.join(os.path.dirname(__file__), "..", "test-data", "preprocessing")

def is_asc(a):
    return np.all(a[:-1] <= a[1:])

class TestSSMImage2d(unittest.TestCase):
    _set_fill_val: float = -999.

    def setUp(self) -> None:
        self.ds = S1Cgls1kmImage(
            os.path.join(testdata_path, "CGLS_SSM1km_V1.1_img",
                         "c_gls_SSM1km_201706030000_CEURO_S1CSAR_V1.1.1.nc"),
            parameters=None,
            flatten=False,
            fillval={'ssm': self._set_fill_val})

    def test_read(self):
        # Image is NOT upside down!
        image = self.ds.read()
        assert list(image.data.keys()) == ['ssm', 'ssm_noise']
        assert self._set_fill_val in image.data['ssm']
        assert np.all(np.unique(self.ds.grid.activearrcell) == [1286,1322])
        assert image.data['ssm'].shape == (448, 448)
        np.testing.assert_almost_equal(image.lat[9,4], 44.91518, 5)
        np.testing.assert_almost_equal(image.lon[9,4], -0.959821, 5)
        np.testing.assert_equal(image.data['ssm'][9,4], 57.5)
        assert image.timestamp == datetime.datetime(2017, 6, 3, 0)
        assert is_asc(np.unique(image.lat))
        assert is_asc(np.unique(image.lon))

    def test_bbox_subset(self):
        """ Read 2d sub image, i.e. use subgrid from bbox of original grid """
        self.ds.read()
        orig_grid = self.ds.grid
        latlonminmax = [44.90625, 44.92411, -0.969, -0.9597]
        new_grid = orig_grid.subgrid_from_gpis(
            orig_grid.get_bbox_grid_points(*latlonminmax))
        new_grid.shape = (2,2)
        self.ds.grid = new_grid
        image = self.ds.read()
        np.testing.assert_almost_equal(image.lat[1,1], 44.91518, 5)
        np.testing.assert_almost_equal(image.lon[1,1], -0.959821, 5)
        np.testing.assert_equal(image.data['ssm'][1,1], 57.5)

class TestSWIImage1d(unittest.TestCase):

    def setUp(self) -> None:
        self.ds = S1Cgls1kmImage(
            os.path.join(testdata_path, "CGLS_SWI1km_V1.0_img",
                         "c_gls_SWI1km_201706021200_CEURO_SCATSAR_V1.0.1.nc"),
            parameters=['SWI_005', 'QFLAG_005'],
            flatten=True)

    def test_read(self):
        # Upside down image was flattened, point [9,4] is now at (448*448)-(448*9)-(448-4)
        idx = (448*448)-(448*9)-(448-4)
        image = self.ds.read()
        assert list(image.data.keys()) == ['SWI_005', 'QFLAG_005']
        assert np.all(np.unique(self.ds.grid.activearrcell) == [1286, 1322])
        assert image.data['SWI_005'].shape == (448*448, )
        np.testing.assert_almost_equal(image.lat[idx], 44.91518, 5)
        np.testing.assert_almost_equal(image.lon[idx], -0.959821, 5)
        np.testing.assert_equal(image.data['SWI_005'][idx], 53.5)
        assert image.timestamp == datetime.datetime(2017, 6, 2, 12)

    def test_bbox_subset(self):
        """ Read 1d sub image, i.e. use subgrid from bbox of original grid """
        self.ds.read()
        orig_grid = self.ds.grid
        latlonminmax = [44.90625, 44.92411, -0.969, -0.9597]
        new_grid = orig_grid.subgrid_from_gpis(
            orig_grid.get_bbox_grid_points(*latlonminmax))
        self.ds.grid = new_grid
        image = self.ds.read()
        np.testing.assert_almost_equal(image.lat[1], 44.91518, 5)
        np.testing.assert_almost_equal(image.lon[1], -0.959821, 5)
        np.testing.assert_equal(image.data['SWI_005'][1], 53.5)

class TestSWIDataset2d(unittest.TestCase):
    def setUp(self) -> None:
        self.ds = S1Cgls1kmDs(
            os.path.join(testdata_path, "CGLS_SWI1km_V1.0_img"),
            parameters=['SWI_005', 'QFLAG_005'],
            fname_templ="c_gls_SWI1km_{datetime}_CEURO_SCATSAR_V*.nc",
            hours=(12,),
            flatten=False)

    def test_timestamps_for_date_range(self):
        ts = self.ds.tstamps_for_daterange(datetime.datetime(2000,1,1),
                                           datetime.datetime(2000,1,10))
        assert np.all(
            ts == pd.date_range(datetime.datetime(2000,1,1,12),
                                datetime.datetime(2000,1,10,12),
                                freq='D').to_pydatetime()
        )

    def test_read(self):
        image = self.ds.read(datetime.datetime(2017,6,2,12))
        assert list(image.data.keys()) == ['SWI_005', 'QFLAG_005']
        assert np.all(np.unique(self.ds.fid.grid.activearrcell) == [1286, 1322])
        assert image.data['SWI_005'].shape == (448, 448)
        np.testing.assert_almost_equal(image.lat[9,4], 44.91518, 5)
        np.testing.assert_almost_equal(image.lon[9,4], -0.959821, 5)
        np.testing.assert_equal(image.data['SWI_005'][9,4], 53.5)
        assert image.timestamp == datetime.datetime(2017, 6, 2, 12)


class TestSSMDataset1d(unittest.TestCase):
    def setUp(self) -> None:
        self.ds = S1Cgls1kmDs(
            os.path.join(testdata_path, "CGLS_SSM1km_V1.1_img"),
            parameters=['ssm'],
            fname_templ="c_gls_SSM1km_{datetime}_CEURO_S1CSAR_V*.nc",
            flatten=True)

    def test_timestamps_for_date_range(self):
        ts = self.ds.tstamps_for_daterange(datetime.datetime(2000,1,1),
                                           datetime.datetime(2000,1,10))
        assert np.all(
            ts == pd.date_range(datetime.datetime(2000,1,1,0),
                                datetime.datetime(2000,1,10,0),
                                freq='D').to_pydatetime()
        )

    def test_read(self):
        idx = (448*448)-(448*9)-(448-4)
        image = self.ds.read(datetime.datetime(2017,6,3,0))
        assert list(image.data.keys()) == ['ssm']
        assert np.all(np.unique(self.ds.fid.grid.activearrcell) == [1286, 1322])
        assert image.data['ssm'].shape == (448*448, )
        np.testing.assert_almost_equal(image.lat[idx], 44.91518, 5)
        np.testing.assert_almost_equal(image.lon[idx], -0.959821, 5)
        np.testing.assert_equal(image.data['ssm'][idx], 57.5)
        assert image.timestamp == datetime.datetime(2017, 6, 3, 0)

if __name__ == '__main__':
    suite = unittest.TestSuite()
    suite.addTest(TestSWIImage1d("test_bbox_subset"))
    runner = unittest.TextTestRunner()
    runner.run(suite)