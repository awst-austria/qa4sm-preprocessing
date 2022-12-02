from qa4sm_preprocessing.ismn_frm.collect import create_frm_csv_for_ismn
import os
import tempfile
import pandas as pd
import numpy as np

test_data_path = os.path.join(os.path.dirname(__file__), '..', 'test-data')

def test_frm_classification_from_tcol_results():
    with tempfile.TemporaryDirectory() as out_path:
        val_results = os.path.join(test_data_path, 'preprocessing', 'ismn_frm',
                                   'tcol_SilverSword.nc')
        create_frm_csv_for_ismn(val_results, out_path=out_path, plot=False)

        df = pd.read_csv(os.path.join(out_path, 'frm_classification.csv'),
                         sep=';')
        assert len(df.index) == 1
        assert df.iloc[0].frm_class == 'undeducible'
        assert df.iloc[0].frm_snr == 8.59662
        assert all(np.isin(['frm_nobs', 'depth_from', 'depth_to', 'network',
                            'station', 'instrument'], df.columns))
        assert df.iloc[0].station == 'SilverSword'
