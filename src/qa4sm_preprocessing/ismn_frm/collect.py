import os.path

import numpy as np
import xarray as xr

try:
    import matplotlib
    matplotlib.use("Qt5Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns
    _enable_plots = True
except ImportError as e:
    print(f"Error during import.: {e}")
    _enable_plots = False

import pandas as pd

"""
See documentation in: docs/ismn_frm.rst

Create FRM qualification csv file from validation results that can be read by 
the ismn package and transferred into the metadata.
"""

class PlottingError(Exception):
    pass

class FrmTcaQualification:
    def __init__(
        self,
        path_val,
        var_snr,
        var_nobs,
        var_snr_ci_lower,
        var_snr_ci_upper,
        var_depth_from,
        var_depth_to,
        out_path,
    ):
        """
        Reader for a QA4SM TripleCollocation Validation result.

        Parameters
        ----------
        path_val: str
            Path to validation results for ISMN sensors
        var_snr: str
            Variable that contains the SNR validation values to use for
            classification.
        var_nobs: str
            Variable that contains the nobs validation values to use for
            classification.
        var_snr_ci_lower: str
            Variable that contains the validation SNR CI lower limit values
            to use for classification.
        var_snr_ci_upper: str
            Variable that contains the validation SNR CI upper limit values
            to use for classification.
        var_depth_from: float, optional (default: None)
            Depth that is assigned to the metadata variable
        var_depth_to: float, optional (default: None)
            Depth that is assigned to the metadata variable
        out_path: str
            Path where output files will be stored.


        """
        self.ds = xr.open_dataset(path_val)

        self.var_snr = var_snr
        self.var_nobs = var_nobs
        self.var_snr_ci_lower = var_snr_ci_lower
        self.var_snr_ci_upper = var_snr_ci_upper
        self.var_depth_from = var_depth_from
        self.var_depth_to = var_depth_to

        self.out_path = out_path
        self.classification = None

    def classify(
        self,
        min_nobs=100,
        snr_thres=(0, 3),
        snr_ci_delta_thres=3,
        include_other_vars=None,
    ):
        """
        Perform classification of ISMN sensors based on the given Triple
        Collocation results.

        Parameters
        ----------

        min_nobs: int, optional (default: 100)
            How many obs must be used in the Triple Collocation for the SNR
            to be considered in the classification.
        snr_thres: tuple, optional (default: (0, 3))
            Thresholds for `representative` and `very representative` sensors.
            i.e. < 0 -> not representative, 0-3 -> representative,
                 > 3 -> very representative
        snr_ci_delta_thres: int, optional (default: 3)
            Threshold for the delta of `SNR CI upper` and `SNR CI lower`.
            If the delta is larger the this value, the sensor is not
            considered to be representative.
        include_other_vars: dict, optional (default: None)
            Keys are variables in the input validation results netcdf file that
            should be transferred into the output file. Values are the new names
            in the output file.
        """

        ds = self.ds

        if (self.var_snr_ci_lower is not None) and (self.var_snr_ci_upper is not None):
            ds['delta_ci'] = ds[self.var_snr_ci_upper] - ds[self.var_snr_ci_lower]
            ds = ds.drop([self.var_snr_ci_lower, self.var_snr_ci_upper])

        flag_low_nobs = (ds[self.var_nobs] < min_nobs).values
        flag_high_ci_range = (ds['delta_ci'] > snr_ci_delta_thres).values

        mask_very_repr = (~flag_low_nobs) & (~flag_high_ci_range) & \
                         (ds[self.var_snr] >= snr_thres[1]).values
        mask_repr = (~flag_low_nobs) & (~flag_high_ci_range) & \
                    ((ds[self.var_snr] < snr_thres[1]).values &
                     (ds[self.var_snr] >= snr_thres[0]).values)
        mask_unrepr = (~flag_low_nobs) & (~flag_high_ci_range) & \
                      (ds[self.var_snr] < snr_thres[0]).values

        ds = ds.assign(
            frm_class=(['loc'], np.full(ds[self.var_snr].values.shape, 'undeducible').astype('<U20')),
            criterion=(['loc'], np.full(ds[self.var_snr].values.shape, 'other').astype('<U20')),
        )

        ds['criterion'].values[flag_low_nobs] = 'low nobs'
        ds['criterion'].values[flag_high_ci_range] = 'large CI range'
        ds['criterion'].values[mask_very_repr | mask_repr | mask_unrepr] = 'ok'

        ds['frm_class'].values[mask_very_repr] = 'very representative'
        ds['frm_class'].values[mask_repr] = 'representative'
        ds['frm_class'].values[mask_unrepr] = 'not representative'

        vars = ['frm_class', self.var_snr, self.var_nobs, 'criterion']

        rename = {self.var_snr: 'frm_snr', self.var_nobs: 'frm_nobs'}

        if include_other_vars is not None:
            vars.extend(list(include_other_vars.keys()))
            rename.update(include_other_vars)
        else:
            include_other_vars = {}

        if self.var_depth_from:
            vars.append(self.var_depth_from)
            rename[self.var_depth_from] = 'depth_from'
        if self.var_depth_to:
            vars.append(self.var_depth_to)
            rename[self.var_depth_to] = 'depth_to'

        drop = [c for c in ('lon', 'lat', 'idx')
                if c not in include_other_vars.keys()]


        self.classification = ds[vars].to_dataframe() \
                                      .rename(columns=rename) \
                                      .drop(columns=drop)
        self.classification['variable'] = 'soil_moisture'

    def export(self):
        """
        Export the current classification to csv file that the ismn reader can
        use via the `ismn.custom.CustomSensorMetadataCsv` class.
        """
        cols = ['network', 'station', 'instrument']


        df = [self.classification.drop(columns=['criterion']),
              self.ds[cols].to_pandas()]

        df = pd.concat(df, axis=1)

        df.to_csv(os.path.join(self.out_path, 'frm_classification.csv'),
                  index=False, sep=';')

    def plot_bar(self):
        """
        Create bar plot of QI classifications in out_path.
        """
        if not _enable_plots:
            raise PlottingError("Seaborn and matplotlib are required but not installed.")

        if self.classification is None:
            raise ValueError("Create classification first.")

        plt.figure()
        ax = sns.countplot(data=self.classification, y='frm_class',
                           hue='criterion',
                           order=['undeducible', 'very representative',
                                  'representative', 'not representative'],
                           hue_order=['low nobs', 'large CI range',
                                      'other', 'ok'])
        plt.tight_layout()
        plt.savefig(os.path.join(self.out_path, 'qi_bar.png'))

        return ax

    def plot_scatter(self, by='frm'):
        """
        Create scatter plot of frm QI classification in out_path.

        Parameters
        ----------
        by: str, optional
            Create scatter for different depths ('depth') or
            different frm classes ('frm').
        """
        if not _enable_plots:
            raise PlottingError("Seaborn and matplotlib are required but not installed.")


        if self.classification is None:
            raise ValueError("Create classification first.")

        if by.lower() == 'frm':
            plt.figure()
            ax = sns.scatterplot(x="frm_nobs", y="frm_snr", hue="frm_class",
                                 data=self.classification)
            plt.tight_layout()
            plt.savefig(os.path.join(self.out_path, 'qi_scatter_frm_class.png'))
            return ax
        elif by.lower() == 'depth':
            plt.figure()
            ax = sns.scatterplot(x="frm_nobs", y="frm_snr", hue="depth_from",
                                 palette='turbo', data=self.classification)

            norm = plt.Normalize(self.classification['depth_from'].min(),
                                 self.classification['depth_from'].max())
            sm = plt.cm.ScalarMappable(cmap="turbo", norm=norm)
            sm.set_array([])

            # Remove the legend and add a colorbar
            ax.get_legend().remove()
            ax.figure.colorbar(sm)
            plt.savefig(os.path.join(self.out_path, 'qi_scatter_depth_class.png'))
            return ax
        else:
            raise NotImplementedError()



def create_frm_csv_for_ismn(
        tcol_val_result,
        var_snr='snr_00-ISMN_between_00-ISMN_and_01-ERA5_LAND_and_02-ESA_CCI_SM_passive',
        var_ci_upper='snr_ci_upper_00-ISMN_between_00-ISMN_and_01-ERA5_LAND_and_02-ESA_CCI_SM_passive',
        var_ci_lower = 'snr_ci_lower_00-ISMN_between_00-ISMN_and_01-ERA5_LAND_and_02-ESA_CCI_SM_passive',
        var_depth_from='instrument_depthfrom_between_00-ISMN_and_01-ERA5_LAND',
        var_depth_to='instrument_depthto_between_00-ISMN_and_01-ERA5_LAND',
        var_nobs='n_obs',
        plot=False,
        out_path='/tmp',
        include_other_vars=None,
):
    """
    Collect triple collocation results from QA4SM validation and compute
    FRM qualification from the relevant variables using the proposed
    thresholds provided by ISMN.

    Parameters
    ----------
    tcol_val_result: str
        Path to a QA4SM / smecv_validation run that contains triple collocation
        results for ISMN.
        e.g. from Projects/FRM4SM/08_scratch/Validations/tcol_sat_tempref/bootstrap_tcol_80p_ci_10nobs/tcol_ismnG_ccip_era5/v1/netcdf/ismn_val_1980-01-01_TO_2021-12-31_in_0_TO_0_1.nc
    var_snr: str, optional
        The relevant SNR variable that is used to classify ISMN sensors.
    var_ci_upper: str, optional
        The relevant SNR CI upper variable that is used to classify ISMN sensors.
    var_ci_lower: str, optional
        The relevant SNR CI lower variable that is used to classify ISMN sensors.
    var_depth_from: str, optional
        The ISMN sensor depth_from variable in the validation results.
    var_depth_to: str, optional
        The ISMN sensor depth_to variable in the validation results.
    var_nobs: str, optional
        The relevant nobs variable that is used to classify ISMN sensors.
    plot: bool, optional (default: False)
        Create plots.
    out_path: str, optional
        Path where the output csv file is stored.
    include_other_vars: dict, optional (default: None)
        Keys are variables in the input validation results netcdf file that
        should be transferred into the output file. Values are the new names
        in the output file.
    """
    frm_qi = FrmTcaQualification(
        tcol_val_result,
        var_snr=var_snr,
        var_snr_ci_lower=var_ci_lower,
        var_snr_ci_upper=var_ci_upper,
        var_nobs=var_nobs,
        var_depth_from=var_depth_from,
        var_depth_to=var_depth_to,
        out_path=out_path,
    )

    frm_qi.classify(
        min_nobs=100,
        snr_thres=(0, 3),
        snr_ci_delta_thres=3,
        include_other_vars=include_other_vars,
    )

    if plot:
        frm_qi.plot_bar()
        frm_qi.plot_scatter('frm')

    frm_qi.export()