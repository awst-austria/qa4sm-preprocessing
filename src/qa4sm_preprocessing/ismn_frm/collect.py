import os.path

import numpy as np
import xarray as xr

import matplotlib
matplotlib.use("Qt5Agg")

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

"""
Create FRM qualification csv file that can be read by the ismn package and
transferred into the metdata.
"""

class FrmTcaQualification:
    def __init__(
        self,
        path_val,
    ):
        self.ds = xr.open_dataset(path_val)

    def classify_from_tca(
        self,
        var_snr,
        var_nobs,
        var_snr_ci_lower,
        var_snr_ci_upper,
        min_nobs=100,
        snr_thres=(0, 3),
        snr_ci_delta_thres=3,
        var_depth_from=None,
        var_depth_to=None,
    ):
        """
        Perform classification.

        Returns
        -------

        """

        ds = self.ds

        if (var_snr_ci_lower is not None) and (var_snr_ci_upper is not None):
            ds['delta_ci'] = ds[var_snr_ci_upper] - ds[var_snr_ci_lower]
            ds = ds.drop([var_snr_ci_lower, var_snr_ci_upper])

        flag_low_nobs = (ds[var_nobs] < min_nobs).values
        flag_high_ci_range = (ds['delta_ci'] > snr_ci_delta_thres).values

        mask_very_repr = (~flag_low_nobs) & (~flag_high_ci_range) & \
                         (ds[var_snr] >= snr_thres[1]).values
        mask_repr = (~flag_low_nobs) & (~flag_high_ci_range) & \
                    ((ds[var_snr] < snr_thres[1]).values &
                     (ds[var_snr] >= snr_thres[0]).values)
        mask_unrepr = (~flag_low_nobs) & (~flag_high_ci_range) & \
                      (ds[var_snr] < snr_thres[0]).values

        ds = ds.assign(
            frm_class=(['loc'], np.full(ds[var_snr].values.shape, 'unknown').astype('<U20')),
            criterion=(['loc'], np.full(ds[var_snr].values.shape, 'other').astype('<U20')),
        )

        ds['criterion'].values[flag_low_nobs] = 'low nobs'
        ds['criterion'].values[flag_high_ci_range] = 'large CI range'
        ds['criterion'].values[mask_very_repr | mask_repr | mask_unrepr] = 'ok'

        ds['frm_class'].values[mask_very_repr] = 'very representative'
        ds['frm_class'].values[mask_repr] = 'representative'
        ds['frm_class'].values[mask_unrepr] = 'not representative'

        vars = ['frm_class', var_snr, var_nobs, 'criterion']
        rename = {var_snr: 'frm_snr', var_nobs: 'frm_nobs'}

        if var_depth_from:
            vars.append(var_depth_from)
            rename[var_depth_from] = 'depth_from'
        if var_depth_to:
            vars.append(var_depth_to)
            rename[var_depth_to] = 'depth_to'


        self.classification = ds[vars].to_dataframe().rename(columns=rename) \
                                      .drop(columns=['lon', 'lat', 'idx'])


    def export(self, format='csv', out_path=None, depth_from_var=None,
               depth_to_var=None):
        """
        Export the current classification to csv file that the ismn reader can
        use.

        Parameters
        ----------
        format

        Returns
        -------

        """
        cols = ['network', 'station', 'instrument']
        if depth_from_var is not None:
            cols.append(depth_from_var)
        if depth_to_var is not None:
            cols.append(depth_to_var)

        df = [self.classification.drop(columns=['criterion']),
              self.ds[cols].to_pandas()]

        df = pd.concat(df, axis=1).drop(columns=['lat', 'lon', 'idx'])

        if depth_from_var is not None:
            df = df.rename(columns={depth_from_var: 'depth_from'})
        if depth_to_var is not None:
            df = df.rename(columns={depth_to_var: 'depth_to'})

        out_path = os.path.dirname(__file__) if out_path is None else out_path

        if format.lower() == 'csv':
            df.to_csv(os.path.join(out_path, 'frm_class.csv'), index=False,
                      sep=';')
        else:
            raise NotImplementedError(f"Unsupported format: {format}")

    def plot(self, plot_bar=False, plot_scatter_by_frm=False,
             plot_scatter_by_depth=True):
        if plot_bar:
            plt.figure()
            ax = sns.countplot(data=self.classification, y='frm_class',
                               hue='criterion',
                               order=['unknown', 'very representative',
                                      'representative', 'not representative'],
                               hue_order=['low nobs', 'large CI range',
                                          'other', 'ok'])
            plt.tight_layout()

        if plot_scatter_by_frm:
            ax = sns.scatterplot(x="frm_nobs", y="frm_snr", hue="frm_class",
                                 data=self.classification)
            plt.tight_layout()


        if plot_scatter_by_depth:
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
