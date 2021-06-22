
from parse import parse
import os

def clear_images(path:str, dryrun: bool = True, prod:str = 'SSM1km'):
    # clean up input data before reshuffling, this removes duplicate images.
    # dryrun means that no files in path will be deleted.

    if prod.startswith('SWI'):
        sens = 'SCATSAR'
    else:
        sens = 'S1CSAR'

    template = "c_gls_%s_{datetime}_CEURO_%s_V{vers}.nc" % (prod, sens)

    vers_files = {}
    N_tot = 0
    for img in os.listdir(path):
        el = parse(template, img).named
        if el['datetime'] not in vers_files.keys():
            vers_files[el['datetime']] = []
        vers_files[el['datetime']].append(el['vers'])
        N_tot += 1
    N = 0
    for date, vers in vers_files.items():
        for v in sorted(vers):
            if v != sorted(vers)[-1]:
                thefile = os.path.join(path, template.format(datetime=date,
                                                             vers=v))
                print(f'Remove {thefile} because a new version ({sorted(vers)[-1]}) was found.')
                N += 1
                if not dryrun:
                    os.remove(os.path.join(path, thefile))
    print(f"Files to remove: {N} of {N_tot}")


if __name__ == '__main__':
    import numpy as np
    import pandas as pd
    from pytesmo.scaling import scale

    def mean_std_scale_ignore_nan(tss, ref_name='ISMN'):

        def scale(src, ref):
            return ((src - np.nanmean(src)) / np.nanstd(src)) * np.nanstd(ref) + np.nanmean(ref)

        dat_cols = tss.columns[tss.columns != ref_name]
        tss_scaled = pd.concat([scale(tss[c], tss[ref_name]) for c in dat_cols], axis=1)
        return tss_scaled


    ref_name = 'ISMN'
    tss = pd.read_csv("/home/wpreimes/Temp/pandas.csv", index_col=0)
    ts_ref = tss[[ref_name]]

    tss_scaled = mean_std_scale_ignore_nan(tss, ref_name=ref_name)

    tss_scaled = pd.DataFrame(index=tss_scaled.index,
                              data={'mean': tss_scaled.mean(axis=1),
                                    'std': tss_scaled.std(axis=1),
                                    'mean+std': tss_scaled.mean(axis=1)+tss_scaled.std(axis=1),
                                    'mean-std': tss_scaled.mean(axis=1)-tss_scaled.std(axis=1),
                                    'N': tss_scaled.count(axis=1)})
