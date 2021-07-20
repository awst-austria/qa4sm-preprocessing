
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

