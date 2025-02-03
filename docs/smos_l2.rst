Processing SMOS L2 swath data to time series
============================================

.. _process_smos_l2:

The smos module is used to convert L2 swath files (also SBPCA) to time series.

Download SMOS L2 swath files
----------------------------

For the SBPCA data use the following FTP (requires access from ESA): ``dpgsrftpserver.smos.eo.esa.int``

For L2 SM use the following FTP (requires an `account <https://earth.esa.int/eogateway/catalog/smos-science-products>`_):
``smos-diss.eo.esa.int``

As of Feb. 2025, there were some issues downloading the files as the server did not allow recursive download via sftp. Instead, run the command

.. code-block:: shell

    lftp -c "set sftp:auto-confirm yes; open -u wpreimesberger sftp://dpgsrftpserver.smos.eo.esa.int; mirror --continue --parallel=5 -L /data/ftp/dpgsr_l2sm_IF/Out_Folder/LOC_DPGS_RC_L2SM ."


Convert ZIP to netcdf
---------------------
This is **only relevant for SBPCA data**.
Download the SMOS Netcdf conversion tool and convert the data to netcdf.

.. code-block:: shell

    in_root=".../LOC_DPGS_RC_L2SM_SBPCAfromv724"
    out_root=".../smos_sbpca_ts"
    year="2013"

    files=$(ls $in_root/$year/*/*/*SMUDP*.zip)

    for file in $files; do
      date_part=$(dirname "$file" | awk -F'/' '{print "/"$(NF-1)"/"$NF}')
      outpath=$out_root/$year/$date_part/
      bash .../smos-ee-to-nc.sh $file --target-directory $outpath --overwrite-target
    done

Convert Swath data to time series
---------------------------------
Use the `SMOSL2Reader` from this package and call the resampling routine.
Add the overpass flag so we can filter for asc and desc. orbits.

.. code-block:: python

    from datetime import datetime
    from pygeogrids.netcdf import load_grid
    from qa4sm_preprocessing.level2.smos import SMOSL2Reader

    path = ".../smos_sbpca_l2nc"
    parameters = ['Chi_2_P', 'M_AVA0', 'N_RFI_X', 'N_RFI_Y', 'RFI_Prob',
                  'Science_Flags', 'Soil_Moisture', 'acquisition_time']

    reader = SMOSL2Reader(path, parameters, add_overpass_flag=True)
    grid = load_grid("5deg_SMOSL2_grid.nc")
    reader.repurpose(
        ".../smos_sbpca_v5_ts",
        start=datetime(2011, 2, 1),
        end=datetime(2013, 1, 31),
        overwrite=True,
        memory=100,
    )

