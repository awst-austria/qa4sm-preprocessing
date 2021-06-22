import os
from repurpose.img2ts import Img2Ts
from s1cgls_nc import S1Cgls1kmDs

def reshuffle(input_root,
              outputpath,
              startdate,
              enddate,
              parameters,
              imgbuffer=50,
              **ds_kwargs):
    """
    Reshuffle method applied to GLDAS data.

    Parameters
    ----------
    input_root: str
        input path where gldas data was downloaded
    outputpath : str
        Output path.
    startdate : datetime
        Start date.
    enddate : datetime
        End date.
    parameters: Iterable
        parameters to read and convert
    imgbuffer: int, optional
        How many images to read at once before writing time series.
    """
    input_dataset = S1Cgls1kmDs(data_path=input_root, parameters=parameters,
                                flatten=True, **ds_kwargs)

    input_dataset.read(startdate) # to build the grid from data
    input_grid = input_dataset.fid.grid

    # build again using the cut grid.
    input_dataset = S1Cgls1kmDs(data_path=input_root,
                                parameters=parameters,
                                flatten=True,
                                grid=input_grid,
                                **ds_kwargs)

    if not os.path.exists(outputpath):
        os.makedirs(outputpath)

    # get time series attributes from first day of data.
    img = input_dataset.read(startdate)
    metadata = img.metadata

    global_attrs = metadata.pop('nc_global_attr')

    reshuffler = Img2Ts(input_dataset=input_dataset, outputpath=outputpath,
                        startdate=startdate, enddate=enddate, input_grid=input_grid,
                        imgbuffer=imgbuffer, cellsize_lat=5.0,
                        cellsize_lon=5.0, global_attr=global_attrs, zlib=True,
                        unlim_chunksize=1000, ts_attributes=metadata)
    reshuffler.calc()

if __name__ == '__main__':
    from datetime import datetime
    #input = "/data-read/USERS/wpreimes/temp/CGLS_SWI1km_V1.0_img/"
    input = "/shares/wpreimes/radar/Projects/QA4SM_HR/07_data/CGLS_SWI1km_V1.0_img/"
    output = "/data-write/USERS/wpreimes/temp/CGLS_SWI1km_V1.0_ts/"

    reshuffle(input, output, datetime(2015,1, 1,12), datetime(2020,7,31,12),
              ['SWI_005', 'QFLAG_005', 'SWI_040', 'QFLAG_040'],
              fname_templ="c_gls_SWI1km_{datetime}1200_CEURO_SCATSAR_V*.nc",
              datetime_format = "%Y%m%d")
