import os
from repurpose.img2ts import Img2Ts
from qa4sm_preprocessing.cgls_hr_ssm_swi.s1cgls_nc import S1Cgls1kmDs
import datetime

def reshuffle(input_root,
              outputpath,
              startdate,
              enddate,
              parameters,
              bbox=None,
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
    startdate : datetime.datetime
        Start date.
    enddate : datetime.datetime
        End date.
    parameters: Iterable
        parameters to read and convert
    bbox: list[float, float, float, float], optional (default: None)
        min_lon min_lat max_lon max_lat
        Bounding box that contains values to reshuffle from the original image.
    imgbuffer: int, optional
        How many images to read at once before writing time series.
    """
    input_dataset = S1Cgls1kmDs(data_path=input_root, parameters=parameters,
                                flatten=True, **ds_kwargs)

    input_dataset.read(startdate) # to build the grid from data
    input_grid = input_dataset.fid.grid

    if bbox is not None:
        input_grid = input_grid.subgrid_from_gpis(
            input_grid.get_bbox_grid_points(*bbox))

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
