import dask
import dask.array as dsa
from dask.distributed import Client
import datetime
import logging
import math
from multiprocessing.pool import ThreadPool
import numpy as np
from pathlib import Path
from typing import Union, TypeVar
import xarray as xr
import shutil
import warnings


from .utils import infer_chunks
from qa4sm_preprocessing.nc_image_reader.readers import DirectoryImageReader


Reader = TypeVar("Reader")


def create_transposed_netcdf(
    reader: Reader,
    outfname: Union[Path, str],
    new_last_dim: str = None,
    start: datetime.datetime = None,
    end: datetime.datetime = None,
    memory: float = 2,
    n_threads: int = 4,
    zlib: bool = True,
    complevel: int = 4,
    distributed: Union[bool, Client] = False,
):
    """
    Creates a stacked and transposed netCDF file from a given reader.

    WARNING: very experimental!

    Parameters
    ----------
    reader : XarrayImageReaderBase
        Reader for the dataset.
    outfname : str or Path
        Output filename.
    start : datetime.datetime, optional
        If not given, start at first timestamp in dataset.
    end : datetime.datetime, optional
        If not given, end at last timestamp in dataset.
    memory : float, optional
        The amount of memory to be used for buffering in GB. Default is 2.
        Higher is faster.
    n_threads : int, optional
        The amount of threads to use. Default is 4.
    zlib : bool, optional
        Whether to use compression when storing the files. Reduces file size,
        but strongly increases write time, and maybe also access time. Default
        is ``False``.
    complevel : int, optional
        Compression level to use. Default is 4. Range is from 1 (low) to 9
        (high).
    distributed : bool or Client, optional
        Whether to use the local or the distributed dask scheduler. If a client
        for a distributed scheduler is used, this is used instead.
    """
    dask_config = {
        "array.slicing.split_large_chunks": False,
    }
    if isinstance(distributed, Client) or not distributed:
        if not distributed:
            dask_config.update(
                {"scheduler": "threads", "pool": ThreadPool(n_threads)}
            )
        with dask.config.set(**dask_config):
            _transpose(
                reader,
                outfname,
                new_last_dim=new_last_dim,
                start=start,
                end=end,
                memory=memory,
                zlib=zlib,
                complevel=complevel,
            )
    elif distributed:
        with dask.config.set(**dask_config), Client(
            n_workers=1,
            threads_per_worker=n_threads,
            memory_limit=f"{memory}GB",
        ) as client:
            print("Dask dashboard accessible at:", client.dashboard_link)
            _transpose(
                reader,
                outfname,
                new_last_dim=new_last_dim,
                start=start,
                end=end,
                memory=memory,
                zlib=zlib,
                complevel=complevel,
            )


def _transpose(
    reader: Reader,
    outfname: Union[Path, str],
    new_last_dim: str = None,
    start: datetime.datetime = None,
    end: datetime.datetime = None,
    memory: float = 2,
    zlib: bool = True,
    complevel: int = 4,
):
    new_last_dim = reader.timename
    timestamps = reader.tstamps_for_daterange(start, end)

    # we need to mask the grid, because it doesn't support pickling
    grid = reader.grid
    reader.grid = None

    variable_fnames = []
    variable_chunks = []
    for varname in reader.varnames:

        # first, get some info about structure of the input file
        first_img = reader.read_block(start=timestamps[0], end=timestamps[0])[
            varname
        ]
        dtype = first_img.dtype
        input_dim_names = first_img.dims
        input_dim_sizes = first_img.shape

        if input_dim_names[-1] == new_last_dim:  # pragma: no cover
            print(f"{new_last_dim} is already the last dimension")

        # get new dim names in the correct order
        new_dim_names = list(input_dim_names)
        new_dim_names.remove(new_last_dim)
        new_dim_names.append(new_last_dim)
        new_dim_names = tuple(new_dim_names)
        new_dim_sizes = [
            input_dim_sizes[input_dim_names.index(dim)]
            for dim in new_dim_names
        ]
        new_dim_sizes[-1] = len(timestamps)
        new_dim_sizes = tuple(new_dim_sizes)

        # netCDF chunks should be about 1MB in size
        chunks = infer_chunks(new_dim_sizes, 1, dtype)
        variable_chunks.append(chunks)
        size = dtype.itemsize
        chunksize_MB = np.prod(chunks) * size / 1024 ** 2
        logging.info(
            f"create_transposed_netcdf: Creating chunks {str(chunks)}"
            f" with chunksize {chunksize_MB:.2f} MB for {varname}"
        )

        # calculate intermediate chunk size in time direction
        len_new_dim = new_dim_sizes[-1]
        imagesize_GB = np.prod(new_dim_sizes[:-1]) * size / 1024 ** 3
        # we need to divide by two, because we need intermediate storage for
        # the transposing, and we need another factor of 2-3 because of
        # unmanaged memory in the dask scheduler
        stepsize = int(math.floor(memory / imagesize_GB)) // 5
        stepsize = min(stepsize, len_new_dim)
        logging.info(
            f"create_transposed_netcdf: Using {stepsize} images as buffer"
        )

        # dask chunks should be about 50MB in size
        # infer_chunks assumes that the last dimension is the continuous dimension
        tmp_dim_sizes = list(new_dim_sizes)
        tmp_dim_sizes[-1] = stepsize
        tmp_dask_chunks = list(infer_chunks(tmp_dim_sizes, 50, dtype))
        dask_chunks = [stepsize] + tmp_dask_chunks[:-1]
        dask_chunks = tuple(dask_chunks)

        tmp_outfname = str(outfname) + f".{varname}.zarr"
        variable_fnames.append(tmp_outfname)

        if not Path(tmp_outfname).exists():
            arrays = []
            logging.debug(
                f"create_transposed_netcdf: starting dask array creation"
                f" for {len(timestamps)} timestamps"
            )

            # I had issues with memory leaks, probably related to
            # https://github.com/dask/dask/issues/3530
            # when using the from_delayed approach that got blocks as numpy
            # arrays.
            # The reader should probably have set use_dask, so that dask arrays
            # are returned by _get_single_image_as_array
            if (
                isinstance(reader, DirectoryImageReader)
                and reader.chunks is None
            ):
                warnings.warn(
                    "Not reading single images as dask arrays. If you run into memory"
                    " issues, use the `use_dask=True` option for your reader."
                )

            def _get_single_image_as_array(tstamp):
                img = reader.read_block(tstamp, tstamp)[varname]
                return img.data[0, ...]

            for t in timestamps:
                arr = _get_single_image_as_array(t)
                arrays.append(arr)

            logging.debug("create_transposed_netcdf: stacking")
            all_data = dsa.stack(arrays).rechunk(dask_chunks)

            coords = {
                c: first_img.coords[c]
                for c in new_dim_names[:-1]
                if c in first_img.coords
            }
            coords[reader.timename] = timestamps
            ds = xr.Dataset(
                {
                    varname: (
                        input_dim_names,
                        all_data,
                        reader.array_metadata[varname],
                    )
                },
                coords=coords,
            )
            ds.attrs.update(reader.dataset_metadata)
            logging.debug("create_transposed_netcdf: transpose")
            ds = ds.transpose(..., reader.timename)
            logging.debug(f"create_transposed_netcdf: Writing {tmp_outfname}")
            ds.to_zarr(tmp_outfname, mode="w", consolidated=True)

    variable_datasets = []
    for fname in variable_fnames:
        # now we can convert to netCDF
        ds = xr.open_zarr(fname, consolidated=True)
        variable_datasets.append(ds)

    ds = xr.merge(variable_datasets)
    ds.to_netcdf(
        outfname,
        encoding={
            varname: {
                "chunksizes": variable_chunks[j],
                "zlib": zlib,
                "complevel": complevel,
            }
            for j, varname in enumerate(reader.varnames)
        },
    )
    reader.grid = grid
    for fname in variable_fnames:
        shutil.rmtree(fname)
    logging.info("create_transposed_netcdf: Finished writing transposed file.")
