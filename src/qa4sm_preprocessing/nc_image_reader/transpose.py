import dask
import datetime
import logging
import math
from multiprocessing.pool import ThreadPool
import numpy as np
from pathlib import Path
from tqdm import trange
from tqdm.dask import TqdmCallback
from typing import Union, Tuple, TypeVar
import warnings
import xarray as xr


from .utils import infer_chunks


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
):
    """
    Creates a stacked and transposed netCDF file from a given reader.

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
    """
    dask_config = {
        "array.slicing.split_large_chunks": False,
        "scheduler": "threads",
        "pool": ThreadPool(n_threads),
    }
    dask.config.set(**dask_config)

    # client = Client(
    #     n_workers=1, threads_per_worker=n_threads, memory_limit=f"{memory}GB"
    # )
    # print("Dask dashboard accessible at:", client.dashboard_link)

    new_last_dim = reader.timename
    timestamps = reader.tstamps_for_daterange(start, end)

    # first, get some info about structure of the input file
    first_img = reader.read_block(start=timestamps[0], end=timestamps[0])[
        reader.varnames[0]
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
        input_dim_sizes[input_dim_names.index(dim)] for dim in new_dim_names
    ]
    new_dim_sizes[-1] = len(timestamps)
    new_dim_sizes = tuple(new_dim_sizes)

    # netCDF chunks should be about 1MB in size
    chunks = infer_chunks(new_dim_sizes, 1, dtype)
    # dask chunks should be about 100MB in size
    dask_chunks = infer_chunks(new_dim_sizes, 100, dtype)
    size = dtype.itemsize * len(reader.varnames)
    chunksize_MB = np.prod(chunks) * size / 1024 ** 2
    logging.info(
        f"create_transposed_netcdf: Creating chunks {chunks}"
        f" with chunksize {chunksize_MB:.2f} MB"
    )

    # calculate block size to use based on given memory size
    len_new_dim = new_dim_sizes[-1]
    imagesize_GB = np.prod(new_dim_sizes[:-1]) * size / 1024 ** 3
    # we need to divide by two, because we need intermediate storage for
    # the transposing
    stepsize = int(math.floor(memory / imagesize_GB)) // 2
    stepsize = min(stepsize, len_new_dim)
    logging.info(
        f"create_transposed_netcdf: Using {stepsize} images as buffer,"
        f" leading to buffer size of {2 * stepsize * imagesize_GB:.2f} GB"
    )
    block_start = list(
        map(int, np.arange(0, len_new_dim + stepsize - 0.5, stepsize))
    )
    block_start[-1] = min(block_start[-1], len_new_dim)

    num_blocks = len(block_start) - 1

    # We are first creating intermediate files, that we will then merge to a
    # single file as final step.
    # The intermediate files will not be compressed, but chunked.
    if num_blocks == 1:
        print("Creating transposed file:")
    else:
        print("Creating intermediate files:")
    temporary_dask_chunks = []
    for i in trange(num_blocks):
        s, e = block_start[:-1][i], block_start[1:][i]

        tmp_chunks = list(chunks[:-1])
        tmp_chunks.append(e - s)
        tmp_chunks = tuple(tmp_chunks)

        tmp_dask_chunks = list(dask_chunks[:-1])
        tmp_dask_chunks.append(e - s)
        tmp_dask_chunks = tuple(tmp_dask_chunks)
        temporary_dask_chunks.append(tmp_dask_chunks)

        block = reader.read_block(
            start=timestamps[s], end=timestamps[e - 1]
        ).chunk(dict(zip(new_dim_names, tmp_dask_chunks)))
        transposed_block = block.transpose(..., new_last_dim)
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="String decoding changed.*",
                category=FutureWarning,
            )
            transposed_block.to_netcdf(
                str(outfname) + f".tmp.{i}",
                encoding={
                    varname: {"chunksizes": tmp_chunks}
                    for varname in reader.varnames
                },
                engine="h5netcdf",
            )

    if num_blocks == 1:
        Path(str(outfname) + f".tmp.{i}").rename(outfname)
    else:
        datasets = []
        for i in range(num_blocks):
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    message="String decoding changed.*",
                    category=FutureWarning,
                )
                ds = xr.open_dataset(
                    str(outfname) + f".tmp.{i}",
                    chunks=dict(zip(new_dim_names, temporary_dask_chunks[i])),
                    engine="h5netcdf",
                )
                datasets.append(ds)
        print("Concatenating intermediate files:")
        final = xr.concat(
            datasets,
            dim=reader.timename,
            compat="override",
            data_vars="minimal",
            coords="minimal",
        ).chunk(dict(zip(new_dim_names, dask_chunks)))
        final.attrs = reader.dataset_metadata
        with TqdmCallback(), warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="String decoding changed.*",
                category=FutureWarning,
            )
            final.to_netcdf(
                outfname,
                encoding={
                    varname: {
                        "chunksizes": chunks,
                        "zlib": zlib,
                        "complevel": complevel,
                    }
                    for varname in reader.varnames
                },
                engine="h5netcdf",
            )
        # remove temporary files
        for i in range(num_blocks):
            Path(str(outfname) + f".tmp.{i}").unlink()

    logging.info("create_transposed_netcdf: Finished writing transposed file.")
