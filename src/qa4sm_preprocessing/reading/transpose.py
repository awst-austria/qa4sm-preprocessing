import copy
import dask
import dask.array as da
from dask.distributed import Client
import datetime
import logging
from multiprocessing.pool import ThreadPool
import numpy as np
from pathlib import Path
from tqdm.auto import tqdm
from typing import Union, TypeVar
import xarray as xr
import shutil
import zarr
from numcodecs.blosc import Blosc

from .utils import infer_chunksizes, nimages_for_memory


Reader = TypeVar("Reader")


def write_transposed_dataset(
    reader: Reader,
    outfname: Union[Path, str],
    start: datetime.datetime = None,
    end: datetime.datetime = None,
    chunks: dict = None,
    memory: float = 5,
    zlib: bool = True,
    complevel: int = 4,
    distributed=False,
    n_threads: int = 1,
    stepsize: int = None,
):
    """
    Creates a stacked and transposed netCDF/zarr file from a given reader.

    WARNING: very experimental!

    Parameters
    ----------
    reader : ImageReaderBase instance
        Reader for the dataset.
    outfname : str or Path
        Output filename. Must end with ".nc" for netCDF output or with ".zarr"
        for zarr output.
    start : datetime.datetime, optional
        If not given, start at first timestamp in dataset.
    end : datetime.datetime, optional
        If not given, end at last timestamp in dataset.
    chunks : dictionary, optional
        The chunk sizes that are used for the transposed file. If none are
        given, chunks with a size of 1MB are used for netCDF, and chunks with a
        size of 100MB are used for zarr output.
    memory : float, optional
        The amount of memory to be used for buffering in GB. Default is 5.
        Higher is faster.
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
    n_threads : int, optional
        The amount of threads to use for the final step of the
        transposing. Default is 2. Only relevant if `distributed` is not a dask
        client instance.
    """
    args = (reader, outfname)
    kwargs = {
        "start": start,
        "end": end,
        "memory": memory,
        "zlib": zlib,
        "complevel": complevel,
        "chunks": chunks,
        "stepsize": stepsize,
    }

    dask_config = {
        "array.slicing.split_large_chunks": False,
    }
    if isinstance(distributed, Client) or not distributed:
        if not distributed:
            logging.info(
                "write_transposed_dataset: using thread pool with "
                f" {n_threads} threads"
            )
            dask_config.update(
                {
                    "scheduler": "threads",
                    "pool": ThreadPool(n_threads),
                }
            )
        else:
            logging.info(
                "write_transposed_dataset: using distributed scheduler"
                f" available at {distributed.dashboard_link}"
            )
        with dask.config.set(**dask_config):
            _transpose(*args, **kwargs)
    else:  # distributed is True
        with dask.config.set(**dask_config), Client(
            n_workers=1,
            threads_per_worker=n_threads,
            memory_limit=f"{memory}GB",
        ) as client:
            print("Dask dashboard accessible at:", client.dashboard_link)
            _transpose(*args, **kwargs)


def _transpose(
    reader: Reader,
    outfname: Union[Path, str],
    start: datetime.datetime = None,
    end: datetime.datetime = None,
    chunks: dict = None,
    memory: float = 5,
    zlib: bool = True,
    complevel: int = 4,
    stepsize: int = None,
):
    # Some reader settings need to be adapted for the duration of this routine
    if hasattr(reader, "open_dataset_kwargs"):
        orig_chunks = reader.open_dataset_kwargs.get("chunks")
        reader.open_dataset_kwargs["chunks"] = None
    if hasattr(reader, "use_tqdm"):  # pragma: no branch
        orig_tqdm = reader.use_tqdm
        new_tqdm = stepsize == 1
        reader.use_tqdm = new_tqdm

    outfname = str(outfname)
    timestamps = reader.tstamps_for_daterange(start, end)

    # figure out coordinates and dimensions based on test image
    # for reading the test image, disable tqdm
    testds = reader._testimg()
    coords = dict(testds.coords)
    coords[reader.timename] = timestamps
    # to get the correct order of dimensions, we use the first variable and
    # assert that the other variables follow the same order
    testvar = testds[reader.varnames[0]]
    dims = dict(zip(testvar.dims, testvar.shape))
    for var in reader.varnames:
        testvar = testds[var]
        assert dims == dict(zip(testvar.dims, testvar.shape))
    del dims[reader.timename]
    dims[reader.timename] = len(timestamps)
    new_dimsizes = tuple(size for size in dims.values())

    # calculate chunk sizes if they are not given
    maxdtypesize = 0
    for var in reader.varnames:
        maxdtypesize = max(maxdtypesize, testds[var].dtype.itemsize)
    if chunks is None:
        if outfname.endswith(".zarr"):
            chunksizes = infer_chunksizes(new_dimsizes, 100, maxdtypesize)
        else:
            chunksizes = infer_chunksizes(new_dimsizes, 1, maxdtypesize)
        chunks = dict(zip(dims, chunksizes))
        chunksizes = list(chunksizes)
    else:
        new_chunks = {}
        for i, name in enumerate(dims):
            if name in chunks and chunks[name] > 0:
                new_chunks[name] = chunks[name]
            else:
                new_chunks[name] = new_dimsizes[i]
        chunks = new_chunks
        chunksizes = list(chunks.values())[:-1] + [len(timestamps)]
    logging.info(f"Chunking to: {chunks}")

    # check if target chunk sizes are not too big
    chunksizes = list(chunks.values())
    if (
        np.prod(chunksizes) * maxdtypesize > 100 * 1024**2
    ):  # pragma: no cover
        logging.warn(
            "The specified chunksizes will lead to chunks larger than 100MB!"
        )

    # The intermediate chunks have 2 size constraints:
    # 1) The size in the time dimension, nsteps, must meet:
    #        imagesize * nsteps <= memory * safety_factor
    #    because we will always read in nsteps images at once.
    #    Reformulated, we get: nsteps = memory/imagesize * safetyfactor
    # 2) The "chunk base size" (i.e. size of the chunk with only one timestep)
    #    must be small enough that always 2 chunks with full time fit into
    #    memory, i.e with ntimes the total number of timesteps, we have:
    #        chunkbasesize * ntimes <= memory * 2
    #    or
    #        chunkbasesize = memory*2/ntimes
    # Additionally, to avoid too large chunks, we want to limit the chunk size
    # to 100MB, i.e. chunkbasesize * nsteps <= 100MB
    ntimes = len(reader.timestamps)

    # First we get the number of allowed steps, with safety factor 2
    maxnsteps = nimages_for_memory(testds, memory) // 2
    if stepsize is None:
        nsteps = min(maxnsteps, ntimes)
    else:
        nsteps = min(stepsize, maxnsteps, ntimes)
    logging.info(f"write_transposed_dataset: Using {nsteps} images per step")

    # Now we get the chunk base size based on the 2 constraints
    size1 = 2 * memory / ntimes * 1024**3  # in bytes (memory was in GB)
    size2 = 100 / nsteps * 1024**2  # in bytes (100 was in MB)
    chunkbasesize = min(size1, size2)

    # To convert the chunkbasesize to a chunk shape, we use the maxdtypesize so
    # that we can be sure that our constraints from above are met for all
    # variables. As a starting point we use the size of the final chunks, and
    # then just adapt the chunk size of the first dimension.
    # The current chunk base size can be calculated with
    #     n_0 * n_1 * ... n_{d-2} * dtypesize = chunk base size
    # if (n_0, ..., n_{d-1}) is the chunk shape (with time dimension).
    # So we get for n_0:
    #     n_0 = (chunk base size) / (dtypesize * n_1 * ... * n_{d-2})
    tmp_chunksizes = copy.copy(chunksizes)
    tmp_chunksizes[-1] = nsteps
    # product of other chunk dimensions than first and last
    nother = np.prod(tmp_chunksizes) / (nsteps * tmp_chunksizes[0])
    n0 = int(np.floor(chunkbasesize / (maxdtypesize * nother)))
    n0 = min(n0, new_dimsizes[0])
    tmp_chunksizes[0] = n0
    # now our constraint should be met
    assert np.prod(tmp_chunksizes[:-1]) * maxdtypesize <= chunkbasesize
    assert np.prod(tmp_chunksizes) * maxdtypesize <= 100 * 1024**2
    logging.info(f"Intermediate chunksizes: {tmp_chunksizes}")

    # create zarr arrays
    tmp_stores = {}
    tmp_fnames = {}
    for var in reader.varnames:
        tmp_fnames[var] = outfname + "." + var + ".tmp.zarr"
        dtype = testds[var].dtype
        default_fill_value = (
            -9999 if np.issubdtype(np.int32, np.integer) else np.nan
        )
        fill_value = testds[var].attrs.get("_FillValue", default_fill_value)
        tmp_stores[var] = zarr.create(
            new_dimsizes,
            chunks=tuple(tmp_chunksizes),
            store=tmp_fnames[var],
            overwrite=True,
            fill_value=fill_value,
            dtype=dtype,
        )

    # Write the images to zarr files, one file for each variable

    pbar = tqdm(range(0, len(timestamps), nsteps))
    for start_idx in pbar:
        pbar.set_description("Reading")
        end_idx = min(start_idx + nsteps - 1, len(timestamps) - 1)

        block = reader.read_block(
            timestamps[start_idx], timestamps[end_idx]
        ).compute()
        block = block.transpose(..., reader.timename)
        pbar.set_description("Writing")
        for var in reader.varnames:
            tmp_stores[var][..., start_idx : end_idx + 1] = block[var].values

    # Now we just have to assemble all arrays to a single big dataset
    arrays = {}
    encoding = {}
    for var in reader.varnames:
        arr = da.from_zarr(tmp_fnames[var])
        arrays[var] = xr.DataArray(arr, dims=dims, coords=coords)
        arrays[var].name = var
        arrays[var].attrs.update(reader.array_attrs[var])
        arrays[var] = arrays[var].chunk(chunks)
        if outfname.endswith(".zarr"):
            encoding[var] = {
                "chunks": tuple(size for size in chunks.values()),
                "compressor": Blosc(cname="zstd", clevel=complevel),
            }
        else:
            encoding[var] = {
                "zlib": zlib,
                "complevel": complevel,
                "chunksizes": tuple(size for size in chunks.values()),
            }
    ds = xr.Dataset(arrays)
    ds.attrs.update(reader.global_attrs)

    # Now we can write the dataset
    logging.info(
        f"write_transposed_dataset: Writing combined file to {str(outfname)}"
    )
    if outfname.endswith(".zarr"):
        ds.to_zarr(outfname, mode="w", consolidated=True)
    else:
        ds.to_netcdf(outfname, encoding=encoding)

    # restore the reader settings
    if hasattr(reader, "use_tqdm"):  # pragma: no branch
        reader.use_tqdm = orig_tqdm
    if hasattr(reader, "open_dataset_kwargs"):  # pragma: no branch
        reader.open_dataset_kwargs["chunks"] = orig_chunks

    for fname in tmp_fnames.values():
        shutil.rmtree(fname)
    logging.info("write_transposed_dataset: Finished writing transposed file.")
