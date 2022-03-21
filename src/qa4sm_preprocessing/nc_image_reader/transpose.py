import copy
import dask
import dask.array as da
from dask.distributed import Client
import datetime
import logging
import math
from multiprocessing.pool import ThreadPool
import numpy as np
from pathlib import Path
from tqdm.auto import tqdm
from typing import Union, TypeVar, Tuple
import xarray as xr
import shutil
import warnings
import zarr


from .utils import infer_chunks
from .readers import DirectoryImageReader


Reader = TypeVar("Reader")


def write_transposed_dataset(
    reader: Reader,
    outfname: Union[Path, str],
    start: datetime.datetime = None,
    end: datetime.datetime = None,
    chunks: dict = None,
    memory: float = 2,
    n_threads: int = 2,
    zlib: bool = True,
    complevel: int = 4,
    distributed: Union[bool, Client] = False,
    use_dask: bool = True,
):
    """
    Creates a stacked and transposed netCDF file from a given reader.

    WARNING: very experimental!

    Parameters
    ----------
    reader : XarrayImageReaderBase
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
        size of 50MB are used for zarr output.
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
    use_dask : bool, optional
        Whether to use dask for the transposing. Default is True, but sometimes
        (especially with large datasets) this fails. If set to False, the data
        is written to an intermediate zarr store.
    """
    dask_config = {
        "array.slicing.split_large_chunks": False,
    }
    args = (reader, outfname)
    kwargs = {
        "start": start,
        "end": end,
        "memory": memory,
        "zlib": zlib,
        "complevel": complevel,
        "chunks": chunks,
    }
    if not use_dask:
        _transpose_no_dask(*args, **kwargs)
    elif isinstance(distributed, Client) or not distributed:
        if not distributed:
            logging.info(
                f"write_transposed_dataset: using thread pool with {n_threads}"
            )
            dask_config.update(
                {"scheduler": "threads", "pool": ThreadPool(n_threads)}
            )
        else:
            logging.info(
                "write_transposed_dataset: using distributed scheduler"
                f" available at {distributed.dashboard_link}"
            )
        with dask.config.set(**dask_config):
            _transpose(*args, **kwargs)
    elif distributed:
        with dask.config.set(**dask_config), Client(
            n_workers=1,
            threads_per_worker=n_threads,
            memory_limit=f"{n_threads*memory}GB",
        ) as client:
            print("Dask dashboard accessible at:", client.dashboard_link)
            _transpose(*args, **kwargs)


def _get_intermediate_chunks(array, chunks, new_last_dim, zarr_output, memory):
    """
    Calculates chunk sizes for the given array for the intermediate output
    files.

    Parameters
    ----------
    array : xr.DataArray
        Array to rechunk and transpose
    chunks : dict or None
        Chunks passed to write_transposed_dataset, None if none were given.
    new_last_dim : str
        Name of the new last dimension, normally "time".
    zarr_output : bool
        Whether the final file will be a zarr file (True) or a netCDf (False).
    memory : float
        The amount of memory to be used for buffering in GB.

    Returns
    -------
    tmp_chunks : dict
        Chunks to be used for rechunking the array to a temporary file. The
        order of keys corresponds to the order of dimensions in the transposed
        array.
    """
    dtype = array.dtype
    dims = dict(zip(array.dims, array.shape))
    transposed_shape = [
        length for dim, length in dims.items() if dim != new_last_dim
    ]
    transposed_shape.append(dims[new_last_dim])

    # If the chunks argument was not given, we have to infer the spatial
    # and temporal chunks for the intermediate file.
    # The spatial chunks will be set such that for a continuous time
    # dimension the chunk size is still reasonable.
    if chunks is None:
        if zarr_output:
            chunksizes = infer_chunks(transposed_shape, 100, dtype)[:-1]
        else:
            chunksizes = infer_chunks(transposed_shape, 1, dtype)[:-1]
        chunks = dict(
            zip([dim for dim in dims if dim != new_last_dim], chunksizes)
        )
        chunks[new_last_dim] = -1
    else:
        chunks = copy.copy(chunks)
    tmp_chunks = {dim: chunks[dim] for dim in dims if dim != new_last_dim}

    # figure out temporary chunk sizes based on image size and available memory
    size = dtype.itemsize
    chunksizes = [
        size if size != -1 else dims[dim] for dim, size in chunks.items()
    ]
    chunksize_MB = np.prod(chunksizes) * size / 1024 ** 2
    img_shape = transposed_shape[:-1]
    len_time = transposed_shape[-1]
    imagesize_GB = np.prod(img_shape) * size / 1024 ** 3
    # we need to divide by two, because we need intermediate storage for
    # the transposing
    stepsize = int(math.floor(memory / imagesize_GB)) // 2
    stepsize = min(stepsize, len_time)

    tmp_chunks[new_last_dim] = stepsize
    tmp_chunks_str = str(tuple(tmp_chunks.values()))
    logging.info(
        f"write_transposed_dataset: Creating chunks {tmp_chunks_str}"
        f" with chunksize {chunksize_MB:.2f} MB for {array.name}."
    )
    return tmp_chunks


def _transpose(
    reader: Reader,
    outfname: Union[Path, str],
    start: datetime.datetime = None,
    end: datetime.datetime = None,
    chunks: dict = None,
    memory: float = 2,
    zlib: bool = True,
    complevel: int = 4,
):
    zarr_output = str(outfname).endswith(".zarr")
    new_last_dim = reader.timename

    if isinstance(reader, DirectoryImageReader) and reader.chunks is None:
        logging.info(
            "You are using DirectoryImageReader without dask. If you run into"
            " memory issues or have large datasets to transpose, consider"
            " setting use_dask=True in the constructor of DirectoryImageReader"
        )

    tmp_outfname = str(outfname) + f".tmp.zarr"
    if Path(tmp_outfname).exists():
        logging.info(
            f"Skipping generating intermediate file {tmp_outfname}"
            " because it exists"
        )
    else:
        logging.info("write_transposed_dataset: Reading block of data")
        ds = reader.read_block(start, end)

        # We process each variable separately and store them as intermediately
        # chunked temporary files. The chunk size in time dimension is inferred
        # from the given memory.
        for var in reader.varnames:

            tmp_chunks = _get_intermediate_chunks(
                ds[var], chunks, new_last_dim, zarr_output, memory
            )

            # make sure that the time dimension will be continuous in the final
            # output
            chunks = copy.copy(tmp_chunks)
            chunks[new_last_dim] = len(ds[var].time)
            ds[var] = ds[var].chunk(tmp_chunks)

        ds = ds.transpose(..., new_last_dim)
        logging.info(
            f"write_transposed_dataset: Writing to temporary file: "
            f"{tmp_outfname}"
        )
        ds.to_zarr(tmp_outfname, consolidated=True)
    ds = xr.open_zarr(tmp_outfname, consolidated=True)

    encoding = {}
    for var in reader.varnames:
        transposed_dims = ds[var].dims
        var_chunks = tuple(chunks[dim] for dim in transposed_dims)
        if not zarr_output:
            encoding[var] = {
                "zlib": zlib,
                "complevel": complevel,
                "chunksizes": var_chunks,
            }
        else:
            del ds[var].encoding["chunks"]
            del ds[var].encoding["preferred_chunks"]
            ds[var] = ds[var].chunk(var_chunks)

    ds.attrs.update(reader.global_attrs)
    logging.info(
        "write_transposed_dataset: Writing final output file: " f"{outfname}"
    )
    if not zarr_output:
        ds.to_netcdf(outfname, encoding=encoding)
    else:
        ds.to_zarr(outfname, mode="w", consolidated=True)

    shutil.rmtree(tmp_outfname)
    logging.info("write_transposed_dataset: Finished writing transposed file.")


def _transpose_no_dask(
    reader: Reader,
    outfname: Union[Path, str],
    start: datetime.datetime = None,
    end: datetime.datetime = None,
    chunks: Tuple = None,
    memory: float = 2,
    zlib: bool = True,
    complevel: int = 4,
):
    warnings.warn(
        "This is an experimental function and not yet ready for public use!"
    )
    zarr_output = str(outfname).endswith(".zarr")
    new_last_dim = reader.timename
    timestamps = reader.tstamps_for_daterange(start, end)

    variable_fnames = {}
    variable_dims = {}
    for varname in reader.varnames:

        tmp_outfname = str(outfname) + f".{varname}.zarr"
        variable_fnames[varname] = tmp_outfname

        # first, get some info about structure of the input file
        first_img = reader.read_block(start=timestamps[0], end=timestamps[0])[
            varname
        ]
        tmp_chunks = _get_intermediate_chunks(
            first_img, chunks, new_last_dim, zarr_output, memory
        )

        # get new dim names in the correct order
        new_dim_names = list(tmp_chunks)
        variable_dims[varname] = new_dim_names

        # this happens this late because we need to set
        # `variable_dims[varname]` in any case
        if Path(tmp_outfname).exists():
            logging.info(f"{str(tmp_outfname)} already exists, skipping.")
            continue

        logging.debug(
            f"write_transposed_dataset: starting zarr array creation"
            f" for {len(timestamps)} timestamps"
        )

        # get shape of transposed target array
        dims = dict(zip(first_img.dims, first_img.shape))
        new_dim_sizes = tuple(dims[name] for name in new_dim_names)
        zarr_array = zarr.create(
            tuple(new_dim_sizes),
            chunks=tuple(size for size in tmp_chunks.values()),
            store=tmp_outfname,
            overwrite=True,
            fill_value=np.nan,
        )

        logging.debug(f"write_transposed_dataset: Writing {tmp_outfname}")
        print(f"Constructing array stack for {varname}:")
        stepsize = tmp_chunks[new_last_dim]
        pbar = tqdm(range(0, len(timestamps), stepsize))
        for start_idx in pbar:
            pbar.set_description("Reading")
            end_idx = min(start_idx + stepsize - 1, len(timestamps) - 1)
            block = reader.read_block(
                timestamps[start_idx], timestamps[end_idx]
            )[varname]
            block = block.transpose(..., new_last_dim)
            pbar.set_description("Writing")
            zarr_array[..., start_idx : end_idx + 1] = block.values

    variable_arrays = {}
    encoding = {}
    for varname, fname in variable_fnames.items():
        logging.debug(f"Reading {str(fname)}")
        arr = da.from_zarr(fname)
        dims = variable_dims[varname]
        metadata = reader.array_attrs[varname]
        if chunks is None:
            if zarr_output:
                chunks = infer_chunks(new_dim_sizes, 100, arr.dtype)
            else:
                # netCDF chunks should be about 1MB
                chunks = infer_chunks(new_dim_sizes, 1, arr.dtype)
        encoding[varname] = {
            "chunksizes": chunks,
            "zlib": zlib,
            "complevel": complevel,
        }
        chunk_dict = dict(zip(dims, chunks))
        arr = xr.DataArray(data=arr, dims=dims, attrs=metadata)
        arr = arr.chunk(chunk_dict)
        arr.encoding = encoding[varname]
        variable_arrays[varname] = arr

    logging.debug("Reading test image")
    test_img = reader.read_block(start=timestamps[0], end=timestamps[0])[
        reader.varnames[0]
    ]
    coords = {
        c: test_img.coords[c] for c in test_img.coords if c != reader.timename
    }
    coords[reader.timename] = timestamps
    logging.debug("Creating dataset")
    ds = xr.Dataset(variable_arrays, coords=coords,)
    ds.attrs.update(reader.global_attrs)

    logging.info(
        f"write_transposed_dataset: Writing combined file to {str(outfname)}"
    )
    if not zarr_output:
        ds.to_netcdf(outfname, encoding=encoding)
    else:
        ds.to_zarr(outfname, mode="w", consolidated=True)

    for fname in variable_fnames.values():
        shutil.rmtree(fname)
    logging.info("write_transposed_dataset: Finished writing transposed file.")
