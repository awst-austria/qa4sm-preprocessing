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
from typing import Union, TypeVar, Tuple, Sequence
import xarray as xr
import shutil
import warnings
import zarr


from .utils import infer_chunks
from qa4sm_preprocessing.nc_image_reader.readers import DirectoryImageReader


Reader = TypeVar("Reader")


def create_transposed_netcdf(
    reader: Reader,
    outfname: Union[Path, str],
    new_last_dim: str = None,
    start: datetime.datetime = None,
    end: datetime.datetime = None,
    chunks: Tuple = None,
    memory: float = 2,
    n_threads: int = 4,
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
    chunks : tuple, optional
        The chunk sizes that are used for the transposed file. The dimension
        order must correspond to the order of the transposed file. If none is
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
        "new_last_dim": new_last_dim,
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
            dask_config.update(
                {"scheduler": "threads", "pool": ThreadPool(n_threads)}
            )
        with dask.config.set(**dask_config):
            _transpose(*args, **kwargs)
    elif distributed:
        with dask.config.set(**dask_config), Client(
            n_workers=1,
            threads_per_worker=n_threads,
            memory_limit=f"{memory}GB",
        ) as client:
            print("Dask dashboard accessible at:", client.dashboard_link)
            _transpose(*args, **kwargs)


def _get_additional_variables(reader, start, variables):
    img = reader._read_image(start)
    return {var: img[var] for var in variables}


def _transpose(
    reader: Reader,
    outfname: Union[Path, str],
    new_last_dim: str = None,
    start: datetime.datetime = None,
    end: datetime.datetime = None,
    chunks: Tuple = None,
    memory: float = 2,
    zlib: bool = True,
    complevel: int = 4,
):
    zarr_output = str(outfname).endswith(".zarr")
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
        if chunks is None:
            if zarr_output:
                chunks = infer_chunks(new_dim_sizes, 100, dtype)
            else:
                chunks = infer_chunks(new_dim_sizes, 1, dtype)
        variable_chunks.append(chunks)

        # calculate intermediate chunk size in time direction
        size = dtype.itemsize
        chunksize_MB = np.prod(chunks) * size / 1024 ** 2
        logging.info(
            f"create_transposed_netcdf: Creating chunks {str(chunks)}"
            f" with chunksize {chunksize_MB:.2f} MB for {varname}"
        )
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

            print("Constructing array stack:")
            for t in tqdm(timestamps):
                arr = _get_single_image_as_array(t)
                arrays.append(arr)

            logging.debug("create_transposed_netcdf: stacking")
            all_data = da.stack(arrays).rechunk(dask_chunks)

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
    encoding = {
        varname: {
            "chunksizes": variable_chunks[j],
            "zlib": zlib,
            "complevel": complevel,
        }
        for j, varname in enumerate(reader.varnames)
    }

    if not zarr_output:
        ds.to_netcdf(outfname, encoding=encoding)
    else:
        ds.to_zarr(outfname, mode="w", consolidated=True)

    # reset the grid for the reader and remove temporary files
    reader.grid = grid
    reader.timestamps = timestamps
    for fname in variable_fnames:
        shutil.rmtree(fname)
    logging.info("create_transposed_netcdf: Finished writing transposed file.")


def _transpose_no_dask(
    reader: Reader,
    outfname: Union[Path, str],
    new_last_dim: str = None,
    start: datetime.datetime = None,
    end: datetime.datetime = None,
    chunks: Tuple = None,
    memory: float = 2,
    zlib: bool = True,
    complevel: int = 4,
):
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
        variable_dims[varname] = new_dim_names

        if Path(tmp_outfname).exists():
            logging.info(f"{str(tmp_outfname)} already exists, skipping.")
            continue

        # get shape of transposed array
        new_dim_sizes = [
            input_dim_sizes[input_dim_names.index(dim)]
            for dim in new_dim_names
        ]
        new_dim_sizes[-1] = len(timestamps)
        new_dim_sizes = tuple(new_dim_sizes)

        # The intermediate zarr array will be chunked in space and in time.
        # Preferably, we use the output chunks in space, otherwise we use
        # chunks of about 100MB size
        if chunks is None:
            chunks = infer_chunks(new_dim_sizes, 100, dtype)
        chunks = tuple(
            [
                cnks if cnks != -1 else sz
                for cnks, sz in zip(chunks, new_dim_sizes)
            ]
        )

        # We will always read an image stack based on the available memory, so
        # we have to find the size of an image
        size = dtype.itemsize
        imagesize_GB = np.prod(new_dim_sizes[:-1]) * size / 1024 ** 3
        # we need to divide by two, because we need intermediate storage for
        # the transposing, and somehow we still need another factor of 2
        stepsize = int(math.floor(memory / imagesize_GB)) // 4
        len_new_dim = new_dim_sizes[-1]
        stepsize = min(stepsize, len_new_dim)
        logging.info(
            f"create_transposed_netcdf: Using {stepsize} images as buffer"
        )
        tmp_chunks = list(chunks)
        tmp_chunks[-1] = stepsize
        tmp_chunks = tuple(tmp_chunks)

        logging.debug(
            f"create_transposed_netcdf: starting zarr array creation"
            f" for {len(timestamps)} timestamps"
        )

        zarr_array = zarr.create(
            tuple(new_dim_sizes),
            chunks=tmp_chunks,
            store=tmp_outfname,
            overwrite=True,
            fill_value=np.nan,
        )

        logging.debug(f"create_transposed_netcdf: Writing {tmp_outfname}")
        print(f"Constructing array stack for {varname}:")
        pbar = tqdm(range(0, len(timestamps), stepsize))
        for start_idx in pbar:
            pbar.set_description("Reading")
            end_idx = min(start_idx + stepsize - 1, len(timestamps) - 1)
            block = reader.read_block(
                timestamps[start_idx], timestamps[end_idx]
            )[varname]
            block = block.transpose(..., reader.timename)
            pbar.set_description("Writing")
            zarr_array[..., start_idx : end_idx + 1] = block.values

    variable_arrays = {}
    encoding = {}
    for varname, fname in variable_fnames.items():
        logging.debug(f"Reading {str(fname)}")
        arr = da.from_zarr(fname)
        dims = variable_dims[varname]
        metadata = reader.array_metadata[varname]
        if chunks is None:
            if zarr_output:
                chunks = infer_chunks(new_dim_sizes, 100, dtype)
            else:
                # netCDF chunks should be about 1MB
                chunks = infer_chunks(new_dim_sizes, 1, dtype)
        encoding[varname] = {
            "chunksizes": chunks,
            "zlib": zlib,
            "complevel": complevel,
        }
        chunk_dict = dict(zip(dims, chunks))
        arr = xr.DataArray(data=arr, dims=dims, attrs=metadata)
        arr = arr.chunk(chunk_dict)
        arr.encoding = encoding[varname]
        # we're writing again to a temporary file, because otherwise the
        # dataset creation fails because dask sucks
        # arr.to_dataset(name=varname).to_zarr(fname + ".tmp", consolidated=True)
        # variable_arrays[varname] = xr.open_zarr(fname + ".tmp", consolidated=True)
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
    ds = xr.Dataset(
        variable_arrays,
        coords=coords,
    )
    ds.attrs.update(reader.dataset_metadata)

    logging.info(
        f"create_transposed_netcdf: Writing combined file to {str(outfname)}"
    )
    if not zarr_output:
        ds.to_netcdf(outfname, encoding=encoding)
    else:
        ds.to_zarr(outfname, mode="w", consolidated=True)

    for fname in variable_fnames.values():
        shutil.rmtree(fname)
    logging.info("create_transposed_netcdf: Finished writing transposed file.")
