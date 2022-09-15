import logging
from pathlib import Path
from tqdm.auto import tqdm
from typing import Union
import xarray as xr

from .utils import nimages_for_memory


def write_images(
    dataset: Union[xr.Dataset, xr.DataArray],
    directory: Union[Path, str],
    dsname: str,
    fmt: str = "%Y%m%dT%H%M",
    dim: str = "time",
    memory: float = 5,
    stepsize: int = 1,
    invertlats: bool = False,
    includetime: bool = True,
):
    """
    Writes a xr.Dataset as images to a directory.

    Parameters
    ----------
    dataset : xr.Dataset
        Dataset to be written to single images files.
    directory : path
        Directory to which the image files are written.
    dsname : str
        Name of the dataset. The final image file names will follow the pattern
        "{dsname}_{fmt}.nc"
    fmt : str, optional (default: "%Y%m%dT%H%M")
        Format string for creating timestamps for the filenames.
    dim : str, optional (default: "time")
        Name of the time dimension.
    memory : float, optional (default: 5)
        Available memory
    stepsize : int, optional (default: 1)
        Number of timesteps that are written into a single image.  The first
        step is used for generating the time stamp.
    invertlats : bool, optional (default: False)
        Whether to ensure that the latitude axis is inverted.
    """
    if isinstance(dataset, xr.DataArray):
        dataset = dataset.to_dataset(name=dsname)
    directory = Path(directory)
    directory.mkdir(exist_ok=True, parents=True)
    ntime = len(dataset[dim])
    # we always load a stack of half the available memory size into memory
    stacksize = nimages_for_memory(dataset.isel(**{dim: 0}), memory / 2)
    # if stepsize is not 1 we need to make sure stacksize is a multiple of the
    # stepsize
    nsteps = stacksize // stepsize
    stacksize = nsteps * stepsize

    for startidx in tqdm(range(0, ntime, stacksize)):
        endidx = min(startidx + stacksize, ntime)
        stack = dataset.isel(**{dim: slice(startidx, endidx)}).compute()
        if invertlats and stack.lat[0] < stack.lat[1]:
            # invert latitudes
            stack = stack.sel(lat=stack.lat[::-1])
        ntime_stack = len(stack.indexes[dim])
        for timeidx in range(0, ntime_stack, stepsize):
            time = stack.indexes[dim][timeidx]
            fname = directory / time.strftime(f"{dsname}_{fmt}.nc")
            img = stack.sel(
                **{dim: stack.indexes[dim][timeidx : timeidx + stepsize]}
            )
            if not includetime and stepsize == 1:
                img = img.drop_vars(dim).isel(**{dim: 0})
            img.to_netcdf(
                fname,
                encoding={
                    varname: {"zlib": True, "complevel": 5}
                    for varname in dataset.data_vars
                },
            )
    logging.info(f"Finished writing images to {str(directory)}")
