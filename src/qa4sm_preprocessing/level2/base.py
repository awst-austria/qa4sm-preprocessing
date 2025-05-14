from abc import abstractmethod
import argparse
import logging
import numpy as np
import os
from pathlib import Path
from typing import Mapping, Tuple, Union, List

from qa4sm_preprocessing.reading import DirectoryImageReader
from qa4sm_preprocessing.reading.base import GridInfo
from qa4sm_preprocessing.reading.exceptions import ReaderError
from qa4sm_preprocessing.reading.utils import mkdate, str2bool


class L2Reader(DirectoryImageReader):
    """
    Base class for Level 2 orbit file readers.

    Level 2 data can often not be read directly with the DirectoryImageReader,
    because the orbit files typically do not contain a global grid, but only the
    grid points of the orbit.

    This base class should make it easier to create custom level 2 image reader
    which can then be used for reshuffling the data to the pynetcf timeseries
    format. To create a subclass for a specific level 2 dataset, the abstract
    methods defined here have to be overriden:
    - ``_read_l2_file``: This routine reads a single orbit file and returns the
      data, as well as indices of the data in the global grid.
    - ``_time_regex_pattern``: Property that specifies how to extract the
      timestamp from the filename (and how to find files)
    - ``_gridinfo``: Returns a ``GridInfo`` object for the global grid.
    - ``_variable_metadata``: Function that specifies the metadata for all the
      variables of interest, .e.g units, long name, validity bounds, etc.
    - ``_global_metadata``: Function that specifies metadata for the full
      dataset, e.g. name, references, version, etc. This is optional, but it is
      recommended to specify it.

    Parameters
    ----------
    directory : path or str
        Directory where to find the level 2 data.
    varnames : list, optional (default: None)
        List of variable names to extract. By default, all variables in
        ``self._variable_metadata`` are read.
    fmt : str, optional (default: "%Y%m%dT%H%M%S")
        Format of the time string in the filename.
    pattern : str, optional (default: None)
        Pattern that the orbit files have to match. By default, the extension
        of ``self._time_regex_pattern`` is extracted and all files ending with
        this extension are used. If another pattern should be used and the
        files are in nested directories, it should probably follow this format:
        ``"**/<pattern>"``.
    add_attrs : dict, optional (default: None)
        Additional variable attributes that cannot be taken from the input data.
    """

    @abstractmethod
    def _read_l2_file(
        self, fname: Union[Path, str]
    ) -> Tuple[Mapping[str, np.ndarray], Union[np.ndarray, Tuple[np.ndarray]]]:
        """
        Reads a single orbit L2 orbit file

        Parameters
        ----------
        fname : path
            Name of the orbit file.

        Returns
        -------
        orbitdata : dict of np.ndarrays
            This is a dictionary mapping from variable names to 1-d arrays
            containing the orbit data.
        gridids : np.ndarray or tuple of np.ndarrays
            If the grid is unstructured, a single array of grid indices,
            otherwise a tuple of column indices and row indices.
        """
        ...  # pragma: no cover

    @property
    @abstractmethod
    def _time_regex_pattern(self) -> str:
        """
        Regular expression pattern string to extract time information.
        Must end with ``.<ext>`` where ``<ext>`` is the filename extension,
        e.g. ".nc" or ".h5".

        By default we assume that the time pattern has the format
        "%Y%m%dT%H%M%S". If it has another format, the ``fmt`` keyword has to
        be passed to the constructor.
        """
        ...  # pragma: no cover

    @abstractmethod
    def _gridinfo(self) -> GridInfo:
        # override, should return a GridInfo object
        ...  # pragma: no cover

    def _gridinfo_from_dataset(self, ds):
        return self._gridinfo()

    @abstractmethod
    def _variable_metadata(self) -> Mapping[str, Mapping[str, str]]:
        # override, should return a dictionary of variable names mapping to
        # variable attributes (which is also a dictionary)
        ...  # pragma: no cover

    def _global_metadata(self) -> dict:
        # should also be overriden, but default is to not override anything
        return {}  # pragma: no cover

    def __init__(
        self,
        directory: Union[Path, str],
        varnames: List[str],
        fmt="%Y%m%dT%H%M%S",
        pattern=None,
        add_attrs=None,
    ):
        gridinfo = self._gridinfo()
        self.gridshape = gridinfo.shape
        time_regex_pattern = self._time_regex_pattern
        if pattern is None:
            file_extension = os.path.splitext(time_regex_pattern)[-1]
            pattern = f"**/*{file_extension}"
        super().__init__(
            directory,
            varnames,
            fmt=fmt,
            time_regex_pattern=time_regex_pattern,
            pattern=pattern,
            add_attrs=add_attrs,
        )

    def _read_file(self, fname) -> Mapping[str, np.ndarray]:
        # overrides the DirectoryImageReader function to be more specific to L2
        # data where we normally don't get nice xarray datasets
        orbitdata, gridids = self._read_l2_file(fname)
        data = {}
        for var in self.varnames:
            data[var] = np.full(self.gridshape, np.nan, dtype=orbitdata[var].dtype)
            self._add_orbitdata(data[var], orbitdata[var], gridids)
            data[var] = data[var][np.newaxis, ...]  # expand the time dimension
        return data

    def _add_orbitdata(self, data: np.ndarray, orbitdata: np.ndarray, gridids):
        if self.gridtype == "unstructured":
            # 1D data array and gridids is an 1D array of gpis
            gpis = self.grid.get_grid_points()[0]
            _, idx1, idx2 = np.intersect1d(
                gridids, gpis, return_indices=True, assume_unique=True
            )
            data[idx2] = orbitdata[idx1]
        else:
            data[gridids[1], gridids[0]] = orbitdata

    def _metadata_from_dataset(self, fname: Union[Path, str]) -> Tuple[dict, dict]:
        global_attrs = self._global_metadata(fname)
        array_attrs = self._variable_metadata(fname)
        return global_attrs, array_attrs #{v: array_attrs[v] for v in self.varnames}

    def _dtype_from_dataset(self, fname: Union[Path, str]) -> Mapping:
        # can also be overriden if the datatypes are known beforehand
        data, _ = self._read_l2_file(fname)
        return {v: data[v].dtype for v in data}

    def _open_dataset(self, fname):
        # the _open_dataset method should return a xarray Dataset, but we don't
        # want to do the effort of creating one every time a file is read,
        # therefore we override _read_single_file and disable _open_dataset
        raise NotImplementedError  # pragma: no cover


def _repurpose_level2_parse_cli_args(description):  # pragma: no cover
    parser = argparse.ArgumentParser(description)
    parser.add_argument("input_path", help="Path where the L2 data is stored.")
    parser.add_argument("output_path", help="Path where the output should be stored.")

    parser.add_argument(
        "--start",
        type=mkdate,
        help=("Startdate. Either in format YYYY-MM-DD or " "YYYY-MM-DDTHH:MM."),
        default=None,
    )
    parser.add_argument(
        "--end",
        type=mkdate,
        help=("Enddate. Either in format YYYY-MM-DD or " "YYYY-MM-DDTHH:MM."),
        default=None,
    )
    parser.add_argument(
        "--zlib",
        type=str2bool,
        default=True,
        help="Whether to use compression or not. Default is true",
    )
    parser.add_argument(
        "--memory",
        type=float,
        default=2,
        help="The amount of memory to use as buffer in GB",
    )
    parser.add_argument(
        "--logfile",
        default=None,
        help="File for logging output",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Whether to overwrite an existing output directory.",
    )
    parser.add_argument(
        "--parameter",
        type=str,
        nargs="+",
        help="Parameters to process.",
    )
    parser.add_argument(
        "--only_land",
        action="store_true",
        help="Process only land pixels.",
    )
    return parser.parse_args()
