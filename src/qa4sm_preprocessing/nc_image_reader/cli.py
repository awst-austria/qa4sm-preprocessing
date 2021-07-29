# The MIT License (MIT)
#
# Copyright (c) 2020, TU Wien
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""
Module for a command line interface to convert the netcdf image data into
timeseries format or a transposed netcdf with time as last dimension
"""

import argparse
from pathlib import Path
import sys

from repurpose.img2ts import Img2Ts

from .readers import XarrayImageReader, DirectoryImageReader
from .transpose import create_transposed_netcdf
from .utils import mkdate, str2bool


class ReaderArgumentParser(argparse.ArgumentParser):
    def __init__(self, description):
        super().__init__(description=description)

        # common arguments for both scripts
        self.add_argument(
            "dataset_root",
            help=(
                "Path where the data is stored, either"
                " a directory or a netCDF file"
            ),
        )
        self.add_argument(
            "output_root", help="Path where the output should be stored."
        )
        self.add_argument(
            "start",
            type=mkdate,
            help=(
                "Startdate. Either in format YYYY-MM-DD or "
                "YYYY-MM-DDTHH:MM."
            ),
        )
        self.add_argument(
            "end",
            type=mkdate,
            help=(
                "Enddate. Either in format YYYY-MM-DD or " "YYYY-MM-DDTHH:MM."
            ),
        )
        self.add_argument(
            "--parameter",
            type=str,
            required=True,
            nargs="+",
            help="Parameters to process.",
        )
        self.add_argument(
            "--pattern",
            type=str,
            default="*.nc",
            help=(
                "If DATASET_ROOT is a directory, glob pattern to match files"
                " Default is '*.nc'"
            ),
        )
        self.add_argument(
            "--time_fmt",
            type=str,
            help=(
                "If DATASET_ROOT is a directory, strptime format string to"
                " deduce the data from the filenames. This can improve the"
                " performance significantly."
            ),
        )
        self.add_argument(
            "--time_regex_pattern",
            type=str,
            help=(
                "If DATASET_ROOT is a directory, a regex pattern to select"
                " the time string from the filename. If this is used, TIME_FMT"
                " must be chosen accordingly. See nc_image_reader.readers for"
                " more info."
            ),
        )
        self.add_argument(
            "--latname",
            type=str,
            default="lat",
            help="Name of the latitude coordinate. Default is 'lat'",
        )
        self.add_argument(
            "--latdim",
            type=str,
            help="Name of the latitude dimension (e.g. north_south)",
        )
        self.add_argument(
            "--lonname",
            type=str,
            default="lon",
            help="Name of the longitude coordinate. Default is 'lon'",
        )
        self.add_argument(
            "--londim",
            type=str,
            help="Name of the longitude dimension (e.g. east_west).",
        )
        self.add_argument(
            "--locdim",
            type=str,
            help="Name of the location dimension for non-regular grids.",
        )
        self.add_argument(
            "--lat",
            type=float,
            metavar=("START", "STEP"),
            nargs=2,
            default=None,
            help=(
                "Start and stepsize for latitude vector, in case it can"
                " not be inferred from the netCDF. Requires that --latdim"
                " is also given."
            ),
        )
        self.add_argument(
            "--lon",
            type=float,
            metavar=("START", "STEP"),
            nargs=2,
            default=None,
            help=(
                "Start and stepsize for longitude vector, in case it can"
                " not be inferred from the netCDF. Requires that --londim"
                " is also given."
            ),
        )
        self.add_argument(
            "--bbox",
            type=float,
            default=None,
            metavar=("MIN_LON", "MIN_LAT", "MAX_LON", "MAX_LAT"),
            nargs=4,
            help=(
                "Bounding Box (lower left and upper right corner) "
                "of area to reshuffle"
            ),
        )
        self.add_argument(
            "--landmask",
            type=str,
            metavar="[FILENAME:]VARNAME",
            help=(
                "Either only the variable name of a variable in the dataset"
                " that is False over non-land points, or"
                "'<filename>:<varname>' if the landmask is in a different"
                " file. The landmask must have the same coordinates as the"
                " dataset."
            ),
        )
        self.add_argument(
            "--level",
            type=str,
            metavar="DIMNAME:IDX",
            nargs="+",
            help=(
                "Dimension names and indices for additional dimensions, e.g."
                " levels, dimension names and indices are given separated by"
                " colons."
            ),
        )
        self.add_argument(
            "--zlib",
            type=str2bool,
            default=True,
            help="Whether to use compression or not. Default is true",
        )


class RepurposeArgumentParser(ReaderArgumentParser):
    def __init__(self):
        super().__init__("Converts data to time series format.")
        self.prog = "repurpose_images"
        self.add_argument(
            "--imgbuffer",
            type=int,
            default=365,
            help=(
                "How many images to read at once. Bigger "
                "numbers make the conversion faster but "
                "consume more memory. Default is 365."
            ),
        )
        self.add_argument(
            "--cellsize",
            type=float,
            default=5.0,
            help=("Size of single file cells. Default is 5.0."),
        )


class TransposeArgumentParser(ReaderArgumentParser):
    def __init__(self):
        super().__init__("Converts data to transposed netCDF.")
        self.prog = "transpose_images"
        self.add_argument(
            "--memory",
            type=float,
            default=2,
            help="The amount of memory to use as buffer in GB",
        )
        self.add_argument(
            "--n_threads",
            type=int,
            default=4,
            help="Number of threads to use.",
        )
        self.add_argument(
            "--complevel",
            type=int,
            default=4,
            help="Compression level. 1 means low, 9 means high, default is 4.",
        )


def parse_args(parser, args):
    """
    Parse command line parameters for recursive download.

    Parameters
    ----------
    args : list of str
        Command line parameters as list of strings.

    Returns
    -------
    reader, args
    """
    args = parser.parse_args(args)
    print(
        f"Converting data from {args.start} to {args.end}"
        f" into directory {args.output_root}."
    )

    level = args.level
    if level is not None:
        level = {}
        for argument in args.level:
            dimname, val = argument.split(":")
            level[dimname] = int(val)

    common_reader_kwargs = dict(
        latname=args.latname,
        lonname=args.lonname,
        latdim=args.latdim,
        londim=args.londim,
        locdim=args.locdim,
        lat=args.lat,
        lon=args.lon,
        bbox=args.bbox,
        landmask=args.landmask,
        level=level,
    )

    input_path = Path(args.dataset_root)
    if input_path.is_file():
        reader = XarrayImageReader(
            input_path, args.parameter, **common_reader_kwargs,
        )
    else:
        reader = DirectoryImageReader(
            input_path,
            args.parameter,
            fmt=args.time_fmt,
            pattern=args.pattern,
            time_regex_pattern=args.time_regex_pattern,
            **common_reader_kwargs,
        )

    return reader, args


def repurpose(args):
    parser = RepurposeArgumentParser()
    reader, args = parse_args(parser, args)

    outpath = Path(args.output_root)
    outpath.mkdir(exist_ok=True, parents=True)

    reshuffler = Img2Ts(
        input_dataset=reader,
        outputpath=args.output_root,
        startdate=args.start,
        enddate=args.end,
        ts_attributes=reader.dataset_metadata,
        zlib=args.zlib,
        imgbuffer=args.imgbuffer,
        # this is necessary currently due to bug in repurpose
        cellsize_lat=args.cellsize,
        cellsize_lon=args.cellsize,
    )
    reshuffler.calc()


def transpose(args):
    parser = TransposeArgumentParser()
    reader, args = parse_args(parser, args)
    create_transposed_netcdf(
        reader,
        args.output_root,
        start=args.start,
        end=args.end,
        memory=args.memory,
        n_threads=args.n_threads,
        zlib=args.zlib,
        complevel=args.complevel,
    )


def run_repurpose():  # pragma: no cover
    repurpose(sys.argv[1:])


def run_transpose():  # pragma: no cover
    transpose(sys.argv[1:])
