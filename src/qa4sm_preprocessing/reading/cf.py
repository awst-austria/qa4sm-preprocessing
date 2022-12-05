"""
This module provides coordinate detection based on CF conventions.
It is based on ``cf_xarray``, but adds detection based on dtype for the time
axis and detection of specified alternative names, as well as
case-insensitivity for the latitude and longitude axes.
"""
import cf_xarray as cfxr
import numpy as np
import re

from .exceptions import ReaderError


custom_criteria = {
    "latitude": {
        "standard_name": re.compile("lat|latitude", re.IGNORECASE),
        "long_name": re.compile("lat|latitude", re.IGNORECASE),
        "units": (
            "degree_north|degree_N|degreeN|degrees_north|degrees_N|degreesN"
        ),
        "_CoordinateAxisType": "Lat",
    },
    "longitude": {
        "standard_name": re.compile("lon|longitude", re.IGNORECASE),
        "long_name": re.compile("lon|longitude", re.IGNORECASE),
        "units": (
            "degree_east|degree_E|degreeE|degrees_east|degrees_E|degreesE"
        ),
        "_CoordinateAxisType": "Lon",
    },
    "time": {
        "standard_name": "time",
        "long_name": "time",
        "axis": "T",
        "units": "[A-z]+ since .+",
        "_CoordinateAxisType": "Time",
    },
}


def get_coord(ds, standardname, alternatives=[]):
    try:
        with cfxr.set_options(custom_criteria=custom_criteria):
            coord = ds.cf[standardname]
    except KeyError:
        candidates = list(filter(lambda alt: alt in ds, alternatives))
        if not candidates:
            raise ReaderError(f"{standardname} can not be inferred!")
        coord = ds[candidates[0]]
    return coord


def get_time(ds):
    try:
        time = get_coord(ds, "time")
        assert time.ndim == 1
    except (ReaderError, AssertionError):
        candidates = []
        candidates = list(
            filter(
                lambda v: ds[v].ndim == 1
                and np.issubdtype(ds[v].dtype, np.datetime64),
                ds.variables,
            )
        )
        if len(candidates) > 1:  # pragma: no cover
            raise ReaderError(
                f"More than one time coordinate found: {candidates}"
            )
        if len(candidates) == 0:  # pragma: no cover
            raise ReaderError("No time coordinate found!")
        time = ds[candidates[0]]
    return time
