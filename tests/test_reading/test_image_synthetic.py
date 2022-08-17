# This module contains tests for the DirectoryImageReader based on synthetic datasets
# The tests typically follow this pattern:
# - take fixture as input dataset
# - modify a copy of the input dataset
# - write to image files
# - read image files so that the original input dataset is returned

import numpy as np
from pathlib import Path
import pandas as pd
import pytest
import shutil
import time
import xarray as xr

from repurpose.img2ts import Img2Ts

from qa4sm_preprocessing.reading import (
    DirectoryImageReader,
)
from qa4sm_preprocessing.reading.utils import mkdate
from qa4sm_preprocessing.reading.write import write_images

# this is defined in conftest.py
from pytest import test_data_path


def validate_reader(reader, dataset):
    expected_timestamps = dataset.indexes["time"]
    assert len(reader.timestamps) == len(expected_timestamps)
    assert np.all(list(reader.timestamps) == expected_timestamps)

    # check if read_block gives the correct result for the first image
    expected_img = dataset.sel(time=reader.timestamps[0])
    img = reader.read_block(reader.timestamps[0], reader.timestamps[0])
    assert expected_img.attrs == reader.global_attrs
    assert expected_img.attrs == img.attrs
    assert sorted(list(expected_img.data_vars)) == sorted(list(img.data_vars))
    for var in img.data_vars:
        np.testing.assert_equal(
            img[var].values.squeeze(), expected_img[var].values.squeeze()
        )

    # check if read_block gives the correct result for the full block
    block = reader.read_block()
    assert dataset.attrs == reader.global_attrs
    assert dataset.attrs == block.attrs
    assert sorted(list(dataset.data_vars)) == sorted(list(block.data_vars))
    for var in block.data_vars:
        np.testing.assert_equal(
            block[var].values.squeeze(), dataset[var].values.squeeze()
        )


def test_directory_image_reader_basic(synthetic_test_args):
    # basic functionality with a regular grid
    ds, kwargs = synthetic_test_args

    write_images(ds, test_data_path / "basic_test", "basic_test")
    reader = DirectoryImageReader(
        test_data_path / "basic_test",
        ["X", "Y"],
        fmt="basic_test_%Y%m%dT%H%M.nc",
        **kwargs
    )

    validate_reader(reader, ds)
    shutil.rmtree(test_data_path / "basic_test")
