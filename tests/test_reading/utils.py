import numpy as np
from pygeogrids.grids import BasicGrid


def validate_reader(reader, dataset, grid=True):
    expected_timestamps = dataset.indexes[reader.timename]
    assert len(reader.timestamps) == len(expected_timestamps)
    assert np.all(list(reader.timestamps) == expected_timestamps)

    if grid:
        assert isinstance(reader.grid, BasicGrid)
    else:
        assert reader.grid is None

    # check if read_block gives the correct result for the first image
    expected_img = dataset.sel(**{reader.timename: reader.timestamps[0]})
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
