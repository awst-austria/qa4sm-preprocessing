=========
Changelog
=========

Unreleased
==========

v0.6.1
======
- Remove request parameter from validate_file_upload call
- Skip user space validation as it's handled in UI layer

v0.6
====
- Add format validation for files uploaded to qa4sm (ZIP, netCDF)

v0.5
====
- Add SMOS L2 land gridfile for land-only processing
- Implement only-land reading option in SMOSL2Reader

v0.4
====
- Updated environment dependencies
- Code cleanup and maintenance
- Test data, config updated

v0.3
====
* Smos package fixes `#29 <https://github.com/awst-austria/qa4sm-preprocessing/pull/29>`_ `#30 <https://github.com/awst-austria/qa4sm-preprocessing/pull/30>`_ `#31 <https://github.com/awst-austria/qa4sm-preprocessing/pull/31>`_

v0.2
====
- FRM4SM Release 2 version

v0.1.4
======
- new subpackage ismn_frm
- level 2 and timeseries data readers
- bug fixes

v0.1.3
==========
- renaming of keyword ``latdim`` to ``ydim`` and ``londim`` to ``xdim`` in the
  reading package
- detection of coordinates based on CF conventions
- better handling of time offset variables

v0.1.2
======

- more features for ``DirectoryImageReader``
  - handling of "image" files with multiple time steps
  - better documentation for subclassing
- renaming of other readers
  - ``XarrayImageStackReader`` to ``StackImageReader``
  - ``XarrayTSReader`` to ``StackTs``
- ``repurpose`` function for image readers
- improved test coverage

v0.1.0
======

- handling empty data frame added (PR `#7 <https://github.com/awst-austria/qa4sm-preprocessing/pull/7>`_)
- Discard attrs (PR `#6 <https://github.com/awst-austria/qa4sm-preprocessing/pull/6>`_)
- Sub-daily to daily averaging methods (PR `#5 <https://github.com/awst-austria/qa4sm-preprocessing/pull/5>`_)
- unified the reader backend; everything is using read_block now (PR `#4 <https://github.com/awst-austria/qa4sm-preprocessing/pull/4>`_)
- a few fixes for unstructured and curvilinear grids (PR `#3 <https://github.com/awst-austria/qa4sm-preprocessing/pull/3>`_)



v0.0.1
======

- Added readers and converters for HR CGLS SSM and SWI image data
- Added time series readers for converted data, averaging all time series in an area
