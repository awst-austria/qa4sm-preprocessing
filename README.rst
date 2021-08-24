===================
qa4sm-preprocessing
===================

.. image:: https://github.com/awst-austria/qa4sm-preprocessing/workflows/Automated%20Tests/badge.svg?branch=master&event=push
   :target: https://github.com/awst-austria/qa4sm-preprocessing/actions

.. image:: https://coveralls.io/repos/awst-austria/qa4sm-preprocessing/badge.png?branch=master
  :target: https://coveralls.io/r/awst-austria/qa4sm-preprocessing?branch=master

.. image:: https://badge.fury.io/py/qa4sm-preprocessing.svg
    :target: https://badge.fury.io/py/qa4sm-preprocessing


This package contains functions to preprocess certain data before using them
in the QA4SM online validation framework.


CGLS HR SSM SWI
===============

Read CGLS HR SSM and SWI images (1km sampling) in netcdf format and convert
them to time series.
The image reader allows reading/converting data for a spatial subset (bbox) only.
Time series are stored in 5*5 DEG cell files, i.e. there are `~250 000 time series`
stored in one single cell file.

Time series reading is done based on cell level. Up to 6 cells are loaded into
memory at a time. The ``read_area`` function allows reading multiple GPI time series
around one location at once (and optionally converting them into a single, averaged
series, to represent the mean SM for an area).

Necessary updates
-----------------
At the moment it is only possible to read a single variable. However, in order
to mask SM time series based in location quality flags, it is necessary to
read multiple parameters. When passing the averaged time series for an area
to `pytesmo` for validation, masking can not be done in `pytesmo`, but must be done
beforehand.


.. _pyscaffold-notes:

Note
====

This project has been set up using PyScaffold 4.0.2. For details and usage
information on PyScaffold see https://pyscaffold.org/.
