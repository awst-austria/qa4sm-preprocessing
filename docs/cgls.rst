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
