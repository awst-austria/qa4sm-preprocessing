import netCDF4
import numpy as np
import io
import re
from typing import Dict, List, Tuple, Any
import zipfile
import io
import re
import pandas as pd


class NetCDFValidator:
    """NetCDF format validator for QA4SM compliance"""

    def __init__(self, nc_dataset: netCDF4.Dataset):
        self.nc = nc_dataset
        self.is_netcdf4 = self._detect_netcdf4_format()
        self.file_info = self._get_file_format_info()
        self.validation_results = {}
        self.errors = []

        # Cache coordinate variables for performance
        self._lat_vars = None
        self._lon_vars = None
        self._time_vars = None
        self._vars = None

    def validate(self) -> Tuple[bool, Dict[str, Any], int]:
        """Main validation method that runs all checks"""
        self.errors = []

        # Run all validation checks
        validation_methods = [
            self._validate_basic_structure,
            self._validate_naming_conventions,
            self._validate_coordinates,
            self._validate_variables,
            self._validate_coordinate_ranges,
            self._validate_netcdf4_specific,
        ]

        for method in validation_methods:
            try:
                method()
            except Exception as e:
                self.errors.append(
                    f"Validation error in {method.__name__}: {str(e)}")

        # Compile results
        if self.errors:
            return False, {
                "errors": self.errors,
                "file_info": self.file_info
            }, 400

        return True, {
            "message": "NetCDF file is valid for QA4SM",
            "file_info": self.file_info,
            "validation_results": self.validation_results
        }, 200

    def _detect_netcdf4_format(self) -> bool:
        """Detect if this is a NetCDF4 file"""
        try:
            # Check if file has groups (NetCDF4 feature)
            if hasattr(self.nc, 'groups') and len(self.nc.groups) > 0:
                return True

            # Check file format attribute
            if hasattr(self.nc, 'file_format'):
                return 'NETCDF4' in self.nc.file_format.upper()

            # Check for NetCDF4-specific data types
            for var in self.nc.variables.values():
                if str(var.dtype) in ['object', 'string', 'unicode']:
                    return True

            return False
        except:
            return False

    def _get_file_format_info(self) -> Dict[str, Any]:
        """Get detailed file format information"""
        info = {
            'format': 'Unknown',
            'version': 'Unknown',
            'features': [],
            'dimensions': len(self.nc.dimensions),
            'variables': len(self.nc.variables)
        }

        if hasattr(self.nc, 'file_format'):
            info['format'] = self.nc.file_format

            if 'NETCDF4' in self.nc.file_format.upper():
                info['features'].append('NetCDF4/HDF5')
                if hasattr(self.nc, 'groups') and len(self.nc.groups) > 0:
                    info['features'].append('Groups')
            elif 'NETCDF3' in self.nc.file_format.upper():
                info['features'].append('NetCDF Classic')

        return info

    def _validate_basic_structure(self):
        """Check basic structure (dimensions and variables exist)"""
        if len(self.nc.dimensions) == 0:
            self.errors.append("No dimensions found")

        if len(self.nc.dimensions) < 3:
            self.errors.append(
                "At least 3 dimensions needed in netcdf (lat, lon and time)")

        if len(self.nc.variables) == 0:
            self.errors.append("No variables found")

        self.validation_results['basic_structure'] = {
            'dimensions': len(self.nc.dimensions),
            'variables': len(self.nc.variables)
        }

    def _validate_naming_conventions(self):
        """Check naming conventions for variables, dimensions, and attributes"""
        name_pattern = re.compile(r'^[a-zA-Z0-9_]+$')
        naming_errors = []

        # Check dimensions
        for dim_name in self.nc.dimensions:
            if not name_pattern.match(dim_name):
                naming_errors.append(f"Invalid dimension name: {dim_name}")

        # Check variables
        for var_name, var in self.nc.variables.items():
            if not name_pattern.match(var_name):
                naming_errors.append(f"Invalid variable name: {var_name}")

            # Check variable attributes
            for attr_name in var.ncattrs():
                if not name_pattern.match(attr_name):
                    naming_errors.append(
                        f"Invalid attribute name: {attr_name} in variable {var_name}"
                    )

        # get only valid vars (with 3 dimensions and not lat, lon)
        vars = self.get_variables()

        # Check variable name length
        for var_name in vars:
            if len(var_name) > 30:
                naming_errors.append(
                    f"Please note that variable names cannot be longer than 30 characters: {var_name}"
                )

        self.errors.extend(naming_errors)
        self.validation_results['naming_conventions'] = {
            'errors': len(naming_errors)
        }

    def _validate_coordinates(self):
        """Validate coordinate variables and their structure"""
        coord_errors = []

        # Find coordinate variables (cached)
        lat_vars = self.get_coordinate_variables('latitude')
        lon_vars = self.get_coordinate_variables('longitude')
        time_vars = self.get_coordinate_variables('time')

        if not lat_vars:
            coord_errors.append("No latitude coordinate variable found")
        if not lon_vars:
            coord_errors.append("No longitude coordinate variable found")
        if not time_vars:
            coord_errors.append("No time coordinate variable found")

        if coord_errors:
            self.errors.extend(coord_errors)
            return

        # Check coordinate dimensions
        lat_var = self.nc.variables[lat_vars[0]]
        lon_var = self.nc.variables[lon_vars[0]]
        time_var = self.nc.variables[time_vars[0]]

        # Check that lat/lon don't depend on time
        if 'time' in lat_var.dimensions:
            coord_errors.append("Latitude coordinate depends on time dimension")
        if 'time' in lon_var.dimensions:
            coord_errors.append(
                "Longitude coordinate depends on time dimension")

        # Check coordinate variable dimensions (1D or 2D)
        if len(lat_var.dimensions) not in [1, 2]:
            coord_errors.append(
                f"Latitude coordinate has {len(lat_var.dimensions)} dimensions (expected 1 or 2)"
            )
        if len(lon_var.dimensions) not in [1, 2]:
            coord_errors.append(
                f"Longitude coordinate has {len(lon_var.dimensions)} dimensions (expected 1 or 2)"
            )

        # Time should be 1D
        if len(time_var.dimensions) != 1:
            coord_errors.append(
                f"Time coordinate has {len(time_var.dimensions)} dimensions (expected 1)"
            )

        self.errors.extend(coord_errors)
        self.validation_results['coordinates'] = {
            'lat_vars': lat_vars,
            'lon_vars': lon_vars,
            'time_vars': time_vars,
            'errors': len(coord_errors)
        }

    def _validate_variables(self):
        """Check for variables"""
        vars = self.get_variables()

        if not vars:
            self.errors.append(
                "Variables need 3 dimensions, a latitude or y, a longitude or x, and a time dimension."
            )

        self.validation_results['variables'] = {
            'variables': vars,
            'count': len(vars)
        }

    def _validate_coordinate_ranges(self):
        """Validate coordinate ranges and check for duplicate timestamps"""
        range_errors = []

        lat_vars = self.get_coordinate_variables('latitude')
        lon_vars = self.get_coordinate_variables('longitude')
        time_vars = self.get_coordinate_variables('time')

        if not (lat_vars and lon_vars and time_vars):
            return  # Already handled in _validate_coordinates

        # Check latitude range
        try:
            lat_var = self.nc.variables[lat_vars[0]]
            lat_data = lat_var[:]
            if np.any(lat_data < -90) or np.any(lat_data > 90):
                range_errors.append(
                    "Latitude values outside valid range (-90 to 90)")
        except Exception as e:
            range_errors.append(f"Could not read latitude data: {str(e)}")

        # Check longitude range
        try:
            lon_var = self.nc.variables[lon_vars[0]]
            lon_data = lon_var[:]
            if np.any(lon_data < -180) or np.any(lon_data > 180):
                range_errors.append(
                    "Longitude values outside valid range (-180 to 180)")
        except Exception as e:
            range_errors.append(f"Could not read longitude data: {str(e)}")

        # Check for duplicate timestamps
        try:
            time_var = self.nc.variables[time_vars[0]]
            time_data = time_var[:]
            unique_times = np.unique(time_data)
            if len(time_data) != len(unique_times):
                range_errors.append("Duplicate timestamps found")

            self.validation_results['time_analysis'] = {
                'total_timestamps': len(time_data),
                'unique_timestamps': len(unique_times),
                'has_duplicates': len(time_data) != len(unique_times)
            }
        except Exception as e:
            range_errors.append(f"Could not read time data: {str(e)}")

        self.errors.extend(range_errors)
        self.validation_results['coordinate_ranges'] = {
            'errors': len(range_errors)
        }

    def _validate_netcdf4_specific(self):
        """NetCDF4-specific validation checks"""
        if not self.is_netcdf4:
            return

        netcdf4_errors = []

        # Check for groups (QA4SM expects flat structure)
        if hasattr(self.nc, 'groups') and len(self.nc.groups) > 0:
            netcdf4_errors.append(
                "NetCDF4 groups found - QA4SM requires flat structure")

        # Check for unsupported NetCDF4 features
        for var_name, var in self.nc.variables.items():
            # Check for compound data types
            if hasattr(var, 'dtype') and hasattr(var.dtype,
                                                 'names') and var.dtype.names:
                netcdf4_errors.append(
                    f"Compound data type not supported for variable {var_name}")

            # Check for variable-length arrays
            if hasattr(var, 'dtype') and 'object' in str(var.dtype):
                netcdf4_errors.append(
                    f"Variable-length arrays not supported for variable {var_name}"
                )

        # Check for user-defined types
        if hasattr(self.nc, 'cmptypes') and len(self.nc.cmptypes) > 0:
            netcdf4_errors.append("User-defined compound types not supported")

        self.errors.extend(netcdf4_errors)
        self.validation_results['netcdf4_specific'] = {
            'errors': len(netcdf4_errors)
        }

    def get_coordinate_variables(self, coord_type: str) -> List[str]:
        """Find coordinate variables by type with caching"""
        if coord_type == 'latitude' and self._lat_vars is not None:
            return self._lat_vars
        elif coord_type == 'longitude' and self._lon_vars is not None:
            return self._lon_vars
        elif coord_type == 'time' and self._time_vars is not None:
            return self._time_vars

        candidates = []

        if coord_type == 'latitude':
            valid_units = [
                'degrees_north', 'degree_north', 'degree_N', 'degrees_N',
                'degreeN', 'degreesN'
            ]
            standard_name = 'latitude'
            axis_value = 'Y'
            var_names = ['latitude', 'lat']
        elif coord_type == 'longitude':
            valid_units = [
                'degrees_east', 'degree_east', 'degree_E', 'degrees_E',
                'degreeE', 'degreesE'
            ]
            standard_name = 'longitude'
            axis_value = 'X'
            var_names = ['longitude', 'lon']
        elif coord_type == 'time':
            valid_units = None
            standard_name = 'time'
            axis_value = 'T'
            var_names = ['time']

        for var_name, var in self.nc.variables.items():
            # Skip scalar variables (0-dimensional)
            if var.ndim == 0:
                continue
            # Check by units
            if coord_type != 'time' and hasattr(
                    var, 'units') and var.units in valid_units:
                candidates.append(var_name)
            # Check by standard_name
            elif hasattr(
                    var,
                    'standard_name') and var.standard_name == standard_name:
                candidates.append(var_name)
            # Check by axis
            elif hasattr(var, 'axis') and var.axis == axis_value:
                candidates.append(var_name)
            # Check by variable name
            elif var_name.lower() in var_names:
                candidates.append(var_name)
            # Special case for time units (contains "since")
            elif coord_type == 'time' and hasattr(
                    var, 'units') and 'since' in var.units.lower():
                candidates.append(var_name)

        # Cache results
        if coord_type == 'latitude':
            self._lat_vars = candidates
        elif coord_type == 'longitude':
            self._lon_vars = candidates
        elif coord_type == 'time':
            self._time_vars = candidates

        return candidates

    def get_variables(self) -> List[str]:
        """Find potential  variables with caching"""
        if self._vars is not None:
            return self._vars

        vars = []

        for var_name, var in self.nc.variables.items():
            # Skip coordinate variables
            if var_name.lower() in [
                    'lat', 'latitude', 'lon', 'longitude', 'time', 'x', 'y'
            ]:
                continue

            # Check if it's a 3D variable (likely data variable)
            if len(var.dimensions) == 3:
                vars.append(var_name)

        self._vars = vars
        return vars


def run_upload_format_check(file, filename):
    """Main function using the NetCDFValidator class"""
    try:
        file.seek(0)

        if filename.endswith(('.nc', '.nc4')):
            # Read header chunk first for format detection
            header_chunk = file.read(8192)
            file.seek(0)

            # Check for both NetCDF classic and NetCDF4/HDF5 magic numbers
            is_valid_format = (
                header_chunk.startswith(b'CDF\x01') or  # NetCDF classic
                header_chunk.startswith(b'CDF\x02') or  # NetCDF 64-bit offset
                header_chunk.startswith(b'\x89HDF\r\n\x1a\n')  # NetCDF4/HDF5
            )

            if not is_valid_format:
                return False, {
                    "error": "Not a valid NetCDF or NetCDF4 file"
                }, 400

            # Read file content into memory
            file_content = file.read()
            file.seek(0)

            # Create in-memory dataset
            memory_file = io.BytesIO(file_content)

            try:
                with netCDF4.Dataset(
                        '', mode='r', memory=memory_file.read()) as nc:
                    # Create validator and run checks
                    validator = NetCDFValidator(nc)
                    return validator.validate()

            except Exception as e:
                return False, {
                    "error": f"Failed to open NetCDF file: {str(e)}"
                }, 400

        return True, "Valid file", 200

    except Exception as e:
        return False, {"error": f"File processing error: {str(e)}"}, 500


class ZipValidator:
    """Validator for ZIP files containing CSV time series data"""

    def __init__(self, zip_file):
        """
        Initialize validator with zip file

        Args:
            zip_file: File-like object or path to zip file
        """
        self.zip_file = zip_file
        self.errors = []
        self.warnings = []  # Added warnings list
        self.file_info = {}
        self.validation_results = {}
        self.csv_files = []
        self.yml_files = []
        self.nc_files = []  # Added nc_files list
        self.dataset_name = None

    def validate(self) -> Tuple[bool, Dict[str, Any], int]:
        """Main validation method that runs all checks"""
        self.errors = []
        self.warnings = []

        # First check ZIP structure to get file lists
        try:
            self._validate_zip_structure()
        except Exception as e:
            self.errors.append(
                f"Validation error in _validate_zip_structure: {str(e)}")
            return False, {
                "errors": self.errors,
                "file_info": self.file_info
            }, 400

        # Special case: If all files are .nc files, skip all other validations
        if self._is_netcdf_only():
            return True, {
                "message": "ZIP file contains NetCDF time series data",
                "file_info": self.file_info,
                "validation_results": {
                    "type": "netcdf_timeseries"
                }
            }, 200

        # Run remaining validation checks for CSV files
        validation_methods = [
            self._validate_file_types, self._validate_naming_conventions,
            self._validate_csv_content
        ]

        for method in validation_methods:
            try:
                method()
            except Exception as e:
                self.errors.append(
                    f"Validation error in {method.__name__}: {str(e)}")

        # Compile results
        if self.errors:
            return False, {
                "errors": self.errors,
                "file_info": self.file_info
            }, 400

        return True, {
            "message": "ZIP file is valid for QA4SM",
            "file_info": self.file_info,
            "validation_results": self.validation_results
        }, 200

    def _is_netcdf_only(self) -> bool:
        """Check if ZIP contains only NetCDF files"""
        if not self.nc_files:
            return False

        # All files must be .nc files (no CSV, YML, or other files)
        total_expected = len(self.nc_files)
        return self.file_info[
            'total_files'] == total_expected and total_expected > 0

    def _validate_zip_structure(self):
        """Validate that the file is a valid ZIP archive"""
        try:
            with zipfile.ZipFile(self.zip_file, 'r') as zip_ref:
                # Test ZIP integrity
                zip_ref.testzip()

                # Get file list
                file_list = zip_ref.namelist()

                if not file_list:
                    self.errors.append("ZIP file is empty")
                    return

                # Separate CSV, YML, and NC files
                self.csv_files = [
                    f for f in file_list if f.lower().endswith('.csv')
                ]
                self.yml_files = [
                    f for f in file_list if f.lower().endswith('.yml')
                ]
                self.nc_files = [
                    f for f in file_list if f.lower().endswith('.nc')
                ]

                # Store file info
                self.file_info = {
                    'total_files':
                        len(file_list),
                    'csv_files':
                        len(self.csv_files),
                    'yml_files':
                        len(self.yml_files),
                    'nc_files':
                        len(self.nc_files),
                    'other_files':
                        len(file_list) - len(self.csv_files) -
                        len(self.yml_files) - len(self.nc_files)
                }

        except zipfile.BadZipFile:
            self.errors.append("File is not a valid ZIP archive")
        except Exception as e:
            self.errors.append(f"Error reading ZIP file: {str(e)}")

    def _validate_file_types(self):
        """Validate that ZIP contains only CSV files and optionally one YML file"""
        if not self.csv_files:
            self.errors.append("ZIP file must contain at least one CSV file")
            return

        # Check for non-CSV, non-YML files
        total_expected = len(self.csv_files) + len(self.yml_files)
        if self.file_info['total_files'] != total_expected:
            self.errors.append(
                "ZIP file must contain only CSV files and optionally one metadata.yml file"
            )

        # Check YML file count and name
        if len(self.yml_files) > 1:
            self.errors.append(
                "ZIP file can contain at most one YML metadata file")
        elif len(self.yml_files) == 1:
            yml_file = self.yml_files[0]
            if yml_file.lower() not in ['metadata.yml']:
                self.warnings.append(
                    f"YML file should be named 'metadata.yml', found: {yml_file}"
                )

    def _validate_naming_conventions(self):
        """Validate CSV file naming conventions"""
        if not self.csv_files:
            return

        # Pattern: <DATASET_NAME>_gpi=<GPI>_lat=<LAT>_lon=<LON>.csv
        pattern = r'^(.+?)_gpi=(\d+)_lat=(-?\d+(?:\.\d+)?)_lon=(-?\d+(?:\.\d+)?)\.csv$'

        dataset_names = set()
        gpi_values = set()
        coordinates = []

        for csv_file in self.csv_files:
            match = re.match(pattern, csv_file)
            if not match:
                self.errors.append(
                    f"Invalid filename format: {csv_file}. Expected format: <DATASET_NAME>_gpi=<GPI>_lat=<LAT>_lon=<LON>.csv"
                )
                continue

            dataset_name, gpi, lat, lon = match.groups()
            dataset_names.add(dataset_name)

            # Validate GPI uniqueness
            if gpi in gpi_values:
                self.errors.append(f"Duplicate GPI value found: {gpi}")
            gpi_values.add(gpi)

            # Validate coordinate ranges
            try:
                lat_float = float(lat)
                lon_float = float(lon)

                if not (-90 <= lat_float <= 90):
                    self.errors.append(
                        f"Invalid latitude {lat_float} in {csv_file}. Must be between -90 and 90"
                    )

                if not (-180 <= lon_float <= 180):
                    self.errors.append(
                        f"Invalid longitude {lon_float} in {csv_file}. Must be between -180 and 180"
                    )

                coordinates.append((lat_float, lon_float))

            except ValueError:
                self.errors.append(f"Invalid coordinate format in {csv_file}")

        # Check dataset name consistency
        if len(dataset_names) > 1:
            self.errors.append(
                f"All CSV files must have the same dataset name. Found: {list(dataset_names)}"
            )
        elif len(dataset_names) == 1:
            self.dataset_name = list(dataset_names)[0]

        # Store validation results
        self.validation_results.update({
            'dataset_name': self.dataset_name,
            'unique_gpi_count': len(gpi_values),
            'coordinate_range': {
                'lat_min':
                    min(coord[0] for coord in coordinates)
                    if coordinates else None,
                'lat_max':
                    max(coord[0] for coord in coordinates)
                    if coordinates else None,
                'lon_min':
                    min(coord[1] for coord in coordinates)
                    if coordinates else None,
                'lon_max':
                    max(coord[1] for coord in coordinates)
                    if coordinates else None,
            }
        })

    def _validate_csv_content(self):
        """Validate CSV file content by checking a sample file"""
        if not self.csv_files:
            return

        # Pick the first CSV file for content validation
        sample_file = self.csv_files[0]

        try:
            with zipfile.ZipFile(self.zip_file, 'r') as zip_ref:
                with zip_ref.open(sample_file) as csv_file:
                    # Try to read with pandas
                    try:
                        df = pd.read_csv(
                            csv_file, index_col=0, parse_dates=True)

                        # Validate structure
                        if df.empty:
                            self.errors.append(
                                f"CSV file {sample_file} is empty")
                            return

                        # Check if index is datetime
                        if not pd.api.types.is_datetime64_any_dtype(df.index):
                            self.errors.append(
                                f"First column in {sample_file} must be datetime parseable"
                            )

                        # Store info about the CSV structure
                        self.validation_results.update({
                            'sample_file': sample_file,
                            'columns': list(df.columns),
                            'column_count': len(df.columns),
                            'row_count': len(df),
                            'date_range': {
                                'start':
                                    str(df.index.min())
                                    if not df.empty else None,
                                'end':
                                    str(df.index.max())
                                    if not df.empty else None
                            }
                        })

                    except pd.errors.ParserError as e:
                        self.errors.append(
                            f"Cannot parse CSV file {sample_file}: {str(e)}")
                    except Exception as e:
                        self.errors.append(
                            f"Error reading CSV file {sample_file}: {str(e)}")

        except Exception as e:
            self.errors.append(
                f"Error accessing CSV file {sample_file}: {str(e)}")
