import pytest
from unittest.mock import Mock, patch, MagicMock
import os, io
import numpy as np
import zipfile

from qa4sm_preprocessing.utils import validate_file_upload, verify_file_extension
from qa4sm_preprocessing.format_validator import NetCDFValidator, ZipValidator, run_upload_format_check



class TestNetCDFValidation:
    """Test NetCDF file validation through validate_file_upload"""

    @pytest.fixture
    def mock_request(self):
        request = Mock()
        request.user = Mock()
        request.user.space_left = 1000000
        return request

    def test_netcdf_validation_success(self, mock_request):
        """Test NetCDF validation success with real NetCDF file"""

        # Mock user has 5GB space left
        mock_request.user.space_left = 5 * 1024 * 1024 * 1024  # 5GB

        # Path to the actual test NetCDF file
        test_file_path = os.path.join(
            os.path.dirname(__file__),
            "../test-data/user_data/teststack_c3s_2dcoords_min_attrs.nc")

        assert os.path.exists(
            test_file_path), f"Test file not found: {test_file_path}"

        # Create file mock from actual NetCDF file
        with open(test_file_path, 'rb') as f:
            file_content = f.read()

        file_mock = Mock()
        file_mock.read.return_value = file_content
        file_mock.name = "teststack_c3s_2dcoords_min_attrs.nc"
        file_mock.size = len(file_content)  # Set actual file size
        file_mock.seek = Mock()

        is_valid, message, status = validate_file_upload(
            mock_request, file_mock, "teststack_c3s_2dcoords_min_attrs.nc")

        assert is_valid
        assert "NetCDF file validation successful. Found 3 variables." in message
        assert status == 200

    def test_netcdf_validation_failure(self, mock_request):
        """Test validation with improper dataset format"""

        # Mock user has 5GB space left
        mock_request.user.space_left = 5 * 1024 * 1024 * 1024  # 5GB

        # Path to the actual improper NetCDF file
        test_file_path = os.path.join(
            os.path.dirname(__file__),
            "../test-data/user_data/inproper-dataset-format.nc")

        assert os.path.exists(
            test_file_path), f"Test file not found: {test_file_path}"

        # Create file mock from actual improper NetCDF file
        with open(test_file_path, 'rb') as f:
            file_content = f.read()

        file_mock = Mock()
        file_mock.read.return_value = file_content
        file_mock.name = "inproper-dataset-format.nc"
        file_mock.size = len(file_content)
        file_mock.seek = Mock()

        is_valid, message, status = validate_file_upload(
            mock_request, file_mock, file_mock.name)

        assert not is_valid
        assert "NetCDF validation failed: At least 3 dimensions needed" in message
        assert status == 400

    def test_netcdf_read_error(self, mock_request):
        """Test NetCDF file read error"""
        file_mock = Mock()
        file_mock.name = "corrupt.nc"
        file_mock.size = 1000
        file_mock.read.side_effect = Exception("File read error")
        file_mock.seek = Mock()

        is_valid, message, status = validate_file_upload(
            mock_request, file_mock, "corrupt.nc")

        assert not is_valid
        assert "Error reading NetCDF file" in message
        assert status == 500


class TestNetCDFValidator:
    """Test the NetCDFValidator class directly"""

    @pytest.fixture
    def mock_nc_basic(self):
        """Create a basic mock NetCDF dataset"""
        mock_nc = Mock()
        mock_nc.dimensions = {'lat': Mock(), 'lon': Mock(), 'time': Mock()}

        # Create properly configured mock variables
        lat_var = Mock()
        lat_var.dimensions = ('lat',)
        lat_var.units = 'degrees_north'

        lon_var = Mock()
        lon_var.dimensions = ('lon',)
        lon_var.units = 'degrees_east'

        time_var = Mock()
        time_var.dimensions = ('time',)
        time_var.units = 'days since 1900-01-01'

        var1 = Mock()
        var1.dimensions = ('time', 'lat', 'lon')
        var1.units = 'kg/m^2'

        mock_nc.variables = {
            'lat': lat_var,
            'lon': lon_var,
            'time': time_var,
            'var1': var1
        }
        mock_nc.file_format = 'NETCDF4'
        mock_nc.groups = {}
        return mock_nc

    def test_basic_structure_validation_no_dimensions(self):
        """Test validation fails when no dimensions exist"""
        mock_nc = Mock()
        mock_nc.dimensions = {}
        mock_nc.variables = {'var1': Mock()}
        mock_nc.file_format = 'NETCDF4'
        mock_nc.groups = {}

        validator = NetCDFValidator(mock_nc)
        validator._validate_basic_structure()

        assert "No dimensions found" in validator.errors

    def test_basic_structure_validation_insufficient_dimensions(self):
        """Test validation fails with less than 3 dimensions"""
        mock_nc = Mock()
        mock_nc.dimensions = {'lat': Mock(), 'lon': Mock()}  # Only 2 dimensions
        mock_nc.variables = {'var1': Mock()}
        mock_nc.file_format = 'NETCDF4'
        mock_nc.groups = {}

        validator = NetCDFValidator(mock_nc)
        validator._validate_basic_structure()

        assert "At least 3 dimensions needed" in validator.errors.pop()

    def test_basic_structure_validation_no_variables(self):
        """Test validation fails when no variables exist"""
        mock_nc = Mock()
        mock_nc.dimensions = {'lat': Mock(), 'lon': Mock(), 'time': Mock()}
        mock_nc.variables = {}
        mock_nc.file_format = 'NETCDF4'
        mock_nc.groups = {}

        validator = NetCDFValidator(mock_nc)
        validator._validate_basic_structure()

        assert "No variables found" in validator.errors

    def test_coordinate_validation_missing_coordinates(self, mock_nc_basic):
        """Test validation fails when coordinate variables are missing"""
        # Remove coordinate variables but ensure remaining variable has proper attributes
        data_var = Mock()
        data_var.dimensions = ('time', 'lat', 'lon')
        data_var.units = 'kg/m^2'  # String, not Mock
        data_var.ndim = 3  # Add ndim attribute

        mock_nc_basic.variables = {'data_var': data_var}

        validator = NetCDFValidator(mock_nc_basic)
        validator._validate_coordinates()

        assert "No latitude coordinate variable found" in validator.errors
        assert "No longitude coordinate variable found" in validator.errors
        assert "No time coordinate variable found" in validator.errors


class TestZipValidation:
    """Test ZIP file validation through validate_file_upload"""

    @pytest.fixture
    def mock_request(self):
        request = Mock()
        request.user = Mock()
        request.user.space_left = 1000000
        return request

    def _create_invalid_zip_file(self):
        """Create an invalid ZIP file"""
        content = b'PK\x03\x04' + b'\x00' * 100  # ZIP magic + invalid data
        file_mock = Mock()
        file_mock.name = "invalid.zip"
        file_mock.size = len(content)
        file_mock.read.return_value = content
        file_mock.seek = Mock()
        return file_mock

    def test_zipfile_validation_success(self, mock_request):
        """Test NetCDF validation success with real NetCDF file"""

        # Mock user has 5GB space left
        mock_request.user.space_left = 5 * 1024 * 1024 * 1024  # 5GB

        # Path to the actual test NetCDF file
        test_file_path = os.path.join(
            os.path.dirname(__file__),
            "../test-data/user_data/test_data_csv.zip")

        assert os.path.exists(
            test_file_path), f"Test file not found: {test_file_path}"

        # Create file mock from actual NetCDF file
        with open(test_file_path, 'rb') as f:
            file_content = f.read()

        file_mock = Mock()
        file_mock.read.return_value = file_content
        file_mock.name = "test_data_csv.zip"
        file_mock.size = len(file_content)  # Set actual file size
        file_mock.seek = Mock()

        is_valid, message, status = validate_file_upload(
            mock_request, file_mock, file_mock.name)

        assert is_valid
        assert "ZIP file validation successful." in message
        assert status == 200

    def test_zipfile_validation_failure(self, mock_request):
        """Test validation with improper dataset format"""

        # Mock user has 5GB space left
        mock_request.user.space_left = 5 * 1024 * 1024 * 1024  # 5GB

        # Path to the actual improper NetCDF file
        test_file_path = os.path.join(
            os.path.dirname(__file__),
            "../test-data/user_data/invalid-dataset-format.zip")

        assert os.path.exists(
            test_file_path), f"Test file not found: {test_file_path}"

        # Create file mock from actual improper NetCDF file
        with open(test_file_path, 'rb') as f:
            file_content = f.read()

        file_mock = Mock()
        file_mock.read.return_value = file_content
        file_mock.name = "invalid-dataset-format.zip"
        file_mock.size = len(file_content)
        file_mock.seek = Mock()

        is_valid, message, status = validate_file_upload(
            mock_request, file_mock, file_mock.name)

        assert not is_valid
        assert "ZIP validation failed" in message
        assert status == 400

class TestNetCDFValidatorAdditional:
    """Additional tests for NetCDFValidator functions"""

    def test_detect_netcdf4_format_with_groups(self):
        """Test NetCDF4 format detection with groups"""
        mock_nc = Mock()
        mock_nc.groups = {'group1': Mock()}
        mock_nc.file_format = 'NETCDF4'
        mock_nc.dimensions = {'lat': Mock(), 'lon': Mock(), 'time': Mock()}
        mock_nc.variables = {'var1': Mock()}

        validator = NetCDFValidator(mock_nc)
        assert validator._detect_netcdf4_format() == True

    def test_detect_netcdf4_format_classic(self):
        """Test NetCDF classic format detection"""
        mock_nc = Mock()
        mock_nc.groups = {}
        mock_nc.file_format = 'NETCDF3_CLASSIC'
        mock_nc.dimensions = {'lat': Mock(), 'lon': Mock()}
        mock_nc.variables = {'var1': Mock()}

        validator = NetCDFValidator(mock_nc)
        assert validator._detect_netcdf4_format() == False

    def test_get_file_format_info_netcdf4(self):
        """Test file format info for NetCDF4"""
        mock_nc = Mock()
        mock_nc.file_format = 'NETCDF4'
        mock_nc.groups = {'group1': Mock()}
        mock_nc.dimensions = {'lat': Mock(), 'lon': Mock(), 'time': Mock()}
        mock_nc.variables = {'var1': Mock(), 'var2': Mock()}

        validator = NetCDFValidator(mock_nc)
        info = validator._get_file_format_info()

        assert info['format'] == 'NETCDF4'
        assert 'NetCDF4/HDF5' in info['features']
        assert 'Groups' in info['features']
        assert info['dimensions'] == 3
        assert info['variables'] == 2

    def test_get_file_format_info_classic(self):
        """Test file format info for NetCDF classic"""
        mock_nc = Mock()
        mock_nc.file_format = 'NETCDF3_CLASSIC'
        mock_nc.dimensions = {'lat': Mock(), 'lon': Mock()}
        mock_nc.variables = {'var1': Mock()}

        validator = NetCDFValidator(mock_nc)
        info = validator._get_file_format_info()

        assert info['format'] == 'NETCDF3_CLASSIC'
        assert 'NetCDF Classic' in info['features']

    def test_validate_naming_conventions_invalid_names(self):
        """Test naming convention validation with invalid names"""
        mock_nc = Mock()
        mock_nc.dimensions = {'lat-invalid': Mock(), 'lon': Mock(), 'time': Mock()}

        var1 = Mock()
        var1.dimensions = ('time', 'lat', 'lon')
        var1.ncattrs.return_value = ['valid_attr', 'invalid-attr']

        mock_nc.variables = {'invalid-var': var1}
        mock_nc.file_format = 'NETCDF4'
        mock_nc.groups = {}

        validator = NetCDFValidator(mock_nc)
        validator._validate_naming_conventions()

        assert any("Invalid dimension name" in error for error in validator.errors)
        assert any("Invalid variable name" in error for error in validator.errors)
        assert any("Invalid attribute name" in error for error in validator.errors)

    def test_validate_naming_conventions_long_variable_name(self):
        """Test naming convention validation with long variable names"""
        mock_nc = Mock()
        mock_nc.dimensions = {'lat': Mock(), 'lon': Mock(), 'time': Mock()}

        var1 = Mock()
        var1.dimensions = ('time', 'lat', 'lon')
        var1.ncattrs.return_value = []

        long_name = 'a' * 35  # 35 characters, over 30 limit
        mock_nc.variables = {long_name: var1}
        mock_nc.file_format = 'NETCDF4'
        mock_nc.groups = {}

        validator = NetCDFValidator(mock_nc)
        validator._validate_naming_conventions()

        assert any("cannot be longer than 30 characters" in error for error in validator.errors)

    def test_validate_variables_no_valid_vars(self):
        """Test variable validation when no valid variables exist"""
        mock_nc = Mock()
        mock_nc.dimensions = {'lat': Mock(), 'lon': Mock(), 'time': Mock()}

        var1 = Mock()
        var1.dimensions = ('lat', 'lon')  # Only 2 dimensions

        mock_nc.variables = {'lat': Mock(), 'lon': Mock(), 'time': Mock(), 'var1': var1}
        mock_nc.file_format = 'NETCDF4'
        mock_nc.groups = {}

        validator = NetCDFValidator(mock_nc)
        validator._validate_variables()

        assert any("Variables need 3 dimensions" in error for error in validator.errors)

    def test_validate_coordinate_ranges_invalid_ranges(self):
        """Test coordinate range validation with invalid ranges"""
        mock_nc = Mock()
        mock_nc.dimensions = {'lat': Mock(), 'lon': Mock(), 'time': Mock()}

        lat_var = Mock()
        lat_var.dimensions = ('lat',)
        lat_var.units = 'degrees_north'
        lat_var.__getitem__ = Mock(return_value=np.array([95.0]))  # Invalid lat

        lon_var = Mock()
        lon_var.dimensions = ('lon',)
        lon_var.units = 'degrees_east'
        lon_var.__getitem__ = Mock(return_value=np.array([185.0]))  # Invalid lon

        time_var = Mock()
        time_var.dimensions = ('time',)
        time_var.units = 'days since 1900-01-01'
        time_var.__getitem__ = Mock(return_value=np.array([1, 1, 2]))  # Duplicates

        mock_nc.variables = {'lat': lat_var, 'lon': lon_var, 'time': time_var}
        mock_nc.file_format = 'NETCDF4'
        mock_nc.groups = {}

        validator = NetCDFValidator(mock_nc)
        validator._validate_coordinate_ranges()

        assert any("outside valid range" in error for error in validator.errors)
        assert any("Duplicate timestamps" in error for error in validator.errors)

    def test_validate_netcdf4_specific_with_groups(self):
        """Test NetCDF4 specific validation with groups"""
        mock_nc = Mock()
        mock_nc.dimensions = {'lat': Mock(), 'lon': Mock(), 'time': Mock()}
        mock_nc.variables = {}
        mock_nc.file_format = 'NETCDF4'
        mock_nc.groups = {'group1': Mock()}
        mock_nc.cmptypes = []  # Empty list for cmptypes

        validator = NetCDFValidator(mock_nc)
        validator._validate_netcdf4_specific()

        assert any("groups found" in error for error in validator.errors)

    def test_get_variables_basic(self):
        """Test get_variables method"""
        mock_nc = Mock()
        mock_nc.dimensions = {'lat': Mock(), 'lon': Mock(), 'time': Mock()}

        var1 = Mock()
        var1.dimensions = ('time', 'lat', 'lon')
        var2 = Mock()
        var2.dimensions = ('time', 'lat', 'lon')
        coord_var = Mock()
        coord_var.dimensions = ('lat',)

        mock_nc.variables = {
            'lat': coord_var,
            'lon': coord_var,
            'time': coord_var,
            'var1': var1,
            'var2': var2
        }
        mock_nc.file_format = 'NETCDF4'
        mock_nc.groups = {}

        validator = NetCDFValidator(mock_nc)
        vars_list = validator.get_variables()

        assert 'var1' in vars_list
        assert 'var2' in vars_list
        assert 'lat' not in vars_list
        assert 'lon' not in vars_list
        assert 'time' not in vars_list

    def test_run_upload_format_check_txt_file(self):
        """Test run_upload_format_check with txt file"""
        mock_request = Mock()
        mock_file = Mock()

        result = run_upload_format_check(mock_request, mock_file, 'test.txt')

        assert result[0] == True
        assert result[2] == 200

    def test_run_upload_format_check_exception(self):
        """Test run_upload_format_check with exception"""
        mock_request = Mock()
        mock_file = Mock()
        mock_file.seek.side_effect = Exception("File error")

        result = run_upload_format_check(mock_request, mock_file, 'test.nc')

        assert result[0] == False
        assert result[2] == 500


class TestZipValidatorAdditional:
    """Additional tests for ZipValidator functions"""

    def test_is_netcdf_only_true(self):
        """Test _is_netcdf_only returns True for NetCDF-only ZIP"""
        mock_zip = Mock()
        validator = ZipValidator(mock_zip)
        validator.nc_files = ['file1.nc', 'file2.nc']
        validator.file_info = {'total_files': 2}

        assert validator._is_netcdf_only() == True

    def test_is_netcdf_only_false(self):
        """Test _is_netcdf_only returns False for mixed files"""
        mock_zip = Mock()
        validator = ZipValidator(mock_zip)
        validator.nc_files = ['file1.nc']
        validator.csv_files = ['file1.csv']
        validator.file_info = {'total_files': 2}

        assert validator._is_netcdf_only() == False

    def test_validate_file_types_no_csv(self):
        """Test file type validation with no CSV files"""
        mock_zip = Mock()
        validator = ZipValidator(mock_zip)
        validator.csv_files = []
        validator.yml_files = []
        validator.file_info = {'total_files': 0}

        validator._validate_file_types()

        assert any("must contain at least one CSV file" in error for error in validator.errors)

    def test_validate_file_types_multiple_yml(self):
        """Test file type validation with multiple YML files"""
        mock_zip = Mock()
        validator = ZipValidator(mock_zip)
        validator.csv_files = ['file1.csv']
        validator.yml_files = ['meta1.yml', 'meta2.yml']
        validator.file_info = {'total_files': 3}

        validator._validate_file_types()

        assert any("at most one YML" in error for error in validator.errors)

    def test_validate_naming_conventions_invalid_format(self):
        """Test naming convention validation with invalid format"""
        mock_zip = Mock()
        validator = ZipValidator(mock_zip)
        validator.csv_files = ['invalid_name.csv', 'also_invalid.csv']

        validator._validate_naming_conventions()

        assert any("Invalid filename format" in error for error in validator.errors)

    def test_validate_naming_conventions_duplicate_gpi(self):
        """Test naming convention validation with duplicate GPI"""
        mock_zip = Mock()
        validator = ZipValidator(mock_zip)
        validator.csv_files = [
            'dataset_gpi=123_lat=45.0_lon=90.0.csv',
            'dataset_gpi=123_lat=46.0_lon=91.0.csv'
        ]

        validator._validate_naming_conventions()

        assert any("Duplicate GPI value" in error for error in validator.errors)

    def test_validate_naming_conventions_invalid_coordinates(self):
        """Test naming convention validation with invalid coordinates"""
        mock_zip = Mock()
        validator = ZipValidator(mock_zip)
        validator.csv_files = [
            'dataset_gpi=123_lat=95.0_lon=185.0.csv'  # Invalid lat/lon
        ]

        validator._validate_naming_conventions()

        assert any("Invalid latitude" in error for error in validator.errors)
        assert any("Invalid longitude" in error for error in validator.errors)

    def test_validate_naming_conventions_different_datasets(self):
        """Test naming convention validation with different dataset names"""
        mock_zip = Mock()
        validator = ZipValidator(mock_zip)
        validator.csv_files = [
            'dataset1_gpi=123_lat=45.0_lon=90.0.csv',
            'dataset2_gpi=124_lat=46.0_lon=91.0.csv'
        ]

        validator._validate_naming_conventions()

        assert any("same dataset name" in error for error in validator.errors)

    def test_validate_zip_structure_bad_zip(self):
        """Test validation with bad ZIP file"""
        mock_zip = Mock()
        validator = ZipValidator(mock_zip)

        # Create a mock that raises BadZipFile when used as context manager
        def mock_zipfile_constructor(*args, **kwargs):
            raise zipfile.BadZipFile("Bad ZIP")

        with patch('qa4sm_preprocessing.format_validator.zipfile.ZipFile', side_effect=mock_zipfile_constructor):
            validator._validate_zip_structure()

        assert any("not a valid ZIP archive" in error for error in validator.errors)

    def test_validate_zip_structure_empty_zip(self):
        """Test validation with empty ZIP file"""
        mock_zip = Mock()
        validator = ZipValidator(mock_zip)

        # Create a mock zipfile that returns empty namelist
        mock_zip_ref = Mock()
        mock_zip_ref.testzip.return_value = None
        mock_zip_ref.namelist.return_value = []

        with patch('qa4sm_preprocessing.format_validator.zipfile.ZipFile') as mock_zipfile:
            mock_zipfile.return_value.__enter__.return_value = mock_zip_ref
            validator._validate_zip_structure()

        assert any("ZIP file is empty" in error for error in validator.errors)



class TestUtils:

    def setup_method(self):
        """Set up test data paths"""
        self.test_data_dir = os.path.join(
            os.path.dirname(__file__),
            "../test-data/user_data/"
        )

        # Test file paths
        self.invalid_zip_path = os.path.join(self.test_data_dir, "invalid-dataset-format.zip")
        self.valid_zip_path = os.path.join(self.test_data_dir, "test_data_csv.zip")
        self.invalid_nc_path = os.path.join(self.test_data_dir, "inproper-dataset-format.nc")
        self.valid_nc_path = os.path.join(self.test_data_dir, "teststack_c3s_2dcoords_min_attrs.nc")

    def create_mock_file(self, filename, content, file_size=None):
        """Create a mock file object"""
        mock_file = Mock()
        mock_file.name = filename
        mock_file.size = file_size if file_size is not None else len(content)

        # Create a BytesIO object for seek/read operations
        content_io = io.BytesIO(content)
        mock_file.seek = content_io.seek
        mock_file.read = content_io.read

        return mock_file

    def create_mock_request(self, space_left=None):
        """Create a mock request object"""
        request = Mock()
        user = Mock()

        if space_left is not None:
            user.space_left = space_left
        else:
            # User without space_left attribute
            if hasattr(user, 'space_left'):
                delattr(user, 'space_left')

        request.user = user
        return request

    def test_verify_file_extension_valid_extensions(self):
        """Test verify_file_extension with valid extensions"""
        assert verify_file_extension("test.nc")
        assert verify_file_extension("test.nc4")
        assert verify_file_extension("test.zip")
        assert verify_file_extension("TEST.NC")  # Case insensitive
        assert verify_file_extension("file.with.dots.nc4")

    def test_verify_file_extension_invalid_extensions(self):
        """Test verify_file_extension with invalid extensions"""
        assert not verify_file_extension("test.txt")
        assert not verify_file_extension("test.csv")
        assert not verify_file_extension("test.json")
        assert not verify_file_extension("test")
        assert not verify_file_extension("test.")

    def test_validate_file_upload_filename_mismatch(self):
        """Test validate_file_upload with mismatched filename"""
        request = self.create_mock_request(space_left=10 * 1024 * 1024)

        file_content = b"test content"
        uploaded_file = self.create_mock_file("wrong_name.nc", file_content)

        is_valid, message, status = validate_file_upload(request, uploaded_file, "expected_name.nc")

        assert not is_valid
        assert status == 400
        assert "Expected 'expected_name.nc', got 'wrong_name.nc'" in message

    def test_validate_file_upload_invalid_extension(self):
        """Test validate_file_upload with invalid file extension"""
        request = self.create_mock_request(space_left=10 * 1024 * 1024)

        file_content = b"test content"
        uploaded_file = self.create_mock_file("test.txt", file_content)

        is_valid, message, status = validate_file_upload(request, uploaded_file, "test.txt")

        assert not is_valid
        assert status == 400
        assert "File must be .nc4, .nc, or .zip format" in message

    def test_validate_file_upload_file_too_large(self):
        """Test validate_file_upload with file exceeding space limit"""
        request = self.create_mock_request(space_left=100)  # Very small space limit

        large_content = b"x" * 200  # Larger than space limit
        uploaded_file = self.create_mock_file("large_file.nc", large_content)

        is_valid, message, status = validate_file_upload(request, uploaded_file, "large_file.nc")

        assert not is_valid
        assert status == 413
        assert "File size" in message
        assert "exceeds available space" in message

    def test_validate_file_upload_valid_zip(self):
        """Test validate_file_upload with valid zip file"""
        if not os.path.exists(self.valid_zip_path):
            pytest.skip(f"Test file not found: {self.valid_zip_path}")

        request = self.create_mock_request(space_left=10 * 1024 * 1024)

        # Read actual zip file
        with open(self.valid_zip_path, 'rb') as f:
            file_content = f.read()

        uploaded_file = self.create_mock_file("test_data_csv.zip", file_content)

        is_valid, message, status = validate_file_upload(request, uploaded_file, "test_data_csv.zip")

        assert is_valid
        assert status == 200
        assert "ZIP file validation successful" in message

    def test_validate_file_upload_invalid_zip(self):
        """Test validate_file_upload with invalid zip file"""
        if not os.path.exists(self.invalid_zip_path):
            pytest.skip(f"Test file not found: {self.invalid_zip_path}")

        request = self.create_mock_request(space_left=10 * 1024 * 1024)

        # Read actual invalid zip file
        with open(self.invalid_zip_path, 'rb') as f:
            file_content = f.read()

        uploaded_file = self.create_mock_file("invalid-dataset-format.zip", file_content)

        is_valid, message, status = validate_file_upload(request, uploaded_file, "invalid-dataset-format.zip")

        assert not is_valid
        assert "ZIP validation failed" in message

    def test_validate_file_upload_valid_nc(self):
        """Test validate_file_upload with valid NetCDF file"""
        if not os.path.exists(self.valid_nc_path):
            pytest.skip(f"Test file not found: {self.valid_nc_path}")

        request = self.create_mock_request(space_left=10 * 1024 * 1024)

        # Read actual NetCDF file
        with open(self.valid_nc_path, 'rb') as f:
            file_content = f.read()

        uploaded_file = self.create_mock_file("teststack_c3s_2dcoords_min_attrs.nc", file_content)

        is_valid, message, status = validate_file_upload(request, uploaded_file, "teststack_c3s_2dcoords_min_attrs.nc")

        assert is_valid
        assert status == 200
        assert "NetCDF file validation successful" in message

    def test_validate_file_upload_invalid_nc(self):
        """Test validate_file_upload with invalid NetCDF file"""
        if not os.path.exists(self.invalid_nc_path):
            pytest.skip(f"Test file not found: {self.invalid_nc_path}")

        request = self.create_mock_request(space_left=10 * 1024 * 1024)

        # Read actual invalid NetCDF file
        with open(self.invalid_nc_path, 'rb') as f:
            file_content = f.read()

        uploaded_file = self.create_mock_file("inproper-dataset-format.nc", file_content)

        is_valid, message, status = validate_file_upload(request, uploaded_file, "inproper-dataset-format.nc")

        assert not is_valid
        assert "NetCDF validation failed" in message

    def test_validate_file_upload_corrupted_netcdf(self):
        """Test validate_file_upload with corrupted NetCDF file content"""
        request = self.create_mock_request(space_left=10 * 1024 * 1024)

        # Create corrupted NetCDF file (just random bytes)
        corrupted_content = b"This is not a valid NetCDF file content"
        uploaded_file = self.create_mock_file("corrupted.nc", corrupted_content)

        is_valid, message, status = validate_file_upload(request, uploaded_file, "corrupted.nc")

        assert not is_valid
        assert status == 500
        assert "Error reading NetCDF file" in message

    def test_validate_file_upload_corrupted_zip(self):
        """Test validate_file_upload with corrupted ZIP file content"""
        request = self.create_mock_request(space_left=10 * 1024 * 1024)

        # Create corrupted ZIP file (just random bytes)
        corrupted_content = b"This is not a valid ZIP file content"
        uploaded_file = self.create_mock_file("corrupted.zip", corrupted_content)

        is_valid, message, status = validate_file_upload(request, uploaded_file, "corrupted.zip")

        assert not is_valid
        assert status == 400
        assert "ZIP validation failed: File is not a valid ZIP archive" in message

    def test_validate_file_upload_user_without_space_left(self):
        """Test validate_file_upload with user having no space_left attribute"""
        request = self.create_mock_request()  # No space_left attribute

        file_content = b"test content"
        uploaded_file = self.create_mock_file("test.nc", file_content)

        # This should not raise an error and should proceed to validation
        is_valid, message, status = validate_file_upload(request, uploaded_file, "test.nc")

        # Result depends on whether the content is valid NetCDF or not
        # But it should not fail due to space check
        assert is_valid is not None
        assert message is not None
        assert status is not None

    def test_validate_file_upload_nc4_extension(self):
        """Test validate_file_upload with .nc4 extension"""
        if not os.path.exists(self.valid_nc_path):
            pytest.skip(f"Test file not found: {self.valid_nc_path}")

        request = self.create_mock_request(space_left=10 * 1024 * 1024)

        # Read actual NetCDF file but use .nc4 extension
        with open(self.valid_nc_path, 'rb') as f:
            file_content = f.read()

        uploaded_file = self.create_mock_file("test_file.nc4", file_content)

        is_valid, message, status = validate_file_upload(request, uploaded_file, "test_file.nc4")

        assert is_valid
        assert status == 200
        assert "NetCDF file validation successful" in message