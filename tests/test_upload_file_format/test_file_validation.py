import pytest
from unittest.mock import Mock
import os

from qa4sm_preprocessing.utils import validate_file_upload
from qa4sm_preprocessing.format_validator import NetCDFValidator


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
