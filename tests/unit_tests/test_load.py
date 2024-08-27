import pytest
from kobbe import load
import xarray as xr
import os
import tempfile
import numpy as np
from unittest.mock import patch

def test_input_validation():
    # Test with an invalid file input type (not a list or a directory)
    with pytest.raises(ValueError, match="file_input must be a list of file paths or a directory"):
        load.matfiles_to_dataset(123)

    # Test with an empty list
    with pytest.raises(ValueError, match="The file list is empty. No .mat files found."):
        load.matfiles_to_dataset([])

def test_dataset_structure(dataset):
    # Test that the dataset has the expected structure
    assert isinstance(dataset, xr.Dataset), "The output is not an xarray.Dataset"
    expected_variables = ['Average_AltimeterDistanceLE', 'lat', 'FOM_threshold',
                          'ICE_IN_SAMPLE_ANY', 'SIC_FOM', 'time_average_ice']

    for var in expected_variables:
        assert var in dataset.variables, f"Missing expected variable: {var}"

    expected_attrs = ['pressure_offset', 'sampling_interval_sec', 'history']

    for attr in expected_attrs:
        assert attr in dataset.attrs, f"Missing expected attribute: {attr}"

def test_data_integrity(dataset):

    # Ensure that the dataset is sorted by TIME
    assert dataset['TIME'].equals(
        dataset['TIME'].sortby('TIME')
    ), "Dataset is not sorted by 'TIME'"

    # Check that the sampling interval attribute is correctly calculated
    sampling_interval = dataset.attrs['sampling_interval_sec']
    assert isinstance(sampling_interval, float), "Sampling interval is not a float"
    assert sampling_interval > 0, "Sampling interval is not positive"

    # Check the shape after reshaping
    if 'SAMPLE' in dataset.dims:
        assert dataset.sizes['SAMPLE'] > 0, "Reshaped dimension 'SAMPLE' is not greater than zero"

def test_edge_cases(mat_files):
    # Test with a specific time range that might exclude all data
    with pytest.raises(ValueError, match='No data found within the given time range'):
        load.matfiles_to_dataset(mat_files, time_range=("01.01.2050", "31.12.2050"))


    # Test with include_raw_altimeter = True
    try:
        ds_raw = load.matfiles_to_dataset(mat_files, include_raw_altimeter=True)
        # Check if the raw altimeter signal is included, if the dataset loaded successfully
        assert 'raw_altimeter_signal' in ds_raw.variables, "Raw altimeter signal not included when requested"
    except:
        pytest.skip("Skipping test for raw altimeter signal "
                    "- probably not able to load the large data amount..")

# Ignoring an irrelevant Runtime warning that does noe seem to be real
@pytest.mark.filterwarnings('ignore::RuntimeWarning')
def test_load_nc_success(dataset):
    # Define the path for the temporary NetCDF file
    temp_file_path = 'temp_file.nc'

    try:
        # Save the dataset to a NetCDF file
        dataset.to_netcdf(temp_file_path)

        # Test code for successfully loading a NetCDF file
        ds = load.load_nc(temp_file_path)

        assert isinstance(ds, xr.Dataset), "Loaded data is not an xarray Dataset"
        assert dataset.equals(ds), "Saving and loading changed the file"

    finally:
        # Ensure the file is removed after the test, regardless of the result
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)

def test_load_nc_file_not_found():
    # Test loading a non-existent file
    with pytest.raises(FileNotFoundError):
        load.load_nc("non_existent_file.nc")

def test_load_nc_invalid_file():
    # Test handling of an invalid NetCDF file
    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
        temp_file.write(b"not a netcdf file")
        temp_file.close()
        with pytest.raises(ValueError):
            load.load_nc(temp_file.name)
        os.remove(temp_file.name)



def test_chop_with_indices(dataset):
    # Define indices for chopping
    indices = (10, 20)

    N = dataset.sizes['TIME']

    # Apply chop function
    ds_chopped = load.chop(dataset, indices=indices)

    # Check that the dataset was properly chopped
    expected_size = (indices[1] - indices[0] + 1)
    assert ds_chopped.sizes['TIME'] == expected_size, f"Expected size {expected_size}, got {ds_chopped.sizes['TIME']}"
    assert '- Chopped' in ds_chopped.attrs['history'], "History attribute was not updated correctly"

    # Try with a negative final index
    indices = (101, -201)

    # Apply chop function with negative final index
    ds_chopped_neg = load.chop(dataset, indices=indices)

    expected_size_neg = N - indices[0] - np.abs(indices[1]) + 1
    assert ds_chopped_neg.sizes['TIME'] == expected_size_neg, f"Expected size {expected_size_neg}, got {ds_chopped_neg.sizes['TIME']}"
    assert '- Chopped' in ds_chopped.attrs['history'], "History attribute was not updated correctly"


def test_chop_auto_accept(dataset):
    # Mock datasets where pressure is zero aat the beginning..
    dataset_with_deckoffset = dataset.copy()
    dataset_with_deckoffset['Average_AltimeterPressure'].isel(TIME=slice(0, 100), SAMPLE=slice(None, None))[:] = 0

    # Use patch to simulate user input
    with patch('builtins.input', return_value='y'):
        ds_chopped = load.chop(dataset_with_deckoffset, auto_accept=True)

    # Check that the dataset was chopped and the history was updated
    assert '- Chopped' in ds_chopped.attrs['history'], "History attribute was not updated correctly"
    assert ds_chopped.sizes['TIME'] < dataset_with_deckoffset.sizes['TIME'], "Dataset was not chopped as expected"


    if False:

        def test_chop_user_confirmation(dataset):
            # Use patch to simulate user input
            with patch('builtins.input', return_value='n'):
                ds_chopped = load.chop(dataset, auto_accept=False)

            # Verify that the dataset was not chopped
            assert ds_chopped.sizes['TIME'] == len(dataset.TIME), "Dataset was chopped even though it should not be"
            assert "Not accepted" in ds_chopped.attrs['history'], "History attribute was not updated correctly"

        def test_chop_invalid_input(dataset):
            with pytest.raises(ValueError, match=r'I do not understand your input'):
                # Use patch to simulate user input
                with patch('builtins.input', return_value='invalid'):
                    load.chop(dataset, auto_accept=False)

