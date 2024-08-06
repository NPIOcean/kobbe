'''
Test function running through the full functionality.

Downloading test files from M1_1 stored on Zenodo 
(https://zenodo.org/record/13223574, DOI 10.5281/zenodo.13223573) 
and works wih these. 

STATUS:
- Setup ok
    - Test files are on zenodo
    - Confirmed elsewhere that I can download files
    - Want to work on packaging for a bit before returning here 
      (skip the nasty import stuff)

'''


import pytest
import os
import requests
from scipy.io import loadmat
import tempfile


# Function: Download files using the requests libary
def download_file(url, local_filename):
    response = requests.get(url, stream=True)
    response.raise_for_status()
    with open(local_filename, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)


# Define a fixture with the two matfiles
@pytest.fixture(scope="module")
def mat_files():
    # List of URLs to download
    urls = [
        'https://zenodo.org/record/13223574/files/S100812A002_AeN_M1_1.mat?download=1',
        'https://zenodo.org/record/13223574/files/S100812A002_AeN_M1_2.mat?download=1',
        # Add more file URLs as needed
    ]

    downloaded_files = []

    # Using the tempfile functionality to assure files are cleaned afer testing
    with tempfile.TemporaryDirectory() as tmpdirname:
        for url in urls:
            local_filename = os.path.join(tmpdirname, url.split('/')[-1].split('?')[0])
            print(f'Downloading {local_filename}...')
            download_file(url, local_filename)
            downloaded_files.append(local_filename)

        yield downloaded_files

        # Cleanup is handled by TemporaryDirectory context manager

# Parameterizing (not sure this will be necessary; the files will be downloaded together)
@pytest.mark.parametrize("file", [
    'https://zenodo.org/record/13223574/files/file1.mat?download=1',
    'https://zenodo.org/record/13223574/files/file2.mat?download=1',
])


### ACTUAL TESTS
def test_mat_file_content(file, mat_files):
    local_filename = next(f for f in mat_files if os.path.basename(f) in file)
    print(f'Loading {local_filename}...')
    mat_data = loadmat(local_filename)
    assert mat_data is not None  # Add your specific tests and assertions here
    print(f'Content of {local_filename}:', mat_data)
