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
from kobbe import load
import xarray as xr


# Function to download files
def download_file(url, local_filename):
    response = requests.get(url, stream=True)
    response.raise_for_status()
    with open(local_filename, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)

# Fixture to handle the mat files
@pytest.fixture(scope="module")
def mat_files():
    urls = [
        'https://zenodo.org/record/13223574/files/S100812A002_AeN_M1_1.mat?download=1',
        'https://zenodo.org/record/13223574/files/S100812A002_AeN_M1_2.mat?download=1',
        # Add more file URLs as needed
    ]

    downloaded_files = []

    with tempfile.TemporaryDirectory() as tmpdirname:
        for url in urls:
            local_filename = os.path.join(tmpdirname, url.split('/')[-1].split('?')[0])
            print(f'Downloading {local_filename}...')
            download_file(url, local_filename)
            downloaded_files.append(local_filename)

        yield downloaded_files

        # Cleanup handled by TemporaryDirectory context manager


def test_load_mat_files(mat_files):
    # Call the function you want to test
    ds = load.matfiles_to_dataset(mat_files, lat = 79.589, lon = 28.097)
    assert isinstance(ds, xr.Dataset), "The variable ds is not an xarray.Dataset instance."