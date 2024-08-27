'''
Grab test files to be used through the test suite

Downloading two test .mat files from M1_1 stored on Zenodo
(https://zenodo.org/record/13223574, DOI 10.5281/zenodo.13223573)
and working with these.

- The download takes a while; for faster executions download the files
  to tests/test_data/ (should already be .gitignored )
'''


import pytest
import os
import requests
from kobbe import load


# Function to download files
def download_file(url, local_filename):
    response = requests.get(url, stream=True)
    try:
        response.raise_for_status()
    except requests.exceptions.HTTPError as e:
        raise RuntimeError(
            f'Test failing because remote test files (zenodo) '
            f'could not be accessed. \n-> Check your internet connection?\n\n'
            f'Failed to download file from {url}\n\n Error: "{str(e)}"')
    with open(local_filename, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)


# Fixture to handle the mat files
@pytest.fixture(scope="session")
def mat_files():
    urls = [
        'https://zenodo.org/record/13223574/files/S100812A002_AeN_M1_1.mat?download=1',
        'https://zenodo.org/record/13223574/files/S100812A002_AeN_M1_2.mat?download=1',
        # Add more file URLs as needed
    ]

    local_dir = 'tests/test_data'
    downloaded_files = []

    for url in urls:
        filename = url.split('/')[-1].split('?')[0]
        local_filename = os.path.join(local_dir, filename)
        if not os.path.exists(local_filename):
            print(f'{filename} not found locally. Downloading...')
            download_file(url, local_filename)
        else:
            print(f'Using local file: {local_filename}')
        downloaded_files.append(local_filename)

    yield downloaded_files


@pytest.fixture(scope="session")
def dataset(mat_files):
    ds = load.matfiles_to_dataset(mat_files, lat=80, lon=30)
    return ds