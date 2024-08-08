'''
Test function running through the full functionality.

Does not test all functionality, runs through the whole processing chain.


Downloading two test .mat files from M1_1 stored on Zenodo 
(https://zenodo.org/record/13223574, DOI 10.5281/zenodo.13223573) 
and working with these.

- The download takes a while; for faster executions download the files
  to tests/test_data/ (should already be .gitignored )

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
import kobbe
import xarray as xr
import numpy as np


# Function to download files
def download_file(url, local_filename):
    response = requests.get(url, stream=True)
    try:
        response.raise_for_status()
    except requests.exceptions.HTTPError as e:
        raise RuntimeError(f'Test failing because remote test files (zenodo) '
            f'could not be accessed. \n-> Check your internet connection?\n\n'
            f'Failed to download file from {url}\n\n Error: "{str(e)}"')
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

def test_load_mat_files(mat_files):

    # Call the load function and 
    ds = kobbe.load.matfiles_to_dataset(mat_files, lat = 85, lon = 20)

    # Test that it produces an xr Dataset.
    assert isinstance(ds, xr.Dataset), "The variable ds is not an xarray.Dataset instance."

    # Create mock SLP data
    mock_time = np.arange(ds.TIME[0], ds.TIME[-1])  
    L_mock = len(mock_time)
    mock_slp = np.random.rand(L_mock) + 10  # Sea level pressure in dbar

    slp_attrs = {'units': 'dbar', 'long_name': 'Sea level pressure at location'}

    # Run the append_atm_pres function
    ds = kobbe.append.append_atm_pres(ds, mock_slp, mock_time, attrs = slp_attrs)

    # Assertions to check the expected outcomes (data + attributes)
    assert 'p_atmo' in ds, "The variable 'p_atmo' was not added to the dataset."
    assert ds['p_atmo'].attrs['units'] == 'dbar'
    assert ds['p_atmo'].attrs['long_name'] == 'Sea level pressure at location'

    # Create mock CTD data
    mock_temp = np.random.rand(L_mock)*5 -1.8  # Temperature 
    mock_sal = np.random.rand(L_mock)*1.5 + 33.5  # Salinity 
    mock_pres = np.random.rand(L_mock) + 20  # Pressure 

    # Run the append_ctd function
    ds = kobbe.append.append_ctd(ds, mock_temp, mock_sal, mock_pres, mock_time,
             instr_SN = '11111', 
           instr_desc = 'RBR Concerto CTD sensor mounted right below the Signature' )
    
    # Assertions to check the expected outcomes (data + attributes)
    assert 'SA_CTD' in ds, "No variable 'SA_CTD' in the dataset."
    assert 'CT_CTD' in ds, "No variable 'CT_CTD' in the dataset."
    assert 'pres_CTD' in ds, "No variable 'pres_CTD' in the dataset."
    assert 'sound_speed_CTD' in ds, "No variable 'sound_speed_CTD' in the dataset."
    assert 'rho_CTD' in ds, "No variable 'rho_CTD' in the dataset."

    # Create mock magnetic declination data
    mock_magdec_time = [mock_time[0], mock_time[-1]] 
    mock_magdec = [20, 20.3] 

    # Run the append_magdec function
    ds = kobbe.append.append_magdec(ds, mock_magdec, mock_magdec_time)
    assert 'magdec' in ds, "The variable 'magdec' was not added to the dataset."
    assert ds['magdec'].attrs['units'] == 'degrees'

    # Create mock SIC data (example of appending non-standard variable)
    mock_sic = np.random.rand(L_mock)*100  
    ds = kobbe.append.add_to_sigdata(ds, mock_sic, mock_time, 'SIC_AMSR2', 
               attrs = {'units':'%'})
    assert 'SIC_AMSR2' in ds, "The variable 'SIC_AMSR2' was not added to the dataset."
    assert ds['SIC_AMSR2'].attrs['units'] == '%'

    # Calculate depth from pressure with density correction
    ds = kobbe.calc.dep_from_p(ds, corr_CTD_density=True)

    # Check that "depth" and "g" were added
    assert 'depth' in ds, "The variable 'depth' was not added to the dataset."
    assert 'g' in ds, "The variable 'g' was not added to the dataset."


    # Calculate sea ice draft
    ds = kobbe.icedraft.calculate_draft(ds)

    # The operation should generate a bunh of new variables:
    var_list_draft = ['SURFACE_DEPTH_LE',
        'SURFACE_DEPTH_AST',
        'SURFACE_DEPTH_LE_UNCAPPED',
        'SEA_ICE_DRAFT_LE',
        'SEA_ICE_DRAFT_MEDIAN_LE',
        'SEA_ICE_DRAFT_AST',
        'SEA_ICE_DRAFT_MEDIAN_AST']

    # Check that these were all generated
    all_elements_in_keys_draft = all(var_name in ds.keys() 
                                     for var_name in var_list_draft)
    assert all_elements_in_keys_draft, ("kobbe.icedraft.calculate_draft did"
        " not generate all 7 expected variables (SEA_ICE_DRAFT_LE, etc).")
    # Check that the "note" metadata field contains a note that we havent 
    # applied OW correction
    assert "OPEN WATER CORRECTION **NOT** APPLIED!" in ds.SEA_ICE_DRAFT_MEDIAN_LE.note
    assert "OPEN WATER CORRECTION **NOT** APPLIED!" in ds.SEA_ICE_DRAFT_AST.note
    assert "OPEN WATER CORRECTION **NOT** APPLIED!" in ds.SURFACE_DEPTH_LE.note

    # Get open water sound speed correction
    ds = kobbe.icedraft.get_Beta_from_OWSD(ds)

    # The operation should generate a bunch of new variables:
    var_list_owc = ['BETA_open_water_corr_LE',
        'BETA_open_water_corr_AST',
        'OW_surface_before_correction_LE',
        'OW_surface_before_correction_AST']
    # Check that these were all generated
    all_elements_in_keys_owc = all(var_name in ds.keys() for var_name in var_list_owc)
    assert all_elements_in_keys_owc, ("kobbe.icedraft.get_Beta_from_OWSD did"
        " not generate all 4 expected variables (BETA_open_water_corr_LE, etc).")


    # Recalculate sea ice draft (should not use OWC)
    ds = kobbe.icedraft.calculate_draft(ds)
    # Check that the "note" metadata field NO LONGER contains a note that we havent 
    # applied OW correction
    assert "OPEN WATER CORRECTION **NOT** APPLIED!" not in ds.SEA_ICE_DRAFT_MEDIAN_LE.note
    assert "OPEN WATER CORRECTION **NOT** APPLIED!" not in ds.SEA_ICE_DRAFT_AST.note
    assert "OPEN WATER CORRECTION **NOT** APPLIED!" not in ds.SURFACE_DEPTH_LE.note

    # Test chopping function
    L0 = ds.sizes['TIME']
    # Chop way the two last ensembles
    ds = kobbe.load.chop(ds, indices = (0, L0-3), auto_accept=True)
    L1 = ds.sizes['TIME']
    # Check that we lost two ensembles
    assert L0-L1==2, 'Chopping did not work as expected'
    # Check that there is a note in the history field
    assert "Chopped 2 ensembles using " in ds.history

    # Calculate sea ice velocity
    ds = kobbe.vel.calculate_ice_vel(ds)

    # The operation should generate a bunch of new variables:
    var_list_ivel = ['uice', 'vice', 'Uice', 'Vice', 'Uice_SD', 'Vice_SD']

    # Check that these were all generated
    all_elements_in_keys_ivel = all(var_name in ds.keys() 
                                    for var_name in var_list_ivel)
    assert all_elements_in_keys_ivel, ("kobbe.icedraft.calculate_ice_vel did"
        " not generate all 6 expected variables (uice, Vice, etc).")



    # Calculate ocean velocity
    ds = kobbe.vel.calculate_ocean_vel(ds)

    # The operation should generate a bunch of new variables:
    var_list_ovel = ['bin_depth', 'uocean', 'vocean', 'Uocean', 'Vocean']

    # Check that these were all generated
    all_elements_in_keys_ovel = all(var_name in ds.keys() for var_name in var_list_ovel)
    assert all_elements_in_keys_ovel, ("kobbe.icedraft.calculate_ocean_vel did"
        " not generate all 5 expected variables (bin_depth, uocean, etc).")


    ### Ocean velocity editing
    
    # Sidelobe rejection
    nans_before_sidelobe = int(np.isnan(ds['uocean']).sum())
    ds = kobbe.vel.reject_sidelobe(ds)
    nans_after_sidelobe = int(np.isnan(ds['uocean']).sum())

    # Check that we NaN'ed out a bunch of entries (NaNs should in fact have 
    # gone from 0 to > 50% of the dataset..) 
    assert nans_after_sidelobe > nans_before_sidelobe, "Sidelobe rejection did not work right"
    # Check metadata note
    assert ("Rejected samples close enough to the surface to be "
            "affected by sidelobe interference" 
             in ds.uocean.processing_history)


    # Range masking (threshold for various parameters like corr, amp, speed, tilt,, amp jumps....)
    nans_before_rangemask = int(np.isnan(ds['uocean']).sum())
    ds = kobbe.vel.uvoc_mask_range(ds)
    nans_after_rangemask = int(np.isnan(ds['uocean']).sum())
    # Check that we NaN'ed out a bunch of entries (NaNs should in fact have 
    # gone from 0 to > 50% of the dataset..) 
    assert nans_after_rangemask > nans_before_rangemask, "Range masking rejection did not work right"
    # Check metadata note
    assert "THRESHOLD-BASED DATA CLEANING" in ds.uocean.processing_history

    # Clearing near-empty velocity bins
    nbins_before_clearing = ds.sizes['BINS']
    ds = kobbe.vel.clear_empty_bins(ds)
    nbins_after_clearing = ds.sizes['BINS']
    # Check that we reduced number of bins
    assert nbins_after_clearing < nbins_before_clearing, "Bin clearing did not work right"
    # Check metadata note
    assert "bins where less than" in ds.history


    # Magnetic declination correction
    ds_urot = ds.copy()
    ds = kobbe.vel.rotate_vels_magdec(ds)

    # Test that original and rotated datasets have the same speed
    # Boolean entries: Are rotated and original speeds the same  
    bool_arr_speed = xr.apply_ufunc(
        np.isclose, 
        ds.uocean**2 + ds.vocean**2, 
        ds_urot.uocean**2 + ds_urot.vocean**2,
        kwargs={"atol": 1e-7},  # Allow some numerical error differences
    )
    # (Setting nan velocities to True - only want the comparison between actual speeds) 
    bool_arr_speed = bool_arr_speed.where(~np.isnan(ds.uocean))
    bool_arr_speed = bool_arr_speed.fillna(value = True)

    assert bool_arr_speed.all(), f"Magnetic correction rotation seems to have changed the speed!"


    # Test that original and rotated datasets no NOT have the same u velocity
    # Boolean entries: Are rotated and original u the same  
    bool_arr_u = xr.apply_ufunc(
        np.isclose, 
        ds.uocean, 
        ds_urot.uocean,
        kwargs={"atol": 1e-7},  # Allow some numerical error differences
    )

    # (Setting nan velocities to True - only want the comparison between actual velocities) 
    bool_arr_u = bool_arr_u.where(~np.isnan(ds.uocean))
    bool_arr_u = bool_arr_u.fillna(value = True)

    assert not bool_arr_u.all(), f"Magnetic correction rotation seems NOT to have changed the u velocity!"


    # Interpolate ocean velocity
    ds = kobbe.vel.interp_oceanvel(ds, 10)
    assert all(var_name in ds.keys() for var_name in ['Vocean_10m', 'Uocean_10m'])

