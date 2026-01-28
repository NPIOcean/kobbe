'''
Test function running through the full functionality.

Does not test all functionality, runs through the whole processing chain.

'''

import pytest
import kobbe
import xarray as xr
import numpy as np


def test_load_mat_files(mat_files):

    # Call the load function and
    ds = kobbe.load.matfiles_to_dataset(mat_files, lat=80, lon=30)

    # Test that it produces an xr Dataset.
    assert isinstance(ds, xr.Dataset), "The variable ds is not an xarray.Dataset instance."

    # Create mock SLP data
    mock_time = np.arange(ds.TIME[0], ds.TIME[-1])
    L_mock = len(mock_time)
    mock_slp = np.random.rand(L_mock) + 10  # Sea level pressure in dbar

    slp_attrs = {'units': 'dbar', 'long_name': 'Sea level pressure'}

    # Run the append_atm_pres function
    ds = kobbe.append.append_atm_pres(ds.copy(), mock_slp, mock_time, attrs = slp_attrs)

    # Assertions to check the expected outcomes (data + attributes)
    assert 'p_atmo' in ds, "append_atm_pres() did not add the variable 'p_atmo' to the dataset."
    assert ds['p_atmo'].attrs['units'] == 'dbar'
    assert ds['p_atmo'].attrs['long_name'] == 'Sea level pressure'
    assert (9 < ds.p_atmo.mean() < 11).item()

    # Droppig for now since it takes so long. Look into this again when making
    # unit tests..
    if False:
        # Run the append_atm_pres-auto function
        ds2 = kobbe.append.append_atm_pres_auto(ds.copy())
        assert 'p_atmo' in ds2, "append_atm_pres() did not add the variable 'p_atmo' to the dataset."
        assert ds2['p_atmo'].attrs['units'] == 'dbar'
        assert ds2['p_atmo'].attrs['long_name'] == 'Sea level pressure'
        assert (9 < ds2.p_atmo.mean() < 11).item()

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
    ds2 = ds.copy()
    ds2 = kobbe.append.append_magdec(ds2, mock_magdec, mock_magdec_time)
    assert 'magdec' in ds2, "The variable 'magdec' was not added to the dataset."
    assert ds2['magdec'].attrs['units'] == 'degrees'

    # Run the append_magdec_auto function
    ds = kobbe.append.append_magdec_auto(ds)
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
    assert 'instr_depth' in ds, "The variable 'instr_depth' was not added to the dataset."
    assert 'g' in ds, "The variable 'g' was not added to the dataset."


    # Calculate sea ice draft
    ds = kobbe.icedraft.calculate_draft(ds)

    # The operation should generate a bunh of new variables:
    var_list_draft = ['SURFACE_DEPTH_LE',
        'SURFACE_DEPTH_AST',
        'SEA_ICE_DRAFT_LE',
        'SEA_ICE_DRAFT_MEDIAN_LE',
        'SEA_ICE_DRAFT_AST',
        'SEA_ICE_DRAFT_MEDIAN_AST']


    # Check that these were all generated
    all_elements_in_keys_draft = all(var_name in ds.keys()
                                     for var_name in var_list_draft)
    assert all_elements_in_keys_draft, ("kobbe.icedraft.calculate_draft did"
        " not generate all 6 expected variables (SEA_ICE_DRAFT_LE, etc).")
    # Check that the "note" metadata field contains a note that we havent
    # applied OW correction
    assert "No fixed offset alpha " in ds.SEA_ICE_DRAFT_MEDIAN_LE.note
    assert "No open water corrective factor beta" in ds.SEA_ICE_DRAFT_MEDIAN_LE.note
    assert "No fixed offset alpha " in ds.SEA_ICE_DRAFT_AST.note
    assert "No open water corrective factor beta" in ds.SEA_ICE_DRAFT_AST.note
    assert "No fixed offset alpha " in ds.SURFACE_DEPTH_LE.note
    assert "No open water corrective factor beta" in ds.SURFACE_DEPTH_LE.note

    # Get open water sound speed correction
    ds = kobbe.icedraft.get_open_water_correction(ds)

    # The operation should generate a bunch of new variables:
    var_list_owc = [
        'alpha_LE', 'alpha_AST', 'beta_LE', 'beta_AST',
        'ow_surface_before_correction_LE_LP',
        'ow_surface_before_correction_LE',
        'ow_surface_before_correction_AST_LP',
        'ow_surface_before_correction_AST']

    # Check that these were all generated
    all_elements_in_keys_owc = all(
        var_name in ds.keys() for var_name in var_list_owc)
    assert all_elements_in_keys_owc, (
        "kobbe.icedraft.get_open_water_correction() did"
        f" not generate all 8 expected variables  {var_list_owc}.")

    # Recalculate sea ice draft (should not use OWC)
    ds = kobbe.icedraft.calculate_draft(ds)
    # Check that the "note" metadata field NO LONGER contains a note that we havent
    # applied OW correction
    assert "No fixed offset alpha " not in ds.SEA_ICE_DRAFT_MEDIAN_LE.note
    assert "No open water corrective factor beta" not in ds.SEA_ICE_DRAFT_MEDIAN_LE.note
    assert "No fixed offset alpha " not in ds.SEA_ICE_DRAFT_AST.note
    assert "No open water corrective factor beta" not in ds.SEA_ICE_DRAFT_AST.note
    assert "No fixed offset alpha " not in ds.SURFACE_DEPTH_LE.note
    assert "No open water corrective factor beta" not in ds.SURFACE_DEPTH_LE.note


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
    var_list_ivel = ['uice', 'vice', 'UICE', 'VICE', 'UICE_SD', 'VICE_SD']

    # Check that these were all generated
    all_elements_in_keys_ivel = all(var_name in ds.keys()
                                    for var_name in var_list_ivel)
    assert all_elements_in_keys_ivel, ("kobbe.icedraft.calculate_ice_vel did"
        " not generate all 6 expected variables (uice, VICE, etc).")


    # Calculate ocean velocity
    ds = kobbe.vel.calculate_ocean_vel(ds)

    # The operation should generate a bunch of new variables:
    var_list_ovel = ['BIN_DEPTH', 'ucur', 'vcur', 'UCUR', 'VCUR']

    # Check that these were all generated
    all_elements_in_keys_ovel = all(var_name in ds.keys() for var_name in var_list_ovel)
    assert all_elements_in_keys_ovel, ("kobbe.icedraft.calculate_ocean_vel did"
        " not generate all 5 expected variables (BIN_DEPTH, ucur, VCUR, etc).")


    ### Ocean velocity editing

    # Sidelobe rejection
    nans_before_sidelobe = int(np.isnan(ds['ucur']).sum())
    ds = kobbe.vel.reject_sidelobe(ds)
    nans_after_sidelobe = int(np.isnan(ds['ucur']).sum())

    # Check that we NaN'ed out a bunch of entries (NaNs should in fact have
    # gone from 0 to > 50% of the dataset..)
    assert nans_after_sidelobe > nans_before_sidelobe, "Sidelobe rejection did not work right"
    # Check metadata note

    # Deprecated: Check processing history ( no longer implementing; 
    # may implement this in the PROCESSING variable in the future)
    if False: 
        assert ("Rejected samples close enough to the surface to be "
                "affected by sidelobe interference"
                in ds.ucur.processing_history)


    # Range masking (threshold for various parameters like corr, amp, speed, tilt,, amp jumps....)
    nans_before_rangemask = int(np.isnan(ds['ucur']).sum())
    ds = kobbe.vel.uvoc_mask_range(ds)
    nans_after_rangemask = int(np.isnan(ds['ucur']).sum())
    # Check that we NaN'ed out a bunch of entries (NaNs should in fact have
    # gone from 0 to > 50% of the dataset..)
    assert nans_after_rangemask > nans_before_rangemask, "Range masking rejection did not work right"
    # Check metadata note


    # Deprecated: Check processing history ( no longer implementing; 
    # may implement this in the PROCESSING variable in the future)
    if False: 
      assert "THRESHOLD-BASED DATA CLEANING" in ds.ucur.processing_history

    # Clearing near-empty velocity bins
    nbins_before_clearing = ds.sizes['VEL_BIN']
    ds = kobbe.vel.clear_empty_bins(ds)
    nbins_after_clearing = ds.sizes['VEL_BIN']
    # Check that we reduced number of bins
    assert nbins_after_clearing < nbins_before_clearing, "Bin clearing did not work right"
    # Check metadata note
    # Deprecated: Check processing history ( no longer implementing; 
    # may implement this in the PROCESSING variable in the future)
    if False: 
        assert "bins where less than" in ds.history


    # Magnetic declination correction
    ds_urot = ds.copy()
    ds = kobbe.vel.rotate_vels_magdec(ds)

    # Test that original and rotated datasets have the same speed
    # Boolean entries: Are rotated and original speeds the same
    bool_arr_speed = xr.apply_ufunc(
        np.isclose,
        ds.ucur**2 + ds.vcur**2,
        ds_urot.ucur**2 + ds_urot.vcur**2,
        kwargs={"atol": 1e-7},  # Allow some numerical error differences
    )
    # (Setting nan velocities to True - only want the comparison between actual speeds)
    bool_arr_speed = bool_arr_speed.where(~np.isnan(ds.ucur))
    bool_arr_speed = bool_arr_speed.fillna(value = True)

    assert bool_arr_speed.all(), f"Magnetic correction rotation seems to have changed the speed!"


    # Test that original and rotated datasets no NOT have the same u velocity
    # Boolean entries: Are rotated and original u the same
    bool_arr_u = xr.apply_ufunc(
        np.isclose,
        ds.ucur,
        ds_urot.ucur,
        kwargs={"atol": 1e-7},  # Allow some numerical error differences
    )

    # (Setting nan velocities to True - only want the comparison between actual velocities)
    bool_arr_u = bool_arr_u.where(~np.isnan(ds.ucur))
    bool_arr_u = bool_arr_u.fillna(value = True)

    assert not bool_arr_u.all(), f"Magnetic correction rotation seems NOT to have changed the u velocity!"


    # Interpolate ocean velocity
    ds = kobbe.vel.interp_oceanvel(ds, 10)
    assert all(var_name in ds.keys() for var_name in ['VCUR_10m', 'UCUR_10m'])

