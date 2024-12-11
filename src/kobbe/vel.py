"""
KOBBE.VEL

Functions for processing ocean and sea ice drift velocity.
"""

import numpy as np
import xarray as xr
from scipy.interpolate import interp1d
import warnings
from typing import Tuple, Optional


def calculate_ice_vel(
    ds: xr.Dataset,
    avg_method: str = "median",
    min_per_ensemble: float = 1,
    sspd_scaling: bool = False,
    true_sspd_name: str = 'sound_speed_CTD'
) -> xr.Dataset:
    """
    Calculate sea ice drift velocity from the AverageIce fields.

    Parameters:
    -----------
    ds : xarray.Dataset
        Input dataset containing the AverageIce velocity components.
    avg_method : str, optional
        Method to calculate ensemble average ('median' or 'mean'). Default is
        'median'.
    min_per_ensemble : int, optional
        Minimum number of samples required to produce a valid ensemble average.
    sspd_scaling : bool, optional
        If True, scales the ice velocity by the ratio of actual sound speed to
        averaged sound speed.
    true_sspd_name : str, optional
        The name of the dataset field containing the true sound speed values.
        Default is 'sound_speed_CTD'.

    Returns:
    --------
    xarray.Dataset
        Updated dataset with 'uice' and 'vice' fields added.
    """

    # Add uice and vice fields for eastward and northward sea ice drift velocity
    ds["uice"] = (
        ("TIME", "SAMPLE"),
        ds.AverageIce_VelEast.data,
        {
            "units": "m s-1",
            "long_name": "Eastward sea ice drift velocity",
            "details": "All average mode samples",
        },
    )
    ds["vice"] = (
        ("TIME", "SAMPLE"),
        ds.AverageIce_VelNorth.data,
        {
            "units": "m s-1",
            "long_name": "Northward sea ice drift velocity",
            "details": "All average mode samples",
        },
    )

    # Mask data where no ice is present
    ds["uice"] = ds["uice"].where(ds.ICE_IN_SAMPLE)
    ds["vice"] = ds["vice"].where(ds.ICE_IN_SAMPLE)

    # Add processing history
    for key in ["uice", "vice"]:
        ds[key].attrs["processing_history"] = "Loaded from AverageIce_VelEast/AverageIce_VelNorth fields.\n"

    # Perform sound speed scaling if applicable
    if sspd_scaling:
        if true_sspd_name in ds:
            ds = _sspd_correction_ice_vel(ds, true_sspd_name=true_sspd_name)
        else:
            print(f"Could not perform sound speed scaling of ice velocity (field '{true_sspd_name}' not found)")

    # Calculate average ice velocity
    ds = _calculate_uvice_avg(ds, avg_method=avg_method,
                              min_per_ensemble=min_per_ensemble)

    return ds


def _sspd_correction_ice_vel(
    ds: xr.Dataset,
    true_sspd_name: str = 'sound_speed_CTD'
) -> xr.Dataset:
    """
    Apply sound speed correction to ice velocity.

    This function scales the ice drift velocity fields ('uice' and 'vice') by the ratio of the true sound speed
    to the average sound speed in the dataset.

    Parameters:
    -----------
    ds : xarray.Dataset
        The dataset containing the velocity and sound speed data.
    true_sspd_name : str
        The name of the dataset field containing the true sound speed values.

    Returns:
    --------
    xarray.Dataset
        The dataset with corrected velocity fields.
    """
    # Calculate the sound speed correction factor
    sound_speed_factor_ice = ds[true_sspd_name] / ds.Average_Soundspeed
    mean_sspd_factor_ice = sound_speed_factor_ice.mean()

    # Apply correction to 'uice' and 'vice'
    for uv_key in ['uice', 'vice']:
        if uv_key in ds:
            ds[uv_key] = ds[uv_key].copy() * sound_speed_factor_ice

            proc_add = (
                'Amplitude scaled by improved sound speed estimate '
                f'(mean scaling factor {mean_sspd_factor_ice:.4f}).'
            )
            ds[uv_key].attrs['processing_history'] = (
                ds[uv_key].attrs.get('processing_history', '') + proc_add)

    return ds



def _calculate_uvice_avg(
        ds: xr.Dataset,
        avg_method: str = "median",
        min_per_ensemble: int = 1,
) -> xr.Dataset:
    """
    Calculate ensemble average sea ice velocity.

    Parameters:
    -----------
    ds : xarray.Dataset
        Input dataset containing sea ice velocity components.
    avg_method : str, optional
        Method to calculate ensemble average ('median' or 'mean'). Default is
        'median'.
    min_per_ensemble : int, optional
        Minimum number of samples required to produce a valid ensemble average.

    Returns:
    --------
    xarray.Dataset
        Updated dataset with 'UICE' and 'VICE' fields added.
    """

    if avg_method == "median":
        ds["UICE"] = ds["uice"].median(dim="SAMPLE")
        ds["VICE"] = ds["vice"].median(dim="SAMPLE")

    elif avg_method == "mean":
        ds["UICE"] = ds["uice"].mean(dim="SAMPLE")
        ds["VICE"] = ds["vice"].mean(dim="SAMPLE")
    else:
        raise Exception(
            f'Invalid "avg_method" ("{avg_method}").'
            ' Must be "mean" or "median".'
        )


    ds.UICE.attrs = {
        "units": "m s-1",
        "long_name": "Eastward sea ice drift velocity",
        "details": "Ensemble average (%s)" % avg_method,
        "processing_history": ds.uice.processing_history,
    }
    ds.VICE.attrs = {
        "units": "m s-1",
        "long_name": "Northward sea ice drift velocity",
        "details": "Ensemble average (%s)" % avg_method,
        "processing_history": ds.vice.processing_history,
    }

    with warnings.catch_warnings():  # Suppressing (benign) warning yielded
        # when computing std() over all-nan slice..
        warnings.filterwarnings(
            action="ignore", message="Degrees of freedom <= 0 for slice"
        )
        ds["UICE_SD"] = ds["uice"].std(dim="SAMPLE", skipna=True)
        ds["VICE_SD"] = ds["vice"].std(dim="SAMPLE", skipna=True)

    ds.UICE_SD.attrs = {
        "units": "m s-1",
        "long_name": (
            "Ensemble standard deviation of " "eastward sea ice drift velocity"
        ),
    }
    ds.VICE_SD.attrs = {
        "units": "m s-1",
        "long_name": (
            "Ensemble standard deviation of "
            "northward sea ice drift velocity"
        ),
    }

    # Set to NaN where the number of valid samples

    n_valid = ds.uice.notnull().sum(dim='SAMPLE')

    for key in ['UICE', 'VICE', 'UICE_SD', 'VICE_SD']:
        ds[key] = ds[key].where(n_valid >= min_per_ensemble)

    return ds


def calculate_ocean_vel(
        ds: xr.Dataset,
        avg_method: str = "median",
        sspd_scaling: bool = False,
        true_sspd_name: str = 'sound_speed_CTD'
    ) -> xr.Dataset:



    """
    Calculate ocean velocity from the Average_VelEast and Average_VelNorth
    fields.

    Parameters:
    -----------
    ds : xarray.Dataset
        Input dataset containing the Average_VelEast and Average_VelNorth
        fields.
    avg_method : str, optional
        Method to calculate ensemble average ('median' or 'mean'). Default is
        'median'.
    sspd_scaling : bool, optional
        If True, scales the ice velocity by the ratio of actual sound speed to averaged sound speed.
    true_sspd_name : str, optional
        The name of the dataset field containing the true sound speed values. Default is 'sound_speed_CTD'.

    Returns:
    --------
    xarray.Dataset
        Updated dataset with 'ucur' and 'vcur' fields added.
    """

    # Calculate bin depths
    ds = _calculate_bin_depths(ds)

    # Extract u, v, data
    ds["ucur"] = (
        ("BINS", "TIME", "SAMPLE"),
        ds.Average_VelEast.data,
        {
            "units": "m s-1",
            "long_name": "Eastward ocean velocity",
            "details": "All average mode samples",
        },
    )
    ds["vcur"] = (
        ("BINS", "TIME", "SAMPLE"),
        ds.Average_VelNorth.data,
        {
            "units": "m s-1",
            "long_name": "Northward ocean velocity",
            "details": "All average mode samples",
        },
    )

    for key in ["ucur", "vcur"]:
        ds[key].attrs["details"] = "All average mode samples"
        ds[key].attrs["units"] = "m s-1"
        ds[key].attrs[
            "processing_history"
        ] = "Loaded from Average_VelEast/Average_VelEast fields.\n"

    # Perform sound speed scaling if applicable
    if sspd_scaling:
        if true_sspd_name in ds:
            ds = _sspd_correction_ocean_vel(ds, true_sspd_name=true_sspd_name)
        else:
            print("Could not perform sound speed scaling of ice velocity "
                  f"(field '{true_sspd_name}' not found)")

    # Calculate sample averages
    ds = _calculate_uvcur_avg(ds, avg_method=avg_method)

    return ds



def _sspd_correction_ocean_vel(
        ds: xr.Dataset,
        true_sspd_name: str = 'sound_speed_CTD') -> xr.Dataset:
    """
    Apply sound speed correction to ocean velocity.

    This function scales the ocean current velocity fields ('ucur' and 'vcur') by
    the ratio of the true sound speed to the average sound speed in the
    dataset.

    Parameters:
    -----------
    ds : xarray.Dataset
        The dataset containing the velocity and sound speed data.
    true_sspd_name : str
        The name of the dataset field containing the true sound speed values.

    Returns:
    --------
    xarray.Dataset
        The dataset with corrected velocity fields.
    """
    # Calculate the sound speed correction factor
    sound_speed_factor = ds[true_sspd_name] / ds.Average_Soundspeed
    mean_sspd_factor = sound_speed_factor.mean()

    # Apply correction to 'uice' and 'vice'
    for uv_key in ['ucur', 'vcur']:
        if uv_key in ds:
            ds[uv_key] = ds[uv_key].copy() * sound_speed_factor

            proc_add = (
                'Amplitude scaled by improved sound speed estimate '
                f'(mean scaling factor {mean_sspd_factor:.4f}).'
            )
            ds[uv_key].attrs['processing_history'] = (
                ds[uv_key].attrs.get('processing_history', '') + proc_add)

    return ds



def _calculate_uvcur_avg(
    ds: xr.Dataset,
    avg_method: str = "median",
    min_good_pct: Optional[float] = None
) -> xr.Dataset:
    """
    Calculate ensemble average ocean velocity.

    Parameters:
    -----------
    ds : xarray.Dataset
        Input dataset containing ocean velocity components.
    avg_method : str, optional
        Method to calculate ensemble average ('median' or 'mean'). Default is
        'median'.
    min_good_pct : float, optional
        Minimum percentage of good samples required for a bin to be included in
        the average. Default is None.

    Returns:
    --------
    xarray.Dataset
        Updated dataset with 'UCUR' and 'VCUR' fields added.
    """

    if avg_method == "median":
        ds["UCUR"] = ds.ucur.median(dim="SAMPLE")
        ds["VCUR"] = ds.vcur.median(dim="SAMPLE")
    elif avg_method == "mean":
        ds["UCUR"] = ds.ucur.mean(dim="SAMPLE")
        ds["VCUR"] = ds.vcur.mean(dim="SAMPLE")
    else:
        raise Exception(
            f'Invalid "avg_method" ("{avg_method}"). '
            'Must be "mean" or "median".'
        )

    if min_good_pct:
        N_before = np.sum(~np.isnan(ds.UCUR))
        good_ind = (np.isnan(ds.ucur).mean(dim="SAMPLE") < 1
                    - min_good_pct / 100)
        N_after = np.sum(good_ind)
        min_good_str = (
            f"\nRejected {N_before - N_after} of {N_before} ensembles"
            f" ({(N_before - N_after) / N_before * 100:.2f}%) with "
            f"<{min_good_pct:.1f}% good samples."
        )
        ds["UCUR"] = ds.UCUR.where(good_ind)
        ds["VCUR"] = ds.VCUR.where(good_ind)

    else:
        min_good_str = ""

    ds.UCUR.attrs = {
        "units": "m s-1",
        "long_name": "Eastward ocean velocity",
        "details": "Ensemble average (%s)" % avg_method,
        "processing_history": ds.ucur.processing_history + min_good_str,
    }
    ds.VCUR.attrs = {
        "units": "m s-1",
        "long_name": "Northward ocean velocity",
        "details": "Ensemble average (%s)" % avg_method,
        "processing_history": ds.vcur.processing_history + min_good_str,
    }

    return ds


def _calculate_bin_depths(ds: xr.Dataset) -> xr.Dataset:
    """
    Calculate time-varying depth of ocean velocity bins.

    From Nortek documentation "Signature Principles of Operation":
        n-th cell is centered at a vertical distance from the transducer
        equal to: Center of n'th cell = Blanking + n * cell size

    Parameters:
    -----------
    ds : xarray.Dataset
        Input dataset containing ocean velocity fields and metadata.

    Returns:
    --------
    xarray.Dataset
        Updated dataset with 'bin_depth' field added.
    """

    dist_from_transducer = (
        ds.INSTRUMENT.blanking_distance_oceanvel
        + ds.INSTRUMENT.cell_size_oceanvel
        * (1 + np.arange(ds.INSTRUMENT.N_cells_oceanvel))
    )

    ds["bin_depth"] = (
        ds.depth.mean(dim="SAMPLE").expand_dims(
            dim={"BINS": ds.sizes["BINS"]})
        - dist_from_transducer[:, np.newaxis]
    )
    ds["bin_depth"].attrs = {
        "long_name": "Sample-average depth of velocity bins",
        "units": "m",
        "note": (
            "Calculated as:\n\n"
            "   bin_depth = instr_depth - (blanking depth + n*cell size)\n\n"
            "where *n* is bin number and *inst_depth* the (sample-mean) depth"
            " of the transducer."
        ),
    }

    return ds


def uvoc_mask_range(
    ds: xr.Dataset,
    uv_max: float = 1.5,
    tilt_max: float = 5,
    sspd_range: Tuple[float, float] = (1400, 1560),
    cor_min: float = 60,
    amp_range: Tuple[float, float] = (30, 85),
    max_amp_increase: float = 20,
) -> xr.Dataset:
    """
    Apply a series of masking criteria to ocean velocity components `ucur`
    and `vcur`.

    Masking criteria: 1. Speed exceeds `uv_max` (m/s). 2. Tilt exceeds
    `tilt_max` (degrees). 3. Instrument-recorded sound speed is outside
    `sspd_range` (m/s). 4. Any beam correlation is below `cor_min` (percent).
    5. Any beam amplitude is outside `amp_range` (dB). 6. There is a bin-to-bin
    amplitude jump greater than `max_amp_increase` (dB).

    Parameters:
    -----------
    ds : xr.Dataset
        The input dataset containing ocean velocity components and related
        data.
    uv_max : float, optional
        Maximum allowable ocean velocity magnitude (default is 1.5 m/s).
    tilt_max : float, optional
        Maximum allowable tilt (default is 5 degrees).
    sspd_range : Tuple[float, float], optional
        Acceptable range of sound speed (default is 1400 to 1560 m/s).
    cor_min : float, optional
        Minimum allowable beam correlation (default is 60%).
    amp_range : Tuple[float, float], optional
        Acceptable range of beam amplitudes (default is 30 to 85 dB).
    max_amp_increase : float, optional
        Maximum allowable bin-to-bin amplitude increase (default is 20 dB).

    Returns:
    --------
    xr.Dataset
        The dataset with `ucur` and `vcur` masked according to the
        specified criteria.
    """

    # N_variables used for counting the effect of each step.
    N_start = float(np.sum(~np.isnan(ds.ucur)).data)

    # Create ds_uv; a copy of ds containing only ucur, vcur.
    # Then feeding these back into ds before returning.
    # (This is because we dont want the ds.where() operation
    # to affect other fields/expand dimensions unrelated to ocean
    # velocities)
    ds_uv = ds[["ucur", "vcur"]]

    # Speed test
    ds_uv = ds_uv.where((ds.ucur**2 + ds.vcur**2) < uv_max**2)
    N_speed = float(np.sum(~np.isnan(ds_uv.ucur)).data)

    # Tilt test
    ds_uv = ds_uv.where(ds.tilt_Average < tilt_max)
    N_tilt = float(np.sum(~np.isnan(ds_uv.ucur)).data)

    # Sound speed test
    ds_uv = ds_uv.where(
        (ds.Average_Soundspeed > sspd_range[0])
        & (ds.Average_Soundspeed < sspd_range[1])
    )
    N_sspd = float(np.sum(~np.isnan(ds_uv.ucur)).data)

    # Correlation test
    if 'Average_CorBeam4' in ds:
        ds_uv = ds_uv.where(
            (ds.Average_CorBeam1 > cor_min)
            | (ds.Average_CorBeam2 > cor_min)
            | (ds.Average_CorBeam3 > cor_min)
            | (ds.Average_CorBeam4 > cor_min)
        )
    else:
        ds_uv = ds_uv.where(
            (ds.Average_CorBeam1 > cor_min)
            | (ds.Average_CorBeam2 > cor_min)
            | (ds.Average_CorBeam3 > cor_min)
        )
    N_cor = float(np.sum(~np.isnan(ds_uv.ucur)).data)

    # Amplitude test
    # Lower bound
    if 'Average_AmpBeam4' in ds:
        ds_uv = ds_uv.where(
            (ds.Average_AmpBeam1 > amp_range[0])
            | (ds.Average_AmpBeam2 > amp_range[0])
            | (ds.Average_AmpBeam3 > amp_range[0])
            | (ds.Average_AmpBeam4 > amp_range[0])
        )
    else:
        ds_uv = ds_uv.where(
            (ds.Average_AmpBeam1 > amp_range[0])
            | (ds.Average_AmpBeam2 > amp_range[0])
            | (ds.Average_AmpBeam3 > amp_range[0])
        )

    # Upper bound
    if 'Average_AmpBeam4' in ds:
        ds_uv = ds_uv.where(
            (ds.Average_AmpBeam1 < amp_range[1])
            | (ds.Average_AmpBeam2 < amp_range[1])
            | (ds.Average_AmpBeam3 < amp_range[1])
            | (ds.Average_AmpBeam4 < amp_range[1])
        )
    else:
        ds_uv = ds_uv.where(
            (ds.Average_AmpBeam1 < amp_range[1])
            | (ds.Average_AmpBeam2 < amp_range[1])
            | (ds.Average_AmpBeam3 < amp_range[1])
        )

    N_amp = float(np.sum(~np.isnan(ds_uv.ucur)).data)

    # Amplitude bump test
    # Find bumps from *diff* in the BIN S dimension
    if 'Average_AmpBeam4' in ds:
        is_bump = (
            (ds.Average_AmpBeam1.diff(dim="BINS") > max_amp_increase)
            | (ds.Average_AmpBeam2.diff(dim="BINS") > max_amp_increase)
            | (ds.Average_AmpBeam3.diff(dim="BINS") > max_amp_increase)
            | (ds.Average_AmpBeam4.diff(dim="BINS") > max_amp_increase)
        )
    else:
        is_bump = (
            (ds.Average_AmpBeam1.diff(dim="BINS") > max_amp_increase)
            | (ds.Average_AmpBeam2.diff(dim="BINS") > max_amp_increase)
            | (ds.Average_AmpBeam3.diff(dim="BINS") > max_amp_increase)
        )

    # Create a boolean (*True* above bumps)
    zeros_firstbin = xr.zeros_like(ds.ucur.isel(BINS=0))
    NOT_ABOVE_BUMP = (
        xr.concat([zeros_firstbin, is_bump.cumsum(axis=0) > 0],
                dim=("BINS")) < 1
    )
    ds_uv = ds_uv.where(NOT_ABOVE_BUMP)
    N_amp_bump = float(np.sum(~np.isnan(ds_uv.ucur)).data)

    proc_string = (
        f"\nTHRESHOLD-BASED DATA CLEANING : "
        f"\nStart: {N_start} initial valid samples.\n"
        f"Dropping (NaNing samples where):\n"
        f"- # Speed < {uv_max:.2f} ms-1 # -> Dropped {N_start - N_speed} pts "
        f"({(N_start - N_speed) / N_start * 100:.2f}%%)\n"
        f"- # Tilt < {tilt_max:.2f} deg # -> Dropped {N_speed - N_tilt} pts "
        f"({(N_speed - N_tilt) / N_speed * 100:.2f}%%)\n"
        f"- # Sound sp in [{sspd_range[0]:.0f}, {sspd_range[1]:.0f}] ms-1 # ->"
        f" Dropped {N_tilt - N_sspd} pts "
        f"({(N_tilt - N_sspd) / N_tilt * 100:.2f}%%)\n"
        f"- # Corr (all beams) < {cor_min:.1f} %% # -> Dropped "
        f"{N_sspd - N_cor} pts ({(N_sspd - N_cor) / N_sspd * 100:.2f}%%)\n"
        f"- # Amp (all beams) in [{amp_range[0]:.0f}, {amp_range[1]:.0f}] db "
        f"# -> Dropped {N_cor - N_amp} pts "
        f"({(N_cor - N_amp) / N_cor * 100:.2f}%%)\n"
        f"- # Above amp bumps > {max_amp_increase:.0f} db # -> Dropped "
        f"{N_amp - N_amp_bump} pts "
        f"({(N_amp - N_amp_bump) / N_amp * 100:.2f}%%)\n"
        f"End: {N_amp_bump} valid samples.\n"
    )

    for key in ["ucur", "vcur"]:
        ds[key] = ds_uv[key]
        ds[key].attrs["processing_history"] += proc_string

    # Recompute sample averages
    ds = _calculate_uvcur_avg(ds)

    return ds


def rotate_vels_magdec(ds: xr.Dataset) -> xr.Dataset:
    """
    Rotate ocean and ice velocities to account for magnetic declination.

    This function rotates the processed fields (`ucur`, `vcur`, `uice`,
    `vice`) by the magnetic declination angle to correct for geographic
    referencing. It does not modify raw variables (e.g., `Average_VelNorth`).

    Average velocities (`UCUR`, `VCUR`, `UICE`,`VICE`) are recalculated
    after rotating.

    Parameters:
    -----------
    ds : xr.Dataset
        The input dataset containing ocean and ice velocity components.

    Returns:
    --------
    xr.Dataset
        The dataset with rotated velocity components to account for magnetic
        declination.
    """

    assert hasattr(ds, "magdec"), (
        "Didn't find magnetic declination (no"
        " *magdec* attribute). Run sig_append.append_magdec().."
    )

    ds0 = ds.copy()

    # Convert to radians
    magdec_rad = ds.magdec * np.pi / 180

    # Make a documentation string (to add as attribute)
    magdec_mean = ds.magdec.mean().data
    magdec_str = (
        "Rotated CW by an average of %.2f degrees" % magdec_mean
        + " to correct for magnetic declination. "
    )

    # Loop through different (u, v) variable pairs and rotate them
    uvpairs = [
        ("uice", "vice"),
        ("ucur", "vcur"),
    ]  #
    #  ('UICE', 'VICE'), ('UCUR', 'VCUR')]

    uvstrs = ""

    for uvpair in uvpairs:

        if hasattr(ds, uvpair[0]) and hasattr(ds, uvpair[1]):
            uvc0_ = ds[uvpair[0]] + 1j * ds[uvpair[1]]
            uvc_rot = uvc0_ * np.exp(-1j * magdec_rad)

            ds[uvpair[0]].data = uvc_rot.real
            ds[uvpair[1]].data = uvc_rot.imag
            for key in uvpair:
                ds[key].attrs["processing_history"] += magdec_str + "\n"
            uvstrs += "\n - (%s, %s)" % uvpair

    if hasattr(ds.INSTRUMENT, "declination_correction"):
        inp_yn = float(
            input(
                "Declination correction rotation has been "
                + "applied to something before. \n -> Continue "
                + "(1) or skip new correction (0): "
            )
        )

        if inp_yn == 1:
            print("-> Applying new correction.")
            ds.INSTRUMENT.attrs["declination_correction"] = (
                "!! NOTE !! Magnetic declination correction has been applied "
                "more than once - !! CAREFUL !!\n"
                + ds.INSTRUMENT.attrs["declination_correction"]
            )
        else:
            print("-> NOT applying new correction.")
            return ds0
    else:
        ds.INSTRUMENT.attrs["declination_correction"] = (
            "Magdec declination correction rotation applied to: %s" % uvstrs
        )
    try:
        ds = _calculate_uvcur_avg(ds, avg_method="median")
    except Exception as e:
        print(f"uvcur_avg failed: {e}")
    try:
        ds = _calculate_uvice_avg(ds, avg_method="median")
    except Exception as e:
        print(f"uvice_avg failed: {e}")

    return ds


def clear_empty_bins(ds: xr.Dataset, thr_perc: float = 5) -> xr.Dataset:
    """
    Remove bins where less than `thr_perc` percent of the samples contain valid
    (non-NaN) data.

    This function examines each bin along the `BINS` dimension and removes bins
    that have a high proportion of missing data (NaNs). The threshold
    percentage (`thr_perc`) defines the minimum percentage of valid samples
    required to retain a bin.

    Parameters:
    -----------
    ds : xr.Dataset
        The input dataset containing the `BINS` dimension and data with
        potential NaNs.
    thr_perc : float, optional
        The threshold percentage of valid (non-NaN) data required to keep a
        bin. Default is 5%.

    Returns:
    --------
    xr.Dataset
        The dataset with bins removed where the percentage of valid data is
        below `thr_perc`.
    """

    # Find indices of empty bins
    empty_bins = np.where(
        np.isnan(ds.UCUR).mean("TIME")
        * 100 > (100 - thr_perc))[0]
    # Count
    Nbins_orig = ds.sizes["BINS"]
    Nbins_drop = len(empty_bins)

    # Drop from dataset
    ds = ds.drop_sel(BINS=empty_bins)
    # Note in history
    ds.attrs["history"] += (
        "\nDropped %i of %i bins where" % (Nbins_drop, Nbins_orig)
        + " less than %.1f%% of samples were" % (thr_perc)
        + "  valid. -> Remaining bins: %i" % (ds.sizes["BINS"])
    )
    return ds


def sound_speed_correction(
        ds:xr.Dataset,
        true_sspd_name:str='sound_speed_CTD'
        ) -> xr.Dataset:
    '''

    '''
    sound_speed_factor = ds[true_sspd_name] / ds.AverageIce_Soundspeed
    sound_speed_factor_ice = ds[true_sspd_name] / ds.Average_Soundspeed
    mean_sspd_factor = sound_speed_factor.mean()
    mean_sspd_factor_ice = sound_speed_factor_ice.mean()

    for uv_key in ['uice', 'vice', 'UICE', 'VICE',]:
        if uv_key in ds:
            ds[uv_key] *= sound_speed_factor_ice

            comment_add = (
                'Amplitude scaled to improved sound speed estimate (mean '
                f'factor {mean_sspd_factor_ice:.4f}).')

            ds[uv_key].attrs['comment'] = (
                ds[uv_key].attrs.get('comment', '') + comment_add)

    for uv_key in ['UCUR', 'VCUR', 'ucur', 'vcur']:
        if uv_key in ds:
            ds[uv_key] *= sound_speed_factor

            comment_add = (
                'Amplitude scaled to improved sound speed estimate (mean '
                f'factor {mean_sspd_factor:.4f}).')

            ds[uv_key].attrs['comment'] = (
                ds[uv_key].attrs.get('comment', '') + comment_add)

    return ds



def reject_sidelobe(ds: xr.Dataset) -> xr.Dataset:
    """
    Mask samples where we expect sidelobe interference based on the calculated
    maximum range (Rmax).

    From Nortek documentation
    (nortekgroup.com/assets/software/N3015-011-SignaturePrinciples.pdf):

    Maximum range Rmax is given by:

    (1)    Rmax = A * cos(θ) - s_c

    Where:
        - A is the distance between the transducer and the surface.
        - θ (theta) is the beam angle, assumed to be 25 degrees for Nortek
          Signatures.
        - s_c is the velocity cell size.

    The distance A is calculated as:

    (2)    A = DEP - ICE_DRAFT

    Where:
        - DEP is the sample-mean depth of the water column.
        - ICE_DRAFT is the sample-mean sea ice draft (using
          SEA_ICE_DRAFT_MEDIAN_LE if available, otherwise 0).

    We use the sample mean tilt for θ.

    This function identifies and masks (sets to NaN) the velocity samples
    (`ucur`, `vcur`) in regions close enough to the surface where sidelobe
     interference is expected, based on the calculated Rmax.

    Parameters:
    -----------
    ds : xr.Dataset
        The input dataset containing ocean velocity components (`ucur`,
        `vcur`), depth, bin depth, and possibly sea ice draft information.

    Returns:
    --------
    xr.Dataset
        The dataset with sidelobe-affected velocity samples masked
        (set to NaN).
    """

    if "depth" in ds.keys():
        DEP = ds.depth.mean(dim="SAMPLE")
    else:
        raise Exception(
            'No "depth" field present. -> cannot reject '
            "measurements influenced by sidelobe interference. "
            "Run sig_calc.dep_from_p() first."
        )

    if "SEA_ICE_DRAFT_LE" in ds.keys():
        ICE_DRAFT = ds.SEA_ICE_DRAFT_MEDIAN_LE.copy().fillna(0)

    else:
        ICE_DRAFT = 0

    A = DEP - ICE_DRAFT
    cos_theta = np.cos(25 * np.pi / 180.0)
    s_c = ds.cell_size_oceanvel

    Rmax = A * cos_theta - s_c

    # Make a copy with only velocities
    ds_uv = ds[["ucur", "vcur"]]
    # NaN instances where bin depth is less than
    #   DEP-Rmax
    ds_uv = ds_uv.where(
        ds.bin_depth > (DEP - Rmax).expand_dims(dim={"BINS": ds.sizes["BINS"]})
    )

    # Feed the NaNed (ucur, vcur) fields back into ds

    N_before = np.sum(~np.isnan(ds.ucur))  # Count samples
    for key in ["ucur", "vcur"]:
        ds[key] = ds_uv[key]
    N_after = np.sum(~np.isnan(ds.ucur))  # Count samples

    # + Add processing history
    proc_string = (
        f"\nRejected samples close enough to the surface "
        f"to be affected by sidelobe interference (rejecting "
        f"{(1 - N_after / N_before) * 100:.2f}%% of velocity samples)."
    )

    for key in ["ucur", "vcur"]:
        ds[key].attrs["processing_history"] += proc_string

    # Recompute sample averages
    ds = _calculate_uvcur_avg(ds)
    return ds


def interp_oceanvel(ds: xr.Dataset, target_depth: float) -> xr.Dataset:
    """
    Interpolate ocean velocity components (`ucur`, `vcur`) to a fixed depth
    (`target_depth`).

    This function interpolates the ocean velocity components from the measured
    bin depths to a specified target depth. The interpolated velocities are
    added to the dataset with names indicating the target depth.

    Parameters:
    -----------
    ds : xr.Dataset
        The input dataset containing the ocean velocity components (`ucur`,
        `vcur`) and their associated depths (`bin_depth`) over time.
    target_depth : float
        The depth (in meters) to which the velocities should be interpolated.

    Returns:
    --------
    xr.Dataset
        The dataset with added velocity components interpolated to the
        specified depth. The new variables are named `UCUR_<depth>m` and
        `VCUR_<depth>m`.
    """
    # Initialize the interpolated velocity arrays
    U_IP = ds.UCUR.mean("BINS", keep_attrs=True).copy()
    V_IP = ds.VCUR.mean("BINS", keep_attrs=True).copy()
    U_IP[:] = np.nan
    V_IP[:] = np.nan

    # Interpolate UCUR onto the fixed depth
    for nn in range(ds.sizes["TIME"]):
        ip_ = interp1d(
            ds.bin_depth.isel(TIME=nn),
            ds.UCUR.isel(TIME=nn),
            bounds_error=False,
            fill_value=np.nan,
        )
        U_IP[nn] = ip_(target_depth)

        if nn % 10 == 0:
            print(
                'Interpolating "UCUR" ('
                f'{100 * nn / ds.sizes["TIME"]:.1f}%%)...\r',
                end="",
            )

    print('Interpolating "UCUR": *DONE*     \r', end="")

    # Interpolate VCUR onto the fixed depth
    for nn in range(ds.sizes["TIME"]):
        ip_ = interp1d(
            ds.bin_depth.isel(TIME=nn),
            ds.VCUR.isel(TIME=nn),
            bounds_error=False,
            fill_value=np.nan,
        )
        V_IP[nn] = ip_(target_depth)

        if nn % 10 == 0:
            print(
                'Interpolating "VCUR" '
                f'({100 * nn / ds.sizes["TIME"]:.1f}%%)...\r',
                end="",
            )

    print('Interpolating "VCUR": *DONE*     \r', end="")

    # Create variable names based on the target depth
    U_IP_name = f"UCUR_{int(np.round(target_depth))}m"
    V_IP_name = f"VCUR_{int(np.round(target_depth))}m"

    # Add interpolated velocities to the dataset with appropriate attributes
    ds[U_IP_name] = U_IP
    ds[U_IP_name].attrs["long_name"] = (
        f"{ds.UCUR.attrs['long_name']} "
        f"interpolated to {target_depth:.1f} m depth"
    )
    ds[U_IP_name].attrs["processing_history"] = (
        f"{ds.UCUR.attrs['processing_history']} "
        f"\nInterpolated to {target_depth:.1f} m depth."
    )

    ds[V_IP_name] = V_IP
    ds[V_IP_name].attrs["long_name"] = (
        f"{ds.VCUR.attrs['long_name']} "
        f"interpolated to {target_depth:.1f} m depth"
    )
    ds[V_IP_name].attrs["processing_history"] = (
        f"{ds.VCUR.attrs['processing_history']} "
        f"\nInterpolated to {target_depth:.1f} m depth."
    )

    print(f"Added interpolated velocities: ({U_IP_name}, {V_IP_name})")
    return ds
