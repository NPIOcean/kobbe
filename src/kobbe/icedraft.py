"""
ICEDRAFT.PY

Functions for calculating sea ice draft
"""

import numpy as np
from scipy.interpolate import interp1d
from matplotlib import pyplot as plt
from kobbe.calc import runningstat, daily_average, clean_nanmedian
from kobbe import append


def calculate_draft(
    ds, corr_sound_speed_CTD=True, qual_thr=8000,
    LE_AST_max_sep=0.5,
    minimum_draft = -0.5,
):
    """
    Calculate ice draft.


    If LE_correction = 'AST', the open water sound speed correction (if
    available) of the LE-derived draft will be based on the AST open water
    offset.
    """

    # Get surface position (LE and AST)
    for le_ast in ["LE", "AST"]:

        ds = calculate_surface_position(
            ds,
            le_ast=le_ast,
            corr_sound_speed_CTD=corr_sound_speed_CTD,
            qual_thr=qual_thr,
        )

    # Reject LE measurements where LE diverges from AST by >LE_AST_max_sep
    if LE_AST_max_sep:
        condition = (
            np.abs(ds.SURFACE_DEPTH_LE - ds.SURFACE_DEPTH_AST)
            < LE_AST_max_sep)

        _N_LE_before = (~np.isnan(ds["SURFACE_DEPTH_LE"])).sum().item()
        ds["SURFACE_DEPTH_LE"] = ds["SURFACE_DEPTH_LE"].where(condition)
        _N_LE_after = (~np.isnan(ds["SURFACE_DEPTH_LE"])).sum().item()
        _pct_reduction = 100 * (_N_LE_before - _N_LE_after) / _N_LE_before

        ds["SURFACE_DEPTH_LE"].attrs["note"] += (
            "\n- Rejected LE measurements where LE and AST length estimates "
            f"diverged by >{LE_AST_max_sep} m, reducing valid LE measurements"
            f" by {_pct_reduction:.1f}% from {_N_LE_before} to {_N_LE_after}."
        )

    # Get sea ice draft (based on ice(no ice criteria))
    # Calculate ensemble medians
    for le_ast in ["LE", "AST"]:

        si_draft_ = ds["SURFACE_DEPTH_%s" % le_ast].data.copy()
        si_draft_[~ds.ICE_IN_SAMPLE.data] = np.nan

        ds["SEA_ICE_DRAFT_%s" % le_ast] = (
            ("TIME", "SAMPLE"),
            si_draft_,
            {
                "long_name": "Sea ice draft at each sample (%s)" % le_ast,
                "units": "m",
                "note": ds["SURFACE_DEPTH_%s" % le_ast].note
                + "\n\nSet to NaN where ICE_IN_SAMPLE==False",
            },
        )

        ds["SEA_ICE_DRAFT_%s" % le_ast] = ds["SEA_ICE_DRAFT_%s" % le_ast].where(
            ds["SEA_ICE_DRAFT_%s" % le_ast] >= minimum_draft
        )

        ds["SEA_ICE_DRAFT_MEDIAN_%s" % le_ast] = (
            ("TIME"),
            clean_nanmedian(si_draft_, axis=1),
            {
                "long_name": "Median sea ice draft of each ensemble (%s)" % le_ast,
                "units": "m",
                "note": ds["SURFACE_DEPTH_%s" % le_ast].note
                + "\n\nOnly counting instances with sea ice presence.",
            },
        )

        ds["SEA_ICE_DRAFT_MEDIAN_%s" % le_ast] = ds[
            "SEA_ICE_DRAFT_MEDIAN_%s" % le_ast
        ].where(ds["SEA_ICE_DRAFT_MEDIAN_%s" % le_ast] >= minimum_draft)

    return ds


def calculate_surface_position(
    ds, corr_sound_speed_CTD=True, qual_thr=8000, le_ast="AST",
):
    """
    Calculate distance between the surface measured by the altimeter
    and the (mean) ocean surface.
    """

    # Determine which source variables to use (LE or AST)
    le_ast = le_ast.upper()
    if le_ast == "AST":
        alt_dist_attr = "Average_AltimeterDistanceAST"
        alt_qual_attr = "Average_AltimeterQualityAST"
    else:
        alt_dist_attr = "Average_AltimeterDistanceLE"
        alt_qual_attr = "Average_AltimeterQualityLE"

    # Start a metadata note
    note_str = (
        "From %s altimeter distances."
        "\n\nComputed with the function "
        "kobbe.icedraft.calculate_surface_position()." % le_ast
    )

    # Obtain tilt factor
    tilt_factor = np.cos(np.pi * ds.tilt_Average / 180)

    note_str += (
        "\n- Altimeter distance adjusted for instrument tilt."
    )

    # Obtain factor of "true" sound speed (from CTD data) vs nominal
    # sound speed (from the Average_Soundspeed field)
    if hasattr(ds, "sound_speed_CTD") and corr_sound_speed_CTD:
        # Ratio between observed and nominal sound speed
        sound_speed_ratio_obs_nom = (
            ds.sound_speed_CTD.data[:, np.newaxis]
            / ds.Average_Soundspeed.data)

        note_str += (
            "\n- Altimeter distance recomputed using updated "
            "sound speed (*sound_speed_CTD* field)"
        )
    # Set ratio to 1 if we do not have CTD sound speed.
    else:
        sound_speed_ratio_obs_nom = 1
        note_str += (
            "\n- (No correction made for CTD sound speed - using Signature "
            "sound speed estimate)."
        )

    # Define beta_LE/_AST (the open water correction factor)
    # to be applied to the data
    alpha_key = "alpha_%s" % le_ast
    beta_key = "beta_%s" % le_ast

    # Set alpha (set to zero if we don't have one)
    if alpha_key in ds:
        alpha = ds[alpha_key].item()
        note_str += (
            "\n- Applying an open water correction fixed offset alpha to "
            "the altimeter distance."
        )
    else:
        alpha = 0
        note_str += (
            "\n- (No fixed offset alpha applied to the altimeter distance)."
        )

    # Wrap out beta_ to the full 2D shape so we can apply it below
    if beta_key in ds:
        beta = ds[beta_key].data[:, np.newaxis] * np.ones(ds.depth.shape)
        note_str += (
            "\n- Applying an open water correction factor beta to "
            f"altimeter distance ({beta_key} field)."
        )
        beta_mean = np.round(np.nanmean(beta).item(), 4)
        beta_max = np.round(np.nanmax(beta).item(), 4)
        beta_min = np.round(np.nanmin(beta).item(), 4)

    # ..or set it to 1 if we don't have a beta
    else:
        beta = 1
        note_str += (
            "\n- (No open water corrective factor beta applied to the "
            "altimeter distance.)"
        )
        beta_mean, beta_max, beta_min = 'N/A', 'N/A', 'N/A'
    # Calculate the surface position (depth of the scattering surface detected
    # by LE or AST algorithm below the water surface)
    surface_position = (
        ds.depth
        - ds[alt_dist_attr]
        * tilt_factor
        * sound_speed_ratio_obs_nom
        * beta
        - alpha
    )

    # Apply a quality threshold criterion (from the Average_AltimeterQuality
    # variable)
    surface_position = surface_position.where(ds[alt_qual_attr] > qual_thr)
    note_str += (
        "\n- Samples where %s>%i" % (alt_qual_attr, qual_thr) + " were discarded."
    )

    # STORE AS VARIABLE
    ds["SURFACE_DEPTH_%s" % le_ast] = (
        ("TIME", "SAMPLE"),
        surface_position.data,
        {
            "long_name": (
                "Depth of the scattering surface observed by the altimeter "
                f"({le_ast})"
            ),
            "units": "m",
            "note": note_str,

            'tilt_correction_mean': np.round(tilt_factor.mean().item(), 4),
            'sound_speed_ratio_mean': np.round(
                sound_speed_ratio_obs_nom.mean().item(), 4),
            'fixed_offset_alpha_cm': np.round(alpha*1e2, 4),
            'varying_ss_factor_beta_mean': beta_mean,
            'varying_ss_factor_beta_max': beta_max,
            'varying_ss_factor_beta_min': beta_min,
        },
    )

    return ds


def get_open_water_surface_depth(ds, method="LE"):
    """
    Get the surface depth during open water periods only.

    Returns DataArray containing the same variable as the input -
    but with ice entries masked.
    """
    open_water_surface_depth = (
        ds["SURFACE_DEPTH_%s" % method].where(ds.ICE_IN_SAMPLE_ANY==False))
    return open_water_surface_depth


def get_open_water_surface_depth_LP(
    open_water_surface_depth,
    thr_reject_from_net_median=0.15,
    min_frac_daily=0.025,
    run_window_days=3,
):
    """
    Compute an estimate of the long-time averaged surface depth in open water
    (open water surface depth, OWSD).

    In an ideal case, OWSD should be equal to zero.

    Steps:
    1. Reject instances where OWSD deviates from the OWSD deployment median by
       more than *thr_reject_from_net_median* (meters, default = 0.15).
    2. Compute ensemble median values of the OWSD resulting from (1).
    3. Compute daily medians of the ensemble means in (2).
       Reject days where less than *min_frac_daily* (default = 0.025) of the
       ensembles contain open-water samples.
    4. Linearly interpolate between missing daily values to get a continuous
       daily time series.
    5. Smoothe this daily time series with a running mean of window length
       *run_window_days* (default=3).
    """

    # 1. Compute initial median and reject values away from the median
    #    by *thr_reject_from_netmedian* [m]
    OWSD_full_median = clean_nanmedian(open_water_surface_depth)

    OWSD_filt = open_water_surface_depth.where(
        np.abs(open_water_surface_depth - OWSD_full_median)
        < thr_reject_from_net_median)

    # 2. Compute ensemble medians
    OWSD_med = OWSD_filt.median(dim="SAMPLE")

    # 3. Compute daily medians ()
    Ad, td = daily_average(
        OWSD_med,
        open_water_surface_depth.TIME,
        min_frac=min_frac_daily,
        axis=-1,
        function="median"
    )

    # 4. Interpolate to continuous function (daily)
    Ad_interp = interp1d(td.data[~np.isnan(Ad)], Ad[~np.isnan(Ad)],
                         bounds_error=False)(
        td.data
    )

    # 5. Smooth with running mean
    RS = runningstat(Ad_interp, run_window_days)

    # Export filtered, ensemble median, daily averaged, smoothed daily OWSD.
    # Also daily time array (td+0.5) of the midpoint of the daily estimates.
    return RS["mean"], td + 0.5

###########


def get_open_water_correction(
    ds,
    fixed_offset=True,
    ss_factor=True,
    thr_reject_from_net_median=0.15,
    min_frac_daily=0.025,
    run_window_days=3,
):
    """ """

    # Obtain (all) estimates Open Water Surface Depths
    ow_surface_depth_full_LE = get_open_water_surface_depth(ds, method="LE")
    ow_surface_depth_full_AST = get_open_water_surface_depth(ds, method="AST")

    # Obtain estimates of daily, smoothed Open Water Surface Depths
    ow_surface_depth_LP_LE, _ = get_open_water_surface_depth_LP(
        ow_surface_depth_full_LE,
        thr_reject_from_net_median=thr_reject_from_net_median,
        min_frac_daily=min_frac_daily,
        run_window_days=run_window_days
    )
    ow_surface_depth_LP_AST, td = get_open_water_surface_depth_LP(
        ow_surface_depth_full_AST,
        thr_reject_from_net_median=thr_reject_from_net_median,
        min_frac_daily=min_frac_daily,
        run_window_days=run_window_days
    )

    # Obtain daily, smoothed instrument depths
    depth_med = ds.depth.median(dim="SAMPLE")
    depth_med_daily, _ = daily_average(
        depth_med, ds.TIME, td=td - 0.5, axis=-1, function="median"
    )
    RS_depth = runningstat(depth_med_daily, run_window_days)
    depth_lp = RS_depth["mean"]  # <--

    # Obtain ALPHA (fixed offset)
    if fixed_offset:
        alpha_LE = np.round(clean_nanmedian(ow_surface_depth_LP_LE), 4)
        alpha_AST = np.round(clean_nanmedian(ow_surface_depth_LP_AST), 4)
    else:
        alpha_LE, alpha_AST = 0, 0

    # Obtain BETA (time-varying sound speed correction)
    if ss_factor:
        beta_LE = ((depth_lp - alpha_LE)
                   / (depth_lp - ow_surface_depth_LP_LE))
        beta_AST = ((depth_lp - alpha_AST)
                    / (depth_lp - ow_surface_depth_LP_AST))
    else:
        beta_LE, beta_AST = np.ones(len(td)), np.ones(len(td))

    # Append alpha and beta to the dataset

    ds["alpha_LE"] = ((), alpha_LE)
    ds["alpha_AST"] = ((), alpha_AST)

    ds = append.add_to_sigdata(ds, beta_LE, td, "beta_LE")
    ds = append.add_to_sigdata(ds, beta_AST, td, "beta_AST")

    # Append the open water estimates as well
    ds = append.add_to_sigdata(
        ds, ow_surface_depth_LP_LE, td, "ow_surface_before_correction_LE_LP"
    )
    ds["ow_surface_before_correction_LE"] = (
        ow_surface_depth_full_LE.median(dim="SAMPLE"))

    ds = append.add_to_sigdata(
        ds, ow_surface_depth_LP_AST, td, "ow_surface_before_correction_AST_LP"
    )
    ds["ow_surface_before_correction_AST"] = (
        ow_surface_depth_full_AST.median(dim="SAMPLE"))

    return ds


def compare_open_water_correction(ds, show_plots=True):
    """
    Note: Run this *after* running *get_Beta_from_OWSD* but *before*
    running *kobbe.icedraft.calculate_draft()* again.
    """

    ds0 = ds.copy()
    ds2 = ds.copy()
    ds2 = calculate_draft(ds2)

    print(
        "LE: Mean (median) offset: %.1f cm (%.1f cm)"
        % (
            ds.ow_surface_before_correction_LE.mean() * 1e2,
            clean_nanmedian(ds.ow_surface_before_correction_LE) * 1e2,
        )
    )

    print(
        "AST: Mean (median) offset: %.1f cm (%.1f cm)"
        % (
            ds.ow_surface_before_correction_AST.mean() * 1e2,
            clean_nanmedian(ds.ow_surface_before_correction_AST) * 1e2,
        )
    )

    print(f"LE: Applied offset alpha: {ds.alpha_LE*1e2:.1f} cm")
    print(
        "LE: Applied time-varying sound speed factor beta: Mean (median)"
        f": {ds.beta_LE.mean():.5f} ({clean_nanmedian(ds.beta_LE):.5f})"
    )

    print(f"AST: Applied offset alpha: {ds.alpha_AST*1e2:.1f} cm")
    print(
        "AST: Applied time-varying sound speed factor beta: Mean (median)"
        f": {ds.beta_AST.mean():.5f} ({clean_nanmedian(ds.beta_AST):.5f})"
    )

    print(
        "LE - MEAN SEA ICE DRAFT:\n"
        f"Before correction: {ds0.SEA_ICE_DRAFT_MEDIAN_LE.mean():.2f} m"
        f"\nAfter correction: {ds2.SEA_ICE_DRAFT_MEDIAN_LE.mean():.2f} m"
    )

    print(
        "AST - MEAN SEA ICE DRAFT:\n"
        f"Before correction: {ds0.SEA_ICE_DRAFT_MEDIAN_AST.mean():.2f} m"
        f"\nAfter correction: {ds2.SEA_ICE_DRAFT_MEDIAN_AST.mean():.2f} m"
    )

    # Figures
    if show_plots:

        #

        fig, ax = plt.subplots(2, 1, figsize=(12, 6), sharex=True)

        ax[0].plot_date(ds2.TIME,
                        ds.ow_surface_before_correction_LE,
                        ".", label="LE", color='tab:blue')
        ax[0].plot_date(ds2.TIME,
                        ds.ow_surface_before_correction_LE_LP,
                        "-", label="LE (LP filtered)",
                        color='tab:blue')
        ax[0].axhline(ds.alpha_LE, ls = ':', color='tab:blue', label='LE Fixed offset $\\alpha$')

        ax[0].plot_date(ds2.TIME,
                        ds.ow_surface_before_correction_AST,
                        ".", label="AST", color='tab:orange')
        ax[0].plot_date(ds2.TIME,
                        ds.ow_surface_before_correction_AST_LP,
                        "-", label="AST (LP filtered)",
                        color='tab:orange')

        ax[0].axhline(ds.alpha_AST, ls = ':', color='tab:orange', label='AST Fixed offset $\\alpha$')

        ax[1].plot_date(ds2.TIME, ds.beta_LE, "-", label="LE")
        ax[1].plot_date(ds2.TIME, ds.beta_AST, "-", label="AST")

        for axn in ax:
            axn.legend(ncol = 2)
            axn.grid()
        labfs = 9
        ax[0].set_ylabel("Estimated open water\nsurface depth [m]",
                         fontsize=labfs)
        ax[1].set_ylabel("$\\beta$ (open water altimeter \ndistance"
                         " correction factor)", fontsize=labfs)
        ax[0].set_ylim(1, -0.4)

        #

        fig, ax = plt.subplots(1, 2, figsize=(12, 6), sharex=True, sharey=True)
        ax[0].scatter(
            ds0.time_average,
            ds0.SURFACE_DEPTH_LE,
            marker=".",
            color="k",
            alpha=0.2,
            s=2,
            label="Uncorrected",
        )
        ax[0].scatter(
            ds.time_average,
            ds2.SURFACE_DEPTH_LE,
            marker=".",
            color="r",
            alpha=0.2,
            s=2,
            label="Corrected",
        )

        ax[1].scatter(
            ds0.TIME,
            ds0.SEA_ICE_DRAFT_MEDIAN_LE,
            marker=".",
            color="k",
            alpha=0.3,
            s=2,
            label="Uncorrected",
        )
        ax[1].scatter(
            ds.TIME,
            ds2.SEA_ICE_DRAFT_MEDIAN_LE,
            marker=".",
            color="r",
            alpha=0.3,
            s=2,
            label="Corrected",
        )
        ax[0].set_title("LE Surface depth (ALL)")
        ax[1].set_title("LE sea ice draft (ice only, ensemble averaged)")

        for axn in ax:
            axn.legend()
            axn.grid()
            axn.set_ylabel("[m]")

        labfs = 9
        ax[0].set_ylabel("Estimated open water\nsurface depth [m]", fontsize=labfs)
        ax[1].set_ylabel("BETA (OWSD correction factor)", fontsize=labfs)

        # Dummy for date axis..
        ax[0].plot_date(ds.time_average[0, 0], ds2.SURFACE_DEPTH_LE[0, 0])

        ax[0].invert_yaxis()

        plt.show()
