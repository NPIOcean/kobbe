"""
Various calculations done on a xarray Dataset with Signature data.
"""

import numpy as np
import gsw
import warnings
import xarray as xr
from typing import Optional, Tuple, Dict, Any


def dep_from_p(
    ds: xr.Dataset,
    corr_atmo: bool = True,
    corr_CTD_density: bool = True
) -> xr.Dataset:
    """
    Calculate depth from absolute pressure in a Signature xarray Dataset.

    Computing depth based on the following:
    - Absolute pressure measured by the instrument (from
      `Average_AltimeterPressure` plus the fixed atmospheric offset which is
      automaically subtracted from this field).
    - Atmospheric pressure (from `p_atmo` field, if available).
    - Gravitational acceleration (calculated from latitude).
    - Ocean density (from `rho_CTD` field, if available, or default to 1027
      kg/m³).

    Args:
        ds (xr.Dataset): xarray Dataset containing Signature data.
        corr_atmo (bool, optional): If True, correct for atmospheric pressure
                                    using the `p_atmo` field. Defaults to True.
        corr_CTD_density (bool, optional): If True, use CTD-derived density
                                           from the `rho_CTD` field.
                                           Defaults to True.

    Returns:
        xr.Dataset: The input Dataset with the "depth" field (TIME, SAMPLE)
                    added.
    """

    note_str = (
        "Altimeter depth calculated from pressure"
        " (*Average_AltimeterPressure* field) as:\n\n "
        "   depth = p / (g * rho)\n"
    )

    # CALCULATE ABSOLUTE PRESSURE
    p_abs = ds.Average_AltimeterPressure + ds.pressure_offset

    # CALCULATE OCEAN PRESSURE
    # Raising issues if we cannot find p_atmo (predefined atmospheric pressure)
    if hasattr(ds, "p_atmo") and corr_atmo:
        p_ocean = (p_abs - ds.p_atmo).data
        note_str += "\n- Atmospheric pressure (*p_atmo* field subtracted)."
    else:
        if corr_atmo:
            warn_str1 = (
                "WARNING!\nCould not find atmospheric pressure "
                "(*p_atmo*) - not recommended continue if you plan to compute "
                "ice draft. \n--> (To add *p_atmo*, run sig_append.append_atm_"
                "pres()=\n\nDepth calculation: Abort (A) or Continue (C): "
            )

            user_input_abort = input(warn_str1).upper()

            while user_input_abort not in ["A", "C"]:
                print('Input ("%s") not recognized.' % user_input_abort)
                user_input_abort = input(
                    'Enter "C" (continue) or "A" (abort): '
                ).upper()

            if user_input_abort == "A":
                raise Exception(
                    "ABORTED BY USER (MISSING ATMOSPHERIC PRESSURE)")

        else:
            user_input_abort = "C"

        p_ocean = ds.Average_AltimeterPressure.data
        print("Continuing without atmospheric correction (careful!)..")
        note_str += (
            "\n- !!! NO TIME_VARYING ATMOSPHERIC CORRECTION APPLIED"
            "  !!!\n (using default atmospheric pressure offset "
            f"{ds.pressure_offset:.2f} db)"
        )

    # CALCULATE GRAVITATIONAL ACCELERATION
    if ds.lat is None:
        raise Exception(
            'No "lat" field in the dataset. Add one using'
            " sig_append.set_lat() and try again."
        )

    g = gsw.grav(ds.lat.data, 0)
    ds["g"] = (
        (), g,
        {"units": "ms-2",
         "note": f"Calculated using gsw.grav() for p=0 and lat={ds.lat:.2f}",
         },
    )
    note_str += f"\n- Using g={g:.4f} ms-2 (calculated using gsw.grav())"

    # CALCULATE OCEAN WATER DENSITY
    if hasattr(ds, "rho_CTD") and corr_CTD_density:
        rho_ocean = ds.rho_CTD.data
        note_str += "\n- Using ocean density from the *rho_CTD* field."
        fixed_rho = False
    else:
        if corr_CTD_density:
            print("\nNo density (*rho_ocean*) field found. ")
            user_input_abort_dense = input(
                'Enter "A" (Abort) or "C" '
                "(Continue using fixed rho = 1027 kg m-3): "
            ).upper()

            while user_input_abort_dense not in ["A", "C"]:
                print(f'Input ("{user_input_abort_dense}") not recognized.')
                user_input_abort_dense = input(
                    'Enter "C" (continue with fixed) or "A" (abort): '
                ).upper()
            if user_input_abort_dense == "A":
                raise Exception("ABORTED BY USER (MISSING OCEAN DENSITY)")

        rho_input = input(
            "Continuing using fixed rho. Choose: \n"
            "(R): Use rho = 1027 kg m-3, or\n"
            "(S): Specify fixed rho\n"
        ).upper()
        while rho_input not in ["R", "S"]:
            print('Input ("%s") not recognized.' % rho_input)
            rho_input = input(
                'Enter "R" (fixed rho = 1027) or "S" (specify): ').upper()

        if rho_input == "R":
            rho_ocean = 1027
        else:
            rho_ocean = np.float(input("Enter rho (kg m-3): "))
        fixed_rho = True

        print(f"Continuing with fixed rho = {rho_ocean:.1f} kg m-3")
        note_str += f"\n- Using FIXED ocean rho = {rho_ocean:.1f} kg m-3."

    # CALCULATE DEPTH
    # Factor 1e4 is conversion db -> Pa
    if fixed_rho:
        depth = 1e4 * p_ocean / g / rho_ocean
    else:
        depth = 1e4 * p_ocean / g / rho_ocean[:, np.newaxis]

    ds["depth"] = (
        ("TIME", "SAMPLE"),
        depth,
        {"units": "m", "long_name": "Transducer depth", "note": note_str},
    )

    return ds

##############################################################################


def daily_average(
    A: np.ndarray,
    t: np.ndarray,
    td: Optional[np.ndarray] = None,
    axis: int = -1,
    min_frac: float = 0.0,
    function: str = "median"
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute daily averages of a time series A on a time grid t.

    This function computes daily averages or medians of the time series data
    `A`, based on the time grid `t`. If the day index `td` is not specified, it
    will be computed automatically. The function supports both 1D and 2D
    arrays, with the time axis being the last axis.

    Args:
        A (np.ndarray): Input array containing the time series data, which can
                        be 1D or 2D.
        t (np.ndarray): Time grid corresponding to the data in `A`.
        td (Optional[np.ndarray], optional): Day index. If not provided, it
                                             will be computed from `t`.
        axis (int, optional): Axis along which the computation is performed.
                              Defaults to -1.
        min_frac (float, optional): Minimum required non-masked fraction of
                                    data to compute the statistic. If the valid
                                    data fraction is less than `min_frac`,
                                    NaN is returned. Defaults to 0.0.
        function (str, optional): Function to use for daily aggregation. Can be
                                  "median" or "mean". Defaults to "median".

    Returns:
        Tuple[np.ndarray, np.ndarray]:
            - `Ad` (np.ndarray): Daily averaged data.
            - `td` (np.ndarray): Corresponding day indices.
    """

    tfl = np.floor(t)
    if td is None:
        td = np.ma.unique(tfl)

    Nt = len(td)

    if A.ndim == 1:
        Ad_shape = Nt
    elif A.ndim == 2:
        Ad_shape = (A.shape[0], Nt)
    else:
        raise ValueError('*daily_median()* only works for 1D or 2D arrays.')

    Ad = np.zeros(Ad_shape) * np.nan

    for nn in np.arange(Nt):
        tind = tfl == td[nn]
        if tind.any():

            if sum(np.isnan(A[..., tind])) / len(A[..., tind]) < 1 - min_frac:
                with (
                    warnings.catch_warnings()
                ):  # np.nanmedian issues a warning for all-nan ensembles
                    # (we don't need to see it, so we suppress warnings
                    # for this operation)
                    warnings.simplefilter("ignore", category=RuntimeWarning)
                    if function == "median":
                        Ad[..., nn] = np.nanmedian(A[..., tind], axis=-1)
                    elif function == "mean":
                        Ad[..., nn] = np.nanmean(A[..., tind], axis=-1)

    return Ad, td


##############################################################################


def runningstat(A: np.ndarray, window_size: int) -> Dict[str, np.ndarray]:
    """
    Calculate running statistics (mean, median, standard deviation) for a time
    series.

    This function computes running statistics (mean, median, and standard
    deviation) for an equally spaced time series using a sliding window
    approach. The window size must be odd, and the boundaries are handled by
    reflecting the data at the ends.

    Note: Reflects at the ends - may have to modify the fringes of the time
    series for some applications..

    Based on script by Maksym Ganenko on this thread:
    https://stackoverflow.com/questions/33585578/
    running-or-sliding-median-mean-and-standard-deviation

    Args:
        A (np.ndarray): Equally spaced time series data. window_size (int):
                        Size of the sliding window (must be odd).

    Returns:
        Dict[str, np.ndarray]: A dictionary containing the running 'mean',
                               'median', and 'std' (standard deviation) of the
                               input time series.

    Raises:
        AssertionError: If `window_size` is not odd or greater than the length
                        of `A`.
    """

    assert window_size % 2 == 1, "window size must be odd"
    assert window_size <= len(A), "sequence must be longer than window"

    # setup index matrix
    half = window_size // 2
    row = np.arange(window_size) - half
    col = np.arange(len(A))
    index = row + col[:, None]

    # reflect boundaries
    row, col = np.triu_indices(half)
    upper = (row, half - 1 - col)
    index[upper] = np.abs(index[upper]) % len(A)
    lower = (len(A) - 1 - row, window_size - 1 - upper[1])
    index[lower] = (len(A) - 2 - index[lower]) % len(A)

    RS = {
        "median": np.median(A[index], axis=1),
        "mean": np.mean(A[index], axis=1),
        "std": np.std(A[index], axis=1),
    }

    return RS

##############################################################################


def clean_nanmedian(a: np.ndarray, **kwargs: Any) -> np.ndarray:
    """
    Compute the median of an array while ignoring NaNs, suppressing warnings
    for all-NaN slices.

    This function is a wrapper around `np.nanmedian` that suppresses the
    RuntimeWarning triggered when computing the median of an all-NaN slice. The
    default behavior of returning NaN for such slices is preserved.

    Args:
        a (np.ndarray): Input array containing numerical data, possibly with
                        NaNs. **kwargs (Any): Additional keyword arguments to
                        pass to `np.nanmedian`.

    Returns:
        np.ndarray: The median of the array along the specified axis, with NaNs
                    ignored.
    """
    with warnings.catch_warnings():
        warnings.filterwarnings(action="ignore",
                                message="All-NaN slice encountered")
        return np.nanmedian(a, **kwargs)
