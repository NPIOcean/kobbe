"""
KOBBE.APPEND

Functions for appending and interpolating external
datasets to an xarray Dataset containing Nortek Signature data.

"""

import numpy as np
import xarray as xr
from scipy.interpolate import interp1d
import gsw
from kval.util.time import matlab_time_to_python_time
from kval.util.magdec import get_declination
from kval.util.era5 import get_era5_time_series_point
from matplotlib.dates import date2num
import pandas as pd
from typing import Optional, Union, List, Dict, Any
from collections.abc import Iterable
from kval.data.moored_tools._moored_decorator import record_processing


def add_to_sigdata(
    ds: xr.Dataset,
    data: Union[np.ndarray, List[float]],
    time: Union[np.ndarray, List[Union[str, np.datetime64, float]]],
    name: str,
    attrs: Optional[Dict[str, Any]] = None,
    time_mat: bool = False,
    extrapolate: bool = False,
    interpolate_input: bool = True,
) -> xr.Dataset:
    """
    Adds a time series to the Signature dataset. Interpolates onto the "TIME"
    coordinate (one entry per ensemble).

    This function can be used to append time series data, such as remote
    sensing data, to an existing xarray dataset for comparison or validation.

    Parameters
    ----------
    ds : xr.Dataset
        The xarray dataset with signature data.
    data : Union[np.ndarray, List[float]]
        The time series data to be added.
    time : Union[np.ndarray, List[Union[str, np.datetime64, float]]]
        The time grid of the data. This can be in datetime64 format, as strings
        (e.g., ['2022-05-21']), or as numeric values (e.g., Python epoch time
        or Matlab timestamp).
    name : str
        The name of the new variable to be added to the dataset.
    attrs : Optional[Dict[str, Any]], optional
        Attributes of the new variable, such as units and long_name, by default
        None.
    time_mat : bool, optional
        Set to True if `time` is provided in MATLAB epoch format. If False
        (default), the time is assumed to be in Python default epoch or
        standard datetime format.
    extrapolate : bool, optional
        If True, values will be extrapolated outside the range of the input
        data by simply extending the edge values. By default, extrapolate is
        set to False.
    interpolate_input: If True, the function will interpolate over gaps in the
                       input data.

    Returns
    -------
    xr.Dataset
        The xarray dataset including the new variable.
    """

    if time_mat:
        time = matlab_time_to_python_time(time)


    # Convert time/data to NumPy arrays if they are not already one
    time = np.asarray(time)
    data = np.asarray(data)

    # Handle string-based time input
    if isinstance(time[0], str):
        time = np.array(pd.to_datetime(time))

    # Check if time is in datetime64 format and convert to numeric
    # (float) if necessary
    if np.issubdtype(time.dtype, np.datetime64):
        time = date2num(time)

    interp1d_kws = {"bounds_error": False}
    if extrapolate:

        first_value = data[~np.isnan(data)][0]
        last_value = data[~np.isnan(data)][-1]

        interp1d_kws["fill_value"] = (first_value, last_value)

    # Interpolatant of the time series
    if interpolate_input:  # Ignore NaNs if "interpolate_input"
        if np.isnan(data).any():
            data_ip = interp1d(time[~np.isnan(data)],
                               data[~np.isnan(data)],
                               **interp1d_kws)
        else:
            data_ip = interp1d(time, data,
                               **interp1d_kws)
    else:
        data_ip = interp1d(time, data, **interp1d_kws)

    # Add interpolated data to ds
    ds[name] = (("TIME"), data_ip(ds.TIME.data), attrs)

    return ds


def append_ctd(
    ds: xr.Dataset,
    temp: np.ndarray,
    sal: np.ndarray,
    pres: np.ndarray,
    CTDtime: np.ndarray,
    instr_SN: Optional[str] = 'N/A',
    instr_desc: Optional[str] = 'N/A',
    time_mat: bool = False,
    extrapolate: bool = True,
    other_attrs: dict = {},
) -> xr.Dataset:
    """
    Append moored CTD data to an xarray Signature Dataset, converting to
    TEOS-10 variables and computing sound speed using the GSW module. The CTD
    data is interpolated onto the time grid of the signature data in the
    provided dataset.

    Parameters
    ----------
    ds : xr.Dataset
        The xarray Dataset containing signature data to which CTD data
        will be added.
    temp : np.ndarray
        In-situ temperature in degrees Celsius.
    sal : np.ndarray
        Practical salinity (dimensionless).
    pres : np.ndarray
        Ocean pressure in dbar.
    CTDtime : np.ndarray
        Timestamps corresponding to the CTD measurements.
    instr_SN : Optional[str], optional
        Instrument serial number, by default 'N/A'.
    instr_desc : Optional[str], optional
        Description of the instrument, by default None.
    time_mat : bool, optional
        If True, time interpolation will use a MATLAB-compatible method,
        by default False.
    extrapolate : bool, optional
        If True, allows extrapolation during interpolation, by default True.
    other_attrs: dict, optional
        Dictonary of other attributes to include.

    Returns
    -------
    xr.Dataset
        The xarray Dataset with added variables for Absolute Salinity (SA),
        Conservative Temperature (CT), Pressure (pres_CTD), Sound Speed
        (sound_speed_CTD), and Ocean Density (rho_CTD), all interpolated
        onto the signature data time grid.
    """

    if 'LATITUDE' not in ds and 'LONGITUDE' not in ds:
        raise Exception('No lat/lon found in the dataset (required for CTD'
                        ' calculations. Add with kobbe.append.set_lat_lon.')

    # Convert practical salinity to absolute salinity to absolute salinity
    SA = gsw.SA_from_SP(SP=sal, p=pres, lon=ds.LONGITUDE.data, lat=ds.LATITUDE.data)

    # Convert in-situ temperature to conservative temperature
    CT = gsw.CT_from_t(SA=SA, t=temp, p=pres)

    # Compute sound speed and density
    ss = gsw.sound_speed(SA=SA, CT=CT, p=pres)
    rho = gsw.rho(SA=SA, CT=CT, p=pres)

    # Define shared attributes
    attrs_all = {
        "Instrument description": instr_desc,
        "Instrument SN": instr_SN,
        "note": ("Calculated using the gsw module. Linearly interpolated"
                 " onto Sig500 time grid."),
    }
    # Append custom attributes
    attrs_all = {**attrs_all, **other_attrs}

    ds = add_to_sigdata(
        ds, SA, CTDtime, "SA_CTD",
        attrs={"long_name": "Absolute Salinity", "units": "g kg-1",
               **attrs_all},
        time_mat=time_mat,
        extrapolate=extrapolate,
    )
    ds = add_to_sigdata(
        ds, CT, CTDtime, "CT_CTD",
        attrs={"long_name": "Conservative Temperature", "units": "degC",
               **attrs_all},
        time_mat=time_mat,
        extrapolate=extrapolate,
    )
    ds = add_to_sigdata(
        ds, pres, CTDtime, "pres_CTD",
        attrs={
            "long_name": "Pressure (CTD measurements)",
            "units": "dbar",
            **attrs_all,
        },
        time_mat=time_mat,
        extrapolate=extrapolate,
    )
    ds = add_to_sigdata(
        ds, ss, CTDtime, "sound_speed_CTD",
        attrs={"long_name": "Sound speed", "units": "m s-1", **attrs_all},
        time_mat=time_mat,
        extrapolate=extrapolate,
    )
    ds = add_to_sigdata(
        ds, rho, CTDtime, "rho_CTD",
        attrs={
            "long_name": "Ocean water density (CTD measurements)",
            "units": "kg m-3",
            **attrs_all,
        },
        time_mat=time_mat,
        extrapolate=extrapolate,
    )
    return ds


def append_atm_pres_auto(
    ds: xr.Dataset,
) -> xr.Dataset:
    """
    Automatically obtain atmospheric pressure from ERA-5 and append to
    the dataset.

    Can take a while (expect a minute or so for most applications).

    Hourly ERA-5 data are obtained over OpenDAP from the Asia-Pacific
    Data Research Center (APDRC): http://apdrc.soest.hawaii.edu/dods/
    public_data/Reanalysis_Data/ERA5/hourly/Surface_pressure.info.

    Inputs
    ------

    ds: xarray dataset with signature data.

    Outputs
    -------
    ds: The xarray dataset including the SLP variable.
    """

    if 'LATITUDE' not in ds or 'LONGITUDE' not in ds:
        raise Exception(
            "['LATITUDE', 'LONGITUDE'] not found in dataset. "
            "Run append.set_lat_lon first..")

    ds_slp = get_era5_time_series_point(
                'SLP', 'hourly',
                ds.LATITUDE.copy(), ds.LONGITUDE.copy(),
                ds.TIME[0].item()-1,
                ds.TIME[-1].item()+1)

    # Define default attributes and update with any provided attributes
    attrs_all = {"long_name": "Sea level pressure", "units": "dbar",
                 "comment": ('Obtained from ERA-5 hourly surface pressure '
                             '(nearest grid point)'),
                 "era5_lon": f'{ds_slp.lon.item()} ({ds_slp.lon.item()-360})',
                 "era5_lat": ds_slp.lat.item(),
                 "era5_variable_name": "sp",
                 "era5_opendap_source": ('http://apdrc.soest.hawaii.edu:80/'
                                         'dods/public_data/Reanalysis_Data/'
                                         'ERA5/hourly/Surface_pressure')
                 }

    # Append the sea level pressure data to the sig500 dataset
    # Factor 1e4 converts from Pa to dbar
    ds = add_to_sigdata(
        ds,
        ds_slp.SLP*1e-4,
        ds_slp.time,
        "p_atmo",
        attrs=attrs_all,
        time_mat=False
    )

    return ds


def append_atm_pres(
    ds: xr.Dataset,
    slp: np.ndarray,
    slptime: np.ndarray,
    attrs: Optional[Dict[str, Any]] = None,
    time_mat: bool = False
) -> xr.Dataset:
    """
    Append sea level pressure from e.g. ERA-5. Note that the
    pressure units should be dbar.

    Append external sea level atmospheric pressure data (from e.g. ERA5)
    to a Signature xarray Dataset.

    Interpolates onto the `TIME` grid of the sig500 data
    and adds to the sig500 data as the variable *SLP*.

    Inputs
    ------

    ds: xarray dataset with signature data.

    slp: Sea level atmospheric pressure [dbar].
    slptime: Time stamp of slp.
    attrs: Attributes (dictionary).
    time_mat: Set to True if *slptime* is matlab epoch
              (False/default: python default epoch or datetime64).

    Outputs
    -------
    ds: The xarray dataset including the SLP variable.
    """
    # Define default attributes and update with any provided attributes
    attrs_all = {"long_name": "Sea level pressure", "units": "dbar"}
    if attrs:
        attrs_all.update(attrs)

    # Append the sea level pressure data to the sig500 dataset
    ds = add_to_sigdata(
        ds,
        slp,
        slptime,
        "p_atmo",
        attrs=attrs_all,
        time_mat=time_mat
    )

    return ds


def append_magdec_auto(
        ds: xr.Dataset,
        model: str = 'auto') -> xr.Dataset:
    '''
    Automatically obtain magnetic declination from the World Magnetic Model.

    Wrapping the pygeomag module (https://pypi.org/project/pygeomag/) which
    obtains declination angle from the World Magnetic Model (WMM).

    model: str
        Which WMM version to use.
        Options: ['2010', '2015', '2020'], otherwise determined based on the
        data. Note that auto detection can cause (small) discontinuities across
        2014-2015 and 2019-2020.
    '''

    if 'LATITUDE' not in ds or 'LONGITUDE' not in ds:
        raise Exception(
            "['LATITUDE', 'LONGITUDE'] not found in dataset. "
            "Run append.set_lat_lon first..")

    dec = get_declination(
        dates=ds.TIME.values, lat=ds.LATITUDE, lon=ds.LONGITUDE, model=model)

    attrs = {'note':
             'Time-varying magnetic declination angle from the World Magnetic'
             ' Model, accessed using the `pymagdec` library '
             '(https://pypi.org/project/pygeomag/).'}

    ds = append_magdec(ds, magdec=dec, magdectime=ds.TIME, attrs=attrs
                       )
    return ds


def append_magdec(
    ds: xr.Dataset,
    magdec: Union[float, np.ndarray],
    magdectime: Optional[np.ndarray] = None,
    attrs: Optional[Dict[str, Any]] = None,
    time_mat: bool = False,
    extrapolate: bool = True
) -> xr.Dataset:

    """
    Append the magnetic declination angle to an xarray Dataset. This angle
    is used for correcting the heading of observed velocities. The magnetic
    declination can be provided as a fixed value or as a time-varying array.

    If time-varying, the declination will be interpolated onto the time grid
    of the signature data in the provided dataset.

    Appended to the sig500 data as the variable `magdec` -
    either as a single number or interpolated onto the `TIME` grid
    of the sig500 data.

    Parameters
    ----------
    ds : xr.Dataset
        The xarray Dataset containing signature data to which the magnetic
        declination will be added.
    magdec : Union[float, np.ndarray]
        Magnetic declination angle in degrees. Can be a single float value
        or an array of time-varying declinations.
    magdectime : Optional[np.ndarray], optional
        Timestamps corresponding to the time-varying magnetic declination.
        Required if `magdec` is an array, by default None.
    attrs : Optional[Dict[str, Any]], optional
        Attributes to add to the magnetic declination variable, by default
        None. If provided, this dictionary will override the default
        attributes.
    time_mat : bool, optional
        If True, indicates that `magdectime` is in MATLAB epoch format,
        by default False.
    extrapolate : bool, optional
        If True, allows extrapolation during interpolation, by default True.

    Returns
    -------
    xr.Dataset
        The xarray Dataset with the added magnetic declination (`magdec`)
        variable.
    """

    # Define default attributes and update with any provided attributes
    attrs_all = {"long_name": "Magnetic declination", "units": "degrees"}
    if attrs:
        attrs_all.update(attrs)

    # Append the magnetic declination to the dataset
    if isinstance(magdec, Iterable) and not isinstance(magdec, str):
        if magdectime is None:
            raise ValueError(
                "Looks like you supplied a time-varying `magdec` but did not "
                "provide the required timestamps `magdectime`."
            )
        ds = add_to_sigdata(
            ds,
            magdec,
            magdectime,
            "magdec",
            attrs=attrs_all,
            time_mat=time_mat,
            extrapolate=extrapolate,
        )
    else:  # If magdec is a single value
        ds["magdec"] = ((), magdec, attrs_all)

    return ds


def set_lat_lon(ds: xr.Dataset, lat: float, lon: float) -> xr.Dataset:
    """
    Append a single latitude value to a Signature xarray Dataset.

    Parameters
    ----------
    ds : xr.Dataset
        The xarray Dataset to which the latitude will be added.
    lat : float
        Latitude in degrees north.

    Returns
    -------
    xr.Dataset
        The xarray Dataset with the added latitude variable (`lat`).
    """
    ds["LATITUDE"] = ((), lat,
                      {"long_name": "Latitude",
                       "standard_name": "latitude",
                       "units": "degrees_north",
                       "axis": "Y",
                       "coverage_content_type": "coordinate"})
    ds["LONGITUDE"] = ((), lon,
                       {"long_name": "Longitude",
                        "standard_name": "longitude",
                        "units": "degrees_east",
                        "axis": 'X',
                        "coverage_content_type": "coordinate"})
    ds = ds.set_coords(['LATITUDE', 'LONGITUDE'])

    return ds


##############################################################################


def _add_tilt(ds: xr.Dataset) -> xr.Dataset:
    """
    Calculate tilt from pitch and roll components in an xarray Dataset.
    Tilt is calculated using the method described in Mantovanelli et al. (2014)
    and Woodgate et al. (2011).

    Parameters
    ----------
    ds : xr.Dataset
        An xarray Dataset containing the pitch and roll fields, typically
        as read by `matfiles_to_dataset`.

    Returns
    -------
    xr.Dataset
        The xarray Dataset with added tilt variables (`tilt_Average` and
        `tilt_AverageIce`), if applicable.

    Raises
    ------
    ValueError
        If required pitch or roll data is missing in the Dataset.
    """

    tilt_attrs: Dict[str, Union[str, float]] = {
        "units": "degrees",
        "desc": "Tilt calculated from pitch+roll",
        "note": (
            "Calculated using the function kobbe.funcs._add_tilt(). "
            "See Mantovanelli et al 2014 and Woodgate et al 2011."
        ),
    }

    try:
        cos_tilt = np.cos(ds.Average_Pitch.data / 180 * np.pi) * np.cos(
            ds.Average_Roll.data / 180 * np.pi
        )

        ds["tilt_Average"] = (
            ("time_average"),
            180 / np.pi * np.arccos(cos_tilt),
            tilt_attrs,
        )
    except AttributeError as e:
        raise ValueError("Pitch and/or Roll data is missing..") from e

    try:
        cos_tilt_avgice = (
            np.cos(ds.AverageIce_Pitch.data / 180 * np.pi)
            * np.cos(ds.AverageIce_Roll.data / 180 * np.pi)
        )

        ds["tilt_AverageIce"] = (
            ("time_average"),
            180 / np.pi * np.arccos(cos_tilt_avgice),
            tilt_attrs,
        )
    except AttributeError:
        # If AverageIce_Pitch or AverageIce_Roll data is missing, skip this
        # step.
        pass

    return ds

##############################################################################


def _add_SIC_FOM(
        ds: xr.Dataset, FOM_thr: Optional[float] = None) -> xr.Dataset:
    """
    Add estimates of sea ice presence in each sample and sea ice concentration
    in each ensemble based on the Figure-of-Merit (FOM) metric reported by the
    four slanted beams in the `AverageIce` data.

    Two estimates are added:
    - Conservative estimate (`ICE_IN_SAMPLE` and `SIC_FOM`):
      FOM < FOM_thr for ALL beams.
    - Alternative estimate (`ICE_IN_SAMPLE_ANY` and `SIC_FOM_ALT`):
      FOM < FOM_thr for ANY of the beams.

    The former seems to give a better estimate of sea ice concentration,
    but the latter is useful for isolating open-water-only samples.

    Note: The sea ice concentration variables "SIC_FOM", "SIC_FOM_ALT"
    (percent) are typically most useful as estimators of sea ice concentration
    when they are averaged over a longer time period (e.g. daily).

    Parameters
    ----------
    ds : xr.Dataset
        The xarray Dataset containing the data.
    FOM_thr : Optional[float], optional
        Figure-of-Merit threshold. If not specified, the "FOM_threshold"
        variable from the dataset is used.

   Returns
    -------
    xr.Dataset
        The xarray Dataset with added fields:
        - `ICE_IN_SAMPLE`: Ice presence per sample (conservative).
        - `ICE_IN_SAMPLE_ANY`: Ice presence per sample (alternative).
        - `SIC_FOM`: Estimated sea ice concentration per ensemble
                     (conservative).
        - `SIC_FOM_ALT`: Estimated sea ice concentration per ensemble
                         (alternative).


    Raises
    ------
    KeyError
        If the required FOM variables or FOM_threshold are missing in
        the Dataset.

    """

    # Use the FOM threshold specified in the dataset unless provided
    if FOM_thr is None:
        try:
            FOM_thr = float(ds.FOM_threshold)
        except AttributeError:
            raise KeyError("FOM_threshold variable is missing in the Dataset.")

    # Initialize boolean arrays for ice detection
    try:
        if 'SAMPLE' in ds.dims:
            ALL_ICE_IN_SAMPLE = np.ones(
                [ds.sizes["TIME"], ds.sizes["SAMPLE"]], dtype=bool)
            ALL_WATER_IN_SAMPLE = np.ones(
                [ds.sizes["TIME"], ds.sizes["SAMPLE"]], dtype=bool)
            for nn in range(1, 5):
                FOMnm = f"AverageIce_FOMBeam{nn}"
                ALL_WATER_IN_SAMPLE &= ds[FOMnm].data > FOM_thr
                ALL_ICE_IN_SAMPLE &= ds[FOMnm].data < FOM_thr
        else:
            print('Working with non-reshaped dataset '
                  '-> Not calculating ice presence')
            return ds
    except KeyError as e:
        FOMnm = "AverageIce_FOMBeam[1-4]"
        raise KeyError(
            f"Required FOM variable {FOMnm} is missing in the Dataset."
            ) from e

    ANY_ICE_IN_SAMPLE = ~ALL_WATER_IN_SAMPLE

    # Add ice presence to dataset
    ds["ICE_IN_SAMPLE"] = (
        ("TIME", "SAMPLE"),
        ALL_ICE_IN_SAMPLE,
        {
            "long_name": ("Identification of sea ice in sample "
                          "(conservative estimate)"),
            "desc": ('Binary classification (ice/not ice), where "ice" is when'
                     f' FOM < {FOM_thr:.0f} in ALL of the 4 slanted beams.'),
        },
    )

    ds["ICE_IN_SAMPLE_ANY"] = (
        ("TIME", "SAMPLE"),
        ANY_ICE_IN_SAMPLE,
        {
            "long_name": ("Identification of sea ice in sample"
                          " (alternative estimate)"),
            "desc": ('Binary classification (ice/not ice), where "ice" is when'
                     f" FOM < {FOM_thr:.0f} in ONE OR MORE of the 4"
                     " slanted beams.")
        },
    )

    # Calculate and add sea ice concentration to dataset
    SIC = ALL_ICE_IN_SAMPLE.mean(axis=1) * 100
    ds["SIC_FOM"] = (
        ("TIME"),
        SIC,
        {
            "long_name": "Sea ice concentration",
            "desc": (
                '"Sea ice concentration" in each ensemble based on FOM '
                'criterion. Calculated as the fraction of samples per'
                f' ensemble where FOM is below {FOM_thr:.0f} for ALL of the'
                ' four slanted beams.'
            ),
            "units": "%",
            "note": ("Typically most useful when averaged over a longer "
                     "period (e.g. daily)."),
        },
    )

    SIC_ALT = ANY_ICE_IN_SAMPLE.mean(axis=1) * 100
    ds["SIC_FOM_ALT"] = (
        ("TIME"),
        SIC_ALT,
        {
            "long_name": "Sea ice concentration (alternative estimate)",
            "desc": (
                '"Sea ice concentration" in each ensemble based on FOM '
                'criterion. Calculated as the fraction of samples per '
                f'ensemble where FOM is below {FOM_thr:.0f} for AT LEAST'
                ' ONE of the four slanted beams.'
            ),
            "units": "%",
            "note": (
                '*SIC_FOM_ALT* seems a bit "trigger happy" - recommended '
                "to use the more conservative *SIC_FOM*. Typically most useful"
                " when averaged over a longer period (e.g. daily)."
            ),
        },
    )

    return ds
