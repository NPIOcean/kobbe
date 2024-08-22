'''
Functions to append external datasets to an xarray Dataset containing Nortek
Signature data.

- General function for adding and interpolating any time series data:

Some specialized wrapper functions used for loading data that
needs to be formatted correctly in later operations:

- Add CTD data and compute sound speed (for ice draft calculations)
- Add air pressure (for instrument depth corrections)
- Add magnetic declination (for correction of velocity directions)

TO DO:
- Check air pressure wrapper.
- Think about whether the ERA-5 picker should be retained here or elsewhere
  (otherwise remove)

'''

import numpy as np
import xarray as xr
from scipy.interpolate import interp1d
import gsw
from kobbe.calc import mat_to_py_time
from matplotlib.dates import date2num, num2date
import pandas as pd
from typing import Optional, Union, List, Dict, Any

def add_to_sigdata(
    ds: xr.Dataset,
    data: Union[np.ndarray, List[float]],
    time: Union[np.ndarray, List[Union[str, np.datetime64, float]]],
    name: str,
    attrs: Optional[Dict[str, Any]] = None,
    time_mat: bool = False,
    extrapolate: bool = False
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
        Attributes of the new variable, such as units and long_name, by default None.
    time_mat : bool, optional
        Set to True if `time` is provided in MATLAB epoch format. If False (default),
        the time is assumed to be in Python default epoch or standard datetime format.
    extrapolate : bool, optional
        If True, values will be linearly extrapolated outside the range of the input
        data. By default, extrapolate is set to False.

    Returns
    -------
    xr.Dataset
        The xarray dataset including the new variable.
    """

    if time_mat:
        time = mat_to_py_time(time)


    # Handle string-based time input
    if isinstance(time, (list, np.ndarray)) and isinstance(time[0], str):
        time = np.array(pd.to_datetime(time))

    # Check if time is in datetime64 format and convert to numeric
    # (float) if necessary
    if np.issubdtype(time.dtype, np.datetime64):
        time = date2num(time)

    tfmt = '%d %b %Y'
    tstrs_ds = (num2date(ds.TIME[0]).strftime(tfmt),
                num2date(ds.TIME[-1]).strftime(tfmt))
    tstrs_input = (num2date(time[0]).strftime(tfmt),
                   num2date(time[-1]).strftime(tfmt))

    interp1d_kws = {'bounds_error':False}
    if extrapolate:
        interp1d_kws['fill_value'] = 'extrapolate'

    # Interpolatant of the time series
    data_ip = interp1d(time, data, **interp1d_kws)

    # Add interpolated data to dx
    ds[name] = (('TIME'), data_ip(ds.TIME.data), attrs)

    return ds


def append_ctd(ds, temp, sal, pres, CTDtime, instr_SN = None, instr_desc = None,
                time_mat = False, extrapolate = True):
    '''
    Read data from a moored CTD - used for sound speed corrections etc.
    Converts to TEOS-10 and computes sound speed using the gsw module.

    Interpolates onto the *time* grid of the sig500 data.

    Note: *temp, sal, pres* should be on the same *time* grid,

    Inputs
    ------

    dx: xarray dataset with signature data.

    temp: In-situ temperature [C].
    salt: Practical salinity [].
    pres: Ocean pressure [dbar].
    CTDtime: Time stamp of CTD measurements .

    Outputs
    -------
    dx: The xarray dataset including the new SA, CT, pres_CTD, and
        sound_speed_CTD variables.
    '''

    SA = gsw.SA_from_SP(sal, pres, ds.lon.data, ds.lat.data)
    CT = gsw.CT_from_t(SA, temp, pres)
    ss = gsw.sound_speed(SA, CT, pres)
    rho = gsw.rho(SA, CT, pres)

    attrs_all = {'Instrument description': instr_desc, 'Instrument SN':instr_SN,
            'note':'Calculated using the gsw module. Linearly'
            ' interpolated onto Sig500 time grid.'}

    ds = add_to_sigdata(ds, SA, CTDtime, 'SA_CTD',
                attrs = {'long_name':'Absolute Salinity', 'units':'g kg-1',
                  **attrs_all},
                time_mat = time_mat, extrapolate = extrapolate)
    ds = add_to_sigdata(ds, CT, CTDtime, 'CT_CTD',
                attrs = {'long_name':'Conservative Temperature', 'units':'degC',
                    **attrs_all},
                time_mat = time_mat, extrapolate = extrapolate)
    ds = add_to_sigdata(ds, pres, CTDtime, 'pres_CTD',
                attrs = {'long_name':'Pressure (CTD measurements)',
                    'units':'dbar', **attrs_all},
                time_mat = time_mat, extrapolate = extrapolate)
    ds = add_to_sigdata(ds, ss, CTDtime, 'sound_speed_CTD',
                attrs = {'long_name':'Sound speed', 'units':'m s-1',
                    **attrs_all},
                time_mat = time_mat,  extrapolate = extrapolate)
    ds = add_to_sigdata(ds, rho, CTDtime, 'rho_CTD',
                attrs = {'long_name':'Ocean water density (CTD measurements)', 'units':'kg m-3',
                    **attrs_all},
                time_mat = time_mat,  extrapolate = extrapolate)
    return ds


def append_atm_pres(ds, slp, slptime, attrs = None,
                    time_mat = False):
    '''
    Append sea level pressure from e.g. ERA-5. Note that the
    pressure units should be dbar.

    Interpolates onto the *time* grid of the sig500 data
    and adds to the sig500 data as the variable *SLP*.

    Inputs
    ------

    ds: xarray dataset with signature data.

    slp: Sea level atmospheric pressure [dbar].
    slptime: Time stamp of slp.
    attrs: Attributes (dictionary).
    time_mat: Set to True if *slptime* is matlab epoch
              (False/default: python default epoch).

    Outputs
    -------
    ds: The xarray dataset including the SLP variable.
    '''

    # Modify the attribute dictionary (specified "units" and "long_name"
    # overrides default ones).
    attrs_all = {'long_name':'Sea level pressure', 'units':'dbar'}
    if attrs:
        for nm in attrs:
            attrs_all[nm] = attrs[nm]

    # Append to sig500 data
    ds = add_to_sigdata(ds, slp, slptime, 'p_atmo',
                attrs = attrs_all,
                time_mat = time_mat)

    return ds


def append_magdec(dx, magdec, magdectime = False, attrs = None,
                    time_mat = False, extrapolate = True):
    '''
    Append magnetic declination angle, used for correcting the heading of
    observed velocities.

    Magnetic declination can be supplied as a fixed number or as a
    time-varying quantity (useful for longer deployments or cases
    where the position is not fixed).

    Appended to the sig500 data as the variable *magdec* -
    either as a single number or interpolated onto the *time* grid
    of the sig500 data.

    Inputs
    ------

    dx: xarray dataset with signature data.

    magdec: Magnetic declination angle [degrees] - single number or array
            of time-varying declination.
    magdectime: Time stamp of magdec if it is time-varying.
    attrs: Attributes (dictionary).
    time_mat: Set to True if *slptime* is matlab epoch
              (False/default: python default epoch).

    Outputs
    -------
    dx: The xarray dataset including the magdec variable.
    '''

    # Modify the attribute dictionary (specified "units" and "long_name"
    # overrides default ones).
    attrs_all = {'long_name':'Magnetic declination', 'units':'degrees'}
    if attrs:
        for nm in attrs:
            attrs_all[nm] = attrs[nm]

    # Append to Signature data
    if hasattr(magdec, '__iter__'): # If this is an array of several
                                    # magdec values
        if not hasattr(magdec, '__iter__'):
            raise Exception('Looks like you supplied a time-varying'
            '*magdec* but not the required time stamps *magdectime*..')
        else:
            add_to_sigdata(dx, magdec, magdectime, 'magdec',
                attrs = attrs_all,
                time_mat = time_mat,
                extrapolate = extrapolate)

    else:
        dx['magdec'] = ((), magdec, attrs_all)

    return dx


def set_lat(dx, lat):
    '''
    Append latitude (degrees north, single value) to the dataset.
    '''
    dx['lat'] = ((), lat, {'long_name':'Latitude', 'units':'degrees_north'})
    return dx


def set_lon(dx, lon):
    '''
    Append latitude (degrees north, single value) to the dataset.
    '''
    dx['lon'] = ((), lon, {'long_name':'Longitude', 'units':'degrees_east'})
    return dx



##############################################################################

def _add_tilt(ds):
    '''
    Calculate tilt from pitch/roll components. See Mantovanelli et al 2014 and
    Woodgate et al 2011.

    Input: xarray dataset with pitch and roll fields as read by
    matfiles_to_dataset

    '''

    tilt_attrs = {
        'units':'degrees',
        'desc': ('Tilt calculated from pitch+roll'),
        'note':('Calculated using the function kobbe.funcs._add_tilt(). '
            'See Mantovanelli et al 2014 and Woodgate et al 2011.'),}

    try:
        cos_tilt = (np.cos(ds.Average_Pitch.data/180*np.pi)
                    * np.cos(ds.Average_Roll.data/180*np.pi))


        ds['tilt_Average'] = (('time_average'),
            180 / np.pi* np.arccos(cos_tilt), tilt_attrs)
      #  ds['tilt_Average'].attrs  =  tilt_attrs
    except:
        print('AAA')
        return tilt_attrs, cos_tilt
        pass

    try:
        cos_tilt_avgice = (np.cos(ds.AverageIce_Pitch.data/180*np.pi)
                    * np.cos(ds.AverageIce_Roll.data/180*np.pi))
        ds['tilt_AverageIce'] = (('time_average'),
            180 / np.pi* np.arccos(cos_tilt_avgice), tilt_attrs)
       # ds['tilt_AverageIce'].attrs  = tilt_attrs
    except:
        pass

    return ds


##############################################################################

def _add_SIC_FOM(ds, FOM_thr = None):
    '''
    Add estimates of sea ice presence in each sample, and sea ice
    concentration in each ensemble, from the Figure-of-Merit (FOM)
    metric reported by the four slanted beam in the AverageIce data.

    - Conservative estimate (no suffix): FOM<FOM_thr for ALL beams
    - Alternative estimate ('_ALT'): FOM<FOM_thr for ANY OF FOUR beams

    The former seems to give a better estimate of sea ice concentration.

    Using the "FOM_threshold" variable in the dataset unless otherwise
    specified.

    The sea ice concentration variables "SIC_FOM", "SIC_FOM_ALT"
    (percent) are typically most useful whean veraging over a longer
    time period (e.g. daily).

    Inputs:
    ------
    ds: xarray Dataset containing the data.
    FOMthr: Figure-of-Merit threshold
            (using "FOM_threshold" specified in ds unless otherwise
            specified)

    Outputs:
    --------
    ds: xarray Dataset containing the data, with added fields:
    - ICE_IN_SAMPLE - Ice presence per sample (conservative)
    - ICE_IN_SAMPLE_ALT - Ice presence per sample (alternative)
    - SIC_FOM - Estimated sea ice concentration per ensemble (conservative)
    - SIC_FOM_ALT - Estimated sea ice concentration per ensemble
                     (alternative)
    '''
    # FOM threshold for ice vs ocean
    if FOM_thr == None:
        FOMthr = float(ds.FOM_threshold)

    # Find ensemble indices where we have ice
    ALL_ICE_IN_SAMPLE = np.bool_(np.ones([ds.sizes['TIME'],
                             ds.sizes['SAMPLE']]))

    ALL_WATER_IN_SAMPLE = np.bool_(np.ones([ds.sizes['TIME'],
                             ds.sizes['SAMPLE']]))

    for nn in np.arange(1, 5):
        FOMnm = 'AverageIce_FOMBeam%i'%nn

        # False if FOM<thr (IS ICE) for ANY beam
        ALL_WATER_IN_SAMPLE *= ds[FOMnm].data > FOMthr

        # False if FOM>thr (IS WATER) for ANY beam
        ALL_ICE_IN_SAMPLE *= ds[FOMnm].data < FOMthr

    ANY_ICE_IN_SAMPLE = ~ALL_WATER_IN_SAMPLE

    ds['ICE_IN_SAMPLE'] = (('TIME', 'SAMPLE'), ALL_ICE_IN_SAMPLE,
        {'long_name':('Identification of sea ice in sample'
                     ' (conservative estimate)'),
         'desc':'Binary classification (ice/not ice), where "ice" '
         'is when FOM < %.0f in ALL of the 4 slanted beams.'%FOMthr})


    ds['ICE_IN_SAMPLE_ANY'] = (('TIME', 'SAMPLE'), ANY_ICE_IN_SAMPLE,
        {'long_name':('Identification of sea ice in sample'),
         'desc':'Binary classification (ice/not ice), where "ice" is when '
         'FOM < %.0f in ONE OR MORE of the 4 slanted beams.'%FOMthr})

    SIC = ALL_ICE_IN_SAMPLE.mean(axis = 1)*100

    ds['SIC_FOM'] = (('TIME'), SIC, {'long_name':'Sea ice concentration',
        'desc':('"Sea ice concentration" in each '
        'ensemble based on FOM criterion. '
        'Calculated as the fraction of '
        'samples per ensemble where FOM is below %.0f for ALL '
        'of the four slanted beams.')%FOMthr,
        'units':'%',
        'note':('Typically most useful when averaged over a longer '
        'period (e.g. daily)')})

    SIC_ALT = ANY_ICE_IN_SAMPLE.mean(axis = 1)*100

    ds['SIC_FOM_ALT'] = (('TIME'), SIC_ALT,
        {'long_name':'Sea ice concentration (alternative)',
        'desc':('"Sea ice concentration" in each '
        'ensemble based on FOM criterion. '
        'Calculated as the fraction of '
        'samples per ensemble where FOM is below %.0f for ALT LEAST ONE '
        'of the four slanted beams.')%FOMthr,
        'units':'%',
        'note':('*SIC_FOM_ALT* seems a bit "trigger happy" - recommended'
                ' to use the more conservative *SIC_FOM*.\nTypically most'
                ' useful when averaged over a longer period (e.g. daily).')})

    return ds



# ERA-5 retrieval deprecated for now (decided for now that this is best suited elsewhere as users may
# do this differently. This way is a bit cumbersome anyways..)

if False:
    def get_era5(dx, temp_2m = False, wind_10m = False):
        '''
        Read ERA-5 variables in the nearest model grid cell:

        - Sea level pressure [dbar] (Used for depth correction)
        - 2-m air temperature [deg C] (Optional - toggle with *temp_2m=True*)
        - 10-m wind components [m s-1] (Optional - toggle with *wind_10m=True*)

        Adds to the sig500 dictionary as 1D variables interpolated to the *time*
        grid.

        Accessing hourly ERA-5 data over OpenDap from the Asia-Pacific Data
        Research Center (http://apdrc.soest.hawaii.edu/datadoc/ecmwf_ERA5.php).

        Note: Accessing the data can take as much as tens of minutes.
        '''
        # Loading remote datasets
        # This operation can take 10-20 seconds
        era5_url = ('http://apdrc.soest.hawaii.edu:80/dods/public_data/'
                    'Reanalysis_Data/ERA5/hourly/')

        print('Connecting to ERA5 MSL remote dataset..')
        era5_msl = xr.open_dataset(era5_url + 'Surface_pressure')
        if temp_2m:
            print('Connecting to ERA5 2-m air temperature remote dataset..')
            era5_msl = xr.open_dataset(era5_url + '2m_temperature')
        if wind_10m:
            print('Connecting to ERA5 10-m wind remote datasets..')
            era5_uwind = xr.open_dataset(era5_url + 'U_wind_component_10m')
            era5_vwind = xr.open_dataset(era5_url + 'V_wind_component_10m')
        print('..done.')

        # Find overlapping time stamps
        t_era_full = date2num(era5.time.data)
        # Start and end points of sig500 deployment
        tind0_era = np.searchsorted(t_era_full, dx.time[0])-2
        tind1_era = np.searchsorted(t_era_full, dx.time[-1])+2
        tsl_era = slice(tind0_era, tind1_era)