'''
Various functions for loading and concatenating Nortek Signature matfiles
produced by Nortek SignatureDeployment.

TO DO:

- Checking, fixing
- Does the *sig_mat_to_dict_join* do anything now?
- Make it easier/clearer to preserve/skip IBurst, RawAltimeter etc..
- Time: Specify epoch?

- Investigate what the RuntimeError is in the tilt function..
- Look over what print messages are necessary (definitely cut some from the reshaper)

'''

##############################################################################

# IMPORTS

import numpy as np
from scipy.io import loadmat
import xarray as xr
from matplotlib.dates import num2date, date2num
import matplotlib.pyplot as plt 
from IPython.display import display
from sigpyproc.sig_calc import mat_to_py_time
from sigpyproc.sig_append import _add_tilt, _add_SIC_FOM, set_lat, set_lon
from datetime import datetime
import warnings

##############################################################################

def matfiles_to_dataset(file_list, reshape = True, lat = None, lon = None,
                include_raw_altimeter = False, FOM_ice_threshold = 1e4,
                time_range = [None, None]):
    '''
    Read, convert, and concatenate .mat files exported from SignatureDeployment.

    Inputs:
    -------

    file_list: list of .mat files.
    reshape: If True, reshape all time series from a single 'time_average'
             dimension to 2D ('TIME', 'SAMPLE') where we TIME is the 
             mean time of each ensemble and SAMPLE is each sample in the 
             ensemble. 
    lat, lon: Lat/lon of deployment (single point)
    include_raw_altimeter: Include raw altimeter signal if available.
                           (Typically on a single time grid) 
    FOM_ice_threshold: Threshold for "Figure of Merit" in the ice ADCP pings
                       used to separate measurements in ice from measurements
                       of water. 
    time_range: Only accept data within this date range (default: including 
                everything) 

                Give a tuple of date strings 'DD.MM.YYYY' 

                E.g.: time_range = [None, '21.08.2022']
                
    Output:
    -------

    xarray Dataset containing the data.

    '''

    # Get max/min times:
    date_fmt = '%d.%m.%Y' 
    if time_range[0]: 
        time_min = date2num(datetime.strptime(time_range[0], date_fmt))
    else: 
        time_min = None
    if time_range[1]: 
        time_max = date2num(datetime.strptime(time_range[1], date_fmt))
    else:
        time_max = None

    ###########################################################################
    # LOAD AND CONCATENATE DATA 

    first = True
    pressure_offsets = np.array([])

    if len(file_list)==0:
        raise Exception(
            'The *file_list* given to the function '
            'sig_funcs.matfiles_to_dataset() is empty.')

    for filename in file_list:
       
        dx, pressure_offset = _matfile_to_dataset(filename,
            lat = lat, lon = lon, 
            include_raw_altimeter = include_raw_altimeter)

        dx = dx.sel({'time_average':slice(time_min, time_max)})

        pressure_offsets = np.append(pressure_offsets, pressure_offset)

        if first: 
            DX = dx
            dx0 = dx.copy()
            first = False
        else:
            print('CONCATENATING: FILE "..%s"..\r'%(filename[-15:]), end = '') 
            try:
                DX = xr.concat([DX, dx], dim = 'time_average')
            except:
                print('Failed at %s'%(fn[-10:]))


    ###########################################################################

    if len(np.unique(pressure_offsets)) == 1:
        DX.attrs['pressure_offset'] = pressure_offsets[0]
    else:
        DX.attrs['pressure_offset'] = pressure_offsets

    # Add tilt (from pitch/roll)
    DX = _add_tilt(DX)

    # Sort by time
    DX = DX.sortby('time_average')

    # Reshape
    if reshape:
        DX = _reshape_ensembles(DX)

    # Add some attributes
    DX = set_lat(DX, lat)
    DX = set_lon(DX, lon)

    # Grab the (de facto) sampling rate
    DX.attrs['sampling_interval_sec'] = np.round(np.ma.median(
        np.diff(DX.time_average)*86400), 3)
    # Add FOM threshold
    DX['FOM_threshold'] = ((), FOM_ice_threshold, {'description':
        'Figure-of merit threshold used to separate ice vs open water'})

    # Add sea ice concetration estimate from FOM
    DX = _add_SIC_FOM(DX)

    # Add history attribute
    DX.attrs['history'] = ('- Loaded from .mat files on' 
        + ' %s'%datetime.now().strftime("%d %b %Y."))

    print('Done. Run sig_funcs.overview() to print some additional details.')


    return DX

##############################################################################

def chop(DX, indices = None, auto_accept = False):
    '''
    Chop a DX array with signature data (across all time-varying variables).

    Can specify the index or go with a simple pressure-based algorithm 
    (first/last indices within 3 SDs of the pressure median)

    DX: xarray Dataset containing signature data.
    indices: Tuple of indexes (start, stop), where start, stop are integers.
    auto_accept: Automatically accept chop suggested based on pressure record. 
    '''

    if not indices:
        p = DX.Average_AltimeterPressure.mean(dim = 'SAMPLE').data
        p_mean = np.ma.median(p)
        p_sd = np.ma.std(p)

        indices = [None, None]

        if p[0]<p_mean-3*p_sd:
            indices[0] = np.where(np.diff(p<p_mean-3*p_sd))[0][0]+1
        if p[-1]<p_mean-3*p_sd:
            indices[1] = np.where(np.diff(p<p_mean-3*p_sd))[0][-1]

        keep_slice = slice(*indices)
        if auto_accept:
            accept = 'y'
        else:
            fig, ax  = plt.subplots(figsize = (8, 4))
            index = np.arange(len(p))
            ax.plot(index, p, 'k')
            ax.plot(index[keep_slice], p[keep_slice], 'r')
            ax.set_xlabel('Index')
            ax.set_ylabel('Pressure [db]')
            ax.invert_yaxis()
            ax.set_title('Suggested chop: %s (to red curve).'%indices
                +' Close this window to continue..')
            plt.show(block=True)
            print('Suggested chop: %s (to red curve)'%indices)
            accept = input('Accept (y/n)?: ')

        if accept=='n':
            print('Not accepted -> Not chopping anything now.')
            print('NOTE: run sig_load.chop(DX, indices =[A, B]) to'
                ' manually set chop.')
            return DX
        elif accept=='y':
            pass
        else:
            raise Exception('I do not understand your input "%s".'%accept 
                + ' Only "y" or "n" works. -> Exiting.')
    else:
        keep_slice = slice(indices[0], indices[1]+1)

    L0 = DX.dims['TIME']
    print('Chopping to index %s'%indices)
    DX = DX.isel(TIME = keep_slice)
    L1 = DX.dims['TIME']
    net_str = 'Chopped %i ensembles using -> %s (total ensembles %i -> %i)'%(
            L0-L1, indices, L0, L1)
    print(net_str)
        
    DX.attrs['history'] += '\n- %s'%net_str
    return DX



##############################################################################

def overview(DX):
    '''
    Prints some basic information about the dataset.
    '''

    # Time range
    datefmt = '%d %b %Y %H:%M'
    starttime = num2date(DX.TIME[0]).strftime(datefmt)
    endtime = num2date(DX.TIME[-1]).strftime(datefmt)
    ndays = DX.TIME[-1]-DX.TIME[0]

    print('\nTIME RANGE:\n%s  -->  %s  (%.1f days)'%(
        starttime, endtime, ndays))
    print('Time between ensembles: %.1f min.'%(
        DX.time_between_ensembles_sec/60))
    print('Time between samples in ensembles: %.1f sec.'%(
        DX.sampling_interval_sec))

    # Pressure
    med_pres = np.ma.median(DX.Average_AltimeterPressure)
    std_pres = np.ma.std(DX.Average_AltimeterPressure)
    print('\nPRESSURE:\nMedian (STD) of altimeter pressure:'
          ' %.1f dbar (%.1f dbar)  - with fixed atm offset %.3f dbar.' %(
            med_pres, std_pres, DX.attrs['pressure_offset']))

    # Size
    print('\nSIZE:\nTotal %i time points.'%(DX.dims['TIME']*DX.dims['SAMPLE']))
    print('Split into %i ensembles with %i sample per ensemble.'%(
          DX.dims['TIME'], DX.dims['SAMPLE']))
    print('Ocean velocity bins: %i.'%(DX.dims['BINS']))

##############################################################################

def _reshape_ensembles(DX):
    '''
    Reshape all time series from a single 'time_average'
    dimension to 2D ('TIME', 'SAMPLE') where we TIME is the 
    mean time of each ensemble and SAMPLE is each sample in the 
    ensemble. 
    '''

    ###########################################################################
    # ADD A "time" COORDINATE (ONE ENTRY PER ENSEMBLE)

    Nt = len(DX.time_average)

    # Find indices where there is a time jump > 7 minutes
    time_jump_inds = np.where(np.diff(DX.time_average)*24*60>7)[0]

    # Start and end times of each ensemble
    ens_starts = np.concatenate([np.array([0]), time_jump_inds+1]) 
    ens_ends = np.concatenate([time_jump_inds, 
                           np.array([Nt-1])])

    # Use the mean time of each ensemble 
    t_ens = 0.5*(DX.time_average.data[ens_starts]
               + DX.time_average.data[ens_ends])
    Nsamp_per_ens = DX.samples_per_ensemble
    Nens = int(Nt/Nsamp_per_ens)

    if Nens != len(t_ens):
        warnings.warn('Expected number of ensembles (%i)'%Nens
        + 'is different from the number of ensembles deduced'
        + 'from time jumps > 7 minutes (%i).\n'%len(t_ens)
        + '!! This is likely to cause problems !!\n'
        + '(Check your time grid!)')

    print('%i time points, %i ensembles. Sample per ensemble: %i'%(
         Nt, Nens, Nsamp_per_ens))


    # NSW XARRAY DATASET  

    # Inheriting dimensions except time_average
    rsh_coords = dict(DX.coords)
    rsh_coords.pop('time_average') 

    # New coordinates: "TIME" (time stamp of each ensemble) and "SAMPLE" 
    # (number of samples within ensemble)

    rsh_coords['TIME'] = (['TIME'], t_ens, 
              {'units':'Days since 1970-01-01', 
               'long_name':('Time stamp of the ensemble averaged'
               ' measurement')})
    rsh_coords['SAMPLE'] = (['SAMPLE'], 
                np.int_(np.arange(1, Nsamp_per_ens+1)),
                {'units':'Sample number', 
               'long_name':('Sample number in ensemble '
               '(%i samples per ensemble)'%Nsamp_per_ens)})

    DXrsh = xr.Dataset(coords = rsh_coords)

    # Inherit attributes
    DXrsh.attrs = DX.attrs
    # Deleting and re-adding the instrument_configuration_details attribute 
    # (want it to be listed last)
    del DXrsh.attrs['instrument_configuration_details']
    DXrsh.attrs['instrument_configuration_details'] = (
        DX.attrs['instrument_configuration_details'])


    # Loop through variables, reshape where necessary
    for var_ in DX.variables:
        if DX[var_].dims == ('time_average',):

            DXrsh[var_] = (('TIME', 'SAMPLE'), 
                np.ma.reshape(DX[var_], (Nens, Nsamp_per_ens)),
                DX[var_].attrs)
        elif DX[var_].dims == ('BINS', 'time_average'):
            DXrsh[var_] = (('BINS', 'TIME', 'SAMPLE'), 
                    np.ma.reshape(DX[var_], (DX.dims['BINS'], 
                        Nens, Nsamp_per_ens)),
                    DX[var_].attrs)
        elif DX[var_].dims == ('time_average', 'xyz'):
            DXrsh[var_] = (('TIME', 'SAMPLE', 'xyz'), 
                    np.ma.reshape(DX[var_], (
                        Nens, Nsamp_per_ens, DX.dims['xyz'])),
                    DX[var_].attrs)
        elif DX[var_].dims == ('time_average', 'beams'):
            DXrsh[var_] = (('TIME', 'SAMPLE', 'beams'), 
                    np.ma.reshape(DX[var_], (
                        Nens, Nsamp_per_ens, DX.dims['beams'])),
                    DX[var_].attrs)


    return DXrsh

##############################################################################

def _matfile_to_dataset(filename, lat = None, lon = None,
                include_raw_altimeter = False):
    '''
    Read and convert single .mat file exported from SignatureDeployment.

    Wrapped into *matfiles_to_dataset*.


    Inputs:
    -------

    file_list: list of .mat files.
    lat, lon: Lat/lon of deployment (single point)
    include_raw_altimeter: Include raw altimeter signal if available.
                           (Typically on a single time grid) 

    Output:
    -------
    dx: xarray Dataset containing the data.
    pressure_offset: Pressure offset used in the data.
    '''

    b = _sig_mat_to_dict(filename)

    # OBTAIN COORDINATES
    coords = {
        'time_average':mat_to_py_time(b['Average_Time']),
        'BINS': np.arange(b['Average_VelEast'].shape[1]),
        'xyz': np.arange(3),
            }

    if include_raw_altimeter:
        try: # If we have AverageRawAltimeter: Add this as well  
            coords.update({'beams':np.arange(
                len(b['AverageRawAltimeter_BeamToChannelMapping'])),
                'along_altimeter':np.arange(
                    b['AverageRawAltimeter_AmpBeam5'].shape[1]),
                'raw_altimeter_time':mat_to_py_time(
                    b['AverageRawAltimeter_Time'])})
        except:
            print('No *AverageRawAltimeter*')

    # CREATE AN XARRAY DATASET TO BE FILLED     
    # Further down: concatenating into a joined dataset.
    dx = xr.Dataset(coords = coords)

    # Check whether we have AverageIce fields..
    try:
        dx['time_average_ice'] = (('time_average'), 
            mat_to_py_time(b['AverageIce_Time']))
    except:
        print('Did not find AverageIce..')

    # IMPORT DATA AND ADD TO XARRAY DATASET
    # Looping through variables. Assigning to dataset 
    # as fields with the appropriate dimensions.
    for key in b.keys():
        
        # AverageIce fields
        if 'AverageIce_' in key:
            if b[key].ndim==0:
                dx[key] = ((), b[key])
            elif b[key].ndim==1:
                dx[key] = (('time_average'), b[key])
            elif b[key].ndim==2:
                dx[key] = (('time_average', 'xyz'), b[key])
    
        # Average fields
        elif 'Average_' in key:
            if b[key].ndim==0:
                dx[key] = ((), b[key])
            elif b[key].ndim==1:
                if len(b[key]) == dx.dims['time_average']:
                    dx[key] = (('time_average'), b[key])
                else:
                    dx[key] = (('beams'), b[key])
            elif b[key].ndim==2:
                if b[key].shape[1] == dx.dims['xyz']:
                    dx[key] = (('time_average', 'xyz'), b[key])
                elif b[key].shape[1] == dx.dims['BINS']:
                    dx[key] = (('BINS', 'time_average'), 
                            b[key].T)

        # AverageRawAltimeter fields
        elif 'AverageRawAltimeter_' in key:
            if include_raw_altimeter:
                if b[key].ndim==0:
                    dx[key] = ((), b[key])
                    
                elif b[key].ndim==1:
                    if len(b[key]) == dx.dims['beams']:
                        dx[key] = (('beams'), b[key])
                    else:
                        dx[key] = (('time_raw_altimeter'), b[key])
                elif b[key].ndim==2:
                    if b[key].shape[1] == dx.dims['xyz']:
                        dx[key] = (('time_raw_altimeter', 'xyz'), b[key])
                    elif b[key].shape[1] == dx.dims['along_altimeter']:
                        dx[key] = (('along_altimeter', 
                                    'time_raw_altimeter'), b[key].T)


    # ASSIGN METADATA ATTRIBUTES
    # Units, description etc
    dx.time_average.attrs['description'] = ('Time stamp for'
     ' "average" fields. Source field: *Average_Time*. Converted'
    ' using sig_funcs.mat_to_py_time().') 
    dx.BINS.attrs['description'] = ('Number of velocity bins.') 
    dx.beams.attrs['description'] = ('Beam number (not 5th).') 
    dx.xyz.attrs['description'] = ('Spatial dimension.') 
    
    if include_raw_altimeter:
        dx.time_raw_altimeter.attrs['description'] = ('Time stamp for'
            ' "AverageRawAltimeter" fields. Source field:'
            ' *AverageRawAltimeter_Time*. Converted'
            ' using mat_to_py_time.') 
        dx.along_altimeter.attrs['description'] = ('Index along altimeter.') 

    # Read the configuration info to a single string (gets cumbersome to 
    # carry these around as attributes..) 
    conf_str = ''
    for conf_ in b['conf']:
        if conf_ in b['units'].keys():
            unit_str = ' %s'%b['units'][conf_]
        else:
            unit_str = ''
        if conf_ in b['desc'].keys():
            desc_str = ' (%s)'%b['desc'][conf_]
        else:
            desc_str = ''

        conf_str += '%s%s: %s%s\n'%(conf_, desc_str, 
                         b['conf'][conf_], unit_str, )
    dx.attrs['instrument_configuration_details'] = conf_str

    # Add some selected attributes that are useful
    dx.attrs['instrument'] = b['conf']['InstrumentName']
    dx.attrs['serial_number'] = b['conf']['SerialNo']
    dx.attrs['samples_per_ensemble'] = int(b['conf']['Average_NPings'])
    dx.attrs['time_between_ensembles_sec'] = int(
        b['conf']['Plan_ProfileInterval'])
    dx.attrs['blanking_distance_oceanvel'] = b['conf'][
                                    'Average_BlankingDistance']
    dx.attrs['cell_size_oceanvel'] = b['conf']['Average_CellSize']
    dx.attrs['N_cells_oceanvel'] = b['conf']['Average_NCells']

    # Read pressure offset
    pressure_offset = b['conf']['PressureOffset']

    return dx, pressure_offset


##############################################################################

def _sig_mat_to_dict(matfn, include_metadata = True, squeeze_identical = True,
                    skip_Burst = True, skip_IBurst = True, 
                    skip_Average = False,  skip_AverageRawAltimeter = False, 
                    skip_AverageIce = False, 
                    skip_fields = []):
    '''
    Reads matfile produced by SignatureDeployment to numpy dictionary.

    include_metadata = False will skip the config, description, and units
    fields. 
    
    squeeze_identical = True will make data arrays of identical entries into
    one single entry - e.g. we just need one entry, [N], showing the number of
    bins, not one entrye per time stamp [N, N, ..., N]. 
    '''

    # Load the mat file
    dmat = loadmat(matfn)

    # Create a dictionary that we will fill in below
    d = {}

    # Add the contents of the non-data fields
    for indict_key, outdict_key in zip(['Config', 'Descriptions', 'Units'],
                                ['conf', 'desc', 'units']):
        outdict = {}
        for varkey in dmat[indict_key].dtype.names:
            outdict[varkey] = _unpack_nested(dmat[indict_key][varkey])
        d[outdict_key] = outdict

    # Making a list of strings based on the *skip_* booleans in the function
    # call. *startsrtings_skip* contains a start strings. If a variable starts
    # with any of these srtings, we won't include it.
    
    startstrings = ['Burst_', 'IBurst_', 'Average_', 
                'AverageRawAltimeter_', 'AverageIce_', ]
    startstrings_skip_bool = [skip_Burst, skip_IBurst, skip_Average,
                skip_AverageRawAltimeter, skip_AverageIce,]
    startstrings_skip = tuple([str_ for (str_, bool_) in zip(startstrings, 
                        startstrings_skip_bool) if bool_])

    # Add the data. Masking NaNs and squeezing/unpacking unused dimensions
    for varkey in dmat['Data'].dtype.names:
        if (not varkey.startswith(startstrings_skip) 
           and varkey not in skip_fields):
           
            d[varkey] = np.ma.squeeze(np.ma.masked_invalid(
                    _unpack_nested(dmat['Data'][varkey])))
            
            if (d[varkey] == d[varkey][0]).all() and squeeze_identical:
                d[varkey] = d[varkey][0]

    # Return the dictionary
    return d


##############################################################################

def _unpack_nested(val):
    '''
    Unpack nested list/arrays down to the base level.
    Need this because data in the SignatureDeployment matfiles end up in
    weird nested structures.
    
    E.g. [['s']] -> 's'
    E.g. [[[[[1, 2], [3, 4]]]] -> [[1, 2], [3, 4]]
    '''
    
    unpack = True
    while unpack == True:
        if hasattr(val, '__iter__') and len(val) == 1:
            val = val[0]
            if isinstance(val, str):
                unpack = False
        else:
            unpack = False
    return val

##############################################################################
def to_nc(DX, file_path, export_vars = [], icedraft = True, icevel = True, 
            oceanvel = False, all = False,):
    '''
    Export to netCDF file.

    Inputs
    ------
    file_path: Path of output file. 
    export_vars: List of variables to influde in the exported ncfile.
    icedraft: Include sea ice draft estimates.
    icevel: Include sea ice drift velocities.
    oceanvel: Include ocean velocities.
    all: Include *all* variables, including everything in the original source 
         files.
    '''

    DXc = DX.copy()

    if all:
        print('Saving *ALL* variables..')
        # Saving
        DXc.to_netcdf(file_path)
        print('Saved data to file:\n%s'%file_path)
    else:
        varlist = export_vars.copy()
        if icedraft:
            varlist += ['SEA_ICE_DRAFT_LE', 'SEA_ICE_DRAFT_MEDIAN_LE', 
                'SEA_ICE_DRAFT_AST', 'SEA_ICE_DRAFT_LE']
        if icevel:
            varlist += ['uice', 'vice', 'Uice', 'Vice']
        if oceanvel:
            varlist += ['uocean', 'vocean', 'Uocean', 'Vocean']

        # Remove any duplicates
        varlist = list(np.unique(np.array(varlist)))

        # Remove any variables not in DX
        for varnm_ in varlist.copy():
            if varnm_ not in list(DXc.keys()):
                varlist.remove(varnm_)
                if varnm_ in export_vars:
                    warnings.warn('No field %s found in the data '%varnm_
                     + '(exporting file without it). Check spelling?')

        # If varlist is empty: Print a note and exit
        if varlist == []:
            print('No existing variables selected -> Not exporting anything.')
            return None

        # Delete some none-useful attributes
        for attr_not_useful in ['pressure_offset']:

            del DXc.attrs[attr_not_useful]

        # Saving
        DXc[varlist].to_netcdf(file_path)
        print('Saved data to file:\n%s'%file_path)
