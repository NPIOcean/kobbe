'''
Various calculations done on a xarray Dataset with signature data.
'''

import numpy as np
import gsw
import warnings

def dep_from_p(DX, corr_atmo = True, corr_CTD_density = True):
    '''
    Calculate depth from:
    
    - Absolute pressure (measured by instrument)
        - Using *Average_AltimeterPressure*
          (very rarely differs from *Average_Pressure* 
           by more than ±5e-3 db).
    - Atmospheric pressure (from *p_atmo* field)
    - Gravitational acceleration (calculated from latitude)
    - Ocean density (from data or default 1025 g/kg)

    Input:
    ------

    DX: xarray Dataset with Signature data.

    Returns:
    --------
    DX where the field "DEPTH" (TIME, SAMPLE) has been added.
    '''

    note_str = ('Altimeter depth calculated from pressure'
        ' (*Average_AltimeterPressure* field) as:\n\n '
        '   depth = p / (g * rho)\n')

    # CALCULATE ABSOLUTE PRESSURE
    p_abs = DX.Average_AltimeterPressure + DX.pressure_offset

    # CALCULATE OCEAN PRESSURE
    # Raising issues if we cannot find p_atmo (predefined atmospheric pressure)
    if hasattr(DX, 'p_atmo') and corr_atmo:
        p_ocean = (p_abs - DX.p_atmo).data
        note_str += '\n- Atmospheric pressure (*p_atmo* field subtracted).'
    else:
        if corr_atmo:
            warn_str1 = ('WARNING!\nCould not find atmospheric pressure '
            '(*p_atmo*) - not recommended continue if you plan to compute ice'
            ' draft. \n--> (To add *p_atmo*, run sig_append.append_atm_'
            'pres()=\n\nDepth calculation: Abort (A) or Continue (C): ')

            user_input_abort = input(warn_str1).upper()

            while user_input_abort not in ['A', 'C']:
                print('Input ("%s") not recognized.'%user_input_abort)
                user_input_abort = input(
                    'Enter "C" (continue) or "A" (abort): ').upper()

            if user_input_abort == 'A':
                raise Exception(
                    'ABORTED BY USER (MISSING ATMOSPHERIC PRESSURE)')
        
        else:
            user_input_abort = 'C'

        p_ocean = DX.Average_AltimeterPressure.data
        print('Continuing without atmospheric correction (careful!)..')
        note_str += (
            '\n- !!! NO TIME_VARYING ATMOSPHERIC CORRECTION APPLIED !!!\n' 
            '  (using default atmospheric pressure offset'
            ' %.2f db)'%DX.pressure_offset)
            

    # CALCULATE GRAVITATIONAL ACCELERATION
    if DX.lat==None:
        raise Exception('No "lat" field in the dataset. Add one using'
                 ' sig_append.set_lat() and try again.')

    g = gsw.grav(DX.lat.data, 0)
    DX['g'] = ((), g, {'units':'ms-2', 
        'note':'Calculated using gsw.grav() for p=0 and lat=%.2f'%DX.lat}) 
    note_str += '\n- Using g=%.4f ms-2 (calculated using gsw.grav())'%g


    # CALCULATE OCEAN WATER DENSITY
    if hasattr(DX, 'rho_CTD') and corr_CTD_density:
        rho_ocean = DX.rho_CTD.data
        note_str += '\n- Using ocean density from the *rho_CTD* field.'
        fixed_rho = False
    else:
        if corr_CTD_density:
            print('\nNo density (*rho_ocean*) field found. ')
            user_input_abort_dense = input('Enter "A" (Abort) or "C" '
            '(Continue using fixed rho = 1027 kg m-3): ').upper()
        
            while user_input_abort_dense not in ['A', 'C']:
                print('Input ("%s") not recognized.'%user_input_abort)
                user_input_abort_dense = input(
                    'Enter "C" (continue with fixed) or "A" (abort): ').upper()
            if user_input_abort_dense == 'A':
                raise Exception('ABORTED BY USER (MISSING OCEAN DENSITY)')

        rho_input = input('Continuing using fixed rho. Choose: \n'
                        '(R): Use rho = 1027 kg m-3, or\n' 
                       '(S): Specify fixed rho\n').upper()
        while rho_input not in ['R', 'S']:
            print('Input ("%s") not recognized.'%rho_input)
            rho_input = input(
                'Enter "R" (fixed rho = 1027) or "S" (specify): ').upper()
        
        if rho_input=='R':
            rho_ocean = 1027
        else:
            rho_ocean = np.float(input('Enter rho (kg m-3): '))
        fixed_rho = True

        print('Continuing with fixed rho = %.1f kg m-3'%rho_ocean)
        note_str += '\n- Using FIXED ocean rho = %.1f kg m-3.'%rho_ocean


    # CALCULATE DEPTH
    # Factor 1e4 is conversion db -> Pa
    if fixed_rho:
        depth = 1e4*p_ocean/g/rho_ocean
    else:
        depth = 1e4*p_ocean/g/rho_ocean[:, np.newaxis]
    
    DX['depth'] = (('TIME', 'SAMPLE'), depth, {'units':'m', 
        'long_name':'Transducer depth', 'note':note_str}) 

    return DX



##############################################################################

def mat_to_py_time(mattime):
    '''
    Convert matlab datenum (days) to Matplotlib dates (days).

    MATLAB base: 00-Jan-0000
    Matplotlib base: 01-Jan-1970
    '''

    mpltime = mattime - 719529.0

    return mpltime


##############################################################################

def daily_average(A, t, td = None, axis = -1, min_frac = 0, 
                function = 'median'):
    '''
    Take a time series A on a time grid t and compute daily averages of A.

    If day index *td* is not specified, it will be computed based on t. 

    A can be 1- or 2-dimensional, but the time axis must be the last axis.

    min_frac: Required non-masked fraction (return nan otherwise)

    Returns:
    --------

    td: Day index (note: not the "mean time" - this would be approx. td+0.5)
    Ad: Daily means
    '''

    tfl = np.floor(t)
    if td is None:
        td = np.ma.unique(tfl)

    Nt = len(td)

    if A.ndim == 1:
        Ad_shape = (Nt)
    elif A.ndim == 2:
        Ad_shape = (A.shape[0], Nt)
    else:
        raise Exception('''
        *daily_median()* only works for 1D or 2D arrays.
        ''')

    Ad = np.zeros(Ad_shape)*np.nan

    for nn in np.arange(Nt):
        tind = tfl==td[nn]
        if tind.any():
         #   print( (sum(np.isnan(A[..., tind]))/len(A[..., tind])).data,  
          #        (sum(np.isnan(A[..., tind]))/len(A[..., tind])).data<1-min_frac)
            if sum(np.isnan(A[..., tind]))/len(A[..., tind]) <1-min_frac:
                    with warnings.catch_warnings(): # np.nanmedian issues a warning for all-nan ensembles
                                            # (we don't need to see it, so we suppress warnings 
                                            # for this operation)
                        warnings.simplefilter("ignore", category=RuntimeWarning)
                        if function == 'median':
                            Ad[..., nn] = np.nanmedian(A[..., tind], axis = -1)
                        elif function == 'mean':
                            Ad[..., nn] = np.nanmean(A[..., tind], axis = -1)

    return Ad, td

##############################################################################


def runningstat(A, window_size):
    """ 
    Calculate running statistics (mean, median, sd).

    Note: Reflects at the ends - may have to modify the fringes of
    the time series for some applications..

    Based on script by Maksym Ganenko on this thread:
    https://stackoverflow.com/questions/33585578/
    running-or-sliding-median-mean-and-standard-deviation   

    Inputs
    ------
    A: Equally spaced time series
    window_size: Window size for boxcar (must be odd)

    Returns
    -------

    RS: Dictionary containing 'mean', 'median' and 'std'

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
    
    RS = {'median' : np.median(A[index], axis = 1),
          'mean' : np.mean(A[index], axis = 1),
          'std' : np.std(A[index], axis = 1)}

    return RS

##### 

def clean_nanmedian(a, **kwargs):
    '''
    Wrapper for the np.nanmedian function, but ignoring the annoying  
    RuntimeWarning when trynig to get the nanmedian of an all-NaN slice.

    (Retaining default behaviour of returning NaN as the median of 
    all-NaN slices)
    '''
    with warnings.catch_warnings():
        warnings.filterwarnings(action='ignore', 
                                message='All-NaN slice encountered')
        return np.nanmedian(a, **kwargs)