'''
SIG_VEL.PY

Functions for processing ocean and sea ice drift velocity.
'''

import numpy as np
import xarray as xr
from scipy.interpolate import interp1d
import warnings 

def calculate_ice_vel(DX, avg_method = 'median'):
    '''
    Calculate sea ice drift velocity from the AverageIce 
    '''

    DX['uice'] = (('TIME', 'SAMPLE'), DX.AverageIce_VelEast.data, 
        {'units':'m s-1', 'long_name':'Eastward sea ice drift velocity',
        'details': 'All average mode samples'})
    DX['vice'] = (('TIME', 'SAMPLE'), DX.AverageIce_VelNorth.data, 
        {'units':'m s-1', 'long_name':'Northward sea ice drift velocity',
        'details': 'All average mode samples'})

    DX['uice'] = DX['uice'].where(DX.ICE_IN_SAMPLE)
    DX['vice'] = DX['vice'].where(DX.ICE_IN_SAMPLE)

    for key in ['uice', 'vice']:
        DX[key].attrs['processing_history'] = (
            'Loaded from AverageIce_VelEast/AverageIce_VelEast fields.\n')

    DX = _calculate_uvice_avg(DX, avg_method = avg_method)

    return DX 



def _calculate_uvice_avg(DX, avg_method = 'median'):
    '''
    Calculate ensemble average ocean velocity
    '''

    if avg_method=='median':
        DX['Uice'] = DX['uice'].median(dim = 'SAMPLE')
        DX['Vice'] = DX['vice'].median(dim = 'SAMPLE')

    elif avg_method=='mean':
        DX['Uice'] = DX['uice'].mean(dim = 'SAMPLE')
        DX['Vice'] = DX['vice'].mean(dim = 'SAMPLE')
    else:
        raise Exception('Invalid "avg_method" ("%s"). '%avg_method
         + 'Must be "mean" or "median".')

    DX.Uice.attrs = {'units':'m s-1', 
        'long_name':'Eastward sea ice drift velocity', 
        'details': 'Ensemble average (%s)'%avg_method, 
        'processing_history' : DX.uice.processing_history}  
    DX.Vice.attrs = {'units':'m s-1', 
        'long_name':'Northward sea ice drift velocity', 
        'details': 'Ensemble average (%s)'%avg_method, 
        'processing_history' : DX.vice.processing_history} 

    with warnings.catch_warnings(): # Suppressing (benign) warning yielded 
        # when computing std() over all-nan slice..
        warnings.filterwarnings(action='ignore', 
            message='Degrees of freedom <= 0 for slice')
        DX['Uice_SD'] = DX['uice'].std(dim = 'SAMPLE', skipna = True)
        DX['Vice_SD'] = DX['vice'].std(dim = 'SAMPLE', skipna = True)

    DX.Uice_SD.attrs = {'units':'m s-1', 
        'long_name':('Ensemble standard deviation of '
                'eastward sea ice drift velocity'), } 
    DX.Vice_SD.attrs = {'units':'m s-1', 
        'long_name':('Ensemble standard deviation of '
                'northward sea ice drift velocity'), }
    
    return DX

def calculate_ocean_vel(DX, avg_method = 'median'):
    '''
    Calculate sea ice drift velocity from the Average_VelEast/Average_VelNorth
    fields.
    '''
    # Calculate bin depths
    DX = _calculate_bin_depths(DX)

    # Extract u, v, data
    DX['uocean'] = (('BINS', 'TIME', 'SAMPLE'), DX.Average_VelEast.data, 
        {'units':'m s-1', 'long_name':'Eastward ocean velocity',
        'details': 'All average mode samples'})
    DX['vocean'] = (('BINS', 'TIME', 'SAMPLE'), DX.Average_VelNorth.data, 
        {'units':'m s-1', 'long_name':'Northward ocean velocity',
        'details': 'All average mode samples'})

    for key in ['uocean', 'vocean']:
        DX[key].attrs['details'] = 'All average mode samples'
        DX[key].attrs['units'] = 'm s-1'
        DX[key].attrs['processing_history'] = (
            'Loaded from Average_VelEast/Average_VelEast fields.\n')

    # Calculate sample averages
    DX = _calculate_uvocean_avg(DX, avg_method = avg_method )

    return DX 




def _calculate_uvocean_avg(DX, avg_method = 'median', min_good_pct = False):
    '''
    TBD Document! (min_good_pct)
    '''

    if avg_method=='median':
        DX['Uocean'] = DX.uocean.median(dim = 'SAMPLE')
        DX['Vocean'] = DX.vocean.median(dim = 'SAMPLE')
    elif avg_method=='mean':
        DX['Uocean'] = DX.uocean.mean(dim = 'SAMPLE')
        DX['Vocean'] = DX.vocean.mean(dim = 'SAMPLE')
    else:
        raise Exception('Invalid "avg_method" ("%s"). '%avg_method
         + 'Must be "mean" or "median".')

    if min_good_pct:
        N_before = np.sum(~np.isnan(DX.Uocean))
        good_ind = np.isnan(DX.uocean).mean(dim='SAMPLE')<1-min_good_pct/100
        N_after = np.sum(good_ind)
        min_good_str = (
            '\nRejected %i of %i ensembles (%.2f%%) with <%.1f%% good samples.'%(
                N_before-N_after, N_before, (N_before-N_after)/N_before*100,
                min_good_pct))
        DX['Uocean'] = DX.Uocean.where(good_ind)
        DX['Vocean'] = DX.Vocean.where(good_ind)
    
    else:
        min_good_str = ''


    DX.Uocean.attrs = {'units':'m s-1', 'long_name':'Eastward ocean velocity', 
        'details': 'Ensemble average (%s)'%avg_method, 
        'processing_history':DX.uocean.processing_history + min_good_str} 
    DX.Vocean.attrs = {'units':'m s-1', 'long_name':'Northward ocean velocity', 
        'details': 'Ensemble average (%s)'%avg_method, 
        'processing_history':DX.vocean.processing_history + min_good_str} 



    return DX


def _calculate_bin_depths(DX):
    '''
    Calculate time-varying depth of ocean velocity bins.

    From Nortek doc "Signature Principles of Operation":
        n-th cell is centered at a vertical distance from the transducer
        equal to: Center of n'th cell = Blanking + n*cell size

    '''

    dist_from_transducer = (DX.blanking_distance_oceanvel 
            + DX.cell_size_oceanvel*(1+np.arange(DX.N_cells_oceanvel)))

    DX['bin_depth'] = (DX.depth.mean(dim='SAMPLE').expand_dims(
                                dim = {'BINS':DX.dims['BINS']}) 
                       - dist_from_transducer[:, np.newaxis])
    DX['bin_depth'].attrs = {
            'long_name':'Sample-average depth of velocity bins',
            'units':'m', 
            'note':('Calculated as:\n\n'
            '   bin_depth = instr_depth - (blanking depth + n*cell size)\n\n'
            'where *n* is bin number and *inst_depth* the (sample-mean) depth'
            ' of the transducer.')}      

    return DX



def uvoc_mask_range(DX, uv_max = 1.5, tilt_max = 5, 
        sspd_range = (1400, 1560), cor_min = 60,
        amp_range = (30, 85), max_amp_increase = 20):
    '''
    Set instances of 

    Setting uocean, vocean data points to NaN where: 

    1. Speed exceeds *uv_max* (m/s).
    2. Tilt exceeds *tilt_max* (degrees).
    3. Instrument-recorded sound speed is outside *sspd_range* (ms-1).
    4. Any one of the beam correlations is below *cor_min* (percent).
    5. Any one of the beam amplitudes are outside *amp_range* (db).
    6. There is a bin-to-bin amplitude jump of *max_amp_increase* (db)
       in any of the beams.
       - Also masking all beams *above* the jump. 

    '''
    # N_variables used for counting the effect of each step.
    N_start = np.float(np.sum(~np.isnan(DX.uocean)).data)

    # Create DX_uv; a copy of DX containing only uocean, vocean.
    # Then feeding these back into DX before returning.
    # (This is because we dont want the DX.where() operation
    # to affect other fields/expand dimensions unrelated to ocean 
    # velocities)
    DX_uv = DX[['uocean', 'vocean']]

    # Speed test
    DX_uv = DX_uv.where((DX.uocean**2 + DX.vocean**2)<uv_max**2)
    N_speed = np.float(np.sum(~np.isnan(DX_uv.uocean)).data)

    # Tilt test
    DX_uv = DX_uv.where(DX.tilt_Average<tilt_max)
    N_tilt = np.float(np.sum(~np.isnan(DX_uv.uocean)).data)
    
    # Sound speed test
    DX_uv = DX_uv.where((DX.Average_Soundspeed>sspd_range[0]) 
        & (DX.Average_Soundspeed<sspd_range[1]))
    N_sspd = np.float(np.sum(~np.isnan(DX_uv.uocean)).data)

    # Correlation test
    DX_uv = DX_uv.where((DX.Average_CorBeam1>cor_min) 
                | (DX.Average_CorBeam2>cor_min)
                | (DX.Average_CorBeam3>cor_min)
                | (DX.Average_CorBeam4>cor_min))
    N_cor = np.float(np.sum(~np.isnan(DX_uv.uocean)).data)

    # Amplitude test
    # Lower bound
    DX_uv = DX_uv.where((DX.Average_AmpBeam1>amp_range[0]) 
                | (DX.Average_AmpBeam2>amp_range[0])
                | (DX.Average_AmpBeam3>amp_range[0])
                | (DX.Average_AmpBeam4>amp_range[0]))
    # Upper bound
    DX_uv = DX_uv.where((DX.Average_AmpBeam1<amp_range[1]) 
                | (DX.Average_AmpBeam2<amp_range[1])
                | (DX.Average_AmpBeam3<amp_range[1])
                | (DX.Average_AmpBeam4<amp_range[1]))

    N_amp = np.float(np.sum(~np.isnan(DX_uv.uocean)).data)

    # Amplitude bump test

    # Find bumps from *diff* in the BIN S dimension
    is_bump = (
        (DX.Average_AmpBeam1.diff(dim = 'BINS')>max_amp_increase)
        | (DX.Average_AmpBeam2.diff(dim = 'BINS')>max_amp_increase)
        | (DX.Average_AmpBeam3.diff(dim = 'BINS')>max_amp_increase)
        | (DX.Average_AmpBeam4.diff(dim = 'BINS')>max_amp_increase))
    
    # Create a boolean (*True* above bumps)
    zeros_firstbin = xr.zeros_like(DX.uocean.isel(BINS=0))
    NOT_ABOVE_BUMP = xr.concat([zeros_firstbin, is_bump.cumsum(axis = 0)>0], 
                        dim = ('BINS'))<1
    DX_uv = DX_uv.where(NOT_ABOVE_BUMP)
    N_amp_bump = np.float(np.sum(~np.isnan(DX_uv.uocean)).data)


    proc_string = ('\nTHRESHOLD_BASED DATA CLEANING : '
     + '\nStart: %i initial valid samples.\n'%N_start
     + 'Dropping (NaNing samples where):\n'
     + '- # Speed < %.2f ms-1 # -> Dropped %i pts (%.2f%%)\n'%(
        uv_max, N_start-N_speed, (N_start-N_speed)/N_start*100)
     + '- # Tilt < %.2f deg # -> Dropped %i pts (%.2f%%)\n'%(
        tilt_max, N_speed-N_tilt, (N_speed-N_tilt)/N_speed*100)       
     + '- # Sound sp in [%.0f, %.0f] ms-1 # -> Dropped %i pts (%.2f%%)\n'%(
        *sspd_range, N_tilt-N_sspd, (N_tilt-N_sspd)/N_tilt*100)    
     + '- # Corr (all beams) < %.1f %% # -> Dropped %i pts (%.2f%%)\n'%(
        cor_min, N_sspd-N_cor, (N_sspd-N_cor)/N_sspd*100) 
     + '- # Amp (all beams) in [%.0f, %.0f] db # -> Dropped %i pts (%.2f%%)\n'%(
        *amp_range, N_cor-N_amp, (N_cor-N_amp)/N_cor*100)    
     + '- # Above amp bumps > %.0f db # -> Dropped %i pts (%.2f%%)\n'%(
        max_amp_increase, N_amp-N_amp_bump, (N_amp-N_amp_bump)/N_amp*100)   
     + 'End: %i valid samples.\n'%N_amp_bump 
        )

    for key in ['uocean', 'vocean']:
        DX[key] = DX_uv[key] 
        DX[key].attrs['processing_history'] += proc_string

    # Recompute sample averages
    DX = _calculate_uvocean_avg(DX)

    return DX



def rotate_vels_magdec(DX):
    '''
    Rotate ocean and ice velocities to account for magnetic declination.

    Only rotates processed fields (uocean, vocean, uice, vice) -
    not raw variables (Average_VelNorth, AverageIce_VelNorth, etc..).
    '''
    
    assert hasattr(DX, 'magdec'), ("Didn't find magnetic declination (no" 
        " *magdec* attribute). Run sig_append.append_magdec()..")

    DX0 = DX.copy()

    # Convert to radians
    magdec_rad = DX.magdec * np.pi/180

    # Make a documentation string (to add as attribute)
    magdec_mean = DX.magdec.mean().data
    magdec_str = ('Rotated CW by an average of %.2f degrees'%magdec_mean
                  +  ' to correct for magnetic declination. ')

    # Loop through different (u, v) variable pairs and rotate them
    uvpairs = [('uice', 'vice'), ('uocean', 'vocean'),] #
              #  ('Uice', 'Vice'), ('Uocean', 'Vocean')]

    uvstrs = ''

    for uvpair in uvpairs:

        if hasattr(DX, uvpair[0]) and hasattr(DX, uvpair[1]):
            uvc0_ = DX[uvpair[0]] + 1j* DX[uvpair[1]]
            uvc_rot = uvc0_ * np.exp(-1j*magdec_rad)

            DX[uvpair[0]].data = uvc_rot.real
            DX[uvpair[1]].data = uvc_rot.imag
            for key in uvpair:
                DX[key].attrs['processing_history'] += magdec_str+'\n'
            uvstrs += '\n - (%s, %s)'%uvpair

    if hasattr(DX, 'declination_correction'):
        inp_yn = float(input('Declination correction rotation has been '
        + 'applied to something before. \n -> Continue '
        + '(1) or skip new correction (0): '))

        if inp_yn ==1:
            print('-> Applying new correction.')
            DX.attrs['declination_correction'] = (
                '!! NOTE !! Magnetic declination correction has been applied more'
                ' than once - !! CAREFUL !!\n' 
                + DX.attrs['declination_correction'])
        else:
            print('-> NOT applying new correction.')
            return DX0
    else:
        DX.attrs['declination_correction'] = (
          'Magdec declination correction rotation applied to: %s'%uvstrs)
    try:
        DX = _calculate_uvocean_avg(DX, avg_method = 'median') 
    except:pass
    try:
        DX = _calculate_uvice_avg(DX, avg_method = 'median') 
    except:pass

    return DX


def clear_empty_bins(DX, thr_perc=5):
    '''
    Remove BINS where less than thr_perc of the bin contains valid
    (non-NaN) samples.
    '''
    # Find indices of empty bins
    empty_bins = np.where(
        np.isnan(DX.Uocean).mean('TIME')*100>(100-thr_perc))[0]
    # Count
    Nbins_orig = DX.dims['BINS']
    Nbins_drop = len(empty_bins)

    # Drop from dataset
    DX = DX.drop_sel(BINS = empty_bins)
    # Note in history
    DX.attrs['history'] += (
        '\nDropped %i of %i bins where'%(Nbins_drop, Nbins_orig)
        + ' less than %.1f%% of samples were'%(thr_perc)
        + '  valid. -> Remaining bins: %i'%(DX.dims['BINS']))
    return DX


def reject_sidelobe(DX):
    '''
    Mask samples where we expect sidelobe interference.

    From Nortek documentation
    (nortekgroup.com/assets/software/N3015-011-SignaturePrinciples.pdf):
    Maximum range Rmax is given by: 

    (1)    Rmax = A cos(θ) - s_c

    Where A is the distance between transducer and surface, θ the beam 
    angle (assumed to be 25 deg for Signatures), and s_c the velocity cell 
    size.

    We calculate A as:

    (2)    A = DEP - ICE_DRAFT

    Where DEPTH is sample-mean depth and ICE_DRAFT is sample-mean
    sea ice draft (SEA_ICE_DRAFT_MEDIAN_LE if available, otherwise 0).

    In (2), we use sample mean tilt for θ.

    '''

    if 'depth' in DX.keys():
        DEP = DX.depth.mean(dim='SAMPLE')
    else:
        raise Exception('No "depth" field present. -> cannot reject '
        'measurements influenced by sidelobe interference. ' 
        'Run sig_calc.dep_from_p() first.')
    
    if 'SEA_ICE_DRAFT_LE' in DX.keys():
        ICE_DRAFT = DX.SEA_ICE_DRAFT_MEDIAN_LE.copy().fillna(0)

    else:
        ICE_DRAFT = 0
    
    A = DEP - ICE_DRAFT
    cos_theta = np.cos(25*np.pi/180.0)
    s_c = DX.cell_size_oceanvel

    Rmax = A * cos_theta - s_c

    # Make a copy with only velocities
    DX_uv = DX[['uocean', 'vocean']]
    # NaN instances where bin depth is less than 
    #   DEP-Rmax
    DX_uv = DX_uv.where(DX.bin_depth > (DEP-Rmax).expand_dims(
                                dim = {'BINS':DX.dims['BINS']}))
    
    # Feed the NaNed (uocean, vocean) fields back into DX
     
    N_before = np.sum(~np.isnan(DX.uocean)) # Count samples
    for key in ['uocean', 'vocean']:
        DX[key] = DX_uv[key] 
    N_after = np.sum(~np.isnan(DX.uocean)) # Count samples

    # + Add processing history   
    proc_string = ('\nRejected samples close enough to the surface '
        'to be affected by sidelobe interference (rejecting '
        '%.2f%% of velocity samples).'%((1-N_after/N_before)*100))
 
    for key in ['uocean', 'vocean']:
        DX[key].attrs['processing_history'] += proc_string

    # Recompute sample averages
    DX = _calculate_uvocean_avg(DX)
    return DX


def interp_oceanvel(DX, ip_depth):
    '''
    Interpolate Uocean, Vocean onto a fixed depth *ip_depth*
    '''

    U_IP = DX.Uocean.mean('BINS', keep_attrs = True).copy()
    V_IP = DX.Vocean.mean('BINS', keep_attrs = True).copy()
    U_IP[:] = np.nan
    V_IP[:] = np.nan

    for nn in np.arange(DX.dims['TIME']):
        ip_ = interp1d(DX.bin_depth.isel(TIME=nn),
                    DX.Uocean.isel(TIME=nn),bounds_error=False,
                    fill_value=np.nan)
        U_IP[nn] = ip_(ip_depth)
        
        if nn/10 == nn//10:
            print('Interpolating "Uocean" (%.1f%%)...\r'%(
                100*nn/DX.dims['TIME']), end = '')
    print('Interpolating "Uocean": *DONE*     \r', end = '')

    for nn in np.arange(DX.dims['TIME']):
        ip_ = interp1d(DX.bin_depth.isel(TIME=nn),
                    DX.Vocean.isel(TIME=nn),bounds_error=False,
                    fill_value=np.nan)
        V_IP[nn] = ip_(ip_depth)
        
        if nn/10 == nn//10:
            print('Interpolating "Vocean" (%.1f%%)...\r'%(
                100*nn/DX.dims['TIME']), end = '')
    print('Interpolating "Vocean": *DONE*     \r', end = '')

    V_IP_name = 'Vocean_%im'%(np.round(ip_depth))
    DX[V_IP_name] = V_IP
    DX[V_IP_name].attrs['long_name'] = (DX.Vocean.attrs['long_name'] 
                            + ' interpolated to %.1f m depth'%ip_depth)
    DX[V_IP_name].attrs['processing_history'] = (
        DX.Vocean.attrs['processing_history'] 
        + '\nInterpolated to %.1f m depth.'%ip_depth)

    U_IP_name = 'Uocean_%im'%(np.round(ip_depth))
    DX[U_IP_name] = U_IP
    DX[U_IP_name].attrs['long_name'] = (DX.Uocean.attrs['long_name'] 
                            + ' interpolated to %.1f m depth'%ip_depth)
    DX[U_IP_name].attrs['processing_history'] = (
        DX.Uocean.attrs['processing_history'] 
        + '\nInterpolated to %.1f m depth.'%ip_depth)

    print('Added interpolated velocities: (%s, %s)'%(U_IP_name, V_IP_name))
    return DX

    
