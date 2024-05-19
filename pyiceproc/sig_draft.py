'''
SIG_DRAFT.PY

Functions for calculating sea ice draft 
'''

import numpy as np
from scipy.interpolate import interp1d
from matplotlib import pyplot as plt
from sigpyproc.sig_calc import runningstat, daily_average, clean_nanmedian
from sigpyproc import sig_append


def calculate_draft(DX, corr_sound_speed_CTD = True, qual_thr = 8000,
                    LE_correction = 'AST'):
    '''
    Calculate ice draft.  


    If LE_correction = 'AST', the open water sound speed correction (if 
    available) of the LE-derived draft will be based on the AST open water 
    offset.   
    '''


    DX = calculate_surface_position(DX, qual_thr = qual_thr,
        corr_sound_speed_CTD = corr_sound_speed_CTD, le_ast ='LE')

    # Get surface position (LE and AST)
    for le_ast in ['LE', 'AST']:
        DX = calculate_surface_position(DX, qual_thr = qual_thr,
             corr_sound_speed_CTD = corr_sound_speed_CTD, le_ast =le_ast,
             LE_correction = LE_correction)

    # Reject LE measurements where LE diverges from AST by >0.5 m
    DX['SURFACE_DEPTH_LE_UNCAPPED'] = DX['SURFACE_DEPTH_LE']

    condition = np.abs(DX.SURFACE_DEPTH_LE - DX.SURFACE_DEPTH_AST)<0.5
    DX['SURFACE_DEPTH_LE'] = DX['SURFACE_DEPTH_LE'].where(condition)

    # Get sea ice draft (based on ice(no ice criteria))
    # Calculate ensemble medians
    for le_ast in ['LE', 'AST']:

        si_draft_ = DX['SURFACE_DEPTH_%s'%le_ast].data.copy()
        si_draft_[~DX.ICE_IN_SAMPLE.data] = np.nan

        DX['SEA_ICE_DRAFT_%s'%le_ast] = (('TIME', 'SAMPLE'), 
            si_draft_, {'long_name':'Sea ice draft at each sample (%s)'%le_ast, 
            'units':'m', 'note':DX['SURFACE_DEPTH_%s'%le_ast].note + 
                '\n\nSet to NaN where ICE_IN_SAMPLE==False'})

        DX['SEA_ICE_DRAFT_%s'%le_ast] = DX['SEA_ICE_DRAFT_%s'%le_ast].where(
                    DX['SEA_ICE_DRAFT_%s'%le_ast]>-0.3)

        DX['SEA_ICE_DRAFT_MEDIAN_%s'%le_ast] = (('TIME'), 
            clean_nanmedian(si_draft_, axis=1),  {'long_name':
            'Median sea ice draft of each ensemble (%s)'%le_ast, 'units':'m', 
            'note':DX['SURFACE_DEPTH_%s'%le_ast].note + 
                '\n\nOnly counting instances with sea ice presence.'})

        DX['SEA_ICE_DRAFT_MEDIAN_%s'%le_ast] = DX['SEA_ICE_DRAFT_MEDIAN_%s'%le_ast].where(
                    DX['SEA_ICE_DRAFT_MEDIAN_%s'%le_ast]>-0.3)

    return DX



def calculate_surface_position(DX, corr_sound_speed_CTD = True, 
                                 qual_thr = 8000, le_ast ='AST', 
                                 LE_correction = 'AST'):
    '''
    Calculate distance between the surface measured by the altimeter 
    and the (mean) ocean surface.  

    If LE_correction = 'AST', the open water sound speed correction (if 
    available) of the LE-derived draft will be based on the AST open water 
    offset.   
    '''
    le_ast = le_ast.upper()
    if le_ast == 'AST':
        alt_dist_attr = 'Average_AltimeterDistanceAST'
        alt_qual_attr = 'Average_AltimeterQualityAST'
    else:
        alt_dist_attr = 'Average_AltimeterDistanceLE'
        alt_qual_attr = 'Average_AltimeterQualityLE'

    note_str = ('From %s altimeter distances.'
                '\n\nComputed with the function '
                'sig_draft.calculate_surface_position().'%le_ast)

    if hasattr(DX, 'sound_speed_CTD') and corr_sound_speed_CTD:
        # Ratio between observed and nominal sound speed
        sound_speed_ratio_obs_nom = (DX.sound_speed_CTD.data[:, np.newaxis]
            /DX.Average_Soundspeed.data)
        note_str += ('\n- Altimeter length recomputed using updated '
        'sound speed (*sound_speed_CTD* field)')
    else:
        sound_speed_ratio_obs_nom = 1
        note_str += ('\n- **NOT** USING OBSERVED SOUND SPEED\n'
           '  (*sound_speed* undefined) -> USING PRE-PROGRAMMED, NOMINAL VALUE!')    

    beta_key = 'BETA_open_water_corr_%s'%le_ast

    if LE_correction=='AST':
        beta_key = 'BETA_open_water_corr_AST'

    if hasattr(DX, beta_key):
        BETA_ow = (DX[beta_key].data[:, np.newaxis]
                     * np.ones(DX.depth.shape))
        note_str += ('\n- Using the open water correction factor to '
            'sound speed (*%s* field)'%beta_key)
    else:
        BETA_ow = 1
        note_str += ('\n- OPEN WATER CORRECTION **NOT** APPLIED!')    

    SURF_POS = DX.depth - (DX[alt_dist_attr]
                           * np.cos(np.pi*DX.tilt_Average/180)
                           * sound_speed_ratio_obs_nom
                           * BETA_ow)

    # APPLY QUALITY CRITERION
    SURF_POS = SURF_POS.where(DX[alt_qual_attr]>qual_thr)
    note_str += ('\n- Samples where %s>%i'%(alt_qual_attr, qual_thr) +
        ' were discarded.')

    # STORE AS VARIABLE
    DX['SURFACE_DEPTH_%s'%le_ast] = (('TIME', 'SAMPLE'), SURF_POS.data, {
        'long_name':'Depth of the scattering surface observed by'
        ' the Altimeter (%s)'%le_ast, 
        'units':'m', 'note':note_str})

    return DX



def get_OWSD(DX, method = 'LE'):
    '''
    Get the surface depth during open water periods only.

    Returns DataArray containing the same variable as the input - 
    but with ice entries masked.
    '''
    OWSD = DX['SURFACE_DEPTH_%s'%method].where(
                            DX.ICE_IN_SAMPLE_ANY==False)
    return OWSD


def get_LP_OWSD(OWSD, thr_reject_from_net_median = 0.15, 
        min_frac_daily = 0.025, run_window_days = 3, ):
    '''
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
    '''
    # 1. Compute initial median and reject values away from the median 
    #    by *thr_reject_from_netmedian* [m]
    OWSD_full_median = OWSD.median()
    OWSD_filt = OWSD.where(
        np.abs(OWSD-OWSD_full_median)<thr_reject_from_net_median)
    
    # 2. Compute ensemble medians
    OWSD_med = OWSD_filt.median(dim = 'SAMPLE')
    
    #fig, ax = plt.subplots(2, 1, sharex = True)
    #ax[0].plot(OWSD.TIME, OWSD_med)

    # 3. Compute daily medians ()
    Ad, td = daily_average(OWSD_med, OWSD.TIME, min_frac = min_frac_daily, 
                         axis = -1, function = 'median')
    
    # 4. Interpolate to continuous function (daily)
    Ad_interp = interp1d(td.data[~np.isnan(Ad)], Ad[~np.isnan(Ad)], 
        bounds_error = False)(td.data)

    # 5. Smooth with running mean
    RS = runningstat(Ad_interp, run_window_days)
    
    # Export filtered, ensemble median, daily averaged, smoothed daily OWSD.
    # Also daily time array (td+0.5) of teh midpoint of the daily estimates.
    return RS['mean'], td+0.5


def get_Beta_from_OWSD(DX, 
        thr_reject_from_net_median = 0.15, 
        min_frac_daily = 0.025, run_window_days = 3,):
    '''
    Estimate sound speed correction BETA.
    '''

    # Obtain (all) estimates of daily, smoothed OWSDs
    OWSD_full_LE = get_OWSD(DX, method = 'LE')
    OWSD_full_AST = get_OWSD(DX, method = 'AST')

    # Obtain estimates of daily, smoothed OWSDs
    OWSD_LP_LE, _ = get_LP_OWSD(OWSD_full_LE, 
        thr_reject_from_net_median = thr_reject_from_net_median,
        min_frac_daily = min_frac_daily)
    OWSD_LP_AST, td = get_LP_OWSD(OWSD_full_AST, 
        thr_reject_from_net_median = thr_reject_from_net_median,
        min_frac_daily = min_frac_daily)


    # Obtain daily, smoothed instrument depths     
    depth_med = DX.depth.median(dim = 'SAMPLE')
    depth_med_daily, _ = daily_average(depth_med, DX.TIME, td = td-0.5, 
                                axis = -1, function = 'median')
    RS_depth = runningstat(depth_med_daily, run_window_days)
    depth_lp = RS_depth['mean']

    # Obtain Beta (sound speed correction factors)
    BETA_LE = depth_lp/(depth_lp - OWSD_LP_LE)
    BETA_AST = depth_lp/(depth_lp - OWSD_LP_AST)

    DX = sig_append.add_to_sigdata(DX, BETA_LE, td, 
            'BETA_open_water_corr_LE')
    DX = sig_append.add_to_sigdata(DX, BETA_AST, td, 
            'BETA_open_water_corr_AST')

    # Append the open water estimates as well
    DX = sig_append.add_to_sigdata(DX, OWSD_LP_LE, td, 
            'OW_surface_before_correction_LE')
    DX = sig_append.add_to_sigdata(DX, OWSD_LP_AST, td, 
            'OW_surface_before_correction_AST')
    return DX


def compare_OW_correction(DX, show_plots = True):
    '''
    Note: Run this *after* running *get_Beta_from_OWSD* but *before*
    running *sig_draft.calculate_draft()* again.
    '''

    DX0 = DX.copy()
    DX2 = DX.copy()
    DX2 = calculate_draft(DX2)


    print('LE: Mean (median) offset: %.1f cm (%.1f cm)'%(
        DX.OW_surface_before_correction_LE.mean()*1e2, 
        clean_nanmedian(DX.OW_surface_before_correction_LE)*1e2))

    print('AST: Mean (median) offset: %.1f cm (%.1f cm)'%(
        DX.OW_surface_before_correction_AST.mean()*1e2, 
        clean_nanmedian(DX.OW_surface_before_correction_AST)*1e2))

    print('LE: Mean (median) dBETA: %.1f (%.1f)'%(
        (DX.BETA_open_water_corr_LE.mean()-1)*1e3, 
        (clean_nanmedian(DX.BETA_open_water_corr_LE)-1)*1e3))

    print('AST: Mean (median) dBETA: %.1f (%.1f)'%(
        (DX.BETA_open_water_corr_AST-1).mean()*1e3, 
        clean_nanmedian((DX.BETA_open_water_corr_AST)-1)*1e3))


    print('LE - MEAN SEA ICE DRAFT:\n'+
          'Before OW correction: %.2f m'%DX0.SEA_ICE_DRAFT_MEDIAN_LE.mean()+
          '\nAfter OW correction: %.2f m'%DX2.SEA_ICE_DRAFT_MEDIAN_LE.mean())

    print('AST - MEAN SEA ICE DRAFT:\n'+
          'Before OW correction: %.2f m'%DX0.SEA_ICE_DRAFT_MEDIAN_AST.mean()+
          '\nAfter OW correction: %.2f m'%DX2.SEA_ICE_DRAFT_MEDIAN_AST.mean())


    # Figures
    if show_plots:
        fig, ax = plt.subplots(2, 1, sharex = True)

        ax[0].plot_date(DX2.TIME, DX.OW_surface_before_correction_LE, '-', label = 'LE')
        ax[0].plot_date(DX2.TIME, DX.OW_surface_before_correction_AST, '-', label = 'AST')

        ax[1].plot_date(DX2.TIME, DX.BETA_open_water_corr_LE, '-', label = 'LE')
        ax[1].plot_date(DX2.TIME, DX.BETA_open_water_corr_AST, '-', label = 'AST')

        for axn in ax: 
            axn.legend()
            axn.grid()
        labfs = 9
        ax[0].set_ylabel('Estimated open water\nsurface depth [m]', 
            fontsize = labfs)
        ax[1].set_ylabel('BETA (OWSD correction factor)', 
            fontsize = labfs)

        fig, ax = plt.subplots(1, 2, sharex = True, sharey = True)
        ax[0].scatter(DX0.time_average, DX0.SURFACE_DEPTH_LE, marker='.', 
                        color = 'k', alpha = 0.05, s = 0.3, label = 'Uncorrected')
        ax[0].scatter(DX.time_average, DX2.SURFACE_DEPTH_LE, marker='.', 
                        color = 'r', alpha = 0.05, s = 0.3, label = 'Corrected')

        ax[1].scatter(DX0.TIME, DX0.SEA_ICE_DRAFT_MEDIAN_LE, marker='.', 
                        color = 'k', alpha = 0.05, s = 0.3, label = 'Uncorrected')
        ax[1].scatter(DX.TIME, DX2.SEA_ICE_DRAFT_MEDIAN_LE, marker='.', 
                        color = 'r', alpha = 0.05, s = 0.3, label = 'Corrected')
        ax[0].set_title('LE Surface depth (ALL)')
        ax[1].set_title('LE sea ice draft (ice only, ensemble averaged)')

        for axn in ax: 
            axn.legend()
            axn.grid()
            axn.set_ylabel('[m]')

        labfs = 9
        ax[0].set_ylabel('Estimated open water\nsurface depth [m]', 
            fontsize = labfs)
        ax[1].set_ylabel('BETA (OWSD correction factor)', 
            fontsize = labfs)

        # Dummy for date axis..
        ax[0].plot_date(DX.time_average[0, 0],DX2.SURFACE_DEPTH_LE[0, 0] )
        
        ax[0].invert_yaxis()
        plt.show()