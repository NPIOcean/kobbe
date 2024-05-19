
import numpy as np
import matplotlib.pyplot as plt

def plot_ellipse_icevel(DX, lp_days = 5, ax = None, 
                return_ax = True):
    '''
    Plot of u and v ice drift components
    low pass filtered with a running mean of *lp_days*.

    Showing the mean current vector, the low-pass-filtered 
    and subsampled currents, and the semi-major and -minor axes
    of the variance ellipse. 
    '''
    assert hasattr(DX, 'Uice'), (
        'No "Uice" field. Run sig_ice_vel.calculate_drift()..')

    print('ELLIPSE PLOT: Interpolate over nans.. \r', end = '')

    uip = DX.Uice.interpolate_na(dim = 'TIME', limit=10).data
    vip = DX.Vice.interpolate_na(dim = 'TIME', limit=10).data

    print('ELLIPSE PLOT: Low pass filtering..    \r', end = '')
    # LPFed 
    wlen = int(np.round(lp_days/(DX.sampling_interval_sec/60/60/24)))
    ULP = np.convolve(uip, np.ones(wlen)/wlen,
                mode = 'valid')[::wlen]
    VLP = np.convolve(vip, np.ones(wlen)/wlen,
                mode = 'valid')[::wlen]

    print('ELLIPSE PLOT: Calculating ellipse (from LPed data).. \r', end = '')

    # Ellipse
    thp, majax, minax = _uv_angle(
        ULP - np.nanmean(ULP), VLP - np.nanmean(VLP))

    # Mean
    UM, VM = np.nanmean(ULP), np.nanmean(VLP)


    print('ELLIPSE PLOT: Plotting..                              \r', end = '')

    if ax == None:
        fig, ax = plt.subplots(figsize = (10, 10))

    ax.set_aspect('equal')

    ax.plot(uip, vip, '.', ms = 1, color = 'Grey', 
        alpha = 0.3, lw = 2, zorder = 0)
    ax.plot(DX.Uice.data[-1], DX.Vice.data[-1], '.', ms= 1,color = 'k', alpha =0.5,
            lw = 2, label ='Full')

    ax.plot(ULP, VLP, '.', ms = 3, color = 'b', alpha = 0.5)
    ax.plot(ULP[0], VLP[0], '.', ms = 5, color = 'b', alpha = 0.5, 
        label = '%.1f-day means'%(lp_days), zorder = 0)

    vmaj = np.array([-majax*np.sin(thp), majax*np.sin(thp)])
    umaj = np.array([-majax*np.cos(thp), majax*np.cos(thp)])
    vmin = np.array([-minax*np.sin(thp+np.pi/2),
                        minax*np.sin(thp+np.pi/2)])
    umin = np.array([-minax*np.cos(thp+np.pi/2),
                        minax*np.cos(thp+np.pi/2)])

    ax.plot(UM + umaj , VM + vmaj, '-k', lw = 2, label ='Maj axis')
    ax.plot(UM + umin , VM + vmin, '--k', lw = 2, label ='Min axis')

    ax.quiver(0, 0, UM, VM, 
        color = 'r', scale_units = 'xy', scale = 1, width = 0.03, 
        headlength = 2, headaxislength = 2, alpha = 0.6, 
        label = 'Mean (u: %.2f, v: %.2f)'%(UM, VM),  edgecolor='k', 
        linewidth = 0.6)

    ax.set_ylabel('v [m s$^{-1}$]'); ax.set_xlabel('u [m s$^{-1}$]')
    ax.legend(fontsize= 10, loc =3, handlelength = 1, ncol = 2) 

    ax.set_title('Ice drift velocity components')
    ax.grid()
    plt.show()

    if return_ax:
        return ax



def histogram(DX, varnm, hrange = None, nbins = 50, 
        return_figure = False):
    '''
    Histogram showing the distribution of a variable - 1D or 2D.

    DXX xarray object with signature data 
    varnm: Name of the variable in DX
    hrange: Max range for the histogram
    nbins: Number of histogram bins
    return_figure: True for returning the figrue object. 
    '''
    fig = plt.figure(figsize = (8, 8))
    ax = plt.subplot2grid((2, 5), (0, 0), colspan =5)
    textax = plt.subplot2grid((2, 5), (1, 0), colspan =3)

    VAR_all = DX[varnm].data[~np.isnan(DX[varnm].data)]

    N_all = len(VAR_all)
    col_1 = (1.0, 0.498, 0.055)
    col_2 = (0.122, 0.467, 0.705)

    # Histogram, all entries
    Hargs = {'density': False, 'range':hrange, 'bins':nbins}
    H_all, H_bins = np.histogram(VAR_all, **Hargs)
    Hargs['bins']= H_bins
    H_width = np.ma.median(np.diff(H_bins))

    # Bar plot
    ax.bar(H_bins[:-1], 100*H_all/N_all, width = H_width, align = 'edge', 
            alpha = 0.4, color = col_1, label = 'All')

    # Cumulative plot
    cumulative = np.concatenate([[0], np.cumsum(100*H_all/N_all)])
    twax = ax.twinx()
    twax.plot(H_bins, cumulative, 'k', clip_on = False)
    twax.set_ylim(0, 105)

    # Axis labels

    # x label: Long description
    ax.set_ylabel('Density per bin [%]')
    twax.set_ylabel('Cumulative density [%]')
    if 'units' in DX[varnm].attrs.keys():
        unit = DX[varnm].attrs['units']
    else:
        unit = ''
        
    ax.set_xlabel(unit)

    attrtext = 'ATTRIBUTES\n------------------\n'
    attrtext += 'DIMENSIONS: %s'%str(DX[varnm].dims)
    for attrnm in DX[varnm].attrs.keys():
        #if len(DX[varnm].attrs[attrnm])>60:
         #   note DX.tilt_Average.attrs['note'][:60]
        attrtext+='\n%s: %s'%(attrnm.upper(), DX[varnm].attrs[attrnm])

    stattext = '\n\nQUICK STATS\n------------------'
    stattext+='\nTOTAL NUMBER NON-NaN VALUES: %.0f '%(len(VAR_all))
    stattext+='\nMEAN: %.2f %s'%(VAR_all.mean(), unit)
    stattext+='\nMEDIAN: %.2f %s'%(np.median(VAR_all), unit)
    stattext+='\nMIN: %.2f %s'%(np.min(VAR_all), unit)
    stattext+='\nMAX: %.2f %s'%(np.max(VAR_all), unit)
    stattext+='\nSD: %.2f %s'%(np.std(VAR_all), unit)

    textax.text(0.01, 0.9, attrtext + stattext, va = 'top',
                transform=textax.transAxes, fontsize=10, wrap = True)

    ax.set_title('%s'%varnm, fontweight = 'bold')

    textax.set_title('%s'%varnm, fontweight = 'bold')
    textax.tick_params(axis = 'both', which = 'both', bottom = False, 
        top=False, labelbottom=False, left=False, right=False, labelleft=False)
    textax.spines["top"].set_visible(False)
    textax.spines["right"].set_visible(False)
    textax.spines["left"].set_visible(False)
    textax.spines["bottom"].set_visible(False)

    if return_figure:
        return fig


def _uv_angle(u, v):
    '''
    Finds the principal angle angle in [-pi/2, pi/2] where the squares
    of the normal distance  to u,v are maximised. Ref Emery/Thompson
    pp 327.

    Also returns the standard deviation along the semimajor and
    semiminor axes.

    '''
    if np.nanmean(u)>1e-7 or np.nanmean(v)>1e-7:
        print('Mean of u and/or v is nonzero. Removing mean.')
        u = u - np.nanmean(u) ; v = v - np.nanmean(v)
    thp = 0.5* np.arctan2(2*np.nanmean(u*v),(np.nanmean(u**2)-\
                          np.nanmean(v**2))) #ET eq 4.3.23b
    uvcr = (u + 1j * v) * np.exp(-1j * thp)
    majax = np.nanstd(uvcr.real)
    minax = np.nanstd(uvcr.imag)

    return thp, majax, minax