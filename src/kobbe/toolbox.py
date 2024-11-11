"""
KOBBE.TOOLBOX

Various useful functions for plotting and examining the data.

"""

from typing import Optional, Tuple
import numpy as np
import matplotlib.pyplot as plt
from kval.ocean import uv
import xarray as xr

def histogram(
    ds: xr.Dataset,
    varnm: str,
    hrange: Optional[Tuple[float, float]] = None,
    nbins: int = 50,
    return_figure: bool = False
) -> Optional[plt.Figure]:
    """
    Plot a histogram showing the distribution of a variable (1D or 2D) in a
    Signature dataset, with optional statistical summary.

    This function generates a histogram for a specified variable in an xarray
    Dataset. It also displays quick statistics about the distribution,
    including mean, median, min, max, and standard deviation.

    Args:
        ds (xr.Dataset):
            The xarray Dataset containing the variable of interest.
        varnm (str):
            The name of the variable in the Dataset to plot.
        hrange (Optional[Tuple[float, float]], optional):
            The range of values to include in the histogram. If None, the range
            is determined from the data. Default is None.
        nbins (int, optional):
            The number of bins for the histogram. Default is 50.
        return_figure (bool, optional):
            If True, the function returns the matplotlib Figure object. Default
            is False.

    Returns:
        Optional[plt.Figure]:
            The matplotlib Figure object if `return_figure` is True, otherwise
            None.

    Example:
        >>> ds = xr.Dataset({"temperature": (["time"], np.random.randn(1000))})
        >>> histogram(ds, "temperature", nbins=30)
    """

    """
    Histogram showing the distribution of a variable - 1D or 2D.

    ds: xarray object with signature data
    varnm: Name of the variable in ds
    hrange: Max range for the histogram
    nbins: Number of histogram bins
    return_figure: True for returning the figrue object.
    """
    fig = plt.figure(figsize=(8, 8))
    ax = plt.subplot2grid((2, 5), (0, 0), colspan=5)
    textax = plt.subplot2grid((2, 5), (1, 0), colspan=3)

    VAR_all = ds[varnm].data[~np.isnan(ds[varnm].data)]

    N_all = len(VAR_all)
    col_1 = (1.0, 0.498, 0.055)

    # Histogram, all entries
    Hargs = {"density": False, "range": hrange, "bins": nbins}
    H_all, H_bins = np.histogram(VAR_all, **Hargs)
    Hargs["bins"] = H_bins
    H_width = np.ma.median(np.diff(H_bins))

    # Bar plot
    ax.bar(
        H_bins[:-1],
        100 * H_all / N_all,
        width=H_width,
        align="edge",
        alpha=0.4,
        color=col_1,
        label="All",
    )

    # Cumulative plot
    cumulative = np.concatenate([[0], np.cumsum(100 * H_all / N_all)])
    twax = ax.twinx()
    twax.plot(H_bins, cumulative, "k", clip_on=False)
    twax.set_ylim(0, 105)

    # Axis labels

    # x label: Long description
    ax.set_ylabel("Density per bin [%]")
    twax.set_ylabel("Cumulative density [%]")
    if "units" in ds[varnm].attrs.keys():
        unit = ds[varnm].attrs["units"]
    else:
        unit = ""

    ax.set_xlabel(unit)

    attr_text = "ATTRIBUTES\n------------------\n"
    attr_text += "DIMENSIONS: %s" % str(ds[varnm].sizes)
    for attrnm in ds[varnm].attrs.keys():
        # if len(ds[varnm].attrs[attrnm])>60:
        #   note ds.tilt_Average.attrs['note'][:60]
        attr_text += "\n%s: %s" % (attrnm.upper(), ds[varnm].attrs[attrnm])

    # Prepare statistics text
    stats_text = "\n\nQUICK STATS\n------------------"
    stats_text += f"\nTOTAL NUMBER NON-NaN VALUES: {N_all:.0f} "
    stats_text += f"\nMEAN: {VAR_all.mean():.2f} {unit}"
    stats_text += f"\nMEDIAN: {np.median(VAR_all):.2f} {unit}"
    stats_text += f"\nMIN: {np.min(VAR_all):.2f} {unit}"
    stats_text += f"\nMAX: {np.max(VAR_all):.2f} {unit}"
    stats_text += f"\nSD: {np.std(VAR_all):.2f} {unit}"

    textax.text(
        0.01,
        0.9,
        attr_text + stats_text,
        va="top",
        transform=textax.transAxes,
        fontsize=10,
        wrap=True,
    )

    ax.set_title("%s" % varnm, fontweight="bold")

    textax.set_title("%s" % varnm, fontweight="bold")
    textax.tick_params(
        axis="both",
        which="both",
        bottom=False,
        top=False,
        labelbottom=False,
        left=False,
        right=False,
        labelleft=False,
    )
    textax.spines["top"].set_visible(False)
    textax.spines["right"].set_visible(False)
    textax.spines["left"].set_visible(False)
    textax.spines["bottom"].set_visible(False)

    if return_figure:
        return fig


def plot_ellipse_icevel(
    ds,
    lp_days: int = 5,
    ax: Optional[plt.Axes] = None,
    return_ax: bool = True
) -> Optional[plt.Axes]:
    """
    Plot of ice drift components (u and v) low pass filtered with a running
    mean of *lp_days*.

    Showing the mean current vector, the low-pass-filtered and subsampled
    currents, and the semi-major and -minor axes of the variance ellipse. Args:
        ds: xarray.Dataset
            The dataset containing ice drift data, specifically "UICE" and
            "VICE" fields.
        lp_days (int, optional):
            The number of days over which to apply the low-pass filter. Default
            is 5 days.
        ax (matplotlib.axes.Axes, optional):
            An existing matplotlib Axes to plot on. If None, a new figure and
            axes are created.
        return_ax (bool, optional):
            If True, returns the matplotlib Axes object used for the plot.
            Default is True.

    Returns:
        Optional[plt.Axes]: The Axes object containing the plot if `return_ax`
        is True, otherwise None.

    Raises:
        AssertionError: If the dataset does not have the required "UICE" field.

    Example:
        >>> ds = xr.Dataset({"UICE": ..., "VICE": ...})
        >>> plot_ellipse_icevel(ds, lp_days=3)
    """

    assert hasattr(ds, "UICE"), ('No "UICE" field. Run '
                                 'vel.calculate_drift()..')

    print("ELLIPSE PLOT: Interpolate over nans.. \r", end="")

    uip = ds.UICE.interpolate_na(dim="TIME", limit=10).data
    vip = ds.VICE.interpolate_na(dim="TIME", limit=10).data

    print("ELLIPSE PLOT: Low pass filtering..    \r", end="")
    # LPFed
    wlen = int(np.round(lp_days / (ds.sampling_interval_sec / 60 / 60 / 24)))
    ULP = np.convolve(uip, np.ones(wlen) / wlen, mode="valid")[::wlen]
    VLP = np.convolve(vip, np.ones(wlen) / wlen, mode="valid")[::wlen]

    print("ELLIPSE PLOT: Calculating ellipse (from LPed data).. \r", end="")

    # Ellipse
    thp, majax, minax = uv.principal_angle(
        ULP - np.nanmean(ULP), VLP - np.nanmean(VLP))

    # Mean
    UM, VM = np.nanmean(ULP), np.nanmean(VLP)

    print("ELLIPSE PLOT: Plotting..                              \r", end="")

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 10))

    ax.set_aspect("equal")

    ax.plot(uip, vip, ".", ms=1, color="Grey", alpha=0.3, lw=2, zorder=0)
    ax.plot(
        ds.UICE.data[-1],
        ds.VICE.data[-1],
        ".",
        ms=1,
        color="k",
        alpha=0.5,
        lw=2,
        label="Full",
    )

    ax.plot(ULP, VLP, ".", ms=3, color="b", alpha=0.5)
    ax.plot(
        ULP[0],
        VLP[0],
        ".",
        ms=5,
        color="b",
        alpha=0.5,
        label="%.1f-day means" % (lp_days),
        zorder=0,
    )

    vmaj = np.array([-majax * np.sin(thp), majax * np.sin(thp)])
    umaj = np.array([-majax * np.cos(thp), majax * np.cos(thp)])
    vmin = np.array([-minax * np.sin(thp + np.pi / 2),
                     minax * np.sin(thp + np.pi / 2)])
    umin = np.array([-minax * np.cos(thp + np.pi / 2),
                     minax * np.cos(thp + np.pi / 2)])

    ax.plot(UM + umaj, VM + vmaj, "-k", lw=2, label="Maj axis")
    ax.plot(UM + umin, VM + vmin, "--k", lw=2, label="Min axis")

    ax.quiver(
        0,
        0,
        UM,
        VM,
        color="r",
        scale_units="xy",
        scale=1,
        width=0.03,
        headlength=2,
        headaxislength=2,
        alpha=0.6,
        label=f"Mean (u: {UM:.2f}, v: {VM:.2f})",
        edgecolor="k",
        linewidth=0.6,
    )

    ax.set_ylabel("v [m s$^{-1}$]")
    ax.set_xlabel("u [m s$^{-1}$]")
    ax.legend(fontsize=10, loc=3, handlelength=1, ncol=2)

    ax.set_title("Ice drift velocity components")
    ax.grid()
    plt.show()

    if return_ax:
        return ax

