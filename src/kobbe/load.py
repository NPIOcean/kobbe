"""
KOBBE.LOAD

Various functions for loading and concatenating Nortek Signature matfiles
produced by Nortek SignatureDeployment.

"""

##############################################################################

# IMPORTS

import numpy as np
from scipy.io import loadmat
import xarray as xr
from matplotlib.dates import num2date, date2num
import matplotlib.pyplot as plt
from kval.util.time import matlab_time_to_python_time
from kobbe.append import _add_tilt, _add_SIC_FOM, set_lat, set_lon
from datetime import datetime
import warnings
import os
import glob2
from typing import List, Optional, Tuple, Union, Dict, Any

##############################################################################


def matfiles_to_dataset(
    file_input: Union[List[str], str],
    reshape: bool = True,
    lat: Optional[float] = None,
    lon: Optional[float] = None,
    include_raw_altimeter: bool = False,
    FOM_ice_threshold: float = 1e4,
    time_range: Optional[Tuple[Optional[str], Optional[str]]] = None,
) -> xr.Dataset:
    """
    Read, convert, and concatenate .mat files exported from
    SignatureDeployment.

    Parameters
    ----------
    file_input : Union[List[str], str]
        List of file paths to .mat files, or a directory containing .mat files.
    reshape : bool, optional
        If True, reshape all time series from a single 'time_average' dimension
        to 2D ('TIME', 'SAMPLE') where TIME is the mean time of each ensemble
        and SAMPLE is each sample in the ensemble. Default is True.
    lat : Optional[float], optional
        Latitude of deployment. If None, this information is not included.
        Default is None.
    lon : Optional[float], optional
        Longitude of deployment. If None, this information is not included.
        Default is None.
    include_raw_altimeter : bool, optional
        Include raw altimeter signal if available (typically on a single time
        grid). Default is False.
    FOM_ice_threshold : float, optional
        Threshold for "Figure of Merit" in the ice ADCP pings used to separate
        measurements in ice from measurements of water. Default is 1e4.
    time_range : Optional[Tuple[Optional[str], Optional[str]]], optional
        Only accept data within this date range. Provide a tuple of date
        trings in 'DD.MM.YYYY' format.
        Default is None, which includes all data.

    Returns
    -------
    xr.Dataset
        Xarray Dataset containing the concatenated data.

    Raises
    ------
    ValueError
        If file_input is empty or if there's a problem during concatenation.

    Notes
    -----
    The function assumes that the provided .mat files are structured in a way
    that can be handled by the internal `_matfile_to_dataset`, `_add_tilt`,
    `_reshape_ensembles`, `set_lat`, `set_lon`, and `_add_SIC_FOM` functions.
    """

    # Convert directory input to a list of .mat files
    if isinstance(file_input, str) and os.path.isdir(file_input):
        file_list = glob2.glob(os.path.join(file_input, "*.mat"))
    elif isinstance(file_input, list):
        file_list = file_input
    else:
        raise ValueError(
            "file_input must be a list of file paths or a directory"
            " containing .mat files."
        )

    if len(file_list) == 0:
        raise ValueError("The file list is empty. No .mat files found.")

    # Get max/min times:
    date_fmt = "%d.%m.%Y"
    if time_range and time_range[0]:
        time_min = date2num(datetime.strptime(time_range[0], date_fmt))
    else:
        time_min = None
    if time_range and time_range[1]:
        time_max = date2num(datetime.strptime(time_range[1], date_fmt))
    else:
        time_max = None

    ##########################################################################
    # LOAD AND CONCATENATE DATA

    first = True
    pressure_offsets = np.array([])

    if len(file_list) == 0:
        raise ValueError(
            "The *file_list* given to the function " "matfiles_to_dataset()"
            " is empty."
        )

    for filename in file_list:
        ds_single, pressure_offset = _matfile_to_dataset(
            filename, include_raw_altimeter=include_raw_altimeter
        )

        ds_single = ds_single.sel({"time_average": slice(time_min, time_max)})

        pressure_offsets = np.append(pressure_offsets, pressure_offset)

        if first:
            ds = ds_single
            first = False
        else:
            print(f'CONCATENATING: FILE "{filename[-15:]}"\r', end="")
            try:
                ds = xr.concat([ds, ds_single], dim="time_average")
            except Exception as e:
                print(f"Failed at {filename[-10:]} with error: {e}")

    # Reads the pressure offset(s), i.e. the fixed atmospheric pressure
    # used to obtain sea pressure-

    if len(np.unique(pressure_offsets)) == 1:
        ds.attrs["pressure_offset"] = pressure_offsets[0]
    else:
        ds.attrs["pressure_offset"] = pressure_offsets

    # Add tilt (from pitch/roll)
    ds = _add_tilt(ds)

    # Sort by time
    ds = ds.sortby("time_average")

    # Reshape
    if reshape:
        ds = _reshape_ensembles(ds)

    # Grab the (de facto) sampling rate
    ds.attrs["sampling_interval_sec"] = np.round(
        np.ma.median(np.diff(ds.time_average) * 86400), 3
    )

    # Add some attributes
    ds = set_lat(ds, lat)
    ds = set_lon(ds, lon)

    # Add FOM threshold
    ds["FOM_threshold"] = (
        (),
        FOM_ice_threshold,
        {"description":
         "Figure-of-merit threshold used to separate ice vs open water"},
    )

    # Add sea ice concentration estimate from FOM
    ds = _add_SIC_FOM(ds)

    # Add history attribute
    now_date_str = datetime.now().strftime(
        "%d %b %Y.")
    ds.attrs["history"] = f"- Loaded from .mat files on {now_date_str}"

    print("Done. Run kobbe.load.overview()) to print some additional details.")

    return ds


##############################################################################


def chop(
    ds: xr.Dataset,
    indices: Optional[Tuple[int, int]] = None,
    auto_accept: bool = False
) -> xr.Dataset:
    """
    Chop an xarray Dataset by removing data outside of a specified range or
    based on a pressure-based algorithm.

    Default behaviour is to display a plot showing the suggested chop. The
    user can then accept or decline.

    If `indices` is not provided, the function will use a pressure-based
    algorithm to suggest a range for chopping.

    Parameters
    ----------
    ds : xr.Dataset
        The xarray Dataset containing the data to be chopped. The dataset
        should include the field `Average_AltimeterPressure`.
    indices : Optional[Tuple[int, int]]
        Tuple of indices (start, stop) for slicing the dataset along the TIME
        dimension. If not provided, a pressure-based algorithm is used.
    auto_accept : bool
        If `True`, automatically accepts the suggested chop based on the
        pressure record. If `False`, prompts the user for confirmation.

    Returns
    -------
    xr.Dataset
        The chopped xarray Dataset.
    """
    if indices is None:
        p = ds.Average_AltimeterPressure.mean(dim="SAMPLE").data
        p_mean = np.ma.median(p)
        p_sd = np.ma.std(p)

        indices = [None, None]

        if p[0] < p_mean - 3 * p_sd:
            indices[0] = np.where(np.diff(p < p_mean - 3 * p_sd))[0][0] + 1
        if p[-1] < p_mean - 3 * p_sd:
            indices[1] = np.where(np.diff(p < p_mean - 3 * p_sd))[0][-1]

        keep_slice = slice(*indices)
        if auto_accept:
            accept = "y"
        else:
            fig, ax = plt.subplots(figsize=(8, 4))
            index = np.arange(len(p))
            ax.plot(index, p, "k", label="Pressure")
            ax.plot(index[keep_slice], p[keep_slice], "r",
                    label="Chopped Range")
            ax.set_xlabel("Index")
            ax.set_ylabel("Pressure [db]")
            ax.invert_yaxis()
            ax.set_title(f"Suggested chop: {indices} (to red curve)."
                         " Close this window to continue..")
            ax.legend()
            plt.show(block=True)
            print(f"Suggested chop: {indices} (to red curve)")
            accept = input("Accept (y/n)?: ")

        if accept.lower() == "n":
            print("Not accepted -> Not chopping anything now.")
            print("NOTE: run chop(ds, indices =[A, B]) to manually set chop.")
            return ds
        elif accept.lower() == "y":
            pass
        else:
            raise ValueError(f'I do not understand your input "{accept}"'
                             '. Only "y" or "n" works. -> Exiting.')
    else:
        keep_slice = slice(indices[0], indices[1] + 1)

    L0 = ds.sizes["TIME"]
    print(f"Chopping to index: {indices}")
    ds = ds.isel(TIME=keep_slice)
    L1 = ds.sizes["TIME"]
    net_str = (f"Chopped {L0 - L1} ensembles using -> {indices} "
               f"(total ensembles {L0} -> {L1})")
    print(net_str)

    ds.attrs["history"] += f"\n- {net_str}"
    return ds


def overview(ds: xr.Dataset) -> None:
    """
    Prints basic information about the given Signature xarray Dataset,
    including time range, pressure statistics, and dataset size.

    Parameters
    ----------
    ds : xr.Dataset
        The xarray Dataset containing the data to be summarized. The dataset
        should be generated using the kobbe.load.module.

    Returns
    -------
    None
    """
    # Validate required fields
    required_fields = ["TIME", "Average_AltimeterPressure",
                       "time_between_ensembles_sec", "sampling_interval_sec"]
    for field in required_fields:
        if field not in ds and field not in ds.attrs:
            raise ValueError(f"Dataset is missing required field: {field}")

    # Time range
    datefmt = "%d %b %Y %H:%M"
    starttime = num2date(ds.TIME[0].values).strftime(datefmt)
    endtime = num2date(ds.TIME[-1].values).strftime(datefmt)
    ndays = (ds.TIME[-1] - ds.TIME[0]).values / (60 * 60 * 24)  # secs->days

    print("\nTIME RANGE:")
    print(f"{starttime}  -->  {endtime}  ({ndays:.1f} days)")
    print(f"Time between ensembles: {ds.time_between_ensembles_sec / 60:.1f}"
          " min.")
    print(f"Time between samples in ensembles: {ds.sampling_interval_sec:.1f}"
          " sec.")

    # Pressure
    med_pres = np.ma.median(ds.Average_AltimeterPressure)
    std_pres = np.ma.std(ds.Average_AltimeterPressure)
    # (Default to "N/A" if key is missing)
    pressure_offset = ds.attrs.get("pressure_offset", "N/A")

    print("\nPRESSURE:")
    print(f"Median (STD) of altimeter pressure: {med_pres:.1f} dbar "
          f"({std_pres:.1f} dbar) - with fixed atm offset "
          f"{pressure_offset:.3f} dbar.")

    # Size
    total_time_points = ds.sizes["TIME"] * ds.sizes["SAMPLE"]
    num_ensembles = ds.sizes["TIME"]
    samples_per_ensemble = ds.sizes["SAMPLE"]
    # (Default to "N/A" if key is missing)
    num_bins = ds.sizes.get("BINS", "N/A")

    print("\nSIZE:")
    print(f"Total {total_time_points} time points.")
    print(f"Split into {num_ensembles} ensembles with {samples_per_ensemble} "
          "sample(s) per ensemble.")
    print(f"Ocean velocity bins: {num_bins}.")

##############################################################################


def _reshape_ensembles(
        ds: xr.Dataset, time_threshold_min: float = 7.0) -> xr.Dataset:
    """
    Reshape a dataset from a single 'time_average' dimension to 2D ('TIME',
    'SAMPLE'), where 'TIME' represents the mean time of each ensemble and
    'SAMPLE' represents each sample within the ensemble.

    Parameters
    ----------
    ds : xr.Dataset
        The xarray Dataset to reshape. It should have the following dimensions:
        - time_average: The dimension to be reshaped
        - Optional dimensions: BINS, xyz, beams
    time_threshold_min : float, optional
        The time jump threshold between ensembles in minutes to determine the
        start and end of ensembles. Default is 7.0 minutes.

    Returns
    -------
    xr.Dataset
        A new xarray Dataset with reshaped dimensions and updated coordinates.
    """

    # Check for required dimension
    if "time_average" not in ds.dims:
        raise ValueError("Dataset must contain the 'time_average' dimension.")

    ###########################################################################
    # ADD A "TTIME" COORDINATE (ONE ENTRY PER ENSEMBLE)

    Nt = len(ds.time_average)

    Nt = len(ds.time_average)
    time_average_data = ds.time_average.data

    # Convert time threshold from minutes to days
    time_threshold_days = time_threshold_min / (24 * 60)

    # Find indices where there is a time jump > threshold
    time_jump_inds = np.where(
        np.diff(time_average_data) > time_threshold_days)[0]

    # Start and end times of each ensemble
    ens_starts = np.concatenate([np.array([0]), time_jump_inds + 1])
    ens_ends = np.concatenate([time_jump_inds, np.array([Nt - 1])])

    # Calculate mean time of each ensemble
    t_ens = 0.5 * (time_average_data[ens_starts] + time_average_data[ens_ends])
    Nsamp_per_ens = ds.samples_per_ensemble
    Nens = int(Nt / Nsamp_per_ens)

    if Nens != len(t_ens):
        warnings.warn(
            f"Expected number of ensembles ({Nens}) differs from the number"
            f" deduced from time jumps ({len(t_ens)}).\nThis discrepancy"
            " might cause issues. Please check your time grid."
        )

    print(f"{Nt} time points, {Nens} ensembles. "
          f"Samples per ensemble: {Nsamp_per_ens}")

    # Prepare new coordinates
    rsh_coords = dict(ds.coords)
    rsh_coords.pop("time_average")

    rsh_coords["TIME"] = (
        ["TIME"],
        t_ens,
        {
            "units": "Days since 1970-01-01",
            "long_name": "Time stamp of the ensemble averaged measurement",
        },
    )
    rsh_coords["SAMPLE"] = (
        ["SAMPLE"],
        np.arange(1, Nsamp_per_ens + 1),
        {
            "units": "Sample number",
            "long_name": (f"Sample number in ensemble ({Nsamp_per_ens}"
                          f" samples per ensemble)"),
        },
    )

    # Create reshaped dataset
    ds_rsh = xr.Dataset(coords=rsh_coords)
    ds_rsh.attrs = ds.attrs
    if "instrument_configuration_details" in ds_rsh.attrs:
        del ds_rsh.attrs["instrument_configuration_details"]
    ds_rsh.attrs["instrument_configuration_details"] = ds.attrs.get(
        "instrument_configuration_details", "N/A")

    # Reshape variables
    for var_ in ds.variables:
        dims = ds[var_].dims
        data = ds[var_].data
        attrs = ds[var_].attrs

        if dims == ("time_average",):
            reshaped_data = np.ma.reshape(data, (Nens, Nsamp_per_ens))
            ds_rsh[var_] = (("TIME", "SAMPLE"), reshaped_data, attrs)
        elif dims == ("BINS", "time_average"):
            reshaped_data = np.ma.reshape(
                data, (ds.sizes["BINS"], Nens, Nsamp_per_ens))
            ds_rsh[var_] = (("BINS", "TIME", "SAMPLE"), reshaped_data, attrs)
        elif dims == ("time_average", "xyz"):
            reshaped_data = np.ma.reshape(
                data, (Nens, Nsamp_per_ens, ds.sizes["xyz"]))
            ds_rsh[var_] = (("TIME", "SAMPLE", "xyz"), reshaped_data, attrs)
        elif dims == ("time_average", "beams"):
            reshaped_data = np.ma.reshape(
                data, (Nens, Nsamp_per_ens, ds.sizes["beams"]))
            ds_rsh[var_] = (("TIME", "SAMPLE", "beams"), reshaped_data, attrs)

    return ds_rsh

##############################################################################


def _matfile_to_dataset(
    filename: str,
    include_raw_altimeter: bool = False
) -> Tuple[xr.Dataset, float]:
    """
    Read and convert a single .mat file exported from SignatureDeployment.

    Parameters
    ----------
    filename : str
        The path to the .mat file.
    include_raw_altimeter : bool, optional
        Include raw altimeter signal if available.
        Default is False.

    Returns
    -------
    ds_single : xr.Dataset
        xarray Dataset containing the data.
    pressure_offset : float
        Pressure offset used in the data.
    """

    # Read .mat file into dictionary
    b = _sig_mat_to_dict(filename)

    # Obtain coordinates
    coords = {
        "time_average": matlab_time_to_python_time(b["Average_Time"]),
        "BINS": np.arange(b["Average_VelEast"].shape[1]),
        "xyz": np.arange(3),
        "beams": np.arange(
                        len(b["Average_BeamToChannelMapping"])
                    )}

    if include_raw_altimeter:
        try:  # If we have AverageRawAltimeter: Add this as well
            coords.update(
                {"along_altimeter": np.arange(
                        b["AverageRawAltimeter_AmpBeam5"].shape[1]
                    ),
                    "raw_altimeter_time": matlab_time_to_python_time(
                        b["AverageRawAltimeter_Time"]),
                 }
            )
        except KeyError:
            print("No *AverageRawAltimeter*")

    # Create an xarray Dataset to be filled
    # Further down: concatenating into a joined dataset.
    ds_single = xr.Dataset(coords=coords)

    # Check and add AverageIce fields if present
    try:
        ds_single["time_average_ice"] = (
            ("time_average"),
            matlab_time_to_python_time(b["AverageIce_Time"]),
        )
    except KeyError:
        print("Did not find AverageIce data")

    # IMPORT DATA AND ADD TO XARRAY DATASET
    # Looping through variables. Assigning to dataset
    # as fields with the appropriate dimensions.
    for key in b.keys():

        # AverageIce fields
        if "AverageIce_" in key:
            if b[key].ndim == 0:
                ds_single[key] = ((), b[key])
            elif b[key].ndim == 1:
                ds_single[key] = (("time_average"), b[key])
            elif b[key].ndim == 2:
                ds_single[key] = (("time_average", "xyz"), b[key])

        # Average fields
        elif "Average_" in key:
            if b[key].ndim == 0:
                ds_single[key] = ((), b[key])
            elif b[key].ndim == 1:
                if len(b[key]) == ds_single.sizes["time_average"]:
                    ds_single[key] = (("time_average"), b[key])
                else:
                    ds_single[key] = (("beams"), b[key])
            elif b[key].ndim == 2:
                if b[key].shape[1] == ds_single.sizes["xyz"]:
                    ds_single[key] = (("time_average", "xyz"), b[key])
                elif b[key].shape[1] == ds_single.sizes["BINS"]:
                    ds_single[key] = (("BINS", "time_average"), b[key].T)

        # AverageRawAltimeter fields
        elif "AverageRawAltimeter_" in key:
            if include_raw_altimeter:
                if b[key].ndim == 0:
                    ds_single[key] = ((), b[key])

                elif b[key].ndim == 1:
                    if len(b[key]) == ds_single.sizes["beams"]:
                        ds_single[key] = (("beams"), b[key])
                    else:
                        ds_single[key] = (("time_raw_altimeter"), b[key])
                elif b[key].ndim == 2:
                    if b[key].shape[1] == ds_single.sizes["xyz"]:
                        ds_single[key] = (("time_raw_altimeter", "xyz"),
                                          b[key])
                    elif b[key].shape[1] == ds_single.sizes["along_altimeter"]:
                        ds_single[key] = (
                            ("along_altimeter", "time_raw_altimeter"),
                            b[key].T,
                        )

    # Assign metadata attributes
    # Units, description etc
    ds_single.time_average.attrs["description"] = (
        "Time stamp for"
        ' "average" fields. Source field: *Average_Time*. Converted'
        " using kval.util.time.matlab_time_to_python_time()."
    )
    ds_single.BINS.attrs["description"] = "Number of velocity bins."
    ds_single.beams.attrs["description"] = "Beam number (not 5th)."
    ds_single.xyz.attrs["description"] = "Spatial dimension."

    if include_raw_altimeter:
        ds_single.time_raw_altimeter.attrs["description"] = (
            "Time stamp for"
            ' "AverageRawAltimeter" fields. Source field:'
            " *AverageRawAltimeter_Time*. Converted"
            " using matlab_time_to_python_time."
        )
        ds_single.along_altimeter.attrs["description"] = (
                    "Index along altimeter.")

    # Read the configuration info to a single string (gets cumbersome to
    # carry these around as attributes..)
    # Read the configuration info into a single string
    conf_str = ""
    for conf_ in b.get("conf", {}):
        unit_str = (
            f" {b['units'].get(conf_, '')}" if conf_ in b["units"] else "")
        desc_str = (
            f" ({b['desc'].get(conf_, '')})" if conf_ in b["desc"] else "")
        conf_str += f"{conf_}{desc_str}: {b['conf'][conf_]}{unit_str}\n"

    ds_single.attrs["instrument_configuration_details"] = conf_str

    # Add some selected attributes that are useful
    ds_single.attrs["instrument"] = b["conf"]["InstrumentName"]
    ds_single.attrs["serial_number"] = b["conf"]["SerialNo"]
    ds_single.attrs["samples_per_ensemble"] = int(b["conf"]["Average_NPings"])
    ds_single.attrs["time_between_ensembles_sec"] = int(
        b["conf"]["Plan_ProfileInterval"]
    )
    ds_single.attrs["blanking_distance_oceanvel"] = b["conf"][
        "Average_BlankingDistance"
    ]
    ds_single.attrs["cell_size_oceanvel"] = b["conf"]["Average_CellSize"]
    ds_single.attrs["N_cells_oceanvel"] = b["conf"]["Average_NCells"]

    # Read pressure offset
    pressure_offset = b["conf"]["PressureOffset"]

    return ds_single, pressure_offset


##############################################################################


def _sig_mat_to_dict(
    matfn: str,
    include_metadata: bool = True,
    squeeze_identical: bool = True,
    skip_Burst: bool = True,
    skip_IBurst: bool = True,
    skip_Average: bool = False,
    skip_AverageRawAltimeter: bool = False,
    skip_AverageIce: bool = False,
    skip_fields: Optional[List[str]] = []
        ) -> Dict[str, Any]:
    """
    Reads matfile produced by SignatureDeployment to numpy dictionary.

    Parameters
    ----------
    matfn : str
        Path to the .mat file.
    include_metadata : bool, optional
        If False, skips the config, description, and units fields.
        Default is True.
    squeeze_identical : bool, optional
        If True, reduces arrays of identical entries to a single entry.
        Default is True.
    skip_Burst : bool, optional
        If True, skips variables starting with 'Burst_'. Default is True.
    skip_IBurst : bool, optional
        If True, skips variables starting with 'IBurst_'. Default is True.
    skip_Average : bool, optional
        If True, skips variables starting with 'Average_'. Default is False.
    skip_AverageRawAltimeter : bool, optional
        If True, skips variables starting with 'AverageRawAltimeter_'.
        Default is False.
    skip_AverageIce : bool, optional
        If True, skips variables starting with 'AverageIce_'. Default is False.
    skip_fields : list of str, optional
        List of specific fields to skip. Default is no defaul skipping.

    Returns
    -------
    Dict[str, Any]
        A dictionary where keys are variable names and values are numpy arrays.

    Notes
    -----
    - The function uses `loadmat` to read the .mat file.
    - Data arrays with identical entries are squeezed into a single entry if
      `squeeze_identical` is True.
    """

    # Load the mat file
    dmat = loadmat(matfn)

    # Create a dictionary that we will fill in below
    d = {}

    # Add metadata fields if requested
    if True:
        metadata_keys = {"Config": "conf", "Descriptions":
                         "desc", "Units": "units"}
        for indict_key, outdict_key in metadata_keys.items():
            outdict = {}
            for varkey in dmat[indict_key].dtype.names:
                outdict[varkey] = _unpack_nested(dmat[indict_key][varkey])
            d[outdict_key] = outdict

    # Making a list of strings based on the *skip_* booleans in the function
    # call. *startstrings_skip* contains a start strings. If a variable starts
    # with any of these srtings, we won't include it.
    startstrings = [
        "Burst_",
        "IBurst_",
        "Average_",
        "AverageRawAltimeter_",
        "AverageIce_",
    ]
    startstrings_skip_bool = [
        skip_Burst,
        skip_IBurst,
        skip_Average,
        skip_AverageRawAltimeter,
        skip_AverageIce,
    ]
    startstrings_skip = tuple(
        [str_ for (str_, bool_) in
         zip(startstrings, startstrings_skip_bool) if bool_]
    )

    # Process data fields.  Masking NaNs and squeezing/unpacking unused
    # dimensions.
    for varkey in dmat["Data"].dtype.names:
        if (
            not varkey.startswith(startstrings_skip)
            and varkey not in skip_fields
        ):

            d[varkey] = np.ma.squeeze(
                np.ma.masked_invalid(_unpack_nested(dmat["Data"][varkey]))
            )

            if (d[varkey] == d[varkey][0]).all() and squeeze_identical:
                d[varkey] = d[varkey][0]

    # Return the dictionary
    return d


##############################################################################

def _unpack_nested(val: Union[List[Any], np.ndarray]) -> Any:
    """
    Recursively unpack nested lists or arrays down to the base level.

    Need this because data in the SignatureDeployment matfiles end up in
    weird nested structures.

    E.g. [['s']] -> 's'
    E.g. [[[[[1, 2], [3, 4]]]] -> [[1, 2], [3, 4]]

    Parameters
    ----------
    val : Any
        The value to be unpacked, which can be nested lists, arrays,
        or other iterables.

    Returns
    -------
    Any
        The unpacked value at its base level.
    """

    unpack = True
    while unpack is True:
        if hasattr(val, "__iter__") and len(val) == 1:
            val = val[0]
            if isinstance(val, str):
                unpack = False
        else:
            unpack = False
    return val


##############################################################################


def to_nc(
    ds: xr.Dataset,
    file_path: str,
    export_vars: Optional[List[str]] = None,
    icedraft: bool = True,
    icevel: bool = True,
    oceanvel: bool = False,
    all: bool = False,
) -> Optional[None]:
    """
    Export a Dataset to a netCDF file.

    Parameters
    ----------
    ds : xr.Dataset
        The xarray Dataset to export.
    file_path : str
        The file path for the output netCDF file.
    export_vars : Optional[List[str]], optional
        A list of variables to include in the exported netCDF file.
        Default is an empty list.
    icedraft : bool, optional
        If True, include sea ice draft estimates in the export.
        Default is True.
    icevel : bool, optional
        If True, include sea ice drift velocities in the export.
        Default is True.
    oceanvel : bool, optional
        If True, include ocean velocities in the export. Default is False.
    all : bool, optional
        If True, include all variables from the Dataset in the export.
        Default is False.

    Returns
    -------
    Optional[None]
        Returns None if the function completes successfully or if no variables
        are selected for export.

    Raises
    ------
    UserWarning
        If a specified variable in `export_vars` is not found in the Dataset.
    """

    dsc = ds.copy()

    if all:
        print("Saving *ALL* variables..")
        dsc.to_netcdf(file_path)
        print(f"Saved data to file:\n{file_path}")
    else:
        varlist = export_vars.copy() if export_vars else []
        if icedraft:
            varlist += [
                "SEA_ICE_DRAFT_LE",
                "SEA_ICE_DRAFT_MEDIAN_LE",
                "SEA_ICE_DRAFT_AST",
                "SEA_ICE_DRAFT_LE",
            ]
        if icevel:
            varlist += ["uice", "vice", "Uice", "Vice"]
        if oceanvel:
            varlist += ["uocean", "vocean", "Uocean", "Vocean"]

        # Remove duplicates
        varlist = list(np.unique(np.array(varlist)))

        # Remove variables not in ds
        for varnm_ in varlist.copy():
            if varnm_ not in dsc.variables:
                varlist.remove(varnm_)
                if export_vars and varnm_ in export_vars:
                    warnings.warn(
                        f"No field '{varnm_}' found in the data "
                        "(exporting file without it). Check spelling?"
                    )

        # If varlist is empty: Print a note and exit
        if not varlist:
            print("No existing variables selected -> Not exporting anything.")
            return None

        # Delete some none-useful attributes
        for attr_not_useful in ["pressure_offset"]:
            del dsc.attrs[attr_not_useful]

        # Saving
        dsc[varlist].to_netcdf(file_path)
        print(f"Saved data to file:\n{file_path}")
