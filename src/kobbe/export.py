import numpy as np
import xarray as xr
from kval.metadata import conventionalize, check_conventions
from kval.util import time
from kobbe.metadata_dicts import _variables
from typing import List, Optional
import warnings


def _add_range_attrs(ds: xr.Dataset) -> xr.Dataset:
    """
    Add time, latitude, longitude, and vertical range attributes to the
    dataset.

    Args:
        ds (xr.Dataset): The input xarray Dataset.

    Returns:
        xr.Dataset: The updated Dataset with added range attributes.
    """
    ds = conventionalize.add_range_attrs(ds)

    ds.attrs['time_coverage_resolution'] = (
        time.seconds_to_ISO8601(ds.INSTRUMENT.time_between_ensembles_sec))

    if 'UCUR' in ds or 'Uocean' in ds:
        ds.attrs["geospatial_vertical_min"] = 0
        ds.attrs["geospatial_vertical_max"] = float(ds.BIN_DEPTH.max())
    else:
        ds.attrs["geospatial_vertical_min"] = 0
        ds.attrs["geospatial_vertical_max"] = 0
        ds.attrs["geospatial_vertical_units"] = 'm'

    ds.attrs["geospatial_vertical_positive"] = "down"
    ds.attrs["geospatial_bounds_vertical_crs"] = "EPSG:5831"

    return ds


def _add_global_attrs(ds: xr.Dataset) -> xr.Dataset:
    """
    Add global metadata attributes to the dataset.

    Args:
        ds (xr.Dataset): The input xarray Dataset.

    Returns:
        xr.Dataset: The updated Dataset with added global attributes.
    """
    ds.attrs["standard_name_vocabulary"] = "CF Standard Name Table, Version 86"
    ds.attrs["source"] = "In-situ measurements from subsurface moored acoustic sensor"
    ds.attrs["processing_level"] = "Data have undergone automatic and manually guided QC and processing"
    ds.attrs["Conventions"] = "CF-1.8, ACDD-1.3"
    ds.attrs["platform"] = 'Water-based Platforms>Buoys>Moored>MOORINGS'
    ds.attrs["sensor_mount"] = "mounted_on_mooring_line"

    ds.attrs["instrument"] = (
        "In Situ/Laboratory Instruments>Profilers/Sounders>Acoustic Sounders>"
        "UPWARD LOOKING SONAR,"
        "In Situ/Laboratory Instruments>Profilers/Sounders>Acoustic Sounders>"
        "ADCP")

    if 'declination_correction' in ds.attrs:
        del ds.attrs['declination_correction']

    return ds


def _add_variable_attrs(ds: xr.Dataset) -> xr.Dataset:
    """
    Add standard attributes (e.g., `standard_name`, `long_name`, `units`) to
    variables in the dataset.

    Args:
        ds (xr.Dataset): The input xarray Dataset.

    Returns:
        xr.Dataset: The updated Dataset with added variable attributes.
    """
    for var_key, var_dict in _variables.var_dict.items():
        if var_key in ds:
            for key, item in var_dict.items():
                ds[var_key].attrs[key] = item

    ds.TIME.attrs['axis'] = 'T'

    return ds


def _choose_LE_or_AST(ds: xr.Dataset, LE_AST: str = 'LE') -> xr.Dataset:
    """
    Retain either LE- or AST-based sea ice draft measurements, discarding the
    other.

    Args:
        ds (xr.Dataset):
            The input xarray Dataset.
        LE_AST (str):
            Choose 'LE' or 'AST' to retain the respective
            measurements.

    Returns:
        xr.Dataset:
            The updated Dataset with the chosen sea ice draft
            measurements.

    Raises:
        Exception: If an invalid `LE_AST` value is provided.
    """
    if LE_AST == 'LE':
        drop_name = 'AST'
    elif LE_AST == 'AST':
        drop_name = 'LE'
    else:
        raise Exception(f'Invalid `LE_AST` value "{LE_AST}". Valid options: "LE", "AST"')

    rename_vars = {
        f'SEA_ICE_DRAFT_{LE_AST}': 'SEA_ICE_DRAFT',
        f'SEA_ICE_DRAFT_MEDIAN_{LE_AST}': 'SEA_ICE_DRAFT_MEDIAN'
    }

    ds = ds.rename_vars(rename_vars)
    ds = ds.drop_vars([f'SEA_ICE_DRAFT_{drop_name}', f'SEA_ICE_DRAFT_MEDIAN_{drop_name}'])

    return ds


def _ice_conc_to_frac(ds: xr.Dataset) -> xr.Dataset:
    """
    Replace SIC_FOM (percentage) variable with SEA_ICE_FRACTION (fraction).

    Args:
        ds (xr.Dataset): The input xarray Dataset.

    Returns:
        xr.Dataset: The updated Dataset with sea ice fraction.
    """
    ds = ds.rename_vars({'SIC_FOM': 'SEA_ICE_FRACTION'})
    ds['SEA_ICE_FRACTION'].values = ds.SEA_ICE_FRACTION * 1e-2
    ds['SEA_ICE_FRACTION'].attrs['units'] = '1'

    for key, item in _variables.var_dict['SEA_ICE_FRACTION'].items():
        ds['SEA_ICE_FRACTION'].attrs[key] = item

    return ds


def _add_gmdc_keywords(ds: xr.Dataset) -> xr.Dataset:
    """
    Add GMDC (Global Marine Data Commons) keywords to the dataset.

    Args:
        ds (xr.Dataset): The input xarray Dataset.

    Returns:
        xr.Dataset: The updated Dataset with GMDC keywords.
    """
    return conventionalize.add_gmdc_keywords_moor(ds)


def check_cf(ds: xr.Dataset, close_button: bool = True) -> None:
    """
    Check the dataset's compliance with CF and ACDD formatting using the IOOS
    compliance checker.

    Args:
        ds (xr.Dataset):
            The input xarray Dataset.
        close_button (bool):
            Whether to include a close button in the compliance checker output. Default is True.
    """
    if close_button:
        check_conventions.check_file_with_button(ds)
    else:
        check_conventions.check_file(ds)


def to_nc(
    ds: xr.Dataset,
    file_path: str,
    export_vars: Optional[List[str]] = None,
    icedraft: bool = True,
    icevel: bool = True,
    oceanvel: bool = False,
    all: bool = False,
    include_latlon: bool = True,
    include_INSTRUMENT: bool = True,
    verbose: bool = True
) -> Optional[None]:
    """
    Export a Dataset to a netCDF file.

    Args:
        ds (xr.Dataset):
            The xarray Dataset to export.
        file_path (str):
            The file path for the output netCDF file.
        export_vars (Optional[List[str]], optional):
            A list of variables to include in the exported netCDF file.
        icedraft (bool, optional):
            Whether to include sea ice draft estimates in the export. Default is True.
        icevel (bool, optional):
            Whether to include sea ice drift velocities in the export. Default is True.
        oceanvel (bool, optional):
            Whether to include ocean velocities in the export. Default is False.
        all (bool, optional):
            Whether to include all variables from the Dataset in the export. Default is False.
        include_latlon (bool, optional):
            Whether to include the LATITUDE and LONGITUDE variables.
        include_INSTRUMENT (bool, optional):
            Whether to include the INSTRUMENT variable containing sampling/instrument info.
        verbose (bool, optional):
            Whether to print a message after successfully saving the file. Default is True.

    Returns:
        Optional[None]: Returns None if the function completes successfully or if no variables are selected for export.

    Raises:
        UserWarning: If a specified variable in `export_vars` is not found in the Dataset.
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
                "SEA_ICE_DRAFT_LE", "SEA_ICE_DRAFT_MEDIAN_LE",
                "SEA_ICE_DRAFT_AST", "SEA_ICE_DRAFT_MEDIAN_AST",
                "SEA_ICE_DRAFT", "SEA_ICE_DRAFT_MEDIAN"
            ]
        if icevel:
            varlist += ["uice", "vice", "Uice", "Vice", "UICE", "VICE"]
        if oceanvel:
            varlist += ["uocean", "vocean", "Uocean", "Vocean", "UCUR", "VCUR"]

        if include_latlon:
            varlist += ['LATITUDE', 'LONGITUDE']

        if include_INSTRUMENT:
            varlist += ['INSTRUMENT']

        varlist = list(np.unique(np.array(varlist)))

        for varnm_ in varlist.copy():
            if varnm_ not in dsc.variables:
                varlist.remove(varnm_)
                if export_vars and varnm_ in export_vars:
                    warnings.warn(
                        f"No field '{varnm_}' found in the data "
                        "(exporting file without it). Check spelling?"
                    )

        if not varlist:
            print("No existing variables selected -> Not exporting anything.")
            return None

        dsc[varlist].to_netcdf(file_path)
        if verbose:
            print(f"Saved data to file:\n{file_path}")


def _reorder_vars(ds: xr.Dataset) -> xr.Dataset:
    """
    Reorder the variables in the dataset based on a predefined order.

    Args:
        ds (xr.Dataset): The input xarray Dataset.

    Returns:
        xr.Dataset: The reordered Dataset.
    """
    ordered_vars = [var for var in _variables.variable_order if var in ds.data_vars]
    remaining_vars = [var for var in ds.data_vars if var not in _variables.variable_order and var != "INSTRUMENT"]

    if "INSTRUMENT" in ds.data_vars:
        reordered_ds = ds[ordered_vars + remaining_vars + ["INSTRUMENT"]]
    else:
        reordered_ds = ds[ordered_vars + remaining_vars]

    return reordered_ds
