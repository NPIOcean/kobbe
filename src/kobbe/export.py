'''
KOBBE.EXPORT

Functions for

- Assign

'''

import numpy as np
import xarray as xr
from kval.metadata import conventionalize, check_conventions
from kval.util import time
from kobbe.metadata_dicts import _variables
from typing import List, Optional
import warnings

def _add_time_geo_attrs(ds):

    ds = conventionalize.add_range_attrs(ds)
    ds.attrs['time_coverage_resolution'] = (
        time.seconds_to_ISO8601(ds.INSTRUMENT.time_between_ensembles_sec))

def _add_variable_attrs(ds):
    '''
    Add variable attributes (standard_name, long_name, units,
    coverage_content_type).
    '''

    for var_key, var_dict in _variables.var_dict.items():
        for key, item in var_dict.items():
            ds[var_key].attrs[key] = item

def _choose_LE_or_AST(ds, LE_AST= 'LE'):
    '''
    Choose to retain either the LE- or AST-based sea ice draft measurements,
    discard the other.

    Rename the chosen variables from

        `SEA_ICE_DRAFT_{LE_AST}`,
        `SEA_ICE_DRAFT_MEDIAN_{LE_AST}'`
    to
        `SEA_ICE_DRAFT`,
        `SEA_ICE_DRAFT_MEDIAN'`
    '''

    if LE_AST == 'LE':
        drop_name = 'AST'
    elif LE_AST == 'AST':
        drop_name = 'LE'
    else:
        raise Exception(
            f'Invalid `LE_AST` value "{LE_AST}". Valid options: "LE", "AST"')

    rename_vars = {
        f'SEA_ICE_DRAFT_{LE_AST}': 'SEA_ICE_DRAFT',
        f'SEA_ICE_DRAFT_MEDIAN_{LE_AST}': 'SEA_ICE_DRAFT_MEDIAN',}

    ds = ds.rename_vars(rename_vars)

    ds = ds.drop_vars([f'SEA_ICE_DRAFT_{drop_name}',
                       f'SEA_ICE_DRAFT_MEDIAN_{drop_name}'])

    return ds


def _ice_conc_to_frac(ds):
    '''
    Replace the SIC_FOM variable (%) with SEA_ICE_FRACTION (fraction).

    For export; conforms more with CF.
    '''

    ds = ds.rename_vars({'SIC_FOM': 'SEA_ICE_FRACTION'})
    ds['SEA_ICE_FRACTION'].values = ds.SEA_ICE_FRACTION*1e-2
    ds['SEA_ICE_FRACTION'].attrs['units'] = 1

    for key, item in _variables.var_dict['SEA_ICE_FRACTION'].items():
        ds['SEA_ICE_FRACTION'].attrs[key] = item

    return ds

def _add_gmdc_keywords(ds):
    ds = conventionalize.add_gmdc_keywords_moor(ds)
    return ds


def check_cf(ds, close_button = True):
    '''
    Use the IOOS compliance checker
    (https://github.com/ioos/compliance-checker-web)
    to check the CF and ACDD formatting.
    '''

    if close_button:
        check_conventions.check_file_with_button(ds)
    else:
        check_conventions.check_file(ds)




##############################################################################

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
    verbose: bool = True,
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
    include_latlon: bool, optional
        Include the LATITUDE and LONGITUDE variables.
    include_INSTRUMENT: bool, optional
        Include the INSTRUMENT variable containing sampling/instrument info.
    verbose: bool, optional
        Whether to print a little statement after successful save.
        Default is True.

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
                "SEA_ICE_DRAFT_MEDIAN_AST",
                "SEA_ICE_DRAFT",
                "SEA_ICE_DRAFT_MEDIAN",_
            ]
        if icevel:
            varlist += ["uice", "vice", "Uice", "Vice", "UICE", "VICE"]
        if oceanvel:
            varlist += ["uocean", "vocean", "Uocean", "Vocean", "UCUR", "VCUR"]


        if include_latlon:
            varlist += ['LATITUDE', 'LONGITUDE']

        if include_INSTRUMENT:
            varlist += ['INSTRUMENT']

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
        if verbose:
            print(f"Saved data to file:\n{file_path}")


def _reorder_vars(ds: xr.Dataset) -> xr.Dataset:
    # Get the variables present in the dataset and in the variable_order list
    ordered_vars = [var for var in _variables.variable_order if var in ds.data_vars]

    # Get the variables that are in the dataset but not in the variable_order list or INSTRUMENT
    remaining_vars = [var for var in ds.data_vars if var
                      not in _variables.variable_order
                      and var != "INSTRUMENT"]

    # If INSTRUMENT exists in the dataset, append it at the end
    if "INSTRUMENT" in ds.data_vars:
        reordered_ds = ds[ordered_vars + remaining_vars + ["INSTRUMENT"]]
    else:
        reordered_ds = ds[ordered_vars + remaining_vars]

    return reordered_ds