'''
KOBBE.EXPORT

Functions for

- Assign

'''

from kval.metadata import conventionalize, check_conventions
from kval.util import time

def _add_attrs(ds):

    ds.attrs['time_coverage_resolution'] = (
        time.seconds_to_ISO8601(ds.time_between_ensembles_sec))


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