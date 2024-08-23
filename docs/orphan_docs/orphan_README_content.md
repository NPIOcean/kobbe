Had written a bunch of stuff for the README.md page.

This was at an earlier stage, so most of it is deprecated now - but there may still be ome useful stuff in here.

_____





Core functionality
------------------

**Read one or multiple (** *.mat* **) data files and transform to Python
dictionary**

- Reads and concatenates data from *.mat*-files produced using Nortek's [SignatureDeployment](https://www.nortekgroup.com/software)
software from
  *.ad2cp* data files output by the instrument.

- Stores the data in an [xarray Dataset](https://docs.xarray.dev/en/stable/generated/xarray.Dataset.html).

- Present version does not read *Burst* or *Waves* data - only *Average*
  (altimeter, ocean velocities) and *AverageIce* (ice velocities).

- The single ``Average_Time`` dimension is reshaped to two dimensions (``TIME``:
  time stamp of each ensemble, and ``SAMPLE``: number of sample in ensemble).

  - This is useful because the instrument usually samples in ensembles, e.g.
    collects 50 samples at 1 Hz once every 20 minutes. We typically want to do
    statistics on each ensembles to arrive at one value per ``TIME``.


**Estimate sea ice presence based on the** *Figure-of-Merit* **metric**

- Classifies each sample as ice/no ice based on the Figure-of-Merit (FOM) value
  of the ice velocity measurement of the four slanted ADCP beams. FOM is an
  indication of the Doppler noise of each ping; low values indicate ice and high
  values indicate open water.

- Sea ice presence is confirmed if FOM of all four beams is below a set
  threshold (default = 10 000). An estimate of "sea ice concentration" is
  calculated as the fraction of samples within an ensemble classified as ice.

  (*NOTE: This "sea ice concentration" is most meaningful when averaged over a
  longer time period, e.g. daily.*)

- Alternative estimates of sea ice presence/concentration (suffix ``_FOM``) are
  made by a less conservative criterion: requiring that FOM of *at least one* of
  four beams is below the threshold. These are not recommended as they tend to
  give false positives for ice.


**Append data from external sources**

- Take for example a record of ocean temperature from another instrument, or sea
  ice concentration from remote sensing product, and add it to the sig500
  dataset interpolated onto the ``TIME`` grid. This is useful for adding CTD
  variables (for sound speed corrections) or atmospheric pressure (for
  instrument depth correction), but can also be useful for analysis of sig500
  data in combination with e.g. remote sensing products.

**Calculate ice draft based on altimeter data**

- Choice of using either of the two peak finding algorithms:

  - *Leading edge (LE)*: Recommended by Nortek for ice applications. However,
    often gives false targets close to the transducer (due to organisms in the
    water column?). This is solved by rejecting points where the LE distance
    deviates greatly from the *AST* distance (which does not have the same issue
    with deep false targets).
  - *Amplitude maximum (Acoustic Surface Tracking, AST)*: Recommended for open
    ocean applications by Nortek. Noisier point-by-point than LE. Typically not
    producing false deep targets, so a larger share of data points can be
    retained for analysis. Absolute values in open water appear to be closer to
    zero than with LE, so a smaller correction is necessary.


- Atmospheric pressure (from e.g. atmospheric reanalysis) and preferably ocean
  salinity and temperature (from e.g. a moored CTD sensor) should be supplied in
  order to get good quality results.
- Ice presence is determined based on the FOM (Figure-of-Merit) reading of the
  slanted beams.

- **Open water sound speed correction**
  - Ad hoc correction which forces the open water mean sea level to be close to zero.
    A correction factor is applied to the sound speed and altimeter distances are recomputed.


**Basic editing/post-processing of ice and ocean velocity (TBW)**

- Applies magnetic declination correction (rotating the velocity vector).
- Thresholds of FOM, quality, and speed used to filter out bad ice drift data.
- Various thresholds (speed, quality) used to filter out ocean velocity data. -
  Near- and above-surface measurements

**Ensemble processing (TBW - partially completed)**

- Mean or median averaging to arrive at a single measurement per ensemble (per
  bin).
- Individual ensembles are retained only if they pass various quality tests and
  tests of internal consistency.


**Chopping of time points and depth bins (TBW)**

- Easy chopping of depth ADCP bins or simgle time entries.

**Depth interpolation (TBW)**

- Interpolation fixed-depth.

**Analysis and visualization (TBW)**

- Print basic statistics of the dataset.
- Some very basic plots.

(Not intended to contain advanced analysis tools - just what is necessary during processing)

**Export (TBW)**

- Functionality to export to smaller "analysis" xarray Dataset where
  unused variables are removed.
- Functionality to export to netCDF file (.nc).

  - *NOTE:* The resulting file *should* be formatted according to CF conventions.
    Do check your dataset closely before publication, however.

______

*Signature_PyProc* is a Python package, and requires Python 3 (will not work on
2, and has currently only been tested on 3.8).

**Standard libraries:**

- ``numpy``
- ``scipy``
- ``matplotlib``
- ``warnings``

**Other:**

-  [xarray](https://docs.xarray.dev/en/stable/) - data are stored and
  manipulated as xarray *Dataset* objects.
- `GSW-Python <https://teos-10.github.io/GSW-Python/>`_ - used for computation
  of depth from pressure as well as density/sound speed/etc from CTD
  measurements.
