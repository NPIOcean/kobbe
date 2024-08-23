# `kobbe`

![](docs/images/logos_illustrations/kobbe_header.png)


Version 0.0.1: *Under development.*


Overview
--------
#### [Documentation page](https://kobbe.readthedocs.io/) *(in development)*


Post-processing and basic analysis of ice and ocean data from Nortek Signature
ADCPs.

Designed for applications where the instrument is deployed looking upward below
the ocean surface.

![](docs/images/logos_illustrations/sea_ice_illustration.png)


_____

  ### **GOAL:**

  **Easy and explicit post-processing to obtain scientific-quality estimates of:**

  - **Sea ice draft**
  - **Sea ice drift velocity**
  - **Upper ocean current velocity**
______


Inputs to the process are:

- *.mat* files exported by Norteks [SignatureDeployment](https://www.nortekgroup.com/software) software *(required)*
- Time series of atmospheric pressure during the deployment *(strongly
  recommended)*
- Time series of ocean temperature/salinity in the upper ocean *(recommended)*
- Magnetic declination correction (single value or time series) *(necessary for
  correct velocity directions)*
______
