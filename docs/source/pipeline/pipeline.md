# Processing pipeline

The following describes a typical `kobbe` processing pipeline for ice and ocean data collected using a moored 5-beam Nortek *Signature250* or *Signature500* instruments.

It describes reading Signature files into the `kobbe` environment, appending auxiliary data if available, and applying post-processing steps editing the data. Finally, metadata are edited  to conform with scientific formatting standards, and the final dataset is exported as a netCDF file.



___

<details>
<summary><b>1. Convert data in <i>SignatureDeployment</i></b></summary>
<p>

Converting from `.ad2cp` file (uploaded from the instrument) to `.mat` file (containing physical variables). The conversion is done in Nortek's *SignatureDeployment* software, outside of the `kobbe` environment.

For long deployments, the export procedure may result in several `.mat`-files per `.ad2cp` file.

</p>
</details>



___

<details>
<summary><b>2. Parse to Python xarray Dataset</b></summary>
<p><br>

- `kobbe.load.matfiles_to_dataset()`
- Loads data from `.mat` files and joins them along a single `Average_TIME` dimension.
- Renames variables and appends relevant metadata.
- Regrids from `Average_TIME` to 2D (`TIME`, `SAMPLE`):

 <a href="../_static/proc_images/kobbe_1d_to_2d.png" target="_blank">
    <img src="../_static/proc_images/kobbe_1d_to_2d.png" width="250" height="100" alt="SIC Spectra">
  </a>

- Calculates absolute tilt from pitch roll.
</p>
</details>


___

<details>
<summary><b>3. Estimate sea ice presence from Figure-of-Merit</b> <i>(automatic within #2)</i></summary>
<p>

- Currently in `kobbe.load.matfiles_to_dataset()`
    - Calls `kobbe.append._add_SIC_FOM`
    - (Should probably force an explicit call for this?)

<div style="display: flex; justify-content: space-between;">
  <a href="../_static/proc_images/FOM.png" target="_blank">
    <img src="../_static/proc_images/FOM.png" width="250" height="100" alt="FOM">
  </a>
  <a href="../_static/proc_images/SIC_spectra.png" target="_blank">
    <img src="../_static/proc_images/SIC_spectra.png" width="250" height="100" alt="SIC Spectra">
  </a>
</div>
</p>
</details>


___

<details>
<summary><b>4. Append external data</b></summary>
<p>

- Various functions in (`kobbe.append`):
    - `append_ctd()` - CTD data if available
    - `append_atm_pres()` - Atmospheric pressure from e.g. reanalysis
    - `append_magdec()` - Magnetic declination data
    - `append_to_sigdata()` - Other contextual data (remote sensing SIC/SIT, for example)
- Interpolates onto `TIME` grid.
- Format and names are standardized for subsequent use in post-processing operations.

</p>
</details>


___
<details>
<summary><b>5. Calculate transducer depth from pressure
</b></summary>
<p>

- `kobbe.calc.dep_from_p()`
  1. $p_{ABS}$ = `Average_AltimeterPressure` + `conf.PressureOffset`
        - $p_{ABS}$: Total pressure measured at transducer.
  2. $p = p_{ABS} - p_{ATMO}$
        - $p_{ATMO}$: Atmospheric pressure; fixed or from e.g. reanalysis.
  3. $\rho$ calculated from e.g. co-mounted CTD.
        - Automatically if CTD data are appended.
        - A fixed value can be specified if no CTD available.
  4. $g$ calculated as $g(\text{latitude})$ using the `gsw` package.

  5. $\ \ \Large{D = \frac{p}{g \rho}}$
    - $D$ is the (time-varying) depth of the instrument transducer head below the sea surface (meters), calculated using the hydrostatic approximation.


</p>
</details>

___

<details>
<summary><b>6. Calculate initial estimate of sea ice draft
</b></summary>
<p>

- `kobbe.icedraft.calculate_draft()`

The vertical distance between the transducer head and the scattering surface detected by the vertical beam, $S_v$ is taken as `Average_AltimeterDistance` (LE or AST) after applying some corrections:

> $S_v$ =
>`Average_AltimeterDistance`
>$\cdot \cos \theta \cdot c_{S, OBS}$/`Average_Soundspeed`$\cdot \beta_{OW}$


- $S_v$: Vertical distance between transducer and scattering surface:
- $\theta$: Instrument tilt (computed in #2).
- $c_{S, OBS}$: Sound speed calculated from sensor
    - (..if available - otherwise, this term is set to 1).
- `Average_Soundspeed`: Sound speed calculated in the Signature500 onboard algorithm (time varying as a function of measured temperature)
- $\beta_{OW}$: Time-varying, empirically derived "open water correction" coefficient. Set to 1 during the initial estimate.

A quality parameter `Average_AltimeterQualityLE/AST` is associated with each `Average_AltimeterDistance` sample. We apply this automatic quality flag by setting $S_v$ to NaN wherever `Average_AltimeterQualityLE/AST` is below a certain thrreshold (default value 8000).

The vertical position $z_S$ of the scattering surface relative to the sea surface (positive downward) is computed from $S_v$: and depth $D$:

> $z_S = D - S_v$

$z_S$ (stored in the variables `SURFACE_DEPTH_LE/AST`) includes measurements of:

- *In open water*: The position of the water surface (should be close to zero on average, but may vary due to waves).
- *In sea ice*: The sea ice draft (vertical distance between  the water surface and the bottom of the sea ice).

Sea ice draft (variables `SEA_ICE_DRAFT_LE/AST`) is equal to $z_S$, but only includes measurements from samples where ice-presence was detected (using the FOM criterion in all four beams). $z_S$ from any open-water or mixed measurements is set to NaN.

In addition, any sea ice draft estimates with values < 30 cm are considered erroneous and removed (set to NaN).

Sea ice draft variables (`SEA_ICE_DRAFT_LE/AST`) are computed for each sample and have dimensions (`TIME`, `SAMPLE`). From these, we compute the median of valid sea ice draft estimates per ensemble and assign them to variables `SEA_ICE_DRAFT_MEDIAN_LE/AST` (with dimensions `TIME`).

</p>
</details>


___

<details>
<summary><b>7. Filter LE distances to reject false near-transducer ice keels </b> <i>(automatic within #6)</i></summary>
<p>


In the LE distance data (`Average_AltimeterDistanceLE`), we typically observe a large number distances that are clearly in the water column between the transducer and the ice or ocean surface, resulting in a broad peak within 0-10 m of the transducer head. This near-transducer peak (referred to here as "false keels") is typically not present in AST distances.

<a href="../_static/proc_images/AST_LE_histograms.png" target="_blank">
    <img src="../_static/proc_images/AST_LE_histograms.png" width="200" height="120" alt="Example of distribution of LE and AST distances">
</a>

We interpret near-transducer values an artifact, i.e. *not* as resulting from very deep ice keels. A likely suspect is the LE algorithm detecting zooplankton or other material in the water column.

We do not want to include such near-transducer "false ice keels" in estimates of sea ice draft. As rough quality control of the LE distances, **we require that the LE distance is within a certain distance (default 0.5 m) of the AST distance.**
This provides an effective filter of false ice keels from the LE dataset.

 - The maximum allowed distance between AST and LE is set
 using the `LE_AST_max_sep` parameter in  `icedraft.calculate_draft()`.

In many instances, this may result in removing quite large parts of the LE distances in the datasets. For example,
in datasets from Sig500s mounted near 20 m depth in th northwestern Barents Sea, this reduces the amount of valid LE measurements by 1/4 to 1/3.



</p>
</details>


___

<details>
<summary><b>8. Get time-varying open water sound speed correction</b></summary>
<p>

**Sources of error** include:

- Erroneous effective sound speed along the acoustic travel path because T and S above the sensor may be *lower* than measured at the Sig500.

    - *Example*:
    - Meltwater layer above the sensor where T, S is lower
    - -> True sound speed is lower than estimated.
    - -> True travel distance is shorter than estimated.
    - -> True ice draft is deeper than estimated.
    - **NOTE: Doe BOE calculation here!!**

- Erroneous instrument pressure
    - From [specs](https://www.nortekgroup.com/products/signature-500/pdf): Error is 0.1% of full scale
    - At 20 dbar, this should give an error of ~2 cm. This could give an error up to 4 cm in the travel distance.

- Erroneous atmospheric pressure
    - Could be errors, but we don't expect a *bias* in one direction or another..
    - A 30hPa error (equivalent to the atmo pressure being *totally* wrong) could give an error of 30 cm.

- Erroneous *g*


- Erroneous instrument tilt
    - From [specs](https://www.nortekgroup.com/products/signature-500/pdf): Error is 2 degrees
    - The associated error in the cos term should be            negligible  (~0.06%)

- Refectors before the surface?

    - Bubbles or similar..
    - Algorithm setting distance *at* point rather than in between




`use-alpha = True`??

- `kobbe.icedraft.get_Beta_from_OWSD()`
- USING A LONG-TIME AVERAGE OFFSET..
- SHOULD MAYBE BE fixed offset plus beta??
</p>
</details>


___

<details>
<summary><b>9. Recalculate sea ice draft, now using OW correction
</b></summary>
<p>

- `kobbe.icedraft.calculate_draft()`
- FORMULA!
- -> Should be done with ice draft here!

</p>
</details>

___

<details>
<summary><b>10. Calculate ice velocity</b></summary>
<p>

- `kobbe.vel.calculate_ice_vel()`
- Auto editing?
</p>
</details>


___

<details>
<summary><b>11. Calculate ocean velocity</b></summary>
<p>

- `kobbe.vel.calculate_ocean_vel()`
- Auto editing?
</p>
</details>


___


<details>
<summary><b>12. Ocean velocity editing</b></summary>
<p>

- Sidelobe rejection
    - `kobbe.vel.reject_sidelobe`
    - What happens here?
    - *"Rejected samples close enough to the surface to be affected by sidelobe interference"*
- Range masking
    - `kobbe.vel.uvoc_mask_range()`
    - Threshold for various parameters like corr, amp, speed, tilt, amp jumps...

- Clear near-empty bins
    - `kobbe.vel.clear_empty_bins()`
</p>
</details>



___

<details>
<summary><b>13. Magnetic declination correction</b></summary>
<p>

- `kobbe.vel.rotate_vels_magdec()`
- Rotates both ice and ocean velocities

</p>
</details>


___

<details>
<summary><b>14. Interpolate ocean velocity onto fixed depths</b><i> (Optional)</i></summary>
<p>

- `kobbe.vel.interp_oceanvel`

</p>
</details>


___

<details>
<summary><b>15. Format metadata using CF/ACDD conventions*</b></summary>
<p><br>

</p>
</details>

___

<details>
<summary><b>16. Export to netCDF*</b></summary>
<p><br>

</p>
</details>

___
___
___

### Q:

- Where is open water stuff calculated?
- Where is the filtering LE by LE-AST distance?
___

### Q Nortek:

___

### To do:

- Grab any useful metadata atthe loading stage!
