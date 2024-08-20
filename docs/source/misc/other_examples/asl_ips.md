# ASL IPS processing

ASL IPS is the "industry standard" for IPS data. Going through a report of their processing to compare their processing procedure to the one in `kobbe`.

> Going by the report `PROCESSING OF ICE DRAFT, FRAM STRAIT, 2021-2022` prepared for the Norwegian Polar Institute by ASL Environmental Sciences Inc.

## About the data

An Ice Profiling Sonar (IPS) manufactured by ASL Environmental Sciences was deployed on the mooring F14-23 in the Fram Strait in August 2021 and recovered in September 2022. Also on the mooring were an RDI ADCP and a SBE37 CTD. The IPS was located at a water depth of 51 m.

The IPS transmits and receives a 420 Hz acoustic pulse. The beam width is 8 degrees at -3db. The data stored are more comprehensive than fir the Sig500, the full acoustic return signal is stored[^tag] , and several targets may be identified per png.

The deployment seems to have been successful and the data processing will presumably have been pretty standard.

[^tag]: Maybe not - the processing report states that the raw data consists of *a)* travel time to the selected target and *b)* the *persistence* ("duration above a threshold amplitude level") of the target eho.

## Processing

Converting time-of-travel measurement recorded internally by the IPS to time series of ice draft in units of meters.

Steps

1. Converting to physical units
2. Correct for clock drift
3. Manually review time series plot of all measured raw data
    - Focused on identifying instrument failures, data gaps, etc.
4. Removing on-deck values
5. Automatic removal of invalid targets
6. Correcting for *double bounce effects* (capturing an acoustic signal that has bounced through the water column)
7. Rejection (interpolation over)
    - *Drop-outs*; range values less than 1 cm (artifact when signal strength is weak)
    - *Artificially high values* where a clipping range is set as *instrument depth at hgh tide + buffer*
8. Marking areas with large waves (to protect it from later despiking)
9. Automatic despiking - standard despiking where the thresholds are refined by the person processing.
10. Patching the large wave segments from #8 back into the record
11. Manually review edited data for additional suspect values (bubbles, other artifacts)
12.  Calculate ice draft from range + tilt + estimated water level.
13. Adjusting ("calibrating") ice draft to adjust for variation in sound speed (open water correction)
14. Identify and flag segments of open water or contaminated by waves.
15. Rejecting (interpolating over) negative ice drift spikes >4.5 cm.

Automated steps resulting in rejecting larger (10-30s) periods of data are manually reviewed.

## Ice draft calculation

> draft = $\eta- \beta\cdot r\cdot \cos \theta$

Where  $\eta$ is the water level (transducer depth), $\beta$ is the sound speed correction, $r$ is the edited ice range, and $\theta$ the tilt.

Water level $\eta$ is calculated as

> $\eta = \frac{P_{IPS}-P_{ATM}}{\rho g}-\Delta D$

where $P_{ATM}$ comes from reanalysis or similar, and $\Delta D$ is the distance between the transducer head and the pressure sensor.

Water density $\rho$ is taken as a onstant, and gravitational acceleration $g$ is taken as the latitude-dependent $g$.

The correction for tilt gives corrections on the order of a few cm.

## Sound speed correction (open water correction)

$\beta$ varied through the deployment between a minumum of around 0.995 and a maximum of around 1.004, corresponding to an open water correction of up to $\pm$ 0.5% of the water depth, so around 25 cm at most. The mean value across the whole depoyment looks to have been around 0.998, corresponding to 0.2% of the water depth, so 10 cm. The mean value below ice seems to have been around 0.996, so around 20 cm.

It is not entirely clear from the report how the final beta correction is calculated. The equivalent sound speed correction was computed for all samples from open water or very thin ice (after correcting the sound speed).

> "***A fit*** *was made to determine the time varying [correction coefficient] based on the empirical computations of [the equivalent sound speed correction].."*

From the plot of the correction, it looks like beta has high (full?) temporal resolution where available. In ice, there are very few open water points, and it looks like the correction is interpolated over periods as long as 1-2 months.

## Wave filtering

A "composite ice draft" is produced by low pass filtering the data in order to remove the effect of surface waves on draft.

> *"A semi-automatic technique was used to search for a spectral gap or a frequency at which the spectral densities reached a minimum in the frequency domain between the ice and waves portions of the spectra"*

Clearly, they are not very specific about this. Based on my own spectra, I thin this cutoff frequency is probably be at 15- 20 s periods. Seems right ; generic SGWs spectra should peak in the 0.05 Hz to 0.5 Hz; this is where my "no-ice" ensemble spectra samples peak as well (note: expect lower frequency for higher wave speeds).

It looks like they **only do this in regions registered as** ***waves-in-ice***.

Both the draft time series with and withough filtering are included in the final datset. The final data product uses flags to tag the various classifications with separate flags.

- Ice *(included in draft)*
- Waves-in-ice *(included in draft)*
- Open water *(not included in draft)*
- Missing or bad data *(not included in draft)*

### Quick note

- Correction is on the order of what I do.
- Should consider at least looking at waves in ice (also interesting in itself..)
