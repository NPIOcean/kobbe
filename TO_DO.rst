TO DO
------

Sea ice draft
''''''''''''''

- Basic functionality for draft seems to work well now! OW correction also seems ok! 
    - (Need to clean up and double check things etc, though. Maybe reconsider
      some naming practices..).

- Need to clarify LE vs AST. Questions for Nortek:
  - Why would there be an open-water difference?
  - Is a ~1% correction typical? 

- Function for cleaning draft data (crazy outliers, etc) - do this before
  ensemble averaging.
    - Include rejection based on quality criterion! (**X**)
       - May still want to do some other statistics? 

t

Ice velocity
''''''''''''
- Mask each SAMPLE by FOM criterion
- Basic outlier and parameter editing on SAMPLE data.
- Ensemble statistics (median?) - rejecting invalid ensembles.
- Magdec correction (can be applied after ensemble averaging).
- Consider a sound speed correction (but drop if it is comlpetely negligible anyways..)
- Save variables to xr dataset

Ocean velocity
''''''''''''''
- Compute bin depths from (updated) depth.
  - I don't think it's worth doing sound speed/density corrections to bin depths/velocity magnitude.
 
- Mask by sidelobe interference and above water measurements.
     - Account for ice draft here! (Probably the easiest approach.) 
     - Reject bins with nothing (<0.1%) in them.
  
- Mask SAMPLES based on the normal, successive threshold criteria
- Ensemble statistics (median?) - rejecting invalid ensembles.
- Magdec correction (can be applied after ensemble averaging).
- Save variables to xr dataset.

Other functionality
'''''''''''''''''''

- Chopping start/end (I think I have good starting point code for this somewhere)
- Basic visualisation/statistics (no need to go overboard)
- Export to reduced xr Dataset (drop non-useful stuff)
- Export to nc
 - For analysis - but maybe eventually also for publication with appropriate conventions.. 

- Interpolation of ocean velocity onto fixed depth (do this at the ensemble stage, 
  cumbersome and not too useful to do it to SAMPLEs)

Documentation
''''''''''''''

- Look over README.md.
- Make working example in README.md
- Make more detailed/realistic example (notebook) 
- Look into sphinx automatic documentation from functions..

Maybe
'''''

- Look into burst stuff
- Time means etc
- Packaging for install