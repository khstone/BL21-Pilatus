# BL21-Pilatus
Python scripts for dealing with powder diffraction data collected on the Pilatus 100K detector installed at SSRL beamline 2-1.

Pilatus_Calibrate.py will use the files and paths specified in Pilatus_Calibration_setup.py to calibrate the sample to detector distance used to convert images to 2-theta.  This should point to a scan where a feature, either the direct beam with appropriate filtering or an isolated diffraction ring, were scanned across the center of the detector in small angular steps.  

Pilatus_Integrate.py will use the files and paths specified in Pilatus_Integration_setup.py to find images and scan data to integrate images into a single 1D diffraction pattern.  The desired 2-theta step size can be changed, although the default of 0.01 degrees seems to work well most of the time.  Steps smaller than 0.005 degrees should not be necessary.  Regions of the Pilatus image may be excluded by setting them to the value of -2.0, this is done in the read_RAW function to exclude the topmost and bottommost pixels which are most likely to be problematic and will have significant overlap between images anyway.
