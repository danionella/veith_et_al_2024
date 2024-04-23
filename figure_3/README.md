# Vibrometry

Matlab code to reproduce phase maps in Figure 3 and Extended Data Figure 11.

## Data
The data is available at https://gin.g-node.org/danionella/Veith_et_al_2024/src/master/vibrometry

## Code

Two Matlab scripts are used to generate these figures: 

1. motionDetect_Veith2024.m 
   
   * loads the data, 
   
   * reshapes the data set to reconstruct the movie of mechanical tissue deformation in response to acoustic stimulation,
   
   * sets particle image velocimetry (PIV) parameters including image pre-processing,
   
   * computes local displacement from one frame to the next along x and y directions, using the PIVlab Matlab library ([GitHub - Shrediquette/PIVlab: Particle Image Velocimetry for Matlab, official repository](https://github.com/Shrediquette/PIVlab), tested with version 2.31 - to use 3.0, a few new parameters must be defined, see code). 

2. phaseMapping_Veith2024.m
   
   * loads the motion data (output from previous code)
   
   * computes the first Fourier component, and convert to proper unit (velocities in um/s and displacement amplitude in um)
   
   * plots the results
   
   * compute mean values of displacement amplitude and phase within region of interests
