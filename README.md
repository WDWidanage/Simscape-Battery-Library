# Simscape-Battery-Library
A Li-ion battery library written in Matlab Simscape language (Matlab 2022b). Battery models include: 
- a thermally coupled Single Particle Model with Electrolyte (TSPMe) 
- a thermally coupled Equivalent Circuit Model with Diffusion (TECMD) 
- a thermally coupled Equivalent Circuit Model (first order) with look-up-tables for the parameters (TECM) 
- a few utility components (a simple battery cycler with CCCV charging and a general electrical source - I/V/P)

# Installation
- Clone (or download) the repository. 
- Add the top level directory in to the Matlab path

# Usage
- See "simpleExamples" folder for running the models with utility components directly in Simulink
- See "viaMatlabCode" for examples that call the model to perform various simulations, e.g. Power discharge, Remaining Useful-Energy calculations, etc
- A parameter estimation example of the TECMD model is also provided (can be a lot further improved)
