# Simscape-Battery-Library
A Li-ion battery library written in Matlab Simscape language (Matlab 2023a). Battery models include: 
- a thermally coupled Equivalent Circuit Model (first order) with look-up-tables for the parameters (TECM)
- a thermally coupled Equivalent Circuit Model with Diffusion (TECMD)
- a thermally coupled Single Particle Model with Electrolyte (TSPMe)   
- a thermally coupled Single Particle Model with Electrolyte with ageing (TSPMeA). Ageing includes SEI growth, Li-plating and negative electrode porosity reduction. 
- a few utility components to define the interface conditions with the battery models
    - a simple battery cycler with CCCV charging, 
    - a general electrical source - I/V/P 
    - a controlled electrical source - I/V/P)

# Installation
- Clone (or download) the repository. 
- Add the top level directory ("BatterySimscape_Library") in to the Matlab [path](https://uk.mathworks.com/help/matlab/matlab_env/add-remove-or-reorder-folders-on-the-search-path.html) (you can do this via the Matlab command '>>addpath(path to BatterySimscape_Library)`).
- Once added to the path, to start using the libarary componets from any location, run the command `>>BatteryLibrary_lib` which will open the battery library and the blocks can be dragged into a Simulink template (.slx or .mdl) file 

# Usage
- See "simpleExamples" folder for running the models with utility components directly in Simulink
- See "viaMatlabCode" for examples that call the model to perform various simulations, e.g. Power discharge, Remaining Useful-Energy calculations, cycle ageing, etc
- A parameter estimation example of the TECMD model is also provided (can be a lot further improved, requires Simulink Design Optimisation Toolbox)

## Dependencies
- Simulink Design Optimisation Toolbox is required for the parameter estimation example

## Model references 
The TSPMe and TSPMeA models are result of the following two papers done in collaboration with Ferran Brosa-Planella. 
- [TSPMe](https://www.sciencedirect.com/science/article/pii/S0013468621008148)
- [TSPMeA](https://www.sciencedirect.com/science/article/pii/S0307904X22005959)

The Parameters for the physcial models are from the work done in collaboration with University of Birmingham
- [TSPMe parameters](https://iopscience.iop.org/article/10.1149/1945-7111/ab9050/meta)

The TECMD is a simplified version (only a single OCV + diffusion) of the work done in collaboration with Chuanxin Fan
- [TECMD](https://wrap.warwick.ac.uk/166065/1/WRAP-Data-driven-identification-of-lithium-ion-batteries-Fan-22.pdf)
