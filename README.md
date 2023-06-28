# Simscape-Battery-Library [![Open in MATLAB Online](https://www.mathworks.com/images/responsive/global/open-in-matlab-online.svg)](https://matlab.mathworks.com/open/github/v1?repo=WDWidanage/Simscape-Battery-Library) [![View A Simscape-Battery-Library on File Exchange](https://www.mathworks.com/matlabcentral/images/matlab-file-exchange.svg)](https://uk.mathworks.com/matlabcentral/fileexchange/102079-a-simscape-battery-library)

A Li-ion battery library written in Matlab Simscape language (Matlab 2023a). Battery library includes: 
1. A thermally coupled Equivalent Circuit Model (first order) with look-up-tables for the parameters (**TECM**)
2. A thermally coupled Equivalent Circuit Model with Diffusion (**TECMD**)
3. A thermally coupled Single Particle Model with electrolyte (**TSPMe**)   
4. A thermally coupled Single Particle Model with electrolyte and Ageing (**TSPMeA**). Ageing includes SEI growth, Li-plating and negative electrode porosity reduction. 
- A few utility components to define the interface conditions with the battery models
    - a simple battery cycler with CCCV charging, 
    - a general electrical source - I/V/P 
    - a controlled electrical source - I/V/P
-  Two general purpose functions `getVariables()` and `plotVariables()` in the +Utilities folder. `BatteryLibrary.Utilities.getVariables()` will extract the relevant time and distributed variables of any of the Battery Library models and  `BatteryLibrary.Utilities.plotVariables()` plots the signals. See "gettingStarted.mlx" in the "Examples\viaMatlabCode" directory on how to use these two functions. 

# Installation
- Clone (or download) the repository. 
- Add the top level directory ("BatterySimscape_Library") in to the Matlab [path](https://uk.mathworks.com/help/matlab/matlab_env/add-remove-or-reorder-folders-on-the-search-path.html) (you can do this via the Matlab command `>>addpath(path to BatterySimscape_Library)`).
- Once added to the path, to start using the libarary componets from any location in Matlab, run the command `>>BatteryLibrary_lib` which will open the battery library and the blocks can be dragged into a Simulink template (.slx or .mdl) file 
- You can also try all the examples online if you have a MathWorks account from here [![Open in MATLAB Online](https://www.mathworks.com/images/responsive/global/open-in-matlab-online.svg)](https://matlab.mathworks.com/open/github/v1?repo=WDWidanage/Simscape-Battery-Library). 
  Make sure to add the repo folder to the Matalb path online as well. You should have a folder called 'Simscape-Battery-Library', right click, go to 'Add to path' and select 'Selected Folder(s) and Subfolders'. 


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
