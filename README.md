# Data-driven characterization of plates
This repository contains all the measurements, data, *Matlab* code and *Comsol Multiphysics* finite element (FE) model related to the article *"A neural network-based method for spruce tonewood characterization", Journal of Acoustic Society of America (JASA)* (David Giuseppe Badiane, Raffaele Malvermi, Sebastian Gonzalez, Fabio Antonacci, Augusto Sarti), which presents a neural network based inverse method to estimate the material properties of thin plates of spruce tonewood. 


# Introduction

  <img align="right" src="/Figures/wood_directions.png" width="250"> The methodology contained in this repository is called *FRF2Params*, as it is necessary to measure a Frequency Response Function (FRF) at prescribed points of a wooden plate surface (along with the plate geometry and density) to estimate the mechanical parameters of the plate (Params). 

Our focus is set on musical acoustic and instrument making, so we devised the method to indentify the elastic properties of thin rectangular plates of spruce tonewood used to build the soundboards of guitars. The figure beside shows how those plates are used to build the soundboards. As it can be seen, the soundboard is carved from two thin plates cut from the same spruce sample glued together. 

Wood is usually modeled as an ortothropic material. Ortothropic materials are characterized by nine elastic constants (3 Young's moduli, 3 shear moduli and 3 Poisson's ratios). The elastic constants are evaluated with reference to three characteristic directions, highlighted in the figure: the direction parallel to the wood fibers direction (*longitudinal*, L), the direction radial with respect to the wood growth rings (*radial*,R) and the direction tangential to the wood growth rings (*tangential*, T). Our aim is to estimate those parameters. 

The flow diagram of *FRF2Params* is reported in the figure below. First it is necessary to define a finite element model of the plate FRF and define where that FRF must be acquired. Once defined, the finite element model is used to generate a dataset containing the eigenfrequencies of the plate and their amplitude in the FRF as the elastic properties, the density, the geometry and the damping of the plate vary. Then, the dataset is used to train two neural networks, one for frequency and one for amplitude. In the meanwhile, the FRF of the plate under test must be acquired and it's peaks must be detected. Finally, the neural networks are employed in a optimization procedure to minimize the frequency distance between the peaks of the measured FRF and their predictions.

# Repository Description

As you can see, you can find three directories:
- **FRF2Params** --> this directory contains:
   - the main files containing all the code that must be executed in order to apply *FRF2Params* or to analyse the dataset;
   - the measured FRFs, geometry and density of ten book-matched thin plates of spruce;
   - the Comsol Multiphysics finite element model of the plate;
   - the generated dataset, with or without modes identification;
   - the estimations of the material parameters along with their associated uncertainties and simulated FRFs;
- **Functions** --> this directory contains all functions and modules employed in FRF2Params;
- **Figures** --> this directory contains the figures that you see in this *README.md* file. 


The dataset is cav_gplates, modeshapes are missing due to their size, the link is available here 


## FRF2Params

make a flow diagram with main codes name and their role

after the flow diagram describe each main program

state the name of every element you can find in the repo

## Functions

create a overleaf pdf file to include in the repo, the pdf file is the complete documentation, over there it will be possible to find the detailed description of each function







