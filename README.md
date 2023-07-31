# Data-driven characterization of plates
This repository contains all the measurements, data, *Matlab* code and *Comsol Multiphysics* finite element (FE) model related to my (David Giuseppe Badiane) master thesis work and the article <!-- *"A neural network-based method for spruce tonewood characterization", Journal of Acoustic Society of America (JASA)* (David Giuseppe Badiane, Raffaele Malvermi, Sebastian Gonzalez, Fabio Antonacci, Augusto Sarti)
-->.

This repository stores the implementation of a novel neural network based inverse method to estimate the material properties of thin plates of spruce tonewood. 

The following README is only introductive - a complete and comprehensive description of this repository can be found [HERE](https://drive.google.com/file/d/16TtDSVASsElo3O6E2Eg_PLCec_CobRyA/view?usp=sharing) or in the repository itself --> *documentation.pdf*.

# Introduction

  <img align="right" src="/Figures/wood_directions.png" width="250"> The methodology contained in this repository is called *FRF2Params*, as it is necessary to measure a Frequency Response Function (FRF) at prescribed points of a wooden plate surface (along with the plate geometry and density) to estimate the mechanical parameters of the plate (Params). 

Our focus is set on musical acoustic and instrument making, so we devised the method to indentify the elastic properties of thin rectangular plates of spruce tonewood used to build the soundboards of guitars. The figure beside shows how those plates are used to build the soundboards. As it can be seen, the soundboard is carved from two thin plates cut from the same spruce sample glued together. 

Wood is usually modeled as an ortothropic material. Ortothropic materials are characterized by nine elastic constants (3 Young's moduli, 3 shear moduli and 3 Poisson's ratios). The elastic constants are evaluated with reference to the three characteristic directions highlighted in the figure beside: the direction parallel to the wood fibers direction (*longitudinal*, L), the direction radial with respect to the wood growth rings (*radial*,R) and the direction tangential to the wood growth rings (*tangential*, T). Our aim is to estimate those parameters. 

The flow diagram of *FRF2Params* is reported in the figure below. First it is necessary to define a finite element model of the plate FRF and define where that FRF must be acquired. Once defined, the finite element model is used to generate a dataset containing the eigenfrequencies of the plate and their amplitude in the FRF as the elastic properties, the density, the geometry and the damping of the plate vary. Then, the dataset is used to train two neural networks: one for frequency and one for amplitude. In the meanwhile, the FRF of the plate under test is estimated with the H1 estimator and its peaks are detected via peak analysis. Finally, the neural networks are employed in a optimization procedure to minimize the frequency distance between the peaks of the measured FRF and their predictions. The results are validated by computing a FE simulation of the FRF with the material parameters of the plate set to the values obtained with *FRF2Params*.

<img align="center" src="/Figures/method Flowchart.png">

# Repository Description

As you can see, you can find three directories:
- **FRF2Params** --> this directory contains:
   - the *main_~.m* files with all the code that must be executed in order to apply *FRF2Params* or to analyse the dataset;
   - the measured FRFs, geometry and density of 10 book-matched thin plates of spruce;
   - the Comsol Multiphysics finite element model of the plate;
   - the generated dataset, with or without modes identification;
   - the estimations of the material parameters along with their associated uncertainties and simulated FRFs;
- **Functions** --> this directory contains all functions and modules employed in FRF2Params;
- **Figures** --> this directory contains the figures that you see in this *README.md* file. 

The dataset directory name is *csv_gplates*, modeshapes are missing due to their size, the link is available [HERE](https://drive.google.com/file/d/1pHcqZKaihc7UNpUfCX5Sw652mwhAkiLH/view?usp=drive_link). modeshapes should be extracted in the folder *Modeshapes* in the *csv_gPlates* directory.

## FRF2Params
<img align="center" src="/Figures/code Flowchart.png">
<img align="center" src="/Figures/FRF2Params_dir_descr.png">

## Functions
<img align="center" src="/Figures/functions_dir_descr.png">
