# Data-driven characterization of plates
This repository contains all the measurements, data, *Matlab* code and *Comsol Multiphysics* finite element (FE) model related to my (David Giuseppe Badiane) master thesis work and the article <!-- *"A neural network-based method for spruce tonewood characterization", Journal of Acoustic Society of America (JASA)* (David Giuseppe Badiane, Raffaele Malvermi, Sebastian Gonzalez, Fabio Antonacci, Augusto Sarti)
-->.

This repository stores the implementation of a novel neural network based inverse method to estimate the material properties of thin plates of spruce tonewood. 

The following README is only introductive - a complete and comprehensive description of this repository can be found [HERE](https://drive.google.com/file/d/16TtDSVASsElo3O6E2Eg_PLCec_CobRyA/view?usp=sharing) or in the repository itself --> *documentation.pdf*.

## Introduction

  <img align="right" src="/Figures/wood_directions.png" width="250"> The methodology contained in this repository is called *FRF2Params*, as it is necessary to measure a Frequency Response Function (FRF) at prescribed points of a wooden plate surface (along with the plate geometry and density) to estimate the mechanical parameters of the plate (Params). 

Our focus is set on musical acoustic and instrument making, so we devised the method to indentify the elastic properties of thin rectangular plates of spruce tonewood used to build the soundboards of guitars. The figure beside shows how those plates are used to build the soundboards. As it can be seen, the soundboard is carved from two thin plates cut from the same spruce sample glued together. 

Wood is usually modeled as an ortothropic material. Ortothropic materials are characterized by nine elastic constants (3 Young's moduli, 3 shear moduli and 3 Poisson's ratios). The elastic constants are evaluated with reference to the three characteristic directions highlighted in the figure beside: the direction parallel to the wood fibers direction (*longitudinal*, L), the direction radial with respect to the wood growth rings (*radial*,R) and the direction tangential to the wood growth rings (*tangential*, T). Our aim is to estimate those parameters. 

The flow diagram of *FRF2Params* is reported in the figure below. First it is necessary to define a finite element model of the plate FRF and define where that FRF must be acquired. Once defined, the finite element model is used to generate a dataset containing the eigenfrequencies of the plate and their amplitude in the FRF as the elastic properties, the density, the geometry and the damping of the plate vary. Then, the dataset is used to train two neural networks: one for frequency and one for amplitude. In the meanwhile, the FRF of the plate under test is estimated with the H1 estimator and its peaks are detected via peak analysis. Finally, the neural networks are employed in a optimization procedure to minimize the frequency distance between the peaks of the measured FRF and their predictions. The results are validated by computing a FE simulation of the FRF with the material parameters of the plate set to the values obtained with *FRF2Params*.

<img align="center" src="/Figures/method Flowchart.png">

## Repository Description

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

## Getting Started

### Dependencies
- Prerequisites: MATLAB® 2020 or successive versions, Comsol Multiphysics 5.6 or successive versions

### Modules
The flow diagram shown before is implemented via five *main_.m* modules and a finite element model as follows:

<img align="center" src="/Figures/code Flowchart.png">

In the following we will describe each module.
#### main_FRF2Params.m
This module applies FRF2Params on ten book matched spruce tonewood plates.

- section 0) - init: sets up the Matlab search path, declares flags and variables, reads geometry and mass measurements;
- section 1 - FRFs + Caldersmith: loads the measured FRFs from *measFRFs.mat* and applies Caldersmith formulas to later compare it with the results of FRF2Params;
- section 2) - dataset generation: generates the dataset with *Comsol Livelink for Matlab*. The dataset is stored in *csv\_gPlates* with three .csv files, namely *inputs.csv, *outputsAmp.csv* and *outputsEig.csv*;
- section 3) - FRF2Params minimization: applies FRF2Params minimization to estimate the mechanical parameters of the plates;
- section 4) validation - eigenfrequencies : uses *Comsol Livelink for Matlab* to compute the plate eigenfrequencies as its mechanical parameters are set to the estimates obtained with *FRF2Params*. The eigenfrequencies are then compared to the frequency values of the peaks of the plate FRF;  
- section 5) - validation FRFs, simulation: uses *Comsol Livelink for Matlab* to compute the plate FRF over a user defined number of points as its mechanical parameters are set to the estimates obtained with *FRF2Params.
- section 5.1) - validation FRFs, postprocessing: visually compares the simulated FRF and the experimentally acquired one with a figure. Computes metrics to assess the similarity between the two.

#### main_hyperparams.m
This module optimizes the topology, i.e. the number of layers and number of neurons per layer, of the feedforward neural networks employed to predict the plate eigenfrequencies and the corresponding FRF amplitude. This task is also known as hyperparameters tuning. 

- section 0) - init: sets up the Matlab search path, declares flags and variables;
- section 1) - split dataset into train and test sets: randomly splits the dataset into train set and test set, saves them back in the directory *csv_gPlates/HyperParameters*;
- section 2) - hyperparameters tuning: performs hyperparameters tuning on both frequency and amplitude neural networks;
- section 3) - max and min R2: finds the best and the worst architectures for both frequency and amplitude neural networks;
- section 4) - Train optimal NNs: trains the neural networks with the optimal architecture and saves them in the directory *csv_gPlates/Neural Netorks*;
- section 5) - plot figures: plots hyperparameters tuning data;

#### main_modesAnalysis.m
This module analyzes the modal shapes of the dataset generated with Comsol Multiphysics livelink for Matlab in this repo.

- section 0) - init: sets up the Matlab search path, declares flags and variables, fetches dataset;
- section 1) - resample modeshapes: resamples modeshapes from Comsol irregular grid to a regular user defined rectangular grid;
- section 2) - compute reference set: computes the reference set of modeshapes with *Comsol livelink with Matlab*;
- section 3) - resample reference modeshapes: resamples the reference set of modeshapes from the irregular grid of Comsol to a user defined regular rectangular grid;
- section 4.1) - see reference modeshapes: plots the reference set of modeshapes;
- section 5) - modes identification}: performs modes identification by computing the normalised cross correlation (NCC) between the modal shapes of each dataset tuple and the reference set. Each mode is identified with the best scoring reference mode name;
- section 5) - modes identification, NCC computation: performs modes identification by computing the normalised cross correlation (NCC) between the modal shapes of each dataset tuple and the reference set;
- section 5.1) - modify reference set: allows to modify the reference set of modal shapes;
- section 6) - modes identification, postprocessing: analyzes NCC data, labels each Labels each mode with the reference mode scoring the highest NCC, discards tules with either repeated modes or at least one mode with NCC < 0.9
- section 6.1) - postprocessing: removes Poisson plates from the dataset;
- section 7) - plot modeshapes: plots the identified modes for each dataset tuple;
- section 8) - define modes order: analyzes modes identification data to find the most common succession of the modes the frequency increases. This succession will define the order in which modes are listed in the dataset ordered by modes;
- section 9) - generate and save ordered dataset: orders the dataset by modes and saves it. The dataset columns are ordered with the ordering defined in the previous section.

#### main_compute_exp_FRF.m
This module computes the H_1 estimator of the mobility (velocity / force) starting from force and acceleration measurements and performs peaks analysis on the estimated FRFs.

#### main_sensitivity_analysis.m
This module analyzes the input/output relationship of the dataset by computing the Pearson's correlation coefficient between each input and each output of the dataset. This allows us to understand how much sensible are the plate eigenfrequencies and the associate FRF amplitudes to the variation of each input parameter of the dataset.  

## Folders
### FRF2Params
<img align="center" src="/Figures/FRF2Params_dir_descr.png">

### Functions
<img align="center" src="/Figures/functions_dir_descr.png">

## Authors
(code and repo) David Giuseppe Badiane
(supervision)   Raffaele Malvermi
                Sebastian Gonzalez
