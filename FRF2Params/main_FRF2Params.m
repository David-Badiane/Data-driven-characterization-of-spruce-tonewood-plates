
%% ____________________ MAIN FRF2PARAMS ___________________________________
% THIS PROGRAM APPLIES FRF2PARAMS TO ESTIMATE MATERIAL PROPERTIES OF PLATES.
% In this program we have:
% A) Dataset generation
% B) FRF2Params application
% C) Validation with Comsol
% -------------------------------------------------------------------------
% ~ Notice that the neural networks are trained and saved with main_hyperparams
% ~ The FRFs are aquired with                                  main_compute_exp_FRF
% ~ Datasets are ordered by mode numbers in                    main_modesAnalysis
% -------------------------------------------------------------------------
% summary:
% section 0) initial setup, reference folders
% section 1) retrieve measured FRFs + apply Caldersmith formulas
% section 2) dataset generation
% section 3) material properties estimation
% section 4) validation on Comsol - eigenfrequency study 
% section 5) validation on Comsol - frequency domain study
% section 5.1) see image of simulated (computed via section 5) and experimental FRFs

%% section 0) Initial setup, reference folder
% =========================================================================

% to remove all previous paths
remPath = false; % flag to remove paths, set to true if you want to remove all paths
if remPath
 cd(baseFolder)
 rmpath(genpath(baseFolder));
end

% directories names
baseFolder = pwd;                                          % base working directory is current directory 
datasetDir = 'csv_gPlates';                                % directory of the dataset
datasetPath = [baseFolder, '\', datasetDir];               % path of the dataset directory
idxs   = strfind(baseFolder,'\');                          % find the fxs path, it is in the directory containing the baseFolder
resultsPath = [datasetPath,'\Results'];                    % path of the results directory
mkdir(resultsPath);                                        % make directory
modeshapesFolder = [datasetPath,'\Modeshapes'];            % directory where modeshapes are saved
geom_mass_path= [baseFolder ,'\geom_mass_measurements'];   % directory of mass and geometry measurements
mkdir(modeshapesFolder);                                   % make directory
cd(baseFolder)                                             % return to base folder
NNPath = [datasetPath, '\Neural networks'];
mkdir(NNPath);

% add all paths 
addpath(genpath([baseFolder(1:idxs(end)), 'functions']));
addpath([baseFolder, '\FRF_data']);
addpath(datasetPath);
addpath(geom_mass_path);
addpath(NNPath);

%comsol Model name
comsolModel = 'gPlate.mph';
% parameters of the dataset
varyingParamsNames = {'rho', 'Ex', 'Ey', 'Ez', 'Gxy', 'Gyz', 'Gxz', 'vxy', 'vyz', 'vxz', 'alpha', 'beta'};   
geomNames = {'L' 'W' 'T'};
inputParamsNames = {varyingParamsNames{:} geomNames{:}};

% labels of the plates (book-matched: L = left, R = right) 
measuredSamples = {'1L' '1R' '2L' '2R' '3L' '3R' '4L' '4R' '5L' '5R'};

% if using Matlab livelink for Comsol, then open the comsol model in matlab
livelink = false;
if livelink, model = mphopen(comsolModel); end

% loading geometry, mass, volume and density measurements
measurement_geom_density = readmatrix('geomParams.csv');
rhos = round(measurement_geom_density(:,end));                  % densities
geomALL = 0.001*round(measurement_geom_density(:,1:3));         % geometry - 0.001* because from mm to meters
%% section 1) retrieve FRFs + apply Caldermith
% =========================================================================

load('measFRFs.mat'); % load measured mobilities --> struct named measFRFs with fields: 
                      % FRFs (mobilities), fAx (frequency axis), f0 (frequencies of the peaks)
                      % fAmps (amplitudes of the peaks), pts (measurement points)
                      % samples (labels of the plates names)

% nominal elastic constants of Picea Abies as reported by Hearmon in 1948
piceaParams = [390 10.9e9 0.64e9 0.42e9 0.58e9 0.026e9 0.59e9 0.39 0.64 0.49];

% idxs of the characteristic modes (f11, f02, f20) for each plate
charModes_Idxs = [1 2 4; 1 2 4; 1 2 4; 1 2 4; 1 2 4;...
                  1 2 3; 1 2 4; 1 2 3; 1 2 3; 1 2 4;];
firstGuess_parameters = zeros(length(measuredSamples), length(varyingParamsNames)+3); % to be filled subsequently

% use Caldersmith formulas for each plate to obtain first guesses
for ii = 1:size(charModes_Idxs,1) % for each plate
    geom = geomALL(ii,:);         % take geometry
    rho  = rhos(ii);              % take density
    f0   = measFRFs.f0s{ii};      % take peaks frequencies
    % apply caldersmith formulas
    [mechParams, normParams] = caldersmith_formulas(f0(charModes_Idxs(ii,:)),rho, geom);
    % store results
    firstguess_parameters(ii,:) = [rho mechParams(1:2) piceaParams(4) mechParams(3) piceaParams(6:end) 5 2e-6 geomALL(ii,:)];
end
% take mean and std of the estimations
firstGuess_mean = mean(firstguess_parameters);
firstGuess_std = std(firstguess_parameters)./firstGuess_mean*100;
%show them
disp([newline, 'obtained subsequent estimations with Caldersmith formulas'])
array2table([firstGuess_mean; firstGuess_std],...
             'VariableNames' , inputParamsNames, 'RowNames', {'mean' 'std [%]'})

% take rho and geometry from firstGuess and put them together to obtain center values of the dataset 
dataset_centerVals = [firstGuess_mean(1), piceaParams(2:end), firstGuess_mean(:,11:end)];
% show them
disp([newline, 'saving subsequent dataset center values'])
array2table(dataset_centerVals, 'VariableNames' , inputParamsNames)
% save them
writeMat2File([dataset_centerVals],'referenceVals.csv', inputParamsNames, length(inputParamsNames), true);

%% section 2) generate Dataset
% =========================================================================

if livelink
    % set to true if you want to write new csv files
    writeNow = 1 ;
    % number of simulations
    nSim = 500;
    % Comsol number of eigenfrequencies computed
    nModes = 20;
    % set mesh size --> 9 = extremely coarse, 1 = extremely fine
    meshSize = 6;
    model.mesh('mesh1').feature('size').set('hauto', int2str(meshSize)); % livelink commands
    model.mesh('mesh1').run;                                             % livelink commands
    % standard Deviation of the dataset input parameters
    standardDev = [0.1 0.25*ones(1,11) 0.005 0.02 0.075] ; 
    [Dataset_FA] = comsolRoutineFA_plate(model, nSim, nModes, dataset_centerVals,...
                                       inputParamsNames,  standardDev, modeshapesFolder,...
                                       datasetPath, writeNow, 'gaussian')      
end

%% section 3) FRF2Params - Material properties estimation
% =========================================================================
% ========================= PRESET VARIABLES ==============================
% ------------ a) set flags -----------------------------------------------
getNNs = true;              % fetch neural networks from files
getOrderedDataset = false;  % True to get NNs ordered by modes, false by peaks
% plotdata = show freq-amp space [end, during, before] minimization
plotData = [1, 1, 1]; 
see_loss_fx_evolution = 0; % true if you want to see the evolution of the loss fx
saveResults = 1;
use_refvals_as_first_guess = true;
% ------------ b) set constants and variables of FRF2PARAMS -----------------------------
nRealizations = 10;                                                    % n° realizations considered to average the result
plateNumbers =  10;                                                  % plate numbers in a array
nPeaks = 12;                                                           % n° peaks considered during minimization
considered_peaks_axis = 1:nPeaks;                                      % axis with considered FRF peaks
alphas =  [5, 5, 5, 5, 4, ...
           2.5, 2.5, 5, 0, 0]*1e-1;       % starting values for alpha

betas   = [8, 3, 2, 3,   5, ...
           5, 5, 8, 4, 6]*1e-7; % starting values for beta

sampleSize = 1000;% first guess are the center values of the dataset

% ------------ c) set parameters to exit the minimization algorithm ------- 
% N.B. both tolFun and tolX must be satisfied to satisfy convergence criteria
tolFun      = 1e-6;     % minimum loss function variation per step
tolX        = 1e6;      % minimum variation of any input per step
maxFunEvals = 1e3;      % max n° evaluations of the loss function
maxIter     = 1.5e3;    % max n° iterations
if see_loss_fx_evolution 
 options = optimset(optimset('fminsearch'), 'TolFun', tolFun,'TolX',tolX,...
                    'MaxFunEvals', maxFunEvals,'MaxIter', maxIter, 'PlotFcns',@optimplotfval); 
else
 options = optimset(optimset('fminsearch'), 'TolFun', tolFun,'TolX',tolX,...
                    'MaxFunEvals', maxFunEvals,'MaxIter', maxIter);
end

%  ------------ d) fetch dataset and NNs ----------------------------------
if getNNs
    modesGet = 15;
    [Dataset_FA, datasetPath, HP] = fetchDataset(baseFolder, modesGet, getOrderedDataset, datasetDir);
    cd(NNPath);
    if getOrderedDataset, order= 'Modes'; else, order = 'Peaks'; end
    load(['optNN_' int2str(modesGet) '_' order '.mat']); 
    cd(baseFolder);
end

%  ------------ e) set which parameters are not updated during minimization 
% name    rho, E_L, E_R, E_T, G_LR, G_RT, G_LT, v_LR, v_RT, v_LT, alpha, beta, L,  W,   T
% index    1    2    3    4    5     6     7     8     9     10    11     12   13  14   15
fixParamsIdxs = [1,11,12,13,14,15]; % density, damping, geometry

% ============================= FRF2Params ================================
for plateN = plateNumbers % for each plate
    disp([newline, 'SAMPLE ' measuredSamples{plateN}, newline])
    
    % A) ---------------------- SET VARIABLES -----------------------------
    resultsFilename = ['Results_', measuredSamples{plateN}, ...
                       '_nR', int2str(nRealizations),'_',int2str(nPeaks)];
    
    % preallocate for the mode matching btw FRF peaks - NNs eigenfrequencies
    % ex. - map = [1 3 5 4...] means (1st peak - 1st eig) - (2nd peak - 3rd eig) - and so on
    mode_matching_maps = zeros(nRealizations,length(considered_peaks_axis)); 
    
    % preallocate for 
    freqMatrix = zeros(nRealizations,length(considered_peaks_axis));         % NNs eigenfrequencies after each minimization
    ampMatrix = zeros(nRealizations, length(considered_peaks_axis));         % NNs amplitudes after each minimization
    parsMatrix = zeros(nRealizations,length(input_parameters_start));        % material parameters after each minimization
    gaussDensity = randn(nRealizations,1); % gaussian distribution for density 
    % N.B. (each minimization starts from a different density value, not updated during minimization)
    densityStd = 0.025;
    
    % take plate FRF data
    FRF  = measFRFs.FRFs(plateN,:);
    f0   = measFRFs.f0s{plateN};
    fAmps = measFRFs.fAmps{plateN};
    fAx  = measFRFs.fAx;
    
    % set first guess for density, geometry, damping
    density = firstguess_parameters(plateN,1);           % density
    geometry = firstguess_parameters(plateN,13:end); % geometry
    damping = [alphas(plateN), betas(plateN)];       % damping
    
    % set first guess for mech params
    if use_refvals_as_first_guess
        input_parameters_start = [density firstguess_parameters(plateN,2:10) damping geometry];
    else
        input_parameters_start = [density dataset_centerVals(2:10) damping geometry];
    end  
    
    % values of constant params
    constant_params_values = input_parameters_start(fixParamsIdxs);
    % show values of constant params
    disp(['constant params names : ',newline, string(inputParamsNames(fixParamsIdxs))])
    disp(['constant params values : ' num2str(input_parameters_start(fixParamsIdxs),2)]);
    rho = input_parameters_start(1);
    
    % B) ------------------------- MINIMIZATIONS --------------------------
    for ii = 1:nRealizations
        tStart = tic;
        constant_params_values(1) = round(rho*(1+densityStd*gaussDensity(ii)));             % override density
        disp(['SAMPLE' num2str(plateN) ' minimization n° ', num2str(ii), ' rho = ', num2str(round(constant_params_values(1)))]);
        
        [estimated_params, mode_matching_map, f_out, amp_out] = ...
          FRF2Params( options, fNet, aNet, {f0}, {fAmps},...
          considered_peaks_axis, plotData, constant_params_values, fixParamsIdxs,input_parameters_start);
      
        % save minimization results
        mode_matching_maps(ii,:) = mode_matching_map;
        freqMatrix(ii,:) = f_out(mode_matching_map).'; 
        ampMatrix(ii,:) = amp_out(mode_matching_map).';
        parsMatrix(ii,:) = estimated_params.';
        array2table(estimated_params(:).', 'variableNames', inputParamsNames)
        disp(['it took ' num2str(toc(tStart)) ' seconds', newline]);
    end
    
    % ======================= SEE AND STORE RESULTS =======================
    % take mean and std of eigenfrequencies from NNs to generate a errorbar plot
    meanFreq = mean(freqMatrix);
    stdFreq = std(freqMatrix);
    figure() 
    plot(1:length(f0), f0, '-o');
    legend('fexp');
    hold on 
    errorbar(1:length(meanFreq),meanFreq,stdFreq)
    legend('fexp','fopt');
    xlabel('mode number  N' );
    ylabel('frequency    [Hz]');

    % take mean and std from the material parameters estimations
    mean_estimated_params = mean(parsMatrix,1);
    std_estimated_params = std(parsMatrix,1)./mean_estimated_params;
    results = [mean_estimated_params; std_estimated_params*100];
    % set names for table
    stdNames = {};
    for ii = 1:length(inputParamsNames)
        stdNames{ii} = ['std', inputParamsNames{ii},'[%]'];
    end
    
    % show results
    array2table(results, 'variableNames', inputParamsNames, 'RowNames', {'params' 'stds'})
    % save results
    cd(resultsPath);
    if saveResults
    finalOutTable = writeMat2File(results,[resultsFilename,'.csv'],inputParamsNames, length(inputParamsNames),true);
    end
end

%% section 4) Comsol eigenfrequency study with estimated parameters to validate the results
% =========================================================================
plateN = 1:10;
nRealizations = 10;
nPeaks = 12;
resultsFilename = ['Results_' measuredSamples{plateN} '_nR', int2str(nRealizations),'_',int2str(nPeaks(plateN)) ];
mechParams = table2array(readtable([resultsFilename,'.csv']));
mechParams = mechParams(1,:);

% get comsol parameters and set them with the estimations
comsolParams_general = mphgetexpressions(model.param);
comsolParams_names = comsolParams_general(1:15,1);
setParams(model, comsolParams_names, mechParams.')

% run eigenfrequency study
model.component('comp1').mesh('mesh1').feature('size').set('hauto', 6); % --> mesh
model.physics('solid').feature('lemm1').feature('dmp1').active(false);  % --> disable damping
model.mesh('mesh1').run;                                                % --> run mesh
model.study('std1').run();                                              % --> run study
model.physics('solid').feature('lemm1').feature('dmp1').active(true);   % --> enable damping

% evaluate Comsol eigenfrequencies and compare with NNs ones
evalFreqz = mpheval(model,'solid.freq','Dataset','dset1','edim',0,'selection',1);
eigenFreqz = real(evalFreqz.d1');
NNs_freqs = fNet(mechParams).';
relErr = (sort(NNs_freqs) - eigenFreqz(1:length(NNs_freqs)))./ eigenFreqz(1:length(NNs_freqs))*100;
disp(['relative error frequencies: ', newline, num2str(relErr)]);

%% section 5) simulate point FRF  to validate results (frequency domain with estimated parameters)
% =========================================================================
% flags 
fetchModel = 1; % --> to fetch the comsol model right here

% variables - some are repeated from (section 0) to opportunistically validate results
datasetDir = 'csv_gPlates_'; 
resultsPath = [baseFolder,'\', datasetDir,'\Results'];
meshSize = 6; 
fLow =30; fHigh = 400;
fBounds = [fLow, fHigh];

fAx  = measFRFs.fAx;
nPointsAxis = 600; 
plateNumbers = 1:10;
nRealizations = 10;
nPeaks = [12 12 12 12 12 12 12 12 12 12];

% to set alpha and beta (damping) in the simulation
alphas_simulation = [8.0 8.0 6.0 6.0 6.0 6.0 6.0 8.0 6.0 6.0];
betas_simulation  = [2.0 2.0 2.0 2.0 2.0 2.3 2.0 2.0 2.0 2.0]*1e-6;
alphaBetaIdxs     = [11,12];

% Comsol routine
if fetchModel, cd(baseFolder); model = mphopen(comsolModel); end

for plateN = plateNumbers
    resultsFilename = ['Results_' measuredSamples{plateN} '_nR', int2str(nRealizations),'_',int2str(nPeaks(plateN)) ];
    dampingParams   = [alphas_simulation(plateN), betas_simulation(plateN)];
    % compute FRF
    [vel, fAxisComsol, resultsFilename] = comsol_point_FRF(model, resultsPath,...
    resultsFilename, meshSize, fAx, fBounds, nPointsAxis, dampingParams, alphaBetaIdxs, {'cpt1'}, 1:15,{'solid.u_tZ'});  
end

%% section 6) See Img of simulated and experimental FRFs
% =========================================================================
% set up img data entries
% flags
chooseFilename = 0;   % --> flag to set wheter to write the filename of the FRF
fetchFromFile = 1;    % --> flag to fetch FRF from file or use previously computed one
highlightPeaks = 0;   % --> flag to highlight peaks with stems in the figure
printAnnotations = 0; % --> flag to print FRAC in the figure itself

% variables
nPeaks = 12;
temp = 1;
nRealizations = 10;
res = [];
resNames = {};
alphas = 5*ones(10,1);
betas = 2e-6*ones(10,1);
plateNumbers = 1:10;

% image setting variables
xLengthImg = 700; 
yLengthImg = round(0.72*xLengthImg);
xLabel = 'Frequency [Hz]'; 
areaColor = [0.1 0.5 0.5];
annotFontSize = 5; 
axFontSize = 16;
areaAlpha = .13; 
paramsTextPos = [.75 .125 .4 .3];
legenda = {'real FRF', 'simulated FRF', 'area difference'};
lineWidth = 1.15;    
cutHigh = 1200; 
cutLow = 30;
fracs = zeros(1,length(plateNumbers));
nmses = zeros(1,length(plateNumbers));
% load FRF data
cd(baseFolder)
load('measFRFs.mat');
cd(resultsPath);

% generate figures
for plateN = plateNumbers
    resultsFilename = ['Results_' measuredSamples{plateN} '_nR', int2str(nRealizations),'_',int2str(nPeaks)]; 
    imgN = 2000+temp;
    figure(imgN); 
    errTextPos = [.75 .425 .4 .3];
    yLabel = ['$|H_{', num2str(plateN), '}(f)|$'];
    subplotN = [1,1,1];
    
    if fetchFromFile
        if chooseFilename  
            completeName = input('insert simulation filename without tag (string like .csv not to include): '); 
        else
            completeName = ['FRF_',resultsFilename,'_beta_' num2str(betas(plateN))]; 
        end 
            simFRFs = readmatrix([completeName, '.csv']);
            fAxisComsol = real(simFRFs(:,1));
            vel =simFRFs(:,2:end);
    end

    [imgData, FRFData] = defImg_comparison_FRFs( xLengthImg, yLengthImg, imgN,...
                                xLabel, yLabel, areaColor, axFontSize,...
                                areaAlpha, legenda, lineWidth, ...
                                cutHigh, cutLow, measFRFs.FRFs(plateN,:), measFRFs.fAx, fAxisComsol, vel,...
                                alphas(plateN), betas(plateN), [], {'log', 'linear'});

    [img, frac,nmse] = export_comparison_FRFs(FRFData, imgData, 'db',2, highlightPeaks,...
                                   0, 50, subplotN);
    disp(['plate n° ', num2str(plateN) ': frac = ' num2str(frac,2) '   nmse = ' num2str(nmse,2)]);
    fracs(plateN)= frac;
    nmses(plateN)= nmse;
    temp = temp+1;
    pause(1e-4);
    close all
end