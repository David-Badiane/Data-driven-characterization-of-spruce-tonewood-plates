clear all;
close all;


%% 0) Initial setup
% rmpath(genpath(baseFolder));
% Reference folders
baseFolder = pwd;
csvName = 'plate_testcsv\gaussianTestSets';
convergencePath = [baseFolder, '\convergenceTest_plate'];
testPath = [baseFolder, '\testSets'];
csvPath = [baseFolder, '\', csvName];
addpath (genpath([baseFolder, '\functions']));
addpath ([baseFolder, '\data']);
addpath(csvPath);
addpath(testPath);

comsolModel = 'woodenPlate';

infosTable = readtable("sampleMeasurements.xlsx");
infosMatrix = table2array(infosTable(:,3:end));
infosMatrix(:,1:2) = infosMatrix(:,1:2)*0.01;
infosMatrix(:,3:7) = infosMatrix(:,3:7)*0.001;

outputFilenames = {'input' 'output'};
outputFiletype = '.csv';

sampleNames = table2array(infosTable(:,1));
Ls      =   infosMatrix(:,1);
Ws      =   infosMatrix(:,2);
ThUp    =   infosMatrix(:,3);
ThDown  =   infosMatrix(:,4);
ThRight =   infosMatrix(:,5);
ThLeft  =   infosMatrix(:,6);
ThAvg   =   infosMatrix(:,7);
rhos    =   infosMatrix(:,9);
comsolParams =infosMatrix(5,1:6);

geom = infosMatrix(5,1:3);
rho = infosMatrix(5,end);

idx = 1;
standardDev = 0.1;

%% 1) OBTAIN EXPERIMENTAL RESULTS

% setp
jj = 1;
sampleNames = {'7b'};
fHigh = 1200;
M = 10;
thresholdPerc = 30;
idx = 1;
load('07_01_21_frf_anima_7b_fronte_G_post');

% 1) SVD
fAxis = f(f <= fHigh);
[HvSVD,threshold,singularVals] = SVD(H1_v, fAxis, M, thresholdPerc, sampleNames, jj);
    
% 2) MODAL ANALYSIS
[Hv,f0, fLocs, csis, Q] = EMASimple(HvSVD, fAxis,1e-4, 3,false);
f0 = f0(2:end);
fLocs = fLocs(2:end);
csis = csis(2:end);

% 3) Compute first guess of Ex

[mechParams, normParams] = computeParams([f0(1), f0(2), f0(3)],rho, geom);

fAmps = abs(Hv(fLocs));

%% 2) SET UP PARAMETERS FOR SIMULATIONS

% open Comsol model
model = mphopen(comsolModel);
% Parameters setup
params = mphgetexpressions(model.param);                  % get initial parameters                          
% get parameters names
varyingParamsNames = params(7:end,1);
steadyParamsNames = params(1:6,1);

% set geometry parameters
geomSet = [infosMatrix(5,1:6), infosMatrix(5,end)];
setParams = cell(length(steadyParamsNames)+1,1);
setParams(1:end-1) = steadyParamsNames;
setParams{end} = 'rho';

for jj = 1:length(setParams)
   model.param.set(setParams(jj), geomSet(jj));
end

% prepare reference values
Ex = mechParams(1);
referenceVals = [rho, Ex, Ex*0.078, Ex*0.043,...
                 Ex*0.061, Ex*0.064, Ex*0.003,...
                 0.467, 0.372, 0.435, 19, 7e-6];
             
%% 3) NEW test set GENERATION

stdVal = [.15 .25 .35 ];
for kk = 1:length(stdVal)
    csvName = ['plate_testcsv\gaussianTestSets\gaussian_', num2str(round(stdVal(kk),2)) ,'std'];
    csvPath = [baseFolder, '\', csvName];
    addpath(csvPath);

    % setup variables 
    simFolder = [csvPath,'\Modeshapes'];
    mkdir(simFolder)
    cd(baseFolder)
    % set if you want to write new csv files
    writeNow = true;
    % number of simulations
    nSim = 100;
    % Comsol model
    model = mphopen(comsolModel);
    % Comsol number of eigenfrequencies computed
    nModes = 20;  

    meshSize = 6;
    model.mesh('mesh1').feature('size').set('hauto', int2str(meshSize));
    model.mesh('mesh1').run;

    % if 'uniform' - density is still gaussian - alpha
    % and beta
    % 'rho' 'Ex' 'Ey' 'Ez' 'Gxy' 'Gyz' 'Gxz' 'vxy' 'vyz' 'vxz' 'alpha' 'beta'
    standardDev = [2 2*ones(size(referenceVals(2:7))) 2*ones(size(referenceVals(8:10))) 2*ones(size(referenceVals(11:12))) ];  
    standardDev = stdVal(kk)*ones(size(referenceVals)) ;                                
%     [testSet] = comsolRoutineFreqAmp(model, nSim, nModes, referenceVals,...
%                                        varyingParamsNames,  standardDev,  simFolder, csvPath, writeNow, 'gaussian');
%     
    [testSet] = comsolRoutineFA_Inputs(model, nSim, nModes,...
                                   varyingParamsNames,  1,  simFolder, csvPath)
                                             
                                   
    rmpath(csvPath);
end
                               
%% FETCH And save TEST SET 
stdSave = num2str(0.15);
testPath = [baseFolder, '\testSets\dset_uu'];
csvName = 'plate_testcsv\uniformTestSets';
csvPath = [baseFolder, '\', csvName];

cd([csvPath, '\gaussian_', stdSave, 'std']);

datasetType = 'ordered'; % choose btw ordered and raw

saveData = true;
saveFilename = ['test_G_',stdSave, 'std_', datasetType,'.mat'];

testSet = struct('inputs',[] ,'outputsEig',[] ,'outputsAmp',[] );
    
if strcmp(datasetType, 'ordered')
    testSet = struct('inputs',[] , 'dataOrder', [], 'modesOrder', [], 'outputsEig',[] ,'outputsAmp',[] );
    testSet.inputs = table2array(readtable('datasetOrdered_Inputs.csv'));
    testSet.outputsEig = table2array(readtable('datasetOrdered_Eig.csv'));
    testSet.outputsAmp = table2array(readtable('datasetOrdered_Amp.csv'));
    testSet.dataOrder = readtable('datasetOrdered_Eig.csv').Properties.VariableNames;
    
    nColsDataset = length(testSet.dataOrder);
    test_modesIdxs = 1:2:nColsDataset;
    test_peaksIdxs = 2:2:nColsDataset;
    modesOrderDataset = testSet.dataOrder(test_modesIdxs);
    testSet.modesOrder = testSet.dataOrder(test_modesIdxs);

end

if strcmp(datasetType, 'raw')
    testSet.inputs = table2array(readtable('inputs.csv'));
    testSet.outputsEig = table2array(readtable('outputsEig.csv'));
    testSet.outputsAmp = table2array(readtable('outputsAmp.csv'));
    test_modesIdxs = 1:length(testSet.outputsEig(1,:));
    test_peakIdxs = [];
end

cd(testPath);
if saveData
    disp(['saving ', saveFilename])
    save(saveFilename, 'testSet', 'test_modesIdxs', 'test_peaksIdxs');
end