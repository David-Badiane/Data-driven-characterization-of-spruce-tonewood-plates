clear all;
close all;


%% 0) Initial setup

% Reference folders
baseFolder = pwd;
csvName = 'csv_wedge_FA';
convergencePath = [baseFolder, '\convergenceTest_wedge'];
csvPath = [baseFolder, '\', csvName];
idxs   = strfind(baseFolder,'\');
addpath(genpath([baseFolder(1:idxs(end)), 'functions']));
addpath(csvPath);

comsolModel = 'wedge_scan.mph';

%% 1)  Convergence test for choosing correct mesh
% =========================================================================

% setup
model = mphopen(comsolModel);
nModes = 15;
meshSizes = {'C+++', 'C++', 'C+', 'C', 'N', 'F', 'F+' , 'F++'};
convergenceNames = {'Eig Error [%]'  'FD Error [%]' 'Eig Time [s]' 'FD Time [s]'};
freqNames = {'f1' 'f2' 'f3' 'f4' 'f5' 'f6' 'f7' 'f8' 'f9' 'f10' 'f11' 'f12' 'f13' 'f14' 'f15'};

% Execute Comsol studies
[eigenFreqzMatrix, pointVelMatrix, eigRate, FDRate, timesEig, timesFD] = ConvergenceTestMesh(model,nModes, convergencePath);
% generate figures with data
convergenceFigures(eigenFreqzMatrix, pointVelMatrix, eigRate, FDRate, timesEig, timesFD );

% save results
cd(convergencePath)
convRates = round([eigRate;FDRate; timesEig; timesFD],2);
errRatesTable = array2table(convRates, 'variableNames',meshSizes, 'rowNames',convergenceNames );
frequencyTable = array2table(eigenFreqzMatrix, 'rowNames',meshSizes, 'variableNames',freqNames );
magnitudeTable = array2table(pointVelMatrix, 'rowNames',meshSizes, 'variableNames',freqNames(1:8) );
writetable(errRatesTable,'errRates.csv');
writetable(frequencyTable,'Eigenfrequencies.csv');
writetable(magnitudeTable,'Magnitude.csv');
cd(baseFolder);

% convergence test on the whole FRF 
fAxis = 200:1:2500;
[timesAmp, pointVelMatrix] = convergenceTestFRF(model, fAxis, convergencePath);
vels = []
for ii = 1:8
[vel] = readTuples(['vel', int2str(ii),'.txt'], 1, false);
vel = vel(4:end);
vels = [vels; vel];
end

figure; hold on;
nVels = length(vels(:,1));
c = [zeros(nVels, 1) linspace(0,1,nVels).' linspace(0,1,nVels).'];

for ii = 1:nVels
    pp = plot(fAxis, db(abs(vels(ii,:))), 'lineWidth', 1.15, 'Color', c(ii,:));
    meshSizes = {'C+++', 'C++', 'C+', 'C', 'N', 'F', 'F+' , 'F++'};
    legend(meshSizes(1:end))
end
Cor = zeros(length(vels(:,1)),1);
normMSE = zeros(length(vels(:,1)),1);
for m1 = 1:length(vels(:,1)) % Create correlations for each experimenter
  Cor(m1) = (corr(abs(vels(m1,:).'), abs(vels(end,:).')));
  normMSE(m1) = NMSE(abs(vels(m1,:)).', abs(vels(end,:)).');
end

figure()
subplot 211
plot(1:length(Cor), Cor);
subplot 212

%% 2) SET UP PARAMETERS FOR SIMULATIONS
% =========================================================================

% open Comsol model
if input('fetch Comsol model? (1/0): '), model = mphopen(comsolModel); end
% Parameters setup
params = mphgetexpressions(model.param);                  % get initial parameters                          
% get parameters names
varyingParamsNames = params(1:12,1);
steadyParamsNames = params(13:end,1);

i1 = strfind(comsolModel, '_'); i2 = strfind(comsolModel, '.');

referenceVals = readmatrix(['referenceVals_', comsolModel(i1+1:i2-1),'.csv']);

for jj = 1:length(varyingParamsNames)
   model.param.set(varyingParamsNames(jj), referenceVals(jj));
end 

%% 3) NEW DATASET GENERATION
% =========================================================================
% setup variables 
modeshapesFolder = [csvPath '\Modeshapes'];
mkdir(modeshapesFolder);
cd(baseFolder)
% set if you want to write new csv files
writeNow = true;
% number of simulations
nSim = 100;
% Comsol model
if input('fetch Comsol model? (1/0): '), model = mphopen(comsolModel); end
% Comsol number of eigenfrequencies computed
nModes = 15;

standardDev = [0.1 0.25*ones(1,9), 0.25*ones(1,2)];
velFilenames = {'H_12' 'H_13' 'H_15'};
meshSize = 4;
model.mesh('mesh1').feature('size').set('hauto', int2str(meshSize));
model.mesh('mesh1').run;

samplingMethod = 'gaussian';

[Dataset_FA] = comsolRoutineFA_wedge(model, nSim, nModes, referenceVals,...
                                   varyingParamsNames, velFilenames, standardDev,...
                                   modeshapesFolder, csvPath, writeNow, samplingMethod);
                               
%% 4) FETCH DATASET - (import csv )
% =========================================================================

[Dataset_FA, csvPath, datasetType] = fetchDataset_wedge(0, baseFolder, 1, csvName);

%% 8) Minimization of the error 
% =========================================================================
% preallocate and set variables
nRealizations = 5;
plotData = true;
NpeaksAxis = 1:8;
freqMatrix = zeros(nRealizations,length(NpeaksAxis));
ampMatrix = zeros(nRealizations,length(NpeaksAxis)); 
parsMatrix = zeros(nRealizations,12);
gauss = randn(nRealizations,1);

% minimiziation process
for ii = 1:nRealizations
    density = rho*(1+0.05*gauss(ii));
    density = rho; % TO DELETE
    [xpar, map, f_out, amp_out, fval] = ...
        minimization_FA(fNet, ampNet, Dataset_FA, f0, fAmps,density, NpeaksAxis, plotData);
    freqMatrix(ii,:) = f_out(map).'; 
    ampMatrix(ii,:) = amp_out(map).';
    parsMatrix(ii,:) = xpar.';
end
 

meanFreq = mean(freqMatrix);
varFreq = var(freqMatrix);

figure() 
plot(1:length(f0(NpeaksAxis)), f0(NpeaksAxis), '-o');
legend('fexp');
hold on 
errorbar(1:length(meanFreq),meanFreq,varFreq)
legend('fexp','fopt');
xlabel('mode number  N' );
ylabel('frequency    [Hz]');


meanMechParams = mean(parsMatrix);
matlabStd = std(parsMatrix)./meanMechParams;

stdNames = {'std rho [%]', 'std Ex [%]', 'std Ey [%]', 'std Ez [%]',...
                    'std Gxy [%]', 'std Gyz [%]', 'std Gxz [%]', 'std vxy [%]', 'std vyz [%]', 'std vxz [%]', 'std alpha[%]', 'std beta[%]'};
varyingParamsNames = {'rho', 'Ex', 'Ey', 'Ez', 'Gxy', 'Gyz', 'Gxz', 'vxy', 'vyz', 'vxz', 'alpha', 'beta'};   

matlabStdTable = array2table(matlabStd*100, 'VariableNames',stdNames);
finalOutTable = writeMat2File([meanMechParams; matlabStd*100],'Results.csv', varyingParamsNames, 10,true); 
    
%% 7.2) Check quality of the result
