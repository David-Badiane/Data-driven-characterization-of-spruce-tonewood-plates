clear all;
close all;


%% 0) Initial setup

% Reference folders
% cd(baseFolder)
% rmpath(genpath(baseFolder));
baseFolder = pwd;
csvName = 'csv_plate_FA_uniform';
convergencePath = [baseFolder, '\convergenceTest_plate'];
testPath = [baseFolder, '\testSets'];
trainPath = [baseFolder, '\trainSets'];
NNPath = [baseFolder,'\NNs'];
csvPath = [baseFolder, '\', csvName];
addpath (genpath([baseFolder, '\functions']));
addpath ([baseFolder, '\data']);
addpath(csvPath);
addpath(testPath);
addpath(trainPath);
addpath(NNPath)
comsolModel = 'woodenPlate';

nnNames = cellstr(ls(NNPath)); nnNames = nnNames(3:end);
for ii = 1:length(nnNames) , nnNames{ii, 2} = ii; end
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

%% 1.1)  Convergence test for choosing correct mesh

% setup
model = mphopen(comsolModel);
nModes = 15;
meshSizes = {'C+++', 'C++', 'C+', 'C', 'N', 'F', 'F+' , 'F++', 'F+++'};
convergenceNames = {'Eig Error [%]' 'Eig Time [s]' };
freqNames = {'f1' 'f2' 'f3' 'f4' 'f5' 'f6' 'f7' 'f8' 'f9' 'f10' 'f11' 'f12' 'f13' 'f14' 'f15' 'f16' 'f17' 'f18' 'f19' 'f20'};

% Execute Comsol studies
[eigenFreqzMatrix, pointVelMatrix, eigRate, timesEig] = ConvergenceTestMesh(model,nModes, convergencePath);
% generate figures with data
convergenceFigures(eigenFreqzMatrix, pointVelMatrix, eigRate, FDRate, timesEig, timesFD, [4,5] );

% save results
cd(convergencePath);
convRates = round([eigRate;FDRate; timesEig; timesFD],2);
errRatesTable = array2table([eigRate; timesEig], 'variableNames',meshSizes, 'rowNames',convergenceNames );
frequencyTable = array2table(eigenFreqzMatrix, 'rowNames',meshSizes, 'variableNames',freqNames );
magnitudeTable = array2table(pointVelMatrix, 'rowNames',meshSizes, 'variableNames',freqNames(1:8) );
writetable(errRatesTable,'errRatesFreq.csv');
writetable(frequencyTable,'Eigenfrequencies.csv');
writetable(magnitudeTable,'Magnitude.csv');
cd(baseFolder);

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
                 0.467, 0.372, 0.435, 19.0001, 7e-6];
             
%% 3) NEW DATASET GENERATION
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
standardDev = 0.60*ones(size(referenceVals)) ;                                
[Dataset_FA] = comsolRoutineFreqAmp(model, nSim, nModes, referenceVals,...
                                   varyingParamsNames,  standardDev,  simFolder, csvPath, writeNow, 'gaussian');
                       
%% 4) Fetch DATASET - (import csv )
datasetType = 'raw'; % choose btw ordered and raw
datasetDistr = 'uniform';
saveData = false;
saveFilename = ['D_',datasetDistr,'_', datasetType, '.mat'];

mechParamsNames = {'\rho' 'E_x' 'E_y' 'E_z' 'G_{xy}' 'G_{yz}' 'G_{xz}' '\nu_{xy}' '\nu_{yz}' '\nu_{xz}' '\alpha' '\beta'};
Dataset_FA = struct('inputs',[] ,'outputsEig',[] ,'outputsAmp',[] );

if strcmp(datasetType, 'ordered')
    Dataset_FA = struct('inputs',[] , 'dataOrder',[], 'modesOrder', [],'outputsEig',[] ,'outputsAmp',[] );
    Dataset_FA.inputs = table2array(readtable('datasetOrdered_Inputs.csv'));
    Dataset_FA.outputsEig = table2array(readtable('datasetOrdered_Eig.csv'));
    Dataset_FA.outputsAmp = table2array(readtable('datasetOrdered_Amp.csv'));
    Dataset_FA.dataOrder = readtable('datasetOrdered_Eig.csv').Properties.VariableNames;
    
    nColsDataset = length(Dataset_FA.dataOrder);
    modesIdxs = 1:2:nColsDataset;
    peaksIdxs = 2:2:nColsDataset;
    Dataset_FA.modesOrder = Dataset_FA.dataOrder(modesIdxs);
    
    dSet = struct('inputs', [], 'modesOrder', [],...
                   'outputsEig', [], 'outputsAmp', []);
    dSet.inputs = Dataset_FA.inputs;
    dSet.modesOrder = Dataset_FA.dataOrder(modesIdxs);
    dSet.outputsEig = Dataset_FA.outputsEig(:,modesIdxs);
    dSet.outputsAmp = Dataset_FA.outputsAmp(:,modesIdxs);
end

if strcmp(datasetType, 'raw')
    Dataset_FA.inputs = table2array(readtable('inputs.csv'));
    Dataset_FA.outputsEig = table2array(readtable('outputsEig.csv'));
    Dataset_FA.outputsAmp = table2array(readtable('outputsAmp.csv'));
    modesIdxs = 1:length(Dataset_FA.outputsEig(1,:));
    peaksIdxs = [];
end

nTuples = length(Dataset_FA.inputs(:,1));
varyingParamsNames = {'rho', 'Ex', 'Ey', 'Ez', 'Gxy', 'Gyz', 'Gxz', 'vxy', 'vyz', 'vxz', 'alpha', 'beta'};

if saveData
    cd(trainPath)
disp(['saving ', saveFilename])
save(saveFilename, 'Dataset_FA', 'modesIdxs', 'peaksIdxs');
cd(baseFolder)
end

%% 5) Check Caldersmith formulas performance
% f11 <-> col 1 || f02 <-> col 3  || f20 <-> col 7
nTuples = length(Dataset_FA.inputs(:,1));
caldersmithIdxs = modesIdxs([1,2,4]);
caldersmithFreqs = Dataset_FA.outputsEig(:, caldersmithIdxs);
caldersmithParams = zeros(nTuples,3);
datasetParamsIdxs = [2,3,5];
datasetParams = Dataset_FA.inputs(:,datasetParamsIdxs);
normDataset = datasetParams./datasetParams(:,1);
densities = Dataset_FA.inputs(:,1);

for ii = 1:nTuples
[mechParams, normParams] = computeParams(caldersmithFreqs(ii,:), densities(ii), geom);
caldersmithParams(ii,:) = mechParams; 
end
normCaldersmith = caldersmithParams./caldersmithParams(:,1);
nPlot =1:2500;
xlabels = {'E_L dataset' 'E_R dataset' 'G_{LR} dataset' };
ylabels = {'E_L caldersmith' 'E_R caldersmith' 'G_{LR} caldersmith' };
titles = {'E_L scatter' 'E_R scatter' 'G_{LR} scatter'};

xlabelsRatio = { 'E_R/E_L dataset' 'G_{LR}/E_L dataset' };
ylabelsRatio = { 'E_R/E_L caldersmith' 'G_{LR}/G_L caldersmith' };
titlesRatio = { 'E_R/E_L scatter' 'G_{LR}/E_L scatter'};

for ii = 1:3   
    tempCaldersmith = caldersmithParams(nPlot,ii);
    tempParams = datasetParams(nPlot,ii);
    
    figure(ii*10)
    scatter( tempParams, tempCaldersmith,9, 'filled');
    hold on;
    plot([0,1e20],[0,1e20]);
    xlim([min(tempParams), max(tempParams)]);
    ylim([min(tempCaldersmith)*0.9, max(tempParams)*1.02]);
    hold off;
    xlabel(xlabels{ii}); ylabel(ylabels{ii}); title(titles{ii}); ax = gca; 
    ax.FontSize = 14;
    
    if ii >1
    figure(11)
    subplot(1,2,ii-1)
    scatter( normDataset(nPlot,ii), normCaldersmith(nPlot,ii), 9, 'filled');
    hold on;
    plot([0,1],[0,1]);
    xlim([min(normDataset(nPlot,ii))*0.9, max(normDataset(nPlot,ii))*1.1])
    hold off;
    xlabel(xlabelsRatio{ii-1}); ylabel(ylabelsRatio{ii-1}); title(titlesRatio{ii-1}); ax = gca; 
    ax.FontSize = 14;
    end
end



%% To analyse how much are present modes for the dataset 
modesPresence = zeros(1,20);
for ii = 1:length(Dataset_FA.dataOrder)/2
    modesPresence(ii) = length(find(~isnan(Dataset_FA.outputsEig(:,modesIdxs(ii)))));
end
writeMat2File(modesPresence,'modesPresence.csv', Dataset_FA.dataOrder(modesIdxs), length(Dataset_FA.dataOrder(modesIdxs)), true)

%% Save NNS
cd(NNPath)
save('NN_u_raw_arch2.mat', 'fNet', 'ampNet')
cd(baseFolder)
%% 7.2) Neural networks
freqNames = {};
count = 1;
for yy = 1:2
figN_f = input('figure number freq: '); figN_a = input('figure number amp: ');
rmpath(testPath);
chooseD = input('choose dataset test dir - (1 gg -- 0 uu): ');
if chooseD == 1 
testPath = [baseFolder, '\testSets\dset_gg']
else
testPath = [baseFolder, '\testSets\dset_uu']
end
addpath(testPath);
% freqNames = {};

for nTest = 1
    splitDataset = false;
    plotData = true;
    loadNNs = 1;

    nModes = 20;
    nNeurons_frq = 15;
    nLayers_frq = 3;
    nNeurons_amp = 15;
    nLayers_amp = 5;
    if ~loadNNs || splitDataset
    disp('choose dataset');
    trainFiles = cellstr(ls(trainPath));
    trainFiles = trainFiles(3:end)
    trainFileIdx = input('choose index of the .mat file: ');
    load(trainFiles{trainFileIdx})
    end

    if splitDataset
        nTrain = round(0.9*length(Dataset_FA.inputs(:,1)));
        nTest = round(0.1*length(Dataset_FA.inputs(:,1)));
        [freq_R2_, freq_R2, fNetVector, fNet, fTestIdxs] = ...
             NN_trainTest_intTest(Dataset_FA.inputs, Dataset_FA.outputsEig(:,modesIdxs), nNeurons_frq, nLayers_frq, nTrain, nTest,'freq', 25);
        [amp_R2_, amp_R2, aNetVector, ampNet, ampTestIdxs] = ...
            NN_trainTest_intTest(Dataset_FA.inputs, db(abs(Dataset_FA.outputsAmp(:,modesIdxs))), nNeurons_amp, nLayers_amp,nTrain, nTest, 'amp',35);
    else

        testFiles = cellstr(ls(testPath));
        testFiles = testFiles(3:end);
        for ii = 1:length(testFiles) , testFiles{ii, 2} = ii; end
        disp(testFiles)
        testFileIdx = input('choose the index of the test .mat file you want to use: ');
        testFilename = testFiles{testFileIdx};
        load(testFilename);

        testIn = testSet.inputs;
        testOut_f = testSet.outputsEig(:, test_modesIdxs);
        testOut_a = db(abs(testSet.outputsAmp(:, test_modesIdxs)));

        if loadNNs
            disp(nnNames)
            NNidx = input('index of the file containing the NNS 2 load: ');
            NNfilename = nnNames{NNidx};
            load(NNfilename);
        else

        [fNet, f_netVector] = NN_train(Dataset_FA.inputs, Dataset_FA.outputsEig(:,modesIdxs), nNeurons_frq, nLayers_frq);
        [ampNet, a_netVector] = NN_train(Dataset_FA.inputs, db(abs(Dataset_FA.outputsAmp(:,modesIdxs))), nNeurons_amp, nLayers_amp);
        end

         [freq_R2] = NN_test(fNet, testIn, testOut_f, 'freq', plotData, figN_f);
         [amp_R2] = NN_test(ampNet, testIn, testOut_a, 'amp' , plotData,  figN_a);
         freqNames{count} = replace([NNfilename(1:end-4), ' - ' , testFilename(1:end-4) ], '_', ' ');
         count = count + 1;
    end

    figure(figN_f)
    legend(freqNames{:})
    figure(figN_f +1)
    legend(freqNames{:})
    figure(figN_a)
    legend(freqNames{:})
    figure(figN_a +1)
    legend(freqNames{:})   
end
end
%% 7.1) Hyperparameters
writeNewFiles = false;

nNeurons_max = 100;
nLayers_max = 5;

fileFreq = 'HPfreq_un_raw.csv';
fileAmp  = 'HPamp_un_raw.csv';


if writeNewFiles 
    HPmat_freq = zeros(nNeurons_max, nLayers_max);
    HPmat_amp = zeros(nNeurons_max, nLayers_max);
else
    HPfreq = readmatrix(fileFreq);
    HPamp = readmatrix(fileAmp);
    [nN,nL] = size(HPfreq);
    nNeurons_max = max([nN, nNeurons_max]);
    nLmax = max([nL, nLayers_max]);
    HPmat_freq = zeros(nNeurons_max , nLmax); 
    HPmat_amp = zeros(max([nN, nNeurons_max]), max([nL, nLayers_max]));
    
    HPmat_freq(1:nN, 1:nL) = HPfreq;
    HPmat_amp(1:nN, 1:nL) = HPamp;
end
% Dataset_FA.outputsEig(isnan(Dataset_FA.outputsEig)) = 10;
% Dataset_FA.outputsAmp(isnan(Dataset_FA.outputsAmp)) = 1e-8;
R2f = {};
R2a = {};

for ii = 31:nNeurons_max
    for jj = 2:nLayers_max
        nTrain = floor(0.9*length(Dataset_FA.inputs(:,1)));
        nTest = floor(0.1*length(Dataset_FA.inputs(:,1)));
        
        [f_R2, fNet, f_outTest, f_predicted] = NN_trainTest_intTest...
            (Dataset_FA.inputs, Dataset_FA.outputsEig(:,modesIdxs),...
            ii, jj, nTrain, nTest);
        
        [a_R2, ampNet, a_outTest, a_predicted] = NN_trainTest_intTest...
            (Dataset_FA.inputs, db(abs(Dataset_FA.outputsAmp(:,modesIdxs))),...
             ii, jj,nTrain, nTest);
         
         f_R2 = table2array(f_R2); 
         a_R2 = table2array(a_R2);
         
         R2f{ii}{jj} = f_R2;
         R2a{ii}{jj} = a_R2;
         
         HPmat_freq(ii,jj) = mean(f_R2);
         HPmat_amp(ii,jj) = mean(a_R2);
    end
end

figure(5)
im = imagesc(HPmat_amp);
xlabel('n layers');
ylabel('nNeurons');
cmap = winter;
colormap(cmap);
colorbar;
set(gca,'XTick',[1:nLayers_max],'YTick',[1:nNeurons_max])
[x,y] = meshgrid(1:nLmax,1:nNeurons_max);
text(x(:),y(:),num2str(round(HPmat_amp(:),2)),'HorizontalAlignment','center')

figure(6)
im=imagesc(HPmat_freq);
xlabel('n layers');
ylabel('nNeurons');
cmap = autumn;
colormap(cmap);
colorbar;
set(gca,'XTick',[1:nLmax],'YTick',[1:nNeurons_max])
text(x(:),y(:),num2str(round(HPmat_freq(:),2)),'HorizontalAlignment','center')

figure(8)
plot(1:40, HPmat_freq(1:40, 1), '-O', 'lineWidth', 1.3)
xlabel('N Neurons')
xx = xlabel('N Neurons');
yy = ylabel('$R^2$');
set(yy,'interpreter', 'latex')
set(xx,'interpreter', 'latex')
ax = gca;
ax.FontSize = 18;
ax.XMinorTick = 'on';
ax.YMinorTick = 'on';
hold on;
xline(38);
hold off;

prompt = input('save files? (1-0): ');
if prompt
writeMat2File(HPmat_freq, fileFreq, {'f'}, 1, false)
writeMat2File(HPmat_amp, fileAmp, {'f'}, 1, false)
end
%% 8) Minimization of the error

for tonight = 7
resultsFilename = ['Results_L2FreqMap_0.5adist_',datasetType,'_',int2str(tonight)];
% preallocate and set variables
nRealizations = 1;
plotData = [1, false, false]; % end figures, loss fx, first guess
NpeaksAxis = 1:tonight;
freqMatrix = zeros(nRealizations,length(NpeaksAxis));
ampMatrix = zeros(nRealizations,length(NpeaksAxis)); 
parsMatrix = zeros(nRealizations,12);
gauss = randn(nRealizations,1);
mapComsol = zeros(nRealizations, length(f0));

options = optimset('fminsearch');
options = optimset(options, 'TolFun',1e-9,'TolX',1e-9, 'MaxFunEvals',10e3,'MaxIter', 10e3,...
    'DiffMinChange', 1, 'DiffMaxChange', 200); 

algorithm = 'fixRayleighParams';
rayleighParams = [60, 20e-6];
if strcmp( algorithm, 'fixRayleighParams')
disp('        FIXED RAYLEIGH PARAMS        ')
disp(['alpha = ',num2str(rayleighParams(end-1)),'  --   beta = ',num2str(rayleighParams(end))]);
end

% minimiziation process
for ii = 1:nRealizations
    tStart = tic;
    density = rho*(1+0.05*gauss(ii));
    disp(['rho = ', num2str(round(density,2))]);
    if strcmp(datasetType,'raw')
        [xpar, map, f_out, amp_out, fval, idxComsol] = ...
        minimization_FA(options, fNet, ampNet, Dataset_FA, f0, fAmps,density, NpeaksAxis, plotData, algorithm, rayleighParams);
    else
        [xpar, map, f_out, amp_out, fval, idxComsol] = ...
        minimization_FA(options, fNet, ampNet, dSet, f0, fAmps,density, NpeaksAxis, plotData, algorithm, rayleighParams);
    end
        
    freqMatrix(ii,:) = f_out(map).'; 
    ampMatrix(ii,:) = amp_out(map).';
    parsMatrix(ii,:) = xpar.';
    mapComsol(ii,:) = idxComsol.';
    disp(toc(tStart));
end
 

meanFreq = mean(freqMatrix);
varFreq = var(freqMatrix);

figure() 
plot(1:length(f0), f0, '-o');
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
cd([pwd,'\Results'])
finalOutTable = writeMat2File([meanMechParams; matlabStd*100],[resultsFilename,'.csv'], varyingParamsNames, 10,true); 
end
%% check Results
cd([csvPath, '\Results']);
filesList = cellstr(ls([csvPath, '\Results']));
filesList = filesList(3:end);

for ii = 1:length(filesList)
    if ii ==1 
        NpeaksAxis = 1:7;
    else
        NpeaksAxis = 1:ii;
    end
    xPars = readmatrix(filesList{ii});
    xPars = xPars(1,:).';
    getFrequencyEstimation(f0, fNet, xPars,NpeaksAxis, plotData)
end

%% 7.2) Check quality of the result  (eigenfrequency study with estimated params)

 for jj = 1:length(varyingParamsNames)
                model.param.set(varyingParamsNames(jj), meanMechParams(jj));
 end
%params = mphgetexpressions(model.param);
model.mesh('mesh1').run;
model.study('std1').run(); 
evalFreqz = mpheval(model,'solid.freq','Dataset','dset1','edim',0,'selection',1);
eigenFreqz = real(evalFreqz.d1');

figure()
plot(1:length(f0(indexReal)), f0(indexReal), '-o');
hold on;
plot(1:length(eigenFreqz(indexComsol)), eigenFreqz(indexComsol),'-x');
xlim([1,length(eigenFreqz(indexComsol))]);
xlabel('mode number N');
ylabel('frequency [Hz]');
legend('real f0','Comsol - minimization');
title('Real Eigenfrequencies vs simulated - minimization f1-f6 , f8-f10');

writeMat2File(eigenFreqz,'EigenfreqzComsol.csv', {'f'}, 1,false); 

%% 7.3) EXPORT & COMPARE POINT RESPONSE OF FRF (frequency domain with estimated parameters)
meshSize = 6;
nPointsAxis = 200;
NpeaksUsed = [7,7];
count = 1;
dSetType = {'ordered' 'raw'};
for files = NpeaksUsed
    % Perform modal analysis first
    mechParams = table2array(readtable(['Results_L2FreqMap_0.5adist_',dSetType{count},'_',int2str(files),'.csv']));
    mechParams = mechParams(1,:);
    mechParams(end-1) = 30;
    mechParams(end) = 5e-6;
    params = mphgetexpressions(model.param);
    paramsNames = params(7:end,1);
    freqAxNames = paramsNames(end-2:end);
    fAxis = f(f <= fHigh);
    fAxisComsol = logspace(log10(fAxis(1)),log10(fAxis(end)),200);

    for ii = (1:length(mechParams))
        model.param.set(paramsNames(ii), mechParams(ii));
    end

    model.physics('solid').feature('lemm1').feature('dmp1').active(true)
    model.component('comp1').physics('solid').feature('lemm1').feature('dmp1').set('DampingType', 'RayleighDamping');
    model.component('comp1').physics('solid').feature('lemm1').feature('dmp1').set('InputParameters', 'AlphaBeta'); 

    % b) Set Rayleigh constants
    model.component('comp1').physics('solid').feature('lemm1').feature('dmp1').set('alpha_dM', 19);
    model.component('comp1').physics('solid').feature('lemm1').feature('dmp1').set('beta_dK', 1e-6);

    % c) Run study and mesh
    model.component('comp1').mesh('mesh1').feature('size').set('hauto', meshSize);
    model.study('std2').feature('freq').set('plist', num2str(fAxisComsol));
    model.mesh('mesh1').run;
    model.study('std2').run();

    % d) Export Results
    dirName = pwd;
    filenameExported = ['vel_L2FreqMap_0.5adist_',dSetType{count},'_' int2str(files)];
    model.result.export('data1').set('transpose', true);
    model.result.export('data1').set('sdim', 'fromdataset');
    exportData(model,'cpt1', dirName, filenameExported, 'solid.u_tZ'); % velocity  
    writeMat2File(fAxisComsol, [filenameExported, '_freqaxis.csv'], {'f'}, 1, false);
    count = count+1;
end

%%
count = 1;
cutLow = 53; cutHigh = 1200;
fAxis_cut = intersect(find(fAxis>=cutLow), find(fAxis<=cutHigh));
fAxisPlot = fAxis(fAxis_cut);
Hvplot = Hv(fAxis_cut);
realFRFint = trapz(fAxisPlot, abs(Hvplot)/max(abs(Hvplot)));
dSetType = {'ordered' 'raw'};
NpeaksUsed = [7,7];

for files = NpeaksUsed
    titleStr = ['n real f', int2str(files), 'gaussian ',dSetType{count}];
    filenameExported = ['vel_L2FreqMap_0.5adist_',dSetType{count},'_' int2str(files)];
    fAxisComsol = readmatrix([filenameExported, '_freqaxis.csv']);
    
    [vel] = readTuples([filenameExported,'.txt'], 1, false);
    vel = vel(4:end);
    
    fAxisComsol_cut = intersect(find(fAxisComsol>=cutLow), find(fAxisComsol<=cutHigh));
    fAxisComsol = fAxisComsol(fAxisComsol_cut);
    vel = vel(fAxisComsol_cut);
    
    [maxVals, maxLocs] = findpeaks(abs(vel)/max(abs(vel)));
    f0Comsol = fAxisComsol(maxLocs);
    f0Comsol = f0Comsol(:);
    
    comsolFRFint = trapz(fAxisComsol, abs(vel)/max(abs(vel)));
    integralError = (realFRFint - comsolFRFint)./realFRFint;
    fPlot_ind = [];
    for ii = 1:length(fAxisComsol)
    diff = abs(fAxisPlot - fAxisComsol(ii));
    [minV, minLoc] = min(diff);
    fPlot_ind = [fPlot_ind minLoc];
    end
    
    normRealFRF =  abs(Hvplot(fPlot_ind).')/max(abs(Hvplot(fPlot_ind)));
    normComsolFRF =  abs(vel)/max(abs(vel));
    figure()
    plot(fAxisPlot(fPlot_ind),normRealFRF, 'LineWidth',1.5);
    hold on;
    plot( fAxisComsol, normComsolFRF,'-.', 'LineWidth',1.5);     
    stem(f0, abs(Hv(fLocs)/max(abs(Hv(fLocs) ) ) ) );
    stem(f0Comsol, abs(vel(maxLocs)/max(abs(vel(maxLocs)))));

    stem(eigs, ones(size(eigs)));

    xlabel('f    [Hz]');
    ylabel(' |H_v(f)| ');
    xlim = ([cutLow, cutHigh]);

    title(titleStr);
    count = count+1;
    
    textErr = ['integral error = ', num2str(abs(round(100*integralError,1))), '%'];
    annotation('textbox',[.73 .325 .4 .3],'String',textErr,'EdgeColor','none', 'fontSize', 16)
    
    x2 = [fAxisComsol, fliplr(fAxisComsol)];
    inBetween = [normRealFRF, fliplr(normComsolFRF)];
    area = fill(x2, inBetween, 'g', 'lineStyle', 'none');
    area.FaceColor = [0.1 0.5 0.2]; area.FaceAlpha = 0.4;
    ax = gca; ax.FontSize = 16; ax.XMinorTick = 'on'; ax.YMinorTick = 'on';
    title(' ')
    legend('real FRF', 'simulated FRF','real peaks', 'Comsol peaks', 'integral error');

end

%% Minimization 2 - narrow band frequency domain studies
    
model = mphopen(comsolModel);

fAmps = abs(Hv(fLocs));

mechParameters = table2array(readtable('Results68.csv'));
mechParameters_1stguess = [mechParameters(1,:), 19, 7e-6];
params = mphgetexpressions(model.param);
paramsNames = params(7:end,1);

model.mesh('mesh1').feature('size').set('hauto', '6');
model.mesh('mesh1').run;

    for ii = (1:length(mechParameters_1stguess))
        model.param.set(paramsNames(ii), mechParameters_1stguess(ii));
    end
         
constraintWeight = 0.05;

options = optimset('fminsearch');
options = optimset(options, 'TolFun',1e-8,'TolX',1e-8, 'MaxFunEvals',100,'MaxIter', 100,...
    'DiffMinChange', 150, 'DiffMaxChange', 300); 
firstGuess = [mechParameters_1stguess(2:end)];

pNames4Min = {paramsNames{2:end-3}, 'alpha', 'beta'};

% fun = @(x) minimizationComsolBased(x,  model, f0, fAmps, constraintWeight, pNames4Min, mechParameters_1stguess);
fun2 = @(x) minimizationComsolBased2(x,  model, f0, fAmps, constraintWeight, pNames4Min, mechParameters_1stguess,false);

% cd('C:\Users\utente\.comsol\v56\llmatlab\codesStudy - Parameters - shear\minComsol1');
% [xpar1,fval1, exitflag, output] = fminsearch(fun, firstGuess, options);

cd('C:\Users\utente\.comsol\v56\llmatlab\codesStudy - Parameters - shear\MinComsol');
[xpar,fval, exitflag, output]  = fminsearch(fun2, firstGuess, options);

% figure()
% syms x
% f = (abs(x) *100+0.01).^-1;
% fplot(f, [-2,2])
% xlabel('x = \chi(r) - \chi(r-1)')
% ylabel('f(x)')

%% GENERATE DATA FOR LOSS FUNCTION EVALUATION AND STUDY
cd(baseFolder);

model = mphopen(comsolModel);
mechParameters = table2array(readtable('Results68.csv'));
mechParameters_1stguess = [mechParameters(1,:), 19, 7e-6];

params = mphgetexpressions(model.param);
paramsNames = params(7:end,1);

model.mesh('mesh1').feature('size').set('hauto', '6');
model.mesh('mesh1').run;
firstGuess = [mechParameters_1stguess(2:end)];

paramsNames4Min = paramsNames(2:end-3);

cd('C:\Users\utente\.comsol\v56\llmatlab\codesStudy - Parameters - shear\alphaBetaVariationStudy')
[inputTable, ampsTable, eigTable] = genFreqAmpData(firstGuess, model, paramsNames4Min);


cd('C:\Users\utente\.comsol\v56\llmatlab\codesStudy - Parameters - shear\Data4LossFunction')
[inputTable, ampsTable, eigTable] = genLFData(firstGuess, model, paramsNames4Min);


    %% 7.4) check the Q factor values for simulated and real FRFs
    
    fAxis = 20:0.5:fHigh;
    [Hv,f0, fLocs, csis, Q] = EMASimple(HvSVD, fAxis,1e-3, 3);
    f0 = f0(2:end); csis = csis(2:end); Q = Q(2:end); fLocs = fLocs(2:end);
    %fAxis = 50:0.5:600;
    [HvComsol,f0Comsol, maxLocs, csisComsol, QComsol] = EMAPoly(vel, fAxis,1e-10, 1, false);
    
    Qs = [Q.'; QComsol(1:length(Q)).'];
    Qtable = array2table(Qs, 'variableNames',{'Q1' 'Q2' 'Q3' 'Q4' 'Q5' 'Q6' 'Q7' 'Q8' 'Q9' 'Q10'}, 'rowNames', {'Real' 'Comsol'});
    writeMat2File(Qs,'Qs.csv', {'Q'}, 1,false); 

    
    %% 7.1a) CORRELATIONS OF INPUTS (check if they actually are gaussian)
nPars = 12;
parsAxis = 1:length(referenceVals);
Cor = zeros(nPars,nPars);
for m1 = parsAxis % Create correlations for each experimenter
 for m2 = parsAxis % Correlate against each experimenter
  Cor(m2,m1) = abs(corr(Dataset_FA.inputs(:,m1),Dataset_FA.inputs(:,m2)));
 end
end

figure()
imagesc(Cor)
colorbar();
set(gca,'YTick',parsAxis,'YTickLabel',mechParamsNames(parsAxis))
set(gca,'XTick',parsAxis,'XTickLabel',mechParamsNames(parsAxis))


%% 7.1b) CORRELATIONS INPUTS/OUTPUTS frequency
parsAxis = 1:length(mechParamsNames);
outAxis = 1:length(Dataset_FA.outputsEig(1,1:10));
Cor = zeros(length(parsAxis),length(outAxis));

fNames = {};
for ii = outAxis
    fNames{ii} = ['f_{', int2str(ii),'}'];
end

figure()
title('inputs vs eigenfrequencies correlation')
for m1 = parsAxis % Create correlations for each experimenter
 for m2 = outAxis % Correlate against each experimenter
  Cor(m1,m2) = abs(corr(Dataset_FA.inputs(:,m1), Dataset_FA.outputsEig(:,m2)));
 end
end
imagesc(Cor)
colorbar();
title('inputs vs frequencies correlation');
set(gca, 'fontSize', 15);
set(gca,'YTick',parsAxis,'YTickLabel',mechParamsNames(parsAxis))
set(gca,'XTick',outAxis,'XTickLabel',fNames)
corrFreq = array2table(Cor, 'variableNames', fNames, 'rowNames', mechParamsNames);


%% 7.1c) CORRELATIONS INPUTS/OUTPUTS amplitude

parsAxis = 1:length(mechParamsNames);
outAxis = 1:length(Dataset_FA.outputsAmp(1,1:10));
Cor = zeros(length(parsAxis),length(outAxis));

fNames = {};
for ii = outAxis
    fNames{ii} = ['a_{', int2str(ii),'}'];
end

figure()

for m1 = parsAxis % Create correlations for each experimenter
 for m2 = outAxis % Correlate against each experimenter
  Cor(m1,m2) = abs(corr(Dataset_FA.inputs(:,m1), db(abs(Dataset_FA.outputsAmp(:,m2))) ));
 end
end
imagesc(Cor)
colorbar();
title('inputs vs amplitudes correlation');
set(gca, 'fontSize', 15);
set(gca,'YTick',parsAxis,'YTickLabel',mechParamsNames(parsAxis))
set(gca,'XTick',outAxis,'XTickLabel',fNames)

corrAmp = array2table(Cor, 'variableNames', fNames, 'rowNames', mechParamsNames);

figure()
scatter(Dataset_FA.inputs(:,2),db( Dataset_FA.outputsAmp(:,8)));
xlabel('\alpha');
ylabel('a_1 |_{dB}');

