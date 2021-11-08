%% main_FA_plate_hyperparameters

%% 0) Init
% =========================================================================
% Reference folders
% cd(baseFolder)
% rmpath(genpath(baseFolder));
% specify paths
baseFolder = pwd;
csvName = 'csv_plate_gaussian';
convergencePath = [baseFolder, '\convergenceTest_plate'];
testPath = [baseFolder, '\testSets'];
trainPath = [baseFolder, '\trainSets'];
NNPath = [baseFolder,'\NNs'];
csvPath = [baseFolder, '\', csvName];

% add paths to workline
idxs   = strfind(baseFolder,'\');
addpath(genpath([baseFolder(1:idxs(end)), 'functions']));
addpath([baseFolder, '\data']);
addpath(csvPath);
addpath(testPath);
addpath(trainPath);
addpath(NNPath);

%% 1) Dataset splitting for hyperparameters
% =========================================================================
% fetch dataset
cd(baseFolder)
rmpath(genpath(csvPath));
saveData = 0;
[Dataset_FA, csvPath, datasetType, datasetDistr] = fetchDataset_plate(saveData, baseFolder);

if strcmp(datasetType, 'ordered')
    cd(csvPath); modesPresence = readtable('modesPresence.csv');
    datasetModes = modesPresence.Properties.VariableNames;
    modesPresence = table2array(modesPresence);
end

HPFolder = [csvPath,'\HyperParameters'];
% setup
nIn = length(Dataset_FA.inputs(1,:));
referenceVals = readmatrix('referenceVals.csv');

test = sort(randsample(2:length(Dataset_FA.inputs(:,1)), 200));
model = setdiff(1:length(Dataset_FA.inputs(:,1)), test); 

% Split Dataset
trainSet = struct('inputs', Dataset_FA.inputs(model,:),...
                  'outputsEig', Dataset_FA.outputsEig(model,:),...
                  'outputsAmp', Dataset_FA.outputsAmp(model,:));
testSet =  struct('inputs', Dataset_FA.inputs(test,:),...
                  'outputsEig', Dataset_FA.outputsEig(test,:),...
                  'outputsAmp', Dataset_FA.outputsAmp(test,:));
% save files
fileName = ['HPsets_', datasetType];
cd(HPFolder)
save(fileName, 'trainSet', 'testSet');
cd(baseFolder) 

%% 2) Hyperparameters
% =========================================================================
%flags = [writeNewFiles doFreq doAmp saveData] 
flags = [0 0 1 1];
nNeuronsVec = 20% 8:4:32 64];
nLayersVec = 1% 2 3 4];
nLaxis_freq =1; nLaxis_amp = [1,2,3,4];
nModesGet = 4;
dDirs =  {'csv_plate_gaussian' 'csv_plate_uniform_0.50' 'csv_plate_uniform_0.75'};
HPResults = {};
for kk = 1:length(dDirs)
[HPData] = NN_hyperparameters(nNeuronsVec, nLayersVec, nLaxis_freq, nLaxis_amp,...
                 nModesGet, baseFolder, dDirs{kk}, flags);
HPResults{kk} = HPData; 
end
%% 3) get min and max values of HP
% =========================================================================

dDirs = {'csv_plate_gaussian', 'csv_plate_uniform_0.50','csv_plate_uniform_0.75'};
minn = []; maxx = [];
for ii = 1:length(dDirs)
HPFolder = [baseFolder, '\', dDirs{ii}, '\HyperParameters'];
cd(HPFolder)
HH = readmatrix(['HPamp_', dDirs{ii}(11:12) ,'_ordered.csv']);
HH = HH(HH~=0);
minn(ii) = min(HH, [], 'all');
maxx(ii) = max(HH, [], 'all');
end
minn = min(minn);
maxx = max(maxx);

%% 4) HP figures 
% =========================================================================
% define count just once if you want to have different figures each time
% count = 0
dDirs = {'csv_plate_gaussian', 'csv_plate_uniform_0.50','csv_plate_uniform_0.75'};
HPnames = {'HPG' 'HPU0.5' 'HPU0.75'};
roundN = [3, 3, 3];
textFontSizes = [20 20 20];
displayCbars = [false false false];
xLengths = [500, 500, 500];

for jj = 1:length(dDirs)
    csvName = dDirs{jj};
    csvPath = [baseFolder,'\', csvName];
    HPFolder = [csvPath,'\HyperParameters']; datasetType = 'ordered';
    datasetDistr = dDirs{jj}(11:12);
    cd(HPFolder);
    HPmat_amp = readmatrix(['HPamp_', datasetDistr(1:2),'_',datasetType,'.csv']);

    % amp figure;
    xIdxs = [1,2,3,4];  yIdxs = [2,4,8,16,32,64];
    xLengthImg = xLengths(jj); yLengthImg = xLengthImg*3/5;
    imgN = 30+count; 
    xLabel = 'N Layers'; yLabel = 'N Neurons';
    colorMap = winter; 
    textFontSize = textFontSizes(jj); axFontSize = 22;
    xTickLabels = {}; yTickLabels = {};
    for ii = 1:length(xIdxs), xTickLabels(ii) = {num2str(xIdxs(ii))}; end
    for ii = 1:length(yIdxs), yTickLabels(ii) = {num2str(yIdxs(ii))}; end
    cbarLabel = '$\overline{R^2}$';
    cbarLim = [minn, maxx];
    displayCbar = displayCbars(jj); 

    imgData = defImg_matrix(xIdxs, yIdxs, xLengthImg, yLengthImg, imgN, xLabel, yLabel, colorMap,...
                       textFontSize, axFontSize, xTickLabels, yTickLabels, cbarLabel, cbarLim, displayCbar);
    img_a = export_matrix(HPmat_amp,imgData,roundN(jj));

cd('C:\Users\utente\Desktop\polimi\Thesis FRF2Params - violins\paperFigures_NNs\HP')
saveas(img_a,[HPnames{jj} '_.png']);
count = count + 1;
end

%% 5) freq figure
% =========================================================================
xIdxs = 1;  yIdxs = [1:8];
xLengthImg = 400; yLengthImg = 500; imgN = 31+count; 
colorMap = summer; textFontSize = 20; axFontSize = 24;
xTickLabels = {}; yTickLabels = {};
for ii = 1:length(xIdxs), xTickLabels(ii) = {num2str(xIdxs(ii))}; end
for ii = 1:length(yIdxs), yTickLabels(ii) = {num2str(yIdxs(ii))}; end

imgData = defImg_matrix(xIdxs, yIdxs, xLengthImg, yLengthImg, imgN, xLabel, yLabel, colorMap,...
                   textFontSize, axFontSize, xTickLabels, yTickLabels);
img_f = export_matrix(HPmat_freq,imgData,4);

count = count + 2;

[ii,jj] = find(HPmat_amp == max(HPmat_amp, [],'all'));
maxHP = HPmat_amp(ii,jj);
disp(' ');
disp([datasetDistr, ' dataset ', datasetType]); 
disp(['HP RESULTS: max at ii = ',int2str(ii) ,' and jj =', int2str(jj), ' with R2 = ', num2str(round(maxHP,3))]); 
disp(' ');




%% 6) Train optimal neural networks
% =========================================================================
dDirs = {'csv_plate_gaussian_G10' 'csv_plate_uniform_0.50' 'csv_plate_uniform_0.75'};
getOrdered = true;
modesGet = 15;

for jj = 1
    % fetch 
    cd(baseFolder);
    [Dataset_FA, csvPath, datasetType, datasetDistr, HPFolder] = ...
        fetchReduceDataset_plate(baseFolder, modesGet, getOrdered, dDirs{jj});
    
    Dataset_FA
    % read HP files
     cd(HPFolder)
     
   HPamp = readmatrix(['HPamp_', datasetDistr(1:2),'_ordered_allModes.csv']);
% 
    % extract number of neurons, number of layers
    [nN_a,nL_a] = find( HPamp == max(HPamp, [], 'all'));
    nN_a = min(nN_a); nL_a = min(nL_a); 

    % train network
    [aNet] = NN_train(Dataset_FA.inputs, ...
                        db(abs(Dataset_FA.outputsAmp)),...
                        nN_a, nL_a);
                    
    [fNet] = NN_train(Dataset_FA.inputs, ...
                        Dataset_FA.outputsEig,...
                        20, 1);
    % save it
    cd(csvPath)
    save(['optNN_' int2str(modesGet) 'Modes'], 'aNet', 'fNet');
end

