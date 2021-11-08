%% main_FA_plate_hyperparameters

%% 0) Init
% =========================================================================
% Reference folders
% cd(baseFolder)
% rmpath(genpath(baseFolder));
% specify paths
baseFolder = pwd;
csvName = 'csv_wedge_scan';
testPath = [baseFolder, '\testSets'];
trainPath = [baseFolder, '\trainSets'];
csvPath = [baseFolder, '\', csvName];
HPPath = [csvPath,'\HyperParameters'];
modesAnalysisPath = [csvPath, '\ModesAnalysis'];
% mkdir(HPPath)
% add paths to workline
idxs   = strfind(baseFolder,'\');
addpath(genpath([baseFolder(1:idxs(end)), 'functions']));
addpath(csvPath);
addpath(testPath);
addpath(trainPath);
addpath(NNPath);
addpath(HPPath);

%% 1) Dataset splitting for hyperparameters
% =========================================================================
% fetch dataset
cd(baseFolder)
rmpath(genpath(csvPath));
saveData = 0;
[Dataset_FA, csvPath, datasetType] = fetchDataset_wedge(saveData, baseFolder, 1, csvName);

if strcmp(datasetType, 'ordered')
    cd(modesAnalysisPath); modesPresence = readtable('modesPresence.csv');
    datasetModes = modesPresence.Properties.VariableNames;
    modesPresence = table2array(modesPresence);
end

% setup
nIn = length(Dataset_FA.inputs(1,:));
cd(baseFolder);
idx = strfind(csvName, '_');
referenceVals = readmatrix(['referenceVals',csvName(idx(2):end),'.csv']);

test = sort(randsample(2:length(Dataset_FA.inputs(:,1)),...
            round(0.1*length(Dataset_FA.inputs(:,1))) ));
train = setdiff(1:length(Dataset_FA.inputs(:,1)), test); 

modesIdxs_Eig1 = Dataset_FA.modesIdxs(1,:);
modesIdxs_Eig2 = Dataset_FA.modesIdxs(2,:);
modesIdxs_Eig3 = Dataset_FA.modesIdxs(3,:);

% Split Dataset
trainSet = struct('inputs', Dataset_FA.inputs(train,:),...
                  'outputsEig', Dataset_FA.outputsEig(train,modesIdxs_Eig1),...
                  'outputsAmp', []);
trainSet.outputsAmp = {Dataset_FA.outputsAmp(train,modesIdxs_Eig1),...
                       Dataset_FA.outputsAmp(train,modesIdxs_Eig2),...
                       Dataset_FA.outputsAmp(train,modesIdxs_Eig3)};
              
testSet =  struct('inputs', Dataset_FA.inputs(test,:),...
                  'outputsEig', Dataset_FA.outputsEig(test,modesIdxs_Eig1),...
                  'outputsAmp', []);
testSet.outputsAmp = {Dataset_FA.outputsAmp(test,modesIdxs_Eig1),...
                      Dataset_FA.outputsAmp(test,modesIdxs_Eig2),...
                      Dataset_FA.outputsAmp(test,modesIdxs_Eig3)};

% save files
fileName = ['HPsets'];
cd(HPPath)
save(fileName, 'trainSet', 'testSet');
cd(baseFolder) 

%% 2) Hyperparameters
% =========================================================================
%flags = [writeNewFiles doFreq doAmp saveData] 
flags = [1 1 1 1];
nNeuronsVec =  [8:4:32 64];
nLayersVec = [1 2];
nLaxis_freq =1; nLaxis_amp = [1,2];
nModesGet = 8;
dDirs =  {'csv_wedge_scan'};
HPResults = {};
isWedge = true;

for kk = 1:length(dDirs)
[HPData] = NN_hyperparameters(nNeuronsVec, nLayersVec, nLaxis_freq, nLaxis_amp,...
                 nModesGet, baseFolder, dDirs{kk}, flags, isWedge);
HPResults{kk} = HPData; 
end

%% 3) get min and max values of HP
% =========================================================================

dDirs =  {'csv_wedge_scan'};
minn = []; maxx = [];
for ii = 1:length(dDirs)
HPPath = [baseFolder, '\', dDirs{ii}, '\HyperParameters'];
cd(HPPath)
HH = readmatrix(['HPamp.csv']);
HH = HH(HH~=0);
minn(ii) = min(HH, [], 'all');
maxx(ii) = max(HH, [], 'all');
end
minn = min(minn);
maxx = max(maxx);

%% 4) amp figure 
% =========================================================================
% define count just once if you want to have different figures each time
% count = 0
dDirs =  {'csv_wedge_scan'};
HPnames = {'HP_scan'};
roundN = [3, 3, 3];
textFontSizes = [20 20 20];
displayCbars = [false false false];
xLengths = [500, 500, 500];


csvName = dDirs{jj};
csvPath = [baseFolder,'\', csvName];
HPPath = [csvPath,'\HyperParameters']; datasetType = 'ordered';
datasetDistr = dDirs{jj}(11:12);
cd(HPPath);
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

cd('C:\Users\utente\Desktop\polimi\Thesis FRF2Params - violins\paperFigures_FRF2Params\HP')
saveas(img_a,[HPnames{jj} '_.png']);
count = count + 1;

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
    [Dataset_FA, csvPath, datasetType, datasetDistr, HPPath] = ...
        fetchReduceDataset_plate(baseFolder, modesGet, getOrdered, dDirs{jj});
    
    Dataset_FA
    % read HP files
     cd(HPPath)
     
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