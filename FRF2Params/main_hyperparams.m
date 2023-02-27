%% __________________ MAIN HYPERPARAMETERS TUNING _________________________
% THIS IS THE MAIN PROGRAM ALLOWING TO TRAIN AND OPTIMIZE THE NEURAL NETWORKS
% ARCHITECTURE
% 
% A) split dataset in train set and test set
% B) hyperparameters tuning --> grid search train NNs varying the number of 
%                               neurons (N) and number of layers (L), each 
%                               architecture is trained and tested
%                               nRealizations times, acquiring the average
%                               coefficient of determination
% C) optimized nns training
% -------------------------------------------------------------------------
% summary:
% section 0) initial setup, reference folders
% section 1) randomly split the dataset into train set and test set
% section 2) hyperparameters (HP) tuning
% section 3) get minimum and maximum R2 values (coefficient of determination) from HP tuning
% section 4) train optimal neural networks 
% section 5) generate figures of HP tuning grid search
%
%% section 0) Init
% =========================================================================
% to remove all previous paths
remPath = false; % fnL_ag to remove paths, set to true if you want to remove all paths
if remPath
 cd(baseFolder)
 rmpath(genpath(baseFolder));
end

% directories and paths nN_ames
baseFolder = pwd;
csvnN_ame = 'csv_gPlates_';
csvPath = [baseFolder, '\', csvnN_ame];
NNsPath = [csvPath,'\Neural networks'];
mkdir(NNsPath);

% add paths to workline
idxs   = strfind(baseFolder,'\');
addpath(genpath([baseFolder(1:idxs(end)), 'functions']));
addpath([baseFolder, '\data']);
addpath(csvPath);
addpath(NNsPath);

% fetch dataset center values
cd(baseFolder)
referenceVals = readmatrix('referenceVals.csv');

%% 1) Dataset splitting into train and test for hyperparameters
% =========================================================================
% flags
saveData = 0;
% variables 
modesGet = 15;
getOrdered = 0;

% fetch dataset
[Dataset_FA, csvPath, HPPath] = ...
        fetchDataset(baseFolder, modesGet, getOrdered, csvnN_ame, saveData);
% make NNs hyperparameters directory    
mkdir(HPPath);

% randomly split dataset into train and test sets
testIdxs = sort(randsample(2:length(Dataset_FA.inputs(:,1)), floor(0.1*length(Dataset_FA.inputs(:,1)))));
trainIdxs = setdiff(2:length(Dataset_FA.inputs(:,1)), testIdxs); 

% fill train set and test set
trainSet = struct('inputs', Dataset_FA.inputs(trainIdxs,:),...
              'outputsEig', Dataset_FA.outputsEig(trainIdxs,:),...
              'outputsAmp', Dataset_FA.outputsAmp(trainIdxs,:));
testSet =  struct('inputs', Dataset_FA.inputs(testIdxs,:),...
              'outputsEig', Dataset_FA.outputsEig(testIdxs,:),...
              'outputsAmp', Dataset_FA.outputsAmp(testIdxs,:));

% save files
if getOrdered == 0, filename = ['HPsets_raw_']; else filename = ['HPsets_ord_']; end 
cd(HPPath)
save(filename, 'trainSet', 'testSet');
cd(baseFolder) 

%% section 2) Hyperparameters tuning
% =========================================================================
% flags = [writeNewFiles doFreq doAmp saveData] 
% writeNewFiles --> write new hyperparameters tuning csv files or continue on previous ones
% doFreq       --> perform hyperparameters tuning for NN that predicts eigenfrequencies
% doAmp        --> perform hyperparameters tuning for NN that predicts amplitudes
% saveData     --> save all data from HP tuning (trained neural networks,
%                  R2 (coefficient of determinN_ation)of each training, training data)
flags = [0 1 1 1];
% variables
nNeuronsVec = [1]; % vector with the number of neurons
nLayersVec = [1 2 3 4];        % vector with the number of layers
nLaxis_freq =[1 2 3];        % chooses number of layers for freq NN 
nLaxis_amp = [1 3 4];        % chooses what number of layers for freq NN
nModesGet = 15;                % number of predicted modes
dataset_directory =  'csv_gPlates_';
nRealizations = 1;            % number of realizations of the training for HP tuning
R2s = [];                  % preallocate for the mean R2s of each training

[HPData, HP_filename_freq, HP_filename_amp] = NN_hyperparameters(nNeuronsVec, nLayersVec, nLaxis_freq, nLaxis_amp,...
             nModesGet, baseFolder, dataset_directory, flags, 'HPSets_raw_', nRealizations);

R2s = [R2s; HPData{1}.R2vecs{end}{end}];

%% section 3) get max values of HP and optimal number of nL_ayers and neurons
% =========================================================================
cd(HPPath)
HP_freq = readmatrix([HP_filename_freq, '.csv']);
HP_amp  = readmatrix([HP_filename_amp,  '.csv']);

% get both minimum and maximum values
max_f = max(HP_freq, [], 'all');
min_f = min(HP_freq, [], 'all');
max_a = max(HP_amp, [], 'all');
min_a = min(HP_amp, [], 'all');

% best number neurons and number layers for both frequency and amplitude
[nN_f, nL_f] = find(HP_freq == max_f);
[nN_a, nL_a] = find(HP_amp == max_a);

% show message 
disp(['frequency HP RESULTS: max at ii = ',int2str(nN_f) ,' and jj =', int2str(nL_f), ' with R2 = ', num2str(round(max_f,5))]); 
disp(['amplitude HP RESULTS: max at ii = ',int2str(nN_a) ,' and jj =', int2str(nL_a), ' with R2 = ', num2str(round(max_a,3)), newline]); 


%% section 4) Train optimal neural networks
% =========================================================================
dataset_directory = 'csv_gPlates_';
getOrdered = true;  % boolean for ordered or raw dataset
saveNow = false;    % boolean to decide whether to save trained NNS
modesGet = 15;

cd(baseFolder);
    [Dataset_FA, csvPath, HPPath] = ...
        fetchDataset(baseFolder, modesGet, getOrdered, dataset_directory);
    
    % train network
    [aNet,tr_a] = NN_train(Dataset_FA.inputs, ...
                        db(abs(Dataset_FA.outputsAmp)),...
                        nN_a, nL_a);
                    
    [fNet, tr_f] = NN_train(Dataset_FA.inputs, ...
                        Dataset_FA.outputsEig,...
                        1, 1);
    % save it
    cd(NNsPath)
    if saveNow
        save(['optNN_' int2str(modesGet) 'Modes'], 'aNet', 'fNet', 'tr_f', 'tr_a');
    end
    
%% section 5) HP figures - skip if you don't want to see them
% =========================================================================
% general settings 
roundN = 3;
textFontSize = 20 ;
axFontSize = 22;
displayCbar = false;
xLengthImg = 500;
yLengthImg = xLengthImg*3/5;
xLabel = 'N Layers'; 
yLabel = 'N Neurons';
cbarLabel = '$\overline{R^2}$';

% --------------------------- amp figure ----------------------------------
% setting
xIdxs = [1,2,4];  
yIdxs = [4,8,16,24,32,64];
xTickLabels = {}; yTickLabels = {};
for ii = 1:length(xIdxs), xTickLabels(ii) = {num2str(xIdxs(ii))}; end
for ii = 1:length(yIdxs), yTickLabels(ii) = {num2str(yIdxs(ii))}; end
colorMap = winter; 
imgN = 30; 
cbarLim = [min_a, max_a];

% read file
HP_amp = readmatrix([HP_filename_amp,'.csv']);
% generate image
imgData = defImg_matrix(xIdxs, yIdxs, xLengthImg, yLengthImg, imgN, xLabel, yLabel, colorMap,...
                   textFontSize, axFontSize, xTickLabels, yTickLabels, cbarLabel, displayCbar);
img_a = export_matrix(HP_amp,imgData,roundN(jj), 0);

% -------------------------- freq figure ----------------------------------
% settings
xIdxs = [1,2,4];  yIdxs = [4 8 16];
imgN = 31; 
colorMap = summer; 
xTickLabels = {}; yTickLabels = {};
for ii = 1:length(xIdxs), xTickLabels(ii) = {num2str(xIdxs(ii))}; end
for ii = 1:length(yIdxs), yTickLabels(ii) = {num2str(yIdxs(ii))}; end

% read file
HP_freq = readmatrix([HP_filename_freq '.csv']);

% generate image
imgData = defImg_matrix(xIdxs, yIdxs, xLengthImg, yLengthImg, imgN, xLabel, yLabel, colorMap,...
                   textFontSize, axFontSize, xTickLabels, yTickLabels, cbarLabel, 1);
img_f = export_matrix(HP_freq,imgData,3,0);
