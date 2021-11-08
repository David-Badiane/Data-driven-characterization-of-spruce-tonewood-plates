%% Multilinear regression main

%% 0) Initial setup

% Reference folders
% cd(baseFolder)
% rmpath(genpath(baseFolder));
% specify paths
baseFolder = pwd;
csvName = 'csv_wedge_scan';
testPath = [baseFolder, '\testSets'];
trainPath = [baseFolder, '\trainSets'];
csvPath = [baseFolder, '\', csvName];
multilinPath = [csvPath,'\Multilin'];
mkdir(multilinPath);

% add paths to workline
idxs   = strfind(baseFolder,'\');
addpath(genpath([baseFolder(1:idxs(end)), 'functions']));
addpath(csvPath);
addpath(testPath);
addpath(trainPath);
idx = strfind(csvName, '_');
referenceVals = readmatrix(['referenceVals',csvName(idx(2):end),'.csv']);
dDirs = {'csv_wedge_scan'};
[Dataset_FA, csvPath, datasetType] = fetchDataset_wedge(0, baseFolder, 1, dDirs{1});
modesOrder = Dataset_FA.modesOrder;
varyingParamsNames = {'rho' 'Ex' 'Ey' 'Ez' 'Gxy' 'Gyz' 'Gxz' 'vxy' 'vyz' 'vxz' 'alpha' 'beta'};
%% 1.1) Gen the train test division Idxs

% initialize image counter just once 
% count = 0;
% Save train/test sets with a single test Set

nSamples = length(Dataset_FA.inputs(:,1));
for jj =  1
    csvDataset = dDirs{jj};
    [Dataset_FA, csvPath, datasetType] = fetchDataset_wedge(0, baseFolder, true, csvDataset);
    testIdxs = randsample(1:nSamples, floor(0.1*nSamples));
    trainIdxs = setdiff(1:nSamples, testIdxs);    
    multilinPath = [csvPath,'\Multilin'];
    cd(multilinPath);
    save('ML_traintestIdxs', 'testIdxs', 'trainIdxs');
end


%% 1.2) Gen inner train / test set 
trainSet = struct('inputs', [], 'outputs', []);
testSet = struct('inputs', [], 'outputs', []);
saveData = true;

for jj = 1
    csvDataset = dDirs{jj};
    [Dataset_FA, csvPath, datasetType] = fetchDataset_wedge(0, baseFolder, true, csvDataset);    
    multilinPath = [csvPath,'\Multilin'];
    cd(multilinPath);
    load('ML_traintestIdxs.mat')
    modesIdxs = Dataset_FA.modesIdxs(1,:);
    trainSet.inputs  = Dataset_FA.inputs(trainIdxs, :);
    trainSet.outputs = Dataset_FA.outputsEig(trainIdxs, modesIdxs);
    testSet.inputs   = Dataset_FA.inputs(testIdxs, :);
    testSet.outputs  = Dataset_FA.outputsEig(testIdxs, modesIdxs);      
    if saveData
        save('setsMLR', 'trainSet', 'testSet');
    end
end

%% 2) ordered vs raw precision of multilinear
getModes = {1:9; 1:8};
[Dataset_FA, csvPath, datasetType] = fetchDataset_wedge(0, baseFolder, 1, dDirs{1});

MLs = {};
for jj = 1
    for kk = 1:2
    % fetch dataset
    csvDataset = dDirs{jj};
    [Dataset_FA, csvPath, datasetType] = fetchDataset_wedge(0, baseFolder, 0, csvDataset);
    modesIdxs = Dataset_FA.modesIdxs(1,getModes{kk});
    modesOrder = Dataset_FA.modesOrder;

    % save folders, load train/test sets
    cd(multilinPath);
    load('ML_trainTestIdxs');
    trainSet.inputs  = Dataset_FA.inputs(trainIdxs, :);
    trainSet.outputs = Dataset_FA.outputsEig(trainIdxs, modesIdxs);
    testSet.inputs  = Dataset_FA.inputs(testIdxs, :);
    testSet.outputs = Dataset_FA.outputsEig(testIdxs, modesIdxs); 
    
    % start multilinear regression
    varNames = modesOrder(getModes{kk});
    [ML] = multilinearRegress(trainSet, testSet,length(getModes{kk}), varNames, referenceVals);
    MLs{kk} = ML;
    actual    = (ML.outs);    predicted = (ML.predOuts);
    predicted = predicted(:,1:length(actual(1,:)));
    disp([csvDataset,' ', ordStr{kk},  ' : mean R2 over ', int2str(length(getModes)), ' modes = ', num2str(mean(computeR2(actual, predicted)))])
    disp(computeR2(actual, predicted))
    end
end

%% 3)  MLR actual algorithm + gen imgs
% define count = 0 before starting
saveImg = false;
if saveImg 
tag = input('tag of the picture name ');
else, tag = ' ' ; end

saveNames = {['ML_scan' tag]};
nModes = 9;

% set const params of imgData
xLength = 800; yLength = 2/3*xLength;
xylims = [0.03, 0.08, 0.08];
xLabel = 'Actual Frequency'; yLabel = 'Predicted Frequency';
tickSize = 16; tickType = '.';
legenda = {};
for ii = 1:nModes
    mode = modesOrder{ii};
    legenda{ii} = ['$', mode(1),'_{',mode(2), ',', mode(3) ,'}$'];
end
legendPos = [0.75 0.4 0.1 0.2];
lineWidth = 1.5;
fontSize = 36; markerAlpha = 0.2;
saveDir = 'C:\Users\utente\Desktop\polimi\Thesis FRF2Params - violins\paperFigures_NNs\';
MLs = {};

for jj = 1
    % fetch dataset
    csvDataset = dDirs{jj}
    [Dataset_FA, csvPath, datasetType] = fetchDataset_wedge(0, baseFolder, true, csvDataset);
    % save folders, load train/test sets
    multilinPath = [csvPath,'\Multilin'];
    mkdir(multilinPath);
    cd(multilinPath);
    load('setsMLR');
    cd(baseFolder);
    % start multilinear regression
    varNames = Dataset_FA.modesOrder;
    [ML] = multilinearRegress(trainSet, testSet,nModes, varNames, referenceVals);
    MLs{jj} = ML;
    % set varying params of imgData
    figureN = mod(15 + count,6); 
    imgN= 100 + count;
    xylim = xylims(jj);
    meanTrain = mean(testSet.outputs(:,1:nModes));
    meanTest  = mean(testSet.outputs(:,1:nModes));
    actual    = (ML.outs);
    predicted = (ML.predOuts); predicted = predicted(:,1:length(actual(1,:)));
    imgData = defImg_scatter(xLength, yLength, imgN, xLabel, yLabel, tickSize,...
                    tickType, lineWidth, legenda, legendPos, fontSize,...
                    markerAlpha, xylim, saveDir, saveNames{jj});
%     [img]   = export_scatter(actual ,predicted, imgData, 'columns', saveImg);
    count   = count+1;
    disp([csvDataset ' : mean R2 over ', int2str(nModes), ' modes = ', num2str(mean(computeR2(actual, predicted)))])
    disp([csvDataset ' R2 ' num2str(computeR2(actual, predicted))])
    cd(csvPath)
end
save(['MLRfreqs_' int2str(nModes)], 'MLs')
%% 4) study linearModels Coefficients
linModels = {};
intVals = [];
coeffsMatr = [];
cd(multilinPath);
load('MLRfreqs_9.mat');
nModes = length(MLs{1}.linMdls);

for ii = 1
    linModels{ii} = MLs{ii}.linMdls;
    [Dataset_FA, csvPath, datasetType] = ...
        fetchDataset_wedge(0, baseFolder, 1, dDirs{ii});
    meanIns = mean(Dataset_FA.inputs).';
    for jj = 1:nModes
           coeffs = table2array(linModels{ii}{jj}.Coefficients); 
           coeffs= coeffs(:,1);
           intercept = coeffs(1);
           normCoeffs = coeffs(2:end);%.*meanIns;
           interceptVals(jj) = coeffs(1);
           coeffsMatr(jj,:)  = [intercept; normCoeffs];
    end
end

coeffsTable = array2table(coeffsMatr,'rowNames', Dataset_FA.modesOrder(1:nModes),...
    'variableNames', {'intercept' varyingParamsNames{1:end-2}})

startWith = 2;
normCoeffs = coeffsMatr(:,startWith:end).*mean(Dataset_FA.inputs(:,startWith-1:end-2));

T = array2table(normCoeffs, 'rowNames', Dataset_FA.modesOrder(1:nModes),...
    'variableNames',  varyingParamsNames(1:end-2))


startWith = 3;
xIdxs = startWith-1:length(normCoeffs(1,:)); yIdxs = 1:length(normCoeffs(:,1))-1;
xLengthImg = 900; yLengthImg = 4/5*xLengthImg; imgN = 33;
xLabel = 'mech Params'; yLabel = 'eigenfrequencies'; 

maxC = 0.95;
bluetogreen = [linspace(0, 0, 100).' linspace(0,maxC,100).' linspace(maxC, 0, 100).'];
greentoyellow = [linspace(0, maxC, 100).' linspace(maxC,maxC,100).' linspace(0, 0, 100).'];
yellowtored = [linspace(maxC, maxC, 100).' linspace(maxC,0,100).' linspace(0, 0, 100).'];
colorMap = [bluetogreen; greentoyellow; yellowtored];

textFontSize = 15; axFontSize = 20; 
xTickLabels = varyingParamsNames(startWith-1:end-2);
yTickLabels = modesOrder(1:nModes);

minImg = min(abs(normCoeffs(yIdxs, xIdxs)), [], 'all');
maxImg = max(abs(normCoeffs(yIdxs, xIdxs)), [], 'all');
cbarLabel = 'relative importance';
displayCbar = true; 

imgData = defImg_matrix(xIdxs, yIdxs, xLengthImg, yLengthImg, imgN, xLabel, yLabel, colorMap,...
                   textFontSize, axFontSize, xTickLabels, yTickLabels, cbarLabel, displayCbar);
               
img = export_matrix(abs(normCoeffs),imgData, 2, true); 

%% 5) Check Caldersmith formulas performance vs MLR coefficients
% f11 <-> col 1 || f02 <-> col 3  || f20 <-> col 7
% setup
nTuples = length(Dataset_FA.inputs(:,1));
MLRIdxs = [1,2,4,5,8];
MLRmodes = Dataset_FA.modesOrder(MLRIdxs)
modesIdxs = Dataset_FA.modesIdxs(1,:);
MLRFreqs = Dataset_FA.outputsEig(:, modesIdxs(MLRIdxs));

% take dataset input params
densities = Dataset_FA.inputs(:,1);
datasetParamsIdxs = [2,3,5,6,7]; % Ex, Ey, Gxy, Gyz, Gxz
datasetParams = Dataset_FA.inputs(:,datasetParamsIdxs);
normDataset = datasetParams./datasetParams(:,1);

poissonRatios = Dataset_FA.inputs(:,8);

% preallocate results 
MLRParams = zeros(nTuples,length(datasetParamsIdxs));

for ii = 1:nTuples
% MLR
mechParamsMLR = computeParamsMultilin_wedge(MLRFreqs(ii,:), densities(ii), coeffsTable, MLRIdxs);
MLRParams(ii,:) = table2array(mechParamsMLR);
end
normMLRParams = MLRParams./(MLRParams(:,1));
% specify variables for plots
nPlot =1:nTuples;
xlabels = {'E_L dataset' 'E_R dataset' 'G_{LR} dataset' 'G_{RT} dataset' 'G_{LT} dataset'}; 
ylabels = {'E_L MLR'     'E_R MLR'     'G_{LR} MLR'     'G_{RT} MLR'     'G_{LT} MLR' };
titles =  {'E_L scatter' 'E_R scatter' 'G_{LR} scatter' 'G_{RT} scatter' 'G_{LT} scatter'};
xlabelsRatio = { 'E_R/E_L dataset' 'G_{LR}/E_L dataset' }; ylabelsRatio = { 'E_R/E_L MLR' 'G_{LR}/G_L MLR' };
titlesRatio = { 'E_R/E_L scatter' 'G_{LR}/E_L scatter'};
testi = {'E_L' 'E_R' 'G_LR'  'G_RT'   'G_LT'};

% do plots
count = 0;
for ii = 1:5  
    tempParam = datasetParams(nPlot,ii);
    tempMLRParam = MLRParams(nPlot, ii);
    
    % figure of parameters
    figure(1+ count); clf reset;
    s = scatter( tempParam, tempMLRParam,9, 'o', 'markerEdgeAlpha', .5); hold on;
    minV = min([tempParam(:); tempMLRParam(:)]);
    maxV = max([tempParam(:); tempMLRParam(:)]);
    plot([minV maxV], [minV maxV], 'lineWidth', 1.2);
    hold off; xlabel(xlabels{ii}); ylabel(ylabels{ii}); title(titles{ii}); ax = gca; ax.FontSize = 14;
      
    R2 = computeR2(tempParam, tempMLRParam);
    errNMSE = MSE(tempParam, tempMLRParam);
    [errMAPE,varMAPE] = MAPE(tempParam, tempMLRParam);
    disp([testi{ii},' MLR        : R2 = ', num2str(R2,2), '    NMSE = ', num2str(db(errNMSE),2), '    MAPE = ', num2str(errMAPE,2) '%   var = ' num2str(varMAPE,2)])

    % figure for normalized params
%     if ii >1
%         figure(40)
%         subplot(1,2,ii-1)
%         hold on;
%         scatter( normDataset(nPlot,ii), normMLRParams(nPlot,ii), 9, 'filled'); hold on;
%         plot([0,1],[0,1]);
%         xlim([min(normDataset(nPlot,ii))*0.9, max(normDataset(nPlot,ii))*1.1]); hold off;
%         xlabel(xlabelsRatio{ii-1}); ylabel(ylabelsRatio{ii-1}); title(titlesRatio{ii-1}); ax = gca; ax.FontSize = 14;
%     end
    count = count+1;
end
