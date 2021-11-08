%% Multilinear regression main

%% 0) Initial setup
% =========================================================================

% Reference folders
% cd(baseFolder)
% rmpath(genpath(baseFolder));
% specify paths
baseFolder = pwd;
csvName = 'csv_plate_gaussian_G10';
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
addpath(NNPath)
referenceVals = readmatrix('referenceVals.csv');
dDirs = {'csv_plate_gaussian_G10', 'csv_plate_uniform_0.50','csv_plate_uniform_0.75'};
[Dataset_FA, csvPath, datasetType, datasetDistr] = fetchDataset_plate(0, baseFolder, 1, dDirs{1});
modesOrder = Dataset_FA.modesOrder;

% data sample measurements
infosTable = readtable("sampleMeasurements.xlsx");
infosMatrix = table2array(infosTable(:,3:end));
infosMatrix(:,1:2) = infosMatrix(:,1:2)*0.01;
infosMatrix(:,3:7) = infosMatrix(:,3:7)*0.001;

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

%% 1.1) Gen the train test division Idxs
% =========================================================================

% initialize image counter just once 
% count = 0;
% Save train/test sets with a single test Set

nSamples = length(Dataset_FA.inputs(:,1));
for jj =  [3,2,1]
    csvDataset = dDirs{jj};
    [Dataset_FA, csvPath, datasetType, datasetDistr] = fetchDataset_plate(0, baseFolder, true, csvDataset);
    testIdxs = randsample(1:nSamples, floor(0.05*nSamples));
    trainIdxs = setdiff(1:nSamples, testIdxs);    
    multilinPath = [csvPath,'\Multilin'];
    cd(multilinPath);
    save('ML_traintestIdxs', 'testIdxs', 'trainIdxs');
end


%% 1.2) Gen inner train / test set 
% =========================================================================
trainSet = struct('inputs', [], 'outputs', []);
testSet = struct('inputs', [], 'outputs', []);
cd([baseFolder, '\', dDirs{1}, '\Multilin']);
load('ML_traintestIdxs');
saveData = true;

for jj = 1
    csvDataset = dDirs{jj};
    [Dataset_FA, csvPath, datasetType, datasetDistr] = fetchDataset_plate(0, baseFolder, true, csvDataset);    
    mean(Dataset_FA.outputsEig(Dataset_FA.modesIdxs(1:4)))
    multilinPath = [csvPath,'\Multilin'];
    cd(multilinPath);
    load('ML_traintestIdxs.mat')
    trainSet.inputs  = Dataset_FA.inputs(trainIdxs, :);
    trainSet.outputs = Dataset_FA.outputsEig(trainIdxs, Dataset_FA.modesIdxs);
    testSet.inputs   = Dataset_FA.inputs(testIdxs, :);
    testSet.outputs  = Dataset_FA.outputsEig(testIdxs, Dataset_FA.modesIdxs);      
    if saveData
        save('setsMLR', 'trainSet', 'testSet');
    end
end

%% 2) ordered vs raw precision of multilinear
% =========================================================================
dDirs = {'csv_plate_gaussian', 'csv_plate_uniform_0.50','csv_plate_uniform_0.75'};
getModes = 1:4;
[Dataset_FA, csvPath, datasetType, datasetDistr] = fetchDataset_plate(0, baseFolder, 1, dDirs{1});
     modesOrder = Dataset_FA.modesOrder;
MLs = {};
ordStr = {'raw'  'ordered'}
for jj = 1
    for kk = 1:2
    % fetch dataset
    csvDataset = dDirs{jj};
    [Dataset_FA, csvPath, datasetType, datasetDistr] = fetchDataset_plate(0, baseFolder, 0, csvDataset);
    % save folders, load train/test sets
    multilinPath = [csvPath,'\Multilin'];
    cd(multilinPath);
    load('ML_trainTestId');
    trainSet.inputs  = Dataset_FA.inputs(trainIdxs, :);
    trainSet.outputs = Dataset_FA.outputsEig(trainIdxs, Dataset_FA.modesIdxs(getModes));
    testSet.inputs  = Dataset_FA.inputs(testIdxs, :);
    testSet.outputs = Dataset_FA.outputsEig(testIdxs, Dataset_FA.modesIdxs(getModes)); 
    cd(testPath);
    cd(baseFolder);
    % start multilinear regression
    varNames = modesOrder(getModes);
    [ML] = multilinearRegress(trainSet, testSet,length(getModes), varNames, referenceVals);
    MLs{kk} = ML;
    actual    = (ML.outs);    predicted = (ML.predOuts);
    predicted = predicted(:,1:length(actual(1,:)));
    disp([csvDataset,' ', ordStr{kk},  ' : mean R2 over ', int2str(length(getModes)), ' modes = ', num2str(mean(computeR2(actual, predicted)))])
    end
end

%% 3)  MLR actual algorithm + gen imgs
% =========================================================================
% define count = 0 before starting
saveImg = false;
if saveImg 
tag = input('tag of the picture name ');
else, tag = ' ' ; end
saveNames = {['ML_G' tag] ['ML_U0.50' tag] ['ML_U0.75' tag]};
nModes = 15;

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
    [Dataset_FA, csvPath, datasetType, datasetDistr] = fetchDataset_plate(0, baseFolder, true, csvDataset);
    % save folders, load train/test sets
    multilinPath = [csvPath,'\Multilin'];
    mkdir(multilinPath);
    cd(multilinPath);
    load('setsMLR');
    cd(baseFolder);
    % start multilinear regression
    varNames = Dataset_FA.dataOrder(Dataset_FA.modesIdxs(1:nModes));
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
    cd(csvPath)

end
save(['MLRfreqs_' int2str(nModes)], 'MLs')
%% 4) study linearModels Coefficients
% =========================================================================
linModels = {};
intVals = [];
coeffsMatr = [];
nModes = length(MLs{1}.linMdls);

for ii = 1
    linModels{ii} = MLs{ii}.linMdls;
    [Dataset_FA, csvPath, datasetType, datasetDistr] = ...
        fetchDataset_plate(0, baseFolder, 1, dDirs{ii});
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

T = array2table(coeffsMatr,...
    'rowNames', Dataset_FA.dataOrder(Dataset_FA.modesIdxs(1:nModes)),...
    'variableNames', {'intercept' varyingParamsNames{:}})

startWith = 2;
normCoeffs = coeffsMatr(:,startWith:end).*mean(Dataset_FA.inputs(:,startWith-1:end));

T = array2table(normCoeffs,...
    'rowNames', Dataset_FA.dataOrder(Dataset_FA.modesIdxs(1:nModes)),...
    'variableNames', { varyingParamsNames{:}})

figure()
startWith = 3;
imagesc(abs(normCoeffs(:,startWith-1:end)));
ax = gca;
set(ax,'XTick', 1:length(varyingParamsNames(startWith-1:end)), 'YTick', 1:length(Dataset_FA.dataOrder(Dataset_FA.modesIdxs(1:nModes))));
ax.TickLabelInterpreter = 'latex';
xticklabels(varyingParamsNames(startWith-1:end))
yticklabels(Dataset_FA.dataOrder(Dataset_FA.modesIdxs(1:nModes)));

%% 5) Check Caldersmith formulas performance vs MLR coefficients
% =========================================================================
% f11 <-> col 1 || f02 <-> col 3  || f20 <-> col 7
% setup
nTuples = length(Dataset_FA.inputs(:,1));
% take caldersmith modes and correspective MLR freqs
caldersmithIdxs = Dataset_FA.modesIdxs([1,2,4]);
caldersmithFreqs = Dataset_FA.outputsEig(:, caldersmithIdxs);
MLRIdxs = [1,2,4];
MLRFreqs = Dataset_FA.outputsEig(:, Dataset_FA.modesIdxs(MLRIdxs));

% take dataset input params
densities = Dataset_FA.inputs(:,1);
datasetParamsIdxs = [2,3,5]; % Ex, Ey, Gxy
datasetParams = Dataset_FA.inputs(:,datasetParamsIdxs);
normDataset = datasetParams./datasetParams(:,1);
poissonRatios = Dataset_FA.inputs(:,8);

% preallocate results 
caldersmithParams = zeros(nTuples,3);
normCaldersmith = zeros(nTuples,3);
MLRParams = zeros(nTuples,3);

for ii = 1:nTuples
% caldersmith
[mechParams, normParams] = computeParams(caldersmithFreqs(ii,:), densities(ii), geom);
normCaldersmith(ii,:) = normParams; 
caldersmithParams(ii,:) = mechParams; 
% MLR
[mechParamsMultilin] = computeParamsMultilin_plate(MLRFreqs(ii,:),densities(ii), coeffsMatr(MLRIdxs,:), 'coeffs');
MLRParams(ii,:) = mechParamsMultilin;
end

% specify variables for plots
nPlot =1:2500;
xlabels = {'E_L dataset' 'E_R dataset' 'G_{LR} dataset'}; ylabels = {'E_L caldersmith' 'E_R caldersmith' 'G_{LR} caldersmith' };
titles =  {'E_L scatter' 'E_R scatter' 'G_{LR} scatter'};
xlabelsRatio = { 'E_R/E_L dataset' 'G_{LR}/E_L dataset' }; ylabelsRatio = { 'E_R/E_L caldersmith' 'G_{LR}/G_L caldersmith' };
titlesRatio = { 'E_R/E_L scatter' 'G_{LR}/E_L scatter'};
testi = {'E_L' 'E_R' 'G_LR' };
% do plots
count = 0;

for ii = 1:3   
    tempCaldersmithParam = caldersmithParams(nPlot,ii);
    tempParam = datasetParams(nPlot,ii);
    tempMLRParam = MLRParams(nPlot, ii);
    
    % figure of parameters
    figure(1+ count)
    hold on;
    s = scatter( tempParam, tempCaldersmithParam,9, 'o', 'markerEdgeAlpha', .3); hold on;
    plot([min([tempParam; tempCaldersmithParam]),max([tempParam; tempCaldersmithParam])],...
         [min([tempParam; tempCaldersmithParam]),max([tempParam; tempCaldersmithParam])]);
    s = scatter( tempParam, tempMLRParam,9, 'o', 'markerEdgeAlpha', .3); hold on;

    hold off; xlabel(xlabels{ii}); ylabel(ylabels{ii}); title(titles{ii}); ax = gca; ax.FontSize = 14;
      
    R2 = computeR2(tempParam, tempCaldersmithParam);
    errNMSe = MSE(tempParam, tempCaldersmithParam);
    [errMAPE,varMAPE] = MAPE(tempParam, tempCaldersmithParam);
    disp(' ')
    disp([testi{ii},' Caldersmith: R2 = ', num2str(R2), '    NMSE = ', num2str(db(errNMSe)), '    MAPE = ', num2str(errMAPE) '%   var ' num2str(varMAPE)])
    R2 = computeR2(tempParam, tempMLRParam);
    errNMSe = MSE(tempParam, tempMLRParam);
    [errMAPE,varMAPE] = MAPE(tempParam, tempMLRParam);
    disp([testi{ii},' MLR        : R2 = ', num2str(R2), '    NMSE = ', num2str(db(errNMSe)), '    MAPE = ', num2str(errMAPE) '%   var ' num2str(varMAPE)])

    % figure for normalized params
    if ii >1
        figure(40)
        subplot(1,2,ii-1)
        hold on;
        scatter( normDataset(nPlot,ii), normCaldersmith(nPlot,ii), 9, 'filled'); hold on;
        plot([0,1],[0,1]);
        xlim([min(normDataset(nPlot,ii))*0.9, max(normDataset(nPlot,ii))*1.1]); hold off;
        xlabel(xlabelsRatio{ii-1}); ylabel(ylabelsRatio{ii-1}); title(titlesRatio{ii-1}); ax = gca; ax.FontSize = 14;
    end
    count = count+1;
end
