%% 0 init 
% to remove all previous paths
remPath = 0; % flag to remove paths, set to true if you want to remove all paths
if remPath
 cd(baseFolder)
 rmpath(genpath(baseFolder));
end

% directories names
baseFolder = pwd;                                          % base working directory is current directory 
datasetDir = 'csv_gPlates';                               % directory of the dataset
datasetPath = [baseFolder, '\', datasetDir];               % path of the dataset directory
idxs   = strfind(baseFolder,'\');                          % find the fxs path, it is in the directory containing the baseFolder
% add all paths 
addpath(genpath([baseFolder(1:idxs(end)), 'functions']));
addpath(datasetPath);
getOrdered = 0; 
nModes = 18;

% parameters of the dataset
varyingParamsNames = {'rho', 'Ex', 'Ey', 'Ez', 'Gxy', 'Gyz', 'Gxz', 'vxy', 'vyz', 'vxz', 'alpha', 'beta'};   
geomNames = {'L' 'W' 'T'};
inputParamsNames = {varyingParamsNames{:} geomNames{:}};
[Dataset_FA, datasetPath, HP] = fetchDataset(baseFolder, nModes, getOrdered, datasetDir);

%% sensitivity 
saveData = 0;
saveFilename = 'percentage_with_damping';
plotPercentage = 1;
imgN = 11;
startWith = 1;
nModes = 18;
xIdxs = [1:15]; 
xTickLabels = inputParamsNames(xIdxs);
 

corrEigs = corr( Dataset_FA.outputsEig, Dataset_FA.inputs(:,xIdxs), 'Rows', 'pairwise');
corrAmps = corr( db(Dataset_FA.outputsAmp), Dataset_FA.inputs(:,xIdxs), 'Rows', 'pairwise');
xIdxs = 1:length(xIdxs);
yIdxs = startWith:nModes;
xLengthImg = 1500; yLengthImg = 0.45*xLengthImg;
xLabel = 'mech Params'; yLabel_f = 'eigenfrequencies'; yLabel_a = 'amplitudes';
maxC = 1;
bluetogreen = [linspace(0, 0, 100).' linspace(0,maxC,100).' linspace(maxC, 0, 100).'];
greentoyellow = [linspace(0, maxC, 100).' linspace(maxC,maxC,100).' linspace(0, 0, 100).'];
yellowtored = [linspace(maxC, maxC, 100).' linspace(maxC,0,100).' linspace(0, 0, 100).'];
colorMap = [bluetogreen; greentoyellow; yellowtored];
textFontSize = 18; axFontSize = 28;
for ii = 1:nModes
yTickLabels_f{ii} = ['$f_{', num2str(ii) '}$'];
yTickLabels_a{ii} = ['$a_{', num2str(ii) '}$'];
end
minImg = min(abs(corrEigs(yIdxs, xIdxs)), [], 'all');
maxImg = max(abs(corrEigs(yIdxs, xIdxs)), [], 'all');
cbarLabel = '$|\mbox{correlation}|$';
displayCbar = true;
imgData_freq = defImg_matrix(xIdxs, yIdxs, xLengthImg, yLengthImg, imgN, xLabel, yLabel_f, colorMap,...
textFontSize, axFontSize, xTickLabels, yTickLabels_f, cbarLabel, displayCbar);
img = export_matrix(corrEigs,imgData_freq, 2, plotPercentage);
box off
if saveData, saveas(img, [saveFilename '_f.png']); end;

imgData_amp = defImg_matrix(xIdxs, yIdxs, xLengthImg, yLengthImg, imgN+1, xLabel, yLabel_a, colorMap,...
textFontSize, axFontSize, xTickLabels, yTickLabels_a, cbarLabel, displayCbar);
img = export_matrix(corrAmps,imgData_amp, 2, plotPercentage);
box off
if saveData, saveas(img, [saveFilename '_a.png']); end;

% writeMat2File(corrEigs, ['correlation_frequency.csv'], xTickLabels, length(xIdxs),1);
% writeMat2File(corrAmps, ['correlation_amplitude.csv'], xTickLabels, length(xIdxs),1);