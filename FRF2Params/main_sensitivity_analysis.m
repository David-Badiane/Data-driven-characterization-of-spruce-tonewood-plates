%% ____________________ MAIN FRF2PARAMS ___________________________________
% THIS PROGRAM APPLIES FRF2PARAMS TO ESTIMATE MATERIAL PROPERTIES OF PLATES.
% In this program we have:
% A) Computation of the correlation between inputs and outputs of the
%    dataset
% b) Representation of the correlation data in two images, one for
% frequency and one for amplitude
% .........................................................................
% summary
%
% 0) init
% 1) correlation computation and image representation

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

% set variables

% parameters of the dataset
varyingParamsNames = {'$\rho$', '$E_x$', '$E_y$', '$E_z$', '$G_{xy}$', '$G_{yz}$', '$G_{xz}$', ...
                      '$\nu_{xy}$', '$\nu_{yz}$', '$\nu_{xz}$', '$\alpha$', '$\beta$'};   
geomNames = {'L' 'W' 'T'};
inputParamsNames = {varyingParamsNames{:} geomNames{:}};

%% sensitivity --> pearson correlation coefficient btw dataset inputs and outputs, both frequency and amplitude
% flags
saveData = 0;
saveFilename = 'percentage_with_damping';
plotPercentage = 0;
getOrdered = 1;   % 0 --> dataset ordered by peaks, 1 by modes 
nModes = 18;
% variables
imgN = 11;
startWith = 1;
nModes = 18;
xIdxs = [1:15]; 
xTickLabels = inputParamsNames(xIdxs);

% fetch dataset
[Dataset_FA, datasetPath, HP] = fetchDataset(baseFolder, nModes, getOrdered, datasetDir);

% compute correlations
corrEigs = corr( Dataset_FA.outputsEig, Dataset_FA.inputs(:,xIdxs), 'Rows', 'pairwise');
corrAmps = corr( db(Dataset_FA.outputsAmp), Dataset_FA.inputs(:,xIdxs), 'Rows', 'pairwise');

% xIndexes and y Indexes are mech params and eigenfrequencies
xIdxs = 1:length(xIdxs);
yIdxs = startWith:nModes;
% figure size
xLengthImg = 1200; yLengthImg = 0.45*xLengthImg;
% x y labels
xLabel = 'mech Params'; 
yLabel_f = 'eigenfrequencies'; 
yLabel_a = 'amplitudes';
% colormap
maxC = 1;
bluetogreen = [linspace(0, 0, 100).' linspace(0,maxC,100).' linspace(maxC, 0, 100).'];
greentoyellow = [linspace(0, maxC, 100).' linspace(maxC,maxC,100).' linspace(0, 0, 100).'];
yellowtored = [linspace(maxC, maxC, 100).' linspace(maxC,0,100).' linspace(0, 0, 100).'];
colorMap = [bluetogreen; greentoyellow; yellowtored];
% fontsizes
textFontSize = 16; axFontSize = 26;

for ii = 1:nModes
    if getOrdered
        yTickLabels_f{ii} = ['$f_{(', Dataset_FA.modesOrder{ii}(2),...
                                ',', Dataset_FA.modesOrder{ii}(3),')}$'];
        yTickLabels_a{ii} = ['$a_{(', Dataset_FA.modesOrder{ii}(2),...
                                ',', Dataset_FA.modesOrder{ii}(3),')}$'];
    else 
        yTickLabels_f{ii} = ['$f_{', num2str(ii) '}$'];
        yTickLabels_a{ii} = ['$a_{', num2str(ii) '}$'];
    end
end
minImg = min(abs(corrEigs(yIdxs, xIdxs)), [], 'all');
maxImg = max(abs(corrEigs(yIdxs, xIdxs)), [], 'all');
cbarLabel = '$|\mbox{correlation}|$';
displayCbar = true;

% generate and save images
% frequency
imgData_freq = defImg_matrix(xIdxs, yIdxs, xLengthImg, yLengthImg, imgN, xLabel, yLabel_f, colorMap,...
textFontSize, axFontSize, xTickLabels, yTickLabels_f, cbarLabel, displayCbar);
img = export_matrix(corrEigs, imgData_freq, 2, plotPercentage);
box off
if saveData, saveas(img, [saveFilename '_f.png']); end;

% amplitude
imgData_amp = defImg_matrix(xIdxs, yIdxs, xLengthImg, yLengthImg, imgN+1, xLabel, yLabel_a, colorMap,...
textFontSize, axFontSize, xTickLabels, yTickLabels_a, cbarLabel, displayCbar);
img = export_matrix(corrAmps,imgData_amp, 2, plotPercentage);
box off
if saveData, saveas(img, [saveFilename '_a.png']); end;

% writeMat2File(corrEigs, ['correlation_frequency.csv'], xTickLabels, length(xIdxs),1);
% writeMat2File(corrAmps, ['correlation_amplitude.csv'], xTickLabels, length(xIdxs),1);