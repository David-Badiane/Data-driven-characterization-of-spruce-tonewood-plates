%% _____________________ MAIN MODES ANALYSIS ______________________________
% PERFORMS MODES ANALYSIS ON A DATASET USING A SET OF REFERENCE
% MODE SHAPES
% pipeline:
% - A) resample the modeshapes of the dataset on a regular rectangular grid
% - B) compare the resampled modeshapes of the dataset with a reference set
%      of modeshapes
% - C) label modeshapes and order the dataset
% -------------------------------------------------------------------------
% SUMMARY:
% section 0)   Init - preset directories, set variables, upload data [...]
% section 1)   Modeshapes Resampling
% section 2)   Compute reference set
% section 3)   Reference set resampling
% section 4)   Modeshapes Labeling --> NCC
% section 4.1) Code to modify the reference set
% section 5)   Postprocessing and outliers removal
% section 5.1) Remove poisson plates
% section 6)   See obtained modeshapes
% section 7)   Define dataset modes order
% section 8)   Generate and save ordered dataset
% _________________________________________________________________________
% copyright: David Giuseppe Badiane
%% SECTION 0) Init
% =========================================================================
% Reference folders
remPath = false; % flag to remove paths, set to true if you want to remove all paths
% directories and paths 
if remPath
 cd(baseFolder)
 rmpath(genpath(baseFolder));
end

% folders, here you choose the dataset folder on which to perform modes analysis
baseFolder = pwd;
csvName = input('insert Dataset (csv_plate) directory name (string): ');
csvPath = [baseFolder, '\', csvName];

% creates directories saves paths
modesAnalysisPath = [csvPath, '\modesAnalysis'];
resampledPath = [csvPath,'\resampledModeshapes'];
mkdir(modesAnalysisPath); mkdir(resampledPath);
addpath(modesAnalysisPath);

% calls functions directory and data folder
idxs   = strfind(baseFolder,'\');
addpath(genpath([baseFolder(1:idxs(end)), 'functions']));
addpath ([baseFolder, '\data']);
addpath(csvPath);

% comsol Model
comsolModel = 'gPlate';

% set modeshapes grid variables
% number of x and y points of the rectuangular grid associated to the plate
pX = 35; % pX = 75;
pY = 15; % pY = 37;

% number of modes analysed for each tuple
nModes = 18;

% fetch the raw dataset (ordered by the ascending order of the eigenfrequencies - modes switching)
[Dataset_FA, csvPath, HPPath] = ...
fetchDataset(baseFolder, nModes, 0, csvName, 0);

nTuples = length(Dataset_FA.inputs(:,1));
varyingParamsNames = {'rho', 'Ex', 'Ey', 'Ez', 'Gxy', 'Gyz', 'Gxz', 'vxy', 'vyz', 'vxz', 'alpha', 'beta'};   
mechParamsNames = {'\rho' 'E_x' 'E_y' 'E_z' 'G_{xy}' 'G_{yz}' 'G_{xz}' '\nu_{xy}' '\nu_{yz}' '\nu_{xz}' '\alpha' '\beta'};

%% SECTION 1) Resample modeshapes
% =========================================================================

resampleShapes = 1; % boolean, to decide wheter to resample modeshapes (takes 2 minutes)
plotData = 0;       % boolean, to plot figures of each resampled modeshape
    
if resampleShapes
    %setup
    addpath([csvPath, '\Modeshapes']);
    % get the number of files that is in the directory modeshapes
    fileList =cellstr(ls([csvPath, '\Modeshapes']));
    fileList = fileList(3:end);
   
    nTuples = length(fileList);
    % call function to resample modeshapes
    modeshapes_resampling(pX,pY,nModes, nTuples,...
        resampledPath, plotData)
    % remove old modeshapes directory from path
    rmpath([csvPath, '\Modeshapes']);
end
cd(csvPath);

%% Section 2) - OBTAIN AND NAME REFERENCE SHAPES 
% =========================================================================

genRefShapes = input('generate reference Modeshapes? (0/1) : ');
nModesRef = 22;

if genRefShapes
    % setup
    cd(baseFolder);
    referenceVals = readmatrix('referenceVals.csv');
    refValsNames = readtable('referenceVals.csv').Properties.VariableNames;
    model = mphopen(comsolModel);
    % set reference params
    modeshapes_compute_reference_set(model, nModesRef, csvPath, referenceVals, refValsNames);
end


%% Section 3) REFERENCE SHAPES RESAMPLING
% =========================================================================
    nModesRef = 22;
    c = jet; % colormap
    
    %a) equally spaced grid x y creation
    ref_raw = abs(readmatrix('refModeshapes_raw.csv'));
    % min max normalization
    ref_raw(:,1) = (ref_raw(:,1)-min(ref_raw(:,1)))./ (max(ref_raw(:,1)) -min(ref_raw(:,1)));
    ref_raw(:,2) = (ref_raw(:,2)-min(ref_raw(:,2)))./ (max(ref_raw(:,2)) -min(ref_raw(:,2)));
    % take minima and maxima
    xMin = min(ref_raw(:,1));    xMax = max(ref_raw(:,1));
    yMin = min(ref_raw(:,2));    yMax = max(ref_raw(:,2));
    % create rectangular grid
    [X,Y] = meshgrid(linspace(xMin, xMax, pX), linspace(yMin, yMax, pY));
    x = X(:); y = Y(:);
    % unroll the grid into an array
    ref = [x,y]; % REF IS THE REFERENCE SET ALL ALONG

    % 2) interpolate to obtain z grid
    for jj = 1:nModesRef
        F1 = scatteredInterpolant(ref_raw(:,1),ref_raw(:,2), ref_raw(:,3+jj), 'natural', 'nearest');
        Z = F1(X,Y);
        z = Z(:);
        ref = [ref, z]; % add modeshape to reference set to ref 
    end
    
    % 3) order the reference modeshapes file and save it
    ref = sortrows(ref);
    
    cd(modesAnalysisPath);
    writeMat2File(ref(:,3:end), ['reference.csv'], {'f'}, 1,false);
    cd(csvPath);
    
    % 4) name the modes by hand
    modesData = readmatrix('reference.csv');
    refModesNames = {};
    for ii = 1:(length(modesData(1,:)))
        % show mode
        modeN = ii+2; % avoid x,y
        modeSurf = reshape(modesData(:,modeN), [pY, pX]);
        figure(100+ii); %clf reset;
        surf(X,Y,modeSurf)
        colormap([jet;])
        shading interp
        view(0.5,0.5)
        % name it 
        refModesNames{ii} = ['f',input('mode: ')];
    end
    % save labels of the reference set 
    cd(modesAnalysisPath);
    writeMat2File(refModesNames, 'refModesNames.csv', {'f'}, 1,false); 
    cd(csvPath);

%% section 4.1) SEE AND SAVE REF SHAPES 
% =========================================================================
seeReferenceModes(modesAnalysisPath,pX, pY,5, 6); % 5 and 6 are the nRows and nCols of the subplot in the figure

%%  section 5) COMPUTATION LOOP - compute NCC with ref modeshapes
%- associate to each mode the best scoring reference mode
% =========================================================================
% set variables
addpath(resampledPath);
ref = readmatrix('reference.csv');
refModesNames = table2cell(readtable('refModesNames.csv'));

% set booleans
plotFigures = 0;
printData   = 0;
cd(modesAnalysisPath)

% call function for the labeling loop
[modesNames, modesNCC] = modeshapes_labeling(pX, pY,pX,pY, nModes, nTuples, plotFigures, ...
                                 abs(ref(:,1:end)), refModesNames(1:end), 'disp', printData);

% save results
cd(modesAnalysisPath);
writeMat2File(modesNames, 'modesNames.csv', {'f'}, 1,false);
writeMat2File(modesNCC, 'modesNCC.csv', {'f'}, 1,false);
cd(csvPath);

%% 5.1) Code to add a column to ref and name to refModes Names 
% =========================================================================
% follow the instructions if you need to modify the reference shapes !
% before running section 5, if you want to add modes to the reference:

% setup & explanation
% - put breakpoints at   outliers(ii) = 1    in section 7 (3 breakpoints);
% - set plotFigures and printData to true -> see not recognised cases;
% - set thresholds for ncc;
% - run the section 5;
% - you'll see figs of not recognised tuples,
% - the variable m = modeshapes of the ii-th dataset tuple, 
% WITH THE CODE BELOW YOU CAN
% - highlight the code you need (add column, substitute column etc,)
%   set variables colTake, colIn, colDel
% - press F9 to execute the code highlighted

skip = true;
if ~skip
    % - to add a column at the end of ref
    colTake = 14+2;
    nameAddEnd = 'f1t';
    ref = [ref m(:, colTake)];
    refModesNames{length(refModesNames) +1} = nameAddEnd;

    % - to add a column inside ref, to preserve the order of the modeshapes
    colIn = 6;
    colTake =6+2;
    nameAddIn = '21';
    ref = [ref(:,1:colIn-1) m(:, colTake) ref(:,colIn:end) ];
    refModesNames = {refModesNames{1:colIn-1} nameAddIn refModesNames{colIn:end}};

    % - to delete a column of ref
    colDel = 31;
    ref(:,colDel) = [];
    refModesNames(colDel) = [];
    
    % - to change a mode in ref
    colChange =34;
    colTake = 16+2;
    name = 'f14';
    ref(:,colChange) = m(:, colTake);
    refModesNames{colChange} = name;

    % to save the reference and the names after modifications
    saveRef(ref, refModesNames, modesAnalysisPath)
end

%% 6) Postprocessing and outliers removal - by metric NCC 
% =========================================================================
% analyse data - discard tuples with repeated modes and tuples that have
% one modeshape with NCC < threshold
plotFigures = 0; 
printData = 0;

modesNCC = readmatrix('modesNCC.csv');
modesNames = table2cell(readtable('modesNames.csv'));
simFolder = [csvPath, '\resampledModeshapes'];
addpath(simFolder);

outliers = zeros(size(modesNames(:,1)));
nccThreshold = 0.9;

repetition_tags = {'_a' '_b' '_c'};
c = [jet;]; % colormap
repetitions = zeros(size(modesNames,1));
low_ncc = zeros(size(modesNames,1));
t = tic;

for ii = 1:length(modesNames(:,1))
    repetition = length(unique(modesNames(ii,:))) < length(modesNames(ii,:));
    %check if there are repeated modes
    if repetition
        % print message
        if printData, disp([newline, 'repetition at tuple ', num2str(ii) ]); end
        % open a figure
        if plotFigures, figure(77); clf reset; end
        % read modeshapes file
        m = readmatrix(['modeshapes',int2str(ii),'.csv']);
        
        repMsgs = {};
        count = 1;
        
        for jj = 1:length(modesNames(1,:))
            % chek how many times they are repeated and what
            checkRepetitions = ismember(modesNames(ii,:), modesNames{ii,jj});
            nRepetitions = length(checkRepetitions(checkRepetitions == 1));
            
            % display a message
            if  nRepetitions > 1
                repMsgs{count} = [modesNames{ii,jj}, '  repeated ', num2str(nRepetitions), ' times'];
                
                % store only first repetition
                if length(find(ismember(repMsgs, repMsgs{count}))) <= 1                   
                    if printData % show messafe
                        disp(repMsgs{count}); 
                        idxs = find( ismember(modesNames(ii,:), modesNames{ii,jj})); 
                        disp([' at idxs ' num2str(find(checkRepetitions == 1))])
                        disp([' ncc = ', num2str(modesNCC(ii,idxs(1))), ' ', num2str(modesNCC(ii,idxs(2))), ' ']); 
                    end
                end
                count = count + 1;
            end
            
            if plotFigures
                figure(77)
                subplot(5,4,jj)
                imagesc(reshape(m(:,jj+2), [pY,pX]));
                title(modesNamesCorrect{jj});
                colormap(c);
            end
        end
        outliers(ii) = 1;
        repetitions(ii) = 1;
    end
    
    
 %  check by NCC
    nccCheck = find(modesNCC(ii,:) < nccThreshold);
    if length(nccCheck) >= 1
        
        if plotFigures
            m = readmatrix(['modeshapes',int2str(ii),'.csv']);            
            figure(77)
            for jj = 1:length(modesNames(1,:))
            subplot(5,4,jj)
            imagesc(reshape(m(:,jj+2), [pY,pX]));
            title(modesNames{ii,jj});
            colormap(c);
            end
            pause(0.00001);
            
        end
        
        if printData 
            disp(['tuple ', num2str(ii) ]);
            disp(['NCC less than threshold = ', num2str(nccThreshold), 'for ' modesNames{ii,nccCheck}])
            disp(num2str(modesNCC(ii, nccCheck)))
        end
        outliers(ii) = 1;
        low_ncc(ii) = 1;
    end
    
    if mod(ii,150) == 0
        disp(['tuple ' num2str(ii) ' time elapsed ', num2str(toc(t),3) ' seconds'])
    end
end

% analyse data
total_Outliers = length(outliers(outliers == 1));
ncc_Outliers = length(low_ncc(low_ncc == 1 & repetitions == 0));
repetition_Outliers = length(repetitions(repetitions==1));

disp(['repetitions outliers ', num2str(repetition_Outliers), newline,...
      'NCC outliers ', num2str(ncc_Outliers), newline,...
      'n° Outliers ', num2str(total_Outliers)])
  
nonOutliers = find(outliers == 0);
outlIdxs = setdiff(1:nTuples, nonOutliers);
size(nonOutliers);

%% Section 6.1) Outliers removal - LOOK FOR POISSON PLATES (if present)
% =========================================================================
simFolder = [csvPath, '\resampledModeshapes'];

aspectRatio =Dataset_FA.inputs(nonOutliers,13)./Dataset_FA.inputs(nonOutliers,14);
delta =1; % percentage of the span in which we look for poisson plates with poissonCheck
plotPlates = 1;
deletePlates = 0;
% do the search 
poissonCheck = (Dataset_FA.inputs(nonOutliers,2)./Dataset_FA.inputs(nonOutliers,3)).^(1/4);
poissonPlates = find( poissonCheck >= .01*(100-delta)*aspectRatio & poissonCheck<=.01*(100+delta)*aspectRatio);
disp(' ');
disp(['N poisson plates :' int2str(length(poissonPlates))]);
disp(['at tuples : [', int2str(nonOutliers(poissonPlates).'),']']);

maxC = 0.95; % maxcolor
bluetogreen = [linspace(0, 0, 100).' linspace(0,maxC,100).' linspace(maxC, 0, 100).'];
greentoyellow = [linspace(0, maxC, 100).' linspace(maxC,maxC,100).' linspace(0, 0, 100).'];
yellowtored = [linspace(maxC, maxC, 100).' linspace(maxC,0,100).' linspace(0, 0, 100).'];
colorMap = [bluetogreen; greentoyellow; yellowtored];

cd(simFolder);
for jj = 1:length(poissonPlates)
    if poissonPlates(jj) >=1
        % plot poisson plates
        if plotPlates
            for ii = 1:3
            numTuple = nonOutliers(poissonPlates(jj));
            modesData = sortrows(readmatrix(['modeshapes',int2str(numTuple),'.csv']));
            figure(142)
            subplot (1,3,ii)
            normM = abs(modesData(:,ii+2)) / max(abs(modesData(:,ii+3)));
            X = reshape(modesData(:,1), [pY,pX]);
            Y = reshape(modesData(:,2), pY,pX);
            surfMode = reshape(modesData(:, ii + 2), pY, pX);
            surf(X,Y, surfMode);  colormap(colorMap);
            shading interp; view(20,82.5);
            set(gca, 'visible', 'off')
%             title(modesNames{nonOutliers(poissonPlates(jj)), ii})
            end
            pause(0.0001);
        end      
    end
end

% delete remaining poisson plates
if deletePlates, nonOutliers(poissonPlates) = []; end

%% 7) SEE obtained modeshapes
% =========================================================================
if plotFigures
    tuples2See = 1:nTuples
    for jj = tuples2See
        nTuple = nonOutliers(jj);
        modesData = readmatrix(['modeshapes', int2str(nTuple), '.csv']);
        for ii = 1:(length(modesData(1,:)) -2)
        modeN = ii+2;
        modeSurf = reshape(modesData(:,modeN), [pY, pX]);
        figure(100)
        subplot(4,5,ii)
        imagesc(modeSurf)
        title(modesNames{nonOutliers(jj), ii})
        end
        sgtitle(['modeshapes tuple ', int2str(jj)])
    end
end

%% section 8) Modes order - Decide the order of the modes in the ordered dataset 
% =========================================================================
% Define and obtain the modes order of the obtained dataset
findRefOrder = 1; % set to 1 if you need to define the mode order now
% otherwise the program will read the dataset mode order from a file

if findRefOrder
    modesFilename = 'modesNames.csv';
    [tupleAppears,maxLoc] = modesOrderAnalysis(nModes, [csvPath,'\ModesAnalysis'], modesFilename, nonOutliers);
    % prepare names (duplicate for more accuracy)
    reVarNames = {};
    addStr = {'_a' '_b' '_c' '_d' '_e' '_f' '_g' };
    for ii = 1:length(refModesNames)
        repetitions = find(ismember(refModesNames, refModesNames(ii)));
        nRepetition = find(repetitions == ii);
        reVarNames{ii} = [refModesNames{ii}, addStr{nRepetition}];
    end
    
    % check presence tuple by tuple (tuple of nModes)
    modesOrder = modesNames(maxLoc,:);
    % check presence mode by mode
    modeAppears = zeros(size(refModesNames));
    for ii = 1:length(refModesNames)
        modeAppears(ii) = length( modesNames(...
            ismember(modesNames(nonOutliers,:), refModesNames{ii}) ...
            ) );  
    end
    
    array2table(modeAppears, 'variablenames', reVarNames)

    % add the rest of the mod{}es in order of appearance
    modes2Add = {};
    for ii = 1:length(modeAppears)
        % the  is not present
        if isempty( find(ismember(modesOrder, refModesNames{ii})) )
            modes2Add = {modes2Add{:} {refModesNames{ii}, modeAppears(ii)}};
        end
    end
    cell2table(modes2Add)

    addedModes = input('input the modes in descending order in a (cell array): ');
    dataset_modesOrder = {modesOrder{:} addedModes{:}};
    writeMat2File(dataset_modesOrder, 'refModesOrder.csv', {'mode '}, 1 , false)

else
    % fetch dataset modes Order
    dataset_modesOrder = table2cell(readtable('refModesOrder.csv'))
end

%% section 9) Generate and save ordered dataset
% =========================================================================
% flags
ampOnGrid = false;
saveData = false;
% variables
nFrfs = 1;
modesNames = table2cell(readtable('modesNames.csv'));
outputsEig  = [];
outputsAmp  = [];

for jj = 1:length(dataset_modesOrder)
    modeFreqs = nan*ones(size(nonOutliers));
    modeAmps = nan*ones(size(nonOutliers));
    nPeaks = nan*ones(size(nonOutliers));

    for ii = 1:length(nonOutliers)
        nPeak = find(ismember(modesNames(nonOutliers(ii),:), dataset_modesOrder{jj}));
        if ~isempty(nPeak)
            nPeaks(ii) = nPeak;
            modeFreqs(ii) =  Dataset_FA.outputsEig(nonOutliers(ii),nPeak);
            modeAmps(ii) = abs(Dataset_FA.outputsAmp(nonOutliers(ii),nPeak));
        end
    end
    outputsEig = [outputsEig, modeFreqs, nPeaks];
    outputsAmp = [outputsAmp, modeAmps, nPeaks];
end

if saveData
        cd(baseFolder)
        refValsNames = readtable('referenceVals.csv').Properties.VariableNames;
    cd(csvPath)

    variableNamesEig = {};
    variableNamesAmp = {};

    countEig = 1;
    countAmp = 1;

    frfs = {};

    for kk = 1:nFrfs
        frfs{kk} = ['pt' num2str(kk)];
        for ii = 1:length(dataset_modesOrder)
                if kk == 1
                    variableNamesEig{countEig} = dataset_modesOrder{ii};
                    countEig = countEig+1;
                    variableNamesEig{countEig} = [dataset_modesOrder{ii},' peak N'];
                    countEig = countEig+1;
                end

                variableNamesAmp{countAmp} = [frfs{kk} ' ' dataset_modesOrder{ii}];
                countAmp = countAmp+1;
                variableNamesAmp{countAmp} = [frfs{kk} ' ' dataset_modesOrder{ii},' peak N'];
                countAmp = countAmp+1;
        end
    end


    writeMat2File(outputsEig, 'datasetOrdered_Eig.csv', variableNamesEig,...
        length(variableNamesEig), true);

    writeMat2File(outputsAmp, 'datasetOrdered_Amp.csv', variableNamesAmp,...
        length(variableNamesAmp), true);

    writeMat2File(Dataset_FA.inputs(nonOutliers,:), 'datasetOrdered_Inputs.csv',...
        refValsNames, length(refValsNames), true);

    modesIdxs = [];
    peaksIdxs  = [];
    count = 0;
    nMod = length(dataset_modesOrder);
    for kk = 1:nFrfs
        modesIdxs = [modesIdxs; count*(2*nMod)+1:2:(count+1)*(2*nMod)];    
        count = count+1;
    end
    peaksIdxs = modesIdxs+1;

    T = array2table(modesIdxs, 'variableNames', dataset_modesOrder, 'rowNames', frfs)
    writetable(T, 'datasetOrdered_modesIdxs.csv');

    T = array2table(peaksIdxs, 'variableNames', dataset_modesOrder, 'rowNames', frfs);
    writetable(T, 'datasetOrdered_peaksIdxs.csv');

    disp( ['END - Saved dataset of ', csvName])
end