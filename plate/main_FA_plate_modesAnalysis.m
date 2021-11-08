

%% 0) Initial setup
% =========================================================================
% Reference folders
% cd(baseFolder);
% rmpath(genpath(baseFolder));
% clear all 
% close all
baseFolder = pwd;
csvName = input('insert Dataset (csv_plate) directory name (string): ');
csvPath = [baseFolder, '\', csvName];
modesAnalysisPath = [csvPath, '\modesAnalysis'];
resampledPath = [csvPath,'\resampledModeshapes'];
mkdir(modesAnalysisPath); mkdir(resampledPath);
addpath(modesAnalysisPath);
idxs   = strfind(baseFolder,'\');
addpath(genpath([baseFolder(1:idxs(end)), 'functions']));
addpath ([baseFolder, '\data']);
addpath(csvPath);


% comsol Model
comsolModel = 'woodenPlate';
% geometry infos
infosTable = readtable("sampleMeasurements.xlsx");
infosMatrix = table2array(infosTable(:,3:end));
infosMatrix(:,1:2) = infosMatrix(:,1:2)*0.01;
Ls      =   infosMatrix(:,1);
Ws      =   infosMatrix(:,2);

% number of points of the grid we will generate
pX = 75;
pY = 37;
% number of modes analysed for each tuple
nModes = 15;

mechParamsNames = {'\rho' 'E_x' 'E_y' 'E_z' 'G_{xy}' 'G_{yz}' 'G_{xz}' '\nu_{xy}' '\nu_{yz}' '\nu_{xz}' '\alpha' '\beta'};
Dataset_FA = struct('inputs',[] ,'outputsEig',[] ,'outputsAmp',[] );
Dataset_FA.inputs = readmatrix('inputs.csv');
Dataset_FA.outputsEig = readmatrix('outputsEig.csv');
Dataset_FA.outputsAmp = readmatrix('outputsAmp.csv');
nTuples = length(Dataset_FA.inputs(:,1));

%% 1.b) Resample all modeshapes
% =========================================================================

resampleShapes = input('resample all modeshapes? (0/1) : ');

if resampleShapes
    %setup
    addpath([csvPath, '\Modeshapes']);
    fileList =cellstr(ls([csvPath, '\Modeshapes']));
    fileList = fileList(3:end);
    nTuples = length(fileList);
    plotData = false;
    % call function
    resampleMshapes(pX,pY,nModes, nTuples,...
        resampledPath, plotData)
    % cleanup
    rmpath([csvPath, '\Modeshapes']);
end
cd(csvPath);

%% 2) - OBTAIN AND NAME REFERENCE SHAPES 
% =========================================================================

genRefShapes = input('generate reference Modeshapes? (0/1) : ');

if genRefShapes
    % setup
    cd(csvPath);
    referenceVals = readmatrix('referenceVals.csv');
    nModesRef = 25;
    model = mphopen(comsolModel);
    % set reference params
    for jj = 1:length(referenceVals)
        model.param.set(varyingParamsNames(jj), referenceVals(jj));
    end
    obtainReferenceModes(model, nModesRef);
    % name the modes
    modesData = readmatrix('reference.csv');
    refModesNames = {};
    for ii = 1:(length(modesData(1,:))-2)
        modeN = ii+2; % avoid x,y
        modeSurf = reshape(modesData(:,modeN), [pY, pX]);
        figure(100)
        subplot(6,5,ii)
        imagesc(modeSurf)
        refModesNames{ii} = ['f',input('mode: ')];
    end
    cd(modesAnalysisPath);
    writeMat2File(refModesNames, 'refModesNames.csv', {'f'}, 1,false); 
    cd(csvPath);


%% 3) REFERENCE SHAPES RESAMPLING
% =========================================================================
    nModesRef = 25;

    %a) equally spaced grid x y creation
    ref_raw = readmatrix('reference_raw.csv');
    xMin = min(ref_raw(:,1));    xMax = max(ref_raw(:,1));
    yMin = min(ref_raw(:,2));    yMax = max(ref_raw(:,2));
    [X,Y] = meshgrid(linspace(xMin, xMax, pX), linspace(yMin, yMax, pY));
    x = X(:); y = Y(:);
    ref = [x,y];

    % interpolate to obtain z grid
    for jj = 1:nModesRef
        F1 = scatteredInterpolant(ref_raw(:,1),ref_raw(:,2), ref_raw(:,3+jj), 'natural');
        Z = F1(X,Y);
        z = Z(:);
        ref = [ref, z];
    end
    % order the file and save it
    ref = sortrows(ref);
    cd(modesAnalysisPath);
    writeMat2File(ref, ['reference.csv'], {'f'}, 1,false);
    cd(csvPath);

%% 4) SEE AND SAVE REF SHAPES 
% =========================================================================
     close all;
     ref = readmatrix('reference.csv');
     refModesNames = table2cell(readtable('refModesNames.csv'));

    % preallocate
    refShapes_Fourier = {};
    refShapes_Disp = {};
    % display and save ref shapes
    for ii = 1:(length(ref(1,:)))
        modeSurf = reshape(ref(:,ii), [pY, pX]);
        refShapes_Fourier{ii} = fftshift(abs(fft2(modeSurf, 200,200)));
        refShapes_Disp{ii} = modeSurf;
        % plot disp
        figure(1256)
        subplot(9,5,ii)
        imagesc(modeSurf); title(refModesNames{ii});
    end
end

%% 5) COMPUTATION LOOP - compute NCC and NMSE with ref modeshapes
%- associate to each one the best scoring one
% =========================================================================
% set variables
addpath(resampledPath);
plotFigures = false;
printData = false;
 ref = readmatrix('reference.csv');
 refModesNames = table2cell(readtable('refModesNames.csv'));

% call function doing computation loop
[modesNames, modesNCC, modesNMSE] = modeshapesLabeling(pX, pY,128,64, nModes, nTuples, plotFigures, ...
                                 ref, refModesNames, 'disp', printData);
% save results
cd(modesAnalysisPath);
writeMat2File(modesNames, 'modesNames.csv', {'f'}, 1,false);
writeMat2File(modesNCC, 'modesNCC.csv', {'f'}, 1,false);
writeMat2File(modesNMSE, 'modesNMSE.csv', {'f'}, 1,false);
cd(csvPath);

%% 6) Code to add a column to ref and name to refModes Names 
% =========================================================================
% follow the instructions if you need to modify the reference shapes !
% before running section 7, if you want to add modes to the reference:

% setup & explanation
% - put breakpoints at   outliers(ii) = 1    in section 7 (3 breakpoints);
% - set plotFigures and printData to true -> see not recognised cases;
% - set thresholds for ncc and nmse;
% - run the section 7;
% - you'll see figs of not recognised tuples,
% - Mshapes of the tuple seen --> variable m;
% to do
% - then:
% - write the entries you need here
% - highlight the code you need (add column, substitute column etc,)
% - press F9 to execute the code highlighted

skip = true;
if ~skip
    % - to add a column at the end of ref
    colTake = 22;
    nameAddEnd = 'f51';
    ref = [ref m(:, colTake)];
    refModesNames{length(refModesNames) +1} = nameAddEnd;

    % - to add a column inside ref, to preserve the order of the modeshapes
    colIn = 27;
    colTake = 19;
    nameAddIn = 'f33';
    ref = [ref(:,1:colIn-1) m(:, colTake) ref(:,colIn:end) ];
    refModesNames = {refModesNames{1:colIn-1} nameAddIn refModesNames{colIn:end}};

    % - to delete a column of ref
    colDel = 37;
    ref(:,colDel) = [];
    refModesNames(colDel) = [];

    % to save the reference and the names after modifications
    cd(modesAnalysisPath)
    reVarNames = {};
    addStr = {'_a' '_b' '_c' '_d' '_e' '_f' };
    for ii = 1:length(refModesNames)
    repetitions = find(ismember(refModesNames, refModesNames(ii)));
    nRepetition = find(repetitions == ii);
    reVarNames{ii} = [refModesNames{ii}, addStr{nRepetition}];
    end
    writeMat2File(ref, ['reference.csv'], reVarNames, length(reVarNames), true);
    writeMat2File(refModesNames, 'refModesNames.csv', {'name '}, 1,false);
end

%% 7) MODESHAPES ANALYSIS - by metrics
% =========================================================================

modesNMSE = readmatrix('modesNMSE.csv');
modesNCC = readmatrix('modesNCC.csv');
modesNames = table2cell(readtable('modesNames.csv'));
simFolder = [csvPath, '\resampledModeshapes'];
addpath(simFolder);


pX = 75;
pY = 37; 
outliers = zeros(size(modesNames(:,1)));
% false - false allows to see fast how many outliers you have eith the set
% of modes
plotFigures = input('plot Figures (0/1):'); 
printData = input('print data while running ? (0/1):');
repeatVec = {};
nmseThreshold = 0.8;
nccThreshold = 0.83;

for ii = 1:length(modesNames(:,1))
    %check if there are repeated modes
    if length(unique(modesNames(ii,:))) < length(modesNames(ii,:))
        m = readmatrix(['modeshapes',int2str(ii),'.csv']);
        repMsgs = {};
        count = 1;
        if printData
            disp( ' ' );
            disp(['tuple ', num2str(ii) ]);
        end
        for jj = 1:length(modesNames(1,:))
            if plotFigures
                figure(77)
                subplot(5,4,jj)
                imagesc(reshape(m(:,jj+2), [pY,pX]));
                title(modesNames{ii,jj});
            end

            % chek how many time they are repeated and what
            checkRepetitions = ismember(modesNames(ii,:), modesNames{ii,jj});
            nRepetitions = length(checkRepetitions(checkRepetitions == 1));
            if  nRepetitions > 1
                repMsgs{count} = [modesNames{ii,jj}, '  repeated ', num2str(nRepetitions), ' times'];
                % store only first repetition
                if length(find(ismember(repMsgs, repMsgs{count}))) <= 1                   
                    if printData, disp(repMsgs{count}); end;
                        if nRepetitions <= 2
                            repeatVec = {repeatVec{:} modesNames{ii,jj}};
                        end
                    end
                count = count + 1;
            end
        end
        outliers(ii) = 1;
    end
    
    % check by NMSE
    nmseCheck = find(modesNMSE(ii,:) > nmseThreshold);
    if length(nmseCheck) >= 2
        if plotFigures
            m = readmatrix(['modeshapes',int2str(ii),'.csv']);
            if length(unique(modesNames(ii,:))) == length(modesNames(ii,:))
            figure(77)
            for jj = 1:length(modesNames(1,:))
            subplot(4,5,jj)
            imagesc(reshape(m(:,jj+2), [pY,pX]));
            title(modesNames{ii,jj});
            end
            end
        end
        
        if printData 
            disp(['tuple ', num2str(ii) ]);
            disp(['NMSE more than threshold = ', num2str(nmseThreshold),' at indexes [', num2str(nccCheck), ']'])
            array2table(modesNMSE(ii, nmseCheck), 'variableNames', modesNames(ii, nmseCheck))
        end
        
        outliers(ii) = 1;
    end
    
 %  check by NCC
    nccCheck = find(modesNCC(ii,:) < nccThreshold);
    if length(nccCheck) >= 2
        if plotFigures
            m = readmatrix(['modeshapes',int2str(ii),'.csv']);
            
            if length(unique(modesNames(ii,:))) == length(modesNames(ii,:))
            figure(77)
            for jj = 1:length(modesNames(1,:))
            subplot(5,4,jj)
            imagesc(reshape(m(:,jj+2), [pY,pX]));
            title(modesNames{ii,jj});
            end
            pause(0.00001);
            end
        end
        if printData 
            disp(['tuple ', num2str(ii) ]);
            disp(['NCC less than threshold = ', num2str(nccThreshold),' at indexes [', num2str(nccCheck), ']'])
            array2table(modesNCC(ii, nccCheck), 'variableNames', modesNames(ii, nccCheck))
        end
        outliers(ii) = 1;
    end
end

nOutliers = length(outliers(outliers == 1));
disp(['n° Outliers + not recognized = ', num2str(nOutliers)])
nonOutliers = find(outliers == 0);
size(nonOutliers);

%% 7,1) LOOK FOR POISSON PLATES (if present)
% =========================================================================
simFolder = [csvPath, '\resampledModeshapes'];
L = Ls(1);
W = Ws(1);
aspectRatio = L/W;
delta = 1.5;
plotPlates = input('plot poisson plates? (0/1): ');
deletePlates = true;

poissonCheck = (Dataset_FA.inputs(nonOutliers,2)./Dataset_FA.inputs(nonOutliers,3)).^(1/4);
poissonPlates = intersect(find( poissonCheck > (100-delta)/100*aspectRatio),...
    find( poissonCheck<(100+delta)/100*aspectRatio));
disp(' ');
disp(['N poisson plates :' int2str(length(poissonPlates))]);
disp(['at tuples : [', int2str(poissonPlates.'),']']);


cd(simFolder);
for jj = 1:length(poissonPlates)
    if poissonPlates(jj) >=1
        % plot poisson plates
        if plotPlates
            for ii = 1:4
            numTuple = nonOutliers(poissonPlates(jj));
            modesData = sortrows(readmatrix(['modeshapes',int2str(numTuple),'.csv']));
            figure(142)
            subplot (2,2,ii)
            normM = modesData(:,ii+2) / max(modesData(:,ii+3));
            idx = find(normM<0.2);
            plot3(modesData(idx,1), modesData(idx,2), modesData(idx,ii+3) ,'.', 'MarkerSize', 5); view(2);
            title(modesNames{nonOutliers(poissonPlates(jj)), ii})
            end
            pause(0.0001);
        end      
    end
end
% delete remaining poisson plates
if deletePlates, nonOutliers(poissonPlates) = []; end;

%% 8) SEE obtained modeshapes
% =========================================================================
if plotFigures
    pX = 75; pY = 37;
    tuples2See = 1:length(nonOutliers);
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

%% 9) Modes Order
% =========================================================================
% Define and obtain the modes order of the obtained dataset
findRefOrder = false;
if findRefOrder
    modesFilename = 'modesNames.csv';
    [tupleAppears,maxLoc] = modesOrderAnalysis(nModes, [csvPath,'\ModesAnalysis'], modesFilename);

    % prepare names (duplicate for more accuracy)
    reVarNames = {};
    addStr = {'_a' '_b' '_c' '_d' '_e' '_f' };
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

    % add the rest of the modes in order of appearance
    modes2Add = {};
    for ii = 1:length(modeAppears)
        % the  is not present
        if isempty( find(ismember(modesOrder, refModesNames{ii})) )
            modes2Add = {modes2Add{:} {refModesNames{ii}, modeAppears(ii)}}
        end
    end
    cell2table(modes2Add)

    addedModes = input('input the modes in descending order in a (cell array): ');
    dataset_modesOrder = {modesOrder{:} addedModes{:}};

else
    % fetch dataset modes Order
    dataset_modesOrder = table2cell(readtable('refModesOrder.csv'));
end

%% 10) Generate ordered dataset + SEE HISTOGRAM DISTRIBUTION OF THE DATASET
% =========================================================================
%close all
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

nModesSee = 15;
startMode = 1;

peakNames = {};
nPoints = 25;
% [N,edges] = histcounts(X);
% [N,edges] = histcounts(X,40);
% edges = edges(2:end) - (edges(2)-edges(1))/2;
% plot(edges, N);
freqFig = 100;
ampFig = freqFig+1;
count = 1;
step = 1;
modesAxis = startMode:step:(startMode + nModesSee*step - 1);
lineStl = {'-' ,'--' ,':' ,'-.'};

for ii = modesAxis
    figure(freqFig)
    
    subplot 211
    hold on
    freqIdx = 1 + (ii-1)*2; % only odd idxs
    [N,edges] = histcounts(outputsEig(~isnan(outputsEig(:, freqIdx)),freqIdx),nPoints);
    edges = edges(2:end) - (edges(2)-edges(1))/2;
    N = lowpass(N,0.625);
    H=area(edges,N, 'LineStyle', lineStl{mod(count,4)+1}, 'lineWidth', 1);

    H.FaceAlpha = 0.3;
    hold off
   
    
    subplot 212
    hold on
    [N,edges] = histcounts(Dataset_FA.outputsEig(:,ii),nPoints);
    edges = edges(2:end) - (edges(2)-edges(1))/2;
    N = lowpass(N,0.625);
    H=area(edges,N,'LineStyle', lineStl{mod(count,4)+1}, 'lineWidth', 1);
    H.FaceAlpha = 0.3;
    hold off
    
    
    figure(ampFig)
    subplot 211
    hold on
    [N,edges] = histcounts(db(outputsAmp(~isnan(outputsEig(:, freqIdx)),freqIdx)),nPoints);
    edges = edges(2:end) - (edges(2)-edges(1))/2;
    N = lowpass(N,0.625);
    H=area(edges,N, 'LineStyle', lineStl{mod(count,4)+1}, 'lineWidth', 1);
    H.FaceAlpha = 0.3;
    hold off
    
    subplot 212
    hold on
    [N,edges] = histcounts(db(abs(Dataset_FA.outputsAmp(:,ii))),nPoints);
    edges = edges(2:end) - (edges(2)-edges(1))/2;
    N = lowpass(N,0.625);
    H=area(edges,N, 'LineStyle', lineStl{mod(count,4)+1}, 'lineWidth', 1);
    H.FaceAlpha = 0.3;
    peakNames{count} = ['peak', int2str(ii)];
    count = count+1;
    hold off
end

% subplot handler;
for ii = 1:2
    figure(freqFig)
    subplot(2,1,ii)
    xx = xlabel('$frequency$ $[Hz]$'); yy = ylabel('N occurences');
    set(xx, 'interpreter', 'latex'); set(yy, 'interpreter', 'latex');
    hold off; ax = gca; ax.FontSize = 15;
    set(gca,'ColorOrder', [cos(linspace(0,0.45*pi,nModesSee).'), abs(sin(linspace(0, 2*pi,nModesSee).')), sin((linspace(0 , pi/3,nModesSee).'))]);
    if ii == 1, legend(dataset_modesOrder{modesAxis});
    else , legend(peakNames); end
    hold off;
    
    figure(ampFig)
    subplot(2,1,ii)
    xx = xlabel('$|H(f_i)|_{dB}$'); yy = ylabel('N occurences');
    set(xx, 'interpreter', 'latex'); set(yy, 'interpreter', 'latex');
    hold off; ax = gca; ax.FontSize = 15;
    set(gca,'ColorOrder', [cos(linspace(0,0.45*pi,nModesSee).'), abs(sin(linspace(0, 2*pi,nModesSee).')), sin((linspace(0 , pi/3,nModesSee).'))]);
    if ii == 1, legend(dataset_modesOrder{startMode:end});
    else , legend(peakNames); end
    hold off;
end

%% 11) Save the obtained dataset ordered by modes
% =========================================================================
cd(csvPath)

variableNamesDataset = {};
count = 1;
for ii = 1:length(dataset_modesOrder)
        variableNamesDataset{count} = dataset_modesOrder{ii};
        count = count+1;
        variableNamesDataset{count} = [dataset_modesOrder{ii},' peak N'];
        count = count+1;
end

varyingParamsNames = {'rho', 'Ex', 'Ey', 'Ez', 'Gxy', 'Gyz', 'Gxz', 'vxy', 'vyz', 'vxz', 'alpha', 'beta'};   
writeMat2File(outputsEig, 'datasetOrdered_Eig.csv', variableNamesDataset,...
    length(variableNamesDataset), true);
writeMat2File(outputsAmp, 'datasetOrdered_Amp.csv', variableNamesDataset,...
    length(variableNamesDataset), true);

writeMat2File(Dataset_FA.inputs(nonOutliers,:), 'datasetOrdered_Inputs.csv',...
    varyingParamsNames, length(varyingParamsNames), true);

disp( ['END - Saved dataset of ', csvName])