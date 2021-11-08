%% 0) Initial setup
% =========================================================================

% Reference folders
% cd(baseFolder);
% rmpath(genpath(baseFolder));
% clear all 
% close all
% 
baseFolder = pwd;
csvName = 'csv_wedge_scan';
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
comsolModel = 'wedge_scan.mph';

% number of points of the grid we will generate
pX = 35;
pY = 15;
% number of modes analysed for each tuple
nModes = 15;
c = [jet; flip(jet)];

mechParamsNames = {'\rho' 'E_x' 'E_y' 'E_z' 'G_{xy}' 'G_{yz}' 'G_{xz}' '\nu_{xy}' '\nu_{yz}' '\nu_{xz}' '\alpha' '\beta'};
Dataset_FA = struct('inputs',[] ,'outputsEig',[] ,'outputsAmp',[] );
Dataset_FA.inputs = readmatrix('inputs.csv');
Dataset_FA.outputsEig = readmatrix('outputsEig.csv');
Dataset_FA.outputsAmp = readmatrix('outputsAmp.csv');
nTuples = length(Dataset_FA.inputs(:,1));

%% 1) RESAMPLE ALL SIMULATED MODESHAPES
% =========================================================================

resampleShapes = input('resample all modeshapes? (0/1) : ');

if resampleShapes
    %setup
    addpath([csvPath, '\Modeshapes']);
    fileList =cellstr(ls([csvPath, '\Modeshapes']));
    fileList = fileList(3:end);
    nTuples = length(fileList);
    plotData = 0;
    % call function
    resampleMshapes(pX,pY,nModes, nTuples,...
        resampledPath, plotData)
    % cleanup
    rmpath([csvPath, '\Modeshapes']);
end
cd(csvPath);

%% 2.a) - OBTAIN REFERENCE SHAPES 
% =========================================================================

genRefShapes = input('generate reference Modeshapes? (0/1) : ');

if genRefShapes
    % setup
    cd(baseFolder)
    i1 = strfind(comsolModel, '_'); i2 = strfind(comsolModel, '.'); 
    referenceVals = readmatrix(['referenceVals',comsolModel(i1:i2-1),'.csv']);
    model = mphopen(comsolModel);

    cd(csvPath);
    nModesRef = 15;
    % set reference params
    for jj = 1:length(referenceVals)
        model.param.set(varyingParamsNames(jj), referenceVals(jj));
    end
    obtainReferenceModes(model, nModesRef, modesAnalysisPath);
end
%% 2.b) REFERENCE SHAPES RESAMPLING
% =========================================================================
cd(modesAnalysisPath)
    %a) equally spaced grid x y creation
    ref_raw = readmatrix('refModeshapes_raw.csv');
    rangeX = max(ref_raw(:,1)) - min(ref_raw(:,1));
    rangeY = max(ref_raw(:,2)) - min(ref_raw(:,2));
    xMin = min(ref_raw(:,1))+0.01*rangeX;    xMax = 0.97*max(ref_raw(:,1));
    yMin = min(ref_raw(:,2))+0.01*rangeY;    yMax = 0.92*max(ref_raw(:,2));
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
    writeMat2File(ref(:,3:end), ['reference.csv'], {'f'}, 1,false);
    cd(csvPath);
%% 2.c) NAME THE REFERENCE MODES 
% =========================================================================
    cd(modesAnalysisPath)
    % name the modes
    modesData = readmatrix('reference.csv');
    refModesNames = {};

    modeSurfX = reshape(modesData(:,1), [pY, pX]);
    modeSurfY = reshape(modesData(:,2), [pY, pX]);
    
    for ii = 1:(length(modesData(1,:))-2)
        modeN = ii+2; % avoid x,y
        modeSurf = reshape(modesData(:,modeN), [pY, pX]);
        figure(100)
        subplot(6,5,ii)
        imagesc(modeSurf);
        colormap(c);

         figure(101)
        surf(modeSurfX, modeSurfY, modeSurf);
        colormap(c);
        refModesNames{ii} = ['f',input('mode: ')];
    end
    cd(modesAnalysisPath);
    writeMat2File(refModesNames, 'refModesNames.csv', {'f'}, 1,false); 
    cd(csvPath);

%% 2.d) SEE REF SHAPES 
% =========================================================================
seeReferenceModes(modesAnalysisPath,pX, pY,5, 6);

%% 3) COMPUTATION LOOP - compute NCC and NMSE with ref modeshapes
%- associate to each one the best scoring one
% =========================================================================
% set variables
addpath(resampledPath);
plotFigures = false;
printData = false;
cd(modesAnalysisPath);
 ref = readmatrix('reference.csv');
 refModesNames = table2cell(readtable('refModesNames.csv'));
nModes = 8;
% call function doing computation loop
[modesNames, modesNCC, modesNMSE] = modeshapesLabeling(pX, pY,pX,pY, nModes, nTuples, plotFigures, ...
                                 ref, refModesNames, 'disp', printData);
% save results
cd(modesAnalysisPath);
writeMat2File(modesNames, 'modesNames.csv', {'f'}, 1,false);
writeMat2File(modesNCC,  'modesNCC.csv',  {'f'}, 1,false);
writeMat2File(modesNMSE, 'modesNMSE.csv', {'f'}, 1,false);
cd(csvPath);

%% 4) Code to add a column to ref and name to refModes Names 
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
    colTake =2+22;
    nameAddEnd = 'f51';
    ref = [ref m(:, colTake)];
    refModesNames{length(refModesNames) +1} = nameAddEnd;

    % - to add a column inside ref, to preserve the order of the modeshapes
    colIn = 22;
    colTake = 2+16;
    nameAddIn = 'f04';
    ref = [ref(:,1:colIn-1) ref(:, colTake) ref(:,colIn:end) ];
    refModesNames = {refModesNames{1:colIn-1} nameAddIn refModesNames{colIn:end}};

    % - to delete a column of ref
    colDel =18;
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

seeReferenceModes(modesAnalysisPath,pX, pY,5, 6);

%% 5) MODESHAPES ANALYSIS - by metrics
% =========================================================================

modesNMSE = readmatrix('modesNMSE.csv');
modesNCC = readmatrix('modesNCC.csv');
modesNames = table2cell(readtable('modesNames.csv'));
simFolder = [csvPath, '\resampledModeshapes'];
addpath(simFolder);
outliers = zeros(size(modesNames(:,1)));
% false - false allows to see fast how many outliers you have eith the set
% of modes
plotFigures = input('plot Figures (0/1):'); 
printData = input('print data while running ? (0/1):');
repeatVec = {};
nmseThreshold = 0.8;
nccThreshold = 0.8;
tags = {'_a' '_b' '_c'};
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
                colormap(c);
            end

            % chek how many time they are repeated and what
            checkRepetitions = ismember(modesNames(ii,:), modesNames{ii,jj});
            nRepetitions = length(checkRepetitions(checkRepetitions == 1));
            if  nRepetitions > 1
                repMsgs{count} = [modesNames{ii,jj}, '  repeated ', num2str(nRepetitions), ' times'];
                % store only first repetition
                if length(find(ismember(repMsgs, repMsgs{count}))) <= 1                   
                    if printData
                        disp(repMsgs{count}); 
                        idxs = find( ismember(modesNames(ii,:), modesNames{ii,jj})); 
                        disp([' ncc = ', num2str(modesNCC(ii,idxs(1))), ' ', num2str(modesNCC(ii,idxs(2))), ' ']); 
                    end
                    if nRepetitions <= 2
                        repeatVec = {repeatVec{:} modesNames{ii,jj}};
                    end
                    idxs = find( ismember(modesNames(ii,:), modesNames{ii,jj})); 
                    [minV, minL] = min(modesNCC(ii, idxs));
                    modesNames{ii,idxs(minL)} = 'no';
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
            disp(['NMSE more than threshold = ', num2str(nmseThreshold),' for ', modesNames{ii,nmseCheck}])
            disp(num2str(modesNMSE(ii, nmseCheck)))
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
            colormap(c);
            end
            pause(0.00001);
            end
        end
        if printData 
            disp(['tuple ', num2str(ii) ]);
            disp(['NCC less than threshold = ', num2str(nccThreshold), 'for ' modesNames{ii,nccCheck}])
            disp(num2str(modesNMSE(ii, nccCheck)))
        end
        outliers(ii) = 1;
    end
end

nOutliers = length(outliers(outliers == 1));
disp(['n° Outliers + not recognized = ', num2str(nOutliers)])
nonOutliers = find(outliers == 0);
size(nonOutliers);

%% 6) SEE obtained modeshapes
% =========================================================================

tuples2See = 1:length(nonOutliers);
for jj = tuples2See
    nTuple = nonOutliers(jj);
    modesData = readmatrix(['modeshapes', int2str(nTuple), '.csv']);
    for ii = 1:nModes
    modeN = ii+2;
    modeSurf = reshape(modesData(:,modeN), [pY, pX]);
    figure(100)
    subplot(4,5,ii)
    imagesc(modeSurf)
    title(modesNames{nonOutliers(jj), ii})
    colormap(c);
    end
    sgtitle(['modeshapes tuple ', int2str(jj)])
end


%% 7)  DEFINITION OF THE MODES ORDER IN THE ORDERED DATASET 
% Define and obtain the modes order of the obtained dataset
findRefOrder = true;
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
    writeMat2File(dataset_modesOrder, 'refModesOrder.csv', {'mode '}, 1 , false)
else
    % fetch dataset modes Order
    dataset_modesOrder = table2cell(readtable('refModesOrder.csv'));
end

%% 8) GENERATE ORDERED DATASET 
% =========================================================================
%close all
modesNames = table2cell(readtable('modesNames.csv'));
outputsEig  = [];
outputsAmps  = {};

nPeaksDataset = 15;
nFRFs = 3;
Dataset_FRFs = {};
count = 0;
for ii = 1:nFRFs
    disp(['range idxs [', int2str(count*nPeaksDataset+1), ' - ',int2str((count+1)*nPeaksDataset) , ']']);
    Dataset_FRFs{ii} = Dataset_FA.outputsAmp(:,count*nPeaksDataset+1: (count+1)*nPeaksDataset);
    outputsAmps{ii} = [];
    count = count+1;
end

for jj = 1:length(dataset_modesOrder)
    modeFreqs = nan*ones(size(nonOutliers));
    modeAmps = nan*ones(length(nonOutliers), 3);
    nPeaks = nan*ones(size(nonOutliers));
   
    for ii = 1:length(nonOutliers)
        nPeak = find(ismember(modesNames(nonOutliers(ii),:), dataset_modesOrder{jj}));
        if ~isempty(nPeak)
            nPeaks(ii) = nPeak;
            modeFreqs(ii) =  Dataset_FA.outputsEig(nonOutliers(ii),nPeak);
            for kk = 1:nFRFs
            modeAmps(ii,kk) = Dataset_FRFs{kk}(nonOutliers(ii),nPeak);
            end
        end
    end
    outputsEig = [outputsEig, modeFreqs, nPeaks];
    for kk = 1:nFRFs
        outputsAmps{kk} = [outputsAmps{kk}, modeAmps(:,kk), nPeaks];
    end
end
outputsAmp = [outputsAmps{1} outputsAmps{2} outputsAmps{3}];

%% 9) SEE HISTOGRAM DISTRIBUTION OF THE DATASET FREQUENCIES
% =========================================================================
% see histogram
nModesSee = length(dataset_modesOrder);
startMode = 1;
peakNames = {};
nPoints = 25;
freqFig = 100;
count = 1;
step = 1;
modesAxis = startMode:step:(startMode + nModesSee*step - 1);

lineStl = {'-' ,'--' ,':' ,'-.'};
figure(freqFig); clf reset;

for ii = modesAxis
    figure(freqFig);
    
    subplot 211
    hold on
    freqIdx = 1 + (ii-1)*2; % only odd idxs
    [N,edges] = histcounts(outputsEig(~isnan(outputsEig(:, freqIdx)),freqIdx),nPoints);
    edges = edges(2:end) - (edges(2)-edges(1))/2;
    N = lowpass(N,0.625);
    H=area(edges,N, 'LineStyle', lineStl{mod(count,4)+1}, 'lineWidth', 1);

    H.FaceAlpha = 0.5;
    hold off
    if ii <= nModes
        subplot 212
        hold on
        [N,edges] = histcounts(Dataset_FA.outputsEig(:,ii),nPoints);
        edges = edges(2:end) - (edges(2)-edges(1))/2;
        N = lowpass(N,0.625);
        H=area(edges,N,'LineStyle', lineStl{mod(count,4)+1}, 'lineWidth', 1);
        H.FaceAlpha = 0.5;
        hold off
        peakNames{count} = ['peak', int2str(ii)];
        count = count +1 ;
    end
end    

% subplot handler
titles = {'by modes' 'by peaks'};
for ii = 1:2
    figure(freqFig)
    subplot(2,1,ii)
    xx = xlabel('$frequency$ $[Hz]$'); yy = ylabel('N occurences');
    set(xx, 'interpreter', 'latex'); set(yy, 'interpreter', 'latex');
    hold off; ax = gca; ax.FontSize = 15;
     set(gca,'ColorOrder', [cos(linspace(-0.45*pi,0.45*pi,nModesSee).'), abs(sin(linspace(pi/4, pi+pi/4,nModesSee).')), sin((linspace(0 , pi/3,nModesSee).'))]);
    if ii == 1, legend(dataset_modesOrder{modesAxis});
    else , legend(peakNames); end
    hold off;
    title(titles{ii});
    ylim([0,480]);
end

%% 10) SEE HISTOGRAM DISTRIBUTION OF THE DATASET AMPLITUDES
% =========================================================================
ampFigs = freqFig+1:freqFig+3;
titles = {'by modes' 'by peaks'};
frfs = {'H12' 'H13' 'H15'};
count = 1;
yPos = [0.73 0.25];
nPoints = 40;
xLimMins = [-70 -90 -105];
xLimMaxs = [-49 -50 -49];
for kk = 1:length(ampFigs)
    figure(ampFigs(kk));  clf reset;
    for ii = modesAxis
        freqIdx = 1 + (ii-1)*2; % only odd idxs
        figure(ampFigs(kk))
        subplot 211
        hold on
        [N,edges] = histcounts(db(outputsAmps{kk}(~isnan(outputsEig(:, freqIdx)),freqIdx)),nPoints);
        edges = edges(2:end) - (edges(2)-edges(1))/2;
%         N = lowpass(N,.99);
        H=area(edges,N, 'LineStyle', lineStl{mod(count,4)+1}, 'lineWidth', 1);
        H.FaceAlpha = 0.5;
        hold off
        if ii<= nModes
            subplot 212
            hold on
            [N,edges] = histcounts(db(abs(Dataset_FRFs{kk}(:,ii))),nPoints);
            edges = edges(2:end) - (edges(2)-edges(1))/2;
    %         N = lowpass(N,.99);
            H=area(edges,N, 'LineStyle', lineStl{mod(count,4)+1}, 'lineWidth', 1);
            H.FaceAlpha = 0.4;
            hold off
            count = count + 1;
        end
    end

    % subplot handler;
    for ii = 1:2    
        figure(ampFigs(kk))
        subplot(2,1,ii)
        xx = xlabel('$|H(f_i)|_{dB}$'); yy = ylabel('N occurences');
        set(xx, 'interpreter', 'latex'); set(yy, 'interpreter', 'latex');
        hold off; ax = gca; ax.FontSize = 15;
        set(gca,'ColorOrder', [cos(linspace(-0.45*pi,0.45*pi,nModesSee).'), abs(sin(linspace(pi/4, pi+pi/4,nModesSee).')), sin((linspace(0 , pi/3,nModesSee).'))]);
        if ii == 1, ll = legend(dataset_modesOrder{startMode:end});
        else , ll = legend(peakNames); end
        set(ll, 'Position', [0.92 yPos(ii) 0.06 0.06]);
        hold off;
        title([frfs{kk} ' : ' titles{ii}]);
        xlim([xLimMins(kk) xLimMaxs(kk)]);
    end
end

%% 11) Save the obtained dataset ordered by modes
% =========================================================================
cd(csvPath)

variableNamesEig = {};
variableNamesAmp = {};
countEig = 1;
countAmp = 1;
frfs = {'H12' 'H13' 'H15'};

for kk = 1:length(frfs)
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

varyingParamsNames = {'rho', 'Ex', 'Ey', 'Ez', 'Gxy', 'Gyz', 'Gxz', 'vxy', 'vyz', 'vxz', 'alpha', 'beta'};   
writeMat2File(outputsEig, 'datasetOrdered_Eig.csv', variableNamesEig,...
    length(variableNamesEig), true);
writeMat2File(outputsAmp, 'datasetOrdered_Amp.csv', variableNamesAmp,...
    length(variableNamesAmp), true);

writeMat2File(Dataset_FA.inputs(nonOutliers,:), 'datasetOrdered_Inputs.csv',...
    varyingParamsNames, length(varyingParamsNames), true);

modesIdxs = [];
peaksIdxs  = [];
count = 0;
nMod = length(dataset_modesOrder);
for kk = 1:length(frfs)
    modesIdxs = [modesIdxs; count*(2*nMod)+1:2:(count+1)*(2*nMod)];    
    count = count+1;
end
peaksIdxs = modesIdxs+1;

T = array2table(modesIdxs, 'variableNames', dataset_modesOrder, 'rowNames', frfs);
writetable(T, 'datasetOrdered_modesIdxs.csv');

T = array2table(peaksIdxs, 'variableNames', dataset_modesOrder, 'rowNames', frfs);
writetable(T, 'datasetOrdered_peaksIdxs.csv');
disp( ['END - Saved dataset of ', csvName])