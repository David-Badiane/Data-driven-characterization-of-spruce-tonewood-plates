clear all 
close all

%% 0) Initial setup

% Reference folders
baseFolder = pwd;
csvName = 'csv_plate_FA';
csvPath = [baseFolder, '\', csvName];
modesAnalysisPath = [csvPath, '\modesAnalysis'];
resampledPath = [csvPath,'\resampledModeshapes'];
addpath(modesAnalysisPath);
addpath ([baseFolder, '\functions']);
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

%% 1) FETCH DATASET - (import csv )
mechParamsNames = {'\rho' 'E_x' 'E_y' 'E_z' 'G_{xy}' 'G_{yz}' 'G_{xz}' '\nu_{xy}' '\nu_{yz}' '\nu_{xz}' '\alpha' '\beta'};
Dataset_FA = struct('inputs',[] ,'outputsEig',[] ,'outputsAmp',[] );
Dataset_FA.inputs = table2array(readtable('inputs.csv'));
Dataset_FA.outputsEig = table2array(readtable('outputsEig.csv'));
Dataset_FA.outputsAmp = table2array(readtable('outputsAmp.csv'));
nTuples = length(Dataset_FA.inputs(:,1));

%% 1.b) Resample all modeshapes
addpath([csvPath, '\Modeshapes']);
fileList = ls([csvPath, '\Modeshapes']);
fileList = fileList(3:end);
nTuples = length(fileList(:,1));
plotData = false;

resampleMshapes(pX,pY,nModes, nTuples,...
    resampledPath, csvPath, plotData)

rmpath([csvPath, '\Modeshapes']);

%% 2) - Obtain and name reference shapes 
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
modesData = readmatrix('reference_raw.csv');
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

%% 3) - reference shapes resampling
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

%% 5) see and save reference Shapes 
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
    figure(121)
    subplot(8,5,ii)
    imagesc(modeSurf); title(refModesNames{ii});
end

%% 6) - Computation loop - compute NCC and NMSE with ref modeshapes
                        %- associate to each one the best scoring one
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

%% 7) Analyse modeshapes by metrics
modesNMSE = readmatrix('modesNMSE.csv');
modesNCC = readmatrix('modesNCC.csv');
modesNames = table2cell(readtable('modesNames.csv'));

pX = 75;
pY = 37; 
outliers = zeros(size(modesNames(:,1)));
plotFigures = false; 
repeatVec = {};

for ii = 1:length(modesNames(:,1))
    %check if there are repeated modes
    if length(unique(modesNames(ii,:))) < length(modesNames(ii,:))
        m = readmatrix(['modeshapes',int2str(ii),'.csv']);
        repMsgs = {};
        count = 1;
        disp( ' ' );
        for jj = 1:length(modesNames(1,:))
            if plotFigures
                figure(77)
                subplot(5,3,jj)
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
                disp(repMsgs{count});
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
    if length(find(modesNMSE(ii,:) > 0.35)) >= 2
        if plotFigures
            m = readmatrix(['modeshapes',int2str(ii),'.csv']);
            figure(77)
            for jj = 1:length(modesNames(1,:))
            subplot(5,3,jj)
            imagesc(reshape(m(:,jj+2), [pY,pX]));
            title(modesNames{ii,jj});
            end
        end
        outliers(ii) = 1;
    end
    
 %       check by NCC
    if length(find(modesNCC(ii,:) < 0.9)) >= 2
        if plotFigures
            m = readmatrix(['modeshapes',int2str(ii),'.csv']);
            figure(77)
            for jj = 1:length(modesNames(1,:))
            subplot(5,3,jj)
            imagesc(reshape(m(:,jj+2), [pY,pX]));
            title(modesNames{ii,jj});
            end
        end
        outliers(ii) = 1;
    end
end

length(outliers(outliers == 1))
nonOutliers = find(outliers == 0);
size(nonOutliers);

%% 5) LOOK FOR POISSON PLATES (if present)
simFolder = [csvPath, '\resampledModeshapes'];

L = Ls(1);
W = Ws(1);
aspectRatio = L/W;
poissonCheck = (Dataset_FA.inputs(nonOutliers,2)./Dataset_FA.inputs(nonOutliers,3)).^(1/4);
poissonPlates = intersect(find( poissonCheck > 99/100*aspectRatio),find( poissonCheck<101/100*aspectRatio));
length(poissonPlates)
plotPlates = true;

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
            end
            pause(0.1);
        end      
    end
end
% delete remaining poisson plates
nonOutliers(poissonPlates) = [];

%% SEE obtained modeshapes
pX = 75; pY = 37;
for jj = 1:length(nonOutliers)
    nTuple = nonOutliers(jj);
    modesData = readmatrix(['modeshapes', int2str(nTuple), '.csv']);
    for ii = 1:(length(modesData(1,:)) -2)
    modeN = ii+2;
    modeSurf = reshape(modesData(:,modeN), [pY, pX]);
    figure(100)
    subplot(3,5,ii)
    imagesc(modeSurf)
    title(modesNames{nonOutliers(jj), ii})
    end
end

%% Modes Order Analysis
modesFilename = 'modesNames.csv';
[appears,maxLoc] = modesOrderAnalysis(nModes, csvPath, modesFilename);

%% Order the Dataset
modesOrder = modesNames(maxLoc,:);

%% Variation of first modes (mode by mode Analysis)
modesNames = table2cell(readtable('modesNames.csv'));
modes2Analyse = {modesOrder{:}, 'f33', 'f05' 'f40' 'f15' 'f41'};
outputsEig  = [];
outputsAmp  = [];
for jj = 1:length(modes2Analyse)
    modeFreqs = nan*ones(size(nonOutliers));
    modeAmps = nan*ones(size(nonOutliers));
    nPeaks = nan*ones(size(nonOutliers));
    
    for ii = 1:length(nonOutliers)
        nPeak = find(ismember(modesNames(nonOutliers(ii),:), modes2Analyse{jj}));
        if ~isempty(nPeak)
            nPeaks(ii) = nPeak;
            modeFreqs(ii) =  Dataset_FA.outputsEig(nonOutliers(ii),nPeak);
            modeAmps(ii) = Dataset_FA.outputsAmp(nonOutliers(ii),nPeak);
        end
    end
    outputsEig = [outputsEig, modeFreqs, nPeaks];
    outputsAmp = [outputsAmp, modeAmps, nPeaks];
end

figure()
hold on;
for ii = 1:length(modes2Analyse)
    freqIdx = 1+(ii-1)*2;
    histogram(outputsEig(~isnan(outputsEig(:, freqIdx)),freqIdx),30);
    xlabel('frequency value');
    ylabel('N occurences');
    title(['modes histogram']);
end
legend(modes2Analyse{:})

%% Save the obtained dataset ordered by modes
variableNamesDataset = {};
count = 1;
for ii = 1:length(modes2Analyse)
        variableNamesDataset{count} = modes2Analyse{ii};
        count = count+1;
        variableNamesDataset{count} = [modes2Analyse{ii},' peak N'];
        count = count+1;
end

varyingParamsNames = {'rho', 'Ex', 'Ey', 'Ez', 'Gxy', 'Gyz', 'Gxz', 'vxy', 'vyz', 'vxz', 'alpha', 'beta'};   

writeMat2File(outputsEig, 'datasetOrdered_Eig.csv', variableNamesDataset,...
    length(variableNamesDataset), true);
writeMat2File(outputsAmp, 'datasetOrdered_Amp.csv', variableNamesDataset,...
    length(variableNamesDataset), true);

writeMat2File(Dataset_FA.inputs(nonOutliers,:), 'datasetOrdered_Inputs.csv',...
    varyingParamsNames, length(varyingParamsNames), true);

