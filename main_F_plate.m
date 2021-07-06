clear all;
close all;

%% Initial setup (ALWAYS RUN IN YOUR WORKING FOLDER)
aseFolder = pwd;
csvPath = [baseFolder, '\csv'];
addpath ([baseFolder, '\functions']);
addpath ([baseFolder, '\data']);
addpath(csvPath);

comsolModel = 'PlateMechParams_WithStudy';

infosTable = readtable("sampleMeasurements.xlsx");
infosMatrix = table2array(infosTable(:,3:end));
infosMatrix(:,1:2) = infosMatrix(:,1:2)*0.01;
infosMatrix(:,3:7) = infosMatrix(:,3:7)*0.001;

outputFilenames = {'input' 'output'};
outputFiletype = '.csv';

sampleNames = table2array(infosTable(:,1));
Ls      =   infosMatrix(:,1);
Ws      =   infosMatrix(:,2);
ThUp    =   infosMatrix(:,3);
ThDown  =   infosMatrix(:,4);
ThRight =   infosMatrix(:,5);
ThLeft  =   infosMatrix(:,6);
ThAvg   =   infosMatrix(:,7);
rhos    =   infosMatrix(:,9);
comsolParams =infosMatrix(1,1:6);

% Reference folders
baseFolder = pwd;
idx = 1;
standardDev = 0.1;

%% 1) OBTAIN EXPERIMENTAL RESULTS

% setup
jj = 1;
sampleNames = {'7b'};
fHigh = 1200;
M = 10;
thresholdPerc = 30;
idx = 1;
load('07_01_21_frf_anima_7b_fronte_G_post');

% 1) SVD
fAxis = f(f <= fHigh);
[HvSVD,threshold,singularVals] = SVD(H1_v, fAxis, M, thresholdPerc, sampleNames, jj);
    
% 2) MODAL ANALYSIS
[Hv,f0, fLocs, csis, Q] = EMASimple(HvSVD, fAxis,1e-4, 3, false);
f0 = f0(2:end);
fLocs = fLocs(2:end);
csis = csis(2:end);

% 3) Compute first guess of Ex
geom = infosMatrix(5,1:3);
rho = infosMatrix(5,end);
[mechParams, normParams] = computeParams([f0(1), f0(2), f0(3)],rho, geom)


fAmps = abs(Hv(fLocs));

%% 1.1) Convergence test for choosing correct mesh

% setup
model = mphopen('PlateMechParams_WithStudy');
nModes = 15;
meshSizes = {'C+++', 'C++', 'C+', 'C', 'N', 'F', 'F+' , 'F++', 'F+++'};
convergenceNames = {'Eig Error [%]'  'FD Error [%]' 'Eig Time [s]' 'FD Time [s]'};
freqNames = {'f1' 'f2' 'f3' 'f4' 'f5' 'f6' 'f7' 'f8' 'f9' 'f10' 'f11' 'f12' 'f13' 'f14' 'f15'};

% Execute Comsol studies
[eigenFreqzMatrix, pointVelMatrix, eigRate, FDRate, timesEig, timesFD] = ConvergenceTestMesh(model,nModes);
% generate figures with data
convergenceFigures(eigenFreqzMatrix, pointVelMatrix, eigRate, FDRate, timesEig, timesFD, [4,5] );

% save results
convRates = round([eigRate;FDRate; timesEig; timesFD],2);
errRatesTable = array2table(convRates, 'variableNames',meshSizes, 'rowNames',convergenceNames );
frequencyTable = array2table(eigenFreqzMatrix, 'rowNames',meshSizes, 'variableNames',freqNames );
magnitudeTable = array2table(pointVelMatrix, 'rowNames',meshSizes, 'variableNames',freqNames(1:8) );
writetable(errRatesTable, 'errRates.csv');
writetable(frequencyTable,'Eigenfrequencies.csv');
writetable(magnitudeTable,'Magnitude.csv');

%% 2) SET UP PARAMETERS FOR SIMULATIONS

% open Comsol model
model = mphopen(comsolModel);
% Parameters setup
params = mphgetexpressions(model.param);                  % get initial parameters                          
% get parameters names
varyingParamsNames = params(7:end,1);
steadyParamsNames = params(1:6,1);

% set geometry parameters
geomSet = [infosMatrix(5,1:6), infosMatrix(5,end)];
setParams = cell(length(steadyParamsNames)+1,1);
setParams(1:end-1) = steadyParamsNames;
setParams{end} = 'rho';

for jj = 1:length(setParams)
   model.param.set(setParams(jj), geomSet(jj));
end

% prepare reference values
Ex = mechParams(1);
referenceVals = [rho, Ex, Ex*0.078, Ex*0.043,...
                 Ex*0.061, Ex*0.064, Ex*0.003,...
                 0.467, 0.372, 0.435];

%% 3) DATASET GENERATION (Eigenfrequency)- in = mech params -- out = eigenfrequencies and modeshapes
% setup variables 
writeNow = true; % do you want to rewrite present files ? 

simFolder = [baseFolder,'\Simulations'];
cd(baseFolder)
% set if you want to write new csv files
writeNew = false;
% number of simulations
nSim = 3;
% Comsol model
model = mphopen(comsolModel);
% Comsol meshSize
meshSize = 9;
nModes = 20;
% setup Comsol
model.component('comp1').physics('solid').feature('lemm1').feature('dmp1').active(false);
model.mesh('mesh1').feature('size').set('hauto', int2str(meshSize));
model.mesh('mesh1').run;
model.study('std1').feature('eig').set('neigs', int2str(nModes));
model.result.export('data1').set('data', 'dset1');

[inputsInfo, inputsTable, outputsALLInfo, outputsALLTable] = comsolRoutineEigenfrequency(model, nSim, nModes, simFolder, csvPath, varyingParamsNames, referenceVals, standardDev,writeNow)

%% 4) LOOK FOR POISSON PLATES (if present)

L = Ls(1)
W = Ws(1)
aspectRatio = L/W;
inputsInfo = table2array(readtable("inputs.csv"));
poissonCheck = (inputsInfo(:,2)./inputsInfo(:,3)).^(1/4);
poissonPlates = intersect(find( poissonCheck > 0.99*aspectRatio),find( poissonCheck<1.01*aspectRatio));
length(poissonPlates)

for jj = 1:length(poissonPlates)
    if poissonPlates(jj) >=1
    ii = poissonPlates(jj);
    meshData = table2array(readtable(['mesh',int2str(ii),'.csv']));
    modesData = table2array(readtable(['modeshapes',int2str(ii),'.csv']));
    figure()
    plot3(meshData(:,1), meshData(:,2), modesData(:,4) ,'.', 'MarkerSize', 5);
    end
end

%% 5 ) Analyze Modeshapes
minimumThreshold = 1e-7;
% obtain csvPath at section 1
nSim = 3;
nModes = 20;
simFolder = [baseFolder, '\Simulations'];
modesFilename = 'modesNames.csv';

[modesNames, namesTable] = obtainModesNames(nSim, nModes,minimumThreshold, csvPath,simFolder, modesFilename);
modesNames = table2array(readtable(modesFilename));
[appears,maxLoc] = modesOrderAnalysis(nModes, csvPath, modesFilename)

%% 6.1) CORRELATIONS OF INPUTS (check if they actually are gaussian)

Cor = zeros(10,10);
for m1 = 1:10 % Create correlations for each experimenter
 for m2 = 1:10 % Correlate against each experimenter
  Cor(m2,m1) = abs(corr(inputsInfo(:,m1),inputsInfo(:,m2)));
 end
end
imagesc(Cor)
colorbar();

%% 6.2) MULTILINEAR REGRESSORS

nModes = 20;
[linearModels,multilinCoeffs, R2, errors, predictedOutputs] = multilinearRegress(inputsInfo,outputsALLInfo,nModes, referenceVals);

% Let's try and see where are outliers
figure()
toRemove = cell(length(linearModels),1);
% select the modes which are flipping
flipping = {'f12' 'f20'};
for ii = 1:length(linearModels)
    
    subplot(5,5,ii)
    plotResiduals(linearModels{ii});
    hold on
    maxResidual = max(abs(linearModels{ii}.Residuals.Raw));
    % select low R2 modes 
    if ii == 3 || ii == 4       
        exact_match_mask = strcmp(modesNames(:,3), flipping{ii-2});
        toRemove{ii} = find(exact_match_mask); 
    end
    legend(['f',int2str(ii)])
end

[linearModels,multilinCoeffs, R2n, errors, predictedOutputs] = multRgrOutliers(inputsInfo,outputsALLInfo, toRemove, nModes);

%% 6.3) See how each frequency is correlated to the mechanical parameters

    namesRef = {'rho' 'Ex' 'Ey' 'Ez' 'Gxy' 'Gyz' 'Gxz' 'vxy' 'vyz' 'vxz'};
    namesPlot = {  'Ex'  'Ey' 'Ez' 'Gxy' 'Gyz' 'Gxz'};
    freqzRef = {'f11' 'f02' 'f20' 'f12' 'f21'	'f03' 'f22'	'f30' 'f13'	'f31'};
    freqzPlot = {'f11' 'f02' 'f20' 'f12' 'f21'	'f03' 'f22'	'f30' };
    freqzIndex = find(ismember(freqzRef(:), freqzPlot(:)) == 1);
    paramsIndex = find(ismember(namesRef(:), namesPlot(:)) == 1);
    [normMultilinCoeffs] = multilinSensitivity(multilinCoeffs,referenceVals, namesRef, freqzRef,  paramsIndex, freqzIndex);

%% 7.1) MINIMIZATION OF THE ERROR 
% 1) first guess of mech parameters
errors = zeros(size(outputsALLInfo(:,1)));
% index for displaying comsol results
displayIndex = [1,2,3,4,5,5,6,6,7,8,9,9,9,10,10,11,12,12,12,12].';
% actual indexes used for the minimization
indexComsol = [1,2,5,14].';
indexReal = [1,2,5,10].';

for ii = 1:length(outputsALLInfo(:,1))
    errors(ii) = norm((f0(indexReal)-outputsALLInfo(ii,indexComsol).')./f0(indexReal),2);
end
[minVals, minimumLoc] = min(errors);

% 2) setup for minimization
nRealizations = 5;
% Check section 4, using where appears most
maxLoc;
plotData = true;
freqMatrix = zeros(nRealizations,length(indexReal)); 
parsMatrix = zeros(nRealizations,10);
gauss = randn(nRealizations,1);

% 3) minimization process
for ii = 1:nRealizations
    density = rho*(1+0.05*gauss(ii));
    [xpar, f_out, fval] = minimizeError(linearModels, inputsInfo, outputsALLInfo,...
                                        f0,minimumLoc,density, indexComsol, indexReal, false);
    diff = (f0(indexReal) - f_out(indexComsol))./f0(indexReal);                               
    err = mean(abs(diff))*100;
    if plotData
        figure()
        plot(displayIndex, f_out, '.', 'markerSize', 10);
        hold on 
        plot(indexReal, f0(indexReal), '-x');
        xlabel('N mode')
        ylabel(' f     [Hz]');
        legend('fMultilin', 'fexp');   
        end
    freqMatrix(ii,:) = f_out(indexComsol).'; 
    parsMatrix(ii,:) = xpar.';
end

% 4) mean and variance of the estimated eigenfrequencies
meanFreq = mean(freqMatrix);
varFreq = var(freqMatrix);

figure() 
plot(1:length(f0(indexReal)), f0(indexReal), '-o');
legend('fexp');
hold on 
errorbar(1:length(meanFreq),meanFreq,varFreq)
legend('fexp','fopt');
xlabel('mode number  N' );
ylabel('frequency    [Hz]');

% 5) estimation of mechanical parameters and save restults
meanMechParams = mean(parsMatrix);
varianceMechParams1 = var(parsMatrix);
finalStd = sqrt(mean( abs( parsMatrix - meanMechParams).^2 ))./meanMechParams;
matlabStd = std(parsMatrix)./meanMechParams;

stdNames = {'std rho [%]', 'std Ex [%]', 'std Ey [%]', 'std Ez [%]',...
                    'std Gxy [%]', 'std Gyz [%]', 'std Gxz [%]', 'std vxy [%]', 'std vyz [%]', 'std vxz [%]'};
varyingParamsNames = {'rho', 'Ex', 'Ey', 'Ez', 'Gxy', 'Gyz', 'Gxz', 'vxy', 'vyz', 'vxz'};   
 
finalStdTable = array2table(finalStd*100, 'VariableNames', stdNames);
myStdTable = array2table(finalStd*100, 'VariableNames', stdNames);
matlabStdTable = array2table(matlabStd*100, 'VariableNames',stdNames);
finalOutTable = writeMat2File([meanMechParams; matlabStd*100],'Results.csv', varyingParamsNames, 10,true); 
writeMat2File(parsMatrix,'ResultsALL.csv', varyingParamsNames, 10,true);


%% 8.1) STUDY MINIMIZATION ERROR

cd(csvPath)
plotData = false;
% setup
comsolIndex = [1,2,4,3,5,7,9,10,12,14,15,16].';
realIndex = [1,2,3,4,5,6,7,8,9,10,11,12].';
presentModes = {'f11' 'f02' 'f20' 'f12' 'f21' 'f22' 'f30' 'f31' 'f32' 'f05' 'f15' };
numMechParams = 10;

errors = zeros(size(outputsALLInfo(:,1)));
for ii = 1:length(outputsALLInfo(:,1))
    errors(ii) = norm((f0(realIndex)-outputsALLInfo(ii,comsolIndex).')./f0(realIndex),2);
end
[minVals, minimumLoc] = min(errors);
nFreq =  length(comsolIndex);

% ModesPermute is an array containing the number of modes you want to permute 
modesPermute = [12];

% minimization with permutations of the indexes
[params,freqz,errs] = studyMinError_eig(realIndex, comsolIndex, modesPermute, modesNames, maxLoc,  linearModels, inputsInfo, outputsALLInfo, f0,minimumLoc,rho, numMechParams,plotData);

% for cycle to save mechParams and eigenfrequencies
for ii = modesPermute
    varyingParamsNames = {'rho', 'Ex', 'Ey', 'Ez', 'Gxy', 'Gyz', 'Gxz', 'vxy', 'vyz', 'vxz'};
    freqzNames = {'f1' 'f2' 'f3' 'f4' 'f5' 'f6' 'f7' 'f8' 'f9' 'f10' 'f11' 'f12'};
    N = length(varyingParamsNames);
    M = length(freqzNames);
    for tt = 1:ii
    varyingParamsNames{N+tt} = ['used f', int2str(tt)];
    freqzNames{M+tt} = ['used f', int2str(tt)];
    end
    writeMat2File(params{ii},['mechParams',int2str(ii),'.csv'], varyingParamsNames, length(varyingParamsNames),true);     
    writeMat2File(freqz{ii},['eigenfreqz',int2str(ii),'.csv'], freqzNames, length(freqzNames),true);   
end

% for cycle to save error
tables = cell(length(modesPermute),1);
for ii = 1:length(modesPermute)
    variableNames = cell(modesPermute(ii)+2, 1);
    for jj = 1:modesPermute(ii)+2
        count = jj-1;
        if jj == 1
            variableNames{jj} = 'err [%]';
        else
            if jj > 1 && jj < modesPermute(ii)+2
               variableNames{jj} = ['used f', int2str(jj)]; 
            else
               variableNames{jj} = 'Mech params'; 
            end
        end
    end
    tables{ii} =  sortrows(cell2table(errs{modesPermute(ii)}, 'VariableNames', variableNames));
    writetable(tables{ii},['errs', int2str(modesPermute(ii)), '.csv']);
end
