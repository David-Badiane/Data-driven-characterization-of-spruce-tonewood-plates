clear all;
close all;


%% 0) Initial setup

% Reference folders
% cd(baseFolder)
% rmpath(genpath(baseFolder));
% specify paths
baseFolder = pwd;
csvName = 'csv_plate_gaussian_G10';
convergencePath = [baseFolder, '\convergenceTest_plate'];
testPath = [baseFolder, '\testSets'];
trainPath = [baseFolder, '\trainSets'];
csvPath = [baseFolder, '\', csvName];
NNPath = [baseFolder '\NNs'];
% add paths to workline
idxs   = strfind(baseFolder,'\');
addpath(genpath([baseFolder(1:idxs(end)), 'functions']));
addpath([baseFolder, '\data']);
addpath(csvPath);
addpath(testPath);
addpath(trainPath);
addpath(NNPath)
%comsol Model name
comsolModel = 'woodenPlate';
varyingParamsNames = {'rho', 'Ex', 'Ey', 'Ez', 'Gxy', 'Gyz', 'Gxz', 'vxy', 'vyz', 'vxz', 'alpha', 'beta'};   
dDirs = {'csv_plate_gaussian_G10', 'csv_plate_gaussian' 'csv_plate_uniform_0.50','csv_plate_uniform_0.75'};

nnNames = cellstr(ls(NNPath)); nnNames = nnNames(3:end);
for ii = 1:length(nnNames) , nnNames{ii, 2} = ii; end

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


%% 1) OBTAIN EXPERIMENTAL RESULTS
% setup
jj = 1;
sampleNames = {'7b'};
fHigh = 1200; fLow = 53;
M = 10;
thresholdPerc = 30;
idx = 1;
load('07_01_21_frf_anima_7b_fronte_G_post');

% 1) SVD
fAxis = f(f <= fHigh);
% [HvSVD,threshold,singularVals] = SVD(H1_v, fAxis, M, thresholdPerc, sampleNames, jj);
    
% 2) MODAL ANALYSIS
[Hv,f0, fLocs, bandsIdxs] = EMASimple(H1_v, fAxis,5e-4, 3,true);
f0 = f0(2:end);
fLocs = fLocs(2:end);

% 3) Compute first guess of Ex
[mechParams, normParams] = computeParams([f0(1), f0(2), f0(3)],rho, geom);
fAmps = abs(Hv(fLocs));


%% 1.1)  Convergence test for choosing correct mesh

% setup
model = mphopen(comsolModel);
nModes = 15;
meshSizes = {'C+++', 'C++', 'C+', 'C', 'N', 'F', 'F+' , 'F++', 'F+++'};
convergenceNames = {'Eig Error [%]' 'Eig Time [s]' };
freqNames = {'f1' 'f2' 'f3' 'f4' 'f5' 'f6' 'f7' 'f8' 'f9' 'f10' 'f11' 'f12' 'f13' 'f14' 'f15' 'f16' 'f17' 'f18' 'f19' 'f20'};

% Execute Comsol studies
[eigenFreqzMatrix, pointVelMatrix, eigRate, timesEig] = ConvergenceTestMesh(model,nModes, convergencePath);
% generate figures with data
convergenceFigures(eigenFreqzMatrix, pointVelMatrix, eigRate, FDRate, timesEig, timesFD, [4,5] );

% save results
cd(convergencePath);
convRates = round([eigRate;FDRate; timesEig; timesFD],2);
errRatesTable = array2table([eigRate; timesEig], 'variableNames',meshSizes, 'rowNames',convergenceNames );
frequencyTable = array2table(eigenFreqzMatrix, 'rowNames',meshSizes, 'variableNames',freqNames );
magnitudeTable = array2table(pointVelMatrix, 'rowNames',meshSizes, 'variableNames',freqNames(1:8) );
writetable(errRatesTable,'errRatesFreq.csv');
writetable(frequencyTable,'Eigenfrequencies.csv');
writetable(magnitudeTable,'Magnitude.csv');
cd(baseFolder);

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
                 0.467, 0.372, 0.435, 21.0001, 1e-6];
             
%% 3) NEW DATASET GENERATION
% setup variables 
simFolder = [csvPath,'\Modeshapes'];
mkdir(simFolder)
cd(baseFolder)
% set if you want to write new csv files
writeNow = true;
% number of simulations
nSim = 4000;
% Comsol model
model = mphopen(comsolModel);
% Comsol number of eigenfrequencies computed
nModes = 20;  

meshSize = 6;
model.mesh('mesh1').feature('size').set('hauto', int2str(meshSize));
model.mesh('mesh1').run;

% if 'uniform' - density is still gaussian - alpha
% and beta
% 'rho' 'Ex' 'Ey' 'Ez' 'Gxy' 'Gyz' 'Gxz' 'vxy' 'vyz' 'vxz' 'alpha' 'beta'
%standardDev = [2 2*ones(size(referenceVals(2:7))) 2*ones(size(referenceVals(8:10))) 2*ones(size(referenceVals(11:12))) ];  
standardDev = 0.10*ones(size(referenceVals)) ; 
alphaCenterVal = 50;
betaCenterVal = 2e-6;
[Dataset_FA] = comsolRoutineFA_plate(model, nSim, nModes, referenceVals,...
                                   varyingParamsNames,  standardDev,  simFolder, csvPath, writeNow, 'gaussian');
                       
%% 4) Fetch DATASET - (import csv )
% restore path
cd(baseFolder)
rmpath(genpath(csvPath));
% specify if save the dataset
saveData = input('save the datasets in .mat files? (0/1): ');
% fetch the dataset
[Dataset_FA, csvPath, datasetType, datasetDistr] = fetchDataset_plate(saveData, baseFolder, 1, dDirs{2});
% if the dataset is ordered retrieve modes presence
if(strcmp(datasetType, 'ordered'))
    cd(csvPath);
    modesPresence = readtable('modesPresence.csv');
end

%% 8) Minimization of the error
% =========================================================================

% preallocate and set variables
nRealizations = 6;
plotData = [1, 0, 0]; % end figures, loss fx, first guess
NpeaksAxis = [1:11];
ampScale = 'db';
resultsFilename = ['Results_nR', int2str(nRealizations),'_',int2str(length(NpeaksAxis)) '_' ampScale];
normalization = 'ratios';
algorithm = 'fixRayleighParams';

rayleighParams = [18, 1e-5];
if strcmp( algorithm, 'fixRayleighParams')
    disp('        FIXED RAYLEIGH PARAMS        ')
    disp(['alpha = ',num2str(rayleighParams(end-1)),'  ---   beta = ',num2str(rayleighParams(end))]);
end

freqMatrix = zeros(nRealizations,length(NpeaksAxis));
ampMatrix = zeros(nRealizations,length(NpeaksAxis)); 
parsMatrix = zeros(nRealizations,12);
gauss = randn(nRealizations,1);
mapComsol = zeros(nRealizations, length(f0));

options = optimset('fminsearch');
options = optimset(options, 'TolFun',1e-8,'TolX',1e-8, 'MaxFunEvals',7.5e3,'MaxIter', 5e3); 
maps = [];

% fetch dataset and regressors
if input('fetch and reduce Dataset, get NNs? (0/1): ')
    getOrdered = true; modesGet = 15;
    [Dataset_FA, csvPath, datasetType, datasetDistr] = fetchReduceDataset_plate(baseFolder, modesGet, getOrdered);
    cd(csvPath); load(['optNN_' int2str(modesGet) 'Modes.mat']); load(['MLRfreqs_' int2str(modesGet) '.mat']); 
    ML = MLs{1};
    cd(baseFolder);
end

scaleBinds = 1e7;
ub = max(Dataset_FA.inputs); 
ub(end) = ub(end)*scaleBinds; ub(end-1) = ub(end-1)*scaleBinds;
lb = min(Dataset_FA.inputs); 
lb(end) = lb(end)*scaleBinds; lb(end-1) = lb(end-1)*scaleBinds;

% minimiziation process
for ii = 1:nRealizations
    tStart = tic;
    density = rho*(1+0.025*gauss(ii));
    %density = round(rho);
    disp(['rho = ', num2str(round(density,2))]);
    
    [xpar, map, f_out, amp_out, fval, idxComsol] = ...
    minimization_FA(options, MLs{1}, aNet, Dataset_FA, f0, fAmps,density, NpeaksAxis,...
                    plotData, algorithm, rayleighParams, normalization, ampScale, lb, ub, scaleBinds);

    maps(ii,:) = map;
    freqMatrix(ii,:) = f_out(map).'; 
    ampMatrix(ii,:) = amp_out(map).';
    parsMatrix(ii,:) = xpar.';
    mapComsol(ii,:) = idxComsol.';
    disp(toc(tStart));
end

meanFreq = mean(freqMatrix);
varFreq = var(freqMatrix);

figure() 
plot(1:length(f0), f0, '-o');
legend('fexp');
hold on 
errorbar(1:length(meanFreq),meanFreq,varFreq)
legend('fexp','fopt');
xlabel('mode number  N' );
ylabel('frequency    [Hz]');

meanMechParams = mean(parsMatrix);
matlabStd = std(parsMatrix)./meanMechParams;

stdNames = {'std rho [%]', 'std Ex [%]', 'std Ey [%]', 'std Ez [%]',...
                    'std Gxy [%]', 'std Gyz [%]', 'std Gxz [%]', 'std vxy [%]', 'std vyz [%]', 'std vxz [%]', 'std alpha[%]', 'std beta[%]'};
varyingParamsNames = {'rho', 'Ex', 'Ey', 'Ez', 'Gxy', 'Gyz', 'Gxz', 'vxy', 'vyz', 'vxz', 'alpha', 'beta'};   

matlabStdTable = array2table(matlabStd*100, 'VariableNames',stdNames);
resultsPath = [csvPath,'\Results'];
mkdir(resultsPath); cd(resultsPath);
if input('save file ? (0/1): ')
finalOutTable = writeMat2File([meanMechParams; matlabStd*100],[resultsFilename,'.csv'], varyingParamsNames, 10,true); 
end

%% 7.2) Check quality of the result  (eigenfrequency study with estimated params)
% =========================================================================

 for jj = 1:length(varyingParamsNames)
     if jj == 1, model.param.set(varyingParamsNames(jj), 400); else
                model.param.set(varyingParamsNames(jj), meanMechParams(jj)); end
 end
%params = mphgetexpressions(model.param);
model.component('comp1').mesh('mesh1').feature('size').set('hauto', 5);
model.mesh('mesh1').run;
model.study('std1').run(); 
evalFreqz = mpheval(model,'solid.freq','Dataset','dset1','edim',0,'selection',1);
eigenFreqz = real(evalFreqz.d1');

figure()
plot(1:length(f0), f0, '-o');
hold on;
plot(1:length(eigenFreqz(map)), eigenFreqz(map),'-x');
xlim([1,length(eigenFreqz(map))]);
xlabel('mode number N');
ylabel('frequency [Hz]');
legend('real f0','Comsol - minimization');

writeMat2File(eigenFreqz,'EigenfreqzComsol.csv', {'f'}, 1,false); 

%% 7.3) EXPORT & COMPARE POINT RESPONSE OF FRF (frequency domain with estimated parameters)
% =========================================================================
% Comsol routine
if input('fetch model? (1/0): '), model = mphopen(comsolModel); end
csvName = 'csv_plate_gaussian_G10'; 
resultsPath = [baseFolder,'\', csvName,'\Results'];
meshSize = 6; fLow = 53; fHigh = 1200;
fAxis = f(f>fLow); 
nPointsAxis = 600; nPeaksUsed = 7;
alpha = 18; beta = 4e-6; nRealizations = 30;
ampScaleUsed = 'db';
getReferenceValsResult = true;
alphas = linspace(1,100,10).';
legenda = {};
colors = [zeros(size(alphas)) linspace(0,1,length(alphas)).' zeros(size(alphas))];
% colors = [zeros(size(betas)) linspace(0,1,length(betas)).' zeros(size(betas))];

for ii = 1
[vel, fAxisComsol, resultsFilename] = comsolPointFRF(model, resultsPath, meshSize, fAxis, fHigh,...
                             nPointsAxis, nPeaksUsed, alpha, beta, nRealizations, getReferenceValsResult, ampScaleUsed);                         
figure(151); hold on;
plot(fAxisComsol, db(abs(vel)), 'lineWidth', 0.9, 'Color', colors(ii,:));
legenda{ii} = ['$\alpha = ', num2str(alphas(ii),7) ,'$'];
pause(0.01);
end                         
ll = legend(legenda);
set(ll, 'Interpreter', 'latex'); set(ll, 'Box', 'off');

fAxis = f(f>fLow & f<fHigh);
Hv = H1_v(f>fLow & f<fHigh);

%% See Img
% =========================================================================
% set up img data entries
% count = 0;

xLengthImg = 1200; yLengthImg = round(7/15 * xLengthImg);
imgN = 45 + count;
xLabel = 'Frequency [Hz]'; yLabel = '$|H_v(f)|$';
areaColor = [0.1 0.5 0.2];
annotFontSize = 18; axFontSize = 24;
areaAlpha = .4; 
errTextPos = [.75 .225 .4 .3]; paramsTextPos = [.75 .125 .4 .3];
legenda = {'real FRF', 'simulated FRF','real peaks', 'Comsol peaks', 'integral error'};
lineWidth = 1.5;

% set up FRF Data entries (fetch Hv at section 1.1)
cutHigh = 1200; cutLow = 53;

if input('fetch fAxisComsol and comsol point FRF ? (0/1): ')
    vec = input('insert [alpha, beta, nRealizations] as string : ') ; 
    vec = sscanf(vec, '%f');
    alpha = vec(1); beta = vec(2); nRealizations = vec(3);
    nPeaksUsed = input('set n Peaks used: ');    
    velName = ['vel_nR',int2str(nRealizations),'_', int2str(nPeaksUsed),'_a_b_', num2str(alpha),'_',num2str(beta)];
    fAxisName = ['vel_fAxis_nR', int2str(nRealizations),'_',int2str(nPeaksUsed),'_a_b_', num2str(alpha),'_',num2str(beta)];   
    [vel] = readTuples([velName,'.txt'], 1, false);
    vel = vel(4:end);
    fAxisComsol = readmatrix([fAxisName, '.csv']);
end

[imgData, FRFData] = defImg_pointFRFCompare( xLengthImg, yLengthImg, imgN,...
                            xLabel, yLabel, areaColor, annotFontSize, axFontSize,...
                            areaAlpha, errTextPos, paramsTextPos, legenda, lineWidth, ...
                            cutHigh, cutLow, Hv, fAxis, fAxisComsol, vel,...
                            alpha, beta, nRealizations);

[img] = export_pointFRFCompare(FRFData, imgData, 'db',4);
count = count +1;
%% See only simulated point FRF
% =========================================================================
resultsPath = [csvPath, '\Results']; cd(resultsPath);
alpha = 19; beta = 1e-6; nRealizations = 5;
nPeaksUsed = 5;    
velName = ['vel_nR',int2str(nRealizations),'_', int2str(nPeaksUsed),'_a_b_', num2str(alpha),'_',num2str(beta)];
fAxisName = ['vel_fAxis_nR', int2str(nRealizations),'_',int2str(nPeaksUsed),'_a_b_', num2str(alpha),'_',num2str(beta)];   
[vel] = readTuples([velName,'.txt'], 1, false);
vel = vel(4:end);
fAxisComsol = readmatrix([fAxisName, '.csv']);

% set up img data entries
cutLow = 53;
cutHigh = 350;
addInset = true;
xLengthImg = 900; yLengthImg = round(6/15 * xLengthImg);
imgN = 45; xLabel = 'Frequency [Hz]'; yLabel = '$|H(f)| \:\: \left[\frac{\mathrm{s}}{\mathrm{kg}}\right]$';
axFontSize = 24;
legenda = {'|H(f)|', 'peaks'};
lineWidth = 2;

[imgData, FRFData] = defImg_pointFRF( addInset, xLengthImg, yLengthImg, imgN,...
                            xLabel, yLabel, axFontSize, legenda, lineWidth,...
                            cutHigh, cutLow, vel, fAxisComsol, baseFolder);
saveImg = true; 
saveDir = 'C:\Users\utente\Desktop\polimi\Thesis FRF2Params - violins\paperFigures_NNs\plate';
saveName = 'pointFRF_noinset';
[img] = export_pointFRF(FRFData, imgData, saveImg, saveDir, saveName);


%% OLD STUFF
%% Minimization 2 - narrow band frequency domain studies
% =========================================================================
    
model = mphopen(comsolModel);

fAmps = abs(Hv(fLocs));

mechParameters = table2array(readtable('Results68.csv'));
mechParameters_1stguess = [mechParameters(1,:), 19, 7e-6];
params = mphgetexpressions(model.param);
paramsNames = params(7:end,1);

model.mesh('mesh1').feature('size').set('hauto', '6');
model.mesh('mesh1').run;

    for ii = (1:length(mechParameters_1stguess))
        model.param.set(paramsNames(ii), mechParameters_1stguess(ii));
    end
         
constraintWeight = 0.05;

options = optimset('fminsearch');
options = optimset(options, 'TolFun',1e-8,'TolX',1e-8, 'MaxFunEvals',100,'MaxIter', 100,...
    'DiffMinChange', 150, 'DiffMaxChange', 300); 
firstGuess = [mechParameters_1stguess(2:end)];

pNames4Min = {paramsNames{2:end-3}, 'alpha', 'beta'};

% fun = @(x) minimizationComsolBased(x,  model, f0, fAmps, constraintWeight, pNames4Min, mechParameters_1stguess);
fun2 = @(x) minimizationComsolBased2(x,  model, f0, fAmps, constraintWeight, pNames4Min, mechParameters_1stguess,false);

% cd('C:\Users\utente\.comsol\v56\llmatlab\codesStudy - Parameters - shear\minComsol1');
% [xpar1,fval1, exitflag, output] = fminsearch(fun, firstGuess, options);

cd('C:\Users\utente\.comsol\v56\llmatlab\codesStudy - Parameters - shear\MinComsol');
[xpar,fval, exitflag, output]  = fminsearch(fun2, firstGuess, options);

% figure()
% syms x
% f = (abs(x) *100+0.01).^-1;
% fplot(f, [-2,2])
% xlabel('x = \chi(r) - \chi(r-1)')
% ylabel('f(x)')

%% GENERATE DATA FOR LOSS FUNCTION EVALUATION AND STUDY
% =========================================================================
cd(baseFolder);

model = mphopen(comsolModel);
mechParameters = table2array(readtable('Results68.csv'));
mechParameters_1stguess = [mechParameters(1,:), 19, 7e-6];

params = mphgetexpressions(model.param);
paramsNames = params(7:end,1);

model.mesh('mesh1').feature('size').set('hauto', '6');
model.mesh('mesh1').run;
firstGuess = [mechParameters_1stguess(2:end)];

paramsNames4Min = paramsNames(2:end-3);

cd('C:\Users\utente\.comsol\v56\llmatlab\codesStudy - Parameters - shear\alphaBetaVariationStudy')
[inputTable, ampsTable, eigTable] = genFreqAmpData(firstGuess, model, paramsNames4Min);


cd('C:\Users\utente\.comsol\v56\llmatlab\codesStudy - Parameters - shear\Data4LossFunction')
[inputTable, ampsTable, eigTable] = genLFData(firstGuess, model, paramsNames4Min);


%% 7.4) check the Q factor values for simulated and real FRFs
% =========================================================================
    
    fAxis = 20:0.5:fHigh;
    [Hv,f0, fLocs, csis, Q] = EMASimple(HvSVD, fAxis,1e-3, 3);
    f0 = f0(2:end); csis = csis(2:end); Q = Q(2:end); fLocs = fLocs(2:end);
    %fAxis = 50:0.5:600;
    [HvComsol,f0Comsol, maxLocs, csisComsol, QComsol] = EMAPoly(vel, fAxis,1e-10, 1, false);
    
    Qs = [Q.'; QComsol(1:length(Q)).'];
    Qtable = array2table(Qs, 'variableNames',{'Q1' 'Q2' 'Q3' 'Q4' 'Q5' 'Q6' 'Q7' 'Q8' 'Q9' 'Q10'}, 'rowNames', {'Real' 'Comsol'});
    writeMat2File(Qs,'Qs.csv', {'Q'}, 1,false); 

    
%% 7.1a) CORRELATIONS OF INPUTS (check if they actually are gaussian)
nPars = 12;
parsAxis = 1:length(referenceVals);
Cor = zeros(nPars,nPars);
for m1 = parsAxis % Create correlations for each experimenter
 for m2 = parsAxis % Correlate against each experimenter
  Cor(m2,m1) = abs(corr(Dataset_FA.inputs(:,m1),Dataset_FA.inputs(:,m2)));
 end
end

figure()
imagesc(Cor)
colorbar();
set(gca,'YTick',parsAxis,'YTickLabel',mechParamsNames(parsAxis))
set(gca,'XTick',parsAxis,'XTickLabel',mechParamsNames(parsAxis))


%% 7.1b) CORRELATIONS INPUTS/OUTPUTS frequency
% =========================================================================
parsAxis = 1:length(mechParamsNames);
outAxis = 1:length(Dataset_FA.outputsEig(1,1:10));
Cor = zeros(length(parsAxis),length(outAxis));

fNames = {};
for ii = outAxis
    fNames{ii} = ['f_{', int2str(ii),'}'];
end

figure()
title('inputs vs eigenfrequencies correlation')
for m1 = parsAxis % Create correlations for each experimenter
 for m2 = outAxis % Correlate against each experimenter
  Cor(m1,m2) = abs(corr(Dataset_FA.inputs(:,m1), Dataset_FA.outputsEig(:,m2)));
 end
end
imagesc(Cor)
colorbar();
title('inputs vs frequencies correlation');
set(gca, 'fontSize', 15);
set(gca,'YTick',parsAxis,'YTickLabel',mechParamsNames(parsAxis))
set(gca,'XTick',outAxis,'XTickLabel',fNames)
corrFreq = array2table(Cor, 'variableNames', fNames, 'rowNames', mechParamsNames);


%% 7.1c) CORRELATIONS INPUTS/OUTPUTS amplitude
% =========================================================================

parsAxis = 1:length(mechParamsNames);
outAxis = 1:length(Dataset_FA.outputsAmp(1,1:10));
Cor = zeros(length(parsAxis),length(outAxis));

fNames = {};
for ii = outAxis
    fNames{ii} = ['a_{', int2str(ii),'}'];
end

figure()

for m1 = parsAxis % Create correlations for each experimenter
 for m2 = outAxis % Correlate against each experimenter
  Cor(m1,m2) = abs(corr(Dataset_FA.inputs(:,m1), db(abs(Dataset_FA.outputsAmp(:,Dataset_FA.modesIdxs(m2)))) ));
 end
end
imagesc(Cor(:,:))
colorbar();
title('inputs vs amplitudes correlation');
set(gca, 'fontSize', 15);
set(gca,'YTick',parsAxis,'YTickLabel',mechParamsNames(parsAxis))
set(gca,'XTick',outAxis,'XTickLabel',fNames)

corrAmp = array2table(Cor, 'variableNames', fNames, 'rowNames', mechParamsNames);

figure()
scatter(Dataset_FA.inputs(:,2),db( Dataset_FA.outputsAmp(:,8)));
xlabel('\alpha');
ylabel('a_1 |_{dB}');

