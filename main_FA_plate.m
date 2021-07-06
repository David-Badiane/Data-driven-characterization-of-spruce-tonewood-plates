clear all;
close all;


%% 0) Initial setup

% Reference folders
baseFolder = pwd;
csvName = 'csv_plate_FA_uniform';
convergencePath = [baseFolder, '\convergenceTest_plate'];
csvPath = [baseFolder, '\', csvName];
addpath ([baseFolder, '\functions']);
addpath ([baseFolder, '\data']);
addpath(csvPath);

comsolModel = 'woodenPlate';

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
[Hv,f0, fLocs, csis, Q] = EMASimple(HvSVD, fAxis,1e-4, 3,false);
f0 = f0(2:end);
fLocs = fLocs(2:end);
csis = csis(2:end);

% 3) Compute first guess of Ex
geom = infosMatrix(5,1:3);
rho = infosMatrix(5,end);
[mechParams, normParams] = computeParams([f0(1), f0(2), f0(3)],rho, geom)

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
                 0.467, 0.372, 0.435, 19.0001, 1e-6];
             
%% 3) NEW DATASET GENERATION
% setup variables 
simFolder = [csvPath,'\Modeshapes'];
mkdir(simFolder)
cd(baseFolder)
% set if you want to write new csv files
writeNow = false;
% number of simulations
nSim = 3000;
% Comsol model
model = mphopen(comsolModel);
% Comsol number of eigenfrequencies computed
nModes = 15;  

meshSize = 6;
model.mesh('mesh1').feature('size').set('hauto', int2str(meshSize));
model.mesh('mesh1').run;

% if 'uniform' - density is still gaussian - Poisson ratios too ? - alpha
% and beta
% 'rho' 'Ex' 'Ey' 'Ez' 'Gxy' 'Gyz' 'Gxz' 'vxy' 'vyz' 'vxz' 'alpha' 'beta'
standardDev = [2 2*ones(size(referenceVals(2:7))) 2*ones(size(referenceVals(8:10))) 2*ones(size(referenceVals(11:12))) ];  
                                
[Dataset_FA, inputsTable, outputsEigTable, outputsAmpTable] = comsolRoutineFreqAmp(model, nSim, nModes, referenceVals,...
                                   varyingParamsNames,  standardDev,  simFolder, csvPath, writeNow, 'uniform');
                       
%% 4) FETCH DATASET - (import csv )
mechParamsNames = {'\rho' 'E_x' 'E_y' 'E_z' 'G_{xy}' 'G_{yz}' 'G_{xz}' '\nu_{xy}' '\nu_{yz}' '\nu_{xz}' '\alpha' '\beta'};
Dataset_FA = struct('inputs',[] ,'outputsEig',[] ,'outputsAmp',[] );
Dataset_FA.inputs = table2array(readtable('inputs.csv'));
Dataset_FA.outputsEig = table2array(readtable('outputsEig.csv'));
Dataset_FA.outpdsetutsAmp = table2array(readtable('outputsAmp.csv'));
nTuples = length(Dataset_FA.inputs(:,1));


%% DATASET PRESELECTION - Parameters
variationRatio = Dataset_FA.inputs./referenceVals;

maskUp = variationRatio(:,2:3)< 5;
maskDown = variationRatio(:,2:3)> 0.5;

selectedIdxs = [];
for ii = 1:length(Dataset_FA.inputs(:,1))
    conditionUp = floor(mean(maskUp(ii,:) == ones(size(maskUp(ii,:)))));
    conditionDown = floor(mean(maskDown(ii,:) == ones(size(maskUp(ii,:)))));
    disp([conditionUp, conditionDown])
    if conditionUp & conditionDown
       selectedIdxs =[idxs; ii];  
    end
end

%% DATASET PRESELECTION - Poisson ratios

maskUp = Dataset_FA.inputs(:,8:10)< 0.1;
poissonIdxs = [];
for ii = 1:length(Dataset_FA.inputs(:,1))
    conditionUp = sum(maskUp(ii,:)) >= 2;
    disp([conditionUp])
    if conditionUp 
       poissonIdxs =[poissonIdxs; ii];  
    end
end




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
  Cor(m1,m2) = abs(corr(Dataset_FA.inputs(:,m1), db(abs(Dataset_FA.outputsAmp(:,m2))) ));
 end
end
imagesc(Cor)
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

%% 7.2) Neural networks
% var = 0.2;
% alphaIdxs = intersect(find(Dataset_FA.inputs(:,end-1) < (1+var)*referenceVals(end-1)), find(Dataset_FA.inputs(:,end-1) > (1-var)*referenceVals(end-1)))
% betaIdxs = intersect(find(Dataset_FA.inputs(:,end) < (1+var)*referenceVals(end)), find(Dataset_FA.inputs(:,end) > (1-var)*referenceVals(end)))
% 
% restrictIdx = intersect(alphaIdxs,betaIdxs); 

 nModes = 20;
% [fRegressors,fCoeffs, fR2, fErrors,fpredOutputs] = multilinearRegress(Dataset_FA.inputs,Dataset_FA.outputsEig, nModes, 'freq', referenceVals);
% [aRegressors,aCoeffs, R2, aErrors,aPredOutputs] = multilinearRegress(Dataset_FA.inputs,db(abs(Dataset_FA.outputsAmp)), nModes, 'amp', referenceVals);

nNeurons_frq = 10;
nLayers_frq = 1;

nNeurons_amp = 15;
nLayers_amp = 3;
nTrain = round(0.9*length(Dataset_FA.inputs(:,1)));
nTest = round(0.1*length(Dataset_FA.inputs(:,1)));

[freq_R2, freq_R2_NN, fNetVector, fNet, fTestIdxs] = ...
    NN_trainTest_ALL(Dataset_FA.inputs, Dataset_FA.outputsEig, nNeurons_frq, nLayers_frq, nTrain, nTest,'freq', 25);

[amp_R2, amp_R2_NN, aNetVector, ampNet, ampTestIdxs] = ...
    NN_trainTest_ALL(Dataset_FA.inputs, db(abs(Dataset_FA.outputsAmp)), nNeurons_amp, nLayers_amp,nTrain, nTest, 'amp',35);

% testing
[freq_R2a, freq_R2_NNa, fNetVectora, fNeta, fTestIdxsa] = ...
    NN_trainTest_ALL(Dataset_FA.inputs, Dataset_FA.outputsEig, nNeurons_frq, nLayers_frq, 550,100,'freq',25);

[freq_R2b, freq_R2_NNb, fNetVectorb, fNetb, fTestIdxsb] = ...
    NN_trainTest_ALL(Dataset_FA.inputs, Dataset_FA.outputsEig, nNeurons_frq, nLayers_frq, 550,100,'freq',25);

[freq_R2c, freq_R2_NNc, fNetVectorc, fNetc, fTestIdxsc] = ...
    NN_trainTest_ALL(Dataset_FA.inputs, Dataset_FA.outputsEig, nNeurons_frq, nLayers_frq, 550,100,'freq',25);


[amp_R2a, amp_R2_NNa, aNetVectora, ampNeta, ampTestIdxsa] = ...
    NN_trainTest_ALL(Dataset_FA.inputs, db(abs(Dataset_FA.outputsAmp)), nNeurons_amp, nLayers_amp, 550,100,'amp',35);

[amp_R2b, amp_R2_NNb, aNetVectorb, ampNetb, ampTestIdxsb] = ...
    NN_trainTest_ALL(Dataset_FA.inputs, db(abs(Dataset_FA.outputsAmp)), nNeurons_amp, nLayers_amp, 550,100,'amp',35);

[amp_R2c, amp_R2_NNc, aNetVectorc, ampNetc, ampTestIdxsc] = ...
    NN_trainTest_ALL(Dataset_FA.inputs, db(abs(Dataset_FA.outputsAmp)), nNeurons_amp, nLayers_amp, 550,100,'amp',35);

amp_R2 = table2array(amp_R2_NN); freq_R2 = table2array(freq_R2_NN);
a550a_R2 = table2array(amp_R2_NNa); f550a_R2 = table2array(freq_R2_NNa);
a550b_R2 = table2array(amp_R2_NNb); f550b_R2 = table2array(freq_R2_NNb);
a550c_R2 = table2array(amp_R2_NNc); f550c_R2 = table2array(freq_R2_NNc);

figure(112)
idx = 1:10;
plot(idx, amp_R2(idx),'-o', 'lineWidth',1.2);
hold on;
plot(idx, a550a_R2(idx), '-.x');
plot(idx, a550b_R2(idx), '-.x');
plot(idx, a550c_R2(idx), '-.x');
hold off;
ax = gca;
ax.XMinorTick = 'on';
ax.YMinorTick = 'on';
ax.FontSize = 16;
xlim([1 10]);
yy = ylabel('$R^2$'); set(yy, 'interpreter', 'latex');
xlabel('N-th eigenfrequency');
legend(['train: ' ,num2str(nTrain), ' test: ', num2str(nTest)], '550a' , '550b' , '550c');
title('amplitude R2')
grid on;

figure(122)
plot(idx, freq_R2(idx),'-o', 'lineWidth',1.2);
hold on;
plot(idx, f550a_R2(idx), '-.x');
plot(idx, f550b_R2(idx), '-.x');
plot(idx, f550c_R2(idx), '-.x');
hold off;
ax = gca;
ax.XMinorTick = 'on';
ax.YMinorTick = 'on';
ax.FontSize = 16;
xlim([1 10]);
legend(['train: ' ,num2str(nTrain), ' test: ', num2str(nTest)], '550a' , '550b' , '550c');
title('frequency R2');
grid on;
yy = ylabel('$R^2$'); set(yy, 'interpreter', 'latex');
xlabel('N-th eigenfrequency');

% % Let's try and see where are outliers
% figure()
% toRemove = cell(length(fRegressors),1);
% % select the modes which are flipping
% flipping = {'f12' 'f20'};
% for ii = 1:length(fRegressors)
%     
%     subplot(5,5,ii)
%     plotResiduals(fRegressors{ii});
%     hold on
%     maxResidual = max(abs(fRegressors{ii}.Residuals.Raw));
%     % select low R2 modes 
%     if ii == 3 || ii == 4       
%         exact_match_mask = strcmp(modesNames(:,3), flipping{ii-2});
%         toRemove{ii} = find(exact_match_mask); 
%     end
%     legend(['f',int2str(ii)])
% end
% 
% [fRegressors,fCoeffs, R2n, errors, predictedOutputs] = multRgrOutliers(inputsInfo,outputsALLInfo, toRemove, nModes);

%% 8) Minimization of the error 
% preallocate and set variables
nRealizations = 5;
plotData = true;
NpeaksAxis = 1:8;
freqMatrix = zeros(nRealizations,length(NpeaksAxis));
ampMatrix = zeros(nRealizations,length(NpeaksAxis)); 
parsMatrix = zeros(nRealizations,12);
gauss = randn(nRealizations,1);

% minimiziation process
for ii = 1:nRealizations
    density = rho*(1+0.05*gauss(ii));
    density = rho; % TO DELETE
    [xpar, map, f_out, amp_out, fval] = ...
        minimization_FA(fNet, ampNet, Dataset_FA, f0, fAmps,density, NpeaksAxis, plotData);
    freqMatrix(ii,:) = f_out(map).'; 
    ampMatrix(ii,:) = amp_out(map).';
    parsMatrix(ii,:) = xpar.';
end
 

meanFreq = mean(freqMatrix);
varFreq = var(freqMatrix);

figure() 
plot(1:length(f0(NpeaksAxis)), f0(NpeaksAxis), '-o');
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
finalOutTable = writeMat2File([meanMechParams; matlabStd*100],'Results.csv', varyingParamsNames, 10,true); 
    
%% 7.2) Check quality of the result  (eigenfrequency study with estimated params)

 for jj = 1:length(varyingParamsNames)
                model.param.set(varyingParamsNames(jj), meanMechParams(jj));
 end
%params = mphgetexpressions(model.param);
model.mesh('mesh1').run;
        model.study('std1').run(); 
evalFreqz = mpheval(model,'solid.freq','Dataset','dset1','edim',0,'selection',1);
eigenFreqz = real(evalFreqz.d1');

figure()
plot(1:length(f0(indexReal)), f0(indexReal), '-o');
hold on;
plot(1:length(eigenFreqz(indexComsol)), eigenFreqz(indexComsol),'-x');
xlim([1,length(eigenFreqz(indexComsol))]);
xlabel('mode number N');
ylabel('frequency [Hz]');
legend('real f0','Comsol - minimization');
title('Real Eigenfrequencies vs simulated - minimization f1-f6 , f8-f10');

writeMat2File(eigenFreqz,'EigenfreqzComsol.csv', {'f'}, 1,false); 

%% 7.3) EXPORT & COMPARE POINT RESPONSE OF FRF (frequency domain with estimated parameters)

% Perform modal analysis first

mechParams = table2array(readtable('Results7.csv'));

params = mphgetexpressions(model.param);
paramsNames = params(7:end,1);
freqAxNames = paramsNames(end-2:end);
fAxis = f(f <= fHigh);
freqVals = [fAxis(end), fAxis(2)-fAxis(1), fAxis(1)];

for ii = (1:length(mechParams))
    model.param.set(paramsNames(ii), mechParams(ii));
end

model.physics('solid').feature('lemm1').feature('dmp1').active(true)
model.component('comp1').physics('solid').feature('lemm1').feature('dmp1').set('DampingType', 'RayleighDamping');
model.component('comp1').physics('solid').feature('lemm1').feature('dmp1').set('InputParameters', 'AlphaBeta'); 

% b) Set Rayleigh constants
model.component('comp1').physics('solid').feature('lemm1').feature('dmp1').set('alpha_dM', 19);
model.component('comp1').physics('solid').feature('lemm1').feature('dmp1').set('beta_dK', 3e-6);

% c) Run study and mesh
model.component('comp1').mesh('mesh1').feature('size').set('hauto', 3);
model.mesh('mesh1').run;
model.study('std2').run();

% d) Export Results
dirName = pwd;
model.result.export('data1').set('transpose', true);
model.result.export('data1').set('sdim', 'fromdataset');
exportData(model,'cpt1', dirName,['velpar'],'solid.u_tZ'); % velocity  

%% e) Compare comsol FRF and Real FRF

[vel] = readTuples(['velCheck.txt'], 1, false);
vel = vel(4:end);
% h) Delete .txt files exported by Comsol
%delete(['vel',int2str(ii),'par.txt']);
fAxisComsol = 20:1:1200;
freqFloor = find(fAxis>=20);
fAxis = fAxis(freqFloor);
%vel = vel(freqFloor);
[maxVals, maxLocs] = findpeaks(abs(vel)/max(abs(vel)));
f0Comsol = fAxisComsol(maxLocs);
f0Comsol = f0Comsol(:);
Hvplot = Hv(freqFloor);
figure()
plot(fAxis, abs(Hvplot/max(abs(Hvplot))), 'LineWidth',1.5);
hold on;
plot( fAxisComsol, abs(vel)/max(abs(vel)), 'LineWidth',1.5);     
stem(f0, abs(Hv(fLocs)/max(abs(Hv(fLocs) ) ) ) );
stem(f0Comsol, abs(vel(maxLocs)/max(abs(vel(maxLocs)))));
eigs = real([ ]);

stem(eigs, ones(size(eigs)));

xlabel('f    [Hz]');
ylabel(' |H_v(f)| ');
legend('real FRF', 'simulated FRF','real peaks', 'Comsol peaks');
title('Real vs simulated FRF ');

toView = 1:8;
ratio = abs(Hv(fLocs(toView)))./abs(vel(maxLocs(toView))).';
factor = mean(ratio);
ampsComsol = factor * abs(vel(maxLocs(toView)));
ampsReal =  abs(Hv(fLocs(toView)));
factorAmp = mean( f0(toView)./(ampsReal) );
ampsComsol = factorAmp*ampsComsol;
ampsReal = factorAmp*ampsReal;

figure()
plot(f0Comsol(toView), ampsComsol , '.', 'markerSize', 10)
hold on;
xlabel('frequency');
ylabel('amplitude');
title('first 8 eigenfrequencies');
plot(f0(toView), ampsReal, '.', 'markerSize' ,10)

pointsComsol = [f0Comsol(toView), ampsComsol.'];
pointsReal = [f0(toView), ampsReal];

minimumDistantPoints = zeros(2, length(toView));

for ii = 1:length(toView)
    dist = sqrt(100*(pointsReal(ii,1) - pointsComsol(:,1)).^2 + (pointsReal(ii,2) - pointsComsol(:,2)).^2);
    [minDist, minLoc] = min(dist);
    minimumDistantPoints(:,ii) =  pointsComsol(minLoc,:);

    lineFreqz =  [f0(ii), f0Comsol(minLoc)];
    lineAmps = [ampsReal(ii), ampsComsol(minLoc)];
    plot(lineFreqz, lineAmps);
end
    legend('Comsol', 'experimental', 'f1', 'f2' , 'f3', 'f4', 'f5', 'f6','f7', 'f8');


%% Minimization 2 - narrow band frequency domain studies
    
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
    
    fAxis = 20:0.5:fHigh;
    [Hv,f0, fLocs, csis, Q] = EMASimple(HvSVD, fAxis,1e-3, 3);
    f0 = f0(2:end); csis = csis(2:end); Q = Q(2:end); fLocs = fLocs(2:end);
    %fAxis = 50:0.5:600;
    [HvComsol,f0Comsol, maxLocs, csisComsol, QComsol] = EMAPoly(vel, fAxis,1e-10, 1, false);
    
    Qs = [Q.'; QComsol(1:length(Q)).'];
    Qtable = array2table(Qs, 'variableNames',{'Q1' 'Q2' 'Q3' 'Q4' 'Q5' 'Q6' 'Q7' 'Q8' 'Q9' 'Q10'}, 'rowNames', {'Real' 'Comsol'});
    writeMat2File(Qs,'Qs.csv', {'Q'}, 1,false); 


