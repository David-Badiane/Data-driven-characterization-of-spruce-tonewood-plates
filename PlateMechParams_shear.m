clear all;
close all;


%% Initial setup

% Reference folders
baseFolder = pwd;
csvPath = [baseFolder, '\csv'];
addpath ([baseFolder, '\functions']);
addpath ([baseFolder, '\data']);
addpath(csvPath);

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
[Hv,f0, fLocs, csis, Q] = EMASimple(HvSVD, fAxis,1e-4, 3);
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
writetable(errRatesTable,'errRates.csv');
writetable(frequencyTable,'Eigenfrequencies.csv');
writetable(magnitudeTable,'Magnitude.csv');


%% 2) SET UP PARAMETERS FOR SIMULATIONS

% open Comsol model
model = mphopen('PlateMechParams');
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
%% NEW DATASET GENERATION
             
             
             
             
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
eigs = real([ ... ]);

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
    
model = mphopen('PlateMechParams_WithStudy');

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

model = mphopen('PlateMechParams_WithStudy');
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
    [HvComsol,f0Comsol, maxLocs, csisComsol, QComsol] = EMAPoly(vel, fAxis,1e-10, 1);
    
    Qs = [Q.'; QComsol(1:length(Q)).'];
    Qtable = array2table(Qs, 'variableNames',{'Q1' 'Q2' 'Q3' 'Q4' 'Q5' 'Q6' 'Q7' 'Q8' 'Q9' 'Q10'}, 'rowNames', {'Real' 'Comsol'});
    writeMat2File(Qs,'Qs.csv', {'Q'}, 1,false); 


