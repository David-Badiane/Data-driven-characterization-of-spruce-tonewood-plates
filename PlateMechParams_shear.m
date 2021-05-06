clear all;
close all;


%% Initial setup
csvPath = 'C:\Users\utente\.comsol\v56\llmatlab\csv3';
addpath 'C:\Users\utente\.comsol\v56\llmatlab\functions'
addpath 'C:\Users\utente\.comsol\v56\llmatlab\data'
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
[Hv,f0, fLocs, csis, Q] = EMASimple(HvSVD, fAxis,1e-4, 3);
f0 = f0(2:end);
fLocs = fLocs(2:end);
csis = csis(2:end);

% 3) Compute first guess of Ex
geom = infosMatrix(5,1:3);
rho = infosMatrix(5,end);
[mechParams, normParams] = computeParams([f0(1), f0(2), f0(3)],rho, geom)

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
            
%% 3) DATASET GENERATION - in mech params -- out eigenfrequencies mesh and modeshapes
% setup variables 
baseFolder = pwd;
simFolder = [baseFolder,'\Simulations3'];
cd(baseFolder)
nSim = 300;
model = mphopen('PlateMechParams');
outputsALLInfo = [];
outputsInfo = [];
inputsInfo = [];

% setup Comsol
model.component('comp1').physics('solid').feature('lemm1').feature('dmp1').active(false);
model.mesh('mesh1').feature('size').set('hauto', '1');
model.mesh('mesh1').run;
nModes = 20;
model.study('std1').feature('eig').set('neigs', int2str(nModes));
model.result.export('data1').set('data', 'dset1');

for ii = 153:nSim
    disp(ii)
    cd(simFolder);
    % 1) gaussian sample mechanical parameters
    if ii == 1
        for jj = 1:length(referenceVals)
            model.param.set(varyingParamsNames(jj), referenceVals(jj));
        end
        currentVals = referenceVals;
    else
        currentVals = createSetParams(model, referenceVals,standardDev, varyingParamsNames);
    end
    % 2) run eigenfrequency study
    model.study('std1').run();

    % 3.a) and save modeshapes
    modesFileName = 'solidDisp';
    expression = {'solid.disp'};
    exportAllModesFromDataset(model, modesFileName,simFolder,expression);
    fileData = readTuples([modesFileName,'.txt'], nModes+3, true);
    meshData =fileData(:,1:3);
    deformationData = fileData(:,4:nModes+3);
    delete([modesFileName,'.txt']); 

    writeMat2File(meshData,['mesh', int2str(ii),'.csv'], {'x' 'y' 'z'}, 3,true);
    writeMat2File(deformationData,['modeshapes', int2str(ii),'.csv'], {'disp f'}, 1, false);

    cd(csvPath)
    
      % 3.b) Evaluate eigenfrequencies
    evalFreqz = mpheval(model,'solid.freq','Dataset','dset1','edim',0,'selection',1);
    eigenFreqz = real(evalFreqz.d1');
    %presentEigenFreqz = [eigenFreqz(2),eigenFreqz(3),eigenFreqz(5),...
                         %eigenFreqz(6),eigenFreqz(10)];
    
    % 4) Extract old values 
    if ii ~=  1
        inputsInfo = table2array(readtable("inputs.csv"));
        %outputsInfo = table2array(readtable("outputs.csv"));
        outputsALLInfo = table2array(readtable("outputsALL.csv"));
    end
    
    % 5) Update results
    inputsInfo = [inputsInfo; currentVals];
    %outputsInfo = [outputsInfo; presentEigenFreqz];
    outputsALLInfo = [outputsALLInfo; eigenFreqz]
    
    % 6) Save results
    inputsTable  = writeMat2File(inputsInfo,'inputs.csv', varyingParamsNames(1:10), 10,true);   
    %outputsTable = writeMat2File(outputsInfo,'outputs.csv', eigenFreqzNames, 5,true);   
    outputsALLTable = writeMat2File(outputsALLInfo(:,1:nModes),'outputsALL.csv', {'f'}, 1,false);   
end

%% 4) LOOK FOR POISSON PLATES (if present)

L = Ls(1)
W = Ws(1)
aspectRatio = L/W;
inputsInfo = table2array(readtable("inputs.csv"));
poissonCheck = (inputsInfo(:,2)./inputsInfo(:,3)).^(1/4);
poissonPlates = intersect(find( poissonCheck > 0.99*aspectRatio),find( poissonCheck<1.01*aspectRatio));
length(poissonPlates)

for jj = 138:length(poissonPlates)
    if poissonPlates(jj) >=1
    ii = poissonPlates(jj);
    meshData = table2array(readtable(['mesh',int2str(ii),'.csv']));
    modesData = table2array(readtable(['modeshapes',int2str(ii),'.csv']));
    figure()
    plot3(meshData(:,1), meshData(:,2), modesData(:,4) ,'.', 'MarkerSize', 5);
    end

end

%% 5.1 ) ANALYZE MESH AND MODESHAPES - OBTAIN THEIR NAMES - Necessary for loss function

%cd(simFolder)
 nModes = 20;
 nSim = 120;
 
% names obtained (if you want to obtain them again, just uncomment what is
% commented )

%modesNames = table2array(readtable("modesNames2.csv"));

modesNames = cell(nSim,nModes);
minimumThreshold = 1e-7;
for ii = 1:nSim
    meshData = table2array(readtable(['mesh',int2str(ii),'.csv']));
    modesData = table2array(readtable(['modeshapes',int2str(ii),'.csv']));
    [modesNamesSim] = recognizeModes(meshData,modesData,minimumThreshold^-1, ii );
    modesNames(ii,1:nModes) = modesNamesSim;
end


cd('C:\Users\utente\.comsol\v56\llmatlab\csv')

% comment this if you want to obtain modeshape names
namesTable  = writeMat2File(modesNames,'modesNames3.csv', {'f'}, 1,false);


%% 5.2) Check the results from modeshapes analysis to find Poisson plates ( ignore, already done in 4) )

ordering = zeros(nSim,1);

for ii = 1:length(ordering)
%    if [modesNames{ii,1:end}] ~= [modesNames{1,1:end}] 
%            ordering(ii) = 0;
%    else
      
   for jj = 1:2
    if modesNames{ii,jj} == 'not'
        ordering(ii) = 2; 
    end
%    end
       
   end
   if ordering(ii) == 2
       figure()
       meshData = table2array(readtable(['mesh',int2str(ii),'.csv']));
       modesData = table2array(readtable(['modeshapes',int2str(ii),'.csv']));
       plot3(meshData(:,1), meshData(:,2), modesData(:,4) ,'.', 'MarkerSize', 10);
   end
   
end

toDeleteIndexes = find(ordering ~= 0 );

% if there are poisson Plates ( now there are not )
cleanOutputsALL =  outputsALLInfo;
cleanOutputsALL (toDeleteIndexes,:) = [];
cleanInputsInfo = inputsInfo;
cleanInputsInfo(toDeleteIndexes,:) = [];

%% 5.3) Check how many time each tuple of modesNames appears in modes Names (check the ordering of modeshapes)

%modesNames = table2cell(readtable('modesNames2.csv'));
appears = zeros(length(modesNames(:,1)),nModes);
for jj = 1:length(modesNames(:,1))
    for ii =  1:length(appears(1,:))
        exact_match_mask = strcmp(modesNames(:,ii), modesNames{jj,ii});
        appears(jj,ii) = length(find(exact_match_mask));
    end
end

[maxVal, maxLoc] = max(mean(appears.'));


 
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
[linearModels,multilinCoeffs, R2, errors, predictedOutputs] = multilinearRegress(inputsInfo,outputsALLInfo,nModes);
%[linearModels,multilinCoeffs, R2, errors, predictedOutputs] = multilinearRegress(cleanInputsInfo,cleanOutputsALL);

% Let's try and see where are outliers
figure()
toRemove = cell(length(linearModels),1);
for ii = 1:length(linearModels)
    subplot(5,5,ii)
    plotResiduals(linearModels{ii});
    hold on
    maxResidual = max(abs(linearModels{ii}.Residuals.Raw));
    if maxResidual > 10
    limit = 0.3;
    else
        limit = 0.9;
    end
    toRemove{ii} = find(abs(linearModels{ii}.Residuals.Raw) > limit*maxResidual);
    legend(['f',int2str(ii)])
end

[linearModels,multilinCoeffs, R2n, errors, predictedOutputs] = multRgrOutliers(inputsInfo,outputsALLInfo, toRemove, nModes);
%[linearModels,multilinCoeffs, R2, errors, predictedOutputs] = multRgrOutliers(cleanInputsInfo,cleanOutputsALL, toRemove);

%% 6.3) See how each frequency is correlated to the mechanical parameters

normMultilinCoeffs = multilinCoeffs(2:end,:).*referenceVals';
figure()
imagesc(abs(normMultilinCoeffs(2:7,1:10)));
colorbar()
names = {'rho' 'Ex' 'Ey' 'Ez' 'Gxy' 'Gyz' 'Gxz' 'vxy' 'vyz' 'vxz'};
% names = { 'Ex' 'Ey' 'Ez' 'Gxy' 'Gyz' 'Gxz'  };
freqz = {'f11' 'f02' 'f20' 'f12' 'f21'	'f03' 'f22'	'f30' 'f13'	'f31'};
freqz = {'f11' 'f02' 'f20' 'f12' 'f21'	'f03' 'f22'	'f30' };
axis off;
hold on
for ii = 1:max([length(names), length(freqz)])
    if ii <= length(freqz)
            text(ii,1, freqz{ii});
    end
    if ii <= length(names)
        text(0,ii, names{ii});
    end
end

%% 7.1) MINIMIZATION OF THE ERROR 

% 1) first guess of mech parameters
errors = zeros(size(outputsALLInfo(:,1)));
indexComsol = [1,2,4,3,5,7,9,10,11,14,15,16];
indexReal = 1:12;
for ii = 1:length(outputsALLInfo(:,1))
    errors(ii) = norm((f0(indexReal)-outputsALLInfo(ii,indexComsol).')./f0(indexReal),2);
end
[minVals, minimumLoc] = min(errors);

% 2) setup for minimization
nRealizations = 20;
% Check section 4, using where appears most
maxLoc;

% I put those here to have a reference of the map we are using
comsolIndex = [1,2,4,3,5,7,9,10,12,14,15,16].';
realIndex =   [1,2,3,4,5,6,7,8,9,10,11,12].';
% actual indexes used for the minimization
indexComsol = [1,2,4,15].';
indexReal = [1,2,3,11].';

freqMatrix = zeros(nRealizations,length(indexReal)); 
parsMatrix = zeros(nRealizations,10);
gauss = randn(nRealizations,1);

% 3) minimization process
for ii = 1:nRealizations
    density = rho*(1+0.05*gauss(ii));
    [xpar, f_out, fval] = minimizeError(linearModels, inputsInfo, outputsALLInfo,...
                                        f0,minimumLoc,density, indexComsol, indexReal, false);
    diff = (f0(realIndex) - f_out(comsolIndex))./f0(realIndex);                               
    err = mean(abs(diff))*100
% If you want to see the figure respresenting the estimation, uncomment it 
    figure()
    plot(realIndex, f_out(comsolIndex), '-o');
    hold on 
    plot(realIndex, f0(realIndex), '-x');
    xlabel('N mode')
    ylabel(' f     [Hz]');
    legend('fMultilin', 'fexp');   

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
varyingParamsNames = {'rho', 'Ex', 'Ey', 'Ez', 'Gxy', 'Gyz', 'Gxz', 'vxy', 'vyz', 'vxz'};
finalOut = array2table(meanMechParams, 'VariableNames', varyingParamsNames);

finalStdTable = array2table(finalStd*100, 'VariableNames', {'std rho [%]', 'std Ex [%]', 'std Ey [%]', 'std Ez [%]',...
                    'std Gxy [%]', 'std Gyz [%]', 'std Gxz [%]', 'std vxy [%]', 'std vyz [%]', 'std vxz [%]'});
                              
matlabStd = std(parsMatrix)./meanMechParams;
myStdTable = array2table(finalStd*100, 'VariableNames', {'std rho [%]', 'std Ex [%]', 'std Ey [%]', 'std Ez [%]',...
                    'std Gxy [%]', 'std Gyz [%]', 'std Gxz [%]', 'std vxy [%]', 'std vyz [%]', 'std vxz [%]'});
matlabStdTable = array2table(matlabStd*100, 'VariableNames', {'std rho [%]', 'std Ex [%]', 'std Ey [%]', 'std Ez [%]',...
'std Gxy [%]', 'std Gyz [%]', 'std Gxz [%]', 'std vxy [%]', 'std vyz [%]', 'std vxz [%]'});


finalOutTable = writeMat2File([meanMechParams; matlabStd*100],'Results.csv', varyingParamsNames, 10,true); 
writeMat2File(parsMatrix,'ResultsALL.csv', varyingParamsNames, 10,true);

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

%% e) Read Data on Matlab and FRF comparison
[vel] = readTuples(['velf7.txt'], 1, false);
vel = vel(4:end);
% h) Delete .txt files exported by Comsol
%delete(['vel',int2str(ii),'par.txt']);
fAxisComsol = 20:0.5:600;
ind = find(fAxis>=20);
fAxis = fAxis(ind);
%vel = vel(ind);
[maxVals, maxLocs] = findpeaks(abs(vel)/max(abs(vel)));
f0Comsol = fAxisComsol(maxLocs);
f0Comsol = f0Comsol(:);
Hvplot = Hv(ind);
figure()
plot(fAxis, abs(Hvplot/max(abs(Hvplot))), 'LineWidth',1.5);
hold on;
plot( fAxisComsol, abs(vel)/max(abs(vel)), 'LineWidth',1.5);     
stem(f0, abs(Hv(fLocs)/max(abs(Hv(fLocs) ) ) ) );
stem(f0Comsol, abs(vel(maxLocs)/max(abs(vel(maxLocs)))));
eigs = real([59.825+1.5991i
            102.62+1.6647i
            135.53+1.7385i
            157.91+1.8004i
            178.69+1.8664i
            277.82+2.2929i
            291.46+2.3660i
            326.79+2.5719i
            372.22+2.8713i
            409.89+3.1490i
            464.85+3.6020i
            516.20+4.0769i
            525.28+4.1660i
            570.73+4.6356i
            688.73+6.0364i
            699.94+6.1831i
            729.89+6.5867i
            760.56+7.0177i
            830.41+8.0651i
            858.67+8.5148i]);

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
legend('Comsol', 'experimental');

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

%% Convergence test for choosing correct mesh

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

%% Minimization 2 - narrow band frequency domain studies
    
model = mphopen('PlateMechParams_WithStudy');

fAmps = abs(Hv(fLocs));

mechParameters = table2array(readtable('Results7.csv'));
mechParameters_1stguess = mechParameters(1,:);
params = mphgetexpressions(model.param);
paramsNames = params(7:end,1);

model.mesh('mesh1').feature('size').set('hauto', '6');
model.mesh('mesh1').run;

    for ii = (1:length(mechParameters_1stguess))
        model.param.set(paramsNames(ii), mechParameters_1stguess(ii));
    end
         
constraintWeight = 0.05;
fun = @(x) minimizationComsolBased(x,  model, f0, fAmps, constraintWeight, paramsNames, mechParameters_1stguess);

options = optimset('fminsearch');
options = optimset(options, 'TolFun',1e-8,'TolX',1e-8, 'MaxFunEvals',5e3,'MaxIter', 5e3,...
    'DiffMinChange', 1, 'DiffMaxChange', 200); 
 
[xpar,fval, exitflag, output] = fminsearch(fun, mechParameters_1stguess, options);


    %% 7.4) check the Q factor values for simulated and real FRFs
    
    fAxis = 20:0.5:fHigh;
    [Hv,f0, fLocs, csis, Q] = EMASimple(HvSVD, fAxis,1e-3, 3);
    f0 = f0(2:end); csis = csis(2:end); Q = Q(2:end); fLocs = fLocs(2:end);
    %fAxis = 50:0.5:600;
    [HvComsol,f0Comsol, maxLocs, csisComsol, QComsol] = EMAPoly(vel, fAxis,1e-10, 1);
    
    Qs = [Q.'; QComsol(1:length(Q)).'];
    Qtable = array2table(Qs, 'variableNames',{'Q1' 'Q2' 'Q3' 'Q4' 'Q5' 'Q6' 'Q7' 'Q8' 'Q9' 'Q10'}, 'rowNames', {'Real' 'Comsol'});
    writeMat2File(Qs,'Qs.csv', {'Q'}, 1,false); 

%% 8.1) STUDY MINIMIZATION ERROR

% setup
comsolIndex = [1,2,4,3,5,7,9,10,12,14,15,16].';
realIndex = [1,2,3,4,5,6,7,8,9,10,11,12].';
presentModes = {'f11' 'f02' 'f20' 'f12' 'f21' 'f22' 'f30' 'f31' 'f32' 'f05' 'f15' };

errors = zeros(size(outputsALLInfo(:,1)));
for ii = 1:length(outputsALLInfo(:,1))
    errors(ii) = norm((f0(realIndex)-outputsALLInfo(ii,comsolIndex).')./f0(realIndex),2);
end
[minVals, minimumLoc] = min(errors);

errs = cell(11,1);
params = cell(11,1);
freqz = cell(11,1);
nFreq =  length(comsolIndex);
nMP = 10;

% minimization with permutations of the indexes
for jj = [4,5,8,10,11]

    comsolPermute = nchoosek(comsolIndex,jj);
    disp(length(comsolPermute(:,1)));
    realPermute = nchoosek(realIndex,jj);

    % preallocate
    errs{jj} = cell(length(comsolPermute(:,1)),jj+2);
    params{jj} = cell(length(comsolPermute(:,1)), jj + nMP );
    freqz{jj} = cell(length(comsolPermute(:,1))+1, jj+nFreq);

    freqz{jj}(1,1:nFreq) = num2cell(f0(realIndex));

    for ii = 1:length(comsolPermute(:,1))
        
        [xpar, f_out, fval] = minimizeError(linearModels, inputsInfo, outputsALLInfo,...
                                            f0,minimumLoc,rho, comsolPermute(ii,:), realPermute(ii,:), false);
        % error                                
        diff = (f0(realIndex) - f_out(comsolIndex))./f0(realIndex);                                
        errs{jj}{ii,1} = mean(abs(diff))*100; 
        % mech Params and eigenfrequencies
        params{jj}(ii,1:nMP) = num2cell(xpar);
        freqz{jj}(ii+1,1:nFreq) = num2cell(f_out(comsolIndex).');
        
        % comment the figure for speed
%         figure()
%         plot(realIndex, f_out(comsolIndex), '-o');
%         hold on 
%         plot(realIndex, f0(realIndex), '-x');
%         xlabel('N mode')
%         ylabel(' f     [Hz]');
%         legend('fMultilin', 'fexp');
        
        % labeling for used modeshapes in the minimization
        for kk = 1:length(comsolPermute(ii,:))
            errs{jj}{ii,kk+1} = modesNames{1,comsolPermute(ii,kk)}; 
            params{jj}{ii, kk+ nMP} = modesNames{1,comsolPermute(ii,kk)};
            freqz{jj}{ii+1, kk+ nFreq} = modesNames{1,comsolPermute(ii,kk)};
        end
        
        % label if there are negative mech parameters or not
        check = find(xpar<0);
        errLength = length(errs{jj}(ii,:));
        if isempty(check)
            errs{jj}{ii, end}  = 'positive MP';        
        else
            errs{jj}{ii, end} = 'negative MP';
        end
        
        % check results at each iteration
        params{jj}(ii,:)
        freqz{jj}(ii+1,:)
        errs{jj}(ii,:)
    end
end


%% 8.2) Save results from minimization study
% save mechParams and eigenfrequencies
for ii = 1:5
    varyingParamsNames = {'rho', 'Ex', 'Ey', 'Ez', 'Gxy', 'Gyz', 'Gxz', 'vxy', 'vyz', 'vxz'};
    freqzNames = {'f1' 'f2' 'f3' 'f4' 'f5' 'f6' 'f7' 'f8' 'f9' 'f10' 'f11' 'f12'};
    N = length(varyingParamsNames);
    M = length(freqzNames);
    for tt = 1:ii
    varyingParamsNames{N+tt} = ['used f', int2str(tt)];
    freqzNames{M+tt} = ['used f', int2str(tt)];
    end

    writeMat2File(paramms{ii},['mechParams',int2str(ii),'.csv'], varyingParamsNames, length(varyingParamsNames),true);     
    writeMat2File(freqqz{ii},['eigenfreqz',int2str(ii),'.csv'], freqzNames, length(freqzNames),true);   
end

% save errors
errs4Table = sortrows(cell2table(errs{4}, 'VariableNames', {'err [%]', 'used f1','used f2','used f3', 'used f4', 'Mech params'}));
errs5Table = sortrows(cell2table(errs{5}, 'VariableNames', {'err [%]', 'used f1','used f2','used f3', 'used f4','used f5', 'Mech params'}));
errs8Table = sortrows(cell2table(errs{8}, 'VariableNames', {'err [%]', 'used f1','used f2','used f3', 'used f4','used f5', 'used f6', 'used f7', 'used f8', 'Mech params'}));
errs10Table = sortrows(cell2table(errs{10}, 'VariableNames', {'err [%]', 'used f1','used f2','used f3', 'used f4','used f5', 'used f6', 'used f7', 'used f8', 'used f9', 'used f10', 'Mech params' }));
errs11Table = sortrows(cell2table(errs{11}, 'VariableNames', {'err [%]',  'used f1','used f2','used f3', 'used f4','used f5', 'used f6', 'used f7', 'used f8', 'used f9', 'used f10', 'used f11' , 'Mech params' }));

writetable(errs4Table,'errs4.csv');
writetable(errs5Table,'errs5.csv');
writetable(errs8Table,'errs8.csv');
writetable(errs10Table,'errs10.csv');
writetable(errs11Table,'errs11.csv');

