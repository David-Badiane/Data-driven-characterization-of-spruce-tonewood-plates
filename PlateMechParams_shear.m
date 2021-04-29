clear all;
close all;

csvPath = 'C:\Users\utente\.comsol\v56\llmatlab\csv2';
addpath 'C:\Users\utente\.comsol\v56\llmatlab\functions'
addpath 'C:\Users\utente\.comsol\v56\llmatlab\data'
addpath(csvPath);
addpath 'data'

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
jj = 1;
sampleNames = {'7b'};

fHigh = 600;
M = 10;
thresholdPerc = 30;
idx = 1;
load('07_01_21_frf_anima_7b_fronte_G_post');

% 1) SVD
fAxis = f(f <= fHigh);
[HvSVD,threshold,singularVals] = SVD(H1_v, fAxis, M, thresholdPerc, sampleNames, jj);
    
% 2) MODAL ANALYSIS
[Hv,f0, fLocs, csis, Q] = EMASimple(HvSVD, fAxis,1e-3, 3);
f0 = f0(2:end);
csis = csis(2:end);

geom = infosMatrix(5,1:3);
rho = infosMatrix(5,end);
[mechParams, normParams] = computeParams([f0(1), f0(2), f0(5)],rho, geom);

% figure()
% semilogy(fAxis, abs(Hv)/max(abs(Hv)), fAxiss, abs(Hvv/max(abs(Hvv))),'LineWidth', 1.3);
% legend('no shear info', ' with shear info');
% xlabel('f    [Hz]');
% ylabel('|H_v(f)| ');
% title('|H_v(f)| with and without shear modes');

%% 2) SET UP PARAMETERS FOR SIMULATIONS

model = mphopen('PlateMechParams');
% Parameters setup
params = mphgetexpressions(model.param);                  % get initial parameters                          
% get parameters names
varyingParamsNames = params(7:end,1);
steadyParamsNames = params(1:6,1);
pastVals = readTuples('inputs.csv', 9, false);

geomSet = [infosMatrix(5,1:6), infosMatrix(5,end)];
setParams = cell(length(steadyParamsNames)+1,1);
setParams(1:end-1) = steadyParamsNames;
setParams{end} = 'rho'; 
for jj = 1:length(setParams)
                model.param.set(setParams(jj), geomSet(jj));
end
%Ex = 2.128363746952024e10;
%Ey = 9.3136e+08;
%Ey = 1.1992e+09;
Ex = mechParams(1);
             
referenceVals = [rho, Ex, Ex*0.078, Ex*0.043,...
                 Ex*0.061, Ex*0.064, Ex*0.003,...
                 0.467, 0.372, 0.435];
             
referenceVals = [rho, Ex, mechParams(2), Ex*0.043,...
                 mechParams(3), Ex*0.064, Ex*0.003,...
                 0.467, 0.372, 0.435];
             
eigenFreqzNames = {'f2' 'f3' 'f5' 'f6' 'f10'};             

for ii = (1:length(steadyParamsNames))
    model.param.set(steadyParamsNames(ii), comsolParams(ii));
end
nSimulations = 50;

%% 3) DATASET GENERATION - in mech params -- out eigenfrequencies mesh and modeshapes
baseFolder = pwd;
simFolder = [baseFolder,'\Simulations'];

cd(baseFolder)
nSim = 200;
inputsInfo = table2array(readtable("inputs.csv"));
model = mphopen('PlateMechParams');
outputsALLInfo = [];
outputsInfo = [];
inputsInfo = [];

model.component('comp1').physics('solid').feature('lemm1').feature('dmp1').active(false);
model.mesh('mesh1').feature('size').set('hauto', '1');
model.mesh('mesh1').run;
nModes = 23;
model.study('std1').feature('eig').set('neigs', int2str(nModes));
model.result.export('data1').set('data', 'dset1');

for ii = 1:nSim
        cd(simFolder);
        if ii == 1
            for jj = 1:length(referenceVals)
                model.param.set(varyingParamsNames(jj), referenceVals(jj));
            end
            currentVals = referenceVals;
        else
            currentVals = createSetParams(model, referenceVals,standardDev, varyingParamsNames);
        end
        
        
        model.study('std1').run(); 
        
        modesFileName = 'solidDisp';
        expression = {'solid.disp'};
        exportAllModesFromDataset(model, modesFileName,simFolder,expression);
        fileData = readTuples([modesFileName,'.txt'], nModes+3, true);
        meshData =fileData(:,1:3);
        deformationData = fileData(:,4:nModes);
        delete([modesFileName,'.txt']); 
        
        writeMat2File(meshData,['mesh', int2str(ii),'.csv'], {'x' 'y' 'z'}, 3,true);
        writeMat2File(deformationData,['modeshapes', int2str(ii),'.csv'], {'disp f'}, 1, false);
    
    cd(csvPath)
    
      % 3) Evaluate eigenfrequencies
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
    outputsALLInfo = [outputsALLInfo; eigenFreqz];
    
    % 6) Save results
    inputsTable  = writeMat2File(inputsInfo,'inputs.csv', varyingParamsNames(1:10), 10,true);   
    %outputsTable = writeMat2File(outputsInfo,'outputs.csv', eigenFreqzNames, 5,true);   
    outputsALLTable = writeMat2File(outputsALLInfo,'outputsALL.csv', {'f'}, 1,false);   

end

% outputsALLTable = writeMat2File(eigenInfos,'outputsALL.csv', {'f'}, 1,false)

figure()
plot(fAxis, abs(Hv)/max(abs(Hv)));
hold on;
stem(eigenFreqz, ones(size(eigenFreqz)));
xlim([20,550]);
xlabel('f    [Hz]');
ylabel('|H_v(f)|');
legend('H_v/max(H_v)', 'Comsol fout');
title('FRF for sample 7b and comsol eigenfrequencies with same density, Simeon formula E_L, E_R, G_{LR}')


%% 4 ) ANALYZE MESH AND MODESHAPES - OBTAIN THEIR NAMES
 cd(simFolder)
 nModes = 23;
 nSim = 200;
 
% names obtaineeed

%modesNames = table2array(readtable("modesNames.csv"));

modesNames = cell(nSim,nModes);
minimumThreshold = 1e-7;
for ii = 1:200
    meshData = table2array(readtable(['mesh',int2str(ii),'.csv']));
    modesData = table2array(readtable(['modeshapes',int2str(ii),'.csv']));
    [modesNamesSim] = recognizeModes(meshData,modesData,minimumThreshold^-1, ii );
    modesNames(ii,1:nModes) = modesNamesSim;
end

cd('C:\Users\utente\.comsol\v56\llmatlab\csv2')
namesTable  = writeMat2File(modesNames,'modesNames.csv', {'f'}, 1,false);

% modesNames2 = modesNames;
% for ii = 1:length(outliers)
%     meshData = table2array(readtable(['mesh',int2str(outliers(ii)),'.csv']));
%     modesData = table2array(readtable(['modeshapes',int2str(outliers(ii)),'.csv']));
%     [modesNamesSim] = recognizeModes(meshData,modesData,minimumThreshold^-1, ii )
%     
%     notNumber = input('how many NOT are there? ');
%     for jj = 1:notNumber
%         str = input('replace not with...','s');
%         index = input('not index...');
%         modesNames2{jj,index} = str;
%     end  
% end
% 
% 
% ordering = zeros(nSim,1);
% 
% for ii = 1:length(ordering)
%    if [modesNames{ii,1:end}] ~= [modesNames{1,1:end}] 
%            ordering(ii) = 1;
%    else
%        
%    for jj = 1:10
%     if modesNames{ii,jj} == 'not'
%         ordering(ii) = 2; 
%     end
%    end
%        
%    end
% end

toDeleteIndexes = find(ordering ~= 0 );

cleanOutputsALL =  outputsALLInfo;
cleanOutputsALL (toDeleteIndexes,:) = [];
cleanInputsInfo = inputsInfo;
cleanInputsInfo(toDeleteIndexes,:) = [];


%% SEE MODES
modeNumber = 2;
%tuples = [78 90 211 3  296 509 516 199];
tuples = [ 3  296 509 516 ];
mins = seeModes( 1e7, 3, tuples);
%% computation of minimum position and max value
tuples = 1:550;
maxLeft = zeros(length(tuples),1);
minPos  = zeros(length(tuples),1);
modeNumber = 2;

for ii = 1:length(tuples)
    meshData = table2array(readtable(['mesh',int2str(tuples(ii)),'.csv']));
    modesData = table2array(readtable(['modeshapes',int2str(tuples(ii)),'.csv']));

    idxX = find(meshData(:,2) == 0);                        % y=0 ---> x axis
    edgeX = lowpass(modesData(idxX,modeNumber), 0.65);
    minLocsX = find(edgeX == min(edgeX(1:floor(length(edgeX)/2)) ));
    
    
    %[minValsX,minLocsX] = findpeaks(1./edgeX, 'MinPeakWidth', 2,'MinPeakProminence', 1e7);

    xPoints =1:length(edgeX);
%     figure(4)
%     plot(xPoints, edgeX, xPoints(minLocsX),edgeX(minLocsX),'r*');
%     xlabel('N element in the edge')
%     ylabel('solid.disp')
%     title(['mode', num2str(modeNumber), '  y = 0   tuple ', num2str(tuples(ii))])
%     
    if isempty(minLocsX)~= 1
        minPos(ii) = minLocsX(1)/xPoints(end);  
    else
        minPos(ii) = 0;
    end 
end



%% Check scatter or information from modeshapes
v = inputsInfo(:,8:10);

idxs = find(minPos == 0);
v(idxs,:) = [];
minPos(idxs) = [];
maxLeft(idxs) = [];

v = inputsInfo(:, 2)
v(idxs,:) = [];
figure()

subplot(2,1,1)
scatter(v(:,1), maxLeft);
hold on 
xlabel('vxy/vyz')
ylabel('value of the peak')
subplot(212)
scatter(v(:,1), minPos);
hold on 
xlabel('vxy/vyz')
ylabel('normalized minimum position')

figure()

for ii = 1:5
subplot(3,2,ii)
scatter(outputsInfo(:,ii), minPos);
hold on 
xlabel(['f',num2str(ii)])
ylabel('norm min pos')

end 
%% CORRELATIONS OF INPUTS
Cor = zeros(10,10);
for m1 = 1:10 % Create correlations for each experimenter
 for m2 = 1:10 % Correlate against each experimenter
  Cor(m2,m1) = abs(corr(inputsInfo(:,m1),inputsInfo(:,m2)));
 end
end
imagesc(Cor)
colorbar();


%% ACTUAL MULTILINEAR REGRESSION


%% FROM ZERO

%[linearModels,multilinCoeffs, R2, errors, predictedOutputs] = multilinearRegress(inputsInfo,outputsALLInfo);
[linearModels,multilinCoeffs, R2, errors, predictedOutputs] = multilinearRegress(cleanInputsInfo,cleanOutputsALL);

% Let's try and see where are outliers
figure()
toRemove = cell(length(linearModels),1);
for ii = 1:length(linearModels)
    subplot(5,2,ii)
    plotResiduals(linearModels{ii});
    hold on
    maxResidual = max(linearModels{ii}.Residuals.Raw);
    if maxResidual > 10
    limit = 0.35;
    else
        limit = 1.2;
    end
    toRemove{ii} = find(linearModels{ii}.Residuals.Raw > limit*maxResidual);
    legend(['f',int2str(ii)])
end

%[linearModels,multilinCoeffs, R2, errors, predictedOutputs] = multRgrOutliers(inputsInfo,outputsALLInfo, toRemove);
[linearModels,multilinCoeffs, R2, errors, predictedOutputs] = multRgrOutliers(cleanInputsInfo,cleanOutputsALL, toRemove);

%%
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

%% SEE SCATTERPLOTS
figure()
scatter(outputsALLInfo(:,1), inputsInfo(:,8));
xlabel('f_{11}')
ylabel('v_{LR}')



%% MINIMIZATION OF THE ERROR 
errors = zeros(size(outputsALLInfo(:,1)));
% indexComsol = [1,2,3,4,5,6,8,9,10];
% indexReal = 1:9;
indexComsol = [1,2,3,4,5,7,9,10];
indexReal = 1:8;

for ii = 1:length(outputsALLInfo(:,1))
    errors(ii) = norm((f0(indexReal)-outputsALLInfo(ii,indexComsol).')./f0(indexReal),2);
end
[minVals, minimumLoc] = min(errors);

nRealizations = 10;
freqMatrix = zeros(nRealizations,10); 
parsMatrix = zeros(nRealizations,10);
gauss = randn(nRealizations,1);


for ii = 1:nRealizations
    density = rho*(1+0.05*gauss(ii));
    [xpar, f_out, fval] = minimizeError(linearModels, inputsInfo, outputsALLInfo,...
                                        f0,minimumLoc,density, indexComsol, indexReal, true);
    freqMatrix(ii,:) = f_out.'; 
    parsMatrix(ii,:) = xpar.';
end
    
freqMatrixDisplay = freqMatrix(:,indexComsol);
meanFreq = mean(freqMatrixDisplay);
varFreq = var(freqMatrixDisplay);

figure() 
plot(1:length(f0(indexReal)), f0(indexReal), '-o');
legend('fexp');
hold on 
errorbar(1:length(meanFreq),meanFreq,varFreq)
legend('fexp','fopt');
xlabel('mode number  N' );
ylabel('frequency    [Hz]');


meanMechParams = mean(parsMatrix);
varianceMechParams1 = var(parsMatrix);


finalStd = sqrt(mean( abs( parsMatrix - meanMechParams).^2 ))./meanMechParams;

finalOut = array2table(meanMechParams, 'VariableNames', {'rho', 'Ex', 'Ey', 'Ez', 'Gxy', 'Gyz', 'Gxz', 'vxy', 'vyz', 'vxz'});

finalStdTable = array2table(finalStd*100, 'VariableNames', {'std rho [%]', 'std Ex [%]', 'std Ey [%]', 'std Ez [%]',...
                    'std Gxy [%]', 'std Gyz [%]', 'std Gxz [%]', 'std vxy [%]', 'std vyz [%]', 'std vxz [%]'});
                
                
matlabStd = std(parsMatrix)./meanMechParams;
myStdTable = array2table(finalStd*100, 'VariableNames', {'std rho [%]', 'std Ex [%]', 'std Ey [%]', 'std Ez [%]',...
                    'std Gxy [%]', 'std Gyz [%]', 'std Gxz [%]', 'std vxy [%]', 'std vyz [%]', 'std vxz [%]'});
matlabStdTable = array2table(matlabStd*100, 'VariableNames', {'std rho [%]', 'std Ex [%]', 'std Ey [%]', 'std Ez [%]',...
'std Gxy [%]', 'std Gyz [%]', 'std Gxz [%]', 'std vxy [%]', 'std vyz [%]', 'std vxz [%]'});


writeMat2File([meanMechParams; matlabStd*100],'Results68.csv', varyingParamsNames, 10,true); 
writeMat2File(parsMatrix,'ResultsALL68.csv', varyingParamsNames, 10,true);
%a = array2table(compare, 'VariableNames', {'f1' 'f2' 'f3' 'f4' 'f5' 'f7' 'f9' 'f10'}, 'RowNames', {'minimization' 'Comsol'})

%% Check quality of the result  (eigenfrequency study)

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
%% EXPORT & COMPARE POINT RESPONSE OF FRF (frequency domain)

% Perform modal analysis first

valuesStudy = meanMechParams;
params = mphgetexpressions(model.param);
paramsNames = params(7:end,1);
freqAxNames = paramsNames(end-2:end);
fAxis = f(f <= fHigh);
freqVals = [fAxis(end), fAxis(2)-fAxis(1), fAxis(1)];


    for ii = (1:length(valuesStudy))
        model.param.set(paramsNames(ii), valuesStudy(ii));
    end
    for ii = (1:length(freqVals))
        model.param.set(freqAxNames(ii), freqVals(ii));
    end
    
    model.physics('solid').feature('lemm1').feature('dmp1').active(true)
    model.component('comp1').physics('solid').feature('lemm1').feature('dmp1').set('DampingType', 'RayleighDamping');
    model.component('comp1').physics('solid').feature('lemm1').feature('dmp1').set('InputParameters', 'AlphaBeta'); 
    ii = length(csis)-1;
    
    indexes = [1,2,3,5,6,8,9,10];
    h = csis(indexes); %adimensional damping coefficient
    w0 = 2*pi*f0(indexes);
    const_mod = [1./(2*w0)  w0/2]\h; %Moore-Penrose pseudo-inverse (least mean square error)
    alpha = const_mod(1);
    beta = const_mod(2);
    
    % b) Set Rayleigh constants
    model.component('comp1').physics('solid').feature('lemm1').feature('dmp1').set('alpha_dM', alpha);
    model.component('comp1').physics('solid').feature('lemm1').feature('dmp1').set('beta_dK', beta);

    % c) Run study and mesh
    % model.component('comp1').mesh('mesh1').feature('size').set('hauto', meshSize);
    model.mesh('mesh1').run;
    model.study('std2').run();

    % d) Export Results
    dirName = pwd;
    model.result.export('data1').set('transpose', true);
    model.result.export('data1').set('sdim', 'fromdataset');
    exportData(model,'cpt1', dirName,['velpar'],'solid.u_tZ'); % velocity  
   
    %% % e) Read Data on Matlab
    [vel] = readTuples(['velRayleigh2f7.txt'], 1, false);
    vel = vel(4:end);
    % h) Delete .txt files exported by Comsol
    %delete(['vel',int2str(ii),'par.txt']);
    fAxis = 20:0.5:600;
    ind = find(fAxis>=20);
    fAxis = fAxis(ind);
    %vel = vel(ind);
    [maxVals, maxLocs] = findpeaks(abs(vel)/max(abs(vel)));
    f0Comsol = fAxis(maxLocs);
    Hvplot = Hv(ind);
    figure()
    plot(fAxis, abs(Hvplot/max(abs(Hvplot))), 'LineWidth',1.5);
    hold on;
    plot( fAxis, abs(vel)/max(abs(vel)), 'LineWidth',1.5);
    hold on ;
    stem(f0, abs(Hv(fLocs)/max(abs(Hv(fLocs) ) ) ) );
    hold on;
    stem(f0Comsol, abs(vel(maxLocs)/max(abs(vel(maxLocs)))));
    
    
    xlabel('f    [Hz]');
    ylabel(' |H_v(f)| ');
    legend('real FRF', 'simulated FRF','real peaks', 'Comsol peaks');
    title('Real vs simulated FRF ');
    
    fAxis = 20:0.5:600;
    [Hv,f0, fLocs, csis, Q] = EMASimple(HvSVD, fAxis,1e-3, 3);
    f0 = f0(2:end); csis = csis(2:end); Q = Q(2:end); fLocs = fLocs(2:end);
    %fAxis = 50:0.5:600;
    [HvComsol,f0Comsol, maxLocs, csisComsol, QComsol] = EMAPoly(vel, fAxis,1e-10, 1);
    
    Qs = [Q.'; QComsol(1:length(Q)).'];
    Qtable = array2table(Qs, 'variableNames',{'Q1' 'Q2' 'Q3' 'Q4' 'Q5' 'Q6' 'Q7' 'Q8' 'Q9' 'Q10'}, 'rowNames', {'Real' 'Comsol'});
    writeMat2File(Qs,'Qs.csv', {'Q'}, 1,false); 

    
%% TRY MULTIQUADRATIC REGRESSION
quadCoeffs = zeros(10,10);
statistics = cell(10);
for ii = 1:10
    [p,S] = polyfit(inputsInfo, outputsALLInfo,2);
    multilinCoeffs(:,ii) = p(1:10);
    multilin95Confidence(:, ii*2-1:ii*2) =  bint(1:10,:);
    statistics(ii,:) = stats;
end

statTable = array2table(statistics, 'VariableNames', {'R^2 stat'  'F stat'   'pValue'  'errorVariance'},...
   'RowNames',{'f1' 'f2' 'f3' 'f4' 'f5' 'f6' 'f7' 'f8' 'f9' 'f10'});
