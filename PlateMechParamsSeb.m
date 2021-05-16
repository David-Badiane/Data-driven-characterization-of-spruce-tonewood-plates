clear all;
close all;

addpath csv
addpath functions
addpath Simulations
outputFilenames = {'input' 'output'};
outputFiletype = '.csv';


%% 1) SET UP PARAMETERS FOR SIMULATIONS

model = mphopen('PlateMechParamsSeb');
% Parameters setup
params = mphgetexpressions(model.param);                  % get initial parameters                          
% get parameters names
varyingParamsNames = params(7:end,1);

Ex = 1e10;
rho = 400;           
referenceVals = [rho, Ex, Ex, Ex*0.04,...
                 Ex*0.06 , Ex*0.06,Ex*0.003,...
                 0.3, 0.3, 0.467];
% referenceVals = [rho,  Ex, Ex, Ex,...
%                  Ex*0.061 ,Ex*0.061, Ex*0.061,...
%                  0.33, 0.33, 0.33];

for jj = 1:length(referenceVals)
     model.param.set(varyingParamsNames(jj), referenceVals(jj));
end 
% referenceVals = [rho,  Ex, Ex, Ex,...
%                  Ex*0.061 ,Ex*0.061, Ex*0.061,...
%                  0.33, 0.33, 0.33];
%              
varyingParamsNames = params(7:end,1);
for jj = 1:length(referenceVals)
                model.param.set(varyingParamsNames(jj), referenceVals(jj));
end
params = mphgetexpressions(model.param)

eigenFreqzNames = {'f1' 'f2' 'f3' 'f4' 'f5'};             


%% 3) DATASET GENERATION - in mech params -- out eigenfrequencies mesh and modeshapes
baseFolder = pwd;
csvPath = [baseFolder,'\csv'];
simFolder = [baseFolder,'\Simulations'];

cd(baseFolder)
nSim = 5;
%model = mphopen('PlateMechParams2');
outputsALLInfo = [];
outputsInfo = [];


model.component('comp1').physics('solid').feature('lemm1').feature('dmp1').active(false);
nModes = 6;



model.study('std1').feature('eig').set('neigs', int2str(nModes));
model.result.export('data1').set('data', 'dset1');
model.study('std1').feature('eig').set('shift', '2[Hz]');


for ii = 1:nSim
    cd(simFolder);
    % import a mesh
    meshFilename = [baseFolder,'\mesh' int2str(ii) ,'.stl'];
    model.mesh('mpart1').feature('imp1').set('filename', meshFilename);
    model.component('comp1').geom('geom1').feature('imp1').set('meshfilename', '');
    model.mesh('mpart1').run;
    model.component('comp1').geom('geom1').run('imp1');
    model.component('mcomp1').baseSystem([]);
    model.geom('mgeom1').lengthUnit('mm');
    model.mesh('mesh1').feature('size').set('hauto', '6');
    model.mesh('mpart1').run;
    model.component('comp1').geom('geom1').feature('imp1').set('mesh', 'mpart1');
    model.component('comp1').mesh('mesh1').run;
    model.mesh('mpart1').feature('imp1').importData;
    
    model.study('std1').run(); 

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
    
      % 3) Evaluate eigenfrequencies
    evalFreqz = mpheval(model,'solid.freq','Dataset','dset1','edim',0,'selection',1);
    eigenFreqz = real(evalFreqz.d1');
    %presentEigenFreqz = [eigenFreqz(2),eigenFreqz(3),eigenFreqz(5),...
                         %eigenFreqz(6),eigenFreqz(10)];
    
    % 4) Extract old values 
    if ii ~=  1
        outputsALLInfo = table2array(readtable("outputsALL.csv"));
    end
    
    % 5) Update results

    %outputsInfo = [outputsInfo; presentEigenFreqz];
    outputsALLInfo = [outputsALLInfo; eigenFreqz];
    
    % 6) Save results
    %outputsTable = writeMat2File(outputsInfo,'outputs.csv', eigenFreqzNames, 5,true);   
    outputsALLTable = writeMat2File(outputsALLInfo,'outputsALL.csv', {'f'}, 1,false);   
end

count = 1;
for ii = 1:nSim
    for jj = 1:6
        meshData = table2array(readtable(['mesh', int2str(ii),'.csv']));
        modesData = table2array(readtable(['modeshapes', int2str(ii),'.csv'])); 
        figure(100)
        subplot (6,6,count)
        z = modesData(:,jj);
        idx = find(z<= 1);
        plot3(meshData(idx,1),meshData(idx,2),z(idx), '.', 'markerSize', 4);
        %view(0,90);
%         xlabel('x  [mm]');
%         ylabel('y  [mm]');
%         zlabel('z  [mm]');
        count = count+1;
    end
end
