function [Dataset_FA] = comsolRoutineFA_plate(model, nSim, nModes, dataset_center_values,...
                                   inputParamsNames,  standardDev,  mshapesPath,...
                                   datasetPath, writeNow, samplingMethod)
% function for dataset generation - randomly sample inputs, calculate
% eigenfrequencies and FRF amplitude at eigenfrequencies
% copyright: David Giuseppe Badiane
% -------------------------------------------------------------------------
% inputs:                       
% model            = comsol finite element model
% nSim             = 1x1 double - number of simulations
% nModes           = 1x1 double - number of modes calculated
% dataset_center_values    = 1x15 double - nominal values of the dataset 
% [density (1) elastic constants (9) Rayleigh constants (2) geometry (3)]
% inputParamsNames = 1x15 cell array - names of the inputs of the dataset
% standardDev      = 1x15 double - std of the dataset inputs
% mshapesPath      = string - directory where modeshapes are stored
% datasetPath      = string - directory where dataset is stored
% writeNow         = boolean - if true --> dataset computation starts from zero
%                              if false --> dataset computation starts from
%                              the data that already are in its directory
% samplingMethod   = string - specifies the distribution of the dataset inputs
%                             can be either 'uniform' or 'gaussian' 
% -------------------------------------------------------------------------
% outputs:
% Dataset_FA       = struct containing the raw Dataset
%                    fields --> inputs
%                               outputsEig, outputsAmp 
% -------------------------------------------------------------------------

% set different behaviors depending number of inputs 
if nargin < 10
    samplingMethod = 'gaussian'; % if not specified differently, sampling method is Gaussian
end

% names of the outputs for the .csv files of the dataset 
fNames = {};
for ii = 1:nModes
    fNames{ii} = ['f_{',int2str(ii),'}' ];
end

% - - - - - - - - - - - - - - - - - - - - - - - - - SETUP
    cd(datasetPath);
    % dataset struct
    Dataset_FA = struct('inputs',[] ,'outputsEig',[] ,'outputsAmp',[] );
    % set starting tuple
    if writeNow, startTuple = 1; % do nothing, it's gonna be filled during the routine
    else        % load previously computed dataset and continue routine
        Dataset_FA.inputs = table2array(readtable('inputs.csv'));
        Dataset_FA.outputsEig = table2array(readtable('outputsEig.csv'));
        Dataset_FA.outputsAmp = table2array(readtable('outputsAmp.csv'));
        startTuple = size(Dataset_FA.inputs,2)+1;
    end
    
    % set variable names for modeshapes .csv files
    varNamesModeShapes = {'x' 'y' 'z'};
    for ii = 1:nModes
       varNamesModeshapes{ii+3} = ['f',int2str(ii)]; 
    end
    
%- - - - - - - - - - - - - - - - - - - - - - - - - - - START SIMULATION LOOP

    for ii = startTuple:nSim
        tStart = tic;                                                       % time counter
        disp(' '); disp(['----- NEW TUPLE - tuple ',int2str(ii),' ;) -----', newline]);
        cd(mshapesPath);                                                    % go to modeshapes path
        
% - - - A) gaussian sample dataset inputs - - - - - - - - - - - - - - - - -
        if ii == 1
            % if first iteration - run dataset centerValues values
            setParams(model, inputParamsNames, dataset_center_values);
            currentVals = dataset_center_values;
        else % choose sampling method 
            
            if strcmp(samplingMethod, 'gaussian')
             currentVals = gaussianSample(dataset_center_values, standardDev);
            elseif strcmp(samplingMethod,'uniform')
             currentVals    = uniformSample(dataset_center_values, standardDev);
            end
            % override sampling of Rayleigh constants with wide uniform distribution
            if length(currentVals)>=8
                % alpha unform btw 0 and 100
                currentVals(11) = 50 + 50*(2*rand(1,1)-1);
                % beta uniform btw 2e-7 and 2e-5
                currentVals(12) = 2*10.^(rand(1,1)*2 - 7); % uniform random variable in the exponent
            end
            % set parameters in Comosol FE model
            setParams(model, inputParamsNames, currentVals);
        end
        
        % generate a table to see chosen params
        array2table([dataset_center_values(:), currentVals(:), currentVals(:)./dataset_center_values(:)].',...
        'rowNames', {'refVals' 'currVals'  'curr/ref'}, 'variableNames', inputParamsNames)

% - - - B) eigenfrequency study (frequency) + modeshapes retrieval - - - -
        model.study('std1').feature('eig').set('neigs', int2str(nModes)); % ---> set number of modes
        model.component('comp1').physics('solid').feature('lemm1').feature('dmp1').active(false); % --> deactivate damping
        model.study('std1').run(); 
        model.component('comp1').physics('solid').feature('lemm1').feature('dmp1').active(true); % --> activate damping
        
        % save eigenfrequencies
        evalFreqz = mpheval(model,'solid.freq','Dataset','dset1','edim',0,'selection',1); % --> evaluate eigenfrequencies
        eigenFreqz = evalFreqz.d1';
        eigenFreqz = real(eigenFreqz(:)).';
        
        % save modeshapes
        modesFileName = 'solidDisp'; % filename exported by comsol
        expression = {'solid.disp'}; % retrieve the displacement of the body
        model.result.export('data1').set('transpose', false);
        model.result.export('data1').set('data', 'dset1');
        % export the modeshapes in a .txt file
        exportAllModesFromDataset(model, modesFileName,mshapesPath,expression); % custom function - check it
        fileData = readTuples([modesFileName,'.txt'], nModes+3, true); % read txt file
        fileData = [fileData(:,1:3) (fileData(:,4:end))];
        delete([modesFileName,'.txt']); % delete .txt file
        writeMat2File(fileData,['modeshapes', int2str(ii),'.csv'], varNamesModeshapes, nModes+3, true); %store them
        cd(datasetPath)
        disp(['modeshape 1st line: ' num2str(fileData(1,4:end))]); % check retrieved modeshape shows no errors (Nans or zero only modeshapes)
        
% - - - C) frequency domain study (amplitude) - - - - - - - - - - - - - - -
        model.study('std2').feature('freq').set('plist', num2str(real(eigenFreqz))); % --> set FD studies at eigenfrequencies value
        model.study('std2').run(); % run study
        % export results
        dirName = pwd;
        model.result.export('data1').set('transpose', true);
        model.result.export('data1').set('sdim', 'fromdataset');
        exportData(model,'cpt1', dirName,['vel'],'solid.u_tZ'); % velocity 
        [vel] = readTuples(['vel.txt'], 1, false);
        delete('vel.txt');
        vel = abs(vel(4:end)); % vel(1:3) are the x,y,z coordinates of the measurement point, useless
        
        % table to check freqz + normalized amp 
        % allows to see if a mode is resonance or antiresonance
        array2table([round(eigenFreqz(:)),round( abs(vel(:))/max(abs(vel(:))),1)].',...
             'rowNames',{'eigenfreqz' 'normalized amp'} , 'variableNames', fNames)

        % Update dataset
        Dataset_FA.inputs     = [Dataset_FA.inputs; currentVals];
        Dataset_FA.outputsEig = [Dataset_FA.outputsEig; eigenFreqz];
        Dataset_FA.outputsAmp = [Dataset_FA.outputsAmp; vel];

        % Save dataset
        inputsTable  = writeMat2File(Dataset_FA.inputs,'inputs.csv', inputParamsNames, length(inputParamsNames),true);
        outputsEigTable = writeMat2File(Dataset_FA.outputsEig,'outputsEig.csv', {'f'}, 1,false);
        outputsAmpTable = writeMat2File(Dataset_FA.outputsAmp,'outputsAmp.csv', {'f'}, 1,false);
        
        % display time and tuple infos
        disp(['elapsed time for 1 tuple = ' , num2str(toc(tStart)), ' seconds', newline]);
        disp(['----- END TUPLE ', int2str(ii),' START TUPLE ', int2str(ii+1),'-----', newline]);
    end
end


function [currentVals] = gaussianSample(dataset_center_values, standardDev) % function for gaussian sampling
    gaussRealization = randn(size(dataset_center_values));
    delta = standardDev.*gaussRealization;
    currentVals = dataset_center_values.*(ones(size(dataset_center_values)) + delta);
end

function [currentVals] = uniformSample(dataset_center_values, range) % function for uniform sampling
    % uniform distribution in [-1,1]
    uniform = (-1 + 2*rand(size(dataset_center_values)));
%     disp([length(idxPos), length(idxNeg)])
    currentVals = dataset_center_values.*( 1+ range.*(uniform));
end

