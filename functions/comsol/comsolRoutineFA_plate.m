function [Dataset_FA] = comsolRoutineFA_plate(model, nSim, nModes, referenceVals,...
                                   varyingParamsNames,  standardDev,  simFolder,...
                                   csvPath, writeNow, samplingMethod)
                               
%COMSOLROUTINEFREQAMP Summary of this function goes here
%   Detailed explanation goes here
if nargin < 10
    samplingMethod = 'gaussian';
end

fNames = {};
for ii = 1:nModes
    fNames{ii} = ['f_{',int2str(ii),'}' ];
end

% - - - - - - - - - - - - - - - - - - - - - - - - - SETUP
    cd(csvPath);
    Dataset_FA = struct('inputs',[] ,'outputsEig',[] ,'outputsAmp',[] );
    if writeNow
    else
        Dataset_FA.inputs = table2array(readtable('inputs.csv'));
        Dataset_FA.outputsEig = table2array(readtable('outputsEig.csv'));
        Dataset_FA.outputsAmp = table2array(readtable('outputsAmp.csv'));
    end
    
    if isempty(Dataset_FA.inputs)
        start = 1;
    else
        start = length(Dataset_FA.inputs(:,1))+1;
    end
    
    varNamesModeshapes = cell(nModes+3,1);
    varNamesxyz = {'x' 'y' 'z'};
    for ii = 1:(nModes +3)
        if ii <4
       varNamesModeshapes{ii} = varNamesxyz{ii}; 
        else 
            varNamesModeshapes{ii} = ['f',int2str(ii-3)]; 
        end
    end
    
%- - - - - - - - - - - - - - - - - - - - - - - - - - - START SIMULATION LOOP

    for ii = start:nSim
        tStart = tic;
        disp(' '); disp(['----- NEW TUPLE - tuple ',int2str(ii),' ;) -----']); disp('  ');
        cd(simFolder);
        % 1) gaussian sample mechanical parameters
        if ii == 0
            % if first iteration - run reference values
            for jj = 1:length(referenceVals)
                model.param.set(varyingParamsNames(jj), referenceVals(jj));
            end
            currentVals = referenceVals;
        else    
            if strcmp(samplingMethod, 'gaussian')
             currentVals = gaussianSample(referenceVals, standardDev);
            end
            if strcmp(samplingMethod,'uniform')
            currentVals = uniformSample(referenceVals, standardDev);
            rhoVals = gaussianSample(referenceVals, 0.1);
            currentVals(1) = rhoVals(1); % density overrun
            currentVals(8:10) = rand(1,3); % Poisson ratios overrun
            end
            
            % alpha unform btw 0 and 100
            currentVals(11) = 50 + 50*(2*rand(1,1)-1);
            % beta uniform btw 2e-7 and 2e-5
            currentVals(12) = 2*10.^(rand(1,1)*2 - 7); 
            
            for jj = 1:length(referenceVals)
            model.param.set(varyingParamsNames(jj), currentVals(jj));
            end
        end
        
        % generate a table to see chosen params
        array2table([referenceVals(:), currentVals(:), currentVals(:)./referenceVals(:)],...
        'variableNames', {'refVals' 'currVals'  'curr/ref'}, 'rowNames', varyingParamsNames)

        
        % eigenfrequency study
        model.study('std1').feature('eig').set('neigs', int2str(nModes)); % ---> set number of modes
        model.component('comp1').physics('solid').feature('lemm1').feature('dmp1').active(false); % --> deactivate damping
        model.study('std1').run(); 
        model.component('comp1').physics('solid').feature('lemm1').feature('dmp1').active(true); % --> activate damping
        
        % save modeshapes
        modesFileName = 'solidDisp';
        expression = {'solid.disp'};
        
        model.result.export('data1').set('transpose', false);
        model.result.export('data1').set('data', 'dset1');
        exportAllModesFromDataset(model, modesFileName,simFolder,expression);
        fileData = readTuples([modesFileName,'.txt'], nModes+3, true);
        fileData = [fileData(:,1:3) imag(fileData(:,4:end))];
        delete([modesFileName,'.txt']); 
        writeMat2File(fileData,['modeshapes', int2str(ii),'.csv'], varNamesModeshapes, nModes+3, true);

        cd(csvPath)
        % save eigenfrequencies
        evalFreqz = mpheval(model,'solid.freq','Dataset','dset1','edim',0,'selection',1); % --> evaluate eigenfrequencies
        eigenFreqz = evalFreqz.d1';
        eigenFreqz = real(eigenFreqz(:)).';
        
        % frequency domain
        model.study('std2').feature('freq').set('plist', num2str(real(eigenFreqz))); % --> set FD studies at eigenfrequencies value
        model.study('std2').run(); 

        % export FD results
        dirName = pwd;
        model.result.export('data1').set('transpose', true);
        model.result.export('data1').set('sdim', 'fromdataset');
        exportData(model,'cpt1', dirName,['vel'],'solid.u_tZ'); % velocity 
        [vel] = readTuples(['vel.txt'], 1, false);
        delete('vel.txt');
        vel = vel(4:end);
        

        % another table to check freqz and normalized amp
        % normalized to check if a mode is present or not
        array2table([round(eigenFreqz(:)),round( abs(vel(:))/max(abs(vel(:))),1)],...
             'variableNames',{'eigenfreqz' 'normalized amp'} , 'rowNames', fNames)

        % Update results
        Dataset_FA.inputs = [Dataset_FA.inputs; currentVals];
        Dataset_FA.outputsEig = [Dataset_FA.outputsEig; eigenFreqz];
        Dataset_FA.outputsAmp = [Dataset_FA.outputsAmp; vel];

        % Save results
        inputsTable  = writeMat2File(Dataset_FA.inputs,'inputs.csv', varyingParamsNames(1:12), 10,true);
        outputsEigTable = writeMat2File(Dataset_FA.outputsEig,'outputsEig.csv', {'f'}, 1,false);
        outputsAmpTable = writeMat2File(Dataset_FA.outputsAmp,'outputsAmp.csv', {'f'}, 1,false);
        
        disp(['elapsed time for 1 tuple = ' , toc(tStart), ' seconds']);
        disp(' '); 
        disp(['----- END TUPLE ', int2str(ii),' START TUPLE ', int2str(ii+1),'-----']);
        disp('  ');

    end
end


function [currentVals] = gaussianSample(referenceVals, standardDev)
    gaussRealization = randn(size(referenceVals));
    currentVals = referenceVals.*(ones(size(referenceVals)) + standardDev.*gaussRealization);
%     disp(referenceVals);
%     disp(currentVals);
end

function [currentVals] = uniformSample(referenceVals, range)
    % uniform distribution in [-1,1]
    uniform = (-1 + 2*rand(size(referenceVals)));
    % find positive and negative values indexes
    idxPos = find(uniform>0); idxNeg = find(uniform<0);
    % the range [n^-1 , n], where n is the single value of range
    % positive part [0,1] -> [1,n] - SCALE BY n-1, translate by 1
    uniform(idxPos) = ( (range(idxPos) -1).*uniform(idxPos) ) + 1;
    % negative part [-1,0] -> [1/n, 1] - Subtract to one 
    uniform(idxNeg) = 1 - (abs(uniform(idxNeg)).* (1 - range(idxNeg).^-1));
%     disp([length(idxPos), length(idxNeg)])
    currentVals = referenceVals.*(uniform);
end

