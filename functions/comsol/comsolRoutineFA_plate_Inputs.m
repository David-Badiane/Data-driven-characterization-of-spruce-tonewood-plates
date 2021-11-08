function [Dataset_FA] = comsolRoutineFA_Inputs(model, nSim, nModes,...
                                   varyingParamsNames,  initialTuple,  simFolder, csvPath)
                               
%COMSOLROUTINEFREQAMP Summary of this function goes here
%   Detailed explanation goes here

% - - - - - - - - - - - - - - - - - - - - - - - - - SETUP
    cd(csvPath);
    Dataset_FA = struct('inputs',[] ,'outputsEig',[] ,'outputsAmp',[] );
    Dataset_FA.inputs = readmatrix('inputs.csv');
    
    fNames = {};
    for ii = 1:nModes
        fNames{ii} = ['f_{',int2str(ii),'}' ];
    end

    if initialTuple > 1
        eig = readmatrix('outputsEig.csv');
        amp = readmatrix('outputsAmp.csv');
        Dataset_FA.outputsEig = eig(1:initialTuple-1,:);
        Dataset_FA.outputsAmp = amp(1:initialTuple-1,:);
    end
    varNamesModeshapes = cell(nModes+3,1);
    varNamesxyz = {'x' 'y' 'z'};
    for ii = 1:(nModes +3)
        if ii <4
       varNamesModeshapes{ii} = varNamesxyz{ii}; 
        else 
            varNamesModeshapes{ii} = ['disp f', int2str(ii-3)]; 
        end
    end
    
%- - - - - - - - - - - - - - - - - - - - - - - - - - - START SIMULATION LOOP
    model.component('comp1').physics('solid').feature('lemm1').feature('dmp1').active(true); % --> activate damping

    for ii = initialTuple:nSim
        disp(' '); disp(['----- NEW TUPLE - tuple ',int2str(ii),' ;) -----']); disp('  ');
        tStart = tic;
        cd(simFolder);
        % 1) set params
        
        currentVals = Dataset_FA.inputs(ii,:);
        for jj = 1:length(Dataset_FA.inputs(1,:))
            model.param.set(varyingParamsNames(jj), currentVals(jj));
        end
        
         % generate a table to see chosen params
        array2table([currentVals(:)],...
        'variableNames', {'curr vals'}, 'rowNames', varyingParamsNames)

        
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
        delete([modesFileName,'.txt']); 
        writeMat2File(fileData,['modeshapes', int2str(ii),'.csv'], varNamesModeshapes, nModes+3, true);

        cd(csvPath)
        % save eigenfrequencies
        evalFreqz = mpheval(model,'solid.freq','Dataset','dset1','edim',0,'selection',1); % --> evaluate eigenfrequencies
        eigenFreqz = real(evalFreqz.d1');
        eigenFreqz = eigenFreqz(:).';
        
        % frequency domain
        model.study('std2').feature('freq').set('plist', num2str(eigenFreqz)); % --> set FD studies at eigenfrequencies value
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
        Dataset_FA.outputsEig = [Dataset_FA.outputsEig; eigenFreqz];
        Dataset_FA.outputsAmp = [Dataset_FA.outputsAmp; vel];

        % Save results
        outputsEigTable = writeMat2File(Dataset_FA.outputsEig,'outputsEig.csv', {'f'}, 1,false);  
        outputsAmpTable = writeMat2File(Dataset_FA.outputsAmp,'outputsAmp.csv', {'f'}, 1,false);
        disp(['elapsed time 1 tuple: ', num2str(round(toc(tStart),2)), ' seconds']);
    end
end

