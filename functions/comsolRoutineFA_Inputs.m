function [Dataset_FA, inputsTable, outputsEigTable, outputsAmpTable] = comsolRoutineFA_Inputs(model, nSim, nModes, referenceVals,...
                                   varyingParamsNames,  standardDev,  simFolder, csvPath)
                               
%COMSOLROUTINEFREQAMP Summary of this function goes here
%   Detailed explanation goes here

% - - - - - - - - - - - - - - - - - - - - - - - - - SETUP
    cd(csvPath);
    Dataset_FA = struct('inputs',[] ,'outputsEig',[] ,'outputsAmp',[] );
    Dataset_FA.inputs = table2array(readtable('inputs.csv'));
    
    start = 1;
    
    varNamesModeshapes = cell(nModes+3,1);
    varNamesxyz = {'x' 'y' 'z'};
    for ii = 1:(nModes +3)
        if ii <4
       varNamesModeshapes{ii} = varNamesxyz{ii}; 
        else 
            varNamesModeshapes{ii} = ['disp f',int2str(ii-3)]; 
        end
    end
    
%- - - - - - - - - - - - - - - - - - - - - - - - - - - START SIMULATION LOOP
    model.component('comp1').physics('solid').feature('lemm1').feature('dmp1').active(true); % --> activate damping

    for ii = start:nSim
        disp(ii)        
        cd(simFolder);
        % 1) gaussian sample mechanical parameters
        if ii == 1
            % if first iteration - run reference values
            for jj = 1:length(referenceVals)
                model.param.set(varyingParamsNames(jj), referenceVals(jj));
            end
            currentVals = referenceVals;
        else
            % else - generate gaussian sampling of MP
            currentVals = Dataset_FA.inputs(ii,:);
            disp(currentVals);
            for jj = 1:length(referenceVals)
                model.param.set(varyingParamsNames(jj), currentVals(jj));
            end
        end
        
        % eigenfrequency study
        model.study('std1').feature('eig').set('neigs', int2str(nModes)); % ---> set number of modes
%         model.component('comp1').physics('solid').feature('lemm1').feature('dmp1').active(false); % --> deactivate damping
        model.study('std1').run(); 
        
        % save modeshapes
        modesFileName = 'solidDisp';
        expression = {'solid.disp'};
        
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

        % Update results
        %Dataset_FA.inputs = [Dataset_FA.inputs; currentVals]
        Dataset_FA.outputsEig = [Dataset_FA.outputsEig; eigenFreqz]
        Dataset_FA.outputsAmp = [Dataset_FA.outputsAmp; vel]

        % Save results
        %inputsTable  = writeMat2File(Dataset_FA.inputs,'inputs.csv', varyingParamsNames(1:12), 10,true);   
        outputsEigTable = writeMat2File(Dataset_FA.outputsEig,'outputsEig.csv', {'f'}, 1,false);  
        outputsAmpTable = writeMat2File(Dataset_FA.outputsAmp,'outputsAmp.csv', {'f'}, 1,false);
    end
end

