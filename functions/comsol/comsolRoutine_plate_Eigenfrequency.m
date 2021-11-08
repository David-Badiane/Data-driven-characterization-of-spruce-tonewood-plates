function [inputsInfo, inputsTable, outputsALLInfo, outputsALLTable] = comsolRoutineEigenfrequency(model, nSim, nModes, simFolder, csvPath, varyingParamsNames, referenceVals,  standardDev, writeNow)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here
    if writeNow
        inputsInfo = [];
        outputsALLInfo = [];
    else
        inputsInfo = table2array(readtable('inputs.csv'));
        outputsALLInfo = table2array(readtable('outputsALL.csv'));
    end

    
    if isempty(inputsInfo)
        start = 1;
    else
        start = length(inputsInfo(:,1))+1;
    end
    
    for ii = start:nSim
        disp(ii)
        cd(simFolder);
        % 1) gaussian sample mechanical parameters
        if ii == 1
            for jj = 1:length(referenceVals)
                model.param.set(varyingParamsNames(jj), referenceVals(jj));
            end
            currentVals = referenceVals;
        else
            currentVals = gaussianSample(referenceVals, standardDev);
            for jj = 1:length(referenceVals)
            model.param.s et(varyingParamsNames(jj), currentVals(jj));
            end
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

        % 4) Extract old values 
        if ii ~=  1
            inputsInfo = table2array(readtable("inputs.csv"));
            outputsALLInfo = table2array(readtable("outputsALL.csv"));
        end

        % 5) Update results
        inputsInfo = [inputsInfo; currentVals];
        outputsALLInfo = [outputsALLInfo; eigenFreqz]

        % 6) Save results
        inputsTable  = writeMat2File(inputsInfo,'inputs.csv', varyingParamsNames(1:10), 10,true);   
        outputsALLTable = writeMat2File(outputsALLInfo(:,1:nModes),'outputsALL.csv', {'f'}, 1,false);   
    end
end

function [currentVals] = gaussianSample(referenceVals, standardDev)
    gaussRealization = randn(size(referenceVals));
    currentVals = referenceVals.*(ones(size(referenceVals)) + standardDev.*gaussRealization);
%     disp(referenceVals);
%     disp(currentVals);
end

