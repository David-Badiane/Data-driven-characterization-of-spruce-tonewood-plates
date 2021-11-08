%% FIX ERRORS IN AMPLITUDE FILE

acqFile = fileread('outputsAmp.csv')
acqFile(1:50) = []
fixedAcqFile = strrep(acqFile, '+-','+');
amps = str2num(fixedAcqFile);
writeMat2File(amps, 'outputsAmp.csv', {'f'}, 1, false);

%%

cd(baseFolder)
rmpath(genpath(csvPath));
saveData = 0;
[Dataset_FA, csvPath, datasetType, datasetDistr] = fetchDataset(saveData, baseFolder);
HPFolder = [csvPath,'\HyperParameters'];

nIn = length(Dataset_FA.inputs(1,:));
referenceVals = readmatrix('referenceVals.csv');
referenceVals(12) = 1e-6;
idxMat = {};
threshold = 0.13;
if(strcmp(datasetDistr, 'gaussian'))
    for ii = 1:nIn
        normDiff = abs(Dataset_FA.inputs(:,ii) - referenceVals(ii))./referenceVals(ii);
        idxMat{ii} = find( normDiff <= threshold);
        disp(length(idxMat{ii}));
        if ii == 1, commonIdxs = idxMat{ii}; end
        commonIdxs = intersect(commonIdxs, idxMat{ii});
        disp(['common idxs at ', int2str(ii), ' length ', num2str(length(commonIdxs))]);   
    end
    % debug
    % check =  abs(Dataset_FA.inputs(commonIdxs,:) - referenceVals)./referenceVals;
    commonIdxs(1) = [];
    splitter = sort(randsample(1:length(commonIdxs), 200));
    test = commonIdxs(splitter);
    model = union( setdiff(1:length(Dataset_FA.inputs(:,1)), commonIdxs),...
                        setdiff(commonIdxs, test) ); 
else
    
    normDiff = abs(Dataset_FA.inputs(:,1) - referenceVals(1))./referenceVals(1);
    threshold = input(['insert threshold for density (< 0.1): ']);
    rhoIdxs = find( normDiff <= threshold);
    rhoIdxs(1) = [];
    disp(['num idxs under threshold: ', int2str(length(rhoIdxs))]);
    splitter = sort(randsample(1:length(rhoIdxs), 200));
    test = rhoIdxs(splitter);
    model = union( setdiff(1:length(Dataset_FA.inputs(:,1)), rhoIdxs),...
                        setdiff(rhoIdxs, test) ); 
end

% Split Dataset
trainSet = struct('inputs', Dataset_FA.inputs(model,:),...
                  'outputsEig', Dataset_FA.outputsEig(model,:),...
                  'outputsAmp', Dataset_FA.outputsAmp(model,:));
testSet =  struct('inputs', Dataset_FA.inputs(test,:),...
                  'outputsEig', Dataset_FA.outputsEig(test,:),...
                  'outputsAmp', Dataset_FA.outputsAmp(test,:));
% save files
fileName = ['HPsets_', datasetType];
cd(HPFolder)
save(fileName, 'trainSet', 'testSet');
cd(baseFolder) 
 
