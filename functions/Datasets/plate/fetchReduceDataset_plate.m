function [Dataset_FA, csvPath, datasetType, datasetDistr, HPFolder] = fetchReduceDataset_plate(baseFolder, modesGet, getOrdered, csvName)
    
    if nargin<3,    getOrdered = false; end
    if nargin > 3
    else,    csvName = input('insert dataset csv directory (string): '); end
    
    if ~getOrdered 
    datasetType = input('insert dataset type (ordered/raw): ');
    else, datasetType = 'ordered'; end
    datasetDistr = csvName(11:12);
    
    % fetch dataset
    [Dataset_FA, csvPath, datasetType, datasetDistr] = fetchDataset_plate(0, baseFolder, getOrdered, csvName);
    HPFolder = [csvPath,'\HyperParameters'];

    % check modes presence, reduce dataset idxs to well represented ones
    cd([csvPath, '\modesAnalysis']); modesPresence = readtable('modesPresence.csv');
    datasetModes = modesPresence.Properties.VariableNames;
    m = table2array(modesPresence);
    getIdxs = 1:modesGet;
    Dataset_FA.modesIdxs = Dataset_FA.modesIdxs(getIdxs);
    
    disp('get modes: ');
    array2table(m(getIdxs), 'variableNames', datasetModes(getIdxs))
    
    Dataset_FA.outputsEig = Dataset_FA.outputsEig(:, Dataset_FA.modesIdxs);
    Dataset_FA.outputsAmp = Dataset_FA.outputsAmp(:, Dataset_FA.modesIdxs);
    
end