function [Dataset_FA, csvPath, datasetType, HPFolder] =...
            fetchReduceDataset_wedge(baseFolder, modesGet, getOrdered, csvName)
        
    if nargin<3,    getOrdered = false; end
    if nargin > 3
    else,    csvName = input('insert dataset csv directory (string): '); end
    
    if ~getOrdered, datasetType = input('insert dataset type (ordered/raw): ');
    else,           datasetType = 'ordered'; end
    
    % fetch dataset
    [Dataset_FA, csvPath, datasetType] = fetchDataset_wedge(0, baseFolder, getOrdered, csvName);
    HPFolder = [csvPath,'\HyperParameters'];

    % check modes presence, reduce dataset idxs to well represented ones
    cd([csvPath, '\modesAnalysis']);
    modesPresence = readtable('modesPresence.csv');
    datasetModes = modesPresence.Properties.VariableNames;
    m = table2array(modesPresence);
    
    getIdxs = 1:modesGet;
    modesIdxsEig = Dataset_FA.modesIdxs(1,getIdxs);
    modesIdxsAmp = Dataset_FA.modesIdxs(:,getIdxs);
    modesIdxsAmp = modesIdxsAmp(:);
    
    disp('get modes: ');
    array2table(m(getIdxs), 'variableNames', datasetModes(getIdxs))
    
    Dataset_FA.outputsEig = Dataset_FA.outputsEig(:, modesIdxsEig);
    Dataset_FA.outputsAmp = Dataset_FA.outputsAmp(:, modesIdxsAmp);
    Dataset_FA.modesOrder = Dataset_FA.modesOrder(1:length(modesIdxsEig));
    Dataset_FA.modesIdxs = [];
    Dataset_FA.peaksIdxs = [];
    Dataset_FA.dataOrder = [];
end

