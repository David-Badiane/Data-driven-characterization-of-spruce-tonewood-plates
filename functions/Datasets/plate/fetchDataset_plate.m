function [Dataset_FA, csvPath, datasetType, datasetDistr] = fetchDataset_plate(saveData, baseFolder, getOrdered, csvName)
    
    if nargin<3,    getOrdered = false; end
    if nargin >3
    else,    csvName = input('insert dataset csv directory (string): '); end
    
    csvPath = [baseFolder,'\', csvName];
    addpath(csvPath);
    
    % choose btw ordered and raw
    if ~getOrdered 
    datasetType = input('insert dataset type (ordered/raw): ');
    else, datasetType = 'ordered'; end
    datasetDistr = csvName(11:12);
    saveFilename = ['D_',datasetDistr,'_', datasetType,'.mat'];
    
    mechParamsNames = {'\rho' 'E_x' 'E_y' 'E_z' 'G_{xy}' 'G_{yz}' 'G_{xz}' '\nu_{xy}' '\nu_{yz}' '\nu_{xz}' '\alpha' '\beta'};
    Dataset_FA = struct('inputs',[] ,'outputsEig',[] ,'outputsAmp',[], 'modesIdxs', [] );

    if strcmp(datasetType, 'ordered')
        Dataset_FA = struct('inputs',[] , 'dataOrder',[], 'modesOrder', [], ...
                            'outputsEig',[] ,'outputsAmp',[], 'modesIdxs', [], 'peaksIdxs', []);
        Dataset_FA.inputs = readmatrix('datasetOrdered_Inputs.csv');
        Dataset_FA.outputsEig = readmatrix('datasetOrdered_Eig.csv');
        Dataset_FA.outputsAmp = readmatrix('datasetOrdered_Amp.csv');
        Dataset_FA.dataOrder = readtable('datasetOrdered_Eig.csv').Properties.VariableNames;

        nColsDataset = length(Dataset_FA.dataOrder);
        modesIdxs = 1:2:nColsDataset;
        peaksIdxs = 2:2:nColsDataset;
        Dataset_FA.modesIdxs = modesIdxs;
        Dataset_FA.peaksIdxs = peaksIdxs;
        Dataset_FA.modesOrder = Dataset_FA.dataOrder(modesIdxs);
    end

    if strcmp(datasetType, 'raw')
        Dataset_FA.inputs = readmatrix('inputs.csv');
        Dataset_FA.outputsEig = readmatrix('outputsEig.csv');
        Dataset_FA.outputsAmp = readmatrix('outputsAmp.csv');
        Dataset_FA.modesIdxs = 1:length(Dataset_FA.outputsEig(1,:));
        peaksIdxs = [];
    end

    nTuples = length(Dataset_FA.inputs(:,1));
    varyingParamsNames = {'rho', 'Ex', 'Ey', 'Ez', 'Gxy', 'Gyz', 'Gxz', 'vxy', 'vyz', 'vxz', 'alpha', 'beta'};

    if saveData
        trainPath = inputs('save inputs where ?: ');
        cd(trainPath)
        disp(['saving ', saveFilename])
        save(saveFilename, 'Dataset_FA', 'modesIdxs', 'peaksIdxs');
        cd(baseFolder)
    end

    if strcmp(datasetType, 'ordered')
        modesPresence = zeros(1,20);
        for ii = 1:length(Dataset_FA.dataOrder)/2
            modesPresence(ii) = length(find(~isnan(Dataset_FA.outputsEig(:,modesIdxs(ii)))));
        end
        cd([csvPath, '\modesAnalysis'])
        writeMat2File(modesPresence,'modesPresence.csv', Dataset_FA.modesOrder, length(Dataset_FA.modesOrder), true);
        cd(baseFolder)
    end
end