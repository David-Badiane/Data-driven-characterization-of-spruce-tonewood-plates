function [Dataset_FA, csvPath, HyperParameters_path] =...
    fetchDataset(baseFolder, modesGet, getOrdered, csvName, saveData)
% this function fetches the dataset with the number of columns specified by modesGet,
% if the fetched data is ordered by modes we will have a dataset with the
% frequency and amplitude of each mode
% copyrigth: David Giuseppe Badiane
% -------------------------------------------------------------------------
% inputs: 
% baseFolder   = string  - directory of the basic folder - ex. FRF2Params/gPlates
% modesGet     = 1x1 int - set how many modes are retrieved in the dataset
% getOrdered   = boolean - if true get the dataset ordered by modes (reduced)
%                          otherwise, get raw dataset
% csvName      = string  - name of the directory containing the dataset
% -------------------------------------------------------------------------
% outputs:
% Dataset_FA   = struct - contains the dataset 
% datasetPath  = string - complete path to dataset directory
% HPFolder     = string - complete path to hyperparameters tuning folder
% -------------------------------------------------------------------------
    % set number of inputs behaviour
     if nargin<3, getOrdered = false; end
     if nargin<4, csvName = input('insert dataset csv directory (string): '); end
     if nargin<5, saveData = 0; end
     
    % choose btw ordered and raw
    if ~getOrdered 
    datasetType = 'raw';
    else, datasetType = 'ordered'; end
    
    % declare the filename of the dataset and add path of its directory
    saveFilename = ['D_',csvName,'_', datasetType,'.mat'];
    csvPath = [baseFolder,'\', csvName];
    addpath(csvPath);
    % array with the indexes of the columns of the dataset to fetch
    getIdxs = 1:modesGet;
    
    % ---------------------------------------------------------------------          
    % if the dataset required is ordered by the ascending value of the eigenfrequencies
    if strcmp(datasetType, 'raw')
        % inputs = dataset inputs, outputsEig = eigenfrequencies value
        % outputsAmp = amplitude value
        Dataset_FA = struct('inputs',[] ,'outputsEig',[] ,'outputsAmp',[]);
        Dataset_FA.inputs = readmatrix('inputs.csv');
        Dataset_FA.outputsEig = readmatrix('outputsEig.csv');
        Dataset_FA.outputsAmp = readmatrix('outputsAmp.csv');
        Dataset_FA.outputsEig = Dataset_FA.outputsEig(:,getIdxs);
        Dataset_FA.outputsAmp = Dataset_FA.outputsAmp(:,getIdxs);
    end
    % ---------------------------------------------------------------------
    % if the dataset required is ordered by mode numbers
    if strcmp(datasetType, 'ordered')
        % inputs = dataset inputs, 
        % outputsEig = eigenfrequencies value (by mode)
        % outputsAmp = amplitude value (by mode), 
        % modesOrder = order of appearance of modes in the dataset 
        Dataset_FA = struct('inputs',[] , 'modesOrder', [], ...
                            'outputsEig',[] ,'outputsAmp',[]);
        % reading data
        Dataset_FA.inputs =     readmatrix('datasetOrdered_Inputs.csv');
        Dataset_FA.outputsEig = readmatrix('datasetOrdered_Eig.csv');
        Dataset_FA.outputsAmp = readmatrix('datasetOrdered_Amp.csv');
        Dataset_FA.modesOrder = readtable('datasetOrdered_modesIdxs.csv').Properties.VariableNames;
        
        % retrieve modes indexes and the order of data (ordered dataset contains both peaks and modes)
        % to better understand it look directly the file 'datasetOrdered_Eig.csv' or at the variable dataOrder
        modesIdxs =  readmatrix('datasetOrdered_modesIdxs.csv');
        dataOrder =  readtable('datasetOrdered_Eig.csv').Properties.VariableNames;

        % study how much each mode is present in the dataset and put it in a csv file
        modesPresence = zeros(1,length(Dataset_FA.modesOrder));
        for ii = 1:length(dataOrder)/2
            modesPresence(ii) = length(find(~isnan(Dataset_FA.outputsEig(:,modesIdxs(ii)))));
        end
        cd([csvPath, '\modesAnalysis'])
        writeMat2File(modesPresence,'modesPresence.csv', Dataset_FA.modesOrder, length(Dataset_FA.modesOrder), true);
        
        % finally get dataset ordered by modes
        modesIdxs = modesIdxs(getIdxs);
        disp('got modes: ');
        array2table(getIdxs, 'variableNames', Dataset_FA.modesOrder(getIdxs))
        Dataset_FA.modesOrder = Dataset_FA.modesOrder(getIdxs);
        Dataset_FA.outputsEig = Dataset_FA.outputsEig(:, modesIdxs);
        Dataset_FA.outputsAmp = Dataset_FA.outputsAmp(:, modesIdxs);
    end
% -------------------------------------------------------------------------7
    % save Dataset
    if saveData
        cd(csvPath)
        disp(['saving ', saveFilename])
        save(saveFilename, 'Dataset_FA');
        cd(baseFolder)
    end
    
    % get hyperparameters path and return to base working directory
    HyperParameters_path = [csvPath,'\HyperParameters'];    
    cd(baseFolder)
end