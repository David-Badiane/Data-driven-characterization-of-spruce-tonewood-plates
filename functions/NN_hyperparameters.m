function [HPData, HP_filename_freq, HP_filename_amp] = NN_hyperparameters(nNeuronsVec, nLayersVec, nLaxis_f, nLaxis_a,...
                                nModesGet, baseFolder, csvName, flags, sets_filename, nRealizations, getOrdered)

% NN_hyperparameters
% this function allows to perform hyperparameters tuning on the neural
% networks for both amplitude and frequency
% copyright: David Giuseppe Badiane
% -------------------------------------------------------------------------
% inputs
% nNeuronsVec    = double - array with the number of neurons of the
%                  grid search performed to tune the HPs of the NNs
% nLayersVec     = double - array with the number of layers of the
%                  grid search performed to tune the HPs of the NNs
% nLaxis_f       = double - array with the number of layers of the frequency NN 
% nLaxis_a       = double - array with the number of layers of the amplitude NN 
% nModesGet      = int    - number of modes on which NNs are trained
% baseFolder     = string - path of the base working directory
% csvName        = string - name of the directory containing the dataset
% flags          = array of 4 booleans [writeNewFiles doFreq doAmp saveData] 
                   % writeNewFiles --> write new hyperparameters tuning csv files or load previous ones
                   % doFreq        --> perform hyperparameters tuning for NN that predicts eigenfrequencies
                   % doAmp         --> perform hyperparameters tuning for NN that predicts amplitudes
                   % saveData      --> save all data from HP tuning (trained neural networks,
% sets_filename  = string - filename of the csv file containing the results of HP tuning
% nRealizations  = int - n° times the grid search is carried on
% getOrdered     = boolean - get sets ordered by modes or by the ascending
%                  value of the eigenfrequencies
% -------------------------------------------------------------------------
% outputs
% HPData         = cell array containing two structs:
%                  HPData_freq or HPData_amp, both have the subsequent members
%                  HP    --> matrice with the double averaged R2 (coefficient of determination)
%                            averaged over all modes and over the number of realizations 
%                  nets  --> trained neural networks
%                  R2    --> R2 of each neural network
% -------------------------------------------------------------------------
    if nargin <9, sets_filename = 'HPsets'; end
    if nargin<10, nRealizations = 1; end
    if nargin<11, getOrdered = 0; end
    
    % load sets
    [Dataset_FA, csvPath, HPPath] = fetchDataset(baseFolder, nModesGet, getOrdered, csvName);  
    cd(HPPath);
    load(sets_filename);
    
    if getOrdered, datasetType = 'ordered';
    else, datasetType = 'raw'; end
    
    % name of the hyperparameters tuning csv files
    HP_filename_freq = ['HPfreq_nModes_', num2str(nModesGet),'_' datasetType];
    HP_filename_amp  = ['HPamp_nModes_', num2str(nModesGet),'_' datasetType];
    
    % ---------------------------------------------------------------------
    % A) retrieve or preallocate matrices and cell arrays for HP tuning data
    if flags(1) % writeNewFiles 
        HPmat_freq = zeros(max(nNeuronsVec), max(nLaxis_f));
        HPmat_amp  = zeros(max(nNeuronsVec), max(nLaxis_a));
    else        % retrieve old files
        cd(HPPath)
        if flags(2) % do Freq
            HPfreq = readmatrix([HP_filename_freq,'.csv']);
            [nN_f,nL_f] = size(HPfreq); % size is (n° neurons x n° layers)
            HPmat_freq = HPfreq; % fill
        end
        if flags(3) % do Amp
            HPamp =  readmatrix([HP_filename_amp, '.csv']);
            [nN_a,nL_a] = size(HPamp); 
            HPmat_amp =  HPamp;        
        end
    end
    
    % preallocate R2 train cell storage
    R2f    = {};  % frequency   
    R2a    = {};  % amplitude
    % training data cell storage
    a_trs = {};   % frequency
    f_trs = {};   % amplitude 
    % networks cell storage
    nets_f = {};  % frequency 
    nets_a = {};  % amplitude
    
    % ---------------------------------------------------------------------
    % B) preallocate structs for HP data 
    % HP = matrice with mean R2        R2vecs = R2 for each NN for each mode
    % nets = neural networks           
    % nNaxis = neurons number axis     nLaxis = layers number axis
    if flags(2) % doFreq
      HPData_freq = struct('HP_f',[], 'R2vecs',[], 'nets', [], 'nNaxis', [], 'nLaxis', []);
      HPData_freq.nNaxis = nNeuronsVec; HPData_freq.nLaxis = nLaxis_f;
    end
    if flags(3)% doAmp
       HPData_amp = struct( 'HP_a', [], 'R2vecs', [], 'nets',[], 'nNaxis', [], 'nLaxis', []);
      HPData_amp.nNaxis = nNeuronsVec; HPData_amp.nLaxis = nLaxis_a;
    end
    
    % assign vars to have a more readable code
    trainIn =  trainSet.inputs;
    testIn = testSet.inputs;
    trainEig = trainSet.outputsEig(:,1:nModesGet); 
    testEig = testSet.outputsEig(:,1:nModesGet);
    trainAmp = db(abs(trainSet.outputsAmp(:,1:nModesGet)));
    testAmp  = db(abs(testSet.outputsAmp(:,1:nModesGet)));
    % ---------------------------------------------------------------------    
    % C) TRAIN Neural Networks - HP tuning
    
    for L = nLayersVec % L = n layers
        tStart = tic;  % timer
        for N = nNeuronsVec % N = n neurons
            R2fs = []; % preallocate R2 freq
            R2as = []; % preallocate R2 amp
            for kk = 1:nRealizations
                % frequency neural network
                if ismember(L, nLaxis_f) && flags(2) %doFreq
                    disp('training frequency neural network')
                    % train snd test frequency neural networks 
                    [f_R2, fNet, ftr] = NN_trainTest_extTest(trainIn,trainEig, testIn, testEig, N, L);
                     f_R2 = table2array(f_R2); 
                     % store results data
                     R2f{kk}{N}{L} = f_R2;       % coefficients of determination mode by mode 
                     f_trs{kk}{N}{L} = ftr;      % training data
                     R2fs = [R2fs, mean(f_R2)];  % mean coefficient of determination
                     nets_f{kk}{N}{L} =fNet;     % neural networks
                     % show results
                     disp([' FREQ Neurons: ', int2str(N), ' Layers: ', int2str(L), '   mean R2: ', num2str(mean(f_R2))]);
                     disp(num2str(f_R2));
                end
                % amplitude neural network
                if ismember(L,nLaxis_a) && flags(3) % doAmp 
                    % train snd test amplitude neural networks 
                    disp('training amplitude neural network')
                    [a_R2, ampNet, atr] = NN_trainTest_extTest(trainIn, trainAmp , testIn, testAmp, N, L);
                     a_R2 = table2array(a_R2);
                    % store results data
                     R2a{kk}{N}{L} = a_R2;        % coefficients of determination mode by mode 
                     a_trs{kk}{N}{L} = atr;       % training data
                     R2as = [ R2as, mean(a_R2)];  % mean coefficient of determination (for each training) 
                     nets_a{kk}{N}{L} = ampNet;   % neural networks
                     % show results
                     disp([' AMP Neurons: ', int2str(N), ' Layers: ', int2str(L), '   mean R2: ', num2str(mean(a_R2))])
                     disp(num2str(a_R2))
                end
            end
        HPmat_amp(N,L) = mean(R2as);  % average the mean coefficient of determination on the nRealizations
        HPmat_freq(N,L) = mean(R2fs); % average the mean coefficient of determination on the nRealizations
        
        % show elapsed time and message
        if (ismember(L, nLaxis_f) && flags(2)) || (ismember(L,nLaxis_a) && flags(3))            
            disp([newline, 'n° Realization ' num2str(kk) ' elapsed time: ', int2str(floor(toc(tStart)/60)),...
                  ' minutes ', int2str(mod(toc(tStart),60)), ' seconds']);
        end
        end
    end
    
    % -----------------------------------------------------------------
    % D) STORING data routine
    
    cd(HPPath)
    HPData = {};
    if flags(2) % doFreq
        HPData_freq.HP_f= HPmat_freq; 
        HPData_freq.R2vecs = R2f;
        HPData_freq.nets =nets_f;
        HPData{length(HPData)+1} = HPData_freq; % store the struct in HPData
        
        if flags(4) % saveData
        writeMat2File(HPmat_freq,[ HP_filename_freq, '.csv'], {' l'}, 1, false);
        save(['HPDATA_freq_', num2str(nModesGet),'modes_' datasetType], 'R2f', 'nets_f', 'f_trs');
        end
    end
    
    if flags(3) %doAmp
        HPData_amp.HP_a= HPmat_amp; 
        HPData_amp.R2vecs = R2a;
        HPData_amp.nets =nets_a;
        HPData{length(HPData)+1} = HPData_amp;
        
        if flags(4) % saveData
        writeMat2File(HPmat_amp, [HP_filename_amp, '.csv'], {' l'}, 1, false);
        save(['HPDATA_amp_', num2str(nModesGet),'modes_' datasetType], 'R2a', 'nets_a', 'a_trs');
        end
    end
end
