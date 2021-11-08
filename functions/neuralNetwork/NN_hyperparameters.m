function [HPData] = NN_hyperparameters(nNeuronsVec, nLayersVec, nLaxis_f, nLaxis_a,...
                                nModesGet, baseFolder, csvName, flags, isWedge)

% flags = array with flags [writeNewFiles doFreq doAmp saveData]
    if nargin < 9, isWedge = false; end
    % load sets
    if isWedge
        [Dataset_FA, csvPath, datasetType, HPFolder] = fetchReduceDataset_wedge(baseFolder, nModesGet, true, csvName);  
    else
        [Dataset_FA, csvPath, datasetType, datasetDistr, HPFolder] = fetchReduceDataset_plate(baseFolder, nModesGet, true, csvName);  
    end
    
    sets_FileName = ['HPsets'];
    cd(HPFolder);
    load(sets_FileName);
    %disp(' '); disp('acquired: '); 
    %disp(Dataset_FA); disp(trainSet);
    if ~isWedge
    HPfileFreq = ['HPfreq_',datasetDistr(1:2),'_', datasetType];
    HPfileAmp  = ['HPamp_',datasetDistr(1:2),'_',datasetType];
    else
        HPfileFreq = ['HPfreq_', num2str(nModesGet),'modes'];
        HPfileAmp  = ['HPamp_', num2str(nModesGet), 'modes'];
    end
    
    if flags(1) % writeNewFiles 
        HPmat_freq = zeros(max(nNeuronsVec), max(nLaxis_f));
        HPmat_amp  = zeros(max(nNeuronsVec), max(nLaxis_a));
    else % retrieve old files
        cd(HPFolder)
        HPfreq = readmatrix([HPfileFreq,'.csv']);
        HPamp =  readmatrix([HPfileAmp, '.csv']);
        [nN_f,nL_f] = size(HPfreq);
        [nN_a,nL_a] = size(HPamp);
        HPmat_freq = zeros(nN_f , nL_f); 
        HPmat_amp = zeros(nN_a, nL_a);   
        HPmat_freq(1:nN_f, 1:nL_f) = HPfreq;
        HPmat_amp(1:nN_a, 1:nL_a) = HPamp;
    end

    R2f    = {};     
    R2a    = {};
    nets_f = {};     
    nets_a = {};
    
    if flags(2) % doFreq
    HPData_freq = struct('HP_f',[], 'R2vecs',[], 'nets', [], 'nNaxis', [], 'nLaxis', []);
    HPData_freq.nNaxis = nNeuronsVec; HPData_freq.nLaxis = nLaxis_f;
    end
    if flags(3)% doAmp
    HPData_amp = struct( 'HP_a', [], 'R2vecs', [], 'nets',[], 'nNaxis', [], 'nLaxis', []);
    HPData_amp.nNaxis = nNeuronsVec; HPData_amp.nLaxis = nLaxis_a;
    end
    
    % assign vars
    trainIn =  trainSet.inputs;
    testIn = testSet.inputs;
    trainEig = trainSet.outputsEig(:,1:nModesGet); 
    testEig = testSet.outputsEig(:,1:nModesGet);
    
    if ~isWedge
        trainAmp = db(abs(trainSet.outputsAmp(:,1:nModesGet)));
        testAmp = db(abs(testSet.outputsAmp(:,1:nModesGet)));
    else
        frfN = input('select FRF 1 --> H12   2 --> H13   3 --> H15 (1/2/3): ');
        trainAmp = db(abs(trainSet.outputsAmp{frfN}(:,1:nModesGet)));
        testAmp =  db(abs(testSet.outputsAmp{frfN}(:,1:nModesGet)));
    end
    
    tStart = tic;
    for jj = nLayersVec     
        for ii = nNeuronsVec 
           if ismember(jj, nLaxis_f) && flags(2) %doFreq
                [f_R2, fNet] = NN_trainTest_extTest(trainIn,trainEig, testIn, testEig, ii, jj);
                 f_R2 = table2array(f_R2); 
                 R2f{ii}{jj} = f_R2;
                 HPmat_freq(ii,jj) = mean(f_R2);
                 nets_f{ii,jj} =fNet; 
                 disp([' FREQ Neurons: ', int2str(ii), ' Layers: ', int2str(jj), '   mean R2: ', num2str(HPmat_freq(ii,jj))]);
           end

            if ismember(jj,nLaxis_a) && flags(3) % doAmp 
                [a_R2, ampNet] = NN_trainTest_extTest(trainIn, trainAmp , testIn, testAmp, ii, jj);
                 a_R2 = table2array(a_R2);
                 nets_a{ii,jj} = ampNet;
                 R2a{ii}{jj} = a_R2;       
                 HPmat_amp(ii,jj) = mean(a_R2);
                 disp([' AMP Neurons: ', int2str(ii), ' Layers: ', int2str(jj), '   mean R2: ', num2str(HPmat_amp(ii,jj))])
             end
        end
    end
    disp(['elapsed time: ', int2str(floor(toc(tStart)/60)), ' minutes ', int2str(mod(toc(tStart),60)), ' seconds']);
    
    HPData = {};
    % store data routine
    if flags(2) % doFreq
        if flags(4) % saveData
        writeMat2File(HPmat_freq,[ HPfileFreq, '.csv'], {' '}, 1, false);
        save(['HPDATA_freq_', num2str(nModesGet),'modes'], 'R2f', 'nets_f');
        end
        HPData_freq.HP_f= HPmat_freq; 
        HPData_freq.R2vecs = R2f;
        HPData_freq.nets =nets_f;
        HPData{length(HPData)+1} = HPData_freq;
    end    
    if flags(3) %doAmp
        if flags(4) % saveData
        writeMat2File(HPfileAmp, {' '}, 1, false);
        save(['HPDATA_amp_', num2str(nModesGet),'modes'], 'R2a', 'nets_a');
        end
        HPData_amp.HP_a= HPmat_amp; 
        HPData_amp.R2vecs = R2a;
        HPData_amp.nets =nets_a;
        HPData{length(HPData)+1} = HPData_amp;
    end
end
