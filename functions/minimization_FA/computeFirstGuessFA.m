function [L2, firstGuessLoc] = computeFirstGuessFA(Dataset_FA, fAmps, f0,...
                               NpeaksAxis, plotData, normalization, ampScale)
%COMPUTELOSSFUNCTIONS Summary of this function goes here
%   Detailed explanation goes here
    %% 1) get Data

    nTuples = length(Dataset_FA.inputs(:,1));                
    L2_raw =    zeros(nTuples , 1);
    
    %% Loss function calculations
   
    for ii = 1:nTuples
        freqData = Dataset_FA.outputsEig(ii, ~isnan(Dataset_FA.outputsEig(ii,:)));
        ampData = abs(Dataset_FA.outputsAmp(ii, ~isnan(Dataset_FA.outputsAmp(ii,:))));
        
        [L2, L2_freq, L2_amp, map] = lossFx_FA(freqData, ampData, f0, fAmps, 1:5, false, normalization, ampScale);
        L2_raw(ii) = L2;
    end
        
    [minVal, firstGuessLoc] = min(L2_raw);
    if plotData
        ii = firstGuessLoc;
            freqData = Dataset_FA.outputsEig(ii, ~isnan(Dataset_FA.outputsEig(ii,:)));
            ampData = abs(Dataset_FA.outputsAmp(ii, ~isnan(Dataset_FA.outputsAmp(ii,:))));  
            [L2, L2_freq, L2_amp, map] = lossFx_FA(freqData, ampData, f0, fAmps, psi, NpeaksAxis, true);
            title('first Guess');
            pause(0.01);
    end
end

