function [L2, L2_freq, L2_amp, map] = errorFA(mechParams, MLR_freq, ampNet, f0, fAmps, rho,...
    NpeaksAxis, plotImg, algorithm, rayleighParams, normalization, ampScale, scaleBinds)
%ERRORFA 
% minimization_FAerror function frequency amplitude
% inputs 
% mechParams = array with mech params to be optimized
% fNet = frequency neural network
% f0 = real FRF eigenfrequencies
% fAmps = real FRF amplitudes
% rho = density of the plate
% NpeaksAxis = peaks numbers considered of the real FRF
% plotImg = boolean to decide whether to plot
% algorithm = string, choose if we want to fix alpha and beta;
    mechParams(end) = mechParams(end)/scaleBinds;
    mechParams(end-1) = mechParams(end-1)/scaleBinds;
    
    if nargin < 9 
        algorithm = 'moveRayleighParams';
        rayleigParams = [];
    end
    
    if strcmp(algorithm, 'fixRayleighParams')
        mechParams(end-1) = rayleighParams(1);
        mechParams(end) = rayleighParams(2);
    end
    mechParameters = [rho; mechParams];
    ampNN = db2mag(ampNet(mechParameters));
    fNN = predictEigenfrequencies(MLR_freq.linMdls , mechParameters.', length(ampNN)).';
%     fNN = MLR_freq(mechParameters);
    [L2, L2_freq, L2_amp, map] = lossFx_FA(fNN, ampNN, f0, fAmps, NpeaksAxis, plotImg, normalization, ampScale);
    if ~isempty(find(mechParams<0))
       L2 = L2 + 1e5; 
    end
    
end

