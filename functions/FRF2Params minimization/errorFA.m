function [L2, mode_matching_map] = errorFA(mechParams, fNet, aNet, f0, fAmps,...
    NpeaksAxis, plotData, fixParamsVals, fixParamsIdxs, nFRFs, figN)
% errorFA - objective function of the minimization - predicts frequency and amplitude of FRF peaks for given material
% properties and computes the loss function; allows to evaluate multiple FRFs computed on multiple points
% copyright: David Giuseppe Badiane
% -------------------------------------------------------------------------
% inputs: 
% mechParams    = array with mech params to be optimized (usually less than 15)
% fNet          = neural network object - neural network predicting eigenfrequencies
% aNet          = neural network object - neural network predicting amplitudes
% f0            = nPts x 1 cell - in each cell we have the frequencies of
%                 the peaks of a single FRF
% fAmps         = nPts x 1 cell - in each cell we have the amplitudes of
%                 the peaks of a single FRF
% rho           = 1x1 double - density of the plate
% NpeaksAxis    = nPeaks x 1 double - axis with the FRF peaks considered 
%                                  in the minimization
% plotData      = boolean to decide whether to plot or not
% fixParamsVals = values for the material properties that are not optimized,
%                 (at least density and geometry)
% fixParamsIdxs = indexes of the fixed params in the mechParams array
% nPts          = 1x1 double - number of point FRFs considered 
% figN          = 1x1 double - figure number 
% -------------------------------------------------------------------------
% outputs: 
% L2            = 1x1 double - loss function value
% mode_matching_map          = mode matching btw FRF peaks - NNs eigenfrequencies
% -------------------------------------------------------------------------
    % set nargin cases
    if nargin<12, nFRFs = 1; end
    if nargin<13, figN = 150; end
    
    % preallocate array with all mechanical parameters
    mechParameters = ones(max(fixParamsIdxs),1);
    % fix some mechanical parameters
    mechParameters(fixParamsIdxs) = fixParamsVals;
    % set in mechParameters the ones that are to be optimized
    mechParameters(setdiff(1:max(fixParamsIdxs),fixParamsIdxs)) = mechParams;
    
    % obtain predictions of freq and amp
    % n.b. length(amps) = (nPts * length(freqs)) 
    amps = db2mag(aNet(mechParameters)); 
    freqs = fNet(mechParameters);

    % preset variables before computation of loss function
    count = 0;
    mode_matching_map = [];
    L2s = [];
    nModes = length(freqs);    
    
    % for each point FRF compute loss function
    for ii = 1:nFRFs
        ampsIdxs = count*nModes+1:nModes*(count+1);
        % compute loss function
        [L2, map] = lossFx_FA(freqs, amps(ampsIdxs), f0{ii}, fAmps{ii},...
                              NpeaksAxis, plotData, figN+count);
        % lower bound to elastic constants
        if ~isempty(find(mechParameters(2:7)<1e7))
            L2 = L2 + 1e5; 
        end
        % avoid negative poisson Ratios
        if ~isempty(find(mechParameters(8:10)<0))
            L2 = L2 + 1e5; 
        end
        % store Loss function value for a given FRF
        L2s = [L2s; L2];
        mode_matching_map = [mode_matching_map; map];
        count = count+1;
    end
    % final loss function is sum of the loss function for each FRF
    L2 = sum(L2s);  
%   optionally see loss function value
%   disp(['Loss function value: ', num2str(L2s)]);
end