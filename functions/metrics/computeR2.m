function [R2] = computeR2(observedData,predictedData)
% COMPUTER2 
% -------------------------------------------------------------------------
% this function allows to compute the coefficient (R2) of determination between
% observed data and predicted data. 
% -------------------------------------------------------------------------
% The R2 says us how much of the standard deviation of the observed 
% data is represented in the predicted data, yielding a metric to
% measure the accuracy of the predicting model.
% -------------------------------------------------------------------------
%   observedData  = nTuples x nCols double - array with observed data
%   predictedData = nTuples x nCols double - array with data predicted by a regressor/predictor
% -------------------------------------------------------------------------
%   R2 = 1xnCols double - coefficient of determination
% -------------------------------------------------------------------------
nCols = length(observedData(1,:)); % number of columns of the data
R2 = []; % preallocate R2

for ii = 1:nCols
    % discard, if present, NaNs
    notNanIdxs = find(~isnan(predictedData(:,ii)) & ~isnan(observedData(:,ii)) & ~isinf(predictedData(:,ii)) & ~isinf(observedData(:,ii)));
    % compute R2
    % step 1: compute residuals --> difference between observed data and predicted data
    residual = observedData(notNanIdxs,ii) - predictedData(notNanIdxs,ii);
    % compute SSE and SST
    SSE = sum(residual.^2);
    SST = sum( (observedData(notNanIdxs,ii) - mean(observedData(notNanIdxs,ii))).^2);
    % R2 is defined as follows
    R2(ii) = 1 - SSE/SST;
end


end

