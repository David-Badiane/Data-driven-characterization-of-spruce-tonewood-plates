function [R2] = computeR2(observedData,predictedData)
%COMPUTER2 Summary of this function goes here
%   Detailed explanation goes here
nModes = length(observedData(1,:));
R2 = [];
for ii = 1:nModes
    notNanIdxs = find(~isnan(predictedData(:,ii)) & ~isnan(observedData(:,ii)));
    residual = observedData(notNanIdxs,ii) - predictedData(notNanIdxs,ii);
    SSE = sum(residual.^2);
    SST = sum( (observedData(notNanIdxs,ii) - mean(observedData(notNanIdxs,ii))).^2);
    R2(ii) = 1 - SSE/SST;
end


end

