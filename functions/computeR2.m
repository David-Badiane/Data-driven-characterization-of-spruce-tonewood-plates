function [R2] = computeR2(observedData,predictedData)
%COMPUTER2 Summary of this function goes here
%   Detailed explanation goes here
nModes = length(observedData(1,:));
R2 = [];
for ii = 1:nModes
    meanOut = mean(observedData(:,ii));
    SS_tot = sum((observedData(:,ii) - meanOut).^2);
    SS_res = sum((observedData(:,ii) - predictedData(:,ii)).^2);
    R2(ii) = 1 - (SS_res/SS_tot);
end
end

