function [currentVals] = createSetParams(model, referenceVals,standardDev, paramsNames)
%CREATEANDSETPARAMS Summary of this function goes here
%   Detailed explanation goes here
gauss = randn(size(referenceVals));
currentVals = referenceVals.*(ones(size(referenceVals)) + standardDev*gauss);
for ii = (1:length(referenceVals))
    model.param.set(paramsNames(ii), currentVals(ii));
end 
end

