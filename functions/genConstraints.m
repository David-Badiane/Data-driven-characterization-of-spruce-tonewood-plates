function [outputArg1,outputArg2] = genConstraints(curvRatio ,referenceVals)
%GENCONSTRAINTS Summary of this function goes here
%   Detailed explanation goes here
    parabolas = cell{length(referenceVals)};
    referenceVals = referenceVals(:);
    points2fitX = [referenceVals/2, referenceVals ,referenceVals*2];
    points2fitY = [0, -curvRatio, 0];
    for ii = 1:length(parabolas)
        parabolas{ii} = polyfit(subBandFreq,subBand,2);

    end
end

