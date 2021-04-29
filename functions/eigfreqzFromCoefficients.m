function [f_out] = eigfreqzFromCoefficients(mechParameters, linearModels, nModes)
%EIGFREQZFROMCOEFFICIENTS Summary of this function goes here
%   Detailed explanation goes here
    f_out = zeros(nModes,1);
    if length(mechParameters(1,:)) ==1
        mechParameters = mechParameters.';
    end

    for jj = 1:nModes
            f_out(jj) = feval(linearModels{jj},mechParameters);
    end
end

