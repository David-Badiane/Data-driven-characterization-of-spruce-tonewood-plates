function [L2, L2_freq] = errorFrequency(mechParams, fNet, f0, rho, NpeaksAxis)
%ERRORFA Summary of this function goes here
%   Detailed explanation goes here
    
    mechParameters = [rho; mechParams];
    fNN = fNet(mechParameters);
    fNN = fNN(1:12);

    pointsNN = [fNN];
    pointsReal = [f0(NpeaksAxis)];
         
    map = [];
    distances = zeros(length(NpeaksAxis));
    
    gains = [2.75,1,1,1.75,1.75].';
    %gains = [3,0.5,0.5,1,2].';
    gains = ones(size(gains));
    hold off;
    
    L2_freq = abs(gains .* ((pointsReal(NpeaksAxis,1) - pointsNN(NpeaksAxis,1))) );
    L2 = sum(L2_freq)/sum(gains);
    
   
end

