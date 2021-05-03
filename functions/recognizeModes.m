function [modesNames] = recognizeModes(meshData,modesData,minPeakVal, tupleIndex)
%RECOGNIZEMODES Summary of this function goes here
%   Detailed explanation goes here
nModes = min(size(modesData));
modesNames = cell(nModes,1);
[mins] = obtainMinima(meshData,modesData, minPeakVal, true, nModes);

for ii = 1:nModes
    %name = 'not';
    
    name = ['f', int2str(length(mins{ii,4})), int2str(length(mins{ii,2}))];

   modesNames{ii} = name; 
end

end

