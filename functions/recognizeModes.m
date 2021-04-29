function [modesNames] = recognizeModes(meshData,modesData,minPeakVal, tupleIndex)
%RECOGNIZEMODES Summary of this function goes here
%   Detailed explanation goes here
nModes = min(size(modesData));
modesNames = cell(nModes,1);
[mins] = obtainMinima(meshData,modesData, minPeakVal, true, nModes);

for ii = 1:nModes
    name = 'not';
%     mins{ii,1}
%     mins{ii,2}
%     mins{ii,3}
%     mins{ii,4}
    % f11 recognizer
    if  length(mins{ii,4}) ==1 && length(mins{ii,2}) ==1  
        name = 'f11';
    end
    
    % f02 recognizer
    if  (isempty(mins{ii,4}) == 1 ) && length(mins{ii,2}) == 2 
        name = 'f02';
    end
    
    % f20 recognizer
    if length(mins{ii,4}) == 2 && (isempty(mins{ii,2}) == 1 )
        name = 'f20';
    end
    
    % f12 recognizer
    if length(mins{ii,4}) == 1 && length(mins{ii,2}) == 2 
        name = 'f12';
    end
    
    % f21 recognizer
    if length(mins{ii,4}) == 2 && length(mins{ii,2}) == 1 
        name = 'f21';
    end
    
    % f03 recognizer
    if (isempty(mins{ii,4}) == 1) && length(mins{ii,2}) == 3 
            name = 'f03';
    end
    
    % f22 recognizer
    if length(mins{ii,4}) == 2 && length(mins{ii,2}) == 2 
       name = 'f22';
    end
    
    % f13 recognizer
    if length(mins{ii,4}) == 1 && length(mins{ii,2}) == 3
        name = 'f13';
    end
    
    
     % f30 recognizer
    if length(mins{ii,4}) == 3 && (isempty(mins{ii,2}) == 1 )
            name = 'f30';
    end
    
    % f31 recognizer
    if length(mins{ii,4}) == 3 && length(mins{ii,2}) == 1 
            name = 'f31';
    end
    
    % f23 recognizer
    if length(mins{ii,4}) == 2 && length(mins{ii,2}) == 3 
            name = 'f23';
    end
    
    if name == 'not'
        a = 1;
    end
   modesNames{ii} = name; 
end

end

