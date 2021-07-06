function [normMultilinCoeffs] = multilinSensitivity(multilinCoeffs,referenceVals, namesRef, freqzRef, paramsIndex, freqzIndex)
%MULTILINSENSITIVITY Summary of this function goes here
%   Detailed explanation goes here

    names = namesRef(paramsIndex);
    freqz = freqzRef(freqzIndex);
    normMultilinCoeffs = multilinCoeffs(2:end,:).*referenceVals';
    figure()
    imagesc(abs(normMultilinCoeffs(paramsIndex,freqzIndex)));
    colorbar()
    axis off;
    hold on
    for ii = 1:length(freqz)
                text(ii,1, freqz{ii});      
    end
    
    for ii = 1:length(names)
       text(0,ii, names{ii});
    end
end

