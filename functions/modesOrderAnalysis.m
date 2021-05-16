function [appears,maxLoc] = modesOrderAnalysis(nModes, csvPath, modesFilename)
% MODESORDERANALYSIS Summary of this function goes here
%   Detailed explanation goes here
    cd(csvPath);
    modesNames = table2cell(readtable(modesFilename));
    appears = zeros(length(modesNames(:,1)),nModes);
    for jj = 1:length(modesNames(:,1))
        for ii =  1:length(appears(1,:))
            exact_match_mask = strcmp(modesNames(:,ii), modesNames{jj,ii});
            appears(jj,ii) = length(find(exact_match_mask));
        end
    end

    [maxVal, maxLoc] = max(mean(appears.'));
    writeMat2File(appears, 'modesOrdering.csv', {'f'}, 1, false);
end

