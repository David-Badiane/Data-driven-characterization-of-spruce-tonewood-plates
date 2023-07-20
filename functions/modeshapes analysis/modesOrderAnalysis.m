function [appears,maxLoc] = modesOrderAnalysis(nModes, csvPath, modesFilename, nonOutliers)
% MODESORDERANALYSIS
% this function says us how many times a mode appears in a given ascending order
% in other words it says us how much times ex. f11 will be the first or
% second eigenfrequency in the dataset
% copyright: David Giuseppe Badiane
% -------------------------------------------------------------------------
% inputs: 
%   nModes = int - number of modes taken into account;
%   csvPath = string - path of the dataset directory
%   modesFilename = string - fileName of the modeshapes file
%   nonOutliers   = boolean array - says us which tuples of the raw dataset
%                 are not outliers (repeated labels - NCC<0.9 - Poisson plates)
% -------------------------------------------------------------------------
% outputs:   
%   appears = int array - map saying us how much times a given mode appears
%                       on a given dataset column (i.e. how much f11 is the 1st peak or 2nd peak)
%   maxLoc  = int - the index of the modes Ordering most frequent in the dataset 
% -------------------------------------------------------------------------
    cd(csvPath);
    modesNames = table2cell(readtable(modesFilename));
    modesNames = modesNames(nonOutliers, :);
    appears = zeros(length(modesNames(:,1)),nModes);
    
    for jj = 1:length(modesNames(:,1))
        for ii =  1:length(appears(1,:))
            exact_match_mask = strcmp(modesNames(:,ii), modesNames{jj,ii});
            appears(jj,ii) = length(find(exact_match_mask));
        end
    end

    [maxVal, maxLoc] = max(mean(appears.')); % takes the tuple with highest mean appears
    writeMat2File(appears, 'modesOrdering.csv', {'f'}, 1, false);
    writeMat2File(modesNames, 'modesNames_nonOutliers.csv', {'f'}, 1, false);

end

