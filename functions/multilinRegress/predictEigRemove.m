function [idxs, predictedOutputs] = predictEigRemove(linearModels , inputsInfo, toRemove)
%PREDICTEIGENFREQUENCIES Summary of this function goes here
%   With this function we predict all the eigenfrequencies of the system

    predictedOutputs = cell(length(linearModels),1);
    indexInputs = 1:length(inputsInfo(:,1));
    idxs = cell(length(linearModels),1);
    
    for jj = 1:length(linearModels)
        indexes = indexInputs; 
        indexes(toRemove{jj}) = [];
        idxs{jj} = indexes;
        inputsNoOutliers = inputsInfo(indexes,:);
        predictedOutputs{jj} = zeros(size(inputsNoOutliers(:,1)));
        for ii = 1:length(inputsNoOutliers(:,1))
            predictedOutputs{jj}(ii) = feval(linearModels{jj},inputsNoOutliers(ii,:));
        end
    end
end

