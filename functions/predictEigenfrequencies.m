function [predictedOutputs] = predictEigenfrequencies(linearModels , inputsInfo)
%PREDICTEIGENFREQUENCIES Summary of this function goes here
%   With this function we predict all the eigenfrequencies of the system

    predictedOutputs = zeros(size( inputsInfo ));

    for ii = 1:length(inputsInfo(:,1))
        for jj = 1:length(linearModels)
            predictedOutputs(ii,jj) = feval(linearModels{jj},inputsInfo(ii,:));
        end
    end
end

