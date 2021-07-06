function [linearModels,multilinCoeffs, R2, errors, predictedOutputs] = multRgrOutliers(inputsInfo,outputsALLInfo, toRemove, nModes)
%MULTILINEARREGRESS Performs multilinear regression on input and output data
%   Detailed explanation goes here
 
    linearModels = cell(nModes,1);
    multilinCoeffs = zeros(11,nModes);
    R2 = zeros(1,10);

    for ii = 1:nModes
        linearModels{ii} = fitlm(inputsInfo, outputsALLInfo(:,ii), 'Exclude', toRemove{ii});   
        multilinCoeffs(:,ii) = table2array(linearModels{ii}.Coefficients(:,1));
        R2(ii) = linearModels{ii}.Rsquared.Adjusted;
    end

    [displayIdxs, predictedOutputs] = predictEigRemove(linearModels , inputsInfo, toRemove);
    
    
    figure()
    for jj = 1:length(linearModels)
        
    normOuts = outputsALLInfo(displayIdxs{jj},jj)./mean(outputsALLInfo(displayIdxs{jj},jj));
    normPredOuts =  predictedOutputs{jj}./mean(predictedOutputs{jj});

        
        subplot(5,5,jj)
        s = scatter(normOuts,normPredOuts);
        s.AlphaData = 10;
        xlabel('Actual');
        ylabel('Predicted');
        hold on;
        plot( linspace(0.8,1.25,1000), linspace(0.8,1.25,1000), 'lineWidth', 1.5);
        title(['f',int2str(jj)]);
        %legend(['f', num2str(ii)], 'y=x line');
    end
    
    
    errors = zeros(nModes,1);
    for ii = 1:nModes
       errors(ii) = NMSE(outputsALLInfo(displayIdxs{ii},ii), predictedOutputs{ii}, ii); 
    end
    
    R2Names = cell(nModes,1);
    for ii = 1:length(R2Names)
        R2Names{ii} = ['f',int2str(ii)];
    end
    
    R2 = array2table(R2, 'VariableNames', R2Names);
end

