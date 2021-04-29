function [linearModels,multilinCoeffs, R2, errors, predictedOutputs] = multRgrOutliers(inputsInfo,outputsALLInfo, toRemove)
%MULTILINEARREGRESS Performs multilinear regression on input and output data
%   Detailed explanation goes here
 
    linearModels = cell(10,1);
    multilinCoeffs = zeros(11,10);
    R2 = zeros(1,10);

    for ii = 1:10
        linearModels{ii} = fitlm(inputsInfo, outputsALLInfo(:,ii), 'Exclude',toRemove{ii});   
        multilinCoeffs(:,ii) = table2array(linearModels{ii}.Coefficients(:,1));
        R2(ii) = linearModels{ii}.Rsquared.Adjusted;
    end

    [predictedOutputs] = predictEigenfrequencies(linearModels , inputsInfo);

    
    normOuts = outputsALLInfo./mean(outputsALLInfo);
    normPredOuts =  predictedOutputs./mean(predictedOutputs);
    
    figure()
    scatter(normOuts,normPredOuts, 'AlphaData', 10);
    xlabel('Actual');
    ylabel('Predicted');
    hold on;
    plot( linspace(0.8,1.25,1000), linspace(0.8,1.25,1000), 'lineWidth', 1.5);
    legend('f1','f2','f3','f4','f5','f6','f7','f8','f9','f10', 'y=x line');

    
    figure()
    for ii = 1:length(normPredOuts(1,:))
        subplot(2,5,ii)
        s = scatter(normOuts(:,ii),normPredOuts(:,ii));
        s.AlphaData = 10;
        xlabel('Actual');
        ylabel('Predicted');
        hold on;
        plot( linspace(0.8,1.25,1000), linspace(0.8,1.25,1000), 'lineWidth', 1.5);
        legend(['f', num2str(ii)], 'y=x line');
    end
    
    
    errors = zeros(10,1);
    for ii = 1:10
       errors(ii) = NMSE(outputsALLInfo(:,ii), predictedOutputs(:,ii), ii); 
    end
    R2Names = {'f1' 'f2' 'f3' 'f4' 'f5' 'f6' 'f7' 'f8' 'f9' 'f10'};
    R2 = array2table(R2, 'VariableNames', R2Names);
end

