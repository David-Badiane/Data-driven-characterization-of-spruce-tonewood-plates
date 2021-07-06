function [linearModels,multilinCoeffs, R2, errors, predictedOutputs] = multilinearRegress(inputsInfo,outputsALLInfo, nModes,name, referenceVals )
%MULTILINEARREGRESS Performs multilinear regression on input and output data
%   Detailed explanation goes here
    
    outNames = cell(nModes,1);
    for ii = 1:nModes
       outNames{ii} = [name,'_{', int2str(ii),'}']; 
    end
    
    nRegressors = length(referenceVals) +1;
    linearModels = cell(nModes,1);
    multilinCoeffs = zeros(nRegressors,nModes);
    R2 = zeros(1,nModes);

    trainIdxs = 1:floor(0.9*length(inputsInfo(:,1)));
    testIdxs = trainIdxs(end)+1 : length(inputsInfo(:,1));
    
    for ii = 1:nModes
        linearModels{ii} = fitlm(inputsInfo(trainIdxs,:), outputsALLInfo(trainIdxs,ii));   
        multilinCoeffs(:,ii) = table2array(linearModels{ii}.Coefficients(:,1));
        R2(ii) = linearModels{ii}.Rsquared.Adjusted;
    end

    [predictedOutputs] = predictEigenfrequencies(linearModels , inputsInfo(testIdxs,:));
    
    normOuts = outputsALLInfo(testIdxs,:)./mean(outputsALLInfo(testIdxs,:));
    normPredOuts =  predictedOutputs./mean(predictedOutputs);
     
    figure()
    s = scatter(normOuts,normPredOuts,50, '.');
    xlabel('Actual');
    ylabel('Predicted');
    hold on;
    plot( [min(normPredOuts, [], 'all'), max(normPredOuts, [], 'all')], [min(normPredOuts, [], 'all'), max(normPredOuts, [], 'all')], 'lineWidth', 1.5);
    legend(outNames{:}, 'y=x line');
    ax = gca;
    ax.XMinorTick = 'on';
    ax.YMinorTick = 'on';
    ax.TickDir = 'out';
    ax.FontSize = 20;

    
    figure()
    for ii = 1:length(normPredOuts(1,1:10))
        if ii == 3 || ii == 4
        subplot(2,1,ii-2)
        s = scatter(normOuts(:,ii),normPredOuts(:,ii),50, '.');
        s.AlphaData = 0.5;
        xlabel('Actual');
        ylabel('Predicted');
        hold on;
        checkLine = linspace(min(normPredOuts(:,ii)),max(normPredOuts(:,ii)),1000);
        plot(checkLine , checkLine, 'lineWidth', 1.5);
        title(outNames{ii});
        ax = gca;
        ax.XMinorTick = 'on';
        ax.YMinorTick = 'on';
        ax.TickDir = 'out';
        ax = gca;
        ax.FontSize = 14;

        end
        %legend(['f', num2str(ii)], 'y=x line');
    end
    
    
    errors = zeros(nModes,1);
    for ii = 1:nModes
       errors(ii) = NMSE(outputsALLInfo(testIdxs,ii), predictedOutputs(:,ii)); 
    end
    R2Names = cell(nModes,1);
    for ii = 1:length(R2Names)
        R2Names{ii} = [name(1), int2str(ii)];
    end
    
    R2 = array2table(round(R2,3), 'VariableNames', R2Names);
end

