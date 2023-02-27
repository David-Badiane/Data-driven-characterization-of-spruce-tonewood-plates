function  [R2_NN, net, tr] = ...
    NN_trainTest_extTest(trainIn, trainOut, testIn, testOut, nNeurons, nLayers, ...
    strTitle, plotData, figNumber)
% NN_TRAINTEST_EXTTEST 
% creates and trains a Multilayer Feedforward Neural Network (MFNN),
% evaluates coefficient of determination and saves training data
% copyright: David Giuseppe Badiane
% -------------------------------------------------------------------------
% inputs
% trainIn   = nTuples train x nInputs double  - train set inputs
% trainOut  = nTuples train x nOutputs double - train set outputs
% testIn    = nTuples test x nInputs double   - test set inputs
% testOut   = nTuples test x nOutputs double  - test set inputs
% nNeurons  = int - number of neurons of the MFNN 
% nLayers   = int - number of layers of the MFNN
% strTitle  = string  - title of a figure with a scatter plot of test results
% plotData  = boolean - select whether to plot the scatter plot or not
% figNumber = int - number of the plotted figure
% -------------------------------------------------------------------------
% outputs
% R2_NN     = coefficient of determination of the neural network (testing)
% net       = trained MFNN
% tr        = training data
% -------------------------------------------------------------------------
% automatic setting if there are less input arguments than expected
if (nargin <= 6 || nargin <= 7 || nargin <= 8) 
    strTitle = ' ';
    plotData = false;
    figNumber = 100;
end 
% create neural network
netLayers = ones(1,nLayers);
netVector = nNeurons*netLayers;
net = feedforwardnet(netVector);
% transfer function is elliot sigmoid fx
for ii = 1:netLayers
    net.layers{ii}.transferFcn = 'elliotsig';
end
% train neural network with Matlab NNTRAINTOOL
[net,tr] = train(net,trainIn.',trainOut.');
% get outputs for test sets
y = net(testIn.').';

% generate names for table
names = {};
for ii = 1:length(y(1,:))
   names{ii} = [strTitle(1), int2str(ii)];
end
% compute the coefficient of determination
[R2] = computeR2(testOut, y);
% put the result in a table
R2_NN = array2table(round(R2,4), 'VariableNames', names);

% generate a scatter plot
% commented you can find the commands to plot the training data
if plotData
    %plotperform(tr); %plottrainstate(tr);
    figure(figNumber)
    for ii = 1:length(y(1,:))
        subplot(2,4,ii)
        scatter(testOut(:,ii), y(:,ii));
        hold on
        plot([min(testOut(:,ii)) max(testOut(:,ii))], [min(testOut(:,ii)) max(testOut(:,ii))]);
        xlabel('test Out');
        ylabel('NN prediction');
        ax = gca;
        ax.FontSize = 13;
        xlim([min(testOut(:,ii)) max(testOut(:,ii))]); ylim([min(testOut(:,ii)) max(testOut(:,ii))]);
    end
end

end



