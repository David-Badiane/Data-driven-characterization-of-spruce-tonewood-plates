function  [R2_NN, net] = ...
    NN_trainTest_extTest(inputs, outputs, testIn, testOut, nNeurons, nLayers, ...
    strTitle, plotData, figNumber)
%NN_TRAINTEST_EXTTEST 
% creates and trains a feedforward neural network, evaluates statistics
%   Detailed explanation goes here

if nargin <= 6
    strTitle = ' ';
    plotData = false;
    figNumber = 100;
end 

netLayers = ones(1,nLayers);
netVector = nNeurons*netLayers;
net = feedforwardnet(netVector);
for ii = 1:netLayers
    net.layers{ii}.transferFcn = 'logsig';
end
[net,tr] = train(net,inputs.',outputs.');
%plotperform(tr); %plottrainstat(tr);


y = net(testIn.').';
names = {};
for ii = 1:length(y(1,:))
   names{ii} = [strTitle(1), int2str(ii)];
end

[R2] = computeR2(testOut, y);
R2_NN = array2table(round(R2,4), 'VariableNames', names);

if plotData
    figure(figNumber)
    for ii = 1:16
        subplot(4,4,ii)
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



