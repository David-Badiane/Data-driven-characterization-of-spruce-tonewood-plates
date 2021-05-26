function [R2, netVector] = NN_trainTest(inputs, outputs, nNeurons, nLayers, strTitle)
%NN_TRAINTEST Summary of this function goes here
%   Detailed explanation goes here
close all

netLayers = ones(1,nLayers);
netVector = nNeurons*netLayers;

net = feedforwardnet(netVector);

for ii = 1:netLayers
    net.layers{ii}.transferFcn = 'logsig';
end


trainIdxs = 1:floor(0.9*length(inputs(:,1)));
testIdxs = trainIdxs(end)+1 : length(inputs(:,1));

linearModels = {};
R2_neuralNetwork = [];

names = {};
for ii = 1:20
out_train = outputs(trainIdxs,ii);
out_test = outputs(testIdxs,ii);

net = train(net,inputs(trainIdxs,:).',out_train.');
inputTest = inputs(testIdxs,:);

y = net(inputTest.');

figure(ii)
scatter( out_test.', y);
hold on 
plot([min(y, [], 'all'), max(y, [], 'all') ], [min(y, [], 'all'), max(y, [], 'all') ] );
xlabel('test data');
ylabel('predicted data');
title([strTitle, int2str(ii)]);

linearModels{ii} = fitlm(out_test, y);   
R2(ii) = linearModels{ii}.Rsquared.Adjusted;
names{ii} = ['f', int2str(ii)];
end

R2_NN = array2table(round(R2,4), 'VariableNames', names)
end

