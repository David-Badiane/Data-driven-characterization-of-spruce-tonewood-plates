function [R2_NN, net, out_test, y] =...
    NN_trainTest_intTest(inputs, outputs, nNeurons, nLayers,nTrain, nTest)
%NN_TRAINTEST Summary of this function goes here
%   Detailed explanation goes here

netLayers = ones(1,nLayers);
netVector = nNeurons*netLayers;

net = feedforwardnet(netVector);
net.outputs{2}.processFcns = {'fixunknowns'};
for ii = 1:netLayers
    net.layers{ii}.transferFcn = 'logsig';
end

datasetIdxs = 1:length(inputs(:,1));
trainIdxs = randsample(datasetIdxs, nTrain);
remainingIdxs = find(ismember(datasetIdxs,trainIdxs) == 0);
testIdxs = randsample(remainingIdxs, nTest);

names = {};
for ii = 1:length(outputs(1,:))
    names{ii} = ['f', int2str(ii)];
end

out_train = outputs(trainIdxs,:);
out_test = outputs(testIdxs,:);

net = train(net,inputs(trainIdxs,:).',out_train.');
inputTest = inputs(testIdxs,:);

y = net(inputTest.').';

[R2] = computeR2(out_test, y);
R2_NN = array2table(round(R2,4), 'VariableNames', names);
end

