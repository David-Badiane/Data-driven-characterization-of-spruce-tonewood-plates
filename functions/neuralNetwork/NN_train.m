function [net] = NN_train(inputs, outputs, nNeurons, nLayers)
    netLayers = ones(1,nLayers);
    netVector = nNeurons*netLayers;
    net = feedforwardnet(netVector);
    for ii = 1:netLayers
    	net.layers{ii}.transferFcn = 'logsig';
    end
    net = train(net,inputs.',outputs.');
end