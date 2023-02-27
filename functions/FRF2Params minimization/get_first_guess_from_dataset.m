function parameters = get_first_guess_from_dataset(Dataset_FA, fReal, aReal, fNet, aNet,...
                                                density, geometry, damping, sampleSize,dataset_centervals)

nTuples = size(Dataset_FA.inputs,1);
NpeaksAxis = 1:12;
L2s = zeros(nTuples,1);
tuplesAxis =[1 randsample(1:nTuples, sampleSize-1)];
t = tic;
disp(['Looking for a first guess:',newline, 'pipeline',newline,...
      '- for [density mechParams <-- (we vary this) damping geom]'...
      '- compute loss function for a random sample of ' num2str(sampleSize) ' dataset tuples',...
      newline, '-use as first guess lowest L2 score of the sample'])
for ii = 1:length(tuplesAxis)
    tuple = tuplesAxis(ii);
    if ii == 1
      parameters = [density dataset_centervals(tuple,2:10) damping geometry]; 
    else
      parameters = [density Dataset_FA.inputs(tuple,2:10) damping geometry]; 
    end
    fEst = fNet(parameters.');
    aEst =  db2mag(aNet(parameters.'));
    [L2, map] = lossFx_FA(fEst, aEst, fReal, aReal, NpeaksAxis, 0, 1);
    L2s(ii) = L2;
    if mod(ii,100) == 0
        disp(['evaluating tuple n°' num2str(ii) '  -->  tuple ' num2str(tuplesAxis(ii))])
        disp(['time elapsed --> ' num2str(toc(t),2) 'seconds']);
    end
end

[minVal, minLoc] = min(L2s);
parameters = [density Dataset_FA.inputs(minLoc,2:10) damping geometry];

end