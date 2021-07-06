
for nn = 1:12
    figure()
indexParam = nn;
name = mechParamsNames{indexParam};

paramVar = Dataset_FA.inputs(:,indexParam);
paramAxis = linspace(min(paramVar), max(paramVar), 200);
for ii = 2:5
hold on
subplot(2,2,ii-1)
freq = [];

for kk = 1:length(paramAxis)
pars = referenceVals;
pars(indexParam) = paramAxis(kk);
freqPredicted = fNet(pars.').';
freq = [freq; freqPredicted(1:5)];
end
plot(paramAxis, freq(:,ii), '.');
end

sgtitle(name);

end