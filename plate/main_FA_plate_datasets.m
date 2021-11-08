%% 5) NN and MLR over Gsigma test sets
dDirs = {'csv_plate_gaussian', 'csv_plate_uniform_0.50','csv_plate_uniform_0.75'};
stds = 0.1:0.1:0.4;
R2s_amp = [];
R2s_freq = [];
modesGet = 4; getOrdered = true;

for ii = 1:length(dDirs)
    [Dataset_FA, csvPath, datasetType, datasetDistr, HPFolder] = ...
        fetchReduceDataset_plate(baseFolder, modesGet, getOrdered, dDirs{ii});
    cd([baseFolder, '\', dDirs{ii}]);
    load('optNNs');
    load('MLRfreqs');
    for jj = 1:length(stds)
       testFileName = ['test_G_',num2str(stds(jj)),'std_ordered.mat'];
       load([testPath,'\',testFileName]);
       pred_amp = sim(aNet, testSet.inputs.').';
       pred_freq = predictEigenfrequencies(linMdls , testSet.inputs, 4);
       real_amp = db(abs(testSet.outputsAmp(:, Dataset_FA.modesIdxs)));
       R2s_amp(ii,jj) = mean(computeR2(real_amp, pred_amp));
        actual = testSet.outputsEig(:, Dataset_FA.modesIdxs);
        predicted = pred_freq;
        R2s_freq(ii,jj) = mean(computeR2(actual,predicted));
    end
end

xLengthImg = 1200; yLengthImg = 0.63*xLengthImg;
xLabel = '$ \sigma $ Test Set'; yLabel = '$\overline{R^2}$';
lineType = {'-o' '-d' '-s'};
legenda = {'$G_{10}$' '$U_{50}$' '$U_{75}$'};
legendPos = [0.2 0.28 0.1 0.2];
lineWidth = 2.1;
markerSize = 14;
fontSize = 40;
imgN = 140;
saveDir = 'C:\Users\utente\Desktop\polimi\Thesis FRF2Params - violins\paperFigures_NNs\R2test';
saveName = 'R2test_amp';
axis = stds;
saveImg = true;

imgData = defImg_multiplot(xLengthImg, yLengthImg, imgN, xLabel,...
                    yLabel, lineType, lineWidth, markerSize, legenda, legendPos,...
                    fontSize, saveDir, saveName);               
[img] = export_multiplot(axis, R2s_amp, imgData, saveImg);

saveName = 'R2test_freq';
imgData = defImg_multiplot(xLengthImg, yLengthImg, imgN+1, xLabel,...
                    yLabel, lineType, lineWidth, markerSize, legenda, legendPos,...
                    fontSize, saveDir, saveName);
[img] = export_multiplot(axis, R2s_freq, imgData, saveImg);

%% 6) Dataset Distributions figures 

axis = -1:.01:1;
stdsG = [.1 .2 .3 ];
stdsU = [.5 .75];
stds = [stdsG]% stdsU];
distributions = [];
legenda = {}
for ii = 1:length(stds)
    if ismember(ii, 1:length(stdsG))
    distributions(ii,:) = normpdf(axis,0,stds(ii));
    legenda{ii} = ['$G_{', num2str(stds(ii)*100),'}$'];          
    else
    distributions(ii,:) = unifpdf(axis,-stds(ii), stds(ii));
    legenda{ii} = ['$U_{', num2str(stds(4)*100),'}$'];
    end
end

% set img variables
xLengthImg = 700; yLengthImg = xLengthImg*2/3; imgN = 19; 
xLabel = '$x$'; yLabel = '$y$ (pdf)';
faceAlpha = 0.35;
lineStyles = { '--' '-' ':' '-.'}; lineWidth = 2.5;

legendPos = [0.7, 0.55, 0.1, 0.2];
fontSize = 30;
saveDir = 'C:\Users\utente\Desktop\polimi\Thesis FRF2Params - violins\paperFigures_NNs\Distr';
saveName = 'distributions';
saveImg = true; 

imgData = defImg_multiarea(xLengthImg, yLengthImg, imgN, xLabel,...
                    yLabel, faceAlpha, lineStyles, lineWidth, legenda, legendPos,...
                    fontSize, saveDir, saveName);
                
[img] = export_multiarea(axis, distributions, imgData, saveImg);
