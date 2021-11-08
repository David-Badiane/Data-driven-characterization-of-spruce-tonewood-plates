function [img] = seeHistDataset_plate(baseFolder, nModesSee, startWithMode, step, csvName, saveDir, saveName)
if nargin<4, step = 1; end
if nargin<6, saveData = false; else, saveData = true; end

if nargin <5
    [Dataset_FA, csvPath, datasetType, datasetDistr] = fetchDataset(saveData, baseFolder, true);
else
    [Dataset_FA, csvPath, datasetType, datasetDistr] = fetchDataset(saveData, baseFolder, true, csvName);
end

Dataset_FA.outputsEig = Dataset_FA.outputsEig(:,Dataset_FA.modesIdxs);
Dataset_FA.outputsAmp = Dataset_FA.outputsAmp(:,Dataset_FA.modesIdxs);

peakNames = {};
nPoints = 25;
freqFig = 100;
ampFig = freqFig+1;
yLength = 650;
xLength = 1.618*yLength;
figure(freqFig); clf reset; set(gcf, 'Position',  [0, 50, xLength, yLength]);
figure(ampFig); clf reset; set(gcf, 'Position',  [0, 50, xLength, yLength]);

count = 1;
modesAxis = startWithMode:step:(startWithMode + nModesSee*step - 1);
lineStyle = {'-' ,'--' ,':' ,'-.'};

smoothing = 0.625;

lWidth = 2;
fSize = 24;

minF = min(Dataset_FA.outputsEig(:,modesAxis), [], 'all');
maxF = max(Dataset_FA.outputsEig(:,modesAxis), [], 'all');

for ii = modesAxis
    figure(freqFig)
    notNanF = ~isnan(Dataset_FA.outputsEig(:, ii));
    hold on
    [N,edges] = histcounts(Dataset_FA.outputsEig(notNanF, ii),nPoints);
    edges = edges(2:end) - (edges(2)-edges(1))/2;
    N = lowpass(N,smoothing);
    H=area(edges,N, 'LineStyle', lineStyle{mod(count, length(lineStyle))+1}, 'lineWidth', lWidth);
    H.FaceAlpha = 0.3;
    hold off 
    if ii >1, mmax = max([N,mmax]); else, mmax = max([N]); end
    figure(ampFig)
    notNanA = ~isnan(Dataset_FA.outputsEig(:, ii));
    hold on
    [N,edges] = histcounts(db(Dataset_FA.outputsAmp(notNanA,ii)),nPoints);
    edges = edges(2:end) - (edges(2)-edges(1))/2;
%     N = lowpass(N,smoothing);
    H=area(edges,N, 'LineStyle', lineStyle{mod(count, length(lineStyle))+1}, 'lineWidth', lWidth);
    H.FaceAlpha = 0.3;
    hold off
    
    count = count + 1;
end


% plot style handler;
figure(freqFig)
ylim([0,mmax]);
hold off; ax = gca; ax.FontSize = 0.7*fSize;
xx = xlabel('$frequency$ $[Hz]$', 'fontSize', fSize); yy = ylabel('N occurences');
set(xx, 'interpreter', 'latex', 'fontSize', fSize); set(yy, 'interpreter', 'latex', 'fontSize', fSize);
set(gca,'ColorOrder', [cos(linspace(0,0.45*pi,nModesSee).'), abs(sin(linspace(0, 2*pi,nModesSee).')), sin((linspace(0 , pi/3,nModesSee).'))]);
box on;
xlim([minF, maxF+25]);
ax = gca; ax.TickLength = [.013,.013];
ax.LineWidth = 1.3;
ax.XMinorTick = 'on';
legenda = {};
legendAxis = startWithMode:step:startWithMode+nModesSee;
for ii = 1:length(legendAxis)
    leg = Dataset_FA.modesOrder{legendAxis(ii)};
    legenda{ii} = ['$' leg(1) '_{' leg(2) ',' leg(3) '}$'];
end
ll = legend(legenda{:}, 'interpreter', 'latex', 'fontSize', fSize*0.9);
set(ll,'Box','off')
ax.TickLabelInterpreter = 'latex';


figure(ampFig)
xx = xlabel('$|H(f_i)|_{dB}$'); yy = ylabel('N occurences');
set(xx, 'interpreter', 'latex'); set(yy, 'interpreter', 'latex');
hold off; ax = gca; ax.FontSize = fSize;
set(gca,'ColorOrder', [cos(linspace(0,0.45*pi,nModesSee).'), abs(sin(linspace(0, 2*pi,nModesSee).')), sin((linspace(0 , pi/3,nModesSee).'))]);
ll = legend(legenda{:}, 'interpreter', 'latex', 'fontSize', 0.9*fSize);
set(ll,'Box','off')

box on;
if saveData
    cd(saveDir)
    fig = figure(freqFig);
    saveas(fig, [saveName,'.png']);
end
ax.TickLabelInterpreter = 'latex';
end