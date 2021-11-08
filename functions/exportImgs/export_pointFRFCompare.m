function [img] = export_pointFRFCompare(FRFData, imgData, imgMode, minPeakWidth)

% FRFData : cutLow, cutHigh, Hv, nPeaksUsed, fAxisComsol, vel, alpha, beta
% ImgData : xyLabel, lineWidth, annotFontSize, axFontSize, legend,
%           areaColor, areaAlpha, errTextPos, paramsTextPos

% cut frequency axis
if strcmp(imgMode, 'db'), plotDb = true; else plotDb = false; end

fAxis_cut = intersect(find(FRFData.fAxis>=FRFData.cutLow),...
                      find(FRFData.fAxis <= FRFData.cutHigh));
fAxisPlot = FRFData.fAxis(fAxis_cut);

fAxisComsol_cut = intersect(find(FRFData.fAxisComsol>=FRFData.cutLow),...
                            find(FRFData.fAxisComsol<=FRFData.cutHigh));
fAxisComsol = FRFData.fAxisComsol(fAxisComsol_cut);
fAxisComsol = fAxisComsol(:).';

Hvplot = FRFData.Hv(fAxis_cut);
vel = FRFData.vel(fAxisComsol_cut);

% resample real frf axis
fPlot_ind = [];
for ii = 1:length(fAxisComsol)
    diff = abs(fAxisPlot - fAxisComsol(ii));
    [minV, minLoc] = min(diff);
    fPlot_ind = [fPlot_ind minLoc];
end
Hvplot = Hvplot(fPlot_ind);
fAxisPlot = fAxisPlot(fPlot_ind);

% normalize
if plotDb
    normRealFRF =  db(abs(Hvplot)/max(abs(Hvplot))); 
    normComsolFRF =  db(abs(vel)/max(abs(vel)));
else
    normRealFRF =  abs(Hvplot)/max(abs(Hvplot));
    normComsolFRF =  abs(vel)/max(abs(vel)); 
end

normRealFRF = normRealFRF(:).'; normComsolFRF = normComsolFRF(:).';

% integral error computed by trapezoidal scheme
realFRFint = trapz(fAxisPlot, normRealFRF);
comsolFRFint = trapz(fAxisComsol, normComsolFRF);
integralError = (realFRFint - comsolFRFint)./realFRFint;

% peak analysis
if plotDb
    [maxVComsol, maxLocsComsol] = findpeaks(abs(1./normComsolFRF));
    [maxVReal, maxLocs] = findpeaks(abs(1./normRealFRF), 'minPeakWidth', minPeakWidth);
    maxVComsol = normComsolFRF(maxLocsComsol);
    maxVReal = normRealFRF(maxLocs);
else
    [maxVComsol, maxLocsComsol] = findpeaks(normComsolFRF);
    [maxVReal, maxLocs] = findpeaks(normRealFRF, 'minPeakWidth', minPeakWidth);
end

f0Comsol = fAxisComsol(maxLocsComsol); f0Comsol = f0Comsol(:).';
f0 = fAxisComsol(maxLocs); f0 = f0(:).';

% start plotting 
img = figure(imgData.imgN);
clf reset; box on;
set(gcf, 'Position',  [0, 50, imgData.xLengthImg, imgData.yLengthImg]);

hold on;
plot(fAxisPlot,normRealFRF, 'LineWidth',imgData.lineWidth);
plot( fAxisComsol, normComsolFRF,'-.', 'LineWidth',imgData.lineWidth);     
stem(f0, maxVReal);
stem(f0Comsol, maxVComsol);

xx = xlabel(imgData.xyLabels{1}); set(xx, 'Interpreter', 'Latex');
yy = ylabel(imgData.xyLabels{2}); set(yy, 'Interpreter', 'Latex');
    
xlim([FRFData.cutLow, FRFData.cutHigh*1.25]);

textErr = ['int err = ', num2str(abs(round(100*integralError,1))), '%'];
annotation('textbox',imgData.errTextPos,  'String',textErr,'EdgeColor','none', 'fontSize', 16)

textRay = ['alpha = ', num2str(FRFData.alpha), newline,...
           'beta = ', num2str(FRFData.beta) , newline,...
           'nR = '   , int2str(FRFData.nRealizations)];
annotation('textbox',imgData.paramsTextPos,'String',textRay,'EdgeColor','none', 'fontSize', 16)

x2 = [fAxisComsol, fliplr(fAxisComsol)];
inBetween = [normRealFRF, fliplr(normComsolFRF)];
area = fill(x2, inBetween, 'g', 'lineStyle', 'none');
area.FaceColor = imgData.areaColor; 
area.FaceAlpha = imgData.areaAlpha;

ax = gca; ax.FontSize = imgData.axFontSize; ax.XMinorTick = 'on'; ax.YMinorTick = 'on';
ll = legend(imgData.legenda{:});
set(ll,'Box', 'off'); set(ll, 'Interpreter', 'latex'); set(ll, 'FontSize', 14);
end