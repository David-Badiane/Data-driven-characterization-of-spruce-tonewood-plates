function [img,FRAC, NMSE] = export_comparison_FRFs(FRFData, imgData, imgMode, minPeakWidth,...
                                            doStem, Xref, Yref, subplotN)

% function to plot the figure comparing the two FRFs
% copyright: David Giuseppe Badiane
% -------------------------------------------------------------------------
% inputs: 
% FRFData = struct with the data of experimental and simulated FRFs --> see defImg_comparison_FRFs
% imgData = struct with the data of the figure to be generated      --> see defImg_comparison_FRFs
% imgMode      = string - type 'db' to plot in dB, otherwise in linear magnitude
% minPeakWidth = 1x1 double - minimum peak width for findpeaks algorithm
% doStem       = boolean    - select whether to highlight peaks to stems
% Xref         = 1x1 double - position of the figure in X axis
% Yref         = 1x1 double - position of the figure in Y axis
% subplotN     = 1x1 double - number of the subplot if plotting in subplot
% -------------------------------------------------------------------------
% outputs:
% img          = matlab figure
% FRAC         = 1x1 double - evaluation of the FRAC
% cut frequency axis
if nargin < 6, Xref = 0; Yref = 50; end
if nargin < 7, subplotN = [1 1 1]; end

% set if plotting in dB
if strcmp(imgMode, 'db'), plotDb = true; else plotDb = false; end

% A) Manipulate FRFs so that they have the same bandwidth and number of points
% -------------------------------------------------------------------------
% cut the fAxis of both FRFs to the bandwidth [fHigh,fLow]
fAxis_cut = find(FRFData.fAxis>=FRFData.cutLow & FRFData.fAxis <= FRFData.cutHigh);
fAxis = FRFData.fAxis(fAxis_cut);
fAxisComsol_cut = find(FRFData.fAxisComsol>=FRFData.cutLow & FRFData.fAxisComsol<=FRFData.cutHigh);
fAxisComsol = FRFData.fAxisComsol(fAxisComsol_cut);
fAxisComsol = fAxisComsol(:).';
fAxis = fAxis(:).';

% select both FRFs to the selected bandwidth
FRFplot = FRFData.FRF(fAxis_cut);
comsol_FRF = FRFData.comsol_FRF(fAxisComsol_cut);
% make both FRFs row vectors
comsol_FRF = comsol_FRF(:).';
FRFplot = FRFplot(:).';

% resample experimental FRF axis so that experimental FRF has the same
% number of points of the simulated one
fPlot_ind = zeros(size(fAxisComsol));
for ii = 1:length(fAxisComsol)
    diff = abs(fAxis - fAxisComsol(ii)); % take the difference
    [minV, minLoc] = min(diff);          % for each point compute the difference
    fPlot_ind(ii) =  minLoc;      % add the index where the difference is minimal
end

% cut experimental FRF with the obtained indexes
FRFplot = FRFplot(fPlot_ind);
fAxis = fAxis(fPlot_ind);
% -------------------------------------------------------------------------
% B) Normalization
if plotDb % then min max normalization
    normRealFRF =  normMinMax(db(abs(FRFplot))); 
    normComsolFRF =  normMinMax(db(abs(comsol_FRF)));
else % classicNormalization
    normRealFRF =  abs(FRFplot)/max(abs(FRFplot));
    normComsolFRF =  abs(comsol_FRF)/max(abs(comsol_FRF)); 
end
% make both FRFs row vectors
normRealFRF = normRealFRF(:).';
normComsolFRF = normComsolFRF(:).';
% -------------------------------------------------------------------------
% C) perform peak analysis
if plotDb
    [maxVComsol, maxLocsComsol] = findpeaks(abs(1./normComsolFRF));
    [maxVReal, maxLocs] = findpeaks(abs(1./normRealFRF), 'minPeakWidth', minPeakWidth);
    maxVComsol = normComsolFRF(maxLocsComsol);
    maxVReal = normRealFRF(maxLocs);
else
    [maxVComsol, maxLocsComsol] = findpeaks(normComsolFRF);
    [maxVReal, maxLocs] = findpeaks(normRealFRF, 'minPeakWidth', minPeakWidth);
end

f0Comsol = fAxisComsol(maxLocsComsol); 
f0 = fAxisComsol(maxLocs); 

% -------------------------------------------------------------------------
% D) start plotting 
img = figure(imgData.imgN); clf reset; % image number and reset image
% set image position
set(gcf, 'Position',  [Xref, Yref, imgData.xLengthImg, imgData.yLengthImg]);

% if subplotN is different than [1 1 1] init the subplot
if mean(subplotN ~= [1 1 1]) == 1, subplot(subplotN(1), subplotN(2), subplotN(3)); end

hold on; % to plot multiple lines on a single figure

% plotting normalized experimental FRF
plot(fAxis,normRealFRF, 'LineWidth',imgData.lineWidth);
% plotting normalized simulated FRF
plot( fAxisComsol, normComsolFRF, 'LineWidth',imgData.lineWidth);
% if specified so, add stems on each peak
if doStem, stem(f0, maxVReal); stem(f0Comsol, maxVComsol); end
% add x and y labels
xx = xlabel(imgData.xyLabels{1}); set(xx, 'Interpreter', 'Latex');
yy = ylabel(imgData.xyLabels{2}); set(yy, 'Interpreter', 'Latex');

% fill the area between the two FRFs    
x2 = [fAxisComsol, fliplr(fAxisComsol)];
inBetween = [normRealFRF, fliplr(normComsolFRF)];
area = fill(x2, inBetween, 'g', 'lineStyle', 'none');
area.FaceColor = imgData.areaColor; 
area.FaceAlpha = imgData.areaAlpha;
% set the fontsize of the axis and add x and y minor ticks
ax = gca; ax.FontSize = imgData.axFontSize; ax.XMinorTick = 'on'; ax.YMinorTick = 'on';

% add legend
ll = legend(imgData.legenda{:}); 
set(ll,'Box', 'off'); set(ll, 'Interpreter', 'latex'); set(ll, 'FontSize', imgData.axFontSize);

% choose x and y scale (log or linear)
ax.XScale = imgData.xScale;
ax.YScale = imgData.yScale;
box on;
% compute frequency response assurance criterion - (frac)
FRAC = frac(FRFplot.', comsol_FRF.');
NMSE = nmse(normRealFRF, normComsolFRF);
end

function [normF] = normMinMax(fun)
    normF = (fun - min(fun))./(max(fun) - min(fun));
end

function [FRAC] = frac(FRF_a, FRF_b)
    FRAC = ( abs(FRF_a' * FRF_b).^2 ) ./ ( (FRF_a' * FRF_a) .* (FRF_b' * FRF_b) );
end

function [NMSE] = nmse(y, x)
    % y is the measured signal
    % x is the simulated one
   NMSE = 20*log10(norm(y-x,2)^2 / norm(y,2)^2);
end