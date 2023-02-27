function [imgData, FRFData] = defImg_comparison_FRFs( xLengthImg, yLengthImg, imgN,...
                            xLabel, yLabel, areaColor, axFontSize,...
                            areaAlpha, legenda, lineWidth,...
                            cutHigh, cutLow, FRF, fAxis, fAxisComsol, comsol_FRF,...
                            alpha, beta, nRealizations, xyScale)
% defImg_pointFRFCompare - function to define the parameters of the image
% comparing experimental and simulated FRFs
% copyright: David Giuseppe Badiane
% -------------------------------------------------------------------------
% ImgData : struct with the data of the image
%           xLengthImg, yLengthImg    --> 1x1 doubles x and y length of figure
%           imgN                      --> image number
%           xyLabels                  --> xLabel and yLabel - strings
%           areaColor, areaAlpha      --> alpha and color of the area between the two FRFs
%            axFontSize               --> 1x1 double, axis font size
%           legenda        --> string - legend of the figure
%           lineWidth      --> 1x1 double width of the line
%           xScale, yScale --> strings - specify linear or log scale for x and y, respectively
%
% FRFData : struct with the data of the experimental and simulated FRFs
%           FRF, fAxis              --> 1xN double, experimental FRF and associated frequency axis
%           comsol_FRF, fAxisComsol --> 1xM double, simulated FRF and associated frequency axis  
%           alpha, beta             --> 1x1 doubles, damping variables
%           nRealizations           --> nRealizations of the estimation associated to the simulated FRF

    
    if nargin < 23
        xyScale = {'linear' 'linear'};
    end
    
    imgData = struct('xLengthImg',[], 'yLengthImg', [], 'imgN', [], 'xyLabels',...
        [], 'areaColor',[], 'axFontSize', [], 'areaAlpha',...
        [], 'legenda', [], 'lineWidth', [], 'xScale', [], 'yScale',[]);

    imgData.xLengthImg    = xLengthImg;
    imgData.yLengthImg    = yLengthImg;
    imgData.imgN          = imgN;
    imgData.xyLabels      = {xLabel yLabel};
    imgData.areaColor     = areaColor;
    imgData.axFontSize    = axFontSize;
    imgData.areaAlpha     = areaAlpha;
    imgData.legenda       = legenda;
    imgData.lineWidth = lineWidth;
    imgData.xScale = xyScale{1};
    imgData.yScale = xyScale{2};
    
    FRFData = struct('FRF', [], 'fAxis', [],...
              'fAxisComsol', [],'comsol_FRF', [], 'alpha', [], 'beta', [],...
              'nRealizations', []);
    
    FRFData.cutHigh     = cutHigh;
    FRFData.cutLow      = cutLow;
    FRFData.FRF          = FRF;
    FRFData.fAxis       = fAxis;
    FRFData.fAxisComsol = fAxisComsol;
    FRFData.comsol_FRF         = comsol_FRF;
    FRFData.alpha       = alpha;
    FRFData.beta        = beta;
    FRFData.nRealizations = nRealizations;    
end