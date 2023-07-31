function [imgData, FRFData] = defImg_comparison_FRFs( xLengthImg, yLengthImg, imgN,...
                            xLabel, yLabel, areaColor, axFontSize,...
                            areaAlpha, legenda, lineWidth,...
                            cutHigh, cutLow, FRF, fAxis, fAxisComsol, comsol_FRF,...
                            alpha, beta, nRealizations, xyScale)
% This function wraps the parameters of an image comparing two FRFs
% INPUTS: 
%  Imgdata entries:
%   xLengthImg = (float) - x length of the figure
%   yLengthImg = (float) - y length of the figure
%   imgN       = (int) - figure number
%   xyLabels   = (cell, len = 2) - cell with xLabel and yLabel strings
%   areaColor  = (1DArray, len = 3) - color of the area between the FRFs 
%   areaAlpha  = (float) - transpacerncy of the area color, in [0,1]
%   axFontSize = (int) - font size of the axis
%   legenda    = (string) - legend of the figure
%   lineWidth  = (float) - width of the FRF line
%   xyscale    = (cell, len = 2) - {xScale, yScale} can be 'log' or 'linear'
%  FRFData entries:
%   cutHigh       = (float) - low bound of the frequency axis
%   cutLow        = (float) - high bound of the frequencye axis
%   FRF           = (1DArray) - measured FRF
%   fAxis         = (1DArray) - axis of the measured FRF
%   fAxisComsol   = (1DArray) - axis of simulated FRF
%   comsol_FRF    = (1DArray) - FRF simulated with Comsol Multiphysics
%   alpha         = (float) - Rayleigh damping parameter alpha value
%   beta          = (float) - Rayleigh damping parameter beta value
%                             n.b. you can annotate them in the figure
%   nRealizations = (int) - number of realizations with which the results
%                            of FRF2Params are computed
% ------------------------------------------------------------------------
% imgData : struct wrapping the parameters of the image
% FRFData : struct wrapping the data of experimental and simulated FRFs
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