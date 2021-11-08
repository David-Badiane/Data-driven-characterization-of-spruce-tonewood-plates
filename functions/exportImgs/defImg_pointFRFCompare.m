function [imgData, FRFData] = defImg_pointFRFCompare( xLengthImg, yLengthImg, imgN,...
                            xLabel, yLabel, areaColor, annotFontSize, axFontSize,...
                            areaAlpha, errTextPos, paramsTextPos, legenda, lineWidth,...
                            cutHigh, cutLow, Hv, fAxis, fAxisComsol, vel,...
                            alpha, beta, nRealizations)

    % FRFData : cutLow, cutHigh, Hv, fAxisComsol, vel, f0
    % ImgData : xyLabel, lineWidth, annotFontSize, axFontSize, legend,
    %           areaColor, areaAlpha, errTextPos, paramsTextPos

    imgData = struct('xLengthImg',[], 'yLengthImg', [], 'imgN', [], 'xyLabels',...
        [], 'areaColor',[],'annotFontSize', [], 'axFontSize', [], 'areaAlpha',...
        [], 'errTextPos', [], 'paramsTextPos', [], 'legenda', [], 'lineWidth', []);

    imgData.xLengthImg    = xLengthImg;
    imgData.yLengthImg    = yLengthImg;
    imgData.imgN          = imgN;
    imgData.xyLabels      = {xLabel yLabel};
    imgData.areaColor     = areaColor;
    imgData.annotFontSize = annotFontSize;
    imgData.axFontSize    = axFontSize;
    imgData.areaAlpha     = areaAlpha;
    imgData.errTextPos    = errTextPos;
    imgData.paramsTextPos = paramsTextPos;
    imgData.legenda       = legenda;
    imgData.lineWidth = lineWidth;
    
    FRFData = struct('Hv', [], 'fAxis', [],...
              'fAxisComsol', [],'vel', [], 'alpha', [], 'beta', [],...
              'nRealizations', []);
    
    FRFData.cutHigh     = cutHigh;
    FRFData.cutLow      = cutLow;
    FRFData.Hv          = Hv;
    FRFData.fAxis       = fAxis;
    FRFData.fAxisComsol = fAxisComsol;
    FRFData.vel         = vel;
    FRFData.alpha       = alpha;
    FRFData.beta        = beta;
    FRFData.nRealizations = nRealizations;    
end