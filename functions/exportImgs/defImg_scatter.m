function imgData = defImg_scatter(xLengthImg, yLengthImg, imgN, xLabel, yLabel,...
                    tickSize, tickType, lineWidth, legenda, legendPos, fontSize,...
                    markerAlpha, xyLim, saveDir, saveName)
    imgData = struct('xLengthImg',[], 'yLengthImg', [], 'imgN', [], 'xyLabel',...
        [], 'tickSize',[],'tickType', [], 'lineWidth', [], 'legend', [], 'legendPos', [],...
        'fontSize', [], 'markerAlpha', [], 'xyLim', [], 'saveDir', [],'saveName', []);

    imgData.xLengthImg  = xLengthImg;
    imgData.yLengthImg  = yLengthImg;
    imgData.imgN        = imgN;
    imgData.xyLabel     = {xLabel yLabel};
    imgData.tickSize    = tickSize;
    imgData.tickType    = tickType;
    imgData.lineWidth   = lineWidth;
    imgData.legend      = legenda;
    imgData.legendPos   = legendPos;
    imgData.fontSize    = fontSize;
    imgData.markerAlpha = markerAlpha;
    imgData.xyLim       = xyLim;
    imgData.saveDir     = saveDir;
    imgData.saveName    = saveName;
end
