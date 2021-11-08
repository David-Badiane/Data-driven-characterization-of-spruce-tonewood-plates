function imgData = defImg_multiplot(xLengthImg, yLengthImg, imgN, xLabel,...
                    yLabel, lineType, lineWidth, markerSize, legenda, legendPos,...
                    fontSize, saveDir, saveName)
    
               imgData = struct('xLengthImg',[], 'yLengthImg', [], 'imgN', [], 'xyLabel',...
        [],'lineType', [], 'lineWidth', [],'markerSize', [], 'legend', [], 'legendPos', [],...
        'fontSize', [], 'saveDir', [], 'saveName', []);

    imgData.xLengthImg  = xLengthImg;
    imgData.yLengthImg  = yLengthImg;
    imgData.imgN        = imgN;
    imgData.xyLabel     = {xLabel yLabel};
    imgData.lineType    = lineType;
    imgData.lineWidth   = lineWidth;
    imgData.legend      = legenda;
    imgData.legendPos   = legendPos;
    imgData.markerSize  = markerSize;
    imgData.fontSize    = fontSize;
    imgData.saveDir     = saveDir;
    imgData.saveName    = saveName;
end