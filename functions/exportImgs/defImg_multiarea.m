function imgData = defImg_multiarea(xLengthImg, yLengthImg, imgN, xLabel,...
                    yLabel, faceAlpha, lineStyles, lineWidth, legenda, legendPos,...
                    fontSize, saveDir, saveName)
    
    imgData = struct('xLengthImg',[], 'yLengthImg', [], 'imgN', [], 'xyLabel',...
        [], 'faceAlpha', [], 'lineStyles', [], 'lineWidth', [], 'legend', [],...
        'legendPos', [], 'fontSize', [], 'saveDir', [], 'saveName', []);

    imgData.xLengthImg  = xLengthImg;
    imgData.yLengthImg  = yLengthImg;
    imgData.imgN        = imgN;
    imgData.xyLabel     = {xLabel yLabel};
    imgData.faceAlpha   = faceAlpha;
    imgData.lineStyles  = lineStyles;
    imgData.lineWidth   = lineWidth;
    imgData.legend      = legenda;
    imgData.legendPos   = legendPos;
    imgData.fontSize    = fontSize;
    imgData.saveDir     = saveDir;
    imgData.saveName    = saveName;
end