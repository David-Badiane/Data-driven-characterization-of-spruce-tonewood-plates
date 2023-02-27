function imgData = defImg_matrix(xIdxs, yIdxs, xLengthImg, yLengthImg, imgN, xLabel, yLabel, colorMap,...
                   textFontSize, axFontSize, xTickLabels, yTickLabels, cbarLabel, displayCbar)
    
    % IMG DATA OF MATRIX IMAGE
    % this function collects the data to make a chessboard representation 
    % with given entries of a matrix
    % copyright: David Giuseppe Badiane
    % ---------------------------------------------------------------------
    % inputs:
    % xLength      = (int) img window xLength
    % yLength      = (int) img window yLength
    % imgN         = (int) img number
    % xLabel       = (string) x label string
    % yLabel       = (string) y label string
    % colorMap     = (double RGB matrix) ex. winter
    % textFontSize = (int) size of text inside matrix 
    % axFontSize   = (int) size of axis text (x y labels, legends, etc)
    % xTyckLabels  = (cell) labels of the x ticks ex. {'1' 'b' 'area'}
    % yTickLabels  = (cell) labels of the y ticks ex. {'1' 'b' 'area'}         
    % ---------------------------------------------------------------------
    % outputs:
    % imgData      = (struct) - struct with members equal to the inputs of the fx
    % ---------------------------------------------------------------------
    imgData = struct('xIdxs', [],'yIdxs', [],'xLengthImg',[], 'yLengthImg', [],...
        'imgN', [], 'xyLabels',[], 'colorMap',[],'textFontSize', [],...
        'axFontSize', [], 'xyTickLabels', [], 'cbarLabel', [],...
        'displayCbar', []);
    
    imgData.xIdxs        = xIdxs;
    imgData.yIdxs        = yIdxs;
    imgData.xLengthImg   = xLengthImg;
    imgData.yLengthImg   = yLengthImg;
    imgData.imgN         = imgN;
    imgData.xyLabels     = {xLabel yLabel};
    imgData.colorMap     = colorMap;
    imgData.textFontSize = textFontSize;
    imgData.axFontSize   = axFontSize;
    imgData.xyTickLabels = {xTickLabels yTickLabels};
    imgData.cbarLabel    = cbarLabel;
    imgData.displayCbar  = displayCbar;
end