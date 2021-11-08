function [imgData, FRFData]  = defImg_pointFRF( addInset, xLengthImg, yLengthImg, imgN,...
                            xLabel, yLabel, axFontSize, legenda, lineWidth,...
                            cutHigh, cutLow, FRF, fAxis, baseFolder)
    % imgData inputs
    % addInset --> specify if add an inset with the surfs (true/false)
    % xLengthImg = xLength of the image window
    % yLengthImg = yLength of the image window
    % imgN       = image number
    % xLabel, yLabel = x and y labels (string)
    % legenda        = legenda (cell)
    % lineWidth      = double for line width
    % cutHigh        = high cut of frequency axis
    % cutLow         = low cut of frequency axis
    
    % FRFData inputs
    % FRF            = FRF to plot
    % fAxis          = axis of the FRF
    
    % Read modes
    % baseFolder     = folder from which read the modeshapes for the inset
                        
                        
    imgData = struct('xLengthImg',[], 'yLengthImg', [], 'imgN', [], 'xyLabels',...
        [], 'axFontSize', [], 'legenda', [], 'lineWidth', []);
    
    imgData.xLengthImg    = xLengthImg;
    imgData.yLengthImg    = yLengthImg;
    imgData.imgN          = imgN;
    imgData.xyLabels      = {xLabel yLabel};
    imgData.axFontSize    = axFontSize;
    imgData.legenda       = legenda;
    imgData.lineWidth     = lineWidth;
    
    
    FRFData = struct('FRF', [], 'fAxis', [],...
                     'peakVals', [], 'peakLocs', [], 'refModesMatr', [], 'addInset', []);
    
    if addInset
       pX = 100; pY = 200;
       cd(baseFolder)
       refModes = readmatrix('referenceModes.csv');
       refModesMatr = {};
       for ii = 1:(length(refModes(1,:)))
           refModesMatr{ii} = reshape(refModes(:,ii), pY, pX).'; 
         
       end 
        figure(1)
        imagesc(refModesMatr{4});
        FRFData.refModesMatr = refModesMatr;
    end
    
    fCutIdxs = intersect(find(fAxis>=cutLow),...
                            find(fAxis<=cutHigh));
    fAxis = fAxis(fCutIdxs);
    FRF = FRF(fCutIdxs);
    [peakVals, peakLocs] = findpeaks(abs(FRF), 'minPeakProminence', 0.1e-6);

    FRFData.FRF         = abs(FRF);
    FRFData.fAxis       = fAxis; 
    FRFData.peakVals    = peakVals;
    FRFData.peakLocs    = peakLocs;
    FRFData.addInset    = addInset;
end