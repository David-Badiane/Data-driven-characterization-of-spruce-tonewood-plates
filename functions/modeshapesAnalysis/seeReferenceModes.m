function [] = seeReferenceModes(modesAnalysisPath,pX, pY,subplotnRows, subplotnCols)
%% 4) SEE REF SHAPES 
     cd(modesAnalysisPath)
     ref = readmatrix('reference.csv');
     ref = ref(:,1:end);
     refModesNames = table2cell(readtable('refModesNames.csv'));

    % preallocate
    refShapes_Fourier = {};
    refShapes_Disp = {};
    % display and save ref shapes
    c = [jet; flip(jet)];
    colormap(c);
    figure(1256); clf reset; 
    figure(1257); clf reset;
    for ii = 1:(length(ref(1,:)))
        modeSurf = reshape(ref(:,ii), [pY, pX]);
        refShapes_Fourier{ii} = fftshift(abs(fft2(modeSurf, floor(pX/2),floor(pY/2))));
        refShapes_Disp{ii} = modeSurf;
        % plot disp
        figure(1256)
        subplot(subplotnRows,subplotnCols,ii)
        imagesc(refShapes_Disp{ii}); title(refModesNames{ii});    colormap(c);

        pause(0.001);
        
        figure(1257)
        subplot(subplotnRows,subplotnCols,ii)
        imagesc(refShapes_Fourier{ii}); title(refModesNames{ii});    colormap(c);

        pause(0.001);
    end

end