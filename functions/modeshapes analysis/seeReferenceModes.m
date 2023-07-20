function [] = seeReferenceModes(modesAnalysisPath,pX, pY,subplotnRows, subplotnCols ,showFourier)
%% 4) SEE REF SHAPES 
% this function generates a figure of the reference modeshapes
% copyright: David Giuseppe Badiane
% -------------------------------------------------------------------------
% inputs:
%   modesAnalysisPath = string - path of the modesAnalysis directory
%   pX                = int - number of points on the x axis
%   pY                = int - number of points on the y axis
%   subplotnRows      = int - subplot of the figure, total number of rows
%   subplotnCols      = int - subplot of the figure, total number of columns
%   showFourier       = boolean - to show reference set in space fourier domain 
%                     or space domain
% -------------------------------------------------------------------------
% outputs:
% ~
% -------------------------------------------------------------------------
% function to see the reference set

if nargin <6, showFourier = 0; end
     cd(modesAnalysisPath)
         ref = readmatrix('reference.csv');
         refModesNames = table2cell(readtable('refModesNames.csv'));
     
    
% we can see both space Fourier domain or space domain
    % preallocate
    refShapes_Fourier = {};
    refShapes_Disp = {};
    % display and save ref shapes
    c = [flip(jet);];% colormap
    colormap(c);
    
    figure(1256); clf reset; 
    if showFourier, figure(1257); clf reset; end
    for ii = 1:(length(ref(1,:)))
        modeSurf = reshape(ref(:,ii), [pY, pX]);
        % displacement
        refShapes_Disp{ii} = modeSurf;
        % plot disp
        figure(1256)
        subplot(subplotnRows,subplotnCols,ii)
        imagesc(refShapes_Disp{ii}); shading interp;
        title(['(', refModesNames{elmnts(ii)}(2) ',' refModesNames{elmnts(ii)}(3),')']);    
        colormap(c);
        axis off;
        box on;
        pause(0.001);
        
        if showFourier
            % space fourier
            refShapes_Fourier{ii} = fftshift(abs(fft2(modeSurf, floor(pX/2),floor(pY/2))));
            
            % plot
            figure(1257)
            subplot(subplotnRows,subplotnCols,ii)
             imagesc(refShapes_Fourier{ii}); title(refModesNames{ii});    colormap(c);
            axis off;
            box on;
            pause(0.001);
        end
    end

end