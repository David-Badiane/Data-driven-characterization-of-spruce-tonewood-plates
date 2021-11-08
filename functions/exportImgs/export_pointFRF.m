function [img] = export_pointFRF(FRFData, imgData,saveImg,saveDir, saveName)

% setup
fAxis = FRFData.fAxis;
Hv = FRFData.FRF;
f0 = fAxis(FRFData.peakLocs);

% start plotting 
img = figure(imgData.imgN);
clf reset; box on; hold on;
set(gcf, 'Position',  [0, 50, imgData.xLengthImg, imgData.yLengthImg]);
plot(fAxis,Hv, 'LineWidth',imgData.lineWidth);

% settings of the axis
xlim([min(fAxis), max(fAxis)]);
ylim([0, 1.1*max(Hv)])
ax = gca; 
ax.FontSize = imgData.axFontSize - 4; 
xx = xlabel(imgData.xyLabels{1}, 'fontSize',imgData.axFontSize ); set(xx, 'Interpreter', 'Latex');
yy = ylabel(imgData.xyLabels{2}, 'fontSize',imgData.axFontSize); set(yy, 'Interpreter', 'Latex');
ax.XMinorTick = 'on'; ax.YMinorTick = 'on';
ax.TickLabelInterpreter = 'latex';
% ll = legend(imgData.legenda{:});
ax.LineWidth = 1.3;

% If add an inset
if FRFData.addInset
    nModesSee = 6;
    % comment if you want just to do the inset 
    [Hv,f0, fLocs, bandsIdxs] = EMASimple(Hv, fAxis, 2e-8 , 1, false);
    bandsIdxs(1,:) = [bandsIdxs(1,1)-5, bandsIdxs(1,2)+10];
    for ii = 1:nModesSee
        area(fAxis(bandsIdxs(ii,:)), [1 1]*1.4*max(Hv), 'FaceAlpha', 0.28, 'edgeColor', 'none') 
    end
    
    % settings for the inset surfaces
    count = 0;         
    ylim([0, 2*max(Hv)]);
    % colormap
    c = [jet;flip(jet)];
    refModesIdxs = [1 2 4 3 5 6];
    refModes = FRFData.refModesMatr;
    X = refModes{1}; Y = refModes{2}; 
    thickness = -1e-7;
    dimY = .15;
    dimX = dimY;
    
    for ii = 1:nModesSee
        % xPos for each inset
        xPos = 0.01+ax.Position(1)+1.01*dimX*count ;
        % modeShape fetch
        Z = refModes{refModesIdxs(ii)+4};
        % create inset
        assi = axes('Parent', gcf,'Position',[xPos 0.7 dimX dimY]);
        hold on ;
        % patch -- solid object
        fv = surf2solid(Y,X,Z, 'thickness', thickness);
        pp = patch(fv,'FaceColor', 0.4*[1 1 1], 'EdgeColor', 'none',...
             'EdgeLighting', 'gouraud', 'AmbientStrength', 0.15, 'EdgeAlpha', 0.1); hold on;
        % surface over the patch
        surf(Y,X,Z, 'edgeColor' , 'none', 'FaceAlpha', 1);
        % set zlim, view, axis features
        zlim([min(Z, [], 'all')+thickness , 1.5*max(Z, [], 'all')])
        view(-200, 110)
        set(gca,'visible','off')
        colormap(c);
        set(gca,'xtick',[]); set(gca,'ytick',[])
        count = count + 1;
    end
end
    if saveImg
        cd(saveDir);
        saveas(img, [saveName, '.png']);
    end
end
