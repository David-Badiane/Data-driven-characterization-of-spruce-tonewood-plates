function img = export_matrix(matr,imgData, roundN, percentage)
    
    figure(imgData.imgN)
    clf reset; 
    ax = gca;     ax.FontSize = imgData.axFontSize*0.75;
    set(gcf, 'Position',  [0, 50, imgData.xLengthImg, imgData.yLengthImg]);
    
    % imgN, yIdxs, xIdxs, xyLabels, colorMap, textFontSize, axFontSize,
    % xyTickLabels 2 cell array with tickLabels {'2','4','8','16','32'}
    nXidxs = length(imgData.xIdxs);
    nYidxs = length(imgData.yIdxs);
    
    M = matr(imgData.yIdxs,imgData.xIdxs);
    if percentage
    textPercentage = [];
       for ii = 1:length(M(:,1))
            sumRow = sum(M(ii,:));
            percVals = M(ii,:)./sumRow;
            for jj = 1:length(percVals)
                textPercentage(ii,jj) = percVals(jj); 
            end
       end
       M = textPercentage;
    end
    img = imagesc(M);
    
    colormap(imgData.colorMap);
    
    cbar = colorbar;
    cbar.Label.String = imgData.cbarLabel;
    cbar.Label.Interpreter = 'latex';
    cbar.Label.FontSize = 22;
%     cbar.Label.Rotation = 0; % to rotate the text
%     cbar.Label.Position = [0.5 0.3 0];
    caxis([min(M, [], 'all')  max(M,[],'all') ])
    
    if ~imgData.displayCbar
        colorbar('off');
    end
    [x,y] = meshgrid(1:nXidxs,(1:nYidxs)+0.15);
    M(round(M,2) == 1) = 0.999;
    
    text(x(:),y(:)-0.1,num2str(round(M(:),roundN)),'HorizontalAlignment','center', 'FontSize',imgData.textFontSize*0.9, 'Interpreter', 'latex')
    
    if imgData.displayCbar
        cbar.TickLabelInterpreter = 'latex';      
        cbar.LineWidth = 1.2; 
    end
    
    ax = gca;
    set(ax,'XTick', 1:nXidxs, 'YTick', 1:nYidxs);
    ax.FontSize = imgData.axFontSize - 4;
    xx = xlabel(imgData.xyLabels{1}, 'FontSize', imgData.axFontSize);
    yy = ylabel(imgData.xyLabels{2}, 'FontSize', imgData.axFontSize);
    set(yy,'interpreter', 'latex'); set(xx,'interpreter', 'latex');
    
    ax.TickLabelInterpreter = 'latex';
    xticklabels(imgData.xyTickLabels{1})
    yticklabels(imgData.xyTickLabels{2})
    
    yline([0, 0.5:nYidxs+1, nYidxs+1.5], 'lineWidth', 1.5)
    xline([0, 0.5:nXidxs+1, nXidxs+1.5], 'lineWidth', 1.5)
end