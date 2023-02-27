function img = export_matrixix(matrix,imgData, roundN, plotPercentage)
    % export_matrixix
    % this function generates a chessboard representation of a matrix
    % the entries of the matrix are specified as text inside each square
    % it is possible to represent a submatrix
    % copyright: David Giuseppe Badiane
    % ---------------------------------------------------------------------
    % inputs:
    % matrix         = (nTuples x nCols) double - matrix to represent
    % imgData        = struct - struct with the various figure settings
    % roundN         = int - number of significant digits represented as
    %                        text in the figure
    % plotPercentage = boolean - selects whether to represent the actual
    %                            matrix values (plotPercentage == 0) 
    %                            or to plot them in percentage of the sum
    %                            of the matrix row (plotPercentage == 1);
    % ---------------------------------------------------------------------
    % outputs:
    % img            = matlab figure
    % ---------------------------------------------------------------------
    
    % set figure number, axis font size and position 
    figure(imgData.imgN)
    clf reset; 
    ax = gca;     ax.FontSize = imgData.axFontSize*0.75;
    set(gcf, 'Position',  [0, 50, imgData.xLengthImg, imgData.yLengthImg]);
    
    % indexes of the matrix to be plotted
    nXidxs = length(imgData.xIdxs);
    nYidxs = length(imgData.yIdxs);
    
    % submatrix
    M = matrix(imgData.yIdxs,imgData.xIdxs);
    % if plotPercentage - update M with the relative values over the row sum
    if plotPercentage
        for ii = 1:length(M(:,1))
            sumRow = sum(M(ii,:));
            M(ii,:) = M(ii,:)./sumRow;
        end
    end
    
    % plot the chessboard
    img = imagesc(abs(M));
    % add the color map
    colormap(imgData.colorMap);
    % add the colorbar and assign settings
    cbar = colorbar;
    cbar.Label.String = imgData.cbarLabel;
    cbar.Label.Interpreter = 'latex';
    cbar.Label.FontSize = 22;
    cbar.TickLabelInterpreter = 'latex';      
    cbar.LineWidth = 1.2; 
    if ~imgData.displayCbar
        colorbar('off');
    end
    
    % set in order to avoid ones in the matrix
    M(round(M,2) == 1) = 0.999;
    
    % add the text with the matrix entries
    [x,y] = meshgrid(1:nXidxs,(1:nYidxs)+0.15);
    text(x(:),y(:)-0.1,num2str(round(M(:),roundN)),'HorizontalAlignment','center', 'FontSize',imgData.textFontSize*0.9, 'Interpreter', 'latex')
    
    % edit the figure axis
    ax = gca;
    set(ax,'XTick', 1:nXidxs, 'YTick', 1:nYidxs);
    ax.FontSize = imgData.axFontSize - 4;
    xx = xlabel(imgData.xyLabels{1}, 'FontSize', imgData.axFontSize, 'Interpreter', 'latex');
    yy = ylabel(imgData.xyLabels{2}, 'FontSize', imgData.axFontSize, 'Interpreter', 'latex');
    % add x and y ticks labels
    ax.TickLabelInterpreter = 'latex';
    xticklabels(imgData.xyTickLabels{1})
    yticklabels(imgData.xyTickLabels{2})
    
    % use xline and yline to highlight the chessboards
    yline([0, 0.5:nYidxs+1, nYidxs+1.5], 'lineWidth', 1.5)
    xline([0, 0.5:nXidxs+1, nXidxs+1.5], 'lineWidth', 1.5)
end