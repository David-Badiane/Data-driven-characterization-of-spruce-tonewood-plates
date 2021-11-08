function [img] = export_scatter(actual, predicted, imgData, imgMode, saveImg)
  
    tickTypes = {'o',  'd' , 's','v', '<', 'p', 'v','>', '_', '|', '+', '*', '.', 'x',};
    colors =  colororder;

    if strcmp(imgMode, 'columns')
        nCols = length(actual(1,:));
        nRowsAx = 1:length(actual(:,1));
        %nRowsAx = sort(randsample(nRowsAx, 200));
        img = figure(imgData.imgN);
        clf reset; 
        ax = gca;         ax.FontSize =0.8*imgData.fontSize;
        set(gcf, 'Position',  [0, 50, imgData.xLengthImg, imgData.yLengthImg]);    
        minL= []; maxL = [];
        hold on

        for ii = 1:nCols       
            minL(ii) = min([min(predicted(nRowsAx,ii), [], 'all'), min(actual(nRowsAx,ii), [], 'all')]);
            maxL(ii) = max([max(predicted(nRowsAx,ii), [], 'all'), max(actual(nRowsAx,ii), [], 'all')]);
        end

        minL = min(minL); maxL = max(maxL);
        bis = [minL - imgData.xyLim, maxL +  imgData.xyLim];
        plot( bis,bis,'--k', 'lineWidth', 1.5, 'HandleVisibility', 'off');
        xlim(bis);
        ylim(bis);
        cAxis = linspace(0.5,1,nCols);
        for ii = 1:nCols
        s = scatter(actual(nRowsAx,ii),predicted(nRowsAx,ii),imgData.tickSize, tickTypes{ii},...
                'MarkerFaceColor', cAxis(ii)*[1,1,1], 'MarkerEdgeColor', colors(mod(ii-1,7) +1 ,:), 'MarkerFaceAlpha',imgData.markerAlpha,...
                'MarkerEdgeAlpha', 0.675, 'lineWidth', imgData.lineWidth);  
        end

        [ll] = legend(imgData.legend, 'position', imgData.legendPos);set(ll,'Interpreter', 'Latex', 'fontSize', imgData.fontSize*0.95);
        set(ll,'Box','off')
        ax = gca;
        xx = xlabel(imgData.xyLabel{1}, 'FontSize', imgData.fontSize -4);
        yy = ylabel(imgData.xyLabel{2}, 'FontSize', imgData.fontSize -4);
        set(xx,'Interpreter', 'Latex'); set(yy,'Interpreter', 'Latex');
    end


    if strcmp(imgMode, 'all')
        img =  figure(imgData.imgN);
        clf reset; 
        set(gcf, 'Position',  [0, 50, imgData.xLengthImg, imgData.yLengthImg]);

        minL = min([min(predicted, [], 'all'), min(actual, [], 'all')]);
        maxL = max([max(predicted, [], 'all'), max(predicted, [], 'all')]);
        bis = [0.95*minL, 1.05*maxL];
        plot( bis,bis, 'lineWidth', 1.5);

        hold on;

        s = scatter(actual(:),predicted(:),imgData.tickSize, tickTypes{1},...
                'MarkerFaceColor', [1,1,1], 'MarkerFaceAlpha',.7, 'lineWidth', 0.5);
        xx = xlabel(imgData.xyLabel{1});
        yy = ylabel(imgData.xyLabel{2});
        set(xx,'Interpreter', 'Latex'); set(yy,'Interpreter', 'Latex');
        hold on;

        xlim(bis);
        ylim(bis);
        ax = gca;
        ax.XMinorTick = 'on';
        ax.YMinorTick = 'on';
        ax.TickDir = 'out';
        ax.FontSize = 20;
        
    end

    box on;
    ax.TickLength = [.013,.013];
    ax.LineWidth = 1.3;
    ax.XMinorTick = 'on'; ax.YMinorTick = 'on';
    ax.TickDir = 'in';
    hold off;
    ax.TickLabelInterpreter = 'latex';

    if saveImg 
           cd(imgData.saveDir)
           saveas(img,[imgData.saveName '.png']); 
    end
end