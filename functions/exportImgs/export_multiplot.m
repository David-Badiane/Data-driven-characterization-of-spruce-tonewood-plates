function [img] = export_multiplot(axis, matrixplot, imgData, saveImg)
    
%     imgData = struct('xLengthImg',[], 'yLengthImg', [], 'imgN', [], 'xyLabel',...
%         [], 'tickSize',[],'lineType', [], 'lineWidth', [], 'legend', [], 'legendPos', [],...
%         'fontSize', [], 'xyLim', []);
    
    
    
    img = figure(imgData.imgN);
    clf reset; 
    set(gcf, 'Position',  [0,0, imgData.xLengthImg, imgData.yLengthImg]);    
    hold on; box on;
    
    for ii = 1:length(imgData.lineType)
        plot(axis, matrixplot(ii,:), imgData.lineType{ii},...
            'lineWidth', imgData.lineWidth, 'markerSize', imgData.markerSize,...
            'markerFaceColor', [1,1,1]);
    end
    xlim([min(axis), max(axis)])
    [ll] = legend(imgData.legend, 'position', imgData.legendPos, 'fontSize', imgData.fontSize*0.95);
    set(ll,'Interpreter', 'Latex');
    set(ll,'Box','off')
    
    ax = gca;
    ax.FontSize = imgData.fontSize-8;
    xx = xlabel(imgData.xyLabel{1}, 'FontSize', imgData.fontSize);
    yy = ylabel(imgData.xyLabel{2}, 'FontSize', imgData.fontSize);
    set(xx,'Interpreter', 'Latex'); set(yy,'Interpreter', 'Latex', 'fontSize', imgData.fontSize*0.9);
    ax.XTick = axis;
    ax.TickLength = [.013,.013];
    ax.LineWidth = 1.3;
    ax.XMinorTick = 'off'; ax.YMinorTick = 'on';
    ax.TickDir = 'in';
    hold off;
        ax.TickLabelInterpreter = 'latex';
    if saveImg 
       cd(imgData.saveDir)
       saveas(img,[imgData.saveName '.png']); 
    end
end