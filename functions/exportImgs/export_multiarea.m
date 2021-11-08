function [img] = export_multiarea(axis, matrixplot, imgData, saveImg)

    img = figure(imgData.imgN);
    clf reset; 
    set(gcf, 'Position',  [0, 50, imgData.xLengthImg, imgData.yLengthImg]);    
    hold on; box on;
    
    for ii = 1:length(matrixplot(:,1))
        lnStylIdx = mod(ii-1, length(imgData.lineStyles)) + 1;
        area(axis, matrixplot(ii,:), 'lineWidth', imgData.lineWidth,...
            'LineStyle', imgData.lineStyles{lnStylIdx},...
            'faceAlpha', imgData.faceAlpha);
    end
    
    [ll] = legend(imgData.legend, 'position', imgData.legendPos);
    set(ll,'Interpreter', 'Latex');
    set(ll,'Box','off')
    
    ax = gca;
    ax.FontSize = imgData.fontSize-4;
    xx = xlabel(imgData.xyLabel{1}, 'FontSize', imgData.fontSize);
    yy = ylabel(imgData.xyLabel{2}, 'FontSize', imgData.fontSize);
    set(xx,'Interpreter', 'Latex'); set(yy,'Interpreter', 'Latex');
    
    ax.TickLength = [.013,.013];
    ax.LineWidth = 1.3;
    ax.XMinorTick = 'on'; ax.YMinorTick = 'on';
    ax.TickDir = 'in';
    hold off;
    
    if saveImg 
       cd(imgData.saveDir)
       saveas(img,[imgData.saveName '.png']); 
    end
end