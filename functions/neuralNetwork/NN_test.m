function [R2_NN] = NN_test(net, testIn, testOut, strTitle, plotData, figNumber)

y = net(testIn.').';
yCut = min([length(y(1,:)), length(testOut(1,:))]);
y = y(:,1:yCut);
testOut = testOut(:,1:yCut);
names = {};
for ii = 1:length(y(1,:))
   names{ii} = [strTitle(1), int2str(ii)];
end

[R2] = computeR2(testOut, y);
R2_NN = array2table(round(R2,4), 'VariableNames', names);

if plotData
    figure(figNumber)
    for ii = 1:6
        subplot(2,3,ii)
        scatter(testOut(:,ii), y(:,ii), 15, 'filled');
        hold on
        plot([min(testOut(:,ii)) max(testOut(:,ii))], [min(testOut(:,ii)) max(testOut(:,ii))]);
        xlabel('test Out');
        ylabel('NN prediction');
        ax = gca;
        ax.FontSize = 13;
        xlim([min(testOut(:,ii)) max(testOut(:,ii))]); ylim([min(testOut(:,ii)) max(testOut(:,ii))]);
    end
    
    figure(figNumber+1)
    plot(1:15, R2(1:15),'-o', 'lineWidth',1.2);
    ax = gca; ax.XMinorTick = 'on'; ax.YMinorTick = 'on';
    ax.FontSize = 12; xlim([1 10]);
    yy = ylabel('$R^2$'); set(yy, 'interpreter', 'latex');
    xlabel('N-th eigenfrequency');
    hold on
    title(strTitle)
end

end