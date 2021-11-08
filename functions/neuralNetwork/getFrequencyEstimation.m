function [fNN, idxComsol] = getFrequencyEstimation(f0, MLR_freq, xpar,NpeaksAxis, plotData)

fNN = predictEigenfrequencies(MLR_freq.linMdls , xpar.', length(MLR_freq.linMdls)).';
% fNN = MLR_freq(xpar);
idxComsol = [];
for ii = 1:length(f0)
    [minVal, minLoc] = min(abs(f0(ii) - fNN));
    idxComsol = [idxComsol; minLoc];
end

diffAbs = abs(f0(1:length(idxComsol)) - fNN(idxComsol));
if plotData    
%     figure()
%     stem(1:length(diffAbs), diffAbs);
%     ylim([0,50]);
%     ax = gca;
%     ax.XMinorTick = 'on';
%     ax.YMinorTick = 'on';
%     ax.TickDir = 'out';
%     ax.FontSize = 15;
%     xlabel('N FRF peak')
%     yy = ylabel('$ | f_r - f_{nn}|$');
%     set(yy, 'interpreter', 'latex');
%     title(['abs value difference' , num2str(NpeaksAxis)]);

    figure();
    plot(1:length(idxComsol), f0(1:length(idxComsol)),'-o', 1:length(idxComsol), fNN(idxComsol), '-x' );
    ax = gca;
    ax.XMinorTick = 'on';
    ax.YMinorTick = 'on';
    ax.TickDir = 'out';
    ax.LineWidth = 1.2; box on;
    grid on;
    ax.FontSize = 17;
    title(['freq difference', num2str(NpeaksAxis)]);
    pause(0.01)
    xlim([1 12]);
    ll = legend('real', 'estimated');
    set(ll, 'Box', 'off');
    ylabel('frequency [Hz]', 'Interpreter', 'latex', 'fontSize',18)
    xlabel('FRF peak number', 'Interpreter', 'latex', 'fontSize',18)
end
end