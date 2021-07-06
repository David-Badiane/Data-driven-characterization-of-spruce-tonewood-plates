function fNN = plotFrequencyEstimation(f0, fNet, idxComsol, rho, xpar,NpeakAxis)
fNN = fNet([rho;xpar]);
diffAbs = abs(f0(1:length(idxComsol)) - fNN(idxComsol));

figure()
stem(1:length(diffAbs), diffAbs);
ylim([0,50]);
ax = gca;
ax.XMinorTick = 'on';
ax.YMinorTick = 'on';
ax.TickDir = 'out';
ax.FontSize = 15;
xlabel('N FRF peak')
yy = ylabel('$ | f_r - f_{nn}|$');
set(yy, 'interpreter', 'latex');
title(num2str(NpeakAxis));

figure();
plot(1:length(idxComsol), f0(1:length(idxComsol)),'-o', 1:length(idxComsol), fNN(idxComsol), '-x' );
ax = gca;
ax.XMinorTick = 'on';
ax.YMinorTick = 'on';
ax.TickDir = 'out';
ax.FontSize = 17;
title(num2str(NpeakAxis));

% figure();
% plot(1:length(idxComsol), f0(1:14),'-o', 1:20, fNN(1:20), '-x' );
% ax = gca;
% ax.XMinorTick = 'on';
% ax.YMinorTick = 'on';
% ax.TickDir = 'out';
% ax.FontSize = 17;
end