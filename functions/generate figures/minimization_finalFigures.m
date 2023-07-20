function [freqs] = minimizations_finalFigures(f0, fNet, xpar,NpeaksAxis, plotData)
% getFrequencyEstimation generates a figure to see how the estimation is
% accurate under the frequency profile
% -------------------------------------------------------------------------
% inputs: 
%   f0         = nPeaksx1 double - frequency of the peaks of one FRF
%   fNet       = neural network object - neural network for prediction of eigenfrequencies
%   xpar       = double - predicted parameters
%   NpeaksAxis = double - axis with FRF peaks considered
%   plotData   = boolean - whether or not to plot a image
% -------------------------------------------------------------------------
% outputs:
%   freqs      = predicted eigenfrequencies
% -------------------------------------------------------------------------
    freqs = fNet(xpar(:));
    % map computation (in frequency space)

    % plot figure
    if plotData    
        figure(33); clf reset;
        plot(1:length(mapFreq), f0(1:length(mapFreq)),'-o', 1:length(mapFreq), freqs(mapFreq), '-x' );
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