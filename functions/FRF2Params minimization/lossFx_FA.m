function [L2, map] = lossFx_FA(fEst, aEst, fReal, aReal, NpeaksAxis,...
            plotData, figN)
% LOSSFX_FA    
% Computes the loss function of the minimization. 
% Loss Fx is computed in 4 steps in the frequency/amplitude (FA) space: 
% 1) normalization btw R = (fReal, aReal) and E = (fEst, aEst)
% 2) euclidean FA space distance btw each R and all E
% 3) for each R select closest E
% 4) Loss Fx = relative frequency difference btw R and closest E
% This because not all eigenfrequencies of the plate correspond to
% peaks in the FRF, we need to discard the "antiresonances"
% ---------------------------------------------------------------------
% inputs:
%   fEst = 12x1 double - frequencies estimated by fNetwork
%   aEst = 12x1 double - amplitudes estimated by aNetwork
%   fReal = nPeaks x 1 double - frequencies of the FRF peaks
%   aReal = nPeaks x 1 double - amplitudes of the FRF peaks
%   NpeaksAxis = 12x1 double  - axis with the number of peaks considered
%                             in the minimization
%   plotData = boolean        - says whether to plot image or not
%   figN     = 1x1 double     - number of the plotted figure
% ---------------------------------------------------------------------
% outputs
%   L2  = 1x1  double - value of the loss function
%   map = 12x1 double - index of the prediction associated to a given FRF peak
%       ex. a map [1 2 4 3 5 6 8 7 10 11 12] indicates the index of the NN
%       estimation associated to [1 2 3 4 5 6 7 8 9 10 11 12]° FRF peak
% ---------------------------------------------------------------------

% setup fx inputs
if nargin<7, figN = 200; end
% setup variables
aEst = aEst(1:length(fEst));
aReal = aReal(NpeaksAxis); 
fReal = fReal(NpeaksAxis);
map = [];

% make them all column vectors
fEst = fEst(:); aEst = aEst(:); 
aReal = aReal(:); fReal = fReal(:);

% A) normalization
K = 4;
gamma = mean(abs(aReal(1:K))./abs(aEst(1:K))); % for db - comsol and real amps to the same order of magnitude      
aEst_norm  = db(aEst*gamma);     
aReal_norm = db(aReal);
eta = abs(mean(fReal./aReal_norm)); % get frequency and amplitudes on same scale
% rename variables
pointsEst = [fEst, aEst_norm];
pointsReal = [fReal, aReal_norm];         

% start the loss fx figure


% B) map computation
for kk = 1:length(NpeaksAxis) % for each real point
    % 1) compute the distance in frequency and amplitude for each point
    % multiply for eta to give same importance to frequency and amp in computation of the distance              
    ampsDiff = abs(eta*(pointsReal(kk,2) - pointsEst(:,2)));  
    fDiff =  abs((pointsReal(kk,1) - pointsEst(:,1)));        

    % 2) compute euclidean distances
    dist = sqrt(fDiff.^2 + ampsDiff.^2);

    % 3) compute minimum of the euclidean distances
    [minDist, minLoc] = min(dist);

    % 4) store map
    map(kk) = minLoc;           

    % 5) plot figure
    if plotData
        if kk == 1
            figure(figN-1);
            clf reset;
            hold on;
            plot(fReal(NpeaksAxis), aReal_norm(NpeaksAxis), '.', 'markerSize' ,20)
            plot(fEst, aEst_norm , 'd', 'markerSize',7,'lineWidth', 1.3)
            xlabel('frequency [Hz]', 'Interpreter' , 'latex');
            ylabel('amplitude [dB]', 'Interpreter' , 'latex');
        end
        lineFreqz =  [fReal(NpeaksAxis(kk)), fEst(map(kk))];
        lineAmps = [aReal_norm(NpeaksAxis(kk)), aEst_norm(map(kk))];
        plot(lineFreqz, lineAmps, 'k', 'lineWidth', 1.4);
        %ylim([0, 500]);
        %xlim([0,600]);
        if kk == length(NpeaksAxis)
            ll = legend( 'measured', 'estimated'); set(ll, 'Box', 'off'); 
            ax = gca;
            ax.XMinorTick = 'on';
            ax.YMinorTick = 'on'; ax.FontSize = 18;
            ax.TickLabelInterpreter = 'latex';
            ax.LineWidth = 1.2; box on;
            set(ll, 'FontSize', 15);
            pause(0.0001);
        end
    end
end

% C) compute loss function
% weights to prioritize matching of lower frequency modes
importantPeaks =ismember(map,[1]);
gains = ones(size(map)) + 1*importantPeaks;

% relative errors
L2_freq_rel = abs( (pointsReal(:,1) - pointsEst(map,1))) ./ pointsReal(:,1);
L2_amp_rel  = abs( (pointsReal(:,2) -  pointsEst(map,2))./ pointsReal(:,2) ); 

% Loss function
% minimize only on frequency if alpha and beta are fixed
L2 = sum(gains(:).*L2_freq_rel(:)); 
% minimizing also amplitude allows to estimate alpha and beta
% L2 = gains.*sum(L2_freq_rel +L2_amp_rel); 

% penalty function to avoid that the same estimation of NN is
% associated to two or more FRF peaks
if length(unique(map)) ~= length(map)
    L2 = (abs(length(unique(map))-length(map)))^2 + L2; 
end  
end