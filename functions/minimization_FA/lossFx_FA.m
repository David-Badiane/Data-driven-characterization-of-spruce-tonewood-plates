function [L2, L2_freq, L2_amp, map] = lossFx_FA(fEst, aEst, fReal, aReal, NpeaksAxis, plotData, normalization, ampScale)
    
    if nargin<8, ampScale = 'mag'; end
    if strcmp(ampScale,'db'), aEst = db(aEst); aReal = db(aReal); end
    
    % setup
    fEst = fEst(:); aEst = aEst(:);
    map = [];
    distances = zeros(length(NpeaksAxis));
    
    % Frequency amplitude scaling, normalization
    if strcmp(normalization, 'normalize')
        if strcmp(ampScale,'db')
         aEst_norm  = (aEst - min(aEst))   ./ (max(aEst)  - min(aEst));
         aReal_norm = (aReal - min(aReal)) ./ (max(aReal) - min(aReal));
         eta = mean(fReal(1:5)./(aReal([1 2 4 3 5 ])) ); % 
         maxF = fReal(NpeaksAxis(round(length(NpeaksAxis)/2)));
         fReal = fReal/maxF;
         fEst = fEst/maxF;
        else
         aEst_norm = aEst/max(aEst);
         aReal_norm =  aReal /max(aReal);
         eta = mean(fReal(1:5)./(aReal([1 2 3 4 5 ])) ); % 
        end
    end
    
    % Frequency amplitude scaling, ratios
    if strcmp(normalization, 'ratios')
     if strcmp(ampScale, 'db')
      gamma = mean(abs(aReal(1:5)-aEst([1 2 4 3 5]))); % for db - comsol and real amps to the same order of magnitude
      aEst_norm  = aEst + gamma;                           % for db
     else
      gamma = mean((aReal(1:5))./(aEst([1 2 4 3 5])));    % for mag - comsol and real amps to the same order of magnitude
      aEst_norm  = aEst*gamma;                            % for mag
     end
     aReal_norm = aReal;
     eta = abs(mean(fReal(NpeaksAxis)./(aReal(NpeaksAxis)) )); % get frequency and amplitudes on same scale
     aEst_norm = aEst_norm* eta;
     aReal_norm = aReal_norm*eta;
    end
    
    % collect the points in two vectors
    pointsEst = [fEst, aEst_norm];
    pointsReal = [fReal(NpeaksAxis), aReal_norm(NpeaksAxis)];         
    
    % start the figure
    if plotData
        figure(200);
        clf reset;
        plot( fEst, aEst_norm , '.', 'markerSize', 15)
        hold on;
        xlabel('frequency');
        ylabel('amplitude');
        plot(fReal, aReal_norm, '.', 'markerSize' ,15)
    end
    
    % map computation
    for kk = 1:length(NpeaksAxis) % for each real point
        % 1) compute the distance in frequency and amplitude for each point
        ampsDiff = (pointsReal(kk,2) - pointsEst(:,2));                
        fDiff =  (pointsReal(kk,1) - pointsEst(:,1));        
        
        % 2) compute euclidean distances
            if strcmp(ampScale, 'mag')
                dist = sqrt((fDiff/eta).^2 + (ampsDiff).^2); 
            else
                dist = sqrt((fDiff).^2 + (ampsDiff).^2);
            end
        
        % 3) compute minimum of the euclidean distances
        [minDist, minLoc] = min(dist);
        
        % 4) store results
        distances(kk) = minDist;
        if ismember(minLoc, map)
           dist(minLoc) = 10000;
           [minDist, minLoc] = min(dist);
        end
        map(kk) = minLoc;           
        
        % 5) plot figure
        if plotData
            lineFreqz =  [fReal(NpeaksAxis(kk)), fEst(minLoc)];
            lineAmps = [aReal_norm(NpeaksAxis(kk)), aEst_norm(minLoc)];
            plot(lineFreqz, lineAmps, 'k', 'lineWidth', 1.4);
           % ylim([0, 500]);
            %xlim([0,600]);
            pause(0.01);
        end
    end
    if plotData
        ll = legend('estimated', 'real'); set(ll, 'Box', 'off'); 
        ax = gca;
        ax.XMinorTick = 'on';
        ax.YMinorTick = 'on'; ax.FontSize = 15;
        ax.LineWidth = 1.2; box on;
        grid on
        set(ll, 'FontSize', 15);
    end
    
    % 1) RISCRIVERE CON NMSE  o errori RELATIVI sullo stesso range!!
    % 2) mettere relativi con ampiezze in db e vedere cosa succede!!
    
    L2_freq_rel = abs( (pointsReal(:,1) - pointsEst(map,1)) ./ pointsReal(:,1) );
    L2_amp_rel  = abs( (pointsReal(:,2) -  pointsEst(map,2))./ pointsReal(:,2) ); 
    L2_freq = abs(pointsReal(:,1) - pointsEst(map,1));
    L2_amp = abs( ((pointsReal(:,2) - pointsEst(map,2))));
%   L2 = sum(L2_freq .^2 + L2_amp.^2);%/sum(gains);
    L2 = sum(L2_freq_rel.^2 + L2_amp_rel.^2);
%     disp(['L2 freq absolute: ', num2str(sum(L2_freq)), '    L2 freq realtive: '  num2str(sum(L2_freq_rel))]);
    disp(['L2  absolute: ', num2str(sum(L2_freq .^2 + L2_amp.^2)),...
      '    L2 freq relative: '  num2str(sum(L2_freq_rel.^2 + L2_amp_rel.^2))]);
    if length(unique(map)) ~= length(map)
        L2 = L2 + 1e1; 
    end  
    %disp(corr(pointsReal(:,2), pointsEst(map,2)));
end