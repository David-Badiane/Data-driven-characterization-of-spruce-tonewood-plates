function [L2, L2_freq, L2_amp] = errorFA(mechParams, fNet, ampNet, f0, fAmps, rho, NpeaksAxis, referenceVals, constraintWeight, plotData)
%ERRORFA Summary of this function goes here
%   Detailed explanation goes here
    
    psi = 600;
    mechParameters = [rho; mechParams];
    fNN = fNet(mechParameters);
    ampNN = db2mag(ampNet(mechParameters));
    
    ratio = (fAmps(1:5))./(ampNN(1:5));
    gamma = mean(ratio);
    % Frequency/amplitude scaling
    ampNN = gamma * ampNN;
    ampsReal =  fAmps;
    eta = mean(f0(NpeaksAxis)./(ampsReal(NpeaksAxis)) );
    ampNN = eta*ampNN;
    ampsReal = eta*ampsReal;
 
    pointsNN = [fNN, ampNN];
    pointsReal = [f0(NpeaksAxis), ampsReal(NpeaksAxis)];

    
%     if plotData
% %         figure(1)
% %         diffFreq = diff(fNN);
% %         plot(1:length(diffFreq), diffFreq, '-o');
%         figure(200)
%         plot( fNN, ampNN , '.', 'markerSize', 10)
%         hold on;
%         xlabel('frequency');
%         ylabel('amplitude');
%         plot(f0(NpeaksAxis), ampsReal(NpeaksAxis), '.', 'markerSize' ,10)
%         xlim([f0(1)-10, abs(fNN(end))+20]);
%     end
    

    
    L2 = 0;
    L2_amp = 0;
    L2_freq = 0;
    
    map = [];
    
    for kk = NpeaksAxis
        ampsDiff = pointsReal(kk,2) - pointsNN(:,2);                
        fDiff = (pointsReal(kk,1) - pointsNN(:,1))./(pointsReal(kk,1));
        
        dist = sqrt(( psi* fDiff).^2 + (ampsDiff).^2);
        [minDist, minLoc] = min(dist);
%         disp(['fDiff = ', num2str(psi*(fDiff(minLoc)))]);
%         disp(['ampDiff = ', num2str((ampsDiff(minLoc)))]);
        if isempty(map)
            map(kk) = minLoc;
        else
            if ~isempty( find(map == minLoc));
                L2 = L2 + 1e6; 
            end
        end

        lineFreqz =  [f0(kk), fNN(minLoc)];
        lineAmps = [ampsReal(kk), ampNN(minLoc)];
%         if plotData
%         plot(lineFreqz, lineAmps);
%         pause(0.01);
%         end
        L2_amp = L2_amp + abs(ampsDiff(minLoc));
        L2_freq = L2_freq + abs( psi* fDiff(minLoc));
        L2 = L2 + minDist;
        map(kk) = minLoc;
        
    end
    hold off;
%     disp(['L2_pre = ' , num2str(L2)]);

%     indexes = [5,6];    
%      for ii = 1:length(indexes)
%         bound =  abs(mechParams(ii) - referenceVals(ii));
%         L2 = L2 + constraintWeight*(mechParams(indexes(ii)) - referenceVals(indexes(ii)))^2; 
%      end
%      
%      indexes = [7,8,9];    
%      for ii = 1:length(indexes)
%         bound =  abs(mechParams(ii) - referenceVals(ii));
%         L2 = L2 + 1000* constraintWeight*(mechParams(indexes(ii)) - referenceVals(indexes(ii)))^2; 
%      end
    
%     disp(['freq = ', num2str(L2_freq)]); 
%     disp(['amp = ', num2str(L2_amp)]);
%     
%     disp(['L2_aft = ' , num2str(L2)]);
 
end

