function [L2, L2_freq, L2_amp, map] = errorFA(mechParams, fNet, ampNet, f0, fAmps, rho, NpeaksAxis, plotData)
%ERRORFA Summary of this function goes here
%   Detailed explanation goes here
    psi = 600;   % frequency distance scaling
    mechParameters = [rho; mechParams];
    fNN = fNet(mechParameters);
    ampNN = db2mag(ampNet(mechParameters));
    
    ratio = (fAmps(1:5))./(ampNN(1:5));    % first five frequencies 1:1 link relation
    gamma = mean(ratio);  % put Comsol amplitudes in same scale of Real amplitudes
    
    % Frequency/amplitude scaling
    ampNN = gamma * ampNN;
    ampsReal =  fAmps;
    eta = mean(f0(1:5)./(ampsReal(1:5)) ); % 
    ampNN = eta*ampNN;
    ampsReal = eta*ampsReal;
 
    pointsNN = [fNN, ampNN];
    pointsReal = [f0(NpeaksAxis), ampsReal(NpeaksAxis)];
         
    map = [];
    distances = zeros(length(NpeaksAxis));
    
%     figure(200)
%     plot( fNN, ampNN , '.', 'markerSize', 10)
%     hold on;
%     xlabel('frequency');
%     ylabel('amplitude');
%     plot(f0(NpeaksAxis), ampsReal(NpeaksAxis), '.', 'markerSize' ,10)
%     
%     
    for kk = 1:length(NpeaksAxis)
        ampsDiff = (pointsReal(kk,2) - pointsNN(:,2));                
        fDiff =  (pointsReal(kk,1) - pointsNN(:,1));        
        dist = sqrt((fDiff).^2 + (ampsDiff).^2);      
        [minDist, minLoc] = min(dist);
        distances(kk) = minDist;        
        if isempty(map)
            map(kk) = minLoc;
        end
        map(kk) = minLoc;           
%         lineFreqz =  [f0(NpeaksAxis(kk)), fNN(minLoc)];
%         lineAmps = [ampsReal(NpeaksAxis(kk)), ampNN(minLoc)];
%         plot(lineFreqz, lineAmps);
%         pause(0.01);
    end
    
    hold off;
    
%         gains = ones(size(NpeaksAxis));
%     gains = [2.75,1,1,1.75,1.75,1,1].';

    L2_amp = abs( (pointsReal(:,2) - pointsNN(map,2)) ); 
    %L2_freq = abs(gains .* ((pointsReal(:,1) - pointsNN(map,1))) );
    L2_freq = abs( pointsReal(:,1) - pointsNN(map,1) );
    L2 = sum(L2_freq);%/sum(gains);
    
   if length(unique(map)) ~= length(map)
        L2 = L2 + 1e3; 
   end   
%    pause(0.1);  
end

