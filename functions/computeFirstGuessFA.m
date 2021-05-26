function [L2, firstGuessLoc] = computeFirstGuessFA(Dataset_FA, fAmps, f0, plotData)
%COMPUTELOSSFUNCTIONS Summary of this function goes here
%   Detailed explanation goes here
    %% 1) get Data

    nTuples = length(Dataset_FA.inputs(:,1));                
    L2 =    zeros(nTuples , 1);
    
    %% Loss function calculations
    NpeaksAxis = 1:12;    
    psi = 600;
    

    
    for jj = 1:nTuples
        L = 0;
        % Amplitude real/comsol scaling
        ratio = (fAmps(1:5))./abs(Dataset_FA.outputsAmp(jj, 1:5).');
        gamma = mean(ratio);
        % Frequency/amplitude scaling
        ampsComsol = gamma * abs(Dataset_FA.outputsAmp(jj, :));
        ampsReal =  fAmps;
        eta = mean(f0(NpeaksAxis)./(ampsReal(NpeaksAxis)) );
        ampsComsol = eta*ampsComsol;
        ampsReal = eta*ampsReal;

        % Allocate points
        eigenFreqz = Dataset_FA.outputsEig(jj, :);
        pointsComsol = [eigenFreqz.', ampsComsol.'];
        pointsReal = [f0(NpeaksAxis), ampsReal(NpeaksAxis)];
        
        if plotData
        figure(150)
        plot( eigenFreqz, ampsComsol , '.', 'markerSize', 10)
        hold on;
        xlabel('frequency');
        ylabel('amplitude');
        title(['jj = ', num2str(jj)]);
        plot(f0(NpeaksAxis), ampsReal(NpeaksAxis), '.', 'markerSize' ,10)
        xlim([f0(1)-10, eigenFreqz(end)+20]);
        end
        
        
        map = [];
        for kk = NpeaksAxis
            ampsDiff = pointsReal(kk,2) - pointsComsol(:,2);                
            fDiffReal = (pointsReal(kk,1) - pointsComsol(:,1))./(pointsReal(kk,1));
            dist = sqrt(( psi* fDiffReal).^2 + (ampsDiff).^2);
            [minDist, minLoc] = min(dist);
            
            if isempty(map)
                map(kk) = minLoc;
            else
                if ~isempty( find(map == minLoc));
                    L = L + 1e6; 
                end
            end
            
            lineFreqz =  [f0(kk), eigenFreqz(minLoc)];
            lineAmps = [ampsReal(kk), ampsComsol(minLoc)];
            if plotData
            plot(lineFreqz, lineAmps);
            pause(0.2);
            end
            L = L + minDist; 
            map(kk) = minLoc;
        end
        hold off
        L2(jj) = L;
    end

[minVal, firstGuessLoc] = min(L2);
end

