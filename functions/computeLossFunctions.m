function [Lvalues, pointsMaps] = computeLossFunctions(mechParams,nPoints, fAmps, f0)
%COMPUTELOSSFUNCTIONS Summary of this function goes here
%   Detailed explanation goes here
    %% 1) get Data
    mechParams = table2array(readtable('FirstGuess_good.csv'));
    eigFreqz = table2array(readtable('Eigenfrequencies.csv'));
    Amps = table2array(readtable('Amplitudes.csv'));
    mechParamsVar = table2array(readtable('inputParams.csv'));
    
    
    pointsMaps = struct('L1', cell(nPoints, length(mechParams)), ...
                        'L2', cell(nPoints, length(mechParams)), ...
                        'L3', cell(nPoints, length(mechParams)));
                    
    Lvalues =    struct('L1', zeros(nPoints, length(mechParams)), ...
                        'L2', zeros(nPoints, length(mechParams)), ...
                        'L3', zeros(nPoints, length(mechParams)));
    
    %% Loss function calculations
    NpeaksAxis = 1:8;
    map1 = [NpeaksAxis; zeros(size(NpeaksAxis))];
    map2 = [NpeaksAxis; zeros(size(NpeaksAxis))];
    map3 = [NpeaksAxis; zeros(size(NpeaksAxis))];
    
    
    psi = 600;
    
    
    for ii = 1:length(mechParams)                
        
        check = [];
        for jj = 1:nPoints
            L1 = 0;
            L2 = 0;
            L3 = 0;
            
            idxMatrix = (ii-1)*(nPoints) + jj;
            
            % Amplitude real/comsol scaling
            ratio = (fAmps(1:5))./abs(Amps(idxMatrix, 1:5).');
            gamma = mean(ratio);

            % Frequency/amplitude scaling
            ampsComsol = gamma * abs(Amps(idxMatrix, :));
            ampsReal =  fAmps;
            eta = mean(f0(NpeaksAxis)./(ampsReal(NpeaksAxis)) );
            ampsComsol = eta*ampsComsol;
            ampsReal = eta*ampsReal;

            % Allocate points
            eigenFreqz = eigFreqz(idxMatrix, :);
            pointsComsol = [eigenFreqz.', ampsComsol.'];
            pointsReal = [f0(NpeaksAxis), ampsReal(NpeaksAxis)];
            
            for kk = NpeaksAxis

                ampsDiff = pointsReal(kk,2) - pointsComsol(:,2);
                ampsDiffNorm = (pointsReal(kk,2) - pointsComsol(:,2))/pointsReal(kk,2);
                
                fDiffComsol = (pointsReal(kk,1) - pointsComsol(:,1))./(pointsComsol(:,1));
                fDiffReal = (pointsReal(kk,1) - pointsComsol(:,1))./(pointsReal(kk,1));
                
                dist1 = sqrt(( psi* fDiffComsol).^2 + (ampsDiff).^2);
                dist2 = sqrt(( psi* fDiffReal).^2 + (ampsDiff).^2);
                dist3 = sqrt( (fDiffReal).^2 + (ampsDiffNorm).^2);
                                
                [minDist1, minLoc1] = min(dist1);
                [minDist2, minLoc2] = min(dist2);
                [minDist3, minLoc3] = min(dist3);
                
                map1(2,kk) =  minLoc1;
                map2(2,kk) =  minLoc2;
                map3(2,kk) =  minLoc3;
                
                if ii == 1
                    pastLoc = 100;
                end

                if (pastLoc - minLoc1) == 0
                   L1 = L1 + 2e6; 
                end
                if (pastLoc - minLoc2) == 0
                   L2 = L1 + 2e6; 
                end
                if (pastLoc - minLoc3) == 0
                   L3 = L1 + 2e6; 
                end

                L1 = L1 + minDist1;           
                L2 = L2 + minDist2;           
                L3 = L3 + minDist3;           
            end
            %disp(check);    
            Lvalues.L1(jj,ii) = L1;   
            Lvalues.L2(jj,ii) = L2;   
            Lvalues.L3(jj,ii) = L3;   
        end
%         figure()
       
    end
    percVector = -10:20/(10-1):+10;
    
    
    
    figure() 
    hold on;
    for ii = 1:length(mechParams)
       plot(percVector, Lvalues.L1(:,ii),'-o', 'lineWidth', 1.2);
    end
    names = {'Ex', 'Ey', 'Ez', 'Gxy', 'Gyz', 'Gxz', 'vxy', 'vyz', 'vxz', 'alpha', 'beta'};  
    legend(names{1:end});
    xlabel('\Delta X  [%]');
    ylabel('Loss function value');
    title('normalized with Comsol - f')
    ax = gca;
    ax.XMinorTick = 'on';
    ax.YMinorTick = 'on';
    ax.TickDir = 'out';
    ax.FontSize = 20;
    
    figure() 
    hold on;
    xlabel('\Delta X  [%]');
    ylabel('Loss function value');
    t = title(['$ L_2 = \sum_{r=1}^R  \left(\min_{s} d_r(s) + J_{\chi}(r) \right)   $']);
    set(t, 'Interpreter', 'latex')
    ax = gca;
    ax.XMinorTick = 'on';
    ax.YMinorTick = 'on';
    ax.TickDir = 'out';
    ax.FontSize = 20;
    ylim([260,390]);
   
    for ii = 1:length(mechParams)
       plot(percVector, Lvalues.L2(:,ii),'-o', 'lineWidth', 1.2);
       legend(names{1:ii});
    end

    
    
    figure() 
    hold on;
    for ii = 1:length(mechParams)
       plot(percVector, Lvalues.L3(:,ii),'-o', 'lineWidth', 1.2);
    end
    legend(names{1:end});
    xlabel('\Delta X  [%]');
    ylabel('Loss function value');
    title('normalized with Real - both f and Amp');    
    ax = gca;
    ax.XMinorTick = 'on';
    ax.YMinorTick = 'on';
    ax.TickDir = 'out';
    ax.FontSize = 20;
    
end

