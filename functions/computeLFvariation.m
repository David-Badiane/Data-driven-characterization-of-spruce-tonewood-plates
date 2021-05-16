function [LvaluesFreq, LvaluesAmps ] = computeLFvariation(nPoints, fAmps, f0, perc)
%COMPUTELOSSFUNCTIONS Summary of this function goes here
%   Detailed explanation goes here
    %% 1) get Data
    eigFreqsRef = table2array(readtable('Eigenfrequencies.csv'));
    AmpsRef = abs(table2array(readtable('Amplitudes.csv')));    
    
    LvaluesFreq =    struct('L1', zeros(nPoints, length(AmpsRef)), ...
                            'L2', zeros(nPoints, length(AmpsRef)), ...
                            'L3', zeros(nPoints, length(AmpsRef)));
    LvaluesAmps=     struct('L1', zeros(nPoints, length(AmpsRef)), ...
                            'L2', zeros(nPoints, length(AmpsRef)), ...
                            'L3', zeros(nPoints, length(AmpsRef)));
                    
    %% Loss function calculations
    NpeaksAxis = 1:8;
    psi = 600;
    
    eigFreqsVar = zeros(nPoints, length(eigFreqsRef));
    AmpsVar = zeros(nPoints, length(AmpsRef));
    
    for ii = 1:length(AmpsRef)
        stepEig =  2*perc*eigFreqsRef(ii)/(nPoints - 1 );
        stepAmp =  2*perc*AmpsRef(ii)/(nPoints - 1 );
        eigFreqsVar(:,ii) = eigFreqsRef(ii)*(1-perc): stepEig : eigFreqsRef(ii)*(1+perc);
        AmpsVar(:,ii) = AmpsRef(ii)*(1-perc): stepAmp : AmpsRef(ii)*(1+perc);
    end
      
    %% Frequency cycle - amplitudes steady
    for ii = 1:length(eigFreqsRef)        
        for jj = 1:nPoints
            L1 = 0;
            L2 = 0;
            L3 = 0;
            
            testFreqs = eigFreqsRef;
            testFreqs(ii) = eigFreqsVar(jj,ii); 
            
            % Amplitude real/comsol scaling
            ratio = (fAmps(1:5))./abs(AmpsRef(1:5).');
            gamma = mean(ratio);

            % Frequency/amplitude scaling
            ampsComsol = gamma * AmpsRef.';
            ampsReal =  fAmps;
            eta = mean(f0(NpeaksAxis)./(ampsReal(NpeaksAxis)) );
            ampsComsol = eta*ampsComsol;
            ampsReal = eta*ampsReal;

            % Allocate points
            eigenFreqz = testFreqs.';
            pointsComsol = [eigenFreqz, ampsComsol];
            pointsReal = [f0(NpeaksAxis), ampsReal(NpeaksAxis)];
            
            for kk = NpeaksAxis
                ampsDiff = pointsReal(kk,2) - pointsComsol(:,2);
                ampsDiffNorm = (pointsReal(kk,2) - pointsComsol(:,2))/pointsReal(kk,1);
                
                fDiffComsol = (pointsReal(kk,1) - pointsComsol(:,1))./(pointsComsol(:,1));
                fDiffReal = (pointsReal(kk,1) - pointsComsol(:,1))./(pointsReal(kk,1));
                
                dist1 = sqrt(( psi* fDiffComsol).^2 + (ampsDiff).^2);
                dist2 = sqrt(( psi* fDiffReal).^2 + (ampsDiff).^2);
                dist3 = sqrt(( fDiffReal).^2 + (ampsDiffNorm).^2);
               
                [minDist1, minLoc1] = min(dist1);
                [minDist2, minLoc2] = min(dist2);
                [minDist3, minLoc3] = min(dist3);

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
                
            LvaluesFreq.L1(jj,ii) = L1; 
            LvaluesFreq.L2(jj,ii) = L2;   
            LvaluesFreq.L3(jj,ii) = L3;  
        end
    end
    
    percVector = -perc*100:2*perc*100/(nPoints-1):perc*100;
       
    figure() 
    hold on;
    xlabel('\Delta X  [%]');
    ylabel('Loss function value');
    title('normalized with Comsol - f')
    names = {'f1', 'f2', 'f3', 'f4', 'f5', 'f6', 'f7', 'f8', 'f9', 'f10', 'f11', 'f12', 'f13', 'f14', 'f15'};  

    for ii = 1:length(AmpsRef)
       plot(percVector, LvaluesFreq.L1(:,ii),'-o', 'lineWidth', 1.05,'MarkerSize',2);
       legend(names{1:ii});
    end
    
    figure() 
    hold on;
    xlabel('\Delta X  [%]');
    yy = ylabel('L ');
    t = title('Landscape frequency   $\qquad \sqrt{\left(\psi \frac{\hat{f}_r - f_s}{\hat{f}_r}\right)^2 + (\hat{a}_r - a_s)^2} $');
    set(t, 'Interpreter', 'latex')
    set(yy, 'Interpreter', 'latex')

    ax = gca;
    ax.XMinorTick = 'on';
    ax.YMinorTick = 'on';
    ax.TickDir = 'out';
    ax.FontSize = 20;
    
    for ii = 1:length(AmpsRef)
       plot(percVector, LvaluesFreq.L2(:,ii),'-o', 'lineWidth', 1.05,'MarkerSize',2);
       legend(names{1:ii});
    end
    
    figure() 
    hold on;
    xlabel('\Delta X  [%]');
    ylabel('Loss function value');
    title('normalized with Real - both f and Amp');
    
    for ii = 1:length(AmpsRef)
       plot(percVector, LvaluesFreq.L3(:,ii),'-o', 'lineWidth', 1.2);
       legend(names{1:ii});
    end   
    
    
    %% Amplitudes varying - Frequencies steady
    for ii = 1:length(AmpsRef)        
        for jj = 1:nPoints
            L1 = 0;
            L2 = 0;
            L3 = 0;
            
            testAmps = AmpsRef;
            testAmps(ii) = AmpsVar(jj,ii); 
            
            % Amplitude real/comsol scaling
            ratio = (fAmps(1:5))./abs(testAmps(1:5).');
            gamma = mean(ratio);

            % Frequency/amplitude scaling
            ampsComsol = gamma * testAmps.';
            ampsReal =  fAmps;
            eta = mean(f0(NpeaksAxis)./(ampsReal(NpeaksAxis)) );
            ampsComsol = eta*ampsComsol;
            ampsReal = eta*ampsReal;

            % Allocate points
            eigenFreqz = eigFreqsRef.';
            pointsComsol = [eigenFreqz, ampsComsol];
            pointsReal = [f0(NpeaksAxis), ampsReal(NpeaksAxis)];
            
            for kk = NpeaksAxis
                ampsDiff = pointsReal(kk,2) - pointsComsol(:,2);
                ampsDiffNorm = (pointsReal(kk,2) - pointsComsol(:,2))/pointsReal(kk,1);
                
                fDiffComsol = (pointsReal(kk,1) - pointsComsol(:,1))./(pointsComsol(:,1));
                fDiffReal = (pointsReal(kk,1) - pointsComsol(:,1))./(pointsReal(kk,1));
                
                dist1 = sqrt(( psi* fDiffComsol).^2 + (ampsDiff).^2);
                dist2 = sqrt(( psi* fDiffReal).^2 + (ampsDiff).^2);
                dist3 = sqrt(( fDiffReal).^2 + (ampsDiffNorm).^2);
               
                [minDist1, minLoc1] = min(dist1);
                [minDist2, minLoc2] = min(dist2);
                [minDist3, minLoc3] = min(dist3);

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
                
            LvaluesAmps.L1(jj,ii) = L1; 
            LvaluesAmps.L2(jj,ii) = L2;   
            LvaluesAmps.L3(jj,ii) = L3;  
        end
    end
    
    percVector = -perc*100:2*perc*100/(nPoints-1):perc*100;
       
    figure() 
    hold on;
    names = {'Amp1', 'Amp2', 'Amp3', 'Amp4', 'Amp5', 'Amp6', 'Amp7', 'Amp8', 'Amp9', 'Amp10', 'Amp11', 'Amp12', 'Amp13', 'Amp14', 'Amp15'};  
    xlabel('\Delta X  [%]');
    ylabel('Loss function value');
    title('normalized with Comsol - f');
    
    for ii = 1:length(AmpsRef)
       plot(percVector, LvaluesAmps.L1(:,ii),'-o', 'lineWidth', 1.2);
       legend(names{1:ii});
    end
    
    
    figure() 
    hold on;
    xlabel('\Delta X  [%]');
    ylabel('Loss function value');
    t = title('Landscape amplitude $\qquad \sqrt{\left(\psi \frac{\hat{f}_r - f_s}{ hat{f}_r}\right)^2 + (\hat{a}_r - a_s)^2}$');
    set(t, 'Interpreter', 'latex')
    set(yy, 'Interpreter', 'latex')

    ax = gca;
    ax.XMinorTick = 'on';
    ax.YMinorTick = 'on';
    ax.TickDir = 'out';
    ax.FontSize = 20;
    for ii = 1:length(AmpsRef)
       plot(percVector, LvaluesAmps.L2(:,ii),'-o', 'lineWidth', 1.05, 'MarkerSize', 5);
       legend(names{1:ii});
    end
    
    figure() 
    hold on;
    xlabel('\Delta X  [%]');
    ylabel('Loss function value');
    title('normalized with Real - both f and Amp');
    
    for ii = 1:length(AmpsRef)
       plot(percVector, LvaluesAmps.L3(:,ii),'-o', 'lineWidth', 1.2);
       legend(names{1:ii});
    end
    
    
end

