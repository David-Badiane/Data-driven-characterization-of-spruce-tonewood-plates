function [err] = minimizationComsolBased2(mechParams, model, f0, fAmps, constraintWeight,paramsNames, referenceVals,create)
%MINIMIZATIONCOMSOLBASED Summary of this function goes here
%   Detailed explanation goes here

%     beta = [3e-7, 7e-7, 1e-6, 3e-6,  7e-6, 9e-6, 1e-5, 2e-5, 3e-5];
%     legendNames = cell(length(beta),1);
    
%     figure()
%     hold on;
%     for jj = 1:length(beta)
%     legendNames{jj} = ['beta = ', num2str(beta(jj))];
    if create
       [files] = fastOpen({'fDistPerc' 'fDistScale' 'ADist' 'ParamsEvolution' 'EigFreqz' },'w', '.csv');
       fclose('all');
    end

    
    fDistPerc = table2array(readtable('fDistPerc.csv'));
    fDistScale = table2array(readtable('fDistScale.csv'));
    ADist = table2array(readtable('ADist.csv'));
    ParamsEvolution = table2array(readtable('ParamsEvolution.csv'));
    eigFreqz = table2array(readtable('EigFreqz.csv'));
    nModes = 15;
    
    model.component('comp1').physics('solid').feature('lemm1').feature('dmp1').set('alpha_dM',mechParams(end-1));
    model.component('comp1').physics('solid').feature('lemm1').feature('dmp1').set('beta_dK',mechParams(end));

    % a) set parameters and points to study
    tstart = tic;
    for ii = (1:length(mechParams))
        model.param.set(paramsNames(ii), mechParams(ii));
    end
    
    model.study('std1').feature('eig').set('neigs', int2str(nModes));
    model.component('comp1').physics('solid').feature('lemm1').feature('dmp1').active(false);
    model.study('std1').run(); 
    model.component('comp1').physics('solid').feature('lemm1').feature('dmp1').active(true);

    evalFreqz = mpheval(model,'solid.freq','Dataset','dset1','edim',0,'selection',1);
    eigenFreqz = real(evalFreqz.d1');
    eigenFreqz = eigenFreqz(:);
    
    model.study('std2').feature('freq').set('plist', num2str(eigenFreqz.'));
    
    % b) run study
    model.study('std2').run();

    % c) Export Results
    dirName = pwd;
    model.result.export('data1').set('transpose', true);
    model.result.export('data1').set('sdim', 'fromdataset');
    exportData(model,'cpt1', dirName,['vel'],'solid.u_tZ'); % velocity 
    
    [vel] = readTuples(['vel.txt'], 1, false);
    vel = vel(4:end).';
       
    NpeaksAxis = 1:8;
    ratio = fAmps(1:5)./abs(vel(1:5));
    gamma = mean(ratio);
    
    ampsComsol = gamma * abs(vel);
    ampsReal =  fAmps;
    eta = mean(f0(NpeaksAxis)./(ampsReal(NpeaksAxis)) );
    ampsComsol = eta*ampsComsol;
    ampsReal = eta*ampsReal;
    
%     plot(1:15, ampsComsol, '-o', 'lineWidth', 1.4, 'markerSize', 7);
%     xlabel('n eigenfrequencies');
%     ylabel('|H_v(f_n)|    [s/kg]');
%     title('amplitudes variation with beta');
%     end
%     legend(legendNames);
%     ax = gca;
%     ax.XMinorTick = 'on';
%     ax.YMinorTick = 'on';
%     ax.TickDir = 'out';
%     ax.FontSize = 20;
%     plot([1,2,3,4,5,7,9,10], ampsReal(1:8), '-.x', 'lineWidth', 1.4, 'markerSize', 7 );
%     legend(legendNames,'real Amps')

    pointsComsol = [eigenFreqz, ampsComsol];
    pointsReal = [f0(NpeaksAxis), ampsReal(NpeaksAxis)];

    minimumDistantPoints = zeros(2, length(NpeaksAxis));

    err = 0;
    pastLoc = [];
    Jchi = zeros(length(NpeaksAxis),1);
    psi = 600;
    
    figure(150)
    plot( eigenFreqz, ampsComsol , '.', 'markerSize', 10)
    hold on;
    xlabel('frequency');
    ylabel('amplitude');
    title(['alpha', num2str(mechParams(end-1)),' beta = ', num2str(mechParams(end))]);
    plot(f0(NpeaksAxis), ampsReal(NpeaksAxis), '.', 'markerSize' ,10)
    xlim([f0(1)-10, eigenFreqz(end)+20])
            
    distFreq = zeros(1,length(NpeaksAxis));
    distAmp = zeros(1,length(NpeaksAxis));
    selectedFreqz = zeros(1,length(NpeaksAxis));
    
    for ii = 1:length(NpeaksAxis)
        ampsDiff = pointsReal(ii,2) - pointsComsol(:,2);
        freqDiff = (pointsReal(ii,1) - pointsComsol(:,1))./(pointsReal(ii,1));
        dist = sqrt(( psi* freqDiff).^2 + (ampsDiff).^2);
        [minDist, minLoc] = min(dist);
        minimumDistantPoints(:,ii) =  pointsComsol(minLoc,:);
        if ii == 1
            pastLoc = 100;
        end

        if (pastLoc - minLoc) == 0
           err = err + 2e6; 
        end
        
        Jchi(ii) = (abs(pastLoc - minLoc) *100+0.01).^-1;

        lineFreqz =  [f0(ii), eigenFreqz(minLoc)];
        lineAmps = [ampsReal(ii), ampsComsol(minLoc)];

        err = err + minDist;
        plot(lineFreqz, lineAmps);
        pastLoc = minLoc;
        
        distFreq(ii) = (freqDiff(minLoc));
        distAmp(ii) = (ampsDiff(minLoc));
        selectedFreqz(ii) = eigenFreqz(minLoc);
        
    end

    fDistPerc = [fDistPerc; 100*distFreq];
    fDistScale = [fDistScale; psi*distFreq];
    ADist = [ADist; distAmp];
    ParamsEvolution = [ParamsEvolution; mechParams];
    eigFreqz = [eigFreqz; selectedFreqz];
    
    writeMat2File(fDistPerc,'fDistPerc.csv', {'[%] f'}, 1,false);     
    writeMat2File(fDistScale,'fDistScale.csv', {'f'}, 1,false);    
    writeMat2File(ADist,'ADist.csv', {'f'}, 1,false);
    writeMat2File(ParamsEvolution,'ParamsEvolution.csv',...
        {'Ex' 'Ey' 'Ez' 'Gxy' 'Gyz' 'Gxz' 'vxy' 'vyz' 'vxz' 'alpha' 'beta'}, 11, true);
    writeMat2File(eigFreqz,'EigFreqz.csv', {'f'}, 1,false);

    
    disp('freqDist');
    disp(distFreq);
    disp('ampDist');
    disp(distAmp);
    
    legend('Comsol', 'experimental', 'f1', 'f2' , 'f3', 'f4', 'f5', 'f6','f7', 'f8');
    hold off;
    tElapsed = toc(tstart);
    disp(tElapsed);
    disp(err);
    
%     figure(151)
%     plot(1:length(NpeaksAxis), Jchi, '.', 'markerSize', 10);
%     ylim([0, max(Jchi)*1.1])
%     xlabel('r  = {1,...,R =8} (real index)')
%     ylabel('J_{\chi}(r)')
%     title('J_{\chi(r)} vs r')
%     xlabel('n     n = 1,...,N = 5');
%     ylabel('dB');


%     figure(152)
%     hold on
%     plot(1:5, db(fAmps(1:5)), '-x', 'markerSize', 7 , 'lineWidth', 1.5);
%     plot(1:5, db(abs(vel(1:5))), '-.o', 'markerSize', 5, 'lineWidth', 1.5);
%     l = legend('$\hat{a}_n^*$', '$a_n^*$');
%     l.FontSize = 25;
%     set(l, 'Interpreter', 'latex')
%     xlabel('n-th peak');
%     ylabel('|H|_{dB}    [s/Kg]');
%     xlim([0.9,5.8])
%     ax = gca;
%     ax.XMinorTick = 'on';
%     ax.YMinorTick = 'on';
%     ax.TickDir = 'out';
%     ax.FontSize = 20;
%     hold off
    
%     figure(153)
%     hold on
%     plot(1:5, db(gamma*ones(1,5)) , 'lineWidth', 1.5);
%     plot(1:5, db(ratio),'-o', 'markerSize', 7, 'lineWidth', 1.5);
%     xlabel('n-th peak');
%     l = legend('$\gamma$','$\frac{\hat{a}_n^*}{a_n^*}$');
%     l.FontSize = 30;
%     set(l, 'Interpreter', 'latex')
%     ax = gca;
%     ax.XMinorTick = 'on';
%     ax.YMinorTick = 'on';
%     ax.TickDir = 'out';
%     ax.FontSize = 20;
%     ylabel('dB');
%     xlim([0.9, 5.8])
     
%     figure(154)
%     hold on
%     plot(1:5, db(fAmps(1:5)), '-x', 'markerSize', 7, 'lineWidth', 1.5);
%     plot(1:5, db(gamma*abs(vel(1:5))), '-.o', 'markerSize', 5,  'lineWidth', 1.5);
%     l = legend( '$\hat{a}_n^*$',  '$\gamma \cdot a_n^*$');
%     l.FontSize = 25;
%     set(l, 'Interpreter', 'latex')
%     xlabel('n-th peak');
%     ylabel('|H|_{dB}    [s/Kg]');
%     ax = gca;
%     ax.XMinorTick = 'on';
%     ax.YMinorTick = 'on';
%     ax.TickDir = 'out';
%     ax.FontSize = 20;
%     xlim([0.9, 5.8])     

    pause(0.8);
    
%      err = 0;
%      indexes = [3,5,6,7,8,9];    
%      for ii = 1:length(indexes)
%         err = err + constraintWeight*(mechParams(indexes(ii)) - referenceVals(indexes(ii)))^2; 
%      end
     
%      for ii = 1:length(NpeaksAxis)
%         err = err + ( (f0(ii) - minimumDistantPoints(1,ii)) /f0(ii))^2 + ...
%                     ( (fAmps(ii) - minimumDistantPoints(2,ii)) / fAmp(ii) )^2;
%      end
     
end

