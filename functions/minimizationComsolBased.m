function [err] = minimizationComsolBased(mechParams, model, f0, fAmps, constraintWeight,paramsNames, referenceVals)
%MINIMIZATIONCOMSOLBASED Summary of this function goes here
%   Detailed explanation goes here
    % a) set parameters and points to study
    tstart = tic;
    for ii = (1:length(mechParams))
        model.param.set(paramsNames(ii), mechParams(ii));
    end
    nModes = 15;
    model.study('std1').feature('eig').set('neigs', int2str(nModes));
    model.component('comp1').physics('solid').feature('lemm1').feature('dmp1').active(false);
    model.study('std1').run(); 
    model.component('comp1').physics('solid').feature('lemm1').feature('dmp1').active(true);

    evalFreqz = mpheval(model,'solid.freq','Dataset','dset1','edim',0,'selection',1);
    eigenFreqz = real(evalFreqz.d1');
    eigenFreqz = eigenFreqz(eigenFreqz<600);
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
    vel = vel(4:end);
    
    toView = 1:8;
    ratio = fAmps(1:5)./abs(vel(1:5)).';
    factor = mean(ratio);
    
    ampsComsol = factor * abs(vel);
    ampsReal =  fAmps;
    factorAmp = mean(f0(toView)./(ampsReal(toView)) );
    ampsComsol = factorAmp*ampsComsol;
    ampsReal = factorAmp*ampsReal;

    figure(150)
    plot( eigenFreqz, ampsComsol , '.', 'markerSize', 10)
    hold on;
    xlabel('frequency');
    ylabel('amplitude');
    title('first 8 eigenfrequencies');
    plot(f0(toView), ampsReal(toView), '.', 'markerSize' ,10)
    

    pointsComsol = [eigenFreqz, ampsComsol.'];
    pointsReal = [f0(toView), ampsReal(toView)];

    minimumDistantPoints = zeros(2, length(toView));

    err = 0;
    
    for ii = 1:length(toView)
        dist = sqrt(100*(pointsReal(ii,1) - pointsComsol(:,1)).^2 + (pointsReal(ii,2) - pointsComsol(:,2)).^2);
        [minDist, minLoc] = min(dist);
        minimumDistantPoints(:,ii) =  pointsComsol(minLoc,:);

        lineFreqz =  [f0(ii), eigenFreqz(minLoc)];
        lineAmps = [ampsReal(ii), ampsComsol(minLoc)];

        err = err + sqrt(100*( ( (lineFreqz(1) - lineFreqz(2))/lineFreqz(1) )^2 + (lineAmps(1) - lineAmps(2))^2 ));
        plot(lineFreqz, lineAmps);
    end
    legend('Comsol', 'experimental', 'f1', 'f2' , 'f3', 'f4', 'f5', 'f6','f7', 'f8');
    hold off;
    tElapsed = toc(tstart);
    disp(tElapsed);
    
%      err = 0;
%      indexes = [3,5,6,7,8,9];    
%      for ii = 1:length(indexes)
%         err = err + constraintWeight*(mechParams(indexes(ii)) - referenceVals(indexes(ii)))^2; 
%      end
     
%      for ii = 1:length(toView)
%         err = err + ( (f0(ii) - minimumDistantPoints(1,ii)) /f0(ii))^2 + ...
%                     ( (fAmps(ii) - minimumDistantPoints(2,ii)) / fAmp(ii) )^2;
%      end
     
end

