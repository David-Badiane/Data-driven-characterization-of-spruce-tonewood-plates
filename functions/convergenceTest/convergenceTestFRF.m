function [timesAmp, pointVelMatrix] = convergenceTestFRF(model, fAxis, convergencePath)
%CONVERGENCETESTFRF Summary of this function goes here
%   Detailed explanation goes here
    % FREQUENCY DOMAIN STUDY
    cd(convergencePath);
    index = [9,8,7,6,5,4,3,2];
    timesAmp = zeros(size(index));
    pointVelMatrix = zeros(length(index), length(fAxis));
    
    fAxis = fAxis(:).';
    model.component('comp1').physics('solid').feature('lemm1').feature('dmp1').active(true);
    model.study('std2').feature('freq').set('plist', num2str(fAxis.'));

    for ii = 1:length(index)
        model.component('comp1').mesh('mesh1').feature('size').set('hauto', index(ii));
        model.mesh('mesh1').run;
        
        tstart = tic;
        model.study('std2').run();
        timesAmp(ii) = toc(tstart);

        model.result.export('data1').set('transpose', true);
        model.result.export('data1').set('sdim', 'fromdataset');
        exportData(model,'cpt1', convergencePath,['vel', int2str(ii)],'solid.u_tZ'); % velocity
        [vel] = readTuples(['vel', int2str(ii),'.txt'], 1, false);
        vel = vel(4:end);
        pointVelMatrix(ii,:) = vel;        
    end
    % relative error in percentage
    cd(baseFolder)
end

