function [eigenFreqzMatrix, pointVelMatrix, eigRate, timesEig] = ConvergenceTestMesh(model,nModes, convergencePath)
%CONVERGENCETESTMESH Summary of this function goes here
%   Detailed explanation goes here
baseFolder = pwd;
cd(convergencePath);
index = [9,8,7,6,5,4,3,2];
eigenFreqzMatrix = zeros(length(index), nModes);
pointVelMatrix = zeros(length(index), 8);

eigRate = zeros(size(index));
FDRate = zeros(size(index)); 

timesEig = zeros(size(index));
timesFD = zeros(size(index));

model.study('std1').feature('eig').set('neigs', int2str(nModes));
model.component('comp1').physics('solid').feature('lemm1').feature('dmp1').active(false);

% EIGENFREQUENCY STUDY
    for ii = 1:length(index)
        model.component('comp1').mesh('mesh1').feature('size').set('hauto', index(ii));
        model.mesh('mesh1').run;
        tstart = tic;
        model.study('std1').run(); 
        timesEig(ii) = toc(tstart);
        
        evalFreqz = mpheval(model, 'solid.freq', 'Dataset', 'dset1', 'edim',0, 'selection', 1);
        eigenFreqz = real(evalFreqz.d1');
     
        eigenFreqzMatrix(ii,:) = eigenFreqz;
    end
    
    % relative error in percentage
    for ii = 1:length(index)
        eigRate(ii) = mean((real(eigenFreqzMatrix(ii,:)) - real(eigenFreqzMatrix(end,:)))./real(eigenFreqzMatrix(end,:)))*100;
    end
    
%     % FREQUENCY DOMAIN STUDY
% model.component('comp1').physics('solid').feature('lemm1').feature('dmp1').active(true);
% model.study('std2').feature('freq').set('plist', num2str(eigenFreqz(1:8).'));
% 
%     for ii = 1:length(index)
%         model.component('comp1').mesh('mesh1').feature('size').set('hauto', index(ii));
%         model.mesh('mesh1').run;
%         
%         tstart = tic;
%         model.study('std2').run();
%         timesFD(ii) = toc(tstart);
% 
%         model.result.export('data1').set('transpose', true);
%         model.result.export('data1').set('sdim', 'fromdataset');
%         exportData(model,'cpt1', convergencePath,['vel'],'solid.u_tZ'); % velocity
%         [vel] = readTuples(['vel.txt'], 1, false);
%         vel = vel(4:end);
%         pointVelMatrix(ii,:) = vel;        
%     end
%     % relative error in percentage
%     for ii = 1:length(index)
%         FDRate(ii) = mean((abs(pointVelMatrix(ii,:)) - abs(pointVelMatrix(end,:)))./abs(pointVelMatrix(end,:)))*100;
%     end
cd(baseFolder)
end

