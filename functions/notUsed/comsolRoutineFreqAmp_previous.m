function [vel,eigenFreqz] = comsolRoutineFreqAmp_previous(model, mechParams,paramsNames)
%COMSOLROUTINE Summary of this function goes here
%   Detailed explanation goes here
    nModes = 15;
    model.component('comp1').physics('solid').feature('lemm1').feature('dmp1').set('alpha_dM',mechParams(end-1));
    model.component('comp1').physics('solid').feature('lemm1').feature('dmp1').set('beta_dK',mechParams(end));

    % a) set parameters and points to study
    tstart = tic;
    for ii = (1:length(paramsNames))
        model.param.set(paramsNames(ii), mechParams(ii));
    end
    %params = mphgetexpressions(model.param)
    
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
    
    tElapsed = toc(tstart);
    disp(tElapsed);
end

