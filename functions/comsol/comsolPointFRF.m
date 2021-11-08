function [vel, fAxisComsol, resultsFilename] = comsolPointFRF(model, resultsPath, meshSize, fAxis, fHigh,...
                             nPointsAxis, nPeaksUsed, alpha, beta, nRealizations, getReferenceValsResult, ampScale)
    ampScale = ['_' ampScale];
    if nargin < 12, ampScale = ''; end
    
    tStart = tic;
    % Perform modal analysis first
    cd(resultsPath)
    if ~getReferenceValsResult
    resultsFilename = ['Results_nR', int2str(nRealizations),'_',int2str(nPeaksUsed) ampScale];
    else
    resultsFilename = input('insert filename without tag (string): ');    
    end
    
    mechParams = table2array(readtable([resultsFilename,'.csv']));
    mechParams = mechParams(1,:);
    mechParams(1) = 400;
    mechParams(end-1) = alpha;
    mechParams(end) = beta;
    
    params = mphgetexpressions(model.param);
    paramsNames = params(7:end,1);
    freqAxNames = paramsNames(end-2:end);
    fAxisComsol_temp = fAxis(fAxis <= fHigh);
    
    fAxisComsol = logspace(log10(fAxisComsol_temp(1)),log10(fAxisComsol_temp(end)), nPointsAxis);
    
    disp('results filename:')
    disp(resultsFilename)
    
    for ii = (1:length(mechParams))
        model.param.set(paramsNames(ii), mechParams(ii));
    end

    model.physics('solid').feature('lemm1').feature('dmp1').active(true)
    model.component('comp1').physics('solid').feature('lemm1').feature('dmp1').set('DampingType', 'RayleighDamping');
    model.component('comp1').physics('solid').feature('lemm1').feature('dmp1').set('InputParameters', 'AlphaBeta'); 

    % b) Set Rayleigh constants
    model.component('comp1').physics('solid').feature('lemm1').feature('dmp1').set('alpha_dM', mechParams(end-1));
    model.component('comp1').physics('solid').feature('lemm1').feature('dmp1').set('beta_dK', mechParams(end));

    % c) Run study and mesh
    model.component('comp1').mesh('mesh1').feature('size').set('hauto', meshSize);
    model.study('std2').feature('freq').set('plist', num2str(fAxisComsol));
    model.mesh('mesh1').run;
    model.study('std2').run();

    % d) Export Results
    dirName = pwd;
    if ~getReferenceValsResult
        filenameExported = ['vel_nR',int2str(nRealizations),'_', int2str(nPeaksUsed),'_a_b_', num2str(alpha),'_',num2str(beta)];
    else
        filenameExported = [resultsFilename, '_beta_', num2str(beta)];
    end
    
    model.result.export('data1').set('transpose', true);
    model.result.export('data1').set('sdim', 'fromdataset');
    exportData(model,'cpt1', dirName, filenameExported, 'solid.u_tZ'); % velocity  
    
    if ~getReferenceValsResult
        fAxisName = ['vel_fAxis_nR', int2str(nRealizations),'_',int2str(nPeaksUsed),'_a_b_', num2str(alpha),'_',num2str(beta)];
    else
        fAxisName = [resultsFilename,'_fAxis_beta_', num2str(beta)];
    end
    writeMat2File(fAxisComsol, [fAxisName,'.csv'], {'f'}, 1, false);
    
    % read velocity
    [vel] = readTuples([filenameExported,'.txt'], 1, false);
    vel = vel(4:end);
    disp(['elapsed time: ', int2str(floor(toc(tStart)/60)), ' minutes ', int2str(mod(toc(tStart),60)), ' seconds']);

end