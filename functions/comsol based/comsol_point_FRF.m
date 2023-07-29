function [simulated_FRFs, fAxisComsol] = ...
            comsol_point_FRF(model, resultsPath, resultsFilename, meshSize, fAx, fBounds, nPointsAxis,...
            dampingParams, dampingParams_idxs, comsolDset, paramsIdxsModel, expression)
    % function to simulate a point FRF on Comsol with Comsol livelink for Matlab
    % copyright: David Giuseppe Badiane
    % -------------------------------------------------------------------------
    % inputs:                       
    % model            = comsol finite element model
    % resultsPath      = string - path to the results directory
    % resultsFilename  = string - name of the results file
    % meshSize         = 1x1 double - mesh size (9=extremely coarse / 1=extremely fine) 
    % fAx              = 1xfreqBins double - frequency axis from experimentally measured FRFs 
    % fHigh            = 1x1 double - high bound for frequency axis
    % nPointsAxis      = 1x1 double - number of points of the frequency
    %                                 axis in the comsol simulation
    % dampingParams      = 1x2 double - Rayleigh damping control variables
    % dampingParams_idxs = 1x2 double - Rayleigh damping control variable
    %                                   indexes as parameters of the comsol model
    % comsolDset       = 1xnPoints cell - datasets from which comsol evaluates the FRF
    %                                 usually {'cpt1'} --> evaluates the FRF for single point of the plate, 
    %                                 you may input also 'cpt2' for another point (n.b. must be previously 
    %                                 defined in the comsol model)
    % paramsIdxsModel  = 1xM double - array with the indexes of the
    %                    parameters to update in the model (for this
    %                    application 1:15 --> [density, mechParams(9), damping(2), geometry(3)]
    % expression       = 1x1 cell   - expression to compute, to have a
    %                                 FRF set it to {'solid.ut_Z'} or {'solid.vel'}
    % -------------------------------------------------------------------------
    % outputs:
    % simulated_FRFs   = 1xnPointsAxis double - array of the FRF
    % fAxisComsol      = 1xnPointsAxis double - frequency axis associated to the FRF
    % -------------------------------------------------------------------------
    % set different function layouts
    if nargin < 9, comsolDset = 'cpt1'; end
    if nargin < 10, paramsIdxsModel = 1:15; end
    if nargin < 11, expression = {'solid.u_tZ'}; end
    tStart = tic; % timer
    
    % set variables
    FRF_filename = ['FRF_' resultsFilename, '_beta_', num2str(beta)];
    params = mphgetexpressions(model.param);
    paramsNames = params(paramsIdxsModel,1);
    simulated_FRFs = [];
    FRF_varNames = {'fAx'};
    for ii = 1:length(comsolDset)
       FRF_varNames{ii+1} = ['H_', int2str(ii)]; 
    end
    
    % read and set estimated parameters
    cd(resultsPath)
    mechParams = table2array(readtable([resultsFilename,'.csv']));
    mechParams = mechParams(1,:);
    mechParams(dampingParams_idxs) = dampingParams;
    disp(['results filename:', newline, resultsFilename, newline, 'setting params :'])
    array2table(mechParams, 'VariableNames', paramsNames)
    setParams(model, paramsNames, mechParams)
    
    % set fAxis to compute the FRF, we use a logarithmic frequency axis
    fAxisComsol_temp = fAx(fAx>=fBounds(1) & fAx <= fBounds(2)); 
    fAxisComsol = logspace(log10(fAxisComsol_temp(1)),log10(fAxisComsol_temp(end)), nPointsAxis);
    
    % activate damping and choose rayleigh damping
    model.physics('solid').feature('lemm1').feature('dmp1').active(true)
    model.component('comp1').physics('solid').feature('lemm1').feature('dmp1').set('DampingType', 'RayleighDamping');
    model.component('comp1').physics('solid').feature('lemm1').feature('dmp1').set('InputParameters', 'AlphaBeta'); 
   
    % Run mesh and frequency domain study
    model.component('comp1').mesh('mesh1').feature('size').set('hauto', meshSize); % --> set mesh
    model.study('std2').feature('freq').set('plist', num2str(fAxisComsol)); % --> set frequency axis
    model.mesh('mesh1').run;                                                % run mesh
    model.component('comp1').geom('geom1').run;                             % run geometry
    model.study('std2').run();                                              % runf frequency domain study

    % d) Export Results
    dirName = pwd;
    filenameExported = [resultsFilename, '_beta_', num2str(beta)];    
    model.result.export('data1').set('transpose', true);
    
    for jj = 1:length(comsolDset)
    exportData(model,comsolDset{jj}, dirName, [filenameExported,'_',num2str(jj)],...
               expression{:}); % velocity  
    % read txt file and convert it to csv
    [simulated_FRF] = readTuples([filenameExported,'_',num2str(jj),'.txt'], 1, false);
    simulated_FRF = simulated_FRF(4:end);
    simulated_FRFs = [simulated_FRFs; simulated_FRF];
    delete([filenameExported,'_',num2str(jj),'.txt']);
    end
    
    writeMat2File([fAxisComsol;simulated_FRFs].', [FRF_filename,'.csv'], FRF_varNames, length(FRF_varNames), 1);
    disp(['elapsed time: ', int2str(floor(toc(tStart)/60)), ' minutes ', int2str(mod(toc(tStart),60)), ' seconds']);
end