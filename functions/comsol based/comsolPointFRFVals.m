function [vels, fAxisComsol] = ...
            comsolPointFRFVals(model, mechParams, path, meshSize, fAxis, fHigh, nPts_fAxis,...
            comsolDset, paramsIdxsModel, writeFiles, printData, expression, outputFilename)
    
    if nargin < 8, comsolDset = 'cpt1'; end
    if nargin < 9, paramsIdxsModel = 1:12; end
    if nargin < 10, writeFiles = 0; end
    if nargin < 11, printData  = 1; end
    if nargin <12, expression = {'solid.u_tZ'}; end
    if nargin <13, outputFilename = 'FRF_data'; end
    tStart = tic;
    % Perform modal analysis first
    cd(path)   
    
    params = mphgetexpressions(model.param);
    paramsNames = params(paramsIdxsModel,1);
    
    if printData
    disp(['setting params :',newline]);
    paramsNames(:).'
    disp(num2str(mechParams));
    end
    
    fAxisComsol_temp = fAxis(fAxis <= fHigh);
    fAxisComsol = logspace(log10(fAxisComsol_temp(1)),log10(fAxisComsol_temp(end)), nPts_fAxis);
    setParams(model, paramsNames, mechParams);
    
    model.physics('solid').feature('lemm1').feature('dmp1').active(true);

    % c) Run study and mesh
    model.component('comp1').mesh('mesh1').feature('size').set('hauto', meshSize);
    model.study('std2').feature('freq').set('plist', num2str(fAxisComsol));
    model.mesh('mesh1').run;
    model.study('std2').run()

    % d) Export Results
    dirName = pwd;
    filenameExported = 'vel';
    
    model.result.export('data1').set('transpose', true);
    model.result.export('data1').set('sdim', 'fromdataset');
    
    vels = [];
    
    for jj = 1:length(comsolDset)
        exportData(model,comsolDset{jj}, dirName, [filenameExported,'_',num2str(jj)],...
                   expression); % velocity  
            % read velocity
        [vel] = readTuples([filenameExported,'_',num2str(jj),'.txt'], 1, false);
        vel = vel(4:end); 
        vels = [vels; vel];
        if ~writeFiles
            delete([filenameExported,'_',num2str(jj) '.txt'])
        end
    end
    
    fAxisName = [filenameExported,'_fAx'];
    if writeFiles
        writeMat2File([fAxisComsol(:), vels.'], [outputFilename,'.csv'], {'f'}, 1, false);
    end

    disp(['elapsed time: ', int2str(floor(toc(tStart)/60)), ' minutes ', int2str(mod(toc(tStart),60)), ' seconds']);

end