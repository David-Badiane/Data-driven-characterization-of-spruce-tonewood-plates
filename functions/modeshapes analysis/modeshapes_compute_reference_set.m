function [refFilename] = modeshapes_compute_reference_set(model, nModesRef, ...
                                                          csvPath,reference_parameters,...
                                                          reference_parameters_names, isWedge)
    % modeshapes_compute_reference_set
    % this function computes and saves the modeshapes for given 
    % input parameter - (material, damping, geometry)
    % ---------------------------------------------------------------------
    % inputs:
    % model                       = comsol model of gPlate.comsol
    % nModesRef                   = int - number of modes to compute
    % csvPath                     = string - path of the dataset directory
    % reference_parameters        = double array - input parameters values
    % reference_parameters_names  = cell array   - input parameters names
    % ---------------------------------------------------------------------
    % outputs:
    % refFilename = string - filename of a file with the generated reference
    %                        modeshapes
    % ---------------------------------------------------------------------
    
    if nargin <6, isWedge = 0; end
    
    % set parameters in the Comsol model
    setParams(model,reference_parameters_names, reference_parameters);
    
    % set variables names (columns names) for the csv files
    varNamesModeshapes = {'x' 'y' 'z'};
    modesFilename = 'mShape';
    for ii = 1:(nModesRef)
        varNamesModeshapes{ii+3} = ['f',int2str(ii)]; 
    end
    
    
    % obtain eigenfrequecnies
    disp(['start eig study with:  nEigs = ', num2str(nModesRef)]);
    tStart = tic;
    % Comsol livelink with Matlab
    model.study('std1').feature('eig').set('neigs', int2str(nModesRef)); % ---> set number of modes
    model.component('comp1').physics('solid').feature('lemm1').feature('dmp1').active(true); % --> deactivate damping
    model.study('std1').run(); 
    model.component('comp1').physics('solid').feature('lemm1').feature('dmp1').active(true);
    disp(['end of the study in ', num2str(toc(tStart)), ' seconds' ]);
    
    % export modeshapes
    % if it is a wedge export the subsequent expression
    if isWedge
        expression = {'solid.u_tZ'};
            refFilename = 'refModeshapes_raw.csv';
    else 
        expression = {'solid.disp'};
        refFilename = 'refModeshapes_raw.csv';
    end
    % do not transpose data
    model.result.export('data1').set('transpose', false);
    
    % if isWedge export on a surface  
    if isWedge, model.result.export('data1').set('data', 'surf1'); 
    % else use the plate dataset
    else, model.result.export('data1').set('data', 'dset1'); end
    
    % show messages
    disp(['exporting in ', csvPath ]);
    disp(['file  ', refFilename, ' expression " ', expression{1} ,' "']);
    
    % export_modes and save them into file
    exportAllModesFromDataset(model, modesFilename,[csvPath,'\modesAnalysis'],expression);
    cd([csvPath,'\modesAnalysis']);
    fileData = readTuples([modesFilename,'.txt'], nModesRef+3, true);
    fileData = [fileData(:,1:3) fileData(:,4:end)];
    delete([modesFilename,'.txt']); 
   
    csvwrite(refFilename, fileData);   
    
end