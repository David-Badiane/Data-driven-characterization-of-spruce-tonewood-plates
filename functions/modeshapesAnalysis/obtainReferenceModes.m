function [] = obtainReferenceModes(model, nModesRef, csvPath)
    % names of the modeshapes file
    varNamesModeshapes = cell(nModesRef+3,1);
    varNamesxyz = {'x' 'y' 'z'};
    modesFilename = 'mShape';
    for ii = 1:(nModesRef +3)
    if ii <4
    varNamesModeshapes{ii} = varNamesxyz{ii}; 
    else 
        varNamesModeshapes{ii} = ['f',int2str(ii-3)]; 
    end
    end
    % obtain eigenfreqeucnies
    disp(['start eig study with:  nEigs = ', num2str(nModesRef)]);
    tStart = tic;
    model.study('std1').feature('eig').set('neigs', int2str(nModesRef)); % ---> set number of modes
    model.component('comp1').physics('solid').feature('lemm1').feature('dmp1').active(false); % --> deactivate damping
    model.study('std1').run(); 
    disp(['end of the study in ', num2str(toc(tStart)), ' seconds' ]);
    % export modeshapes
    expression = {'solid.u_tX*nx+ solid.u_tY*ny + solid.u_tZ*nz'};

    model.result.export('data1').set('transpose', false);
    model.result.export('data1').set('data', 'surf1');
    disp(['exporting in ', csvPath ]);
    disp(['file  ', modesFilename, ' expression " ', expression{1} ,' "']);

    exportAllModesFromDataset(model, modesFilename,csvPath,expression);
    cd(csvPath);
    fileData = readTuples([modesFilename,'.txt'], nModesRef+3, true);
    fileData = [fileData(:,1:3) imag(fileData(:,4:end))];
    delete([modesFilename,'.txt']); 
    writeMat2File(fileData,['refModeshapes_raw.csv'], varNamesModeshapes, nModesRef+3, true);

end