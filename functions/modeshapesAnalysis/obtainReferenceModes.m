function [] = obtainReferenceModes(model, nModesRef)
    % names of the modeshapes file
    varNamesModeshapes = cell(nModesRef+3,1);
    varNamesxyz = {'x' 'y' 'z'};
    for ii = 1:(nModesRef +3)
    if ii <4
    varNamesModeshapes{ii} = varNamesxyz{ii}; 
    else 
        varNamesModeshapes{ii} = ['f',int2str(ii-3)]; 
    end
    end
    % obtain eigenfreqeucnies
    model.study('std1').feature('eig').set('neigs', int2str(nModesRef)); % ---> set number of modes
    model.component('comp1').physics('solid').feature('lemm1').feature('dmp1').active(false); % --> deactivate damping
    model.study('std1').run(); 
    % export modeshapes
    expression = {'solid.disp'};
    model.result.export('data1').set('transpose', false);
    model.result.export('data1').set('data', 'dset1');
    exportAllModesFromDataset(model, 'solidDisp',simFolder,expression);
    fileData = readTuples('solidDisp.txt', nModes+3, true);
    delete('solidDisp.txt'); 
    % save in back into .csv file
    writeMat2File(fileData,'reference_raw.csv', varNamesModeshapes, nModes+3, true);
end